import numpy as np
import torch

from src.model.ensemble.ensemble_base import BaseEnsemble, ModelTrainer


class BoostingEnsemble(BaseEnsemble):
    def __init__(self, models_config=None, n_estimators=4, learning_rate=0.8):
        super().__init__(models_config)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model_weights = []
        self.models_sequence = []

    def fit(self, X_train, y_train, X_val, y_val, input_size, sequence_length=None, epochs=50):
        """Train models sequentially with adaptive sample weighting."""
        print('=' * 60)
        print('BOOSTING ENSEMBLE TRAINING')
        print('=' * 60)

        # Initialize sample weights (uniform)
        n_samples = len(X_train)
        sample_weights = np.ones(n_samples) / n_samples
        ensemble_pred = np.zeros(n_samples)

        # Get model names to cycle through
        model_names = [name for name, include in self.models_config.items() if include]

        for round_idx in range(self.n_estimators):
            print(f'\nBoosting Round {round_idx + 1}/{self.n_estimators}')

            # Select model type for this round (cycle through available models)
            model_name = model_names[round_idx % len(model_names)]
            print(f'Training {model_name} model...')

            # Create weighted training set
            X_weighted, y_weighted = self._create_weighted_sample(X_train, y_train, sample_weights)

            # Train single model
            models_config_single = {name: (name == model_name) for name in self.models_config.keys()}
            model_dict = ModelTrainer.train_individual_models(
                X_weighted,
                y_weighted,
                X_val,
                y_val,
                input_size,
                sequence_length,
                models_config_single,
                epochs // 2,  # Shorter training for boosting
            )

            # Get the trained model
            trained_model = model_dict[model_name]
            self.models_sequence.append((model_name, trained_model))

            # Get predictions on training set
            with torch.no_grad():
                if model_name == 'cnn':
                    pred = self._get_cnn_predictions(trained_model, X_train)
                else:
                    pred = trained_model(torch.FloatTensor(X_train)).numpy().flatten()

            # Calculate weighted error
            errors = np.abs(pred - y_train)
            weighted_error = np.average(errors, weights=sample_weights)

            # Calculate model weight (alpha)
            if weighted_error < 1e-10:  # Perfect model
                alpha = 10.0
            elif weighted_error >= 0.5:  # Worse than random
                alpha = 0.1
            else:
                alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)

            alpha *= self.learning_rate
            self.model_weights.append(alpha)

            print(f'Model weighted error: {weighted_error:.4f}, weight: {alpha:.4f}')

            # Update ensemble prediction
            ensemble_pred += alpha * pred

            # Update sample weights (focus more on poorly predicted samples)
            sample_weights = sample_weights * np.exp(alpha * errors / np.max(errors))
            sample_weights /= np.sum(sample_weights)  # Normalize

        print(f'\nBoosting completed! Model weights: {self.model_weights}')

    def _create_weighted_sample(self, X_train, y_train, sample_weights, oversample_factor=2):
        """Create weighted training sample by importance sampling."""
        n_samples = int(len(X_train) * oversample_factor)

        # Sample indices based on weights
        indices = np.random.choice(len(X_train), size=n_samples, p=sample_weights, replace=True)

        return X_train[indices], y_train[indices]

    def _get_cnn_predictions(self, model, X_data):
        """Get predictions from CNN model."""
        from .ensemble_base import StockDatasetCNN

        test_dataset = StockDatasetCNN(X_data)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        predictions = []
        for batch_X in test_loader:
            with torch.no_grad():
                batch_pred = model(batch_X).numpy()
                predictions.extend(batch_pred)

        return np.array(predictions).flatten()

    def predict(self, X_test):
        """Make ensemble predictions using weighted combination."""
        if not self.models_sequence:
            raise ValueError('Ensemble must be fitted before making predictions')

        ensemble_pred = np.zeros(len(X_test))

        for (model_name, model), weight in zip(self.models_sequence, self.model_weights):
            model.eval()
            with torch.no_grad():
                if model_name == 'cnn':
                    pred = self._get_cnn_predictions(model, X_test)
                else:
                    pred = model(torch.FloatTensor(X_test)).numpy().flatten()

            ensemble_pred += weight * pred

        # Normalize by total weight
        total_weight = sum(self.model_weights)
        return ensemble_pred / total_weight if total_weight > 0 else ensemble_pred


class GradientBoostingEnsemble(BaseEnsemble):
    def __init__(self, models_config=None, n_estimators=4, learning_rate=0.1):
        super().__init__(models_config)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models_sequence = []

    def fit(self, X_train, y_train, X_val, y_val, input_size, sequence_length=None, epochs=50):
        """Train models sequentially on residuals."""
        print('=' * 60)
        print('GRADIENT BOOSTING ENSEMBLE TRAINING')
        print('=' * 60)

        # Initialize with zero predictions
        current_pred = np.zeros(len(y_train))
        model_names = [name for name, include in self.models_config.items() if include]

        for round_idx in range(self.n_estimators):
            print(f'\nGradient Boosting Round {round_idx + 1}/{self.n_estimators}')

            # Calculate residuals
            residuals = y_train - current_pred
            print(f'Mean absolute residual: {np.mean(np.abs(residuals)):.4f}')

            # Select model type for this round
            model_name = model_names[round_idx % len(model_names)]
            print(f'Training {model_name} on residuals...')

            # Train model on residuals
            models_config_single = {name: (name == model_name) for name in self.models_config.keys()}
            model_dict = ModelTrainer.train_individual_models(
                X_train,
                residuals,
                X_val,
                y_val - current_pred[: len(X_val)] if len(current_pred) > len(X_val) else y_val,
                input_size,
                sequence_length,
                models_config_single,
                epochs // 3,
            )

            trained_model = model_dict[model_name]
            self.models_sequence.append((model_name, trained_model))

            # Update predictions
            with torch.no_grad():
                if model_name == 'cnn':
                    step_pred = self._get_cnn_predictions(trained_model, X_train)
                else:
                    step_pred = trained_model(torch.FloatTensor(X_train)).numpy().flatten()

            current_pred += self.learning_rate * step_pred

        print('Gradient boosting completed!')

    def predict(self, X_test):
        """Make ensemble predictions by summing all model predictions."""
        if not self.models_sequence:
            raise ValueError('Ensemble must be fitted before making predictions')

        ensemble_pred = np.zeros(len(X_test))

        for model_name, model in self.models_sequence:
            model.eval()
            with torch.no_grad():
                if model_name == 'cnn':
                    pred = self._get_cnn_predictions(model, X_test)
                else:
                    pred = model(torch.FloatTensor(X_test)).numpy().flatten()

            ensemble_pred += self.learning_rate * pred

        return ensemble_pred

    def _get_cnn_predictions(self, model, X_data):
        """Get predictions from CNN model."""
        from src.model.ensemble.ensemble_base import StockDatasetCNN

        test_dataset = StockDatasetCNN(X_data)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        predictions = []
        for batch_X in test_loader:
            with torch.no_grad():
                batch_pred = model(batch_X).numpy()
                predictions.extend(batch_pred)

        return np.array(predictions).flatten()
