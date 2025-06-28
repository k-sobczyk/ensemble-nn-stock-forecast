import numpy as np
from sklearn.linear_model import LinearRegression

from src.model.ensemble.ensemble_base import BaseEnsemble, ModelTrainer


class VotingEnsemble(BaseEnsemble):
    def __init__(self, models_config=None, voting_type='simple', optimize_weights=True):
        super().__init__(models_config)
        self.voting_type = voting_type
        self.optimize_weights = optimize_weights
        self.weights = None

    def fit(self, X_train, y_train, X_val, y_val, input_size, sequence_length=None, epochs=50):
        """Train individual models and optionally optimize weights."""
        print('=' * 60)
        print('VOTING ENSEMBLE TRAINING')
        print('=' * 60)

        # Train individual models
        self.trained_models = ModelTrainer.train_individual_models(
            X_train, y_train, X_val, y_val, input_size, sequence_length, self.models_config, epochs
        )

        if self.voting_type == 'weighted' and self.optimize_weights:
            print('\nOptimizing ensemble weights...')
            self._optimize_weights(X_val, y_val)
        else:
            # Equal weights
            n_models = len(self.trained_models)
            self.weights = {name: 1.0 / n_models for name in self.trained_models.keys()}

        print(f'\nFinal ensemble weights: {self.weights}')

    def _optimize_weights(self, X_val, y_val):
        """Optimize weights using validation data."""
        # Get predictions from all models on validation set
        val_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_val)

        # Stack predictions for optimization
        pred_matrix = np.column_stack([val_predictions[name] for name in self.trained_models.keys()])

        # Use linear regression to find optimal weights (constrained to be positive)
        # This is a simplified approach - you could use more sophisticated optimization
        lr = LinearRegression(fit_intercept=False, positive=True)
        lr.fit(pred_matrix, y_val)

        # Normalize weights to sum to 1
        raw_weights = lr.coef_
        normalized_weights = raw_weights / raw_weights.sum()

        self.weights = {name: weight for name, weight in zip(self.trained_models.keys(), normalized_weights)}

    def predict(self, X_test):
        """Make ensemble predictions using weighted averaging."""
        if not self.trained_models:
            raise ValueError('Ensemble must be fitted before making predictions')

        # Get predictions from all models
        test_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_test)

        # Weighted average
        ensemble_pred = np.zeros(len(X_test))
        for model_name, predictions in test_predictions.items():
            weight = self.weights[model_name]
            ensemble_pred += weight * predictions

        return ensemble_pred

    def get_individual_predictions(self, X_test):
        """Get predictions from individual models for analysis."""
        return ModelTrainer.get_model_predictions(self.trained_models, X_test)


class AdaptiveVotingEnsemble(VotingEnsemble):
    def __init__(self, models_config=None, window_size=100, adaptation_rate=0.1):
        super().__init__(models_config, voting_type='adaptive')
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.recent_errors = {name: [] for name in models_config.keys() if models_config[name]}

    def predict_adaptive(self, X_test, y_true=None):
        if not self.trained_models:
            raise ValueError('Ensemble must be fitted before making predictions')

        predictions = []
        individual_preds = ModelTrainer.get_model_predictions(self.trained_models, X_test)

        for i in range(len(X_test)):
            # Get current predictions for this sample
            current_preds = {name: preds[i] for name, preds in individual_preds.items()}

            # Make ensemble prediction
            ensemble_pred = sum(self.weights[name] * pred for name, pred in current_preds.items())
            predictions.append(ensemble_pred)

            # Update weights if we have ground truth
            if y_true is not None and i < len(y_true):
                self._update_weights(current_preds, y_true[i])

        return np.array(predictions)

    def _update_weights(self, current_preds, true_value):
        """Update weights based on recent prediction errors."""
        # Calculate errors for each model
        for model_name, pred in current_preds.items():
            error = abs(pred - true_value)
            self.recent_errors[model_name].append(error)

            # Keep only recent errors
            if len(self.recent_errors[model_name]) > self.window_size:
                self.recent_errors[model_name].pop(0)

        # Update weights based on recent performance (lower error = higher weight)
        if all(len(errors) >= 10 for errors in self.recent_errors.values()):  # Need some history
            avg_errors = {name: np.mean(errors) for name, errors in self.recent_errors.items()}

            # Convert errors to weights (inverse relationship)
            inv_errors = {name: 1.0 / (error + 1e-8) for name, error in avg_errors.items()}
            total_inv_error = sum(inv_errors.values())

            # Update weights with exponential smoothing
            for name in self.weights:
                new_weight = inv_errors[name] / total_inv_error
                self.weights[name] = (1 - self.adaptation_rate) * self.weights[name] + self.adaptation_rate * new_weight
