import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from src.model.ensemble.ensemble_base import BaseEnsemble, ModelTrainer


class StackingEnsemble(BaseEnsemble):
    def __init__(self, models_config=None, meta_model_type='linear', cv_folds=5):
        super().__init__(models_config)
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.meta_model = None

    def fit(self, X_train, y_train, X_val, y_val, input_size, sequence_length=None, epochs=50):
        """Train base models and meta-model using cross-validation."""
        print('=' * 60)
        print('STACKING ENSEMBLE TRAINING')
        print('=' * 60)

        # Step 1: Generate meta-features using cross-validation
        print(f'Generating meta-features using {self.cv_folds}-fold cross-validation...')
        meta_features_train = self._generate_meta_features(X_train, y_train, input_size, sequence_length, epochs)

        # Step 2: Train final base models on full training set
        print('\nTraining final base models on full training data...')
        self.trained_models = ModelTrainer.train_individual_models(
            X_train, y_train, X_val, y_val, input_size, sequence_length, self.models_config, epochs
        )

        # Step 3: Generate meta-features for validation set
        print('Generating meta-features for validation set...')
        meta_features_val = self._get_meta_features_from_trained_models(X_val)

        # Step 4: Train meta-model
        print(f'Training meta-model ({self.meta_model_type})...')
        self._train_meta_model(meta_features_train, y_train, meta_features_val, y_val)

        print('Stacking ensemble training completed!')

    def _generate_meta_features(self, X_train, y_train, input_size, sequence_length, epochs):
        """Generate meta-features using cross-validation on training data."""
        n_samples = len(X_train)
        n_models = sum(self.models_config.values())
        meta_features = np.zeros((n_samples, n_models))

        # Create cross-validation folds with indices
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        model_names = [name for name, include in self.models_config.items() if include]

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f'  Processing fold {fold_idx + 1}/{self.cv_folds}')

            # Extract fold data
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            # Train models on this fold
            fold_models = ModelTrainer.train_individual_models(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                input_size,
                sequence_length,
                self.models_config,
                epochs // 2,  # Shorter training for CV
            )

            # Get predictions for validation samples in this fold
            fold_predictions = ModelTrainer.get_model_predictions(fold_models, X_val_fold)

            # Store predictions in meta-features matrix using actual validation indices
            for model_idx, model_name in enumerate(model_names):
                if model_name in fold_predictions:
                    meta_features[val_idx, model_idx] = fold_predictions[model_name]

        return meta_features

    def _get_meta_features_from_trained_models(self, X_data):
        """Get meta-features from already trained models."""
        predictions = ModelTrainer.get_model_predictions(self.trained_models, X_data)
        model_names = [name for name, include in self.models_config.items() if include]

        meta_features = np.column_stack([predictions[name] for name in model_names if name in predictions])
        return meta_features

    def _train_meta_model(self, meta_features_train, y_train, meta_features_val, y_val):
        """Train the meta-model on meta-features."""
        if self.meta_model_type == 'linear':
            self.meta_model = LinearRegression()
        elif self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        elif self.meta_model_type == 'lasso':
            self.meta_model = Lasso(alpha=0.1)
        elif self.meta_model_type == 'rf':
            self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.meta_model_type == 'neural':
            self.meta_model = NeuralMetaModel(meta_features_train.shape[1])
            self._train_neural_meta_model(meta_features_train, y_train, meta_features_val, y_val)
            return
        else:
            raise ValueError(f'Unknown meta_model_type: {self.meta_model_type}')

        # Train sklearn-based meta-model
        self.meta_model.fit(meta_features_train, y_train)

        # Evaluate on validation set
        val_pred = self.meta_model.predict(meta_features_val)
        val_score = np.sqrt(np.mean((val_pred - y_val) ** 2))
        print(f'Meta-model validation RMSE: {val_score:.4f}')

    def _train_neural_meta_model(self, meta_features_train, y_train, meta_features_val, y_val):
        """Train neural network meta-model."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(meta_features_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(meta_features_val)
        y_val_tensor = torch.FloatTensor(y_val)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)

        # Training loop
        epochs = 100
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            self.meta_model.train()
            optimizer.zero_grad()
            outputs = self.meta_model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            self.meta_model.eval()
            with torch.no_grad():
                val_outputs = self.meta_model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Meta-model early stopping at epoch {epoch + 1}')
                break

        print(f'Meta-model final validation RMSE: {np.sqrt(best_val_loss.item()):.4f}')

    def predict(self, X_test):
        """Make ensemble predictions using the meta-model."""
        if not self.trained_models or self.meta_model is None:
            raise ValueError('Ensemble must be fitted before making predictions')

        # Get meta-features from base models
        meta_features_test = self._get_meta_features_from_trained_models(X_test)

        # Make prediction with meta-model
        if self.meta_model_type == 'neural':
            self.meta_model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(meta_features_test)
                ensemble_pred = self.meta_model(test_tensor).squeeze().numpy()
        else:
            ensemble_pred = self.meta_model.predict(meta_features_test)

        return ensemble_pred

    def get_meta_features(self, X_data):
        """Get meta-features for analysis."""
        return self._get_meta_features_from_trained_models(X_data)


class NeuralMetaModel(nn.Module):
    """Neural network meta-model for stacking."""

    def __init__(self, input_size, hidden_size=32, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.network(x)


class DynamicStackingEnsemble(StackingEnsemble):
    def __init__(
        self, models_config=None, meta_model_type='neural', cv_folds=5, adaptation_window=200, retrain_frequency=50
    ):
        super().__init__(models_config, meta_model_type, cv_folds)
        self.adaptation_window = adaptation_window
        self.retrain_frequency = retrain_frequency
        self.recent_meta_features = []
        self.recent_targets = []
        self.prediction_count = 0

    def predict_adaptive(self, X_test, y_true=None):
        """Make predictions with periodic meta-model retraining."""
        predictions = []

        for i, x_sample in enumerate(X_test):
            # Make prediction
            meta_features = self._get_meta_features_from_trained_models(x_sample.reshape(1, -1))

            if self.meta_model_type == 'neural':
                self.meta_model.eval()
                with torch.no_grad():
                    pred = self.meta_model(torch.FloatTensor(meta_features)).item()
            else:
                pred = self.meta_model.predict(meta_features)[0]

            predictions.append(pred)

            # Store recent data for adaptation
            if y_true is not None and i < len(y_true):
                self.recent_meta_features.append(meta_features[0])
                self.recent_targets.append(y_true[i])

                # Keep only recent data
                if len(self.recent_meta_features) > self.adaptation_window:
                    self.recent_meta_features.pop(0)
                    self.recent_targets.pop(0)

                # Retrain meta-model periodically
                self.prediction_count += 1
                if self.prediction_count % self.retrain_frequency == 0 and len(self.recent_meta_features) >= 50:
                    print(f'Retraining meta-model at prediction {self.prediction_count}')
                    self._retrain_meta_model()

        return np.array(predictions)

    def _retrain_meta_model(self):
        """Retrain meta-model on recent data."""
        recent_features = np.array(self.recent_meta_features)
        recent_targets = np.array(self.recent_targets)

        if self.meta_model_type == 'neural':
            # Quick fine-tuning for neural model
            self.meta_model.train()
            optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.0001)
            criterion = nn.MSELoss()

            for _ in range(10):  # Quick fine-tuning
                optimizer.zero_grad()
                outputs = self.meta_model(torch.FloatTensor(recent_features))
                loss = criterion(outputs.squeeze(), torch.FloatTensor(recent_targets))
                loss.backward()
                optimizer.step()
        else:
            # Retrain sklearn model
            self.meta_model.fit(recent_features, recent_targets)
