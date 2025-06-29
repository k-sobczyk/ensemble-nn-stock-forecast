import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from src.model.ensemble.ensemble_base import BaseEnsemble, ModelTrainer


class BlendingEnsemble(BaseEnsemble):
    def __init__(self, models_config=None, meta_model_type='ridge', blend_ratio=0.2):
        super().__init__(models_config)
        self.meta_model_type = meta_model_type
        self.blend_ratio = blend_ratio
        self.meta_model = None

    def fit(self, X_train, y_train, X_val, y_val, input_size, sequence_length=None, epochs=50):
        """Train base models and meta-model using data splitting."""
        print('=' * 60)
        print('BLENDING ENSEMBLE TRAINING')
        print('=' * 60)

        # Step 1: Split training data into base model training and blending sets
        print(f'Splitting training data (blend_ratio={self.blend_ratio})...')
        X_base, X_blend, y_base, y_blend = train_test_split(
            X_train, y_train, test_size=self.blend_ratio, random_state=42
        )

        print(f'Base model training set: {len(X_base)} samples')
        print(f'Blending set: {len(X_blend)} samples')

        # Step 2: Train base models on base training set
        print('\nTraining base models...')
        self.trained_models = ModelTrainer.train_individual_models(
            X_base, y_base, X_val, y_val, input_size, sequence_length, self.models_config, epochs
        )

        # Step 3: Generate predictions on blending set
        print('Generating predictions for blending set...')
        blend_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_blend)

        # Step 4: Train meta-model on blending set
        print(f'Training meta-model ({self.meta_model_type})...')
        self._train_meta_model(blend_predictions, y_blend, X_val, y_val)

        print('Blending ensemble training completed!')

    def _train_meta_model(self, blend_predictions, y_blend, X_val, y_val):
        """Train the meta-model on blending predictions."""
        # Prepare meta-features matrix
        model_names = list(blend_predictions.keys())
        meta_features = np.column_stack([blend_predictions[name] for name in model_names])

        # Initialize meta-model
        if self.meta_model_type == 'linear':
            self.meta_model = LinearRegression()
        elif self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        elif self.meta_model_type == 'elastic':
            self.meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif self.meta_model_type == 'rf':
            self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.meta_model_type == 'neural':
            self.meta_model = BlendingNeuralModel(meta_features.shape[1])
            self._train_neural_meta_model(meta_features, y_blend, X_val, y_val)
            return
        else:
            raise ValueError(f'Unknown meta_model_type: {self.meta_model_type}')

        # Train sklearn-based meta-model
        self.meta_model.fit(meta_features, y_blend)

        # Evaluate on validation set
        val_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_val)
        val_meta_features = np.column_stack([val_predictions[name] for name in model_names])
        val_pred = self.meta_model.predict(val_meta_features)
        val_score = np.sqrt(np.mean((val_pred - y_val) ** 2))
        print(f'Meta-model validation RMSE: {val_score:.4f}')

        # Print model coefficients/feature importance if available
        self._print_model_insights(model_names)

    def _train_neural_meta_model(self, meta_features, y_blend, X_val, y_val):
        """Train neural network meta-model."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(meta_features)
        y_tensor = torch.FloatTensor(y_blend)

        # Get validation meta-features
        val_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_val)
        model_names = list(val_predictions.keys())
        val_meta_features = np.column_stack([val_predictions[name] for name in model_names])
        X_val_tensor = torch.FloatTensor(val_meta_features)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)

        # Training loop with early stopping
        epochs = 200
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            self.meta_model.train()
            optimizer.zero_grad()
            outputs = self.meta_model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()

            # Validation every 10 epochs
            if epoch % 10 == 0:
                self.meta_model.eval()
                with torch.no_grad():
                    val_outputs = self.meta_model(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), torch.FloatTensor(y_val))

                print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        print(f'Meta-model final validation RMSE: {np.sqrt(best_val_loss.item()):.4f}')

    def _print_model_insights(self, model_names):
        """Print insights about meta-model weights/importance."""
        if hasattr(self.meta_model, 'coef_'):
            print('\nMeta-model weights:')
            for name, weight in zip(model_names, self.meta_model.coef_):
                print(f'  {name}: {weight:.4f}')
        elif hasattr(self.meta_model, 'feature_importances_'):
            print('\nMeta-model feature importances:')
            for name, importance in zip(model_names, self.meta_model.feature_importances_):
                print(f'  {name}: {importance:.4f}')

    def predict(self, X_test):
        """Make ensemble predictions using the meta-model."""
        if not self.trained_models or self.meta_model is None:
            raise ValueError('Ensemble must be fitted before making predictions')

        # Get predictions from base models
        test_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_test)

        # Prepare meta-features
        model_names = list(test_predictions.keys())
        meta_features = np.column_stack([test_predictions[name] for name in model_names])

        # Make prediction with meta-model
        if self.meta_model_type == 'neural':
            self.meta_model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(meta_features)
                ensemble_pred = self.meta_model(test_tensor).squeeze().numpy()
        else:
            ensemble_pred = self.meta_model.predict(meta_features)

        return ensemble_pred

    def get_model_contributions(self, X_test):
        """Analyze individual model contributions to final predictions."""
        test_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_test)
        model_names = list(test_predictions.keys())

        if hasattr(self.meta_model, 'coef_'):
            # Linear model - can compute exact contributions
            contributions = {}
            for name, predictions in test_predictions.items():
                weight_idx = model_names.index(name)
                weight = self.meta_model.coef_[weight_idx]
                contributions[name] = weight * predictions

            # Add intercept contribution if exists
            if hasattr(self.meta_model, 'intercept_'):
                contributions['intercept'] = np.full(len(X_test), self.meta_model.intercept_)

            return contributions
        else:
            # Non-linear model - return raw predictions and let user analyze
            return test_predictions


class BlendingNeuralModel(nn.Module):
    """Neural network meta-model for blending."""

    def __init__(self, input_size, hidden_size=16, dropout=0.1):
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


class AdaptiveBlendingEnsemble(BlendingEnsemble):
    def __init__(
        self, models_config=None, meta_model_type='ridge', blend_ratio=0.2, learning_rate=0.01, forgetting_factor=0.99
    ):
        super().__init__(models_config, meta_model_type, blend_ratio)
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.online_samples = 0

    def update_meta_model(self, X_new, y_new):
        """Update meta-model with new samples (online learning)."""
        if self.meta_model is None:
            raise ValueError('Meta-model must be trained before online updates')

        # Get predictions from base models
        new_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_new)
        model_names = list(new_predictions.keys())
        new_meta_features = np.column_stack([new_predictions[name] for name in model_names])

        if self.meta_model_type == 'neural':
            self._update_neural_model(new_meta_features, y_new)
        elif hasattr(self.meta_model, 'coef_'):
            self._update_linear_model(new_meta_features, y_new)
        else:
            print('Warning: Online update not supported for this meta-model type')

    def _update_neural_model(self, new_meta_features, y_new):
        """Update neural meta-model with new data."""
        self.meta_model.train()
        optimizer = torch.optim.SGD(self.meta_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Single gradient step
        X_tensor = torch.FloatTensor(new_meta_features)
        y_tensor = torch.FloatTensor(y_new)

        optimizer.zero_grad()
        outputs = self.meta_model(X_tensor)
        loss = criterion(outputs.squeeze(), y_tensor)
        loss.backward()
        optimizer.step()

        self.online_samples += len(y_new)

    def _update_linear_model(self, new_meta_features, y_new):
        """Update linear meta-model using online gradient descent."""
        for i in range(len(y_new)):
            x = new_meta_features[i]
            y = y_new[i]

            # Predict with current model
            y_pred = np.dot(self.meta_model.coef_, x) + self.meta_model.intercept_

            # Compute error
            error = y - y_pred

            # Update weights
            self.meta_model.coef_ += self.learning_rate * error * x
            self.meta_model.intercept_ += self.learning_rate * error

            # Apply forgetting factor
            self.meta_model.coef_ *= self.forgetting_factor
            self.meta_model.intercept_ *= self.forgetting_factor

        self.online_samples += len(y_new)
