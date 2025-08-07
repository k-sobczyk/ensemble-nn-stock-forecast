import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from src.model.ensemble.ensemble_base import BaseEnsemble, ModelTrainer


class BlendingEnsemble(BaseEnsemble):
    """Enhanced Blending Ensemble with support for hyperparameter optimization."""

    def __init__(
        self,
        models_config=None,
        meta_model_type='ridge',
        blend_ratio=0.2,
        # Ridge parameters
        alpha=1.0,
        # ElasticNet parameters
        l1_ratio=0.5,
        # RandomForest parameters
        n_estimators=100,
        max_depth=5,
        random_state=42,
    ):
        super().__init__(models_config)
        self.meta_model_type = meta_model_type
        self.blend_ratio = blend_ratio
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.meta_model = None

    def fit(self, X_train, y_train, X_val, y_val, input_size, sequence_length=None, epochs=50):
        """Train base models and meta-model using data splitting."""
        print('=' * 60)
        print('ENHANCED BLENDING ENSEMBLE TRAINING')
        print('=' * 60)

        # Step 1: Split training data into base model training and blending sets
        print(f'Splitting training data (blend_ratio={self.blend_ratio})...')
        X_base, X_blend, y_base, y_blend = train_test_split(
            X_train, y_train, test_size=self.blend_ratio, random_state=self.random_state
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

        print('Enhanced blending ensemble training completed!')

    def _train_meta_model(self, blend_predictions, y_blend, X_val, y_val):
        """Train the meta-model with enhanced hyperparameter support."""
        # Prepare meta-features matrix
        model_names = list(blend_predictions.keys())
        meta_features = np.column_stack([blend_predictions[name] for name in model_names])

        # Initialize meta-model with hyperparameters
        if self.meta_model_type == 'linear':
            self.meta_model = LinearRegression()
        elif self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.meta_model_type == 'elastic':
            self.meta_model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state)
        elif self.meta_model_type == 'rf':
            self.meta_model = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
            )
        else:
            raise ValueError(f'Unknown meta_model_type: {self.meta_model_type}')

        # Train meta-model
        self.meta_model.fit(meta_features, y_blend)

        # Evaluate on validation set
        val_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_val)
        val_meta_features = np.column_stack([val_predictions[name] for name in model_names])
        val_pred = self.meta_model.predict(val_meta_features)
        val_score = np.sqrt(np.mean((val_pred - y_val) ** 2))
        print(f'Meta-model validation RMSE: {val_score:.4f}')

        # Print model coefficients/feature importance if available
        self._print_model_insights(model_names)

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
