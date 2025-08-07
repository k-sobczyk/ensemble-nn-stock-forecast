import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from src.model.ensemble.ensemble_base import BaseEnsemble, ModelTrainer


class StackingEnsemble(BaseEnsemble):
    """Enhanced Stacking Ensemble with support for hyperparameter optimization."""

    def __init__(
        self,
        models_config=None,
        meta_model_type='ridge',
        cv_folds=5,
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
        self.cv_folds = cv_folds
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.meta_model = None

    def fit(self, X_train, y_train, X_val, y_val, input_size, sequence_length=None, epochs=50):
        """Train base models and meta-model using simple train/validation split."""
        print('=' * 60)
        print('STACKING ENSEMBLE TRAINING (Train/Val Split)')
        print('=' * 60)

        # Step 1: Train base models on training set
        print('Training base models on training data...')
        self.trained_models = ModelTrainer.train_individual_models(
            X_train, y_train, X_val, y_val, input_size, sequence_length, self.models_config, epochs
        )

        # Step 2: Generate meta-features from validation predictions
        print('Generating meta-features from validation predictions...')
        val_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_val)
        model_names = list(val_predictions.keys())
        val_meta_features = np.column_stack([val_predictions[name] for name in model_names])

        # Step 3: Train meta-model on validation meta-features
        print(f'Training meta-model ({self.meta_model_type}) on validation predictions...')
        self._train_meta_model(val_meta_features, y_val, model_names)

        print('Stacking ensemble training completed!')

    def _train_meta_model(self, meta_features, y_val, model_names):
        """Train the meta-model with enhanced hyperparameter support."""
        # Initialize meta-model based on type with hyperparameters
        if self.meta_model_type == 'linear':
            self.meta_model = LinearRegression()
        elif self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.meta_model_type == 'lasso':
            self.meta_model = Lasso(alpha=self.alpha, random_state=self.random_state)
        elif self.meta_model_type == 'elastic':
            self.meta_model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state)
        elif self.meta_model_type == 'rf':
            self.meta_model = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
            )
        else:
            raise ValueError(f'Unknown meta_model_type: {self.meta_model_type}')

        # Train meta-model on validation meta-features
        self.meta_model.fit(meta_features, y_val)
        
        print(f'Meta-model trained on {len(meta_features)} validation samples')

        # Print model insights
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
