import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from src.model.ensemble.ensemble_base import BaseEnsemble, ModelTrainer, create_cross_validation_folds


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
        """Train base models and meta-model using cross-validation."""
        print('=' * 60)
        print('ENHANCED STACKING ENSEMBLE TRAINING')
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
        val_predictions = ModelTrainer.get_model_predictions(self.trained_models, X_val)
        model_names = list(val_predictions.keys())
        val_meta_features = np.column_stack([val_predictions[name] for name in model_names])

        # Step 4: Train meta-model
        print(f'Training meta-model ({self.meta_model_type})...')
        self._train_meta_model(meta_features_train, y_train, val_meta_features, y_val, model_names)

        print('Enhanced stacking ensemble training completed!')

    def _generate_meta_features(self, X_train, y_train, input_size, sequence_length, epochs):
        """Generate meta-features using cross-validation."""
        # Create cross-validation folds
        folds = create_cross_validation_folds(X_train, y_train, self.cv_folds)

        # Initialize meta-features array
        n_models = sum(self.models_config.values())
        meta_features = np.zeros((len(X_train), n_models))

        for fold_idx, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(folds):
            print(f'  Processing fold {fold_idx + 1}/{self.cv_folds}')

            # Train models on this fold
            fold_models = ModelTrainer.train_individual_models(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                input_size,
                sequence_length,
                self.models_config,
                epochs=1,  # Reduced epochs for CV
            )

            # Get predictions for validation set of this fold
            fold_predictions = ModelTrainer.get_model_predictions(fold_models, X_val_fold)

            # Store meta-features for this fold
            model_names = list(fold_predictions.keys())
            fold_meta = np.column_stack([fold_predictions[name] for name in model_names])

            # Get indices for this validation fold
            val_indices = list(range(fold_idx * len(X_val_fold), (fold_idx + 1) * len(X_val_fold)))
            if fold_idx == self.cv_folds - 1:  # Last fold might be different size
                val_indices = list(range(fold_idx * (len(X_train) // self.cv_folds), len(X_train)))

            meta_features[val_indices] = fold_meta

        return meta_features

    def _train_meta_model(self, meta_features, y_train, val_meta_features, y_val, model_names):
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

        # Train meta-model
        self.meta_model.fit(meta_features, y_train)

        # Evaluate on validation set
        val_pred = self.meta_model.predict(val_meta_features)
        val_score = np.sqrt(np.mean((val_pred - y_val) ** 2))
        print(f'Meta-model validation RMSE: {val_score:.4f}')

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
