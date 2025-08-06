"""Optimized Ensemble Runner using pre-optimized hyperparameters.

This runner uses the best hyperparameters found through Optuna optimization,
stored in ensemble_config.py, similar to how individual models use config.py.
"""

import os
import time
from datetime import datetime

from src.model.ensemble.blending_ensemble import EnhancedBlendingEnsemble
from src.model.ensemble.enhanced_ensemble_runner import EnhancedEnsembleRunner
from src.model.ensemble.ensemble_config import (
    USE_OPTIMIZED_ENSEMBLE_PARAMS,
    VERBOSE_ENSEMBLE_PARAMS,
    get_optimized_blending_params,
    get_optimized_stacking_params,
    get_optimized_voting_params,
)
from src.model.ensemble.stacking_ensemble import EnhancedStackingEnsemble
from src.model.ensemble.voting_ensemble import VotingEnsemble


class OptimizedEnsembleRunner(EnhancedEnsembleRunner):
    def __init__(
        self,
        df_path='dataset_1_full_features.csv',
        sequence_length=None,
        test_start_year=2021,
        epochs=30,
        output_dir='output_optimized',
        use_optimized_params=True,
    ):
        super().__init__(df_path, sequence_length, test_start_year, epochs, output_dir)
        self.use_optimized_params = use_optimized_params and USE_OPTIMIZED_ENSEMBLE_PARAMS

    def run_combination(self, combination_name, models_config, ensemble_methods=['voting', 'stacking', 'blending']):
        """Run combination using optimized hyperparameters."""
        print(f'\n{"=" * 80}')
        print(f'OPTIMIZED ENSEMBLE: {combination_name}')
        print(f'Models: {[model for model, enabled in models_config.items() if enabled]}')
        if self.use_optimized_params:
            print('🎯 Using pre-optimized hyperparameters from ensemble_config.py')
        else:
            print('⚙️  Using default hyperparameters')
        print(f'{"=" * 80}')

        combination_results = {}
        combination_dir = os.path.join(self.output_dir, combination_name.lower())
        os.makedirs(combination_dir, exist_ok=True)

        for method in ensemble_methods:
            print(f'\n{"-" * 60}')
            print(f'Running OPTIMIZED {method.upper()} Ensemble for {combination_name}')
            print(f'{"-" * 60}')

            start_time = time.time()

            try:
                if method == 'voting':
                    ensemble = self._create_optimized_voting_ensemble(combination_name, models_config)
                elif method == 'stacking':
                    ensemble = self._create_optimized_stacking_ensemble(combination_name, models_config)
                elif method == 'blending':
                    ensemble = self._create_optimized_blending_ensemble(combination_name, models_config)
                else:
                    print(f'Unknown ensemble method: {method}')
                    continue

                # Train ensemble
                ensemble.fit(
                    self.X_train,
                    self.y_train,
                    self.X_val,
                    self.y_val,
                    self.input_size,
                    self.sequence_length_used,
                    self.epochs,
                )

                # Make predictions
                predictions = ensemble.predict(self.X_test)

                # Evaluate with base metrics
                metrics = ensemble.evaluate(self.y_test, predictions, self.scaler_y)

                # Calculate additional metrics
                actual = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
                predicted = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
                mape, mase, smape, mape_log = self.calculate_additional_metrics(actual, predicted)

                training_time = time.time() - start_time

                # Create comprehensive results
                comprehensive_results = {
                    'combination_name': combination_name,
                    'ensemble_method': method,
                    'models_used': [model for model, enabled in models_config.items() if enabled],
                    'optimized_params': self._get_used_params(combination_name, method),
                    'metrics': {
                        'rmse': float(metrics['rmse']),
                        'mae': float(metrics['mae']),
                        'r2': float(metrics['r2']),
                        'mse': float(metrics['mse']),
                        'mape': float(mape),
                        'smape': float(smape),
                        'mape_log': float(mape_log),
                        'mase': float(mase),
                    },
                    'training_data': {
                        'training_time': training_time,
                        'epochs': self.epochs,
                        'sequence_length': self.sequence_length_used,
                        'input_size': self.input_size,
                        'used_optimized_params': self.use_optimized_params,
                    },
                    'predictions': {
                        'actual_values': [float(x) for x in actual],
                        'predicted_values': [float(x) for x in predicted],
                    },
                    'timestamp': datetime.now().isoformat(),
                }

                combination_results[method] = comprehensive_results

                # Save individual ensemble method results
                self.save_individual_results(combination_name, method, comprehensive_results, combination_dir)

                print(f'✅ {method.upper()} completed in {training_time:.2f}s')
                print(f'   RMSE: {metrics["rmse"]:.4f}, MAE: {metrics["mae"]:.4f}, R²: {metrics["r2"]:.4f}')

            except Exception as e:
                print(f'❌ Error in {method}: {str(e)}')
                combination_results[method] = {'error': str(e)}

        self.results[combination_name] = combination_results

        # Create combination-level visualizations
        self.create_combination_visualizations(combination_name, combination_results, combination_dir)

        return combination_results

    def _create_optimized_voting_ensemble(self, combination_name, models_config):
        """Create voting ensemble with optimized parameters."""
        if self.use_optimized_params:
            params = get_optimized_voting_params(combination_name)
            if VERBOSE_ENSEMBLE_PARAMS:
                print(f'🎯 Optimized Voting params: {params}')
        else:
            params = {'voting_type': 'weighted', 'optimize_weights': True}

        return VotingEnsemble(
            models_config, voting_type=params['voting_type'], optimize_weights=params['optimize_weights']
        )

    def _create_optimized_stacking_ensemble(self, combination_name, models_config):
        """Create stacking ensemble with optimized parameters."""
        if self.use_optimized_params:
            params = get_optimized_stacking_params(combination_name)
            if VERBOSE_ENSEMBLE_PARAMS:
                print(f'🎯 Optimized Stacking params: {params}')
        else:
            params = {'meta_model_type': 'ridge', 'cv_folds': 5, 'alpha': 1.0}

        return EnhancedStackingEnsemble(
            models_config,
            meta_model_type=params['meta_model_type'],
            cv_folds=params['cv_folds'],
            alpha=params.get('alpha', 1.0),
            l1_ratio=params.get('l1_ratio', 0.5),
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
        )

    def _create_optimized_blending_ensemble(self, combination_name, models_config):
        """Create blending ensemble with optimized parameters."""
        if self.use_optimized_params:
            params = get_optimized_blending_params(combination_name)
            if VERBOSE_ENSEMBLE_PARAMS:
                print(f'🎯 Optimized Blending params: {params}')
        else:
            params = {'meta_model_type': 'ridge', 'blend_ratio': 0.2, 'alpha': 1.0}

        return EnhancedBlendingEnsemble(
            models_config,
            meta_model_type=params['meta_model_type'],
            blend_ratio=params['blend_ratio'],
            alpha=params.get('alpha', 1.0),
            l1_ratio=params.get('l1_ratio', 0.5),
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
        )

    def _get_used_params(self, combination_name, method):
        """Get the parameters that were actually used for this combination/method."""
        if not self.use_optimized_params:
            return {'source': 'default_parameters'}

        if method == 'voting':
            return {'source': 'ensemble_config.py', 'params': get_optimized_voting_params(combination_name)}
        elif method == 'stacking':
            return {'source': 'ensemble_config.py', 'params': get_optimized_stacking_params(combination_name)}
        elif method == 'blending':
            return {'source': 'ensemble_config.py', 'params': get_optimized_blending_params(combination_name)}
        else:
            return {'source': 'unknown'}


def run_optimized_research_analysis(epochs=50, use_optimized_params=True):
    """Run complete research analysis using optimized parameters."""
    print('🚀 Optimized Neural Network Ensemble Analysis')
    print('=' * 80)
    print(f'🎯 Using optimized parameters: {use_optimized_params}')
    print(f'🔄 Individual model epochs: {epochs}')
    print('=' * 80)

    runner = OptimizedEnsembleRunner(
        epochs=epochs, use_optimized_params=use_optimized_params, output_dir='output_optimized_research'
    )

    runner.prepare_data()

    # Run all combinations
    runner.run_all_pairs()
    runner.run_all_triplets()

    # Create comprehensive summary
    runner.create_comprehensive_summary()

    print('\n🎉 Optimized research analysis completed!')
    print(f'📁 All results saved to: {runner.output_dir}')

    return runner.results


def run_optimized_high_diversity(epochs=30, use_optimized_params=True):
    """Run high-diversity pairs with optimized parameters."""
    print('🚀 High-Diversity Pairs with Optimized Parameters')
    print('=' * 80)
    print(f'🎯 Using optimized parameters: {use_optimized_params}')
    print('🎯 Testing: LSTM+CNN, GRU+CNN, BiLSTM+CNN')
    print('=' * 80)

    runner = OptimizedEnsembleRunner(
        epochs=epochs, use_optimized_params=use_optimized_params, output_dir='output_optimized_high_diversity'
    )

    runner.prepare_data()

    # Run only high-diversity pairs
    high_diversity_pairs = ['LSTM_CNN', 'GRU_CNN', 'BiLSTM_CNN']

    for pair_name in high_diversity_pairs:
        if pair_name in runner.pair_combinations:
            runner.run_combination(pair_name, runner.pair_combinations[pair_name])

    runner.create_comprehensive_summary()

    print('\n🎉 High-diversity analysis completed!')
    return runner.results


if __name__ == '__main__':
    print('🔬 Optimized Ensemble Analysis')
    print('Choose an option:')
    print('1. High-diversity pairs (optimized)')
    print('2. Complete analysis (optimized)')
    print('3. High-diversity pairs (default params - for comparison)')

    choice = input('Enter choice (1-3): ').strip()

    if choice == '1':
        results = run_optimized_high_diversity(epochs=30, use_optimized_params=True)
    elif choice == '2':
        results = run_optimized_research_analysis(epochs=50, use_optimized_params=True)
    elif choice == '3':
        results = run_optimized_high_diversity(epochs=30, use_optimized_params=False)
    else:
        print('Running default: High-diversity pairs with optimized parameters')
        results = run_optimized_high_diversity(epochs=30, use_optimized_params=True)
