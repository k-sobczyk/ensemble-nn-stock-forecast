import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.model.ensemble.blending_ensemble import BlendingEnsemble
from src.model.ensemble.ensemble_base import prepare_ensemble_data
from src.model.ensemble.stacking_ensemble import StackingEnsemble
from src.model.ensemble.voting_ensemble import VotingEnsemble


class EnhancedEnsembleRunner:
    """Enhanced runner for neural network ensemble combinations with comprehensive output."""

    def __init__(
        self,
        df_path='dataset_1_full_features.csv',
        sequence_length=None,
        test_start_year=2021,
        epochs=30,
        output_dir='output',
    ):
        self.df_path = df_path
        self.sequence_length = sequence_length
        self.test_start_year = test_start_year
        self.epochs = epochs
        self.output_dir = output_dir
        self.results = {}

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Define the specific combinations as per research requirements
        self.pair_combinations = {
            'LSTM_GRU': {'lstm': True, 'gru': True, 'bi_lstm': False, 'cnn': False},
            'LSTM_CNN': {'lstm': True, 'gru': False, 'bi_lstm': False, 'cnn': True},
            'GRU_BiLSTM': {'lstm': False, 'gru': True, 'bi_lstm': True, 'cnn': False},
            'GRU_CNN': {'lstm': False, 'gru': True, 'bi_lstm': False, 'cnn': True},
            'BiLSTM_CNN': {'lstm': False, 'gru': False, 'bi_lstm': True, 'cnn': True},
        }

        self.triplet_combinations = {
            'LSTM_GRU_CNN': {'lstm': True, 'gru': True, 'bi_lstm': False, 'cnn': True},
            'GRU_BiLSTM_CNN': {'lstm': False, 'gru': True, 'bi_lstm': True, 'cnn': True},
        }

    def prepare_data(self):
        """Load and prepare data for ensemble training."""
        print('Loading and preparing data...')
        df = pd.read_csv(self.df_path)

        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            self.scaler_X,
            self.scaler_y,
            self.feature_cols,
        ) = prepare_ensemble_data(
            df,
            sequence_length=self.sequence_length,
            test_start_year=self.test_start_year,
            auto_sequence_length=True,
            split_validation=True,
        )

        self.input_size = len(self.feature_cols)
        self.sequence_length_used = self.X_train.shape[1]

        print(f'Training set: {self.X_train.shape}')
        print(f'Validation set: {self.X_val.shape}')
        print(f'Test set: {self.X_test.shape}')
        print(f'Input features: {self.input_size}')
        print(f'Sequence length: {self.sequence_length_used}')

    def calculate_additional_metrics(self, actual, predicted):
        """Calculate additional metrics similar to individual models."""
        # MAPE (Mean Absolute Percentage Error)
        try:
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        except ZeroDivisionError:
            mape = np.nan

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        try:
            smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
        except ZeroDivisionError:
            smape = np.nan

        # MAPE with log transformation
        try:
            actual_log = np.log1p(np.maximum(actual, 0))
            predicted_log = np.log1p(np.maximum(predicted, 0))
            mape_log = np.mean(np.abs((actual_log - predicted_log) / actual_log)) * 100
        except Exception:
            mape_log = np.nan

        # MASE (Mean Absolute Scaled Error) - simplified version
        try:
            naive_forecast_error = np.mean(np.abs(np.diff(actual)))
            mae = np.mean(np.abs(actual - predicted))
            mase = mae / naive_forecast_error if naive_forecast_error != 0 else np.nan
        except Exception:
            mase = np.nan

        return mape, mase, smape, mape_log

    def run_combination(self, combination_name, models_config, ensemble_methods=['voting', 'stacking', 'blending']):
        """Run a specific neural network combination with selected ensemble methods."""
        print(f'\n{"=" * 80}')
        print(f'TESTING COMBINATION: {combination_name}')
        print(f'Models: {[model for model, enabled in models_config.items() if enabled]}')
        print(f'{"=" * 80}')

        combination_results = {}
        combination_dir = os.path.join(self.output_dir, combination_name.lower())
        os.makedirs(combination_dir, exist_ok=True)

        for method in ensemble_methods:
            print(f'\n{"-" * 60}')
            print(f'Running {method.upper()} Ensemble for {combination_name}')
            print(f'{"-" * 60}')

            start_time = time.time()

            try:
                if method == 'voting':
                    ensemble = VotingEnsemble(models_config, voting_type='weighted', optimize_weights=True)
                elif method == 'stacking':
                    ensemble = StackingEnsemble(models_config, meta_model_type='ridge', cv_folds=3)
                elif method == 'blending':
                    ensemble = BlendingEnsemble(models_config, meta_model_type='ridge', blend_ratio=0.2)
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

    def save_individual_results(self, combination_name, method, results, combination_dir):
        """Save individual ensemble method results."""
        method_dir = os.path.join(combination_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Save comprehensive results to JSON
        with open(os.path.join(method_dir, f'{combination_name.lower()}_{method}_detailed_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Save metrics to CSV
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(os.path.join(method_dir, f'{combination_name.lower()}_{method}_metrics.csv'), index=False)

        # Save predictions to CSV
        actual = results['predictions']['actual_values']
        predicted = results['predictions']['predicted_values']
        pred_df = pd.DataFrame(
            {
                'actual': actual,
                'predicted': predicted,
                'residual': np.array(actual) - np.array(predicted),
                'abs_error': np.abs(np.array(actual) - np.array(predicted)),
                'percentage_error': np.abs((np.array(actual) - np.array(predicted)) / np.array(actual)) * 100,
            }
        )
        pred_df.to_csv(os.path.join(method_dir, f'{combination_name.lower()}_{method}_predictions.csv'), index=False)

    def create_combination_visualizations(self, combination_name, combination_results, combination_dir):
        """Create visualizations for a specific combination."""
        valid_methods = {method: results for method, results in combination_results.items() if 'error' not in results}

        if not valid_methods:
            return

        # 1. Create predictions vs actual scatter plot for each method
        n_methods = len(valid_methods)
        if n_methods == 0:
            return

        fig, axes = plt.subplots(1, min(n_methods, 3), figsize=(5 * min(n_methods, 3), 5))
        if n_methods == 1:
            axes = [axes]
        elif n_methods == 2:
            axes = axes

        for idx, (method, results) in enumerate(valid_methods.items()):
            if idx >= 3:  # Limit to 3 subplots
                break

            ax = axes[idx] if n_methods > 1 else axes[0]

            actual = results['predictions']['actual_values'][:100]  # Limit for clarity
            predicted = results['predictions']['predicted_values'][:100]

            ax.scatter(actual, predicted, alpha=0.6, s=20)
            ax.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{method.upper()} - R² = {results["metrics"]["r2"]:.4f}')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{combination_name} - Predictions vs Actual', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(combination_dir, f'{combination_name.lower()}_predictions_comparison.png'),
            dpi=300,
            bbox_inches='tight',
        )
        plt.close()

        # 2. Create metrics comparison bar chart
        metrics_to_plot = ['rmse', 'mae', 'r2', 'mape']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            methods = []
            values = []

            for method, results in valid_methods.items():
                if metric in results['metrics'] and not np.isnan(results['metrics'][metric]):
                    methods.append(method.upper())
                    values.append(results['metrics'][metric])

            if values:
                bars = ax.bar(methods, values, alpha=0.7)
                ax.set_title(f'{metric.upper()}')
                ax.set_ylabel(metric.upper())

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{value:.4f}', ha='center', va='bottom')

        plt.suptitle(f'{combination_name} - Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(combination_dir, f'{combination_name.lower()}_metrics_comparison.png'),
            dpi=300,
            bbox_inches='tight',
        )
        plt.close()

    def run_all_pairs(self):
        """Run all pair combinations."""
        print('\n' + '=' * 100)
        print('TESTING ALL PAIR COMBINATIONS')
        print('=' * 100)

        for pair_name, models_config in self.pair_combinations.items():
            self.run_combination(pair_name, models_config)

    def run_all_triplets(self):
        """Run all triplet combinations."""
        print('\n' + '=' * 100)
        print('TESTING ALL TRIPLET COMBINATIONS')
        print('=' * 100)

        for triplet_name, models_config in self.triplet_combinations.items():
            self.run_combination(triplet_name, models_config)

    def run_complete_analysis(self):
        """Run complete analysis with all combinations."""
        print('🚀 Running Complete Neural Network Ensemble Analysis')
        print('=' * 100)

        # Run all combinations
        self.run_all_pairs()
        self.run_all_triplets()

        # Create comprehensive summary
        self.create_comprehensive_summary()

        return self.results

    def create_comprehensive_summary(self):
        """Create comprehensive summary with all results."""
        print('\n' + '=' * 120)
        print('CREATING COMPREHENSIVE SUMMARY')
        print('=' * 120)

        # Collect all results
        all_results = []
        for combination, methods_results in self.results.items():
            for method, results in methods_results.items():
                if 'error' not in results:
                    result_row = {
                        'combination': combination,
                        'ensemble_method': method,
                        **results['metrics'],
                        'training_time': results['training_data']['training_time'],
                    }
                    all_results.append(result_row)

        if not all_results:
            print('No valid results to summarize')
            return

        # Create summary DataFrame
        df_results = pd.DataFrame(all_results)

        # Save comprehensive results CSV
        df_results.to_csv(os.path.join(self.output_dir, 'ensemble_comprehensive_results.csv'), index=False)

        # Create summary visualizations
        self.create_summary_visualizations(df_results)

        # Print best results
        self.print_best_results(df_results)

    def create_summary_visualizations(self, df_results):
        """Create summary visualizations across all combinations."""
        # 1. Overall performance heatmap
        plt.figure(figsize=(14, 8))

        # Pivot for heatmap (combinations vs methods)
        pivot_rmse = df_results.pivot(index='combination', columns='ensemble_method', values='rmse')

        sns.heatmap(pivot_rmse, annot=True, fmt='.4f', cmap='RdYlBu_r', cbar_kws={'label': 'RMSE'})
        plt.title('Ensemble Performance Heatmap (RMSE)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ensemble_performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Performance comparison by combination type
        df_results['combination_type'] = df_results['combination'].apply(
            lambda x: 'Pair' if len(x.split('_')) == 2 else 'Triplet'
        )

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['rmse', 'mae', 'r2', 'mape']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # Box plot by combination type and method
            df_metric = df_results.dropna(subset=[metric])
            if not df_metric.empty:
                sns.boxplot(data=df_metric, x='ensemble_method', y=metric, hue='combination_type', ax=ax)
                ax.set_title(f'{metric.upper()} by Ensemble Method and Type')
                ax.legend(title='Combination Type')

        plt.suptitle('Performance Distribution by Combination Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, 'ensemble_performance_distribution.png'), dpi=300, bbox_inches='tight'
        )
        plt.close()

        # 3. Best combinations ranking
        plt.figure(figsize=(12, 8))

        # Get top 10 combinations by RMSE
        top_combinations = df_results.nsmallest(10, 'rmse')

        # Create combination_method label
        top_combinations['combo_method'] = top_combinations['combination'] + '_' + top_combinations['ensemble_method']

        plt.barh(range(len(top_combinations)), top_combinations['rmse'], alpha=0.7)
        plt.yticks(range(len(top_combinations)), top_combinations['combo_method'])
        plt.xlabel('RMSE')
        plt.title('Top 10 Best Performing Ensemble Combinations', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        # Add value labels
        for i, v in enumerate(top_combinations['rmse']):
            plt.text(v, i, f' {v:.4f}', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_ensemble_combinations.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def print_best_results(self, df_results):
        """Print summary of best results."""
        print('\n' + '=' * 80)
        print('BEST ENSEMBLE COMBINATIONS SUMMARY')
        print('=' * 80)

        # Best overall by RMSE
        best_rmse = df_results.loc[df_results['rmse'].idxmin()]
        print(f'🏆 Best RMSE: {best_rmse["combination"]}_{best_rmse["ensemble_method"]}')
        print(f'   RMSE: {best_rmse["rmse"]:.4f}, R²: {best_rmse["r2"]:.4f}, MAE: {best_rmse["mae"]:.4f}')

        # Best R²
        best_r2 = df_results.loc[df_results['r2'].idxmax()]
        print(f'🏆 Best R²: {best_r2["combination"]}_{best_r2["ensemble_method"]}')
        print(f'   R²: {best_r2["r2"]:.4f}, RMSE: {best_r2["rmse"]:.4f}, MAE: {best_r2["mae"]:.4f}')

        # Best by ensemble method
        print('\n📊 Best performance by ensemble method:')
        for method in df_results['ensemble_method'].unique():
            method_data = df_results[df_results['ensemble_method'] == method]
            best_method = method_data.loc[method_data['rmse'].idxmin()]
            print(f'   {method.upper()}: {best_method["combination"]} (RMSE: {best_method["rmse"]:.4f})')

        # Best pair vs triplet
        pairs = df_results[df_results['combination'].str.count('_') == 1]
        triplets = df_results[df_results['combination'].str.count('_') == 2]

        if not pairs.empty:
            best_pair = pairs.loc[pairs['rmse'].idxmin()]
            print(
                f'\n🥇 Best Pair: {best_pair["combination"]}_{best_pair["ensemble_method"]} (RMSE: {best_pair["rmse"]:.4f})'
            )

        if not triplets.empty:
            best_triplet = triplets.loc[triplets['rmse'].idxmin()]
            print(
                f'🥇 Best Triplet: {best_triplet["combination"]}_{best_triplet["ensemble_method"]} (RMSE: {best_triplet["rmse"]:.4f})'
            )

        print(f'\n📁 All results saved to: {self.output_dir}')
        print('📊 Visualizations created:')
        print('   - Individual combination plots in each subdirectory')
        print('   - Overall performance heatmap')
        print('   - Performance distribution by type')
        print('   - Top combinations ranking')


def main():
    """Main function to run the enhanced ensemble analysis."""
    print('🚀 Enhanced Neural Network Ensemble Analysis')
    print('=' * 80)

    # Initialize runner
    runner = EnhancedEnsembleRunner(epochs=30)

    # Prepare data
    runner.prepare_data()

    # Run complete analysis
    results = runner.run_complete_analysis()

    print('\n🎉 Enhanced ensemble analysis completed!')
    print(f'📁 Check {runner.output_dir} for comprehensive results and visualizations')

    return results


if __name__ == '__main__':
    results = main()
