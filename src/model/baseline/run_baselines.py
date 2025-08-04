import pandas as pd

from src.model.baseline.last_value_baseline import last_value_baseline_evaluation
from src.model.metrics.metrics import calculate_weighted_average_metrics


def run_last_value_baseline(dataset_path='data/datasets/dataset_1_full_features.csv', create_visualizations=True):
    """Run Last Value baseline model evaluation."""
    print('=' * 60)
    print('RUNNING LAST VALUE BASELINE EVALUATION')
    print('=' * 60)
    print(f'Dataset: {dataset_path}')
    print('Test Period: 2021-2022')
    print('=' * 60)

    # Run Last Value baseline
    print('\n🚀 RUNNING LAST VALUE BASELINE...')
    last_value_results = last_value_baseline_evaluation(dataset_path, create_visualizations)

    # Process results
    if last_value_results is not None:
        print('\n' + '=' * 60)
        print('LAST VALUE BASELINE SUMMARY')
        print('=' * 60)

        # Calculate weighted averages (sample-size weighted)
        lv_metrics = calculate_weighted_average_metrics(last_value_results)

        # Create results summary
        summary_data = {
            'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)', 'MAPE_Log (%)', 'MASE', 'R2'],
            'Last Value': [
                f'{lv_metrics["rmse"]:.4f}',
                f'{lv_metrics["mae"]:.4f}',
                f'{lv_metrics["mape"]:.2f}',
                f'{lv_metrics["smape"]:.2f}',
                f'{lv_metrics["mape_log"]:.2f}',
                f'{lv_metrics["mase"]:.4f}',
                f'{lv_metrics["r2"]:.4f}',
            ],
        }

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Performance analysis
        print('\n📊 PERFORMANCE ANALYSIS:')
        print(f'• Last Value processed: {len(last_value_results)} tickers')
        print(f'• Total test samples: {last_value_results["n_test_samples"].sum()}')
        print(f'• Average RMSE: {lv_metrics["rmse"]:.4f}')
        print(f'• Average MAE: {lv_metrics["mae"]:.4f}')
        print(f'• Average R²: {lv_metrics["r2"]:.4f}')
        print(f'• Average MAPE_Log: {lv_metrics["mape_log"]:.2f}%')

        # Save results to output directory
        import os

        output_dir = 'src/model/baseline/output'
        os.makedirs(output_dir, exist_ok=True)

        # Save baseline results
        baseline_results = {
            'model': ['Last Value'],
            'avg_rmse': [lv_metrics['rmse']],
            'avg_mae': [lv_metrics['mae']],
            'avg_mape': [lv_metrics['mape']],
            'avg_smape': [lv_metrics['smape']],
            'avg_mape_log': [lv_metrics['mape_log']],
            'avg_mase': [lv_metrics['mase']],
            'avg_r2': [lv_metrics['r2']],
            'n_successful_tickers': [len(last_value_results)],
            'total_test_samples': [last_value_results['n_test_samples'].sum()],
        }

        results_df = pd.DataFrame(baseline_results)
        summary_path = os.path.join(output_dir, 'last_value_baseline_summary.csv')
        results_df.to_csv(summary_path, index=False)
        print(f'\n💾 Results saved to: {summary_path}')

        # Save detailed metrics for reference
        detailed_metrics = {
            'metric': ['RMSE', 'MAE', 'MAPE', 'SMAPE', 'MAPE_Log', 'MASE', 'R2'],
            'last_value_avg': [
                lv_metrics['rmse'],
                lv_metrics['mae'],
                lv_metrics['mape'],
                lv_metrics['smape'],
                lv_metrics['mape_log'],
                lv_metrics['mase'],
                lv_metrics['r2'],
            ],
        }

        metrics_df = pd.DataFrame(detailed_metrics)
        metrics_path = os.path.join(output_dir, 'last_value_baseline_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f'📊 Detailed metrics saved to: {metrics_path}')

        if create_visualizations:
            print('📊 Visualizations saved to: src/model/baseline/output/visualizations/')

        # Save best and worst performers analysis
        try:
            print('\n📊 SAVING BEST/WORST PERFORMER ANALYSIS...')

            best_lv_idx = last_value_results['rmse'].idxmin()
            worst_lv_idx = last_value_results['rmse'].idxmax()

            detailed_comparison_data = [
                {
                    'model': 'Last Value',
                    'performance': 'Best',
                    'ticker': last_value_results.loc[best_lv_idx, 'ticker'],
                    'rmse': last_value_results.loc[best_lv_idx, 'rmse'],
                    'mae': last_value_results.loc[best_lv_idx, 'mae'],
                    'mape': last_value_results.loc[best_lv_idx, 'mape'],
                    'smape': last_value_results.loc[best_lv_idx, 'smape'],
                    'mape_log': last_value_results.loc[best_lv_idx, 'mape_log'],
                    'mase': last_value_results.loc[best_lv_idx, 'mase'],
                    'r2': last_value_results.loc[best_lv_idx, 'r2'],
                },
                {
                    'model': 'Last Value',
                    'performance': 'Worst',
                    'ticker': last_value_results.loc[worst_lv_idx, 'ticker'],
                    'rmse': last_value_results.loc[worst_lv_idx, 'rmse'],
                    'mae': last_value_results.loc[worst_lv_idx, 'mae'],
                    'mape': last_value_results.loc[worst_lv_idx, 'mape'],
                    'smape': last_value_results.loc[worst_lv_idx, 'smape'],
                    'mape_log': last_value_results.loc[worst_lv_idx, 'mape_log'],
                    'mase': last_value_results.loc[worst_lv_idx, 'mase'],
                    'r2': last_value_results.loc[worst_lv_idx, 'r2'],
                },
            ]

            detailed_comparison_df = pd.DataFrame(detailed_comparison_data)
            detailed_comparison_path = os.path.join(output_dir, 'last_value_best_worst_performers.csv')
            detailed_comparison_df.to_csv(detailed_comparison_path, index=False)
            print(f'📊 Best/worst performers saved to: {detailed_comparison_path}')

        except Exception as e:
            print(f'⚠ Warning: Could not save detailed comparison: {e}')

        return results_df
    else:
        print('❌ Last Value baseline evaluation failed!')
        return None


def run_single_baseline(dataset_path='data/datasets/dataset_1_full_features.csv', create_visualizations=True):
    """Run the Last Value baseline model."""
    print('Running Last Value baseline...')
    return last_value_baseline_evaluation(dataset_path, create_visualizations)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Last Value baseline model for stock forecasting')
    parser.add_argument(
        '--dataset', type=str, default='data/datasets/dataset_1_full_features.csv', help='Path to the dataset'
    )
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization generation')

    args = parser.parse_args()

    create_visualizations = not args.no_viz

    results = run_last_value_baseline(args.dataset, create_visualizations)

    print('\n✅ Last Value baseline evaluation completed!')
