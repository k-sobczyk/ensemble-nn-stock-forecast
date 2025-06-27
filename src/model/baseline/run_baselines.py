import pandas as pd

from src.model.baseline.arima_baseline import arima_baseline_evaluation
from src.model.baseline.last_value_baseline import last_value_baseline_evaluation


def compare_baselines(dataset_path='data/datasets/dataset_1_full_features.csv', create_visualizations=True):
    """Run both baseline models and compare results."""
    print('=' * 60)
    print('RUNNING BASELINE MODEL COMPARISON')
    print('=' * 60)
    print(f'Dataset: {dataset_path}')
    print('Test Period: 2021-2022')
    print('=' * 60)

    # Run Last Value baseline (faster)
    print('\n🚀 RUNNING LAST VALUE BASELINE...')
    last_value_results = last_value_baseline_evaluation(dataset_path, create_visualizations)

    # Run ARIMA baseline (slower)
    print('\n🚀 RUNNING ARIMA BASELINE...')
    arima_results = arima_baseline_evaluation(dataset_path, create_visualizations)

    # Compare results
    if last_value_results is not None and arima_results is not None:
        print('\n' + '=' * 60)
        print('BASELINE COMPARISON SUMMARY')
        print('=' * 60)

        # Create comparison table (only RMSE, MAPE, MASE)
        comparison_data = {
            'Metric': ['RMSE', 'MAPE (%)', 'MASE'],
            'Last Value': [
                f'{last_value_results["rmse"].mean():.4f}',
                f'{last_value_results["mape"].mean():.2f}',
                f'{last_value_results["mase"].mean():.4f}',
            ],
            'ARIMA': [
                f'{arima_results["rmse"].mean():.4f}',
                f'{arima_results["mape"].mean():.2f}',
                f'{arima_results["mase"].mean():.4f}',
            ],
        }

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Winner analysis
        print('\n📊 PERFORMANCE ANALYSIS:')
        print(f'• Last Value processed: {len(last_value_results)} tickers')
        print(f'• ARIMA processed: {len(arima_results)} tickers')

        # Determine winner for each metric (lower is better for all metrics)
        lv_rmse = last_value_results['rmse'].mean()
        ar_rmse = arima_results['rmse'].mean()
        rmse_winner = 'Last Value' if lv_rmse < ar_rmse else 'ARIMA'

        lv_mape = last_value_results['mape'].mean()
        ar_mape = arima_results['mape'].mean()
        mape_winner = 'Last Value' if lv_mape < ar_mape else 'ARIMA'

        lv_mase = last_value_results['mase'].mean()
        ar_mase = arima_results['mase'].mean()
        mase_winner = 'Last Value' if lv_mase < ar_mase else 'ARIMA'

        print(f'• RMSE Winner: {rmse_winner} ({lv_rmse:.4f} vs {ar_rmse:.4f})')
        print(f'• MAPE Winner: {mape_winner} ({lv_mape:.2f}% vs {ar_mape:.2f}%)')
        print(f'• MASE Winner: {mase_winner} ({lv_mase:.4f} vs {ar_mase:.4f})')

        # Overall winner (based on number of metrics won)
        wins = {'Last Value': 0, 'ARIMA': 0}
        wins[rmse_winner] += 1
        wins[mape_winner] += 1
        wins[mase_winner] += 1

        overall_winner = max(wins, key=wins.get)
        print(f'\n🏆 OVERALL WINNER: {overall_winner} ({wins[overall_winner]}/3 metrics)')

        # Save combined results
        combined_results = {
            'model': ['Last Value', 'ARIMA'],
            'avg_rmse': [lv_rmse, ar_rmse],
            'avg_mape': [lv_mape, ar_mape],
            'avg_mase': [lv_mase, ar_mase],
            'n_successful_tickers': [len(last_value_results), len(arima_results)],
        }

        combined_df = pd.DataFrame(combined_results)
        combined_df.to_csv('baseline_comparison_summary.csv', index=False)
        print('\n💾 Combined results saved to: baseline_comparison_summary.csv')

        if create_visualizations:
            print('📊 Visualizations saved to: src/model/baseline/visualization_images/')

        return combined_df
    else:
        print('❌ One or both baseline evaluations failed!')
        return None


def run_single_baseline(
    model_type='last_value', dataset_path='data/datasets/dataset_1_full_features.csv', create_visualizations=True
):
    """Run a single baseline model."""
    if model_type.lower() == 'last_value':
        print('Running Last Value baseline...')
        return last_value_baseline_evaluation(dataset_path, create_visualizations)
    elif model_type.lower() == 'arima':
        print('Running ARIMA baseline...')
        return arima_baseline_evaluation(dataset_path, create_visualizations)
    else:
        print(f'Unknown model type: {model_type}')
        print("Available options: 'last_value', 'arima'")
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline models for stock forecasting')
    parser.add_argument(
        '--model', type=str, choices=['last_value', 'arima', 'both'], default='both', help='Which baseline model to run'
    )
    parser.add_argument(
        '--dataset', type=str, default='data/datasets/dataset_1_full_features.csv', help='Path to the dataset'
    )
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization generation')

    args = parser.parse_args()

    create_visualizations = not args.no_viz

    if args.model == 'both':
        results = compare_baselines(args.dataset, create_visualizations)
    else:
        results = run_single_baseline(args.model, args.dataset, create_visualizations)

    print('\n✅ Baseline evaluation completed!')
