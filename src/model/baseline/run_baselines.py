import pandas as pd

from src.model.baseline.arima_baseline import arima_baseline_evaluation
from src.model.baseline.last_value_baseline import last_value_baseline_evaluation
from src.model.metrics.metrics import calculate_weighted_average_metrics


def compare_baselines(dataset_path='data/datasets/dataset_1_full_features.csv', create_visualizations=True):
    """Run both baseline models and compare results."""
    print('=' * 60)
    print('RUNNING BASELINE MODEL COMPARISON')
    print('=' * 60)
    print(f'Dataset: {dataset_path}')
    print('Test Period: 2021-2022')
    print('=' * 60)

    # Run ARIMA baseline (slower)
    print('\n🚀 RUNNING ARIMA BASELINE...')
    arima_results = arima_baseline_evaluation(dataset_path, '2021-01-01', create_visualizations)

    # Run Last Value baseline (faster)
    print('\n🚀 RUNNING LAST VALUE BASELINE...')
    last_value_results = last_value_baseline_evaluation(dataset_path, create_visualizations)

    # Compare results
    if last_value_results is not None and arima_results is not None:
        print('\n' + '=' * 60)
        print('BASELINE COMPARISON SUMMARY')
        print('=' * 60)

        # Calculate weighted averages (sample-size weighted)
        lv_metrics = calculate_weighted_average_metrics(last_value_results)
        ar_metrics = calculate_weighted_average_metrics(arima_results)

        # Create comparison table
        comparison_data = {
            'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'MASE', 'R2'],
            'Last Value': [
                f'{lv_metrics["rmse"]:.4f}',
                f'{lv_metrics["mae"]:.4f}',
                f'{lv_metrics["mape"]:.2f}',
                f'{lv_metrics["mase"]:.4f}',
                f'{lv_metrics["r2"]:.4f}',
            ],
            'ARIMA': [
                f'{ar_metrics["rmse"]:.4f}',
                f'{ar_metrics["mae"]:.4f}',
                f'{ar_metrics["mape"]:.2f}',
                f'{ar_metrics["mase"]:.4f}',
                f'{ar_metrics["r2"]:.4f}',
            ],
        }

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Winner analysis
        print('\n📊 PERFORMANCE ANALYSIS:')
        print(f'• Last Value processed: {len(last_value_results)} tickers')
        print(f'• ARIMA processed: {len(arima_results)} tickers')
        print(f'• Last Value total test samples: {last_value_results["n_test_samples"].sum()}')
        print(f'• ARIMA total test samples: {arima_results["n_test_samples"].sum()}')

        # Winner analysis (using sample-size weighted averages)
        print('\n🏆 WINNER ANALYSIS:')

        # Determine winner for each metric (lower is better for RMSE, MAE, MAPE, MASE; higher is better for R2)
        lv_rmse = lv_metrics['rmse']
        ar_rmse = ar_metrics['rmse']
        rmse_winner = 'Last Value' if lv_rmse < ar_rmse else 'ARIMA'

        lv_mae = lv_metrics['mae']
        ar_mae = ar_metrics['mae']
        mae_winner = 'Last Value' if lv_mae < ar_mae else 'ARIMA'

        lv_mape = lv_metrics['mape']
        ar_mape = ar_metrics['mape']
        mape_winner = 'Last Value' if lv_mape < ar_mape else 'ARIMA'

        lv_mase = lv_metrics['mase']
        ar_mase = ar_metrics['mase']
        mase_winner = 'Last Value' if lv_mase < ar_mase else 'ARIMA'

        lv_r2 = lv_metrics['r2']
        ar_r2 = ar_metrics['r2']
        r2_winner = 'Last Value' if lv_r2 > ar_r2 else 'ARIMA'  # Higher R2 is better

        print(f'• RMSE Winner: {rmse_winner} ({lv_rmse:.4f} vs {ar_rmse:.4f})')
        print(f'• MAE Winner: {mae_winner} ({lv_mae:.4f} vs {ar_mae:.4f})')
        print(f'• MAPE Winner: {mape_winner} ({lv_mape:.2f}% vs {ar_mape:.2f}%)')
        print(f'• MASE Winner: {mase_winner} ({lv_mase:.4f} vs {ar_mase:.4f})')
        print(f'• R2 Winner: {r2_winner} ({lv_r2:.4f} vs {ar_r2:.4f})')

        # Overall winner (based on number of metrics won)
        wins = {'Last Value': 0, 'ARIMA': 0}
        wins[rmse_winner] += 1
        wins[mae_winner] += 1
        wins[mape_winner] += 1
        wins[mase_winner] += 1
        wins[r2_winner] += 1

        overall_winner = max(wins, key=wins.get)
        print(f'\n🏆 OVERALL WINNER: {overall_winner} ({wins[overall_winner]}/5 metrics)')

        # Save combined results (sample-size weighted averages)
        combined_results = {
            'model': ['Last Value', 'ARIMA'],
            'avg_rmse': [lv_metrics['rmse'], ar_metrics['rmse']],
            'avg_mae': [lv_metrics['mae'], ar_metrics['mae']],
            'avg_mape': [lv_metrics['mape'], ar_metrics['mape']],
            'avg_mase': [lv_metrics['mase'], ar_metrics['mase']],
            'avg_r2': [lv_metrics['r2'], ar_metrics['r2']],
            'n_successful_tickers': [len(last_value_results), len(arima_results)],
            'total_test_samples': [last_value_results['n_test_samples'].sum(), arima_results['n_test_samples'].sum()],
        }

        combined_df = pd.DataFrame(combined_results)
        # Save to output directory
        import os

        output_dir = 'src/model/baseline/output'
        os.makedirs(output_dir, exist_ok=True)

        summary_path = os.path.join(output_dir, 'baseline_comparison_summary.csv')
        combined_df.to_csv(summary_path, index=False)
        print(f'\n💾 Combined results saved to: {summary_path}')

        # Save baseline averages for easy reference
        baseline_averages = {
            'metric': ['RMSE', 'MAE', 'MAPE', 'MASE', 'R2'],
            'last_value_avg': [
                lv_metrics['rmse'],
                lv_metrics['mae'],
                lv_metrics['mape'],
                lv_metrics['mase'],
                lv_metrics['r2'],
            ],
            'arima_avg': [
                ar_metrics['rmse'],
                ar_metrics['mae'],
                ar_metrics['mape'],
                ar_metrics['mase'],
                ar_metrics['r2'],
            ],
            'best_model': [rmse_winner, mae_winner, mape_winner, mase_winner, r2_winner],
        }

        baseline_avg_df = pd.DataFrame(baseline_averages)
        avg_metrics_path = os.path.join(output_dir, 'baseline_average_metrics.csv')
        baseline_avg_df.to_csv(avg_metrics_path, index=False)
        print(f'📊 Baseline average metrics saved to: {avg_metrics_path}')

        if create_visualizations:
            print('📊 Visualizations saved to: src/model/baseline/output/visualizations/')

        # Save detailed comparison of best and worst performers across both models
        try:
            print('\n📊 SAVING BEST/WORST PERFORMER ANALYSIS...')

            # Create comprehensive comparison
            detailed_comparison_data = []

            if last_value_results is not None:
                best_lv_idx = last_value_results['rmse'].idxmin()
                worst_lv_idx = last_value_results['rmse'].idxmax()

                detailed_comparison_data.extend(
                    [
                        {
                            'model': 'Last Value',
                            'performance': 'Best',
                            'ticker': last_value_results.loc[best_lv_idx, 'ticker'],
                            'rmse': last_value_results.loc[best_lv_idx, 'rmse'],
                            'mae': last_value_results.loc[best_lv_idx, 'mae'],
                            'mape': last_value_results.loc[best_lv_idx, 'mape'],
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
                            'mase': last_value_results.loc[worst_lv_idx, 'mase'],
                            'r2': last_value_results.loc[worst_lv_idx, 'r2'],
                        },
                    ]
                )

            if arima_results is not None:
                best_ar_idx = arima_results['rmse'].idxmin()
                worst_ar_idx = arima_results['rmse'].idxmax()

                detailed_comparison_data.extend(
                    [
                        {
                            'model': 'ARIMA',
                            'performance': 'Best',
                            'ticker': arima_results.loc[best_ar_idx, 'ticker'],
                            'rmse': arima_results.loc[best_ar_idx, 'rmse'],
                            'mae': arima_results.loc[best_ar_idx, 'mae'],
                            'mape': arima_results.loc[best_ar_idx, 'mape'],
                            'mase': arima_results.loc[best_ar_idx, 'mase'],
                            'r2': arima_results.loc[best_ar_idx, 'r2'],
                        },
                        {
                            'model': 'ARIMA',
                            'performance': 'Worst',
                            'ticker': arima_results.loc[worst_ar_idx, 'ticker'],
                            'rmse': arima_results.loc[worst_ar_idx, 'rmse'],
                            'mae': arima_results.loc[worst_ar_idx, 'mae'],
                            'mape': arima_results.loc[worst_ar_idx, 'mape'],
                            'mase': arima_results.loc[worst_ar_idx, 'mase'],
                            'r2': arima_results.loc[worst_ar_idx, 'r2'],
                        },
                    ]
                )

            if detailed_comparison_data:
                detailed_comparison_df = pd.DataFrame(detailed_comparison_data)
                detailed_comparison_path = os.path.join(output_dir, 'detailed_metrics_comparison.csv')
                detailed_comparison_df.to_csv(detailed_comparison_path, index=False)
                print(f'📊 Detailed best/worst comparison saved to: {detailed_comparison_path}')

        except Exception as e:
            print(f'⚠ Warning: Could not save detailed comparison: {e}')

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
        return arima_baseline_evaluation(dataset_path, '2021-01-01', create_visualizations)
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
