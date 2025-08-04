import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.model.baseline.last_value_baseline import predict_last_value_for_ticker
from src.model.metrics.metrics import calculate_mape, calculate_mase


def visualize_baseline_comparison(
    ticker_data, ticker, test_start_date='2021-01-01', save_dir='src/model/baseline/output/visualizations'
):
    """Create visualization comparing ARIMA and Last Value baselines for a ticker."""
    from src.model.baseline.arima_baseline import predict_arima_for_ticker

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get predictions from both methods
    y_true_arima, y_pred_arima, y_train_arima = predict_arima_for_ticker(ticker_data, test_start_date)
    y_true_last, y_pred_last, y_train_last = predict_last_value_for_ticker(ticker_data, test_start_date)

    if y_true_arima is None or y_true_last is None:
        print(f'Skipping visualization for {ticker} - insufficient data')
        return

    # Prepare date information
    ticker_data_sorted = ticker_data.sort_values('end_of_period')
    ticker_data_sorted['end_of_period'] = pd.to_datetime(ticker_data_sorted['end_of_period'])

    train_dates = ticker_data_sorted[ticker_data_sorted['end_of_period'] < test_start_date]['end_of_period']
    test_dates = ticker_data_sorted[ticker_data_sorted['end_of_period'] >= test_start_date]['end_of_period']

    # Calculate metrics for both methods
    rmse_arima = np.sqrt(mean_squared_error(y_true_arima, y_pred_arima))
    mape_arima = calculate_mape(y_true_arima, y_pred_arima)
    mase_arima = calculate_mase(y_true_arima, y_pred_arima, y_train_arima)

    rmse_last = np.sqrt(mean_squared_error(y_true_last, y_pred_last))
    mape_last = calculate_mape(y_true_last, y_pred_last)
    mase_last = calculate_mase(y_true_last, y_pred_last, y_train_last)

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Plot training data
    plt.plot(train_dates, y_train_arima, label='Training Data', color='blue', linewidth=2)

    # Plot actual test values
    plt.plot(test_dates, y_true_arima, label='Actual Test Values', color='black', linewidth=2, marker='o', markersize=4)

    # Plot ARIMA predictions
    plt.plot(
        test_dates,
        y_pred_arima,
        label='ARIMA Predictions',
        color='red',
        linewidth=2,
        linestyle='--',
        marker='s',
        markersize=4,
    )

    # Plot Last Value predictions
    plt.plot(
        test_dates,
        y_pred_last,
        label='Last Value Predictions',
        color='orange',
        linewidth=2,
        linestyle=':',
        marker='^',
        markersize=4,
    )

    # Add vertical line at test start
    plt.axvline(x=pd.to_datetime(test_start_date), color='gray', linestyle='-', alpha=0.7, label='Train/Test Split')

    plt.title(
        f'Baseline Comparison for {ticker}\n'
        f'ARIMA: RMSE={rmse_arima:.4f}, MAPE={mape_arima:.2f}%, MASE={mase_arima:.4f}\n'
        f'Last Value: RMSE={rmse_last:.4f}, MAPE={mape_last:.2f}%, MASE={mase_last:.4f}',
        fontsize=14,
        fontweight='bold',
    )

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Target Value (Original Scale)', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    filename = f'{ticker}_baseline_comparison.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Visualization saved: {filepath}')


def save_best_worst_performers(results_df, model_name, base_save_dir='src/model/baseline/output'):
    """Save best and worst performing ticker results and visualizations to separate folders."""
    # Create directories for best and worst performers
    best_dir = os.path.join(base_save_dir, f'best_performers_{model_name}')
    worst_dir = os.path.join(base_save_dir, f'worst_performers_{model_name}')

    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    # Find best and worst performers by RMSE
    best_idx = results_df['rmse'].idxmin()
    worst_idx = results_df['rmse'].idxmax()

    best_ticker = results_df.loc[best_idx, 'ticker']
    worst_ticker = results_df.loc[worst_idx, 'ticker']

    # Save detailed results
    best_result = results_df.loc[best_idx:best_idx]
    worst_result = results_df.loc[worst_idx:worst_idx]

    best_result.to_csv(os.path.join(best_dir, f'best_performer_{model_name}.csv'), index=False)
    worst_result.to_csv(os.path.join(worst_dir, f'worst_performer_{model_name}.csv'), index=False)

    # Copy visualization images if they exist
    viz_source_dir = os.path.join(base_save_dir, 'visualizations')

    if model_name == 'linear_regression':
        viz_pattern = 'linear_regression_comparison'
    else:
        viz_pattern = 'baseline_comparison'

    # Look for visualization files
    if os.path.exists(viz_source_dir):
        for filename in os.listdir(viz_source_dir):
            if best_ticker in filename and viz_pattern in filename:
                src_path = os.path.join(viz_source_dir, filename)
                dst_path = os.path.join(best_dir, filename)
                shutil.copy2(src_path, dst_path)
                print(f'✓ Copied best performer visualization: {dst_path}')

            if worst_ticker in filename and viz_pattern in filename:
                src_path = os.path.join(viz_source_dir, filename)
                dst_path = os.path.join(worst_dir, filename)
                shutil.copy2(src_path, dst_path)
                print(f'✓ Copied worst performer visualization: {dst_path}')

    print(f'✓ Best performer ({best_ticker}) saved to: {best_dir}')
    print(f'✓ Worst performer ({worst_ticker}) saved to: {worst_dir}')

    return best_ticker, worst_ticker, best_dir, worst_dir
