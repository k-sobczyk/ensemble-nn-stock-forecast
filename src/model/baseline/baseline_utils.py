import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from src.model.baseline.last_value_baseline import predict_last_value_for_ticker


def visualize_baseline_comparison(
    ticker_data, ticker, test_start_date='2021-01-01', save_dir='src/model/baseline/output/visualizations'
):
    """Create professional visualization comparing ARIMA and Last Value baselines for thesis presentation."""
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

    # Set professional styling
    plt.style.use('default')
    plt.rcParams.update(
        {
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 1.2,
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'grid.alpha': 0.3,
            'grid.linewidth': 0.8,
        }
    )

    # Create the plot with professional dimensions
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('#fafafa')

    # Professional color palette
    colors = {
        'training': '#2E86C1',  # Professional blue
        'actual': '#1B2631',  # Dark charcoal
        'arima': '#E74C3C',  # Professional red
        'last_value': '#F39C12',  # Professional orange
        'split': '#85929E',  # Subtle gray
    }

    # Plot training data with subtle styling
    ax.plot(train_dates, y_train_arima, label='Historical Data', color=colors['training'], linewidth=2.5, alpha=0.8)

    # Plot actual test values with emphasis
    ax.plot(
        test_dates,
        y_true_arima,
        label='Actual Values',
        color=colors['actual'],
        linewidth=3,
        marker='o',
        markersize=5,
        markerfacecolor='white',
        markeredgewidth=2,
        markeredgecolor=colors['actual'],
    )

    # Plot ARIMA predictions
    ax.plot(
        test_dates,
        y_pred_arima,
        label='ARIMA Forecast',
        color=colors['arima'],
        linewidth=2.5,
        linestyle='--',
        marker='s',
        markersize=4,
        markerfacecolor=colors['arima'],
        alpha=0.9,
    )

    # Plot Last Value predictions
    ax.plot(
        test_dates,
        y_pred_last,
        label='Last Value Forecast',
        color=colors['last_value'],
        linewidth=2.5,
        linestyle=':',
        marker='^',
        markersize=4,
        markerfacecolor=colors['last_value'],
        alpha=0.9,
    )

    # Add professional vertical line at test start
    ax.axvline(
        x=pd.to_datetime(test_start_date),
        color=colors['split'],
        linestyle='-',
        linewidth=2,
        alpha=0.7,
        label='Training/Testing Split',
    )

    # Clean, professional title - only company name
    ax.set_title(f'{ticker}', fontsize=18, fontweight='bold', color='#1B2631', pad=20)

    # Professional axis labels
    ax.set_xlabel('Date', fontsize=14, fontweight='medium', color='#333333')
    ax.set_ylabel('Stock Price (PLN)', fontsize=14, fontweight='medium', color='#333333')

    # Professional legend
    legend = ax.legend(
        fontsize=11,
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor='#dddddd',
        facecolor='white',
    )
    legend.get_frame().set_linewidth(1)

    # Professional grid
    ax.grid(True, alpha=0.3, linewidth=0.8, color='#cccccc')
    ax.set_axisbelow(True)

    # Format x-axis dates professionally
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # Tight layout with padding
    plt.tight_layout(pad=2.0)

    # Save with high quality for thesis
    filename = f'{ticker}_baseline_comparison.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='png')
    plt.close()

    # Reset style to default
    plt.rcParams.update(plt.rcParamsDefault)

    print(f'✓ Professional visualization saved: {filepath}')


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
