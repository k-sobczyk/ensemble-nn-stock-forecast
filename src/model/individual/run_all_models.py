import json
import os
import random
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.model.individual.bi_lstm import main as bi_lstm_main
from src.model.individual.cnn import main as cnn_main
from src.model.individual.gru import main as gru_main

# Import individual models
from src.model.individual.lstm import main as lstm_main

# Import utilities
from src.model.individual.model_utils import calculate_additional_metrics

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """Set random seeds for reproducibility across all models."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def determine_best_model(model_summaries):
    """Determine the best model based on R² score (higher is better)."""
    df = pd.DataFrame(model_summaries)
    # Filter out failed models
    df_valid = df.dropna(subset=['R²'])

    if len(df_valid) == 0:
        return None

    # Find model with highest R² score
    best_idx = df_valid['R²'].idxmax()
    best_model = df_valid.loc[best_idx, 'Model']
    best_r2 = df_valid.loc[best_idx, 'R²']

    return best_model, best_r2


def create_detailed_prediction_plot(actual, predicted, model_name, output_dir):
    """Create detailed prediction vs actual plot for the best model."""
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    plt.scatter(actual, predicted, alpha=0.6, s=30, label='Predictions')

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # Calculate R²
    from sklearn.metrics import r2_score

    r2 = r2_score(actual, predicted)

    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{model_name} - Predictions vs Actual Values\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics text
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}'
    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'{model_name.lower()}_detailed_predictions.png'), dpi=300, bbox_inches='tight'
    )
    plt.close()


def create_residuals_plot(actual, predicted, model_name, output_dir):
    """Create residuals analysis plot for the best model."""
    residuals = actual - predicted

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Residuals Analysis', fontsize=16, fontweight='bold')

    # Residuals vs Predicted
    axes[0, 0].scatter(predicted, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot (approximate)
    stats.probplot(residuals, dist='norm', plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals vs Index (time series pattern)
    axes[1, 1].plot(residuals, alpha=0.7)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Observation Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_residuals_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def run_all_models(
    sequence_length=None,
    auto_sequence_length=True,
    epochs=50,
    early_stopping_patience=10,
    save_detailed_results=True,
    seed=42,
):
    # Set random seeds for reproducibility
    set_seed(seed)

    print('=' * 80)
    print('RUNNING ALL INDIVIDUAL NEURAL NETWORK MODELS')
    print('=' * 80)
    print('Configuration:')
    print(f'- Sequence Length: {"Auto" if auto_sequence_length else sequence_length}')
    print(f'- Epochs: {epochs}')
    print(f'- Early Stopping Patience: {early_stopping_patience}')
    print(f'- Random Seed: {seed}')
    print(f'- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)

    # Create results directory
    results_dir = 'src/model/individual/output'
    os.makedirs(results_dir, exist_ok=True)

    # Define models to run
    models_config = [
        {'name': 'LSTM', 'function': lstm_main, 'color': '#1f77b4'},
        {'name': 'GRU', 'function': gru_main, 'color': '#ff7f0e'},
        {'name': 'Bi-LSTM', 'function': bi_lstm_main, 'color': '#2ca02c'},
        {'name': 'CNN', 'function': cnn_main, 'color': '#d62728'},
    ]

    all_results = {}
    model_summaries = []

    for i, model_config in enumerate(models_config, 1):
        model_name = model_config['name']
        model_function = model_config['function']

        print(f'\n[{i}/4] Running {model_name} Model...')
        print('-' * 50)

        start_time = time.time()

        try:
            # Run the model
            model, results = model_function(
                sequence_length=sequence_length,
                auto_sequence_length=auto_sequence_length,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience,
            )

            end_time = time.time()
            training_time = end_time - start_time

            if model is not None and results is not None:
                # Calculate additional metrics
                actual = results['actual']
                predicted = results['predictions']
                mape, mase = calculate_additional_metrics(actual, predicted)

                # Store results
                all_results[model_name.lower()] = {
                    'model': model,
                    'results': results,
                    'training_time': training_time,
                    'timestamp': datetime.now().isoformat(),
                    'mape': mape,
                    'mase': mase,
                }

                # Note: Individual model results with training losses are saved by each model's main() function

                # Create summary for comparison
                model_summaries.append(
                    {
                        'Model': model_name,
                        'RMSE': results['rmse'],
                        'MAE': results['mae'],
                        'R²': results['r2'],
                        'MAPE': mape,
                        'MASE': mase,
                        'Training Time (min)': training_time / 60,
                        'Color': model_config['color'],
                    }
                )

                print(f'✅ {model_name} completed successfully!')
                print(f'   RMSE: {results["rmse"]:.4f}')
                print(f'   MAE: {results["mae"]:.4f}')
                print(f'   R²: {results["r2"]:.4f}')
                print(f'   MAPE: {mape:.4f}%' if not np.isnan(mape) else '   MAPE: N/A')
                print(f'   MASE: {mase:.4f}' if not np.isnan(mase) else '   MASE: N/A')
                print(f'   Training Time: {training_time / 60:.2f} minutes')

            else:
                print(f'❌ {model_name} failed to complete!')
                model_summaries.append(
                    {
                        'Model': model_name,
                        'RMSE': np.nan,
                        'MAE': np.nan,
                        'R²': np.nan,
                        'MAPE': np.nan,
                        'MASE': np.nan,
                        'Training Time (min)': training_time / 60,
                        'Color': model_config['color'],
                    }
                )

        except Exception as e:
            print(f'❌ Error running {model_name}: {str(e)}')
            end_time = time.time()
            training_time = end_time - start_time
            model_summaries.append(
                {
                    'Model': model_name,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'R²': np.nan,
                    'MAPE': np.nan,
                    'MASE': np.nan,
                    'Training Time (min)': training_time / 60,
                    'Color': model_config['color'],
                }
            )

    # Create and save comparison results
    if save_detailed_results:
        save_comparison_results(model_summaries, all_results, results_dir)

    print('\n' + '=' * 80)
    print('ALL MODELS COMPLETED')
    print('=' * 80)

    # Print summary table
    df_summary = pd.DataFrame(model_summaries)
    print('\nModel Performance Summary:')
    print(df_summary.drop('Color', axis=1).to_string(index=False, float_format='%.4f'))

    return all_results


def save_comparison_results(model_summaries, all_results, results_dir):
    """Save comparison plots and summary results."""
    # Create comparison plots directory
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Convert to DataFrame for easier handling
    df_summary = pd.DataFrame(model_summaries)

    # Filter out failed models for plotting
    df_valid = df_summary.dropna(subset=['RMSE', 'MAE', 'R²', 'MAPE', 'MASE'])

    if len(df_valid) == 0:
        print('⚠️ No models completed successfully - skipping comparison plots')
        return

    # Create performance comparison plots
    create_performance_comparison_plots(df_valid, comparison_dir)

    # Create training time comparison
    create_training_time_plot(df_summary, comparison_dir)

    # Create predictions comparison plot
    create_predictions_comparison_plot(all_results, comparison_dir)

    # Save detailed results to JSON
    save_detailed_results_json(all_results, model_summaries, comparison_dir)

    # Save summary CSV
    df_summary.drop('Color', axis=1).to_csv(os.path.join(comparison_dir, 'model_summary.csv'), index=False)

    print(f'📊 Comparison results saved to: {comparison_dir}')


def create_performance_comparison_plots(df_valid, output_dir):
    """Create performance comparison bar charts."""
    # Metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # RMSE comparison
    axes[0, 0].bar(df_valid['Model'], df_valid['RMSE'], color=df_valid['Color'], alpha=0.7)
    axes[0, 0].set_title('Root Mean Square Error (RMSE)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # MAE comparison
    axes[0, 1].bar(df_valid['Model'], df_valid['MAE'], color=df_valid['Color'], alpha=0.7)
    axes[0, 1].set_title('Mean Absolute Error (MAE)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # R² comparison
    axes[0, 2].bar(df_valid['Model'], df_valid['R²'], color=df_valid['Color'], alpha=0.7)
    axes[0, 2].set_title('R² Score')
    axes[0, 2].set_ylabel('R²')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # MAPE comparison
    axes[1, 0].bar(df_valid['Model'], df_valid['MAPE'], color=df_valid['Color'], alpha=0.7)
    axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE)')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # MASE comparison
    axes[1, 1].bar(df_valid['Model'], df_valid['MASE'], color=df_valid['Color'], alpha=0.7)
    axes[1, 1].set_title('Mean Absolute Scaled Error (MASE)')
    axes[1, 1].set_ylabel('MASE')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Combined metrics (normalized) - last subplot
    # Normalize metrics for combined view (lower is better for RMSE/MAE/MAPE/MASE, higher for R²)
    rmse_norm = (
        (df_valid['RMSE'] - df_valid['RMSE'].min()) / (df_valid['RMSE'].max() - df_valid['RMSE'].min())
        if df_valid['RMSE'].max() != df_valid['RMSE'].min()
        else [0] * len(df_valid)
    )
    mae_norm = (
        (df_valid['MAE'] - df_valid['MAE'].min()) / (df_valid['MAE'].max() - df_valid['MAE'].min())
        if df_valid['MAE'].max() != df_valid['MAE'].min()
        else [0] * len(df_valid)
    )
    # R² is inverted (higher is better)
    r2_norm = (
        (df_valid['R²'] - df_valid['R²'].min()) / (df_valid['R²'].max() - df_valid['R²'].min())
        if df_valid['R²'].max() != df_valid['R²'].min()
        else [1] * len(df_valid)
    )

    x = np.arange(len(df_valid))
    width = 0.15

    axes[1, 2].bar(x - 2 * width, rmse_norm, width, label='RMSE', alpha=0.7)
    axes[1, 2].bar(x - width, mae_norm, width, label='MAE', alpha=0.7)
    axes[1, 2].bar(x, r2_norm, width, label='R²', alpha=0.7)
    axes[1, 2].set_title('Normalized Metrics Overview')
    axes[1, 2].set_ylabel('Normalized Score (0-1)')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(df_valid['Model'], rotation=45)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_training_time_plot(df_summary, output_dir):
    """Create training time comparison plot."""
    plt.figure(figsize=(10, 6))

    # Include all models (even failed ones) for training time
    colors = [row['Color'] for _, row in df_summary.iterrows()]
    bars = plt.bar(df_summary['Model'], df_summary['Training Time (min)'], color=colors, alpha=0.7)

    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Training Time (minutes)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, time_val in zip(bars, df_summary['Training Time (min)']):
        if not np.isnan(time_val):
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{time_val:.1f}m', ha='center', va='bottom'
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results_json(all_results, model_summaries, output_dir):
    """Save detailed results to JSON file."""
    # Prepare data for JSON serialization
    json_data = {'timestamp': datetime.now().isoformat(), 'summary': []}

    for summary in model_summaries:
        json_summary = summary.copy()
        # Convert numpy types to Python types for JSON serialization
        for key, value in json_summary.items():
            if isinstance(value, np.integer | np.floating | np.ndarray):
                if np.isnan(value):
                    json_summary[key] = None
                else:
                    json_summary[key] = float(value)
        json_data['summary'].append(json_summary)

    # Save detailed metrics for each model
    json_data['detailed_results'] = {}
    for model_name, model_data in all_results.items():
        if 'results' in model_data:
            results = model_data['results']
            json_data['detailed_results'][model_name] = {
                'rmse': float(results['rmse']),
                'mae': float(results['mae']),
                'r2': float(results['r2']),
                'mape': float(model_data.get('mape', np.nan)),
                'mase': float(model_data.get('mase', np.nan)),
                'training_time_minutes': float(model_data['training_time'] / 60),
                'timestamp': model_data['timestamp'],
            }

    # Save to JSON file
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(json_data, f, indent=2)


def create_predictions_comparison_plot(all_results, output_dir):
    """Create a plot comparing predictions vs actual values for all models."""
    valid_models = {
        name: data for name, data in all_results.items() if 'results' in data and 'predictions' in data['results']
    }

    if len(valid_models) == 0:
        print('⚠️ No valid prediction data available for comparison plot')
        return

    # Create subplot for each model
    n_models = len(valid_models)
    cols = 2
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, model_data) in enumerate(valid_models.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        results = model_data['results']
        actual = results['actual'][:100]  # Limit to first 100 points for clarity
        predicted = results['predictions'][:100]

        ax.scatter(actual, predicted, alpha=0.6, s=20)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name.upper()} - R² = {results["r2"]:.4f}')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        elif cols > 1:
            axes[col].set_visible(False)

    plt.suptitle('Predictions vs Actual Values Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Run all models with default settings
    results = run_all_models(
        sequence_length=None,  # Auto-select optimal sequence length
        auto_sequence_length=True,
        epochs=100,  # Reduce for testing, increase for production
        early_stopping_patience=20,
        save_detailed_results=True,
        seed=42,
    )

    print('\n🎉 All models completed! Check the results folder for detailed output.')
