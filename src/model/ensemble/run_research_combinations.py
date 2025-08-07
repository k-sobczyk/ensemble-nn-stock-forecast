#!/usr/bin/env python3
"""Research-specific ensemble combinations runner with professional academic visualizations.

This script runs the exact neural network combinations specified in the research:
- Pair Combinations: LSTM+GRU, LSTM+CNN, GRU+BiLSTM, GRU+CNN, BiLSTM+CNN
- Triplet Combinations: LSTM+GRU+CNN, GRU+BiLSTM+CNN
- Ensemble Methods: Voting, Stacking, Blending for each combination

Features:
- Uses optimized hyperparameters from ensemble_config.py (found through Optuna optimization)
- Professional academic visualizations matching individual models style
- Training history plots, residuals analysis, detailed predictions
- Publication-ready PNG and PDF outputs for research papers
- Comprehensive CSV files and statistical analysis
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

from src.model.ensemble.enhanced_ensemble_runner import EnhancedEnsembleRunner

# ============================================================================
# PROFESSIONAL ACADEMIC VISUALIZATION FUNCTIONS
# ============================================================================


def set_professional_style():
    """Set professional styling matching individual models."""
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


def reset_style():
    """Reset matplotlib style to default."""
    plt.rcParams.update(plt.rcParamsDefault)


def create_professional_training_history_plot(
    train_losses, validation_losses, combination_name, method_name, output_dir
):
    """Create detailed training history plot with professional styling matching individual models."""
    set_professional_style()

    # Create the plot with professional dimensions
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('#fafafa')

    # Professional color palette (matching individual models)
    colors = {
        'training': '#2E86C1',  # Professional blue
        'validation': '#E74C3C',  # Professional red
        'best_marker': '#F39C12',  # Professional orange
        'best_line': '#85929E',  # Subtle gray
    }

    epochs = range(1, len(train_losses) + 1)

    # Plot training loss with professional styling
    ax.plot(
        epochs,
        train_losses,
        color=colors['training'],
        linewidth=2.5,
        label='Training Loss',
        alpha=0.8,
        marker='o',
        markersize=3,
        markerfacecolor=colors['training'],
        markevery=max(1, len(epochs) // 20),
    )

    # Plot validation loss with professional styling
    ax.plot(
        epochs,
        validation_losses,
        color=colors['validation'],
        linewidth=2.5,
        linestyle='--',
        label='Validation Loss',
        alpha=0.9,
        marker='s',
        markersize=3,
        markerfacecolor=colors['validation'],
        markevery=max(1, len(epochs) // 20),
    )

    # Find best validation loss epoch
    best_val_epoch = np.argmin(validation_losses) + 1
    best_val_loss = min(validation_losses)

    # Mark best validation loss with professional styling
    ax.axvline(
        x=best_val_epoch,
        color=colors['best_line'],
        linestyle='-',
        alpha=0.7,
        linewidth=2,
        label='Best Validation Epoch',
    )
    ax.plot(
        best_val_epoch,
        best_val_loss,
        color=colors['best_marker'],
        marker='^',
        markersize=8,
        markerfacecolor=colors['best_marker'],
        markeredgewidth=2,
        markeredgecolor='white',
        label=f'Best Validation (Epoch {best_val_epoch})',
    )

    # Clean, professional title
    ax.set_title(
        f'{combination_name} {method_name.title()} Ensemble Training History',
        fontsize=18,
        fontweight='bold',
        color='#1B2631',
        pad=20,
    )

    # Professional axis labels
    ax.set_xlabel('Epoch', fontsize=14, fontweight='medium', color='#333333')
    ax.set_ylabel('Loss', fontsize=14, fontweight='medium', color='#333333')

    # Professional legend
    legend = ax.legend(
        fontsize=11,
        loc='upper right',
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

    # Format axes professionally
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # Add statistics text with professional styling
    final_train_loss = train_losses[-1]
    final_val_loss = validation_losses[-1]
    best_train_loss = min(train_losses)

    stats_text = f'Final Train Loss: {final_train_loss:.6f}\n'
    stats_text += f'Final Val Loss: {final_val_loss:.6f}\n'
    stats_text += f'Best Train Loss: {best_train_loss:.6f}\n'
    stats_text += f'Best Val Loss: {best_val_loss:.6f}\n'
    stats_text += f'Total Epochs: {len(train_losses)}'

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='#dddddd'),
        fontsize=10,
        color='#333333',
    )

    # Tight layout with padding
    plt.tight_layout(pad=2.0)

    # Save with high quality for academic use
    filepath = os.path.join(output_dir, f'{combination_name.lower()}_{method_name.lower()}_training_history.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='png')
    plt.close()

    reset_style()


def create_detailed_prediction_plot(actual, predicted, combination_name, method_name, output_dir):
    """Create detailed prediction vs actual plot matching individual models style."""
    set_professional_style()

    plt.figure(figsize=(12, 8), facecolor='white')

    # Create scatter plot with professional styling
    plt.scatter(actual, predicted, alpha=0.6, s=30, label='Predictions', color='#2E86C1')

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # Calculate R²
    r2 = r2_score(actual, predicted)

    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(
        f'{combination_name} {method_name.title()} Ensemble - Predictions vs Actual Values\nR² = {r2:.4f}',
        fontsize=14,
        fontweight='bold',
    )
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
    filepath = os.path.join(output_dir, f'{combination_name.lower()}_{method_name.lower()}_detailed_predictions.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    reset_style()


def create_residuals_analysis_plot(actual, predicted, combination_name, method_name, output_dir):
    """Create residuals analysis plot matching individual models style."""
    set_professional_style()

    residuals = actual - predicted

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='white')
    fig.suptitle(
        f'{combination_name} {method_name.title()} Ensemble - Residuals Analysis', fontsize=16, fontweight='bold'
    )

    # Residuals vs Predicted
    axes[0, 0].scatter(predicted, residuals, alpha=0.6, color='#2E86C1')
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='#2E86C1')
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(residuals, dist='norm', plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals vs Index (time series pattern)
    axes[1, 1].plot(residuals, alpha=0.7, color='#2E86C1')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Observation Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{combination_name.lower()}_{method_name.lower()}_residuals_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    reset_style()


def create_research_summary_plots(all_results, output_dir):
    """Create comprehensive research summary plots."""
    # Collect all valid results
    results_data = []
    for combination, methods_results in all_results.items():
        for method, results in methods_results.items():
            if 'error' not in results:
                results_data.append(
                    {
                        'combination': combination,
                        'method': method,
                        'rmse': results['metrics']['rmse'],
                        'mae': results['metrics']['mae'],
                        'r2': results['metrics']['r2'],
                        'mape': results['metrics'].get('mape', 0),
                        'training_time': results['training_data']['training_time'],
                    }
                )

    if not results_data:
        return

    df = pd.DataFrame(results_data)

    # 1. Performance heatmap with professional styling
    set_professional_style()

    plt.figure(figsize=(14, 8), facecolor='white')
    pivot_rmse = df.pivot(index='combination', columns='method', values='rmse')

    sns.heatmap(
        pivot_rmse,
        annot=True,
        fmt='.4f',
        cmap='RdYlBu_r',
        cbar_kws={'label': 'RMSE'},
        linewidths=0.5,
        linecolor='white',
    )

    plt.title('Ensemble Performance Heatmap (RMSE)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ensemble Method', fontsize=12, fontweight='medium')
    plt.ylabel('Neural Network Combination', fontsize=12, fontweight='medium')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'research_performance_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 2. Best performers ranking
    plt.figure(figsize=(14, 8), facecolor='white')

    # Get top 10 combinations by RMSE
    top_combinations = df.nsmallest(10, 'rmse').copy()
    top_combinations['combo_method'] = top_combinations['combination'] + '_' + top_combinations['method']

    bars = plt.barh(range(len(top_combinations)), top_combinations['rmse'], alpha=0.7, color='#2E86C1')

    plt.yticks(range(len(top_combinations)), top_combinations['combo_method'])
    plt.xlabel('RMSE', fontsize=12, fontweight='medium')
    plt.title('Top 10 Best Performing Ensemble Combinations', fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()

    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, top_combinations['rmse'])):
        plt.text(v + max(top_combinations['rmse']) * 0.01, i, f'{v:.4f}', va='center', fontsize=10)

    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'top_combinations_research.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    reset_style()


class ResearchEnsembleRunner(EnhancedEnsembleRunner):
    """Enhanced ensemble runner with integrated professional visualizations."""

    def save_individual_results(self, combination_name, method, results, combination_dir):
        """Save individual ensemble method results with professional visualizations."""
        method_dir = os.path.join(combination_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Save comprehensive results to JSON
        with open(os.path.join(method_dir, f'{combination_name.lower()}_{method}_detailed_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Save metrics to CSV
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(os.path.join(method_dir, f'{combination_name.lower()}_{method}_metrics.csv'), index=False)

        # Save predictions to CSV
        actual = np.array(results['predictions']['actual_values'])
        predicted = np.array(results['predictions']['predicted_values'])
        pred_df = pd.DataFrame(
            {
                'actual': actual,
                'predicted': predicted,
                'residual': actual - predicted,
                'abs_error': np.abs(actual - predicted),
                'percentage_error': np.abs((actual - predicted) / actual) * 100,
            }
        )
        pred_df.to_csv(os.path.join(method_dir, f'{combination_name.lower()}_{method}_predictions.csv'), index=False)

        # Create professional visualizations matching individual models
        print(f'📊 Creating professional visualizations for {combination_name} {method}...')

        # 1. Detailed prediction plot
        create_detailed_prediction_plot(actual, predicted, combination_name, method, method_dir)

        # 2. Residuals analysis plot
        create_residuals_analysis_plot(actual, predicted, combination_name, method, method_dir)

        # 3. Training history plot (if available)
        training_history = results['training_data'].get('training_history')
        if training_history and isinstance(training_history, dict):
            train_losses = training_history.get('train_losses', [])
            val_losses = training_history.get('validation_losses', [])

            if train_losses and val_losses:
                create_professional_training_history_plot(
                    train_losses, val_losses, combination_name, method, method_dir
                )

                # Save training history to CSV
                history_df = pd.DataFrame(
                    {
                        'epoch': range(1, len(train_losses) + 1),
                        'train_loss': train_losses,
                        'validation_loss': val_losses,
                    }
                )
                history_df.to_csv(
                    os.path.join(method_dir, f'{combination_name.lower()}_{method}_training_history.csv'), index=False
                )

    def create_comprehensive_summary(self):
        """Create comprehensive summary with professional research visualizations."""
        # Call parent method first
        super().create_comprehensive_summary()

        # Add professional research summary plots
        print('📊 Creating professional research summary visualizations...')
        create_research_summary_plots(self.results, self.output_dir)

        print('🎓 ACADEMIC RESEARCH OUTPUTS CREATED:')
        print('   📊 Professional visualizations matching individual models style')
        print('   📄 High-resolution PNG files (300 DPI) for academic papers')
        print('   📈 Training history, predictions, and residuals analysis plots')
        print('   📋 Comprehensive CSV data tables for research appendices')
        print('   🔬 Performance heatmaps and ranking charts')


# ============================================================================
# RESEARCH FUNCTIONS WITH PROFESSIONAL VISUALIZATIONS
# ============================================================================


def run_research_pairs_only(epochs=50, dataset_path=None, output_dir='src/model/ensemble/research_output/pairs'):
    """Run only the pair combinations from the research with professional visualizations."""
    print('🔬 RESEARCH ANALYSIS: PAIR COMBINATIONS (With Professional Visualizations)')
    print('=' * 80)
    print('Testing combinations:')
    print('  1. LSTM + GRU (Recurrent pair with different gating mechanisms)')
    print('  2. LSTM + CNN (High diversity: Sequential + Local pattern detection)')
    print('  3. GRU + Bi-LSTM (Recurrent pair with bidirectional context)')
    print('  4. GRU + CNN (High diversity: Sequential + Local pattern detection)')
    print('  5. Bi-LSTM + CNN (High diversity: Bidirectional + Feature extraction)')
    print('📊 Professional visualizations will match individual models style!')
    print('=' * 80)

    runner = ResearchEnsembleRunner(
        epochs=epochs,
        df_path=dataset_path or 'src/model/ensemble/dataset_1_full_features.csv',
        use_optimized_params=True,
        output_dir=output_dir,
    )

    runner.prepare_data()
    runner.run_all_pairs()
    runner.create_comprehensive_summary()

    return runner.results


def run_research_triplets_only(epochs=50, dataset_path=None, output_dir='src/model/ensemble/research_output/triplets'):
    """Run only the triplet combinations from the research with professional visualizations."""
    print('🔬 RESEARCH ANALYSIS: TRIPLET COMBINATIONS (With Professional Visualizations)')
    print('=' * 80)
    print('Testing combinations:')
    print('  1. LSTM + GRU + CNN (Traditional recurrent networks + CNN)')
    print('  2. GRU + Bi-LSTM + CNN (Modern approach with bidirectional context)')
    print('📊 Professional visualizations will match individual models style!')
    print('=' * 80)

    runner = ResearchEnsembleRunner(
        epochs=epochs,
        df_path=dataset_path or 'src/model/ensemble/dataset_1_full_features.csv',
        use_optimized_params=True,
        output_dir=output_dir,
    )

    runner.prepare_data()
    runner.run_all_triplets()
    runner.create_comprehensive_summary()

    return runner.results


def run_complete_research_analysis(epochs=50, dataset_path=None, output_dir='src/model/ensemble/research_output'):
    """Run the complete research analysis with all combinations and professional visualizations."""
    print('🔬 COMPLETE RESEARCH ANALYSIS: ALL COMBINATIONS (With Professional Visualizations)')
    print('=' * 80)
    print('This will test all pair and triplet combinations with all ensemble methods:')
    print('\nPair Combinations:')
    print('  • LSTM + GRU')
    print('  • LSTM + CNN')
    print('  • GRU + Bi-LSTM')
    print('  • GRU + CNN')
    print('  • Bi-LSTM + CNN')
    print('\nTriplet Combinations:')
    print('  • LSTM + GRU + CNN')
    print('  • GRU + Bi-LSTM + CNN')
    print('\nEnsemble Methods (applied to each combination):')
    print('  • Voting (Optimized parameters from Optuna)')
    print('  • Stacking (Optimized parameters from Optuna)')
    print('  • Blending (Optimized parameters from Optuna)')
    print('\n📊 Professional Visualizations (matching individual models):')
    print('  • Training history plots with academic formatting')
    print('  • Detailed prediction vs actual plots with statistics')
    print('  • Residuals analysis (4-panel diagnostic plots)')
    print('  • Performance heatmaps and ranking charts')
    print('  • High-resolution PNG files (300 DPI) for academic papers')
    print('=' * 80)

    start_time = time.time()

    runner = ResearchEnsembleRunner(
        epochs=epochs,
        df_path=dataset_path or 'src/model/ensemble/dataset_1_full_features.csv',
        use_optimized_params=True,
        output_dir=output_dir,
    )

    runner.prepare_data()
    results = runner.run_complete_analysis()

    total_time = time.time() - start_time

    print(
        f'\n🎉 Complete research analysis with professional visualizations finished in {total_time / 60:.1f} minutes!'
    )
    print(f'📁 All results saved to: {runner.output_dir}')
    print('🎓 ACADEMIC RESEARCH READY:')
    print('   📊 Professional visualizations matching individual models style')
    print('   📄 High-resolution PNG files for academic papers')
    print('   📈 Training history, predictions, and residuals analysis')
    print('   📋 Comprehensive CSV data tables')
    print('   🔬 Performance heatmaps and statistical analysis')

    return results


def run_quick_demo(epochs=50, output_dir='src/model/ensemble/research_output/demo'):
    """Run a quick demo with high-diversity pairs for testing."""
    print('🚀 QUICK DEMO: High-Diversity Pairs')
    print('=' * 60)
    print(f'Testing: LSTM+CNN, GRU+CNN, BiLSTM+CNN with {epochs} epochs each')
    print('=' * 60)

    runner = ResearchEnsembleRunner(epochs=epochs, use_optimized_params=True, output_dir=output_dir)
    runner.prepare_data()

    # Run only high-diversity pairs
    high_diversity_pairs = ['LSTM_CNN', 'GRU_CNN', 'BiLSTM_CNN']

    for pair_name in high_diversity_pairs:
        if pair_name in runner.pair_combinations:
            runner.run_combination(pair_name, runner.pair_combinations[pair_name])

    runner.create_comprehensive_summary()

    return runner.results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run ensemble combinations for stock prediction research with professional academic visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_research_combinations.py --mode complete --epochs 30
  python run_research_combinations.py --mode pairs --epochs 25
  python run_research_combinations.py --mode triplets --epochs 20
  python run_research_combinations.py --mode demo --epochs 15

Professional Visualization Features:
  • Training history plots with academic formatting (matching individual models)
  • Detailed prediction vs actual scatter plots with statistics
  • 4-panel residuals analysis (distribution, Q-Q plot, time series)
  • Performance heatmaps and ranking charts
  • High-resolution PNG files (300 DPI) for academic papers
  • Comprehensive CSV data tables for research appendices
  • Statistical analysis and architecture comparison charts

All visualizations match the professional style used in individual models
for consistent academic research presentation.
        """,
    )

    parser.add_argument(
        '--mode',
        choices=['complete', 'pairs', 'triplets', 'demo'],
        default='complete',
        help='Analysis mode to run (default: complete)',
    )

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset CSV file (default: src/model/ensemble/dataset_1_full_features.csv)',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: depends on mode)',
    )

    args = parser.parse_args()

    # Set default output directories based on mode if not specified
    if args.output_dir is None:
        output_dirs = {
            'complete': 'src/model/ensemble/research_output',
            'pairs': 'src/model/ensemble/research_output/pairs',
            'triplets': 'src/model/ensemble/research_output/triplets',
            'demo': 'src/model/ensemble/research_output/demo',
        }
        output_dir = output_dirs[args.mode]
    else:
        output_dir = args.output_dir

    print('🔬 Research Ensemble Analysis with Professional Academic Visualizations')
    print(f'📅 Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'⚙️  Mode: {args.mode}')
    print(f'🔄 Epochs: {args.epochs}')
    print(f'📁 Output Directory: {output_dir}')
    if args.dataset:
        print(f'📊 Dataset: {args.dataset}')
    print('📊 Visualizations: Professional style matching individual models')
    print()

    try:
        if args.mode == 'complete':
            results = run_complete_research_analysis(args.epochs, args.dataset, output_dir)
        elif args.mode == 'pairs':
            results = run_research_pairs_only(args.epochs, args.dataset, output_dir)
        elif args.mode == 'triplets':
            results = run_research_triplets_only(args.epochs, args.dataset, output_dir)
        elif args.mode == 'demo':
            results = run_quick_demo(args.epochs, output_dir)
        else:
            print(f'Unknown mode: {args.mode}')
            sys.exit(1)

        print('\n✅ Analysis completed successfully!')
        print(f'📈 Total combinations tested: {len([c for c in results.keys()])}')

        # Count total ensemble tests
        total_tests = sum(len([m for m in methods.keys() if 'error' not in methods[m]]) for methods in results.values())
        print(f'🧪 Total ensemble tests: {total_tests}')

        print('\n🎓 ACADEMIC RESEARCH OUTPUTS READY:')
        print('   📊 Professional visualizations matching individual models style')
        print('   📄 High-resolution PNG files (300 DPI) for academic papers')
        print('   📈 Training history, predictions, and residuals analysis plots')
        print('   📋 Comprehensive CSV data tables for research appendices')
        print('   🔬 Performance heatmaps and statistical analysis charts')
        print('   ✅ All results consistent with individual models presentation!')
        print(f'   📁 Find all outputs in: {output_dir}')

    except KeyboardInterrupt:
        print('\n⚠️  Analysis interrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ Error during analysis: {str(e)}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
