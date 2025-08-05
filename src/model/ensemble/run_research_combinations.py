#!/usr/bin/env python3
"""Research-specific ensemble combinations runner.

This script runs the exact neural network combinations specified in the research:
- Pair Combinations: LSTM+GRU, LSTM+CNN, GRU+BiLSTM, GRU+CNN, BiLSTM+CNN
- Triplet Combinations: LSTM+GRU+CNN, GRU+BiLSTM+CNN
- Ensemble Methods: Voting, Stacking, Blending for each combination

Results are saved with comprehensive CSV files and PNG visualizations similar to individual models.
"""

import argparse
import sys
import time
from datetime import datetime

from src.model.ensemble.enhanced_ensemble_runner import EnhancedEnsembleRunner


def run_research_pairs_only(epochs=30, dataset_path=None):
    """Run only the pair combinations from the research."""
    print('🔬 RESEARCH ANALYSIS: PAIR COMBINATIONS')
    print('=' * 80)
    print('Testing combinations:')
    print('  1. LSTM + GRU (Recurrent pair with different gating mechanisms)')
    print('  2. LSTM + CNN (High diversity: Sequential + Local pattern detection)')
    print('  3. GRU + Bi-LSTM (Recurrent pair with bidirectional context)')
    print('  4. GRU + CNN (High diversity: Sequential + Local pattern detection)')
    print('  5. Bi-LSTM + CNN (High diversity: Bidirectional + Feature extraction)')
    print('=' * 80)

    runner = EnhancedEnsembleRunner(epochs=epochs, df_path=dataset_path or 'dataset_1_full_features.csv')

    runner.prepare_data()
    runner.run_all_pairs()
    runner.create_comprehensive_summary()

    return runner.results


def run_research_triplets_only(epochs=30, dataset_path=None):
    """Run only the triplet combinations from the research."""
    print('🔬 RESEARCH ANALYSIS: TRIPLET COMBINATIONS')
    print('=' * 80)
    print('Testing combinations:')
    print('  1. LSTM + GRU + CNN (Traditional recurrent networks + CNN)')
    print('  2. GRU + Bi-LSTM + CNN (Modern approach with bidirectional context)')
    print('=' * 80)

    runner = EnhancedEnsembleRunner(epochs=epochs, df_path=dataset_path or 'dataset_1_full_features.csv')

    runner.prepare_data()
    runner.run_all_triplets()
    runner.create_comprehensive_summary()

    return runner.results


def run_complete_research_analysis(epochs=30, dataset_path=None):
    """Run the complete research analysis with all combinations."""
    print('🔬 COMPLETE RESEARCH ANALYSIS: ALL COMBINATIONS')
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
    print('  • Voting (Weighted with optimized weights)')
    print('  • Stacking (Ridge meta-model with 3-fold CV)')
    print('  • Blending (Ridge meta-model with 20% blend ratio)')
    print('=' * 80)

    start_time = time.time()

    runner = EnhancedEnsembleRunner(epochs=epochs, df_path=dataset_path or 'dataset_1_full_features.csv')

    runner.prepare_data()
    results = runner.run_complete_analysis()

    total_time = time.time() - start_time

    print(f'\n🎉 Complete research analysis finished in {total_time / 60:.1f} minutes!')
    print(f'📁 All results saved to: {runner.output_dir}')

    return results


def run_quick_demo(epochs=15):
    """Run a quick demo with reduced epochs for testing."""
    print('🚀 QUICK DEMO: High-Diversity Pairs (Reduced Epochs)')
    print('=' * 60)
    print(f'Testing: LSTM+CNN, GRU+CNN, BiLSTM+CNN with {epochs} epochs each')
    print('=' * 60)

    runner = EnhancedEnsembleRunner(epochs=epochs)
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
        description='Run ensemble combinations for stock prediction research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_research_combinations.py --mode complete --epochs 30
  python run_research_combinations.py --mode pairs --epochs 25
  python run_research_combinations.py --mode triplets --epochs 20
  python run_research_combinations.py --mode demo --epochs 15
        """,
    )

    parser.add_argument(
        '--mode',
        choices=['complete', 'pairs', 'triplets', 'demo'],
        default='complete',
        help='Analysis mode to run (default: complete)',
    )

    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset CSV file (default: dataset_1_full_features.csv)',
    )

    args = parser.parse_args()

    print('🔬 Research Ensemble Analysis')
    print(f'📅 Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'⚙️  Mode: {args.mode}')
    print(f'🔄 Epochs: {args.epochs}')
    if args.dataset:
        print(f'📊 Dataset: {args.dataset}')
    print()

    try:
        if args.mode == 'complete':
            results = run_complete_research_analysis(args.epochs, args.dataset)
        elif args.mode == 'pairs':
            results = run_research_pairs_only(args.epochs, args.dataset)
        elif args.mode == 'triplets':
            results = run_research_triplets_only(args.epochs, args.dataset)
        elif args.mode == 'demo':
            results = run_quick_demo(args.epochs)
        else:
            print(f'Unknown mode: {args.mode}')
            sys.exit(1)

        print('\n✅ Analysis completed successfully!')
        print(f'📈 Total combinations tested: {len([c for c in results.keys()])}')

        # Count total ensemble tests
        total_tests = sum(len([m for m in methods.keys() if 'error' not in methods[m]]) for methods in results.values())
        print(f'🧪 Total ensemble tests: {total_tests}')

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
