#!/usr/bin/env python3
"""Run missing stacking combinations to complete the research output."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.model.ensemble.enhanced_ensemble_runner import EnhancedEnsembleRunner


def run_missing_stacking_combinations():
    """Run the missing stacking combinations identified in the analysis."""
    print('🔍 Running Missing Stacking Combinations')
    print('=' * 80)
    print('This will add the missing stacking results for:')
    print('  • LSTM_GRU')
    print('  • GRU_CNN')
    print('  • GRU_BiLSTM_CNN')
    print('=' * 80)

    # Initialize runner with full dataset path
    runner = EnhancedEnsembleRunner(
        df_path='src/model/ensemble/dataset_1_full_features.csv',
        epochs=30,
        use_optimized_params=True,
        output_dir='src/model/ensemble/research_output',
    )

    # Prepare data
    print('📊 Preparing data...')
    runner.prepare_data()

    # Missing combinations
    missing_combinations = [('LSTM_GRU', 'pair'), ('GRU_CNN', 'pair'), ('GRU_BiLSTM_CNN', 'triplet')]

    for combo_name, combo_type in missing_combinations:
        print(f'\n{"=" * 60}')
        print(f'Running STACKING for {combo_name} ({combo_type})')
        print(f'{"=" * 60}')

        try:
            # Get the model configuration
            if combo_type == 'pair':
                models_config = runner.pair_combinations[combo_name]
            else:
                models_config = runner.triplet_combinations[combo_name]

            # Run only stacking for this combination
            result = runner.run_combination(combo_name, models_config, ensemble_methods=['stacking'])

            if 'stacking' in result and 'error' not in result['stacking']:
                print(f'✅ {combo_name} stacking completed successfully!')
                metrics = result['stacking']['metrics']
                print(f'   RMSE: {metrics["rmse"]:.4f}')
                print(f'   MAE: {metrics["mae"]:.4f}')
                print(f'   R²: {metrics["r2"]:.4f}')
                print(f'   MASE: {metrics["mase"]:.4f}')
            else:
                print(f'❌ {combo_name} stacking failed')
                if 'stacking' in result:
                    print(f'   Error: {result["stacking"].get("error", "Unknown error")}')

        except Exception as e:
            print(f'❌ Error running {combo_name}: {str(e)}')

    # Update comprehensive summary
    print(f'\n{"=" * 60}')
    print('📊 Updating comprehensive summary...')
    print(f'{"=" * 60}')
    runner.create_comprehensive_summary()

    print('\n🎉 Missing stacking combinations completed!')
    print(f'📁 Updated results saved to: {runner.output_dir}')
    print('📊 Check ensemble_comprehensive_results.csv for the new stacking entries')


if __name__ == '__main__':
    run_missing_stacking_combinations()
