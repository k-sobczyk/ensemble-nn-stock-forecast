#!/usr/bin/env python3
"""Re-run stacking combinations with corrected time series cross-validation.

CRITICAL FIX: The previous stacking results used KFold which causes data leakage
in time series forecasting. This script re-runs stacking with TimeSeriesSplit
to prevent future data from being used to predict the past.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.model.ensemble.enhanced_ensemble_runner import EnhancedEnsembleRunner


def run_corrected_stacking():
    """Re-run all stacking combinations with proper time series cross-validation."""
    print('🚨 CRITICAL FIX: Re-running Stacking with TimeSeriesSplit')
    print('=' * 80)
    print('ISSUE: Previous stacking used KFold which causes data leakage in time series')
    print('FIX: Now using TimeSeriesSplit to preserve temporal order')
    print('=' * 80)
    print('This will re-run ALL stacking combinations:')
    print('  • LSTM_GRU')
    print('  • LSTM_CNN')
    print('  • GRU_CNN')
    print('  • GRU_BiLSTM')
    print('  • BiLSTM_CNN')
    print('  • LSTM_GRU_CNN')
    print('  • GRU_BiLSTM_CNN')
    print('=' * 80)

    # Initialize runner
    runner = EnhancedEnsembleRunner(
        df_path='src/model/ensemble/dataset_1_full_features.csv',
        epochs=30,
        use_optimized_params=True,
        output_dir='src/model/ensemble/research_output',
    )

    # Prepare data
    print('📊 Preparing data...')
    runner.prepare_data()

    # All combinations that have stacking
    all_combinations = [
        ('LSTM_GRU', 'pair'),
        ('LSTM_CNN', 'pair'),
        ('GRU_CNN', 'pair'),
        ('GRU_BiLSTM', 'pair'),
        ('BiLSTM_CNN', 'pair'),
        ('LSTM_GRU_CNN', 'triplet'),
        ('GRU_BiLSTM_CNN', 'triplet'),
    ]

    successful_runs = 0
    failed_runs = 0

    for combo_name, combo_type in all_combinations:
        print(f'\n{"=" * 60}')
        print(f'Re-running CORRECTED STACKING for {combo_name} ({combo_type})')
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
                print(f'✅ {combo_name} corrected stacking completed successfully!')
                metrics = result['stacking']['metrics']
                print(f'   RMSE: {metrics["rmse"]:.4f}')
                print(f'   MAE: {metrics["mae"]:.4f}')
                print(f'   R²: {metrics["r2"]:.4f}')
                print(f'   MASE: {metrics["mase"]:.4f}')
                successful_runs += 1
            else:
                print(f'❌ {combo_name} stacking failed')
                if 'stacking' in result:
                    print(f'   Error: {result["stacking"].get("error", "Unknown error")}')
                failed_runs += 1

        except Exception as e:
            print(f'❌ Error running {combo_name}: {str(e)}')
            failed_runs += 1

    # Update comprehensive summary
    print(f'\n{"=" * 60}')
    print('📊 Updating comprehensive summary...')
    print(f'{"=" * 60}')
    runner.create_comprehensive_summary()

    print('\n🎉 Corrected stacking re-run completed!')
    print(f'✅ Successful: {successful_runs}')
    print(f'❌ Failed: {failed_runs}')
    print(f'📁 Updated results saved to: {runner.output_dir}')
    print('📊 Check ensemble_comprehensive_results.csv for the corrected stacking entries')
    print('\n🚨 IMPORTANT: These results should now be much more realistic')
    print("   Previous 'good' results were likely due to data leakage")


if __name__ == '__main__':
    run_corrected_stacking()
