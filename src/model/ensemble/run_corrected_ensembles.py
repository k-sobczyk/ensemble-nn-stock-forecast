#!/usr/bin/env python3
"""Re-run all ensemble methods with proper temporal data handling.

CORRECTIONS MADE:
- Voting: Already correct (no changes needed)
- Blending: Fixed to use temporal split instead of random split
- Stacking: Fixed to use simple train/validation approach (no cross-validation)

All methods now use proper temporal order preservation for time series forecasting.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.model.ensemble.enhanced_ensemble_runner import EnhancedEnsembleRunner

def run_corrected_ensembles():
    """Re-run all ensemble combinations with corrected temporal handling."""
    print("🔧 CORRECTED ENSEMBLES: Proper Temporal Data Handling")
    print("=" * 80)
    print("CORRECTIONS APPLIED:")
    print("  ✅ Voting: Already correct (uses train/val properly)")
    print("  🔧 Blending: Fixed temporal split (was using random split)")
    print("  🔧 Stacking: Fixed simple train/val approach (was using CV)")
    print("\n🎯 ALL METHODS NOW:")
    print("  • Preserve temporal order")
    print("  • Use clean train/validation/test separation")
    print("  • Consistent with individual neural network methodology")
    print("=" * 80)
    print("Re-running ALL ensemble combinations with ALL methods:")
    print("  • LSTM_GRU")
    print("  • LSTM_CNN") 
    print("  • GRU_CNN")
    print("  • GRU_BiLSTM")
    print("  • BiLSTM_CNN")
    print("  • LSTM_GRU_CNN")
    print("  • GRU_BiLSTM_CNN")
    print("=" * 80)
    
    # Initialize runner
    runner = EnhancedEnsembleRunner(
        df_path='src/model/ensemble/dataset_1_full_features.csv',
        epochs=30,
        use_optimized_params=True,
        output_dir='src/model/ensemble/research_output'
    )
    
    # Prepare data
    print("📊 Preparing data...")
    runner.prepare_data()
    
    # All combinations
    all_combinations = [
        ('LSTM_GRU', 'pair'),
        ('LSTM_CNN', 'pair'),
        ('GRU_CNN', 'pair'),
        ('GRU_BiLSTM', 'pair'),
        ('BiLSTM_CNN', 'pair'),
        ('LSTM_GRU_CNN', 'triplet'),
        ('GRU_BiLSTM_CNN', 'triplet')
    ]
    
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    
    for combo_name, combo_type in all_combinations:
        print(f"\n{'='*80}")
        print(f"Running ALL CORRECTED ENSEMBLES for {combo_name} ({combo_type})")
        print(f"{'='*80}")
        
        try:
            # Get the model configuration
            if combo_type == 'pair':
                models_config = runner.pair_combinations[combo_name]
            else:
                models_config = runner.triplet_combinations[combo_name]
            
            # Run all ensemble methods for this combination
            result = runner.run_combination(
                combo_name, 
                models_config, 
                ensemble_methods=['voting', 'blending', 'stacking']
            )
            
            # Count results
            for method in ['voting', 'blending', 'stacking']:
                total_tests += 1
                if method in result and 'error' not in result[method]:
                    print(f"  ✅ {combo_name} {method}: RMSE {result[method]['metrics']['rmse']:.4f}, R² {result[method]['metrics']['r2']:.4f}")
                    successful_tests += 1
                else:
                    print(f"  ❌ {combo_name} {method}: Failed")
                    if method in result:
                        print(f"     Error: {result[method].get('error', 'Unknown error')}")
                    failed_tests += 1
                
        except Exception as e:
            print(f"❌ Error running {combo_name}: {str(e)}")
            failed_tests += 3  # All 3 methods failed
            total_tests += 3
    
    # Update comprehensive summary
    print(f"\n{'='*80}")
    print("📊 Updating comprehensive summary...")
    print(f"{'='*80}")
    runner.create_comprehensive_summary()
    
    print(f"\n🎉 Corrected ensemble methods completed!")
    print(f"📊 RESULTS SUMMARY:")
    print(f"   Total tests: {total_tests}")
    print(f"   ✅ Successful: {successful_tests}")
    print(f"   ❌ Failed: {failed_tests}")
    print(f"   Success rate: {successful_tests/total_tests*100:.1f}%")
    print(f"\n📁 Updated results saved to: {runner.output_dir}")
    print("📊 Check ensemble_comprehensive_results.csv for all corrected results")
    print("\n🎯 METHODOLOGY: All ensembles now use proper temporal data handling")
    print("✅ No random shuffling of time series data")
    print("✅ Consistent with individual neural network approach")
    print("✅ Results are now methodologically sound for time series forecasting")

if __name__ == "__main__":
    run_corrected_ensembles()
