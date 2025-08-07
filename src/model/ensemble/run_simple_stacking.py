#!/usr/bin/env python3
"""Re-run stacking with simple train/validation approach (like other neural networks).

This uses the same clean approach as individual neural networks:
- Train base models on training data
- Generate meta-features from validation predictions  
- Train meta-model on validation meta-features
- Test on unseen test data

No cross-validation complexity, no data leakage risk.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.model.ensemble.enhanced_ensemble_runner import EnhancedEnsembleRunner

def run_simple_stacking():
    """Re-run all stacking combinations with simple train/validation approach."""
    print("🔧 CLEAN STACKING: Simple Train/Validation Approach")
    print("=" * 80)
    print("Using the same approach as individual neural networks:")
    print("  1. Train base models on training data")
    print("  2. Generate meta-features from validation predictions")
    print("  3. Train meta-model on validation meta-features")
    print("  4. Test on unseen test data")
    print("\n✅ No cross-validation complexity")
    print("✅ No data leakage risk")
    print("✅ Consistent with individual model methodology")
    print("=" * 80)
    print("Re-running ALL stacking combinations:")
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
    
    # All combinations that have stacking
    all_combinations = [
        ('LSTM_GRU', 'pair'),
        ('LSTM_CNN', 'pair'),
        ('GRU_CNN', 'pair'),
        ('GRU_BiLSTM', 'pair'),
        ('BiLSTM_CNN', 'pair'),
        ('LSTM_GRU_CNN', 'triplet'),
        ('GRU_BiLSTM_CNN', 'triplet')
    ]
    
    successful_runs = 0
    failed_runs = 0
    
    for combo_name, combo_type in all_combinations:
        print(f"\n{'='*60}")
        print(f"Running SIMPLE STACKING for {combo_name} ({combo_type})")
        print(f"{'='*60}")
        
        try:
            # Get the model configuration
            if combo_type == 'pair':
                models_config = runner.pair_combinations[combo_name]
            else:
                models_config = runner.triplet_combinations[combo_name]
            
            # Run only stacking for this combination
            result = runner.run_combination(
                combo_name, 
                models_config, 
                ensemble_methods=['stacking']
            )
            
            if 'stacking' in result and 'error' not in result['stacking']:
                print(f"✅ {combo_name} simple stacking completed successfully!")
                metrics = result['stacking']['metrics']
                print(f"   RMSE: {metrics['rmse']:.4f}")
                print(f"   MAE: {metrics['mae']:.4f}")
                print(f"   R²: {metrics['r2']:.4f}")
                print(f"   MASE: {metrics['mase']:.4f}")
                successful_runs += 1
            else:
                print(f"❌ {combo_name} stacking failed")
                if 'stacking' in result:
                    print(f"   Error: {result['stacking'].get('error', 'Unknown error')}")
                failed_runs += 1
                
        except Exception as e:
            print(f"❌ Error running {combo_name}: {str(e)}")
            failed_runs += 1
    
    # Update comprehensive summary
    print(f"\n{'='*60}")
    print("📊 Updating comprehensive summary...")
    print(f"{'='*60}")
    runner.create_comprehensive_summary()
    
    print(f"\n🎉 Simple stacking approach completed!")
    print(f"✅ Successful: {successful_runs}")
    print(f"❌ Failed: {failed_runs}")
    print(f"📁 Updated results saved to: {runner.output_dir}")
    print("📊 Check ensemble_comprehensive_results.csv for the clean stacking results")
    print("\n✅ METHODOLOGY: Clean and consistent with individual neural networks")

if __name__ == "__main__":
    run_simple_stacking()
