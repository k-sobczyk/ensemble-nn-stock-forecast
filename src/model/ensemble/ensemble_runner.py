
import pandas as pd

from src.model.ensemble.blending_ensemble import BlendingEnsemble
from src.model.ensemble.boosting_ensemble import BoostingEnsemble
from src.model.ensemble.ensemble_base import prepare_ensemble_data
from src.model.ensemble.stacking_ensemble import StackingEnsemble
from src.model.ensemble.voting_ensemble import VotingEnsemble


def run_ensemble_comparison(
    df_path='src/model/individual/dataset_1_full_features.csv', sequence_length=None, test_start_year=2021, epochs=30
):
    """Run comprehensive comparison of all ensemble methods."""
    print('=' * 80)
    print('ENSEMBLE LEARNING COMPARISON FOR STOCK PREDICTION')
    print('=' * 80)

    # Load and prepare data
    print('\nLoading and preparing data...')
    df = pd.read_csv(df_path)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y, feature_cols = prepare_ensemble_data(
        df,
        sequence_length=sequence_length,
        test_start_year=test_start_year,
        auto_sequence_length=True,
        split_validation=True,
    )

    input_size = len(feature_cols)
    sequence_length_used = X_train.shape[1]

    print(f'Training set: {X_train.shape}')
    print(f'Validation set: {X_val.shape}')
    print(f'Test set: {X_test.shape}')
    print(f'Input features: {input_size}')
    print(f'Sequence length: {sequence_length_used}')

    # Configuration for models to include
    models_config = {'lstm': True, 'gru': True, 'bi_lstm': True, 'cnn': True}

    results = {}

    # 1. Voting Ensemble (Simple)
    print('\n' + '=' * 50)
    print('1. SIMPLE VOTING ENSEMBLE')
    print('=' * 50)

    voting_simple = VotingEnsemble(models_config, voting_type='simple')
    voting_simple.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs)

    voting_pred = voting_simple.predict(X_test)
    voting_results = voting_simple.evaluate(y_test, voting_pred, scaler_y)
    results['Voting (Simple)'] = voting_results

    # 2. Voting Ensemble (Weighted)
    print('\n' + '=' * 50)
    print('2. WEIGHTED VOTING ENSEMBLE')
    print('=' * 50)

    voting_weighted = VotingEnsemble(models_config, voting_type='weighted')
    voting_weighted.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs)

    weighted_pred = voting_weighted.predict(X_test)
    weighted_results = voting_weighted.evaluate(y_test, weighted_pred, scaler_y)
    results['Voting (Weighted)'] = weighted_results

    # 3. Stacking Ensemble
    print('\n' + '=' * 50)
    print('3. STACKING ENSEMBLE')
    print('=' * 50)

    stacking = StackingEnsemble(models_config, meta_model_type='ridge', cv_folds=3)
    stacking.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs)

    stacking_pred = stacking.predict(X_test)
    stacking_results = stacking.evaluate(y_test, stacking_pred, scaler_y)
    results['Stacking (Ridge)'] = stacking_results

    # 4. Blending Ensemble
    print('\n' + '=' * 50)
    print('4. BLENDING ENSEMBLE')
    print('=' * 50)

    blending = BlendingEnsemble(models_config, meta_model_type='ridge', blend_ratio=0.2)
    blending.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs)

    blending_pred = blending.predict(X_test)
    blending_results = blending.evaluate(y_test, blending_pred, scaler_y)
    results['Blending (Ridge)'] = blending_results

    # 5. Boosting Ensemble
    print('\n' + '=' * 50)
    print('5. BOOSTING ENSEMBLE')
    print('=' * 50)

    boosting = BoostingEnsemble(models_config, n_estimators=4, learning_rate=0.8)
    boosting.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs)

    boosting_pred = boosting.predict(X_test)
    boosting_results = boosting.evaluate(y_test, boosting_pred, scaler_y)
    results['Boosting'] = boosting_results

    # Print comparison results
    print('\n' + '=' * 80)
    print('ENSEMBLE COMPARISON RESULTS')
    print('=' * 80)

    print(f'{"Method":<20} {"RMSE":<10} {"MAE":<10} {"R²":<10}')
    print('-' * 50)

    for method, metrics in results.items():
        print(f'{method:<20} {metrics["rmse"]:<10.4f} {metrics["mae"]:<10.4f} {metrics["r2"]:<10.4f}')

    # Find best method
    best_method = min(results.keys(), key=lambda x: results[x]['rmse'])
    print(f'\nBest performing method: {best_method}')
    print(f'Best RMSE: {results[best_method]["rmse"]:.4f}')

    return results


def run_quick_ensemble_test(
    models_to_test=['lstm', 'gru'], ensemble_types=['voting', 'stacking', 'blending', 'boosting']
):
    print('=' * 60)
    print('QUICK ENSEMBLE TEST')
    print('=' * 60)

    # Load data
    df = pd.read_csv('src/model/individual/dataset_1_full_features.csv')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y, feature_cols = prepare_ensemble_data(
        df, auto_sequence_length=True, split_validation=True
    )

    input_size = len(feature_cols)
    sequence_length_used = X_train.shape[1]

    # Create models config
    models_config = {name: (name in models_to_test) for name in ['lstm', 'gru', 'bi_lstm', 'cnn']}
    print(f'Testing models: {models_to_test}')
    print(f'Testing ensemble types: {ensemble_types}')

    results = {}

    if 'voting' in ensemble_types:
        print('\n--- Testing Voting Ensemble ---')
        voting = VotingEnsemble(models_config, voting_type='weighted')
        voting.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs=20)

        voting_pred = voting.predict(X_test)
        voting_results = voting.evaluate(y_test, voting_pred, scaler_y)
        results['Voting'] = voting_results

    if 'stacking' in ensemble_types:
        print('\n--- Testing Stacking Ensemble ---')
        stacking = StackingEnsemble(models_config, meta_model_type='linear', cv_folds=3)
        stacking.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs=20)

        stacking_pred = stacking.predict(X_test)
        stacking_results = stacking.evaluate(y_test, stacking_pred, scaler_y)
        results['Stacking'] = stacking_results

    if 'blending' in ensemble_types:
        print('\n--- Testing Blending Ensemble ---')
        blending = BlendingEnsemble(models_config, meta_model_type='linear')
        blending.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs=20)

        blending_pred = blending.predict(X_test)
        blending_results = blending.evaluate(y_test, blending_pred, scaler_y)
        results['Blending'] = blending_results

    if 'boosting' in ensemble_types:
        print('\n--- Testing Boosting Ensemble ---')
        boosting = BoostingEnsemble(models_config, n_estimators=3, learning_rate=0.8)
        boosting.fit(X_train, y_train, X_val, y_val, input_size, sequence_length_used, epochs=15)

        boosting_pred = boosting.predict(X_test)
        boosting_results = boosting.evaluate(y_test, boosting_pred, scaler_y)
        results['Boosting'] = boosting_results

    # Print results
    print('\n' + '=' * 40)
    print('QUICK TEST RESULTS')
    print('=' * 40)

    for method, metrics in results.items():
        print(f'{method}: RMSE={metrics["rmse"]:.4f}, R²={metrics["r2"]:.4f}')

    return results


if __name__ == '__main__':
    # Test only blending to verify the fix
    print('Testing only blending ensemble...')
    quick_results = run_quick_ensemble_test(['lstm', 'gru'], ['blending', 'stacking', 'boosting', 'voting'])

    # Uncomment below for full comparison (takes longer)
    # print("\n\nRunning full ensemble comparison...")
    # full_results = run_ensemble_comparison(epochs=25)
