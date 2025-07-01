import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.model.metrics.metrics import calculate_mape, calculate_mase


def predict_last_value_for_ticker(ticker_data, test_start_date='2021-01-01'):
    """Predict using last value method for a single ticker."""
    ticker_data = ticker_data.sort_values('end_of_period')
    ticker_data['end_of_period'] = pd.to_datetime(ticker_data['end_of_period'])

    train_data = ticker_data[ticker_data['end_of_period'] < test_start_date].copy()
    test_data = ticker_data[ticker_data['end_of_period'] >= test_start_date].copy()

    if len(train_data) < 1 or len(test_data) < 1:
        return None, None, None

    last_value_log = train_data['target_log'].iloc[-1]

    y_pred_log = np.full(len(test_data), last_value_log)
    y_true_log = test_data['target_log'].values
    y_train_log = train_data['target_log'].values

    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    y_train_orig = np.expm1(y_train_log)

    return y_true_orig, y_pred_orig, y_train_orig


def last_value_baseline_evaluation(
    dataset_path='data/datasets/dataset_1_full_features.csv', create_visualizations=True
):
    """Evaluate Last Value baseline across all tickers."""
    print('Loading dataset...')
    df = pd.read_csv(dataset_path)

    print(f'Dataset shape: {df.shape}')
    print(f'Unique tickers: {df["ticker"].nunique()}')

    all_results = []
    successful_tickers = []

    for ticker in df['ticker'].unique():
        print(f'\nProcessing ticker: {ticker}')

        ticker_data = df[df['ticker'] == ticker].copy()

        y_true, y_pred, y_train = predict_last_value_for_ticker(ticker_data)

        if y_true is not None:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = calculate_mape(y_true, y_pred)
            mase = calculate_mase(y_true, y_pred, y_train)

            all_results.append(
                {
                    'ticker': ticker,
                    'rmse': rmse,
                    'mape': mape,
                    'mase': mase,
                    'n_test_samples': len(y_true),
                    'last_train_value': y_train[-1] if len(y_train) > 0 else np.nan,
                    'avg_test_value': np.mean(y_true),
                }
            )

            successful_tickers.append(ticker)
            print(f'✓ Successfully processed {ticker} - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, MASE: {mase:.4f}')

            if create_visualizations:
                try:
                    from src.model.baseline.baseline_utils import visualize_baseline_comparison
                    visualize_baseline_comparison(ticker_data, ticker)
                except Exception as e:
                    print(f'⚠ Warning: Could not create visualization for {ticker}: {e}')

        else:
            print(f'✗ Failed to process {ticker} (insufficient data)')

    if all_results:
        results_df = pd.DataFrame(all_results)

        print(f'\n{"=" * 50}')
        print('LAST VALUE BASELINE RESULTS')
        print(f'{"=" * 50}')
        print(f'Successfully processed tickers: {len(successful_tickers)}/{df["ticker"].nunique()}')
        print(f'Total test samples: {results_df["n_test_samples"].sum()}')

        print('\nAVERAGE METRICS ACROSS ALL TICKERS:')
        print(f'RMSE: {results_df["rmse"].mean():.4f} ± {results_df["rmse"].std():.4f}')
        print(f'MAPE: {results_df["mape"].mean():.2f}% ± {results_df["mape"].std():.2f}%')
        print(f'MASE: {results_df["mase"].mean():.4f} ± {results_df["mase"].std():.4f}')

        print('\nADDITIONAL INSIGHTS:')
        print(f'Average last training value: {results_df["last_train_value"].mean():.2f}')
        print(f'Average test value: {results_df["avg_test_value"].mean():.2f}')
        print(
            f'Correlation between last train and avg test: {results_df["last_train_value"].corr(results_df["avg_test_value"]):.4f}'
        )

        results_df.to_csv('last_value_baseline_detailed_results.csv', index=False)
        print('\nDetailed results saved to: last_value_baseline_detailed_results.csv')

        if create_visualizations:
            print('Visualizations saved to: src/model/baseline/visualization_images/')

        return results_df
    else:
        print('No successful predictions made!')
        return None


if __name__ == '__main__':
    results = last_value_baseline_evaluation()
