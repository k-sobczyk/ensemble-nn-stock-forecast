import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from src.model.baseline.baseline_utils import visualize_baseline_comparison
from src.model.metrics.metrics import calculate_mape, calculate_mase


def find_best_arima_order(data, max_p=3, max_d=2, max_q=3):
    best_aic = float('inf')
    best_order = (1, 1, 1)

    if hasattr(data, 'values'):
        data = data.values

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                except Exception:
                    continue

    return best_order


def create_ticker_time_series(df, test_start_date='2021-01-01'):
    print('Creating individual time series for each ticker...')
    print(f'Requiring data before {test_start_date} (training) AND after {test_start_date} (testing)')

    ticker_time_series = {}
    skipped_reasons = {'insufficient_total': 0, 'no_train_data': 0, 'no_test_data': 0, 'insufficient_train': 0}

    test_start_dt = pd.to_datetime(test_start_date)

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()

        ticker_data = ticker_data.sort_values('end_of_period')
        ticker_data['end_of_period'] = pd.to_datetime(ticker_data['end_of_period'])

        time_series = ticker_data.set_index('end_of_period').drop('ticker', axis=1)

        if len(time_series) < 11:
            print(f'✗ {ticker}: Insufficient total data ({len(time_series)} periods)')
            skipped_reasons['insufficient_total'] += 1
            continue

        train_data = time_series[time_series.index < test_start_dt]
        if len(train_data) == 0:
            print(f'✗ {ticker}: No training data (all data is after {test_start_date})')
            skipped_reasons['no_train_data'] += 1
            continue

        test_data = time_series[time_series.index >= test_start_dt]
        if len(test_data) == 0:
            print(f'✗ {ticker}: No test data (all data is before {test_start_date})')
            skipped_reasons['no_test_data'] += 1
            continue

        if len(train_data) < 10:
            print(f'✗ {ticker}: Insufficient training data ({len(train_data)} periods, need ≥10)')
            skipped_reasons['insufficient_train'] += 1
            continue

        ticker_time_series[ticker] = time_series
        print(
            f'✓ {ticker}: {len(time_series)} periods total | Train: {len(train_data)} periods ({train_data.index.min().strftime("%Y-%m")} to {train_data.index.max().strftime("%Y-%m")}) | Test: {len(test_data)} periods ({test_data.index.min().strftime("%Y-%m")} to {test_data.index.max().strftime("%Y-%m")})'
        )

    print('\n📊 TICKER FILTERING SUMMARY:')
    print(f'Total tickers in dataset: {df["ticker"].nunique()}')
    print(f'✓ Passed all checks: {len(ticker_time_series)}')
    print(f'✗ Skipped - insufficient total data: {skipped_reasons["insufficient_total"]}')
    print(f'✗ Skipped - no training data: {skipped_reasons["no_train_data"]}')
    print(f'✗ Skipped - no test data (missing 2021-2022): {skipped_reasons["no_test_data"]}')
    print(f'✗ Skipped - insufficient training data: {skipped_reasons["insufficient_train"]}')

    return ticker_time_series


def predict_arima_for_time_series(time_series, test_start_date='2021-01-01'):
    train_ts = time_series[time_series.index < test_start_date]['target_log']
    test_ts = time_series[time_series.index >= test_start_date]['target_log']

    if len(train_ts) < 10 or len(test_ts) < 1:
        return None, None, None

    try:
        best_order = find_best_arima_order(train_ts.values)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = ARIMA(train_ts.values, order=best_order)
            fitted_model = model.fit()

            forecast_steps = len(test_ts)
            forecast = fitted_model.forecast(steps=forecast_steps)

        y_true_orig = np.expm1(test_ts.values)
        y_pred_orig = np.expm1(forecast)
        y_train_orig = np.expm1(train_ts.values)

        return y_true_orig, y_pred_orig, y_train_orig

    except Exception as e:
        print(f'Error fitting ARIMA: {e}')
        return None, None, None


def evaluate_single_ticker_arima(ticker, time_series, test_start_date='2021-01-01'):
    print(f'\nProcessing {ticker}...')

    y_true, y_pred, y_train = predict_arima_for_time_series(time_series, test_start_date)

    if y_true is None:
        print(f'✗ Failed to process {ticker} (insufficient data or model error)')
        return None

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    mase = calculate_mase(y_true, y_pred, y_train)

    print(f'✓ Successfully processed {ticker} - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, MASE: {mase:.4f}')

    return {
        'ticker': ticker,
        'rmse': rmse,
        'mape': mape,
        'mase': mase,
        'n_test_samples': len(y_true),
        'n_train_samples': len(y_train),
        'best_arima_order': 'Unknown',
    }


def arima_baseline_evaluation(
    dataset_path='data/datasets/dataset_1_full_features.csv', test_start_date='2021-01-01', create_visualizations=True
):
    print('=' * 60)
    print('ARIMA BASELINE EVALUATION')
    print('=' * 60)

    print('Loading dataset...')
    df = pd.read_csv(dataset_path)
    print(f'Dataset shape: {df.shape}')
    print(f'Unique tickers: {df["ticker"].nunique()}')

    ticker_time_series = create_ticker_time_series(df, test_start_date)

    print(f'\n{"=" * 50}')
    print('PROCESSING INDIVIDUAL TIME SERIES')
    print(f'{"=" * 50}')

    all_results = []
    successful_tickers = []

    warnings.filterwarnings('ignore')

    for ticker, time_series in ticker_time_series.items():
        result = evaluate_single_ticker_arima(ticker, time_series, test_start_date)

        if result is not None:
            all_results.append(result)
            successful_tickers.append(ticker)

            if create_visualizations:
                try:
                    ticker_data = time_series.reset_index()
                    ticker_data['ticker'] = ticker
                    ticker_data['end_of_period'] = ticker_data.index

                    ticker_data['end_of_period'] = time_series.index
                    ticker_data = ticker_data.reset_index(drop=True)

                    visualize_baseline_comparison(ticker_data, ticker, test_start_date)
                except Exception as e:
                    print(f'⚠ Warning: Could not create visualization for {ticker}: {e}')

    if all_results:
        results_df = pd.DataFrame(all_results)

        print(f'\n{"=" * 50}')
        print('ARIMA BASELINE RESULTS')
        print(f'{"=" * 50}')
        print(f'Successfully processed: {len(successful_tickers)}/{len(ticker_time_series)} tickers')
        print(f'Total test samples: {results_df["n_test_samples"].sum()}')
        print(f'Average train samples per ticker: {results_df["n_train_samples"].mean():.1f}')

        print('\nAVERAGE METRICS ACROSS ALL TICKERS:')
        print(f'RMSE: {results_df["rmse"].mean():.4f} ± {results_df["rmse"].std():.4f}')
        print(f'MAPE: {results_df["mape"].mean():.2f}% ± {results_df["mape"].std():.2f}%')
        print(f'MASE: {results_df["mase"].mean():.4f} ± {results_df["mase"].std():.4f}')

        print('\nPERFORMANCE DISTRIBUTION:')
        print(f'Best RMSE: {results_df["rmse"].min():.4f} ({results_df.loc[results_df["rmse"].idxmin(), "ticker"]})')
        print(f'Worst RMSE: {results_df["rmse"].max():.4f} ({results_df.loc[results_df["rmse"].idxmax(), "ticker"]})')
        print(f'Median RMSE: {results_df["rmse"].median():.4f}')

        results_df.to_csv('arima_baseline_detailed_results.csv', index=False)
        print('\nDetailed results saved to: arima_baseline_detailed_results.csv')

        if create_visualizations:
            print('Visualizations saved to: src/model/baseline/visualization_images/')

        return results_df
    else:
        print('❌ No successful predictions made!')
        return None


def predict_arima_for_ticker(ticker_data, test_start_date='2021-01-01'):
    ticker_data_sorted = ticker_data.sort_values('end_of_period')
    ticker_data_sorted['end_of_period'] = pd.to_datetime(ticker_data_sorted['end_of_period'])
    time_series = ticker_data_sorted.set_index('end_of_period')

    return predict_arima_for_time_series(time_series, test_start_date)


if __name__ == '__main__':
    results = arima_baseline_evaluation()
