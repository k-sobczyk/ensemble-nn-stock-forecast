import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def fill_missing_quarters(df):
    new_rows = []

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]

        ticker_data = ticker_data.sort_values('end_of_period')

        first_date = ticker_data['end_of_period'].min()
        last_date = ticker_data['end_of_period'].max()

        day_of_month = ticker_data['end_of_period'].dt.day.mode()[0]

        all_quarters = []
        current_date = first_date

        while current_date <= last_date:
            all_quarters.append(current_date)
            year = current_date.year + (current_date.month + 3) // 12
            month = (current_date.month + 3 - 1) % 12 + 1
            current_date = pd.Timestamp(year=year, month=month, day=day_of_month)

        existing_dates = set(ticker_data['end_of_period'])
        all_quarters_set = set(all_quarters)
        missing_dates = all_quarters_set - existing_dates

        for missing_date in missing_dates:
            new_row = {'ticker': ticker, 'end_of_period': missing_date}
            new_rows.append(new_row)

    if new_rows:
        missing_df = pd.DataFrame(new_rows)

        for col in df.columns:
            if col not in ['ticker', 'end_of_period']:
                missing_df[col] = np.nan

        result_df = pd.concat([df, missing_df], ignore_index=True)
        result_df = result_df.sort_values(['ticker', 'end_of_period'])

        return result_df

    return df


def update_null_targets(df, df_stooq, tolerance_days=7):
    result_df = df.copy()

    if result_df['end_of_period'].dtype != 'datetime64[ns]':
        result_df['end_of_period'] = pd.to_datetime(result_df['end_of_period'])

    df_stooq_prep = df_stooq.copy()
    df_stooq_prep['DATE'] = pd.to_datetime(df_stooq_prep['DATE'])
    df_stooq_prep = df_stooq_prep.rename(
        columns={
            'TICKER': 'ticker',
            'DATE': 'end_of_period',
        }
    )

    null_target_rows = result_df[result_df['target'].isna()].copy()

    if len(null_target_rows) == 0:
        return result_df

    merged_groups = []

    for ticker, group in null_target_rows.groupby('ticker'):
        group = group.sort_values('end_of_period')

        stooq_group = df_stooq_prep[df_stooq_prep['ticker'] == ticker].sort_values('end_of_period')

        if stooq_group.empty:
            merged_groups.append(group)
            continue

        stooq_group = stooq_group[['ticker', 'end_of_period', 'target']]

        try:
            merged = pd.merge_asof(
                group,
                stooq_group,
                on='end_of_period',
                by='ticker',
                direction='nearest',
                tolerance=pd.Timedelta(days=tolerance_days),
                suffixes=('', '_stooq'),
            )

            if 'target_stooq' in merged.columns:
                merged['target'] = merged['target_stooq'].combine_first(merged['target'])
                merged = merged.drop('target_stooq', axis=1)

            merged_groups.append(merged)

        except Exception as _:
            merged_groups.append(group)

    updated_rows = pd.concat(merged_groups, ignore_index=True) if merged_groups else pd.DataFrame()

    if not updated_rows.empty:
        non_null_rows = result_df[~result_df['target'].isna()].copy()
        result_df = pd.concat([non_null_rows, updated_rows], ignore_index=True)
        result_df = result_df.sort_values(['ticker', 'end_of_period'])

    return result_df


def process_missing_data():
    df = pd.read_csv('data/processed/data.csv')

    # max number of rows that can occur: 99
    num_unique_tickers = df['ticker'].nunique()
    print(f'Number of unique tickers: {num_unique_tickers}')

    print('\n')

    unique_tickers = df['ticker'].unique()
    print(unique_tickers)

    ticker_counts = df['ticker'].value_counts()

    print('\nSummary statistics:')
    print(f'Maximum entries for a ticker: {ticker_counts.max()}')
    print(f'Minimum entries for a ticker: {ticker_counts.min()}')
    print(f'Average entries per ticker: {ticker_counts.mean():.2f}')
    print(f'Median entries per ticker: {ticker_counts.median():.2f}')

    # Drop tickers that are insufficient
    tickers_to_drop = [
        'AGT',
        'ANR',
        'ASB',
        'BBT',
        'BCS',
        'BCX',
        'CRI',
        'CRJ',
        'CTS',
        'CTX',
        'DAD',
        'DNP',
        'GIF',
        'GOP',
        'GPP',
        'HLD',
        'HUG',
        'ICE',
        'IMC',
        'KDM',
        'KER',
        'MLK',
        'MLS',
        'MOC',
        'NNG',
        'NTU',
        'OND',
        'PCF',
        'PTG',
        'PUR',
        'SFG',
        'SHO',
        'SIM',
        'SLV',
        'SLZ',
        'SPH',
        'SPR',
        'STH',
        'SVRS',
        'TEN',
        'TMR',
        'TXM',
        'VRC',
    ]

    df = df[~df['ticker'].isin(tickers_to_drop)]

    df['end_of_period'] = df['end_of_period'].astype('datetime64[ns]')
    cutoff_date = pd.to_datetime('2022-07-01')
    df = df[df['end_of_period'] <= cutoff_date]

    # Fill missing quarters
    df = fill_missing_quarters(df)

    # Load stooq data
    df_stooq = pd.read_csv('data/processed/stooq_data.csv')

    df_stooq.info()

    # Update null targets
    df = update_null_targets(df, df_stooq, tolerance_days=7)

    # Prepare data for imputation
    datetime_cols = ['end_of_period']
    object_cols = ['file_name', 'ticker', 'sector']
    numeric_cols = [col for col in df.columns if col not in datetime_cols + object_cols]
    numeric_data = df[numeric_cols].copy()

    # Initialize the MICE imputer
    # Using RandomForestRegressor as the estimator often gives good results
    mice_imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42), max_iter=10, random_state=42, verbose=2
    )

    # Fit and transform the data
    imputed_numeric_data = mice_imputer.fit_transform(numeric_data)
    imputed_df = pd.DataFrame(imputed_numeric_data, columns=numeric_cols)

    # Add back the non-numeric columns
    for col in datetime_cols + object_cols:
        imputed_df[col] = df[col].values

    # Verify the imputation results
    print('Missing values before imputation:')
    print(df[numeric_cols].isna().sum())

    print('\nMissing values after imputation:')
    print(imputed_df[numeric_cols].isna().sum())

    imputed_df.to_csv('data/filled_data.csv', index=False)
    return imputed_df


if __name__ == '__main__':
    process_missing_data()
