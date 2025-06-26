import pandas as pd
import pyarrow as pa


def clean_general_company_info():
    df_general = pd.read_csv('data/processed/general_company_info.csv')

    rename = {'Name': 'company_name', 'TICKER': 'ticker', 'Sector': 'sector', 'File_Name': 'file_name'}
    dtypes = {
        'company_name': pd.ArrowDtype(pa.string()),
        'ticker': pd.ArrowDtype(pa.string()),
        'sector': pd.ArrowDtype(pa.string()),
        'file_name': pd.ArrowDtype(pa.string()),
    }

    df_general = df_general.rename(columns=rename)
    df_general = df_general.astype(dtypes)

    return df_general


def clean_stooq_data():
    df_market_value = pd.read_csv('data/processed/stooq_data.csv')

    rename = {
        'TICKER': 'ticker',
        'DATE': 'end_of_period',
    }

    df_market_value = df_market_value.rename(columns=rename)
    df_market_value = df_market_value[['ticker', 'end_of_period', 'target']]

    dtypes = {'ticker': pd.ArrowDtype(pa.string()), 'end_of_period': 'datetime64[s]', 'target': 'float32[pyarrow]'}

    df_market_value = df_market_value.astype(dtypes)

    return df_market_value


def clean_detailed_company_info():
    df_detailed = pd.read_csv('data/processed/details_company_info.csv')

    rename = {'date': 'end_of_period', 'filename': 'file_name', 'assets': 'total_assets'}

    dtypes = {
        'end_of_period': 'datetime64[s]',
        'total_assets': 'float32[pyarrow]',
        'non_current_assets': 'float32[pyarrow]',
        'current_assets': 'float32[pyarrow]',
        'property_plant_equipment': 'float32[pyarrow]',
        'intangible_assets': 'float32[pyarrow]',
        'inventories': 'float32[pyarrow]',
        'trade_receivables': 'float32[pyarrow]',
        'cash_and_cash_equivalents': 'float32[pyarrow]',
        'equity_shareholders_of_the_parent': 'float32[pyarrow]',
        'share_capital': 'float32[pyarrow]',
        'retained_earning_accumulated_losses': 'float32[pyarrow]',
        'non_current_liabilities': 'float32[pyarrow]',
        'current_liabilities': 'float32[pyarrow]',
        'non_current_loans_and_borrowings': 'float32[pyarrow]',
        'financial_liabilities_loans_borrowings': 'float32[pyarrow]',
        'total_shares': 'float32[pyarrow]',
        'file_name': pd.ArrowDtype(pa.string()),
    }

    df_detailed = df_detailed.rename(columns=rename)
    df_detailed = df_detailed.astype(dtypes)

    return df_detailed


def merge_dataframes(df_detailed, df_general, df_market_value):
    df_detailed = df_detailed.merge(df_general[['file_name', 'ticker', 'sector']], how='left', on='file_name')
    merged_groups = []

    for ticker, group in df_detailed.groupby('ticker'):
        group = group.sort_values('end_of_period')
        market_group = df_market_value[df_market_value['ticker'] == ticker].sort_values('end_of_period')

        merged = pd.merge_asof(
            group, market_group, on='end_of_period', direction='nearest', tolerance=pd.Timedelta(days=7)
        )
        merged_groups.append(merged)

    df_merged = pd.concat(merged_groups, ignore_index=True)

    df_merged = df_merged.dropna(
        subset=['end_of_period', 'target', 'total_assets', 'current_assets', 'non_current_assets']
    )
    df_merged = df_merged.fillna(0)

    df = df_merged.copy()
    df = df.round(2)
    df.drop(columns='ticker_y', inplace=True)
    df.rename(columns={'ticker_x': 'ticker'}, inplace=True)
    return df


def adjust_date_columns(df):
    """Adjusting Date columns to the same quarters.

    move every 01, 02 to 03
    move every 04, 05 to 06
    move every 07, 08 to 09
    move every 10, 11 to 12

    to keep the same quarters per files.
    """
    df['end_of_period'] = df['end_of_period'].apply(
        lambda x: pd.Timestamp(
            year=x.year,
            month=(3 if x.month in [1, 2] else 6 if x.month in [4, 5] else 9 if x.month in [7, 8] else 12),
            day=1,
        )
    )

    return df


def create_missing_quarters_report(df):
    company_id = 'ticker' if 'ticker' in df.columns else 'file_name'
    df[company_id] = df[company_id].astype(str)
    df['present'] = 1

    # Create a pivot table with companies as rows and end_of_period as columns.
    # Using aggfunc='max' ensures that if there is at least one record for that quarter, the value is 1.
    pivot_df = df.pivot_table(index=company_id, columns='end_of_period', values='present', aggfunc='max', fill_value=0)
    pivot_df = pivot_df.reindex(fill_value=0)
    pivot_df = pivot_df.astype(int)

    file_path = 'data/missing_quarters_report.xlsx'
    pivot_df.to_excel(file_path)
    print(f'File saved successfully: {file_path}')


def normalization_and_merging():
    df_general = clean_general_company_info()
    df_market_value = clean_stooq_data()
    df_detailed = clean_detailed_company_info()

    # Merging into one dataframe
    df = merge_dataframes(df_detailed, df_general, df_market_value)

    # Adjusting Date columns to the same quarters
    df = adjust_date_columns(df)

    df.to_csv('data/processed/data.csv', index=False)

    create_missing_quarters_report(df)

    return df


if __name__ == '__main__':
    normalization_and_merging()
