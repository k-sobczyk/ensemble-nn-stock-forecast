import pandas as pd


def calculate_financial_differences(df, attributes, periods=[1, 2], company_col='file_name', sort_col='end_of_period'):
    df = df.sort_values([company_col, sort_col])
    result_df = df.copy()

    # Calculate differences for each attribute and period
    for attr in attributes:
        for period in periods:
            diff_col_name = f'{attr}_diff_{period}q'
            result_df[diff_col_name] = result_df.groupby(company_col)[attr].diff(periods=period)

    return result_df


def feature_engineering():
    data = pd.read_csv('data/filled_data.csv')

    data = (data.sort_values(by=['file_name', 'end_of_period'], ascending=False))

    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)

    data['total_liabilities'] = data['current_liabilities'] + data['non_current_liabilities']

    attributes = [
        'total_assets', 'non_current_assets', 'current_assets',
        'property_plant_equipment', 'intangible_assets', 'inventories',
        'trade_receivables', 'cash_and_cash_equivalents', 'equity_shareholders_of_the_parent',
        'share_capital', 'retained_earning_accumulated_losses', 'non_current_liabilities',
        'current_liabilities', 'non_current_loans_and_borrowings', 'financial_liabilities_loans_borrowings'
    ]

    df = calculate_financial_differences(data, attributes)

    df = df.dropna(subset='financial_liabilities_loans_borrowings_diff_2q')

    df['target'] = df.pop('target')

    df.to_csv('data/data_with_features.csv', index=False)

    return df


if __name__ == '__main__':
    feature_engineering()
