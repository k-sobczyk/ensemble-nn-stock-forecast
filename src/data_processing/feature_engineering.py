import numpy as np
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


def categorize_sectors(df, sector_col='sector'):
    sector_mapping = {
        # Technology and Digitalization
        'gry': 'Technology and Digitalization',
        'oprogramowanie': 'Technology and Digitalization',
        'systemy informatyczne': 'Technology and Digitalization',
        'e-commerce': 'Technology and Digitalization',
        'telekomunikacja': 'Technology and Digitalization',
        'portale internetowe': 'Technology and Digitalization',
        'elektronika': 'Technology and Digitalization',
        'nowe technologie': 'Technology and Digitalization',
        'IT': 'Technology and Digitalization',
        'technologie': 'Technology and Digitalization',
        # Manufacturing and Industry
        'maszyny i urządzenia': 'Manufacturing and Industry',
        'urządzenia elektryczne': 'Manufacturing and Industry',
        'materiały budowlane': 'Manufacturing and Industry',
        'energetyka': 'Manufacturing and Industry',
        'tworzywa sztuczne': 'Manufacturing and Industry',
        'wyroby metalowe': 'Manufacturing and Industry',
        'metalurgia': 'Manufacturing and Industry',
        'chemia': 'Manufacturing and Industry',
        'górnictwo i wydobycie': 'Manufacturing and Industry',
        'części motoryzacyjne': 'Manufacturing and Industry',
        'motoryzacja': 'Manufacturing and Industry',
        'recykling': 'Manufacturing and Industry',
        'przemysł': 'Manufacturing and Industry',
        'produkcja': 'Manufacturing and Industry',
        # Services and Trade
        'odzież i obuwie': 'Services and Trade',
        'spożywczy': 'Services and Trade',
        'hotele i restauracje': 'Services and Trade',
        'biotechnologia': 'Services and Trade',
        'produkcja farmaceutyczna': 'Services and Trade',
        'usługi biznesowe': 'Services and Trade',
        'budownictwo': 'Services and Trade',
        'sprzęt medyczny': 'Services and Trade',
        'transport': 'Services and Trade',
        'handel': 'Services and Trade',
        'szpitale': 'Services and Trade',
        'usługi': 'Services and Trade',
        'handel detaliczny': 'Services and Trade',
        'finanse': 'Services and Trade',
    }

    df['sector_category'] = df[sector_col].map(sector_mapping).fillna('Services and Trade')

    return df


def create_financial_ratios(df):
    df_result = df.copy()

    # Liquidity ratios
    df_result['current_ratio'] = df_result['current_assets'] / (df_result['current_liabilities'] + 1e-8)
    df_result['cash_ratio'] = df_result['cash_and_cash_equivalents'] / (df_result['current_liabilities'] + 1e-8)

    # Leverage ratios
    df_result['debt_to_assets'] = df_result['total_liabilities'] / (df_result['total_assets'] + 1e-8)
    df_result['debt_to_equity'] = df_result['total_liabilities'] / (
        df_result['equity_shareholders_of_the_parent'] + 1e-8
    )

    # Asset efficiency ratios
    df_result['asset_turnover'] = df_result['target'] / (df_result['total_assets'] + 1e-8)  # Using share price as proxy
    df_result['ppe_intensity'] = df_result['property_plant_equipment'] / (df_result['total_assets'] + 1e-8)

    # Replace infinite values with NaN and then fill with median
    ratio_cols = ['current_ratio', 'cash_ratio', 'debt_to_assets', 'debt_to_equity', 'asset_turnover', 'ppe_intensity']
    for col in ratio_cols:
        df_result[col] = df_result[col].replace([np.inf, -np.inf], np.nan)
        df_result[col] = df_result[col].fillna(df_result[col].median())

    return df_result


def create_log_target(df, target_col='target'):
    df['target_log'] = np.log1p(df[target_col])
    return df


def round_float_columns(df, decimals=2):
    df_result = df.copy()

    # Get all numeric columns
    numeric_columns = df_result.select_dtypes(include=[np.number]).columns

    # Round numeric columns
    for col in numeric_columns:
        df_result[col] = df_result[col].round(decimals)

    return df_result


def feature_engineering():
    data = pd.read_csv('data/filled_data.csv')

    data = data.sort_values(by=['file_name', 'end_of_period'], ascending=False)

    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)

    # 1. Create total_liabilities (already implemented)
    data['total_liabilities'] = data['current_liabilities'] + data['non_current_liabilities']

    # 2. Sector categorization and consolidation
    data = categorize_sectors(data)

    # 3. Create additional financial ratios
    data = create_financial_ratios(data)

    # 4. Log transformation for target - Handle price discrepancies (2 zł to 500 zł)
    data = create_log_target(data)

    # 5. Time-lagging of financial structure variables
    attributes = [
        'total_assets',
        'non_current_assets',
        'current_assets',
        'property_plant_equipment',
        'intangible_assets',
        'inventories',
        'trade_receivables',
        'cash_and_cash_equivalents',
        'equity_shareholders_of_the_parent',
        'share_capital',
        'retained_earning_accumulated_losses',
        'non_current_liabilities',
        'current_liabilities',
        'non_current_loans_and_borrowings',
        'financial_liabilities_loans_borrowings',
        'total_liabilities',
    ]

    df = calculate_financial_differences(data, attributes)

    # Drop rows with insufficient data for time-lagging
    df = df.dropna(subset='financial_liabilities_loans_borrowings_diff_2q')

    # 6. Round all float values to 2 decimal places
    df = round_float_columns(df, decimals=2)

    # Move target to the end (keep original target as main target)
    df['target_log'] = df.pop('target_log')

    df.to_csv('data/data_with_features.csv', index=False)

    return df


if __name__ == '__main__':
    feature_engineering()
