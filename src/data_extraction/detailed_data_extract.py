import os

import pandas as pd
from tqdm import tqdm


def extract_financial_details(file_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(file_path, sheet_name='QS', header=None)

        data = {
            'date': df.iloc[3, 3:].values,
            'assets': df.iloc[12, 3:].values,
            'non_current_assets': df.iloc[13, 3:].values,
            'current_assets': df.iloc[14, 3:].values,
            'property_plant_equipment': df.iloc[31, 3:].values,
            'intangible_assets': df.iloc[33, 3:].values,
            'inventories': df.iloc[45, 3:].values,
            'trade_receivables': df.iloc[48, 3:].values,
            'cash_and_cash_equivalents': df.iloc[51, 3:].values,
            'equity_shareholders_of_the_parent': df.iloc[61, 3:].values,
            'share_capital': df.iloc[62, 3:].values,
            'retained_earning_accumulated_losses': df.iloc[68, 3:].values,
            'non_current_liabilities': df.iloc[70, 3:].values,
            'current_liabilities': df.iloc[81, 3:].values,
            'non_current_loans_and_borrowings': df.iloc[72, 3:].values,
            'financial_liabilities_loans_borrowings': df.iloc[83, 3:].values,
            'total_shares': df.iloc[18, 3:].values,
        }

        result_df = pd.DataFrame(data)
        result_df['filename'] = os.path.basename(file_path)

        return result_df

    except Exception as e:
        print(f'Error processing file {file_path}: {str(e)}')
        return None


def process_financial_details(folder_path: str) -> pd.DataFrame:
    all_data: list[pd.DataFrame] = []
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    for file in tqdm(excel_files, desc='Processing financial details files'):
        file_path = os.path.join(folder_path, file)
        data = extract_financial_details(file_path)
        if data is not None:
            all_data.append(data)

    return pd.concat(all_data, ignore_index=True)


def main():
    data_folder = 'C:/Users/ksobc/PycharmProjects/ensemble-nn-stock-forecast/data/raw'
    df_financial_details = process_financial_details(data_folder)
    df_financial_details.dropna(subset='date', inplace=True)
    df_financial_details.to_csv(
        'C:/Users/ksobc/PycharmProjects/ensemble-nn-stock-forecast/data/processed/details_company_info.csv', index=False
    )
    print(f'Processed {df_financial_details.shape[0]} records and saved to details_company_info.csv')


if __name__ == '__main__':
    main()
