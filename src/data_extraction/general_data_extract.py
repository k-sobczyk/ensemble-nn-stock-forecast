import os

import pandas as pd
from tqdm import tqdm


def extract_general_info(file_path: str) -> dict[str, str] | None:
    try:
        info_sheet = pd.read_excel(file_path, sheet_name='Info', header=None)
        name = info_sheet.iloc[2, 1]
        ticker = info_sheet.iloc[12, 1]
        sector = info_sheet.iloc[20, 4]
        file_name = os.path.basename(file_path)
        return {'Name': name, 'TICKER': ticker, 'Sector': sector, 'File_Name': file_name}
    except Exception as e:
        print(f'Error processing file {file_path}: {str(e)}')
        return None


def process_general_info(folder_path: str) -> pd.DataFrame:
    all_data: list[dict[str, str]] = []
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    for file in tqdm(excel_files, desc='Processing general info files'):
        file_path = os.path.join(folder_path, file)
        data = extract_general_info(file_path)
        if data:
            all_data.append(data)
    return pd.DataFrame(all_data)


def update_sector(row):
    sector_updates = {
        '3LP SA': 'handel',
        'BMW AG': 'motoryzacja',
        'Mercedes-Benz Group AG': 'motoryzacja',
        'Uf Games SA': 'gry',
    }
    if row['Name'] in sector_updates:
        return sector_updates[row['Name']]
    return row['Sector']


def main():
    data_folder = 'C:/Users/ksobc/PycharmProjects/ensemble-nn-stock-forecast/data/raw'
    df_general_info = process_general_info(data_folder)
    df_general_info['Sector'] = df_general_info.apply(update_sector, axis=1)
    df_general_info.to_csv(
        'C:/Users/ksobc/PycharmProjects/ensemble-nn-stock-forecast/data/processed/general_company_info.csv', index=False
    )
    print(f'Processed {df_general_info.shape[0]} records and saved to general_company_info.csv')


if __name__ == '__main__':
    main()
