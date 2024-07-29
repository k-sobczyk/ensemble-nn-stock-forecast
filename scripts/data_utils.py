from typing import Dict, List, Optional

import os
import pandas as pd
from tqdm import tqdm


def count_files_in_directory(directory_path: str) -> int:
    """Count the number of files in the specified directory."""
    try:
        if not os.path.exists(directory_path):
            print(f"The directory {directory_path} does not exist.")
            return 0

        count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        return count

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0


def extract_general_info(file_path: str) -> Optional[Dict[str, str]]:
    """Extract general information from an Excel file."""
    try:
        info_sheet = pd.read_excel(file_path, sheet_name='Info', header=None)

        name = info_sheet.iloc[2, 1]  # B3
        ticker = info_sheet.iloc[12, 1]  # B13
        sector = info_sheet.iloc[20, 4]  # E21
        file_name = os.path.basename(file_path)

        return {
            'Name': name,
            'TICKER': ticker,
            'Sector': sector,
            'File_Name': file_name
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None


def process_general_info(folder_path: str) -> pd.DataFrame:
    """Process all Excel files in a folder to extract general information."""
    all_data: List[Dict[str, str]] = []
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    for file in tqdm(excel_files, desc="Processing general info files"):
        file_path = os.path.join(folder_path, file)
        data = extract_general_info(file_path)
        if data:
            all_data.append(data)

    return pd.DataFrame(all_data)


def extract_financial_details(file_path: str) -> Optional[pd.DataFrame]:
    """Extract financial details from an Excel file."""
    try:
        df = pd.read_excel(file_path, sheet_name='QS', header=None)

        data = {
            'date': df.iloc[0, 3:].values,
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
            'total_shares': df.iloc[18, 3:].values
        }

        result_df = pd.DataFrame(data)
        result_df['filename'] = os.path.basename(file_path)

        return result_df

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None


def process_financial_details(folder_path: str) -> pd.DataFrame:
    """Process all Excel files in a folder to extract financial details."""
    all_data: List[pd.DataFrame] = []
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    for file in tqdm(excel_files, desc="Processing financial details files"):
        file_path = os.path.join(folder_path, file)
        data = extract_financial_details(file_path)
        if data is not None:
            all_data.append(data)

    return pd.concat(all_data, ignore_index=True)


def process_market_value_files(directory_path: str) -> pd.DataFrame:
    all_dfs = []
    columns = ['TICKER', 'PER', 'DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'OPENINT']

    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

    # Process each file with a progress bar
    for filename in tqdm(txt_files, desc="Processing market value files"):
        file_path = os.path.join(directory_path, filename)
        try:
            temp_df = pd.read_csv(file_path, delimiter=',')
            if temp_df.shape[1] == len(columns):
                temp_df.columns = columns
                all_dfs.append(temp_df)
            else:
                print(f'File {file_path} does not match the expected number of columns.')
        except Exception as e:
            print(f'Error processing file {file_path}: {str(e)}')

    if not all_dfs:
        print("No valid files were processed.")
        return pd.DataFrame()

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Convert DATE to datetime and format it
    combined_df['DATE'] = pd.to_datetime(combined_df['DATE'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')

    # Remove rows with invalid dates
    combined_df = combined_df.dropna(subset=['DATE'])

    # Sort the DataFrame by TICKER and DATE
    combined_df = combined_df.sort_values(['TICKER', 'DATE'])

    print(f"Processed {len(all_dfs)} files. Final DataFrame shape: {combined_df.shape}")

    return combined_df
