import os
import pandas as pd
from tqdm import tqdm


def count_files_in_directory(directory_path) -> int:
    """
    Count the number of files in the specified directory.

    Args:
    directory_path (str): Path to the directory

    Returns:
    int: Number of files
    """
    try:
        # Ensure the directory exists
        if not os.path.exists(directory_path):
            print(f"The directory {directory_path} does not exist.")
            return 0

        count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

        return count

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0


# General Info Functions
def extract_general_info(file_path):
    try:
        # Read the 'Info' sheet
        info_sheet = pd.read_excel(file_path, sheet_name='Info', header=None)

        # Extract required information
        name = info_sheet.iloc[2, 1]  # B3
        ticker = info_sheet.iloc[12, 1]  # B13
        sector = info_sheet.iloc[20, 4]  # E21

        # Get the file name
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


def process_general_info(folder_path):
    all_data = []

    # Get all xlsx files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # Process each file with a progress bar
    for file in tqdm(excel_files, desc="Processing general info files"):
        file_path = os.path.join(folder_path, file)
        data = extract_general_info(file_path)
        if data:
            all_data.append(data)

    # Create a DataFrame from all extracted data
    df = pd.DataFrame(all_data)
    return df


# Financial Details Functions
def extract_financial_details(file_path):
    try:
        # Read the 'QS' sheet
        df = pd.read_excel(file_path, sheet_name='QS', header=None)

        # Extract data for each metric
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

        # Create a DataFrame
        result_df = pd.DataFrame(data)

        # Add filename column
        result_df['filename'] = os.path.basename(file_path)

        return result_df

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None


def process_financial_details(folder_path):
    all_data = []

    # Get all xlsx files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # Process each file with a progress bar
    for file in tqdm(excel_files, desc="Processing financial details files"):
        file_path = os.path.join(folder_path, file)
        data = extract_financial_details(file_path)
        if data is not None:
            all_data.append(data)

    # Concatenate all DataFrames
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df
