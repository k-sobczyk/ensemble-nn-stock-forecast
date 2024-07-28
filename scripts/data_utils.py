import re
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


# Company General Info
def extract_info_from_excel(file_path):
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


def process_all_files(folder_path):
    all_data = []

    # Get all xlsx files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # Process each file with a progress bar
    for file in tqdm(excel_files, desc="Processing files"):
        file_path = os.path.join(folder_path, file)
        data = extract_info_from_excel(file_path)
        if data:
            all_data.append(data)

    # Create a DataFrame from all extracted data
    df = pd.DataFrame(all_data)
    return df


def clean_column_names(columns):
    cleaned_columns = []
    for column in columns:
        column = column.lower()
        column = re.sub(r'\(.*?\)', '', column)
        column = re.sub(r'[^\w\s]', '', column)
        column = re.sub(r'\s+', '_', column)
        column = column.rstrip('_')
        cleaned_columns.append(column)
    return cleaned_columns

