import os

import pandas as pd


def main():
    data_folder = 'C:/Users/ksobc/PycharmProjects/ensemble-nn-stock-forecast/data/stooq/historic_market_value'
    columns = ['TICKER', 'PER', 'DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'OPENINT']
    all_dfs = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_folder, filename)

            if os.path.getsize(file_path) == 0:
                continue

            try:
                temp_df = pd.read_csv(file_path, delimiter=',')
                if temp_df.shape[1] == len(columns):
                    temp_df.columns = columns
                    all_dfs.append(temp_df)
                else:
                    print(f'File {file_path} does not match the expected number of columns.')
            except Exception as e:
                print(f'Error reading file {filename}: {str(e)}')

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['DATE'] = pd.to_datetime(combined_df['DATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    combined_df['target'] = (combined_df['OPEN'] + combined_df['CLOSE']) / 2

    output_file_path = 'C:/Users/ksobc/PycharmProjects/ensemble-nn-stock-forecast/data/processed/stooq_data.csv'
    combined_df.to_csv(output_file_path, index=False)
    print(f'Processed {combined_df.shape[0]} records and saved to stooq_data.csv')


if __name__ == '__main__':
    main()
