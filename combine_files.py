import os
import pandas as pd

directory_path = 'wse stocks/'

columns = ["TICKER", "PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "OPENINT"]

all_dfs = []

for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        try:
            temp_df = pd.read_csv(file_path, delimiter=',')
            if temp_df.shape[1] == len(columns):
                all_dfs.append(temp_df)
            else:
                print(f'File {file_path} does not match the expected number of columns.')
        except Exception as e:
            print(f'Error reading {file_path}: {e}')

combined_df = pd.concat(all_dfs, ignore_index=True)

output_file_path = 'combined_data.csv'
combined_df.to_csv(output_file_path, index=False)

print(f'Combined data saved to {output_file_path}')
