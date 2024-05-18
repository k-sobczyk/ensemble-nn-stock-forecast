import re


def count_rows_by_filename(df):
    counts = df.groupby('filename').size().reset_index(name='count')
    return counts


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
