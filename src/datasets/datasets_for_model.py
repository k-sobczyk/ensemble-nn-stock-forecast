import pandas as pd


def prepare_model_data():
    df = pd.read_csv('data/data_with_features.csv')
    print(df.columns)
    # One Hot Encoding
    # df = (pd.get_dummies(df, columns=['sector'])
    #         .drop(columns=['file_name']))

    df.to_csv('data/processed/model_with_features.csv', index=False)

    return df


if __name__ == '__main__':
    prepare_model_data()
