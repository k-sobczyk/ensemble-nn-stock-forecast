import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_scale_split(csv_path, split_percentage=0.2, random_state=42):
    df = pd.read_csv(csv_path)

    # Identify date column and target column
    date_column = 'end_of_period'
    target_column = 'target'

    # Convert date column to datetime64
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    target = df.pop(target_column)
    df[target_column] = target

    # Separate features and target
    y = df.pop(target_column)
    X = df

    # Identify numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns

    # Scale the numeric features
    scaler = StandardScaler()
    X.loc[:, numeric_columns] = scaler.fit_transform(X[numeric_columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_percentage, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
