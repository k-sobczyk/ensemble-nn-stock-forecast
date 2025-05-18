import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_scale_split(df, split_percentage=0.2, random_state=42):
    # Identify date column and target column
    date_column = 'end_of_period'
    target_column = 'target'

    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Identify numeric columns (excluding the date column)
    numeric_columns = X.select_dtypes(include=[np.number]).columns.drop(date_column, errors='ignore')

    # Scale the numeric features
    scaler = StandardScaler()
    X.loc[:, numeric_columns] = scaler.fit_transform(X[numeric_columns])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage, random_state=random_state)

    return X_train, X_test, y_train, y_test


def preprocess_data(X):
    for column in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[column]):
            # Convert datetime to timestamp (seconds since epoch)
            X[column] = X[column].astype(int) / 10**9
        elif pd.api.types.is_object_dtype(X[column]):
            # Convert categorical variables to numeric
            X[column] = pd.Categorical(X[column]).codes

    # Ensure all data is float
    X = X.astype(float)

    return X
