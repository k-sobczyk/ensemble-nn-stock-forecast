import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_split_data(
    file_path: str, date_column: str, test_years: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Loads data, splits into train/test based on years."""
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=[date_column])

    test_mask = df[date_column].dt.year.isin(test_years)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    total_samples = len(df)
    train_samples = len(train_df)
    test_samples = len(test_df)

    test_ratio = test_samples / total_samples * 100 if total_samples > 0 else 0

    print(f'Data Split: Total: {total_samples}, Train: {train_samples}, Test: {test_samples} ({test_ratio:.2f}%)')

    if 'ticker' in df.columns:
        print(f'Number of unique companies (tickers): {df["ticker"].nunique()}')
        print(f'Train companies: {train_df["ticker"].nunique()}, Test companies: {test_df["ticker"].nunique()}')

    return train_df, test_df, test_ratio


def prepare_base_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str, ticker_column: str = 'ticker'
) -> tuple:
    """Scales features and target, handles ticker information."""
    train_tickers = train_df[ticker_column].values
    test_tickers = test_df[ticker_column].values

    # Encode company tickers
    ticker_encoder = LabelEncoder()
    train_ticker_encoded = ticker_encoder.fit_transform(train_tickers)
    test_ticker_encoded = ticker_encoder.transform(test_tickers)

    print(f'Encoded {len(ticker_encoder.classes_)} unique companies')
    num_companies = len(ticker_encoder.classes_)
    numeric_cols = [
        col
        for col in train_df.columns
        if col != target_column
        and col != ticker_column
        and (pd.api.types.is_numeric_dtype(train_df[col]) or col.startswith('sector_'))
    ]

    print(f'Using {len(numeric_cols)} numeric features')

    X_train_raw = train_df[numeric_cols].values
    y_train_raw = train_df[target_column].values.reshape(-1, 1)
    X_test_raw = test_df[numeric_cols].values
    y_test_raw = test_df[target_column].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    num_features = X_train_scaled.shape[1]

    print(f'Number of features: {num_features}')
    print(f'Scaled Train Shapes: X={X_train_scaled.shape}, y={y_train_scaled.shape}')
    print(f'Scaled Test Shapes: X={X_test_scaled.shape}, y={y_test_scaled.shape}')

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        y_train_raw.flatten(),
        scaler_X,
        scaler_y,
        num_features,
        train_ticker_encoded,
        test_ticker_encoded,
        num_companies,
    )


def create_sequences_by_company(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    company_ids: np.ndarray,
    sequence_length: int = 2,
    min_samples_per_company: int = 4,
) -> tuple:
    """Creates sequences for LSTM input, respecting company boundaries with handling for sparse data."""
    X_seq, y_seq, company_seq = [], [], []

    unique_companies = np.unique(company_ids)
    for company in unique_companies:
        # Get indices for this company
        company_mask = company_ids == company
        company_X = X_scaled[company_mask]
        company_y = y_scaled[company_mask]

        if len(company_X) < min_samples_per_company:
            print(
                f'Company ID {company} has only {len(company_X)} samples, skipping (min required: {min_samples_per_company})'
            )
            continue

        # Use shorter sequence length if needed but at least 2
        actual_seq_length = min(sequence_length, len(company_X) - 1)
        actual_seq_length = max(actual_seq_length, 2)  # At least 2 for minimal sequence

        if actual_seq_length < sequence_length:
            print(
                f'Reducing sequence length to {actual_seq_length} for company ID {company} (original: {sequence_length})'
            )

        for i in range(len(company_X) - actual_seq_length):
            X_seq.append(company_X[i : i + actual_seq_length])
            y_seq.append(company_y[i + actual_seq_length])
            company_seq.append(company)

    if not X_seq:
        print(
            f'Warning: No sequences created with sequence_length={sequence_length}. Consider reducing sequence length.'
        )
        num_features = X_scaled.shape[1] if X_scaled.ndim > 1 else 1
        return np.empty((0, sequence_length, num_features)), np.empty((0, 1)), np.empty((0,))

    max_seq_len = max(len(seq) for seq in X_seq)
    padded_X_seq = []

    for seq in X_seq:
        if len(seq) < max_seq_len:
            # Pad the sequence with zeros
            pad_size = max_seq_len - len(seq)
            padded_seq = np.vstack([np.zeros((pad_size, seq.shape[1])), seq])
            padded_X_seq.append(padded_seq)
        else:
            padded_X_seq.append(seq)

    return np.array(padded_X_seq), np.array(y_seq), np.array(company_seq)
