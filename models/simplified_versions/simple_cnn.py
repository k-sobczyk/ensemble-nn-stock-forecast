import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from models.metrics.metrics import print_metrics

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in divide')


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ModelCNN(nn.Module):
    def __init__(self, num_features: int, num_filters1: int, num_filters2: int, fc1_neurons: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(num_filters1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(num_filters2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.flattened_size = num_filters2 * num_features

        self.fc1 = nn.Linear(self.flattened_size, fc1_neurons)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc1_neurons, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


def load_and_split_data(
    file_path: str, date_column: str, ticker_column: str, test_years: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=[ticker_column, date_column])

    test_mask = df[date_column].dt.year.isin(test_years)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    # Keep ticker_column for now - will be processed in prepare_data
    train_df = train_df.drop(columns=[date_column])
    test_df = test_df.drop(columns=[date_column])

    total_samples = len(df)
    train_samples = len(train_df)
    test_samples = len(test_df)

    test_ratio = test_samples / total_samples * 100 if total_samples > 0 else 0

    print(f'Data Split: Total: {total_samples}, Train: {train_samples}, Test: {test_samples} ({test_ratio:.2f}%)')

    return train_df, test_df, test_ratio


def prepare_data_for_cnn(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str, ticker_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
    # Handle company ticker column with one-hot encoding
    print(f'Unique tickers in training data: {len(train_df[ticker_column].unique())}')

    # Option 1: Drop the ticker column
    X_train_raw = train_df.drop(columns=[target_column, ticker_column]).values
    y_train_raw = train_df[target_column].values.reshape(-1, 1)
    X_test_raw = test_df.drop(columns=[target_column, ticker_column]).values
    y_test_raw = test_df[target_column].values.reshape(-1, 1)

    # Option 2 (commented out): One-hot encode the ticker column
    # This can be uncommented if you want to include company information
    # Note: This can significantly increase the feature dimension
    """
    # Get all columns except target and ticker for numerical features
    num_features = train_df.drop(columns=[target_column, ticker_column])
    
    # One-hot encode the ticker column
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    train_tickers_encoded = encoder.fit_transform(train_df[[ticker_column]])
    test_tickers_encoded = encoder.transform(test_df[[ticker_column]])
    
    # Scale numerical features
    scaler_X = StandardScaler()
    X_train_num_scaled = scaler_X.fit_transform(num_features)
    X_test_num_scaled = scaler_X.transform(test_df.drop(columns=[target_column, ticker_column]))
    
    # Combine numerical and one-hot features
    X_train_raw = np.hstack([X_train_num_scaled, train_tickers_encoded])
    X_test_raw = np.hstack([X_test_num_scaled, test_tickers_encoded])
    """

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    num_features = X_train_scaled.shape[1]
    X_train = X_train_scaled.reshape(-1, 1, num_features, 1)
    X_test = X_test_scaled.reshape(-1, 1, num_features, 1)

    return X_train, X_test, y_train_scaled, y_test_scaled, y_train_raw.flatten(), scaler_y, num_features


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 100,
    device: torch.device = torch.device('cpu'),
) -> tuple[pd.DataFrame, nn.Module]:
    model.to(device)
    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Evaluation phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * inputs.size(0)

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f'Epoch {epoch + 1:03d}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f}'
            )

    training_time = time.time() - start_time
    print(f'Training finished in {training_time:.2f} seconds')

    loss_df = pd.DataFrame(
        {'Epoch': list(range(1, len(train_losses) + 1)), 'Train Loss': train_losses, 'Test Loss': test_losses}
    )

    return loss_df, model


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, scaler_y: StandardScaler, device: torch.device = torch.device('cpu')
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions_scaled = []
    actuals_scaled = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions_scaled.extend(outputs.cpu().numpy())
            actuals_scaled.extend(targets.cpu().numpy())

    predictions_scaled = np.array(predictions_scaled)
    actuals_scaled = np.array(actuals_scaled)

    predictions = scaler_y.inverse_transform(predictions_scaled)
    actuals = scaler_y.inverse_transform(actuals_scaled)

    return predictions.flatten(), actuals.flatten()


def main():
    DATA_FILE = 'C:/Users/ksobc/PycharmProjects/ensemble-nn-stock-forecast/models/model_with_features.csv'
    DATE_COLUMN = 'end_of_period'
    TARGET_COLUMN = 'target'
    TICKER_COLUMN = 'ticker'  # Added ticker column
    TEST_YEARS = [2021, 2022]

    # Set fixed hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 100
    NUM_FILTERS1 = 32
    NUM_FILTERS2 = 64
    FC1_NEURONS = 64
    DROPOUT_RATE = 0.3

    # Set device to CUDA if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load and prepare data
    train_df, test_df, _ = load_and_split_data(DATA_FILE, DATE_COLUMN, TICKER_COLUMN, TEST_YEARS)
    X_train, X_test, y_train, y_test, y_train_raw, scaler_y, num_features = prepare_data_for_cnn(
        train_df, test_df, TARGET_COLUMN, TICKER_COLUMN
    )

    # Create datasets and data loaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    pin_memory = True if device.type == 'cuda' else False

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    # Create model with fixed parameters
    model = ModelCNN(
        num_features=num_features,
        num_filters1=NUM_FILTERS1,
        num_filters2=NUM_FILTERS2,
        fc1_neurons=FC1_NEURONS,
        dropout_rate=DROPOUT_RATE,
    )

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print('\n--- Training CNN model ---')
    _, trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=EPOCHS, device=device)

    # Evaluate the model
    print('\n--- Evaluating model ---')
    y_pred, y_true = evaluate_model(trained_model, test_loader, scaler_y, device=device)
    print_metrics(y_true, y_pred, y_train_raw)

    # Save predictions
    results_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    results_df.to_csv('simple_cnn_predictions.csv', index=False, float_format='%.4f')
    print('\nModel predictions saved to simple_cnn_predictions.csv')

    # Save model
    torch.save(trained_model.state_dict(), 'simple_cnn_model.pt')
    print('Model saved to simple_cnn_model.pt')


if __name__ == '__main__':
    main()
