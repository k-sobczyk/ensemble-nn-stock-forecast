import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y


def reshape_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    return X_train, X_test, y_train, y_test


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100, patience=20):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        train_losses.append(epoch_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    loss_df = pd.DataFrame(
        {'Epoch': list(range(1, len(train_losses) + 1)), 'Train Loss': train_losses[:len(test_losses)],
         'Test Loss': test_losses})
    loss_df.to_csv("mini_lstm_training_results.csv", index=False)


def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            actuals.extend(targets.numpy())
    return np.array(predictions), np.array(actuals)


def main():
    set_seed(42)
    # nn_df = pd.read_csv('nn_df_scaled.csv')
    # RMSE: 0.2934167683124542
    # MAE: 0.03908737748861313
    # R²: 0.9165587083698881

    mini_df = pd.read_csv('model_data/mini_df_scaled.csv')
    # RMSE: 0.2669451832771301
    # MAE: 0.04278215765953064
    # R²: 0.9219936070023995

    X, y = preprocess_data(mini_df)
    X_train, X_test, y_train, y_test = reshape_data(X, y)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X_train.shape[2]
    hidden_size = 50
    num_layers = 2

    model = LSTMModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100)

    y_pred, y_true = evaluate_model(model, test_loader)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R²: {r2}')


if __name__ == "__main__":
    main()
