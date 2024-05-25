import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ModelCNN(nn.Module):
    def __init__(self, input_shape):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1))
        self.fc1 = nn.Linear(32 * (input_shape[2] - 4), 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def prepare_data(df, target_column='target'):
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    return X, y


def reshape_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, 1, X_train.shape[1], 1)
    X_test = X_test.reshape(-1, 1, X_test.shape[1], 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, X_test, y_train, y_test


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50, patience=5):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets.squeeze())
                test_loss += loss.item() * inputs.size(0)
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}')

        #Early stopping mechanism
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    loss_df = pd.DataFrame({'Epoch': list(range(1, len(train_losses) + 1)), 'Train Loss': train_losses[:len(test_losses)], 'Test Loss': test_losses})
    loss_df.to_csv("mini_cnn_training_results.csv", index=False)


def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    return np.array(predictions), np.array(actuals)


def main():
    # nn_df = pd.read_csv('nn_df_scaled.csv')
    #RMSE: 0.37572747468948364
    #MAE: 0.08156466484069824
    #R²: 0.8631776570559896

    mini_df = pd.read_csv('mini_df_scaled.csv')
    #RMSE: 0.25536617636680603
    #MAE: 0.08266189694404602
    #R²: 0.928614054627815

    X, y = prepare_data(mini_df)
    X_train, X_test, y_train, y_test = reshape_data(X, y)

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ModelCNN(input_shape=X_train.shape)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50)

    y_pred, y_true = evaluate_model(model, test_loader)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R²: {r2}')


if __name__ == "__main__":
    main()
