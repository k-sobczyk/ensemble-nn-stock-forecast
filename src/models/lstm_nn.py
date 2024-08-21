from model_utils import load_scale_split
from scripts.calculate_metrics import calculate_metrics
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class StackedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.0):
        super(StackedLSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_sizes[i-1],
                    hidden_sizes[i],
                    1,
                    batch_first=True,
                    dropout=dropout if i < len(hidden_sizes) - 1 else 0)
            for i in range(len(hidden_sizes))
        ])
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        out = self.fc(x[:, -1, :])
        return out


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
    # Set random seed for reproducibility
    set_seed(42)
    df = pd.read_csv('../../data/processed/model_with_features.csv')

    # Split the data
    X_train, X_test, y_train, y_test = load_scale_split(df)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model parameters
    input_size = X_train.shape[2]  # Assuming X_train is of shape (samples, time_steps, features)
    hidden_size = 64
    num_layers = 2
    output_size = 1
    dropout = 0.2

    # Initialize models
    models = {
        'LSTM': LSTMModel(input_size, hidden_size, num_layers, output_size, dropout),
        'BiLSTM': BiLSTMModel(input_size, hidden_size, num_layers, output_size, dropout),
        'StackedLSTM': StackedLSTMModel(input_size, [64, 32, 16], output_size, dropout)
    }

    # Training parameters
    criterion = nn.MSELoss()
    epochs = 100
    patience = 20

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        optimizer = torch.optim.Adam(model.parameters())

        # Train the model
        train_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience)

        # Evaluate the model
        predictions, actuals = evaluate_model(model, test_loader)

        # Calculate metrics
        metrics = calculate_metrics(actuals, predictions)

        # Save results
        results_df = pd.DataFrame({
            'Actual': actuals.flatten(),
            'Predicted': predictions.flatten()
        })
        results_df.to_csv(f'results/{model_name}_predictions.csv', index=False)

        # Save metrics
        with open(f'results/{model_name}_metrics.txt', 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")


if __name__ == "__main__":
    main()
