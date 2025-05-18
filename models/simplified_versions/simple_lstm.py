import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import datetime

# Listing 1: LSTM Model Definition [cite: 348]
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Corrected from self.fcnn.Linear to self.fc

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Listing 2: Setting the seed for reproducibility [cite: 406]
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # [cite: 404]
        torch.cuda.manual_seed(seed) # [cite: 404]
        torch.cuda.manual_seed_all(seed) # [cite: 405]
        torch.backends.cudnn.deterministic = True # [cite: 405]
        torch.backends.cudnn.benchmark = False # [cite: 406]

# Listing 3: Preprocessing the data with time-based split
def preprocess_data(df):
    # Convert date string to datetime
    df['end_of_period'] = pd.to_datetime(df['end_of_period'])
    
    # Create a year column for filtering
    df['year'] = df['end_of_period'].dt.year
    
    # Split based on year
    train_df = df[df['year'] < 2021]
    test_df = df[(df['year'] >= 2021) & (df['year'] <= 2022)]
    
    # Drop non-numeric columns that won't be used for modeling
    drop_cols = ['end_of_period', 'year', 'ticker']
    
    X_train = train_df.drop(['target'] + drop_cols, axis=1)
    y_train = train_df['target']
    
    X_test = test_df.drop(['target'] + drop_cols, axis=1)
    y_test = test_df['target']
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

# Listing 4: Reshaping the data for LSTM
def reshape_data_for_lstm(X_train, X_test, y_train, y_test, seq_length=1):
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Reshape for LSTM [batch, seq_len, features]
    if seq_length > 1:
        # Create sequences
        # This is a simplified approach - for a real application, you'd need to 
        # ensure sequences are created properly considering time order
        # and not crossing different stocks/tickers
        pass
    else:
        # Just add sequence dimension of 1
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# Listing 5: Training the model [cite: 429]
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100, patience=10): # [cite: 418]
    train_losses = [] # [cite: 418]
    test_losses = [] # [cite: 418]
    best_test_loss = float('inf') # [cite: 419]
    patience_counter = 0 # [cite: 419]

    for epoch in range(epochs): # [cite: 419]
        model.train() # [cite: 420]
        running_loss = 0.0 # [cite: 420]
        for inputs, targets in train_loader: # [cite: 420]
            optimizer.zero_grad() # [cite: 421]
            outputs = model(inputs) # [cite: 421]
            loss = criterion(outputs, targets) # [cite: 421]
            loss.backward() # [cite: 422]
            optimizer.step() # [cite: 422]
            running_loss += loss.item() * inputs.size(0) # [cite: 422, 423]
        epoch_loss = running_loss / len(train_loader.dataset) # [cite: 423]
        train_losses.append(epoch_loss) # [cite: 423]

        model.eval() # [cite: 424]
        test_loss = 0.0 # [cite: 424]
        with torch.no_grad(): # [cite: 424]
            for inputs, targets in test_loader: # [cite: 424]
                outputs = model(inputs) # [cite: 424]
                loss = criterion(outputs, targets) # [cite: 425]
                test_loss += loss.item() * inputs.size(0) # [cite: 425]
        test_loss = test_loss / len(test_loader.dataset) # [cite: 425]
        test_losses.append(test_loss) # [cite: 426]

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}') # [cite: 426]

        if test_loss < best_test_loss: # [cite: 426]
            best_test_loss = test_loss # [cite: 426]
            patience_counter = 0 # [cite: 426]
            # Save the best model
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else: # [cite: 426]
            patience_counter += 1 # [cite: 426]
            if patience_counter >= patience: # [cite: 426]
                print("Early stopping triggered") # [cite: 426]
                break # [cite: 426]
    
    loss_df = pd.DataFrame({'Epoch': list(range(1, len(train_losses) + 1)), 'Train Loss': train_losses, 'Test Loss': test_losses}) # [cite: 427]
    loss_df.to_csv("lstm_training_results.csv", index=False) # [cite: 428]
    return loss_df # Added return for completeness

# Listing 6: Evaluating the model [cite: 435]
def evaluate_model(model, test_loader): # [cite: 430]
    model.eval() # [cite: 431]
    predictions, actuals = [], [] # [cite: 431]
    with torch.no_grad(): # [cite: 432]
        for inputs, targets in test_loader: # [cite: 432]
            outputs = model(inputs) # [cite: 432]
            predictions.extend(outputs.cpu().numpy()) # Corrected: Added .cpu() before .numpy() [cite: 433]
            actuals.extend(targets.cpu().numpy()) # Corrected: Added .cpu() before .numpy() [cite: 433]
    return np.array(predictions), np.array(actuals) # [cite: 434]

# Listing 7: Main function [cite: 442]
def main(): # [cite: 436]
    set_seed(42) # [cite: 436]
    
    # Load the real data
    print("Loading data from CSV...")
    df = pd.read_csv('models/model_with_features.csv')
    print(f"Loaded data with shape: {df.shape}")
    
    # Preprocess the data with time-based split for 2021-2022 test data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Reshape data for LSTM
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = reshape_data_for_lstm(
        X_train, X_test, y_train, y_test
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get input size from data
    input_size = X_train_tensor.shape[2]  # Number of features
    hidden_size = 64  # Increased from 50
    num_layers = 2
    
    print(f"Model configuration: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
    
    # Initialize model, criterion, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training model...")
    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100, patience=15)
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred, y_true = evaluate_model(model, test_loader)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\nTest Set Evaluation Results:")
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2: {r2:.4f}')
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'Actual': y_true.flatten(),
        'Predicted': y_pred.flatten(),
        'Error': y_true.flatten() - y_pred.flatten()
    })
    results_df.to_csv("lstm_predictions.csv", index=False)
    print("Predictions saved to lstm_predictions.csv")

if __name__ == "__main__": # [cite: 441]
    main() # [cite: 441]