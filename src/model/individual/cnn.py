import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.model.individual.config import (
    CNN_BATCH_SIZE,
    CNN_CONV1_CHANNELS,
    CNN_CONV2_CHANNELS,
    CNN_CONV3_CHANNELS,
    CNN_DROPOUT,
    CNN_FC_SIZE_1,
    CNN_FC_SIZE_2,
    CNN_KERNEL_SIZE_1,
    CNN_KERNEL_SIZE_2,
    CNN_KERNEL_SIZE_3,
    CNN_LEARNING_RATE,
    CNN_LR_SCHEDULER_FACTOR,
    CNN_LR_SCHEDULER_PATIENCE,
    CNN_MAX_GRAD_NORM,
    CNN_WEIGHT_DECAY,
    EARLY_STOPPAGE,
    EPOCHS,
)
from src.model.individual.model_utils import EarlyStopping, create_data_loaders, evaluate_model, prepare_data

warnings.filterwarnings('ignore')


class CNN(nn.Module):
    def __init__(
        self,
        input_size,
        sequence_length,
        conv1_channels=CNN_CONV1_CHANNELS,
        conv2_channels=CNN_CONV2_CHANNELS,
        conv3_channels=CNN_CONV3_CHANNELS,
        kernel_size_1=CNN_KERNEL_SIZE_1,
        kernel_size_2=CNN_KERNEL_SIZE_2,
        kernel_size_3=CNN_KERNEL_SIZE_3,
        fc_size_1=CNN_FC_SIZE_1,
        fc_size_2=CNN_FC_SIZE_2,
        dropout=CNN_DROPOUT,
    ):
        super().__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.dropout = dropout

        # Enhanced 1D convolutional layers with proper regularization
        # First conv layer - captures short-term patterns
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=conv1_channels, kernel_size=kernel_size_1, padding=kernel_size_1 // 2
        )
        self.bn1 = nn.BatchNorm1d(conv1_channels)

        # Second conv layer - captures medium-term patterns
        self.conv2 = nn.Conv1d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=kernel_size_2,
            padding=kernel_size_2 // 2,
        )
        self.bn2 = nn.BatchNorm1d(conv2_channels)

        # Third conv layer - captures longer-term patterns
        self.conv3 = nn.Conv1d(
            in_channels=conv2_channels,
            out_channels=conv3_channels,
            kernel_size=kernel_size_3,
            padding=max(0, kernel_size_3 // 2),
        )
        self.bn3 = nn.BatchNorm1d(conv3_channels)

        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)

        # Adaptive pooling to handle variable sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Enhanced fully connected layers with proper regularization
        self.fc = nn.Sequential(
            nn.Linear(conv3_channels, fc_size_1),
            nn.BatchNorm1d(fc_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size_1, fc_size_2),
            nn.BatchNorm1d(fc_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size_2, 1),
        )

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Input shape: (batch_size, features, time_steps) for CNN
        # Note: data loader should handle the transpose for CNNs

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        # Apply pooling only if sequence is long enough
        if x.size(2) > 1:
            x = self.pool(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        # Apply pooling only if sequence is long enough
        if x.size(2) > 1:
            x = self.pool(x)

        # Third conv block (only if we have enough time steps)
        if x.size(2) >= self.conv3.kernel_size[0]:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.dropout_layer(x)

        # Adaptive pooling to get fixed size output
        x = self.adaptive_pool(x)  # Shape: (batch_size, channels, 1)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, channels)

        # Fully connected layers
        output = self.fc(x)

        return output


def train_cnn_model(
    X_train,
    y_train,
    X_test,
    y_test,
    input_size,
    sequence_length,
    epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPAGE,
    device=None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    # Create datasets and dataloaders
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, CNN_BATCH_SIZE, model_type='cnn')

    # Initialize model and move to device
    model = CNN(input_size=input_size, sequence_length=sequence_length).to(device)
    criterion = nn.MSELoss()

    # Enhanced optimizer with L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY)

    # Dynamic learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CNN_LR_SCHEDULER_FACTOR, patience=CNN_LR_SCHEDULER_PATIENCE, min_lr=1e-6
    )

    # Enhanced early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.0001, restore_best_weights=True)

    # Training tracking
    train_losses = []
    test_losses = []

    print('\nTraining CNN model...')
    print(
        f'Architecture: Conv1={CNN_CONV1_CHANNELS}, Conv2={CNN_CONV2_CHANNELS}, Conv3={CNN_CONV3_CHANNELS}, LR={CNN_LEARNING_RATE}'
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), CNN_MAX_GRAD_NORM)

            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Get current learning rate for display
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch [{epoch + 1:3d}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}'
        )

        # Step the learning rate scheduler
        scheduler.step(test_loss)

        # Early stopping check
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch + 1}, best loss: {early_stopping.best_loss:.4f}')
            break

    # Restore best model weights
    early_stopping.restore_best_weights_to_model(model)
    print(f'Training completed. Best validation loss: {early_stopping.best_loss:.4f}')

    return model, train_losses, test_losses


def main(
    sequence_length=None,
    auto_sequence_length=True,
    epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPAGE,
):
    print('CNN Stock Price Prediction')
    print('Loading dataset...')
    df = pd.read_csv('src/model/individual/dataset_1_full_features.csv')
    print(f'Dataset: {df.shape[0]} samples, {df.shape[1]} features')

    # Prepare data using common utility
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_cols = prepare_data(
        df,
        sequence_length=sequence_length,
        test_start_year=2021,
        auto_sequence_length=auto_sequence_length,
        model_type='cnn',
    )

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print('❌ ERROR: No sequences created! Check your data and sequence length.')
        return None, None

    # Get input size and sequence length
    input_size = len(feature_cols)
    sequence_length_used = X_train.shape[1]
    print(f'Input: {input_size} features, {sequence_length_used} time steps')

    # Train model with optimized parameters from config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_losses, test_losses = train_cnn_model(
        X_train,
        y_train,
        X_test,
        y_test,
        input_size,
        sequence_length_used,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
    )

    # Save loss plot
    os.makedirs('src/model/individual/output/cnn_results', exist_ok=True)
    from src.model.individual.model_utils import plot_and_save_loss, save_individual_model_results

    plot_and_save_loss(train_losses, test_losses, 'src/model/individual/output/cnn_results/cnn_loss.png', 'CNN')

    # Final evaluation
    print('\nFinal evaluation:')
    results = evaluate_model(model, X_test, y_test, scaler_y, model_type='cnn', model_name='CNN')
    # Attach training target history in price-space for consistent MASE
    y_train_log = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    results['y_train_original'] = np.expm1(y_train_log)

    # Add training losses to results for use by run_all_models.py
    results['train_losses'] = train_losses
    results['validation_losses'] = test_losses

    # Save comprehensive results including MAPE, MASE, and training data
    save_individual_model_results('CNN', results, train_losses, test_losses, 'src/model/individual/output')

    return model, results


if __name__ == '__main__':
    model, results = main()
