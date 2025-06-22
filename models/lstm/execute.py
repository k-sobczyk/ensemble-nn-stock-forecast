import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from metrics.metrics import print_metrics
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data_processing import create_sequences_by_company, load_and_split_data, prepare_base_data
from .hyperparameter_search import run_hyperparameter_search
from .model import ModelLSTM, StockDatasetLSTM
from .training import evaluate_model, train_model


def run_lstm_model(
    data_file,
    date_column='end_of_period',
    target_column='target',
    ticker_column='ticker',
    test_years=[2021, 2022],
    min_samples_per_company=4,
    n_trials=1,
    optuna_timeout=3600,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
        num_workers = 2
    else:
        num_workers = 0

    print(f'Using {num_workers} dataloader workers.')

    # Load and prepare data
    train_df, test_df, test_ratio = load_and_split_data(data_file, date_column, test_years)

    # Check the number of data points per company
    if ticker_column in train_df.columns:
        print('\n--- Samples per company (training set) ---')
        company_counts = train_df[ticker_column].value_counts()
        print(
            f'Min samples: {company_counts.min()}, Max samples: {company_counts.max()}, Median: {company_counts.median()}'
        )
        print(
            f'Companies with less than {min_samples_per_company} samples: {sum(company_counts < min_samples_per_company)}'
        )

        # Display sample sizes for a few companies
        print('\nSample counts for first 10 companies:')
        for company, count in company_counts.head(10).items():
            print(f'{company}: {count} samples')

    (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        y_train_raw,
        scaler_X,
        scaler_y,
        num_features,
        train_ticker_encoded,
        test_ticker_encoded,
        num_companies,
    ) = prepare_base_data(train_df, test_df, target_column, ticker_column)

    # Run hyperparameter search
    print(f'\n--- Starting Optuna Hyperparameter Optimization ({n_trials} trials) ---')
    best_params = run_hyperparameter_search(
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        train_ticker_encoded,
        test_ticker_encoded,
        num_features,
        num_companies,
        device,
        n_trials=n_trials,
        timeout=optuna_timeout,
    )

    if best_params is None:
        print('Hyperparameter optimization failed. Exiting.')
        return

    # Retrain model with best parameters
    print('\n--- Retraining model with best parameters ---')

    best_sequence_length = best_params['sequence_length']
    print(f'Using best sequence length: {best_sequence_length}')

    min_samples_per_company = best_params.get('min_samples_per_company', min_samples_per_company)

    # Create sequences for final model with best parameters
    X_train_final, y_train_final, train_company_final = create_sequences_by_company(
        X_train_scaled,
        y_train_scaled,
        train_ticker_encoded,
        best_sequence_length,
        min_samples_per_company=min_samples_per_company,
    )
    X_test_final, y_test_final, test_company_final = create_sequences_by_company(
        X_test_scaled,
        y_test_scaled,
        test_ticker_encoded,
        best_sequence_length,
        min_samples_per_company=min_samples_per_company,
    )

    # Print sequence shape information
    print(f'\nCreated {X_train_final.shape[0]} training sequences and {X_test_final.shape[0]} test sequences')
    print(f'Sequence shape: {X_train_final.shape}')

    if X_train_final.shape[0] == 0 or X_test_final.shape[0] == 0:
        print('Error: Cannot retrain model as the best sequence length resulted in empty data.')
        return

    # Create datasets for final model
    final_train_dataset = StockDatasetLSTM(X_train_final, y_train_final, train_company_final)
    final_test_dataset = StockDatasetLSTM(X_test_final, y_test_final, test_company_final)

    pin_memory = True if device.type == 'cuda' else False
    final_train_loader = DataLoader(
        final_train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    final_test_loader = DataLoader(
        final_test_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Initialize final model with best parameters
    embedding_dim = best_params.get('embedding_dim', 8)
    final_model = ModelLSTM(
        num_features=num_features,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate'],
        use_batch_norm=best_params['use_batch_norm'],
        num_companies=num_companies,
        embedding_dim=embedding_dim,
    )
    criterion = nn.MSELoss()

    optimizer_name = best_params['optimizer']
    print(f'Using optimizer: {optimizer_name}')
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(
            final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']
        )
    else:
        optimizer = optim.RMSprop(
            final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']
        )

    final_epochs = 200
    final_patience = 25
    final_lr_patience = 10
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=final_lr_patience, verbose=True, min_lr=1e-8
    )

    loss_df, final_trained_model, _ = train_model(
        final_model,
        final_train_loader,
        final_test_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=final_epochs,
        patience=final_patience,
        device=device,
        verbose=True,
    )

    # Evaluate model
    print('\n--- Evaluating final best model on Test Set ---')
    y_pred, y_true, company_ids = evaluate_model(final_trained_model, final_test_loader, scaler_y, device=device)

    if len(y_true) > 0 and len(y_pred) > 0:
        print_metrics(y_true, y_pred, y_train_raw)

        results_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})

        # Add company information if available
        if company_ids is not None and len(company_ids) == len(y_true):
            ticker_encoder = LabelEncoder()
            ticker_encoder.fit(train_df[ticker_column])
            results_df['Company'] = ticker_encoder.inverse_transform(company_ids)

            # Calculate metrics per company
            print('\n--- Metrics by Company ---')
            for company in results_df['Company'].unique():
                company_mask = results_df['Company'] == company
                company_true = results_df.loc[company_mask, 'Actual'].values
                company_pred = results_df.loc[company_mask, 'Predicted'].values

                if len(company_true) > 5:  # Reduced threshold for quarterly data
                    print(f'\nCompany: {company} (samples: {len(company_true)})')
                    print_metrics(company_true, company_pred, y_train_raw)

        # Save results
        results_df.to_csv('lstm_best_model_predictions.csv', index=False, float_format='%.4f')
        print('\nBest LSTM model predictions saved to lstm_best_model_predictions.csv')

        loss_df.to_csv('lstm_best_model_training_history.csv', index=False, float_format='%.6f')
        print('Best LSTM model training history saved to lstm_best_model_training_history.csv')

        model_save_path = 'lstm_best_model.pth'
        torch.save(final_trained_model.state_dict(), model_save_path)
        print(f'Best LSTM model state dict saved to {model_save_path}')

        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_csv('lstm_best_model_hyperparameters.csv', index=False)
        print('Best LSTM model hyperparameters saved to lstm_best_model_hyperparameters.csv')
    else:
        print('Evaluation failed: No predictions generated.')

    return final_trained_model, y_pred, y_true, results_df if 'results_df' in locals() else None
