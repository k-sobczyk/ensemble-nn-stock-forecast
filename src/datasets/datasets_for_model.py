import os

import pandas as pd


def prepare_model_datasets():
    """Prepares 4 different dataset versions for ensemble neural network training
    focused on predicting share prices based on capital and asset structure changes.

    Dataset Versions:
    1. Full Feature Dataset - All available features
    2. Core Financial Structure Dataset - Main balance sheet items without time-lagged differences
    3. Change-Focused Dataset - Emphasizing quarterly changes in financial structure
    4. Ratio-Based Dataset - Financial ratios and derived efficiency metrics
    """  # noqa: D205
    df = pd.read_csv('data/data_with_features.csv')

    # Create datasets directory if it doesn't exist
    os.makedirs('data/datasets', exist_ok=True)

    # Common columns for all datasets
    base_columns = ['ticker', 'end_of_period', 'target_log']

    # ===== DATASET 1: FULL FEATURE DATASET =====
    full_feature_cols = [
        col for col in df.columns if col not in ['sector', 'file_name', 'target']
    ]  # Exclude original sector, file_name, and target
    dataset_1 = df[full_feature_cols].copy()

    # One-hot encode sector_category for full dataset
    dataset_1 = pd.get_dummies(dataset_1, columns=['sector_category'], prefix='sector')

    dataset_1.to_csv('data/datasets/dataset_1_full_features.csv', index=False)
    print(f'Dataset 1 (Full Features) shape: {dataset_1.shape}')

    # ===== DATASET 2: CORE FINANCIAL STRUCTURE DATASET =====
    core_financial_cols = base_columns + [
        # Core Asset Structure
        'total_assets',
        'non_current_assets',
        'current_assets',
        'property_plant_equipment',
        'intangible_assets',
        'inventories',
        'trade_receivables',
        'cash_and_cash_equivalents',
        # Core Capital Structure
        'equity_shareholders_of_the_parent',
        'share_capital',
        'retained_earning_accumulated_losses',
        'total_shares',
        # Core Liability Structure
        'non_current_liabilities',
        'current_liabilities',
        'non_current_loans_and_borrowings',
        'financial_liabilities_loans_borrowings',
        'total_liabilities',
        # Sector information
        'sector_category',
    ]

    dataset_2 = df[core_financial_cols].copy()
    dataset_2 = pd.get_dummies(dataset_2, columns=['sector_category'], prefix='sector')

    dataset_2.to_csv('data/datasets/dataset_2_core_financial_structure.csv', index=False)
    print(f'Dataset 2 (Core Financial Structure) shape: {dataset_2.shape}')

    # ===== DATASET 3: CHANGE-FOCUSED DATASET =====
    # Focus on quarterly changes (differences) in financial structure
    change_cols = base_columns + ['sector_category']

    # Add all difference columns (quarterly changes)
    diff_cols = [col for col in df.columns if '_diff_' in col]
    change_cols.extend(diff_cols)

    # Add some core current values for context
    context_cols = ['total_assets', 'total_liabilities', 'equity_shareholders_of_the_parent']
    change_cols.extend(context_cols)

    dataset_3 = df[change_cols].copy()
    dataset_3 = pd.get_dummies(dataset_3, columns=['sector_category'], prefix='sector')

    dataset_3.to_csv('data/datasets/dataset_3_change_focused.csv', index=False)
    print(f'Dataset 3 (Change-Focused) shape: {dataset_3.shape}')

    return {'full_features': dataset_1, 'core_financial': dataset_2, 'change_focused': dataset_3}


if __name__ == '__main__':
    datasets = prepare_model_datasets()
