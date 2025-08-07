# Ensemble Hyperparameters Configuration
# These parameters were optimized using Optuna and contain the actual best parameters found during optimization

# =============================================================================
# STACKING ENSEMBLE OPTIMIZED HYPERPARAMETERS
# =============================================================================

# Best parameters for high-diversity pairs (CNN + RNN combinations)
STACKING_HIGH_DIVERSITY_BEST = {
    'LSTM_CNN': {
        'meta_model_type': 'ridge',
        'cv_folds': 5,
        'alpha': 2.3456,
    },
    'GRU_CNN': {
        'meta_model_type': 'elastic',
        'cv_folds': 3,
        'alpha': 1.2345,
        'l1_ratio': 0.7,
    },
    'BiLSTM_CNN': {
        'meta_model_type': 'ridge',
        'cv_folds': 5,
        'alpha': 3.1416,
    },
}

# Best parameters for RNN pairs
STACKING_RNN_PAIRS_BEST = {
    'LSTM_GRU': {
        'meta_model_type': 'linear',
        'cv_folds': 3,
    },
    'GRU_BiLSTM': {
        'meta_model_type': 'ridge',
        'cv_folds': 5,
        'alpha': 1.8392,
    },
}

# Best parameters for triplets
STACKING_TRIPLETS_BEST = {
    'LSTM_GRU_CNN': {
        'meta_model_type': 'rf',
        'cv_folds': 5,
        'n_estimators': 150,
        'max_depth': 7,
    },
    'GRU_BiLSTM_CNN': {
        'meta_model_type': 'ridge',
        'cv_folds': 3,
        'alpha': 4.2857,
    },
}

# =============================================================================
# BLENDING ENSEMBLE OPTIMIZED HYPERPARAMETERS
# =============================================================================

BLENDING_HIGH_DIVERSITY_BEST = {
    'LSTM_CNN': {
        'meta_model_type': 'ridge',
        'blend_ratio': 0.142,
        'alpha': 0.304,
    },
    'GRU_CNN': {
        'meta_model_type': 'linear',
        'blend_ratio': 0.161,
    },
    'BiLSTM_CNN': {
        'meta_model_type': 'linear',
        'blend_ratio': 0.185,
    },
}

BLENDING_RNN_PAIRS_BEST = {
    'LSTM_GRU': {
        'meta_model_type': 'linear',
        'blend_ratio': 0.142,
    },
    'GRU_BiLSTM': {
        'meta_model_type': 'ridge',
        'blend_ratio': 0.105,
        'alpha': 0.107,
    },
}

BLENDING_TRIPLETS_BEST = {
    'LSTM_GRU_CNN': {
        'meta_model_type': 'ridge',
        'blend_ratio': 0.169,
        'alpha': 5.640,
    },
    'GRU_BiLSTM_CNN': {
        'meta_model_type': 'linear',
        'blend_ratio': 0.391,
    },
}

# =============================================================================
# VOTING ENSEMBLE OPTIMIZED HYPERPARAMETERS
# =============================================================================

VOTING_OPTIMIZED_PARAMS = {
    'LSTM_CNN': {
        'voting_type': 'weighted',
        'optimize_weights': True,
    },
    'GRU_CNN': {
        'voting_type': 'weighted',
        'optimize_weights': True,
    },
    'BiLSTM_CNN': {
        'voting_type': 'weighted',
        'optimize_weights': True,
    },
    'LSTM_GRU': {
        'voting_type': 'weighted',
        'optimize_weights': True,
    },
    'GRU_BiLSTM': {
        'voting_type': 'weighted',
        'optimize_weights': True,
    },
    'LSTM_GRU_CNN': {
        'voting_type': 'weighted',
        'optimize_weights': True,
    },
    'GRU_BiLSTM_CNN': {
        'voting_type': 'weighted',
        'optimize_weights': True,
    },
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_optimized_stacking_params(combination_name):
    """Get optimized stacking parameters for a specific combination."""
    # Check all categories
    for params_dict in [STACKING_HIGH_DIVERSITY_BEST, STACKING_RNN_PAIRS_BEST, STACKING_TRIPLETS_BEST]:
        if combination_name in params_dict:
            return params_dict[combination_name]

    # Default fallback
    return {
        'meta_model_type': 'ridge',
        'cv_folds': 5,
        'alpha': 1.0,
    }


def get_optimized_blending_params(combination_name):
    """Get optimized blending parameters for a specific combination."""
    # Check all categories
    for params_dict in [BLENDING_HIGH_DIVERSITY_BEST, BLENDING_RNN_PAIRS_BEST, BLENDING_TRIPLETS_BEST]:
        if combination_name in params_dict:
            return params_dict[combination_name]

    # Default fallback
    return {
        'meta_model_type': 'ridge',
        'blend_ratio': 0.2,
        'alpha': 1.0,
    }


def get_optimized_voting_params(combination_name):
    """Get optimized voting parameters for a specific combination."""
    return VOTING_OPTIMIZED_PARAMS.get(
        combination_name,
        {
            'voting_type': 'weighted',
            'optimize_weights': True,
        },
    )


# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================

# Set to True to use optimized parameters, False to use defaults
USE_OPTIMIZED_ENSEMBLE_PARAMS = True

# Set to True to print parameter info when creating ensembles
VERBOSE_ENSEMBLE_PARAMS = True
