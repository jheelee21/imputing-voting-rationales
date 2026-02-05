"""
Configuration file for voting rationales prediction project.
Centralized settings for models, data, and evaluation.

Modeling scope (strict):
- We model the reason for dissent conditional on dissent having occurred (ind_dissent=1).
- Training uses only dissent rows with at least one labeled rationale.
- Prediction (imputation) targets dissent rows where all rationales are missing.
"""

from pathlib import Path
from typing import List, Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data configuration
DATA_CONFIG = {
    "data_path": str(DATA_DIR / "Imputing Rationales.csv"),
    "min_meetings_rat": 1,  # Filter: minimum meetings with rationales
    "min_dissent": 5,  # Filter: minimum dissenting votes
    "test_size": 0.2,
    "random_seed": 21,
}

# Feature engineering configuration
# Default: use all variables with missing rate below threshold (drop_high_missing)
FEATURE_CONFIG = {
    "use_all_features": True,  # Default: use all available columns (subject to missing threshold)
    "drop_high_missing": 0.5,   # Drop features with missing rate > this (0.0-1.0)
    "exclude_cols": [          # Columns to always exclude from features
        "meeting_id",          # Don't use as feature (per PDF)
        "ind_dissent",         # Target-related, not a feature
    ],
}

# Rationales configuration
CORE_RATIONALES = ["diversity", "indep", "tenure", "busyness", "combined_ceo_chairman"]

ALL_RATIONALES = [
    "diversity", "boardstructure", "indep", "tenure", "governance",
    "combined_ceo_chairman", "esg_csr", "responsiveness", "attendance",
    "compensation", "busyness", "norat_misc",
]

# Required features for each rationale (from PDF instructions)
REQUIRED_FEATURES = {
    "diversity": ["Per_female", "D_Per_female"],
    "tenure": ["AvTenure", "D_AvTenure"],
    "indep": ["Per_Independent", "D_Independent"],
}

# General features (must include)
GENERAL_FEATURES = ["frac_vote_against"]
CATEGORICAL_IDS = ["investor_id", "pid", "ProxySeason"]

# Model configurations
MODEL_CONFIGS = {
    "logistic": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "C": 1.0,
        "random_state": 21,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": 21,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": 21,
    },
    "mc_dropout": {
        "hidden_dims": [64, 32],
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "num_epochs": 100,
        "batch_size": 256,
        "num_samples": 50,
        "weight_decay": 1e-4,
        "random_state": 21,
    },
}

# Evaluation configuration
EVAL_CONFIG = {
    "prediction_threshold": 0.5,
    "calibrate_models": True,
    "save_plots": True,
    "mc_dropout_samples": 50,
}

# ID columns for outputs
ID_COLUMNS = ["investor_id", "pid", "ProxySeason", "meeting_id"]