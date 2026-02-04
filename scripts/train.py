#!/usr/bin/env python3
"""
Unified training script for all model types.

Usage:
    python train.py --model_type logistic --rationales diversity indep tenure
    python train.py --model_type mc_dropout --rationales diversity indep tenure busyness
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    DATA_CONFIG, MODELS_DIR, CORE_RATIONALES, MODEL_CONFIGS
)
from src.data.data_manager import DataManager
from src.models.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train voting rationale prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest", "gradient_boosting", "mc_dropout"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--rationales",
        nargs="+",
        default=CORE_RATIONALES,
        help="Rationales to predict"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_CONFIG["data_path"],
        help="Path to data file"
    )
    parser.add_argument(
        "--min_meetings_rat",
        type=int,
        default=DATA_CONFIG["min_meetings_rat"],
        help="Minimum N_Meetings_Rat filter"
    )
    parser.add_argument(
        "--min_dissent",
        type=int,
        default=DATA_CONFIG["min_dissent"],
        help="Minimum N_dissent filter"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=DATA_CONFIG["test_size"],
        help="Test set proportion"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=DATA_CONFIG["random_seed"],
        help="Random seed"
    )
    
    # Training configuration
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save models (default: models/{model_type})"
    )
    parser.add_argument(
        "--no_calibrate",
        action="store_true",
        help="Disable probability calibration (supervised models only)"
    )
    
    # Model-specific hyperparameters
    parser.add_argument("--C", type=float, help="Regularization (logistic)")
    parser.add_argument("--max_depth", type=int, help="Max depth (tree models)")
    parser.add_argument("--n_estimators", type=int, help="Number of estimators (tree models)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Training epochs (MC Dropout)")
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate (MC Dropout)")
    
    # Feature engineering configuration
    parser.add_argument(
        "--use_all_features",
        action="store_true",
        help="Use all available columns as features (subject to missing threshold)"
    )
    parser.add_argument(
        "--drop_high_missing",
        type=float,
        default=1.0,
        help="Drop features with missing rate > threshold (0.0-1.0, default=1.0 means keep all)"
    )
    parser.add_argument(
        "--exclude_cols",
        nargs="+",
        default=None,
        help="Columns to exclude from features (in addition to defaults)"
    )
    
    args = parser.parse_args()
    
    # Determine save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = MODELS_DIR / args.model_type
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("="*80)
    print("VOTING RATIONALE MODEL TRAINING")
    print("="*80)
    print(f"Model type: {args.model_type}")
    print(f"Rationales: {args.rationales}")
    print(f"Data: {args.data_path}")
    print(f"Save dir: {save_dir}")
    print("="*80 + "\n")
    
    # Load and prepare data
    print("Loading data...")
    data_manager = DataManager()
    data_manager.load_data(args.data_path)
    data_manager.apply_filters(
        min_meetings_rat=args.min_meetings_rat,
        min_dissent=args.min_dissent
    )
    
    train_df, val_df = data_manager.split_data(
        test_size=args.test_size,
        random_seed=args.random_seed,
        rationales=args.rationales,
    )
    
    # Build model configuration
    config = MODEL_CONFIGS.get(args.model_type, {}).copy()
    config['calibrate'] = not args.no_calibrate
    config['random_seed'] = args.random_seed
    
    # Add feature configuration
    config['use_all_features'] = args.use_all_features
    config['drop_high_missing'] = args.drop_high_missing
    if args.exclude_cols:
        config['exclude_cols'] = args.exclude_cols
    
    # Override with command-line arguments
    custom_params = {}
    if args.C is not None:
        custom_params['C'] = args.C
    if args.max_depth is not None:
        custom_params['max_depth'] = args.max_depth
    if args.n_estimators is not None:
        custom_params['n_estimators'] = args.n_estimators
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.dropout_rate is not None:
        config['dropout_rate'] = args.dropout_rate
    
    if custom_params:
        config['custom_params'] = custom_params
    
    # Train models
    trainer = ModelTrainer(
        model_type=args.model_type,
        rationales=args.rationales,
        config=config,
    )
    
    models = trainer.train_all(
        train_df=train_df,
        rationales=args.rationales,
        data_manager=data_manager,
        val_df=val_df if args.model_type == "mc_dropout" else None,
        save_dir=save_dir,
    )
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    main()