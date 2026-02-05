#!/usr/bin/env python3
"""
Extended training script for all probabilistic models.

Usage Examples:
    # Bayesian Neural Network
    python train_extended.py --model_type bnn --rationales diversity indep tenure
    
    # Calibrated Boosting Models
    python train_extended.py --model_type catboost --rationales diversity indep tenure
    python train_extended.py --model_type lightgbm --rationales diversity indep tenure
    python train_extended.py --model_type xgboost --rationales diversity indep tenure
    
    # Gaussian Process Models
    python train_extended.py --model_type sparse_gp --rationales diversity indep
    python train_extended.py --model_type deep_kernel_gp --rationales diversity indep
    
    # Original Models
    python train_extended.py --model_type mc_dropout --rationales diversity indep tenure
    python train_extended.py --model_type logistic --rationales diversity indep tenure
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    DATA_CONFIG, MODELS_DIR, CORE_RATIONALES
)
from src.data.data_manager import DataManager
from src.pipeline.extended_trainer import ExtendedModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train probabilistic voting rationale prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="catboost",
        choices=[
            # Original models
            "logistic", "random_forest", "gradient_boosting", "mc_dropout",
            # New probabilistic models
            "bnn", "catboost", "lightgbm", "xgboost",
            "sparse_gp", "deep_kernel_gp"
        ],
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
        help="Disable probability calibration (boosting models only)"
    )
    
    # Feature engineering configuration
    parser.add_argument(
        "--use_all_features",
        action="store_true",
        help="Use all available columns as features"
    )
    parser.add_argument(
        "--drop_high_missing",
        type=float,
        default=0.5,
        help="Drop features with missing rate > threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--exclude_cols",
        nargs="+",
        default=None,
        help="Columns to exclude from features"
    )
    
    # Model-specific hyperparameters
    # Boosting
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of estimators (boosting)")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth (boosting)")
    parser.add_argument("--learning_rate", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--calibration_method", type=str, default="isotonic", 
                       choices=["isotonic", "sigmoid"], help="Calibration method (boosting)")
    
    # Neural networks
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs (NN models)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (NN models)")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (MC Dropout)")
    parser.add_argument("--prior_scale", type=float, default=1.0, help="Prior scale (BNN)")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 32], 
                       help="Hidden layer dimensions (NN models)")
    
    # Gaussian Process
    parser.add_argument("--num_inducing", type=int, default=500, help="Number of inducing points (GP)")
    parser.add_argument("--kernel_type", type=str, default="rbf", 
                       choices=["rbf", "matern"], help="Kernel type (GP)")
    parser.add_argument("--no_ard", action="store_true", help="Disable ARD (GP)")
    
    args = parser.parse_args()
    
    # Determine save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = MODELS_DIR / args.model_type
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("="*80)
    print("PROBABILISTIC VOTING RATIONALE MODEL TRAINING")
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
    config = {
        'random_seed': args.random_seed,
        'use_all_features': args.use_all_features,
        'drop_high_missing': args.drop_high_missing,
    }
    
    if args.exclude_cols:
        config['exclude_cols'] = args.exclude_cols
    
    # Model-specific configs
    if args.model_type in ['catboost', 'lightgbm', 'xgboost']:
        config['calibrate'] = not args.no_calibrate
        config['calibration_method'] = args.calibration_method
        config['custom_params'] = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
        }
    
    elif args.model_type in ['mc_dropout', 'bnn']:
        config['hidden_dims'] = args.hidden_dims
        config['learning_rate'] = args.learning_rate
        config['num_epochs'] = args.num_epochs
        config['batch_size'] = args.batch_size
        
        if args.model_type == 'mc_dropout':
            config['dropout_rate'] = args.dropout_rate
        else:  # bnn
            config['prior_scale'] = args.prior_scale
    
    elif args.model_type in ['sparse_gp', 'deep_kernel_gp']:
        config['kernel_type'] = args.kernel_type
        config['num_inducing'] = args.num_inducing
        config['learning_rate'] = args.learning_rate
        config['num_epochs'] = args.num_epochs
        config['batch_size'] = args.batch_size
        config['hidden_dims'] = args.hidden_dims
        config['use_ard'] = not args.no_ard
    
    # Train models
    trainer = ExtendedModelTrainer(
        model_type=args.model_type,
        rationales=args.rationales,
        config=config,
    )
    
    models = trainer.train_all(
        train_df=train_df,
        rationales=args.rationales,
        data_manager=data_manager,
        val_df=val_df,
        save_dir=save_dir,
    )
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Models saved to: models/{args.model_type}")
    print(f"\nTo evaluate, run:")
    print(f"python scripts/evaluate.py --model_dir models/{args.model_type}")


if __name__ == "__main__":
    main()