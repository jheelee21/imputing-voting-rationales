#!/usr/bin/env python3
"""
Semi-Supervised Learning Training Script.

Trains models using both labeled and unlabeled dissent observations.
This is particularly effective when you have many dissent rows with missing rationales.

Usage:
    # Pseudo-labeling with logistic regression
    python train_semi_supervised.py --method pseudo_labeling --base_model_type logistic --rationales diversity indep tenure
    
    # Co-training with random forest
    python train_semi_supervised.py --method co_training --base_model_type random_forest --rationales diversity indep
    
    # With custom confidence threshold
    python train_semi_supervised.py --method pseudo_labeling --confidence_threshold 0.95 --max_iterations 10
    
    # With boosting models (requires calibrated_boosting.py)
    python train_semi_supervised.py --method pseudo_labeling --base_model_type catboost --rationales diversity tenure
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    DATA_CONFIG, MODELS_DIR, CORE_RATIONALES, FEATURE_CONFIG,
)
from src.data.data_manager import DataManager
from src.pipeline.semi_supervised_trainer import (
    SemiSupervisedTrainer,
    prepare_semi_supervised_data
)


def main():
    parser = argparse.ArgumentParser(
        description="Train semi-supervised models for voting rationale prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--method",
        type=str,
        default="pseudo_labeling",
        choices=["pseudo_labeling", "co_training"],
        help="Semi-supervised method"
    )
    parser.add_argument(
        "--base_model_type",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest", "gradient_boosting", "catboost", "lightgbm", "xgboost"],
        help="Base supervised model type"
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
    
    # Semi-supervised configuration
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for pseudo-labels (0.0-1.0)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum self-training iterations"
    )
    parser.add_argument(
        "--min_pseudo_labels",
        type=int,
        default=10,
        help="Minimum pseudo-labels to add per iteration"
    )
    
    # Co-training specific
    parser.add_argument(
        "--agreement_threshold",
        type=float,
        default=0.1,
        help="Agreement threshold for co-training (max prob difference)"
    )
    parser.add_argument(
        "--feature_split_ratio",
        type=float,
        default=0.5,
        help="Feature split ratio for co-training (0.0-1.0)"
    )
    
    # Training configuration
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save models (default: models/semi_supervised_{method})"
    )
    parser.add_argument(
        "--no_calibrate",
        action="store_true",
        help="Disable probability calibration"
    )
    
    # Feature configuration
    parser.add_argument(
        "--required_only",
        action="store_true",
        help="Use only required features per rationale"
    )
    parser.add_argument(
        "--drop_high_missing",
        type=float,
        default=None,
        help="Drop features with missing rate > threshold"
    )
    parser.add_argument(
        "--exclude_cols",
        nargs="+",
        default=None,
        help="Columns to exclude from features"
    )
    
    # Base model hyperparameters
    parser.add_argument("--C", type=float, help="Regularization (logistic)")
    parser.add_argument("--max_depth", type=int, help="Max depth (tree models)")
    parser.add_argument("--n_estimators", type=int, help="Number of estimators (tree models)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (boosting)")
    
    args = parser.parse_args()
    
    # Determine save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = MODELS_DIR / f"semi_supervised_{args.method}"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("="*80)
    print("SEMI-SUPERVISED LEARNING FOR VOTING RATIONALE PREDICTION")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Base model: {args.base_model_type}")
    print(f"Rationales: {args.rationales}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Max iterations: {args.max_iterations}")
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
    
    # Prepare semi-supervised data split
    train_df, unlabeled_df, test_df = prepare_semi_supervised_data(
        data_manager=data_manager,
        rationales=args.rationales,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    
    # Configure trainer
    config = {
        'base_model_type': args.base_model_type,
        'confidence_threshold': args.confidence_threshold,
        'max_iterations': args.max_iterations,
        'min_pseudo_labels': args.min_pseudo_labels,
        'calibrate': not args.no_calibrate,
        'random_seed': args.random_seed,
        'use_all_features': not args.required_only,
        'drop_high_missing': args.drop_high_missing if args.drop_high_missing is not None else FEATURE_CONFIG.get('drop_high_missing', 0.5),
        'exclude_cols': args.exclude_cols if args.exclude_cols is not None else FEATURE_CONFIG.get('exclude_cols'),
        'missing_strategy': FEATURE_CONFIG.get('missing_strategy', 'median'),
    }
    
    # Add co-training specific params
    if args.method == "co_training":
        config['agreement_threshold'] = args.agreement_threshold
        config['feature_split_ratio'] = args.feature_split_ratio
    
    # Add base model hyperparameters
    custom_params = {}
    if args.C is not None:
        custom_params['C'] = args.C
    if args.max_depth is not None:
        custom_params['max_depth'] = args.max_depth
    if args.n_estimators is not None:
        custom_params['n_estimators'] = args.n_estimators
    if args.learning_rate is not None:
        custom_params['learning_rate'] = args.learning_rate
    
    if custom_params:
        config['custom_params'] = custom_params
    
    # Create trainer
    trainer = SemiSupervisedTrainer(
        method=args.method,
        rationales=args.rationales,
        config=config,
    )
    
    # Train models
    models = trainer.train_all(
        train_df=train_df,
        unlabeled_df=unlabeled_df,
        rationales=args.rationales,
        data_manager=data_manager,
        val_df=test_df,  # Use test as validation for monitoring
        save_dir=save_dir,
    )
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Models saved to: {save_dir}")
    print(f"\nTo evaluate, run:")
    print(f"  python scripts/evaluate.py --model_dir models/semi_supervised_{args.method}")
    print(f"\nTo generate predictions, run:")
    print(f"  python scripts/predict.py --model_dir models/semi_supervised_{args.method}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()