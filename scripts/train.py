#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    DATA_CONFIG,
    MODELS_DIR,
    CORE_RATIONALES,
    MODEL_CONFIGS,
)
from src.data.data_manager import DataManager
from src.pipeline.trainer import ModelTrainer
from src.pipeline.workflow import (
    WorkflowConfig,
    build_feature_config,
    load_and_filter_data,
    resolve_save_dir,
    split_labeled_data,
)

try:
    from src.pipeline.extended_trainer import ExtendedModelTrainer

    EXTENDED_AVAILABLE = True
except ImportError:
    EXTENDED_AVAILABLE = False

ORIGINAL_MODEL_TYPES = ["logistic", "random_forest", "gradient_boosting", "mc_dropout"]
EXTENDED_MODEL_TYPES = [
    "bnn",
    "catboost",
    "lightgbm",
    "xgboost",
    "sparse_gp",
    "deep_kernel_gp",
    "pca",
    "hierarchical",
]
ALL_MODEL_TYPES = ORIGINAL_MODEL_TYPES + EXTENDED_MODEL_TYPES


def main():
    parser = argparse.ArgumentParser(
        description="Train voting rationale prediction models (original + extended + PCA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=ALL_MODEL_TYPES,
        help="Type of model to train",
    )
    parser.add_argument(
        "--rationales", nargs="+", default=CORE_RATIONALES, help="Rationales to predict"
    )

    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_CONFIG["data_path"],
        help="Path to data file",
    )
    parser.add_argument(
        "--min_meetings_rat",
        type=int,
        default=DATA_CONFIG["min_meetings_rat"],
        help="Minimum N_Meetings_Rat filter",
    )
    parser.add_argument(
        "--min_dissent",
        type=int,
        default=DATA_CONFIG["min_dissent"],
        help="Minimum N_dissent filter",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=DATA_CONFIG["test_size"],
        help="Test set proportion",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=DATA_CONFIG["random_seed"],
        help="Random seed",
    )

    # Training configuration
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save models (default: models/{model_type})",
    )
    parser.add_argument(
        "--no_calibrate",
        action="store_true",
        help="Disable probability calibration (supervised/boosting models only)",
    )

    # Original model hyperparameters
    parser.add_argument("--C", type=float, help="Regularization (logistic)")
    parser.add_argument("--max_depth", type=int, help="Max depth (tree models)")
    parser.add_argument(
        "--n_estimators", type=int, help="Number of estimators (tree models)"
    )
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Training epochs (NN models)")
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate (MC Dropout)")

    # Feature engineering configuration (default: all variables with missing rate below threshold)
    parser.add_argument(
        "--required_only",
        action="store_true",
        help="Use only required features per rationale (default: use all variables with missing rate < threshold)",
    )
    parser.add_argument(
        "--drop_high_missing",
        type=float,
        default=None,
        help="Drop features with missing rate > threshold (0.0-1.0). Default from config (0.5)",
    )
    parser.add_argument(
        "--exclude_cols",
        nargs="+",
        default=None,
        help="Columns to exclude from features. Default from config (meeting_id, ind_dissent)",
    )

    # Extended model hyperparameters
    parser.add_argument(
        "--calibration_method",
        type=str,
        default="isotonic",
        choices=["isotonic", "sigmoid"],
        help="Calibration method (boosting)",
    )
    parser.add_argument(
        "--prior_scale", type=float, default=1.0, help="Prior scale (BNN)"
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[64, 32],
        help="Hidden layer dimensions (NN models)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size (NN models)"
    )
    parser.add_argument(
        "--num_inducing", type=int, default=500, help="Number of inducing points (GP)"
    )
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="rbf",
        choices=["rbf", "matern"],
        help="Kernel type (GP)",
    )
    parser.add_argument("--no_ard", action="store_true", help="Disable ARD (GP)")

    # PCA-specific hyperparameters
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Number of PCA components (None = auto-select based on variance_threshold)",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.95,
        help="Variance threshold for PCA component selection (0.0-1.0)",
    )
    parser.add_argument(
        "--pca_classifier",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest", "gradient_boosting"],
        help="Classifier to use on PCA features",
    )
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="Apply whitening to PCA components",
    )

    args = parser.parse_args()

    if args.model_type in EXTENDED_MODEL_TYPES and not EXTENDED_AVAILABLE:
        sys.exit(
            f"Extended model '{args.model_type}' requires ExtendedModelTrainer. Install optional dependencies."
        )

    workflow = WorkflowConfig(
        data_path=args.data_path,
        min_meetings_rat=args.min_meetings_rat,
        min_dissent=args.min_dissent,
        random_seed=args.random_seed,
        test_size=args.test_size,
    )

    # Determine save directory
    save_dir = resolve_save_dir(MODELS_DIR, args.model_type, args.save_dir)

    # Print configuration
    print("=" * 80)
    print("VOTING RATIONALE MODEL TRAINING")
    print("=" * 80)
    print(f"Model type: {args.model_type}")
    print(f"Rationales: {args.rationales}")
    print(f"Data: {args.data_path}")
    print(f"Save dir: {save_dir}")

    # Print PCA-specific config if applicable
    if args.model_type == "pca":
        print(f"\nPCA Configuration:")
        print(
            f"  Components: {args.n_components or f'auto ({args.variance_threshold:.0%} variance)'}"
        )
        print(f"  Classifier: {args.pca_classifier}")
        print(f"  Whitening: {args.whiten}")

    print("=" * 80 + "\n")

    # Load and prepare data
    print("Loading data...")
    data_manager = DataManager()
    load_and_filter_data(data_manager, workflow)

    train_df, val_df = split_labeled_data(
        data_manager=data_manager,
        rationales=args.rationales,
        workflow=workflow,
    )

    use_extended = args.model_type in EXTENDED_MODEL_TYPES

    if use_extended:
        # Config for extended trainer (default: all variables with missing rate < threshold)
        config = {
            "random_seed": args.random_seed,
            **build_feature_config(
                required_only=args.required_only,
                drop_high_missing=args.drop_high_missing,
                exclude_cols=args.exclude_cols,
            ),
        }

        # PCA-specific configuration
        if args.model_type == "pca":
            config["n_components"] = args.n_components
            config["variance_threshold"] = args.variance_threshold
            config["pca_classifier"] = args.pca_classifier
            config["whiten"] = args.whiten
            config["calibrate"] = not args.no_calibrate
            # Add custom params for the PCA's base classifier
            custom_params = {}
            if args.C is not None:
                custom_params["C"] = args.C
            if args.max_depth is not None:
                custom_params["max_depth"] = args.max_depth
            if args.n_estimators is not None:
                custom_params["n_estimators"] = args.n_estimators
            if custom_params:
                config["custom_params"] = custom_params

        elif args.model_type in ["catboost", "lightgbm", "xgboost"]:
            config["calibrate"] = not args.no_calibrate
            config["calibration_method"] = args.calibration_method
            config["custom_params"] = {
                "n_estimators": args.n_estimators or 500,
                "max_depth": args.max_depth or 6,
                "learning_rate": args.learning_rate or 0.03,
            }
        elif args.model_type in ["mc_dropout", "bnn"]:
            config["hidden_dims"] = args.hidden_dims
            config["learning_rate"] = args.learning_rate or (
                0.001 if args.model_type == "mc_dropout" else 0.01
            )
            config["num_epochs"] = args.num_epochs or 100
            config["batch_size"] = args.batch_size
            if args.model_type == "mc_dropout":
                config["dropout_rate"] = args.dropout_rate or 0.2
            else:
                config["prior_scale"] = args.prior_scale
        elif args.model_type in ["sparse_gp", "deep_kernel_gp"]:
            config["kernel_type"] = args.kernel_type
            config["num_inducing"] = args.num_inducing
            config["learning_rate"] = args.learning_rate or 0.01
            config["num_epochs"] = args.num_epochs or 100
            config["batch_size"] = args.batch_size
            config["hidden_dims"] = args.hidden_dims
            config["use_ard"] = not args.no_ard

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
    else:
        # Config for original trainer (default: all variables with missing rate < threshold)
        config = MODEL_CONFIGS.get(args.model_type, {}).copy()
        config["calibrate"] = not args.no_calibrate
        config["random_seed"] = args.random_seed
        config.update(
            build_feature_config(
                required_only=args.required_only,
                drop_high_missing=args.drop_high_missing,
                exclude_cols=args.exclude_cols,
            )
        )

        custom_params = {}
        if args.C is not None:
            custom_params["C"] = args.C
        if args.max_depth is not None:
            custom_params["max_depth"] = args.max_depth
        if args.n_estimators is not None:
            custom_params["n_estimators"] = args.n_estimators
        if args.learning_rate is not None:
            config["learning_rate"] = args.learning_rate
        if args.num_epochs is not None:
            config["num_epochs"] = args.num_epochs
        if args.dropout_rate is not None:
            config["dropout_rate"] = args.dropout_rate
        if custom_params:
            config["custom_params"] = custom_params

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

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Models saved to: {save_dir}")
    print(f"\nTo evaluate, run:")
    print(f"  python scripts/evaluate.py --model_dir {save_dir}")
    print()
    print(f"To generate predictions, run:")
    print(f"  python scripts/predict.py --model_dir {save_dir}")

    if args.model_type == "pca":
        print(f"\nPCA models trained successfully!")
        print(
            f"Check models/{args.model_type}/training_info.json for component details"
        )


if __name__ == "__main__":
    main()
