import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.models.supervised import SupervisedModel
from src.utils.types import CORE_RATIONALES


def train_single_rationale(
    rationale: str,
    train_df: pd.DataFrame,
    model_type: str = "logistic",
    calibrate: bool = True,
    random_seed: int = 21,
    additional_features: list = None,
    custom_params: dict = None,
) -> tuple:
    print(f"Training model for: {rationale.upper()}")

    # Filter to only samples with this rationale labeled
    train_df_clean = train_df.dropna(subset=[rationale]).copy()
    print(f"Training samples (after removing NaN): {len(train_df_clean)}")

    if len(train_df_clean) == 0:
        print(f"No training data available for {rationale}")
        return None, None, None

    preprocessor = DataPreprocessor(rationales=[rationale])

    X_train_df = preprocessor.prepare_features(
        train_df_clean, additional_features=additional_features
    )
    X_train_df = preprocessor.encode_categorical(X_train_df, fit=True)
    X_train_df = preprocessor.handle_missing(X_train_df)

    y_train = (train_df_clean[rationale] == 1).astype(int).values

    categorical_encoded_cols = [c for c in X_train_df.columns if c.endswith("_encoded")]
    numerical_cols = [
        c for c in X_train_df.columns if c not in categorical_encoded_cols
    ]

    X_train_scaled = preprocessor.scale_features(X_train_df[numerical_cols], fit=True)

    X_train = np.hstack([X_train_scaled, X_train_df[categorical_encoded_cols].values])

    feature_names = list(numerical_cols) + list(categorical_encoded_cols)

    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Positive samples: {y_train.sum()} ({y_train.mean():.2%})")

    model = SupervisedModel(
        rationale=rationale,
        calibrate=calibrate,
        base_model=model_type,
        custom_params=custom_params,
        random_seed=random_seed,
    )

    model.feature_names = feature_names

    print(f"Training {model_type.upper()} model...")
    model.fit(X_train, y_train)

    if model.model is not None:
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)

        train_accuracy = (y_train == y_train_pred).mean()
        print(f"Training accuracy: {train_accuracy:.4f}")

        if model.feature_importance is not None:
            print("\nTop 10 important features:")
            top_features = model.get_top_features(top_k=10)
            print(top_features.to_string(index=False))
    else:
        train_accuracy = 0.0

    training_info = {
        "rationale": rationale,
        "model_type": model_type,
        "calibrate": calibrate,
        "n_train": len(y_train),
        "n_positive": int(y_train.sum()),
        "positive_rate": float(y_train.mean()),
        "n_features": X_train.shape[1],
        "train_accuracy": train_accuracy,
        "timestamp": datetime.now().isoformat(),
    }

    return model, preprocessor, training_info


def train_all_rationales(
    rationales: list,
    train_df: pd.DataFrame,
    model_type: str = "logistic",
    calibrate: bool = True,
    random_seed: int = 21,
    save_dir: Path = None,
    additional_features: list = None,
    custom_params: dict = None,
) -> dict:
    results = {}

    for rationale in rationales:
        model, preprocessor, info = train_single_rationale(
            rationale,
            train_df,
            model_type,
            calibrate,
            random_seed,
            additional_features,
            custom_params,
        )

        if model is None:
            continue

        results[rationale] = info

        if save_dir:
            model_path = save_dir / f"{rationale}_model.pkl"
            preprocessor_path = save_dir / f"{rationale}_preprocessor.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved to {model_path}")

            with open(preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)
            print(f"Preprocessor saved to {preprocessor_path}")

    if results:
        print("=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)

        summary_df = pd.DataFrame(results).T
        print(
            summary_df[
                ["n_train", "n_positive", "positive_rate", "train_accuracy"]
            ].to_string()
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train supervised models for voting rationales"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/Imputing Rationales.csv",
        help="Path to data file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models/trained",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--rationales",
        nargs="+",
        default=CORE_RATIONALES,
        help="Rationales to train models for",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest", "gradient_boosting"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--no_calibrate",
        action="store_true",
        help="Disable probability calibration",
    )
    parser.add_argument(
        "--min_meetings_rat",
        type=int,
        default=1,
        help="Minimum N_Meetings_Rat filter",
    )
    parser.add_argument(
        "--min_dissent",
        type=int,
        default=5,
        help="Minimum N_dissent filter",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size (for splitting)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--additional_features",
        nargs="+",
        default=None,
        help="Additional features to include beyond required features",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Number of estimators for tree-based models",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Max depth for tree-based models",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for gradient boosting",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=None,
        help="Regularization parameter for logistic regression",
    )

    args = parser.parse_args()

    custom_params = {}
    if args.n_estimators is not None:
        custom_params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        custom_params["max_depth"] = args.max_depth
    if args.learning_rate is not None:
        custom_params["learning_rate"] = args.learning_rate
    if args.C is not None:
        custom_params["C"] = args.C

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VOTING RATIONALE MODEL TRAINING")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Save directory: {save_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Calibrate: {not args.no_calibrate}")
    print(f"Rationales: {args.rationales}")
    print(f"Random seed: {args.random_seed}")
    if custom_params:
        print(f"Custom params: {custom_params}")
    print("=" * 80)

    loader = DataLoader()
    df = loader.load_data(args.data_path)

    df = loader.apply_filters(
        min_meetings_rat=args.min_meetings_rat, min_dissent=args.min_dissent
    )

    train_df, test_df = loader.split_train_test(
        test_size=args.test_size,
        random_seed=args.random_seed,
        label=args.rationales,
    )

    print(f"\nTraining set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    results = train_all_rationales(
        rationales=args.rationales,
        train_df=train_df,
        model_type=args.model_type,
        calibrate=not args.no_calibrate,
        random_seed=args.random_seed,
        save_dir=save_dir,
        additional_features=args.additional_features,
        custom_params=custom_params if custom_params else None,
    )

    if results:
        summary_path = save_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nTraining summary saved to {summary_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
