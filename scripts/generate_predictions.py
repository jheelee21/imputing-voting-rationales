import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.models.supervised import SupervisedModel
from src.utils.types import CORE_RATIONALES


def load_model(model_path: Path) -> SupervisedModel:
    """Load a trained model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_preprocessor(preprocessor_path: Path) -> DataPreprocessor:
    """Load a fitted preprocessor from disk."""
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor


def generate_predictions_single_rationale(
    rationale: str,
    unlabeled_df: pd.DataFrame,
    model: SupervisedModel,
    preprocessor: DataPreprocessor,
) -> pd.DataFrame:
    """
    Generate predictions for a single rationale.

    Args:
        rationale: Name of the rationale
        unlabeled_df: DataFrame with unlabeled observations
        model: Trained SupervisedModel
        preprocessor: Fitted DataPreprocessor

    Returns:
        DataFrame with predictions
    """
    print(f"\nGenerating predictions for: {rationale}")
    print(f"Unlabeled samples: {len(unlabeled_df)}")

    if len(unlabeled_df) == 0:
        return pd.DataFrame()

    # Prepare features
    X_df = preprocessor.prepare_features(unlabeled_df)
    X_df = preprocessor.encode_categorical(X_df, fit=False)
    X_df = preprocessor.handle_missing(X_df)

    # Separate encoded categorical features from numerical features
    categorical_encoded_cols = [c for c in X_df.columns if c.endswith("_encoded")]
    numerical_cols = [c for c in X_df.columns if c not in categorical_encoded_cols]

    # Scale numerical features
    X_scaled = preprocessor.scale_features(X_df[numerical_cols], fit=False)

    # Combine scaled numerical with encoded categorical
    X = np.hstack([X_scaled, X_df[categorical_encoded_cols].values])

    # Generate predictions
    y_prob = model.predict_proba(X)

    # Create results DataFrame
    results_df = unlabeled_df[
        ["investor_id", "pid", "ProxySeason", "meeting_id"]
    ].copy()
    results_df[f"{rationale}_prob"] = y_prob

    return results_df


def generate_all_predictions(
    rationales: list,
    unlabeled_df: pd.DataFrame,
    models_dir: Path,
) -> pd.DataFrame:
    """
    Generate predictions for all rationales.

    Args:
        rationales: List of rationales
        unlabeled_df: DataFrame with unlabeled observations
        models_dir: Directory containing trained models

    Returns:
        DataFrame with all predictions
    """
    # Start with base columns
    base_cols = ["investor_id", "pid", "ProxySeason", "meeting_id"]
    predictions_df = unlabeled_df[base_cols].copy()

    for rationale in rationales:
        model_path = models_dir / f"{rationale}_model.pkl"
        preprocessor_path = models_dir / f"{rationale}_preprocessor.pkl"

        if not model_path.exists():
            print(f"\nModel not found for {rationale}: {model_path}")
            continue

        if not preprocessor_path.exists():
            print(f"\nPreprocessor not found for {rationale}: {preprocessor_path}")
            continue

        # Load model and preprocessor
        model = load_model(model_path)
        preprocessor = load_preprocessor(preprocessor_path)

        # Skip if model wasn't trained
        if model.model is None:
            print(f"\nModel for {rationale} was not trained (no positive samples)")
            continue

        # Generate predictions
        rationale_preds = generate_predictions_single_rationale(
            rationale, unlabeled_df, model, preprocessor
        )

        if not rationale_preds.empty:
            # Merge predictions
            predictions_df = predictions_df.merge(
                rationale_preds[[f"{rationale}_prob", f"{rationale}_pred"]],
                left_index=True,
                right_index=True,
                how="left",
            )

    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for unlabeled voting rationales"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/Imputing Rationales.csv",
        help="Path to data file",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="../models/trained",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/predictions",
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--rationales",
        nargs="+",
        default=CORE_RATIONALES,
        help="Rationales to predict",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
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

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(args.models_dir)

    print("=" * 80)
    print("VOTING RATIONALE PREDICTION GENERATION")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Models directory: {models_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Rationales: {args.rationales}")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)

    # Load data
    loader = DataLoader(data_dir="../data")
    df = loader.load_data(args.data_path)

    # Apply filters
    df = loader.apply_filters(
        min_meetings_rat=args.min_meetings_rat, min_dissent=args.min_dissent
    )

    # Get unlabeled data (dissent but no rationales)
    unlabeled_df = loader.get_unlabeled_data(args.rationales)

    print(f"\nUnlabeled observations to predict: {len(unlabeled_df)}")

    if len(unlabeled_df) == 0:
        print("No unlabeled data found. Exiting.")
        return

    # Generate predictions
    predictions_df = generate_all_predictions(
        args.rationales, unlabeled_df, models_dir, args.threshold
    )

    # Add metadata
    predictions_df["prediction_date"] = datetime.now().isoformat()
    predictions_df["threshold"] = args.threshold

    # Save predictions
    output_path = output_dir / "unlabeled_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)

    for rationale in args.rationales:
        pred_col = f"{rationale}_pred"
        prob_col = f"{rationale}_prob"

        if pred_col in predictions_df.columns:
            n_positive = predictions_df[pred_col].sum()
            pct_positive = predictions_df[pred_col].mean()
            avg_prob = predictions_df[prob_col].mean()

            print(f"\n{rationale}:")
            print(f"  Predicted positive: {n_positive} ({pct_positive:.2%})")
            print(f"  Average probability: {avg_prob:.4f}")

    # Save summary with original data
    full_output_path = output_dir / "unlabeled_predictions_with_features.csv"
    full_df = unlabeled_df.merge(
        predictions_df,
        on=["investor_id", "pid", "ProxySeason", "meeting_id"],
        how="left",
    )
    full_df.to_csv(full_output_path, index=False)
    print(f"\nFull predictions with features saved to {full_output_path}")

    print("\n" + "=" * 80)
    print("PREDICTION GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
