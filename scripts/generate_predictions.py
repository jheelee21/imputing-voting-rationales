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
from src.utils.types import CORE_RATIONALES


def load_model_and_preprocessor(model_dir: Path, rationale: str):
    """Load a trained model and its preprocessor."""
    model_path = model_dir / f"{rationale}_model.pkl"
    preprocessor_path = model_dir / f"{rationale}_preprocessor.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(f"Model or preprocessor not found for {rationale}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


def prepare_unlabeled_data(
    df: pd.DataFrame, preprocessor: DataPreprocessor, rationale: str
):
    """Prepare unlabeled data for prediction - optimized for large datasets."""
    # Keep original index to map back
    original_indices = df.index.copy()

    # Prepare features
    X_df = preprocessor.prepare_features(df)

    # OPTIMIZED: Encode categorical variables more efficiently
    # Instead of using the slow encode_categorical method, do it directly
    from src.utils.types import CATEGORICAL_IDS

    for col in CATEGORICAL_IDS:
        if col not in X_df.columns:
            continue

        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]

            # Convert to string for consistency
            col_values = X_df[col].astype(str)

            # Create a mapping dict for faster lookup (vectorized approach)
            mapping = {val: idx for idx, val in enumerate(le.classes_)}

            # Use map with the dictionary (much faster than lambda with transform)
            X_df[f"{col}_encoded"] = col_values.map(mapping).fillna(-1).astype(int)

    # Handle missing values
    X_df = preprocessor.handle_missing(X_df)

    # Separate categorical and numerical columns
    categorical_encoded_cols = [c for c in X_df.columns if c.endswith("_encoded")]
    numerical_cols = [c for c in X_df.columns if c not in categorical_encoded_cols]

    # Use scaler's feature names if available
    if hasattr(preprocessor.scaler, "feature_names_in_"):
        scaler_feature_names = preprocessor.scaler.feature_names_in_
        X_numerical_df = X_df[scaler_feature_names].copy()
    else:
        X_numerical_df = X_df[numerical_cols].copy()

    # Scale numerical features
    X_scaled = preprocessor.scaler.transform(X_numerical_df)

    # Combine scaled numerical and categorical features
    X = np.hstack([X_scaled, X_df[categorical_encoded_cols].values])

    return X, original_indices


def generate_predictions(
    rationales: list,
    model_dir: Path,
    unlabeled_df: pd.DataFrame,
    id_columns: list = None,
    batch_size: int = None,
) -> pd.DataFrame:
    """Generate probability predictions for all rationales on unlabeled data.

    Returns a DataFrame with ID columns and predicted probabilities for each rationale.

    Args:
        rationales: List of rationales to predict
        model_dir: Directory containing trained models
        unlabeled_df: DataFrame with unlabeled observations
        id_columns: List of ID columns to include in output
        batch_size: If provided, process data in batches (useful for very large datasets)
    """
    # Default ID columns
    if id_columns is None:
        id_columns = ["investor_id", "pid", "ProxySeason", "meeting_id"]

    # Filter to columns that exist
    existing_id_cols = [col for col in id_columns if col in unlabeled_df.columns]

    print(f"\n{'=' * 80}")
    print("GENERATING PROBABILITY PREDICTIONS")
    print(f"{'=' * 80}")
    print(f"Unlabeled samples: {len(unlabeled_df):,}")
    print(f"Rationales: {rationales}")
    print(f"ID columns: {existing_id_cols}")
    if batch_size:
        n_batches = (len(unlabeled_df) + batch_size - 1) // batch_size
        print(f"Batch processing: {n_batches} batches of {batch_size:,} samples")
    print(f"{'=' * 80}\n")

    # Start with ID columns
    predictions_df = unlabeled_df[existing_id_cols].copy()

    # Add predictions for each rationale
    for rationale in rationales:
        print(f"Processing {rationale}...", end=" ", flush=True)

        try:
            # Load model and preprocessor
            model, preprocessor = load_model_and_preprocessor(model_dir, rationale)

            if batch_size and len(unlabeled_df) > batch_size:
                # Process in batches for memory efficiency
                all_probs = []
                n_batches = (len(unlabeled_df) + batch_size - 1) // batch_size

                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(unlabeled_df))
                    batch_df = unlabeled_df.iloc[start_idx:end_idx]

                    # Prepare and predict batch
                    X_batch, _ = prepare_unlabeled_data(
                        batch_df, preprocessor, rationale
                    )
                    y_prob_batch = model.predict_proba(X_batch)
                    all_probs.append(y_prob_batch)

                    # Show progress for large datasets
                    if n_batches > 10 and (i + 1) % max(1, n_batches // 10) == 0:
                        print(
                            f"{((i + 1) / n_batches * 100):.0f}%...", end="", flush=True
                        )

                # Concatenate all batches
                y_prob = np.concatenate(all_probs)
            else:
                # Process all at once
                X, original_indices = prepare_unlabeled_data(
                    unlabeled_df, preprocessor, rationale
                )
                y_prob = model.predict_proba(X)

            # Add to dataframe with rationale name as column
            predictions_df[f"{rationale}_prob"] = y_prob

            print(f" ✓ (mean: {y_prob.mean():.3f}, std: {y_prob.std():.3f})")

        except Exception as e:
            print(f" ✗ Error: {e}")
            predictions_df[f"{rationale}_prob"] = np.nan

    print(f"\n{'=' * 80}")
    print(f"Predictions generated for {len(predictions_df):,} samples")
    print(f"{'=' * 80}\n")

    return predictions_df


def evaluate_prediction_confidence(
    predictions_df: pd.DataFrame, rationales: list
) -> pd.DataFrame:
    """Evaluate prediction confidence across all rationales.

    Returns summary statistics about prediction probabilities.
    """
    stats = []

    for rationale in rationales:
        prob_col = f"{rationale}_prob"

        if prob_col not in predictions_df.columns:
            continue

        probs = predictions_df[prob_col].dropna()

        if len(probs) == 0:
            continue

        # Calculate statistics
        stat = {
            "rationale": rationale,
            "n_predictions": len(probs),
            "mean_prob": probs.mean(),
            "std_prob": probs.std(),
            "min_prob": probs.min(),
            "q25_prob": probs.quantile(0.25),
            "median_prob": probs.quantile(0.50),
            "q75_prob": probs.quantile(0.75),
            "q90_prob": probs.quantile(0.90),
            "q95_prob": probs.quantile(0.95),
            "max_prob": probs.max(),
            "n_high_conf": (probs >= 0.7).sum(),
            "pct_high_conf": (probs >= 0.7).mean() * 100,
            "n_medium_conf": ((probs >= 0.3) & (probs < 0.7)).sum(),
            "pct_medium_conf": ((probs >= 0.3) & (probs < 0.7)).mean() * 100,
            "n_low_conf": (probs < 0.3).sum(),
            "pct_low_conf": (probs < 0.3).mean() * 100,
        }

        stats.append(stat)

    return pd.DataFrame(stats)


def analyze_multi_label_predictions(
    predictions_df: pd.DataFrame, rationales: list
) -> dict:
    """Analyze how many rationales are predicted per observation.

    Uses a threshold of 0.5 to determine "predicted" rationales.
    """
    prob_cols = [
        f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns
    ]

    if not prob_cols:
        return {}

    # Create binary predictions at 0.5 threshold
    binary_preds = (predictions_df[prob_cols] >= 0.5).astype(int)

    # Count number of rationales per observation
    n_rationales_per_obs = binary_preds.sum(axis=1)

    analysis = {
        "total_observations": int(len(predictions_df)),
        "n_with_0_rationales": int((n_rationales_per_obs == 0).sum()),
        "n_with_1_rationale": int((n_rationales_per_obs == 1).sum()),
        "n_with_2_rationales": int((n_rationales_per_obs == 2).sum()),
        "n_with_3_rationales": int((n_rationales_per_obs == 3).sum()),
        "n_with_4plus_rationales": int((n_rationales_per_obs >= 4).sum()),
        "mean_rationales_per_obs": float(n_rationales_per_obs.mean()),
        "median_rationales_per_obs": float(n_rationales_per_obs.median()),
    }

    # Calculate percentage
    total = analysis["total_observations"]
    analysis["pct_with_0_rationales"] = float(
        (analysis["n_with_0_rationales"] / total) * 100
    )
    analysis["pct_with_1_rationale"] = float(
        (analysis["n_with_1_rationale"] / total) * 100
    )
    analysis["pct_with_2_rationales"] = float(
        (analysis["n_with_2_rationales"] / total) * 100
    )
    analysis["pct_with_3_rationales"] = float(
        (analysis["n_with_3_rationales"] / total) * 100
    )
    analysis["pct_with_4plus_rationales"] = float(
        (analysis["n_with_4plus_rationales"] / total) * 100
    )

    return analysis


def print_confidence_summary(confidence_stats: pd.DataFrame):
    """Print a formatted summary of prediction confidence."""
    print(f"\n{'=' * 80}")
    print("PREDICTION CONFIDENCE SUMMARY")
    print(f"{'=' * 80}\n")

    # Main statistics
    print("Mean Predicted Probabilities by Rationale:")
    print(f"{'-' * 80}")
    for _, row in confidence_stats.iterrows():
        print(
            f"{row['rationale']:25s}  Mean: {row['mean_prob']:.3f}  "
            f"Std: {row['std_prob']:.3f}  Median: {row['median_prob']:.3f}"
        )

    print(f"\n{'-' * 80}")
    print("Confidence Distribution (% of predictions):")
    print(f"{'-' * 80}")
    print(
        f"{'Rationale':<25s} {'High (≥0.7)':>12s} {'Medium (0.3-0.7)':>18s} {'Low (<0.3)':>12s}"
    )
    print(f"{'-' * 80}")
    for _, row in confidence_stats.iterrows():
        print(
            f"{row['rationale']:25s} "
            f"{row['pct_high_conf']:>11.1f}% "
            f"{row['pct_medium_conf']:>17.1f}% "
            f"{row['pct_low_conf']:>11.1f}%"
        )

    print(f"\n{'-' * 80}")
    print("Percentile Distribution:")
    print(f"{'-' * 80}")
    print(
        f"{'Rationale':<25s} {'Min':>6s} {'25%':>6s} {'50%':>6s} {'75%':>6s} {'90%':>6s} {'95%':>6s} {'Max':>6s}"
    )
    print(f"{'-' * 80}")
    for _, row in confidence_stats.iterrows():
        print(
            f"{row['rationale']:25s} "
            f"{row['min_prob']:>6.3f} "
            f"{row['q25_prob']:>6.3f} "
            f"{row['median_prob']:>6.3f} "
            f"{row['q75_prob']:>6.3f} "
            f"{row['q90_prob']:>6.3f} "
            f"{row['q95_prob']:>6.3f} "
            f"{row['max_prob']:>6.3f}"
        )


def print_multi_label_analysis(analysis: dict):
    """Print multi-label prediction analysis."""
    print(f"\n{'=' * 80}")
    print("MULTI-LABEL PREDICTION ANALYSIS (Threshold = 0.5)")
    print(f"{'=' * 80}\n")

    print(f"Total observations: {analysis['total_observations']:,}")
    print(f"Mean rationales per observation: {analysis['mean_rationales_per_obs']:.2f}")
    print(
        f"Median rationales per observation: {analysis['median_rationales_per_obs']:.1f}"
    )

    print(f"\n{'-' * 80}")
    print("Distribution of Rationale Count per Observation:")
    print(f"{'-' * 80}")
    print(
        f"0 rationales:   {analysis['n_with_0_rationales']:>8,}  ({analysis['pct_with_0_rationales']:>5.1f}%)"
    )
    print(
        f"1 rationale:    {analysis['n_with_1_rationale']:>8,}  ({analysis['pct_with_1_rationale']:>5.1f}%)"
    )
    print(
        f"2 rationales:   {analysis['n_with_2_rationales']:>8,}  ({analysis['pct_with_2_rationales']:>5.1f}%)"
    )
    print(
        f"3 rationales:   {analysis['n_with_3_rationales']:>8,}  ({analysis['pct_with_3_rationales']:>5.1f}%)"
    )
    print(
        f"4+ rationales:  {analysis['n_with_4plus_rationales']:>8,}  ({analysis['pct_with_4plus_rationales']:>5.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate probability predictions for unlabeled voting rationales"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/Imputing Rationales.csv",
        help="Path to data file",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models/trained/supervised",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/predictions",
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--rationales", nargs="+", default=CORE_RATIONALES, help="Rationales to predict"
    )
    parser.add_argument(
        "--id_columns",
        nargs="+",
        default=["investor_id", "pid", "ProxySeason", "meeting_id"],
        help="ID columns to include in output",
    )
    parser.add_argument(
        "--min_meetings_rat", type=int, default=1, help="Minimum N_Meetings_Rat filter"
    )
    parser.add_argument(
        "--min_dissent", type=int, default=5, help="Minimum N_dissent filter"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="unlabeled_predictions.csv",
        help="Output filename for predictions",
    )
    parser.add_argument(
        "--save_stats",
        action="store_true",
        help="Save confidence statistics to separate file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Process data in batches (useful for very large datasets, e.g., 10000)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    print("=" * 80)
    print("UNLABELED VOTING RATIONALE PREDICTION")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Rationales: {args.rationales}")
    print(f"ID columns: {args.id_columns}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    loader = DataLoader()
    df = loader.load_data(args.data_path)

    # Apply filters
    df = loader.apply_filters(
        min_meetings_rat=args.min_meetings_rat, min_dissent=args.min_dissent
    )

    # Get unlabeled data (dissents with missing rationales)
    unlabeled_df = loader.get_unlabeled_data(args.rationales)

    if len(unlabeled_df) == 0:
        print("\n⚠️  No unlabeled data found!")
        print("All dissent observations have at least one rationale labeled.")
        return

    print(f"\n✓ Found {len(unlabeled_df):,} unlabeled observations")

    # Generate predictions
    predictions_df = generate_predictions(
        rationales=args.rationales,
        model_dir=model_dir,
        unlabeled_df=unlabeled_df,
        id_columns=args.id_columns,
        batch_size=args.batch_size,
    )

    # Evaluate prediction confidence
    confidence_stats = evaluate_prediction_confidence(predictions_df, args.rationales)

    # Analyze multi-label predictions
    ml_analysis = analyze_multi_label_predictions(predictions_df, args.rationales)

    # Print summaries
    print_confidence_summary(confidence_stats)
    if ml_analysis:
        print_multi_label_analysis(ml_analysis)

    # Save predictions
    output_path = output_dir / args.output_filename
    predictions_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"✓ Predictions saved to: {output_path}")
    print(f"  Total rows: {len(predictions_df):,}")
    print(f"  Total columns: {len(predictions_df.columns)}")

    # Save confidence statistics if requested
    if args.save_stats:
        stats_path = output_dir / "confidence_statistics.csv"
        confidence_stats.to_csv(stats_path, index=False)
        print(f"✓ Confidence statistics saved to: {stats_path}")

        ml_stats_path = output_dir / "multi_label_statistics.json"
        with open(ml_stats_path, "w") as f:
            json.dump(ml_analysis, f, indent=2)
        print(f"✓ Multi-label statistics saved to: {ml_stats_path}")

    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "data_path": args.data_path,
        "model_dir": str(model_dir),
        "rationales": args.rationales,
        "id_columns": args.id_columns,
        "filters": {
            "min_meetings_rat": args.min_meetings_rat,
            "min_dissent": args.min_dissent,
        },
        "n_predictions": len(predictions_df),
        "mean_probabilities": {
            rationale: float(predictions_df[f"{rationale}_prob"].mean())
            for rationale in args.rationales
            if f"{rationale}_prob" in predictions_df.columns
        },
    }

    metadata_path = output_dir / "prediction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")

    print(f"{'=' * 80}\n")

    # Print sample of predictions
    print(f"{'=' * 80}")
    print("SAMPLE PREDICTIONS (First 10 rows)")
    print(f"{'=' * 80}")
    print(predictions_df.head(10).to_string(index=False))
    print(f"\n{'=' * 80}")
    print("PREDICTION COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
