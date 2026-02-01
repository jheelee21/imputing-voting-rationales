import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.evaluation.metrics import (
    EvaluationMetrics,
    evaluate_probabilistic_predictions,
    plot_calibration_curves,
    plot_probability_distributions,
    analyze_prediction_confidence,
)
from src.utils.types import CORE_RATIONALES


def load_trained_model(model_path: Path, preprocessor_path: Path):
    """Load a trained model and its preprocessor."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


def prepare_test_data(
    test_df: pd.DataFrame,
    preprocessor: DataPreprocessor,
    rationale: str,
    additional_features: list = None,
):
    """Prepare test data using the same preprocessing steps as training.

    This must exactly match the feature preparation in train_model.py to ensure
    the scaler receives features in the same order with the same column names.
    """
    # Filter to only samples with this rationale labeled
    test_df_clean = test_df.dropna(subset=[rationale]).copy()

    if len(test_df_clean) == 0:
        return None, None

    # Prepare features - use same additional_features as training if any
    X_test_df = preprocessor.prepare_features(
        test_df_clean, additional_features=additional_features
    )

    # Encode categorical variables
    X_test_df = preprocessor.encode_categorical(X_test_df, fit=False)

    # Handle missing values
    X_test_df = preprocessor.handle_missing(X_test_df)

    # Get true labels
    y_test = (test_df_clean[rationale] == 1).astype(int).values

    # Separate categorical and numerical columns (MUST be in same order as training)
    categorical_encoded_cols = [c for c in X_test_df.columns if c.endswith("_encoded")]
    numerical_cols = [c for c in X_test_df.columns if c not in categorical_encoded_cols]

    # CRITICAL: If scaler has feature_names_in_, use those to ensure exact column order
    if hasattr(preprocessor.scaler, "feature_names_in_"):
        # Use the exact feature names the scaler expects, in the exact order
        scaler_feature_names = preprocessor.scaler.feature_names_in_
        X_test_numerical_df = X_test_df[scaler_feature_names].copy()
    else:
        # Fallback to numerical_cols if scaler doesn't have feature_names_in_
        X_test_numerical_df = X_test_df[numerical_cols].copy()

    # Scale numerical features - scaler expects DataFrame with same column names as during fit
    X_test_scaled = preprocessor.scaler.transform(X_test_numerical_df)

    # Combine scaled numerical and categorical features in the SAME ORDER as training
    # Training uses: [numerical_cols (scaled), categorical_encoded_cols]
    X_test = np.hstack([X_test_scaled, X_test_df[categorical_encoded_cols].values])

    return X_test, y_test


def evaluate_single_rationale(
    rationale: str,
    model_dir: Path,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
    save_plots: bool = True,
    output_dir: Path = None,
    additional_features: list = None,
) -> dict:
    """Evaluate a single trained model on test data."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {rationale.upper()}")
    print(f"{'=' * 80}")

    # Load model and preprocessor
    model_path = model_dir / f"{rationale}_model.pkl"
    preprocessor_path = model_dir / f"{rationale}_preprocessor.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        print(f"Model or preprocessor not found for {rationale}")
        return None

    model, preprocessor = load_trained_model(model_path, preprocessor_path)

    # Prepare test data
    X_test, y_test = prepare_test_data(
        test_df, preprocessor, rationale, additional_features
    )

    if X_test is None:
        print(f"No test data available for {rationale}")
        return None

    print(f"Test samples: {len(y_test)}")
    print(f"Positive samples: {y_test.sum()} ({y_test.mean():.2%})")

    # Get predictions
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=threshold)

    # Create DataFrames for evaluation
    y_true_df = pd.DataFrame({rationale: y_test})
    y_pred_df = pd.DataFrame({rationale: y_pred})
    y_prob_df = pd.DataFrame({rationale: y_prob})

    # Evaluate - with error handling for metrics.py bugs
    try:
        evaluator = EvaluationMetrics(rationales=[rationale])
        results = evaluator.evaluate(y_true_df, y_pred_df, y_prob_df)

        # Print results
        evaluator.print_reports(results)

        # Get metrics for summary
        if len(results["per_label"]) > 0:
            per_label_metrics = (
                results["per_label"].iloc[0].to_dict()
                if hasattr(results["per_label"], "iloc")
                else results["per_label"][0]
            )
        else:
            per_label_metrics = {}
    except Exception as e:
        print(f"Warning: Error using EvaluationMetrics class: {e}")
        print("Falling back to direct metric calculation...")

        # Calculate metrics directly
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            average_precision_score,
            log_loss,
            confusion_matrix,
        )

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            roc_auc = np.nan

        try:
            avg_prec = average_precision_score(y_test, y_prob)
        except:
            avg_prec = np.nan

        try:
            logloss = log_loss(y_test, y_prob)
        except:
            logloss = np.nan

        cm = confusion_matrix(y_test, y_pred)

        per_label_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "avg_precision": avg_prec,
            "log_loss": logloss,
            "support": int(y_test.sum()),
            "positive_rate": float(y_test.mean()),
        }

        # Print metrics
        print(f"\n{'-' * 80}")
        print("PERFORMANCE METRICS")
        print(f"{'-' * 80}")
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Precision:          {precision:.4f}")
        print(f"Recall:             {recall:.4f}")
        print(f"F1-Score:           {f1:.4f}")
        print(f"ROC-AUC:            {roc_auc:.4f}")
        print(f"Avg Precision:      {avg_prec:.4f}")
        print(f"Log Loss:           {logloss:.4f}")

        print(f"\n{'-' * 80}")
        print("CONFUSION MATRIX")
        print(f"{'-' * 80}")
        print(f"                 Predicted")
        print(f"                 Neg    Pos")
        print(f"Actual  Neg    {cm[0, 0]:5d}  {cm[0, 1]:5d}")
        print(f"        Pos    {cm[1, 0]:5d}  {cm[1, 1]:5d}")

        results = {
            "per_label": pd.DataFrame([per_label_metrics]),
            "confusion_matrices": {rationale: cm},
        }

    # Additional probabilistic metrics
    print(f"\n{'-' * 80}")
    print("PROBABILISTIC METRICS")
    print(f"{'-' * 80}")
    try:
        prob_results = evaluate_probabilistic_predictions(
            y_true_df, y_prob_df, [rationale]
        )
        print(prob_results.to_string(index=False))
    except Exception as e:
        print(f"Could not compute probabilistic metrics: {e}")

    # Prediction confidence analysis
    print(f"\n{'-' * 80}")
    print("PREDICTION CONFIDENCE ANALYSIS")
    print(f"{'-' * 80}")
    try:
        confidence_analysis = analyze_prediction_confidence(y_prob_df, [rationale])
        print(confidence_analysis.to_string(index=False))
    except Exception as e:
        print(f"Could not compute confidence analysis: {e}")
        # Compute basic confidence stats manually
        quantiles = np.percentile(y_prob, [25, 50, 75, 90, 95])
        print(f"Min:      {y_prob.min():.3f}")
        print(f"25th:     {quantiles[0]:.3f}")
        print(f"Median:   {quantiles[1]:.3f}")
        print(f"75th:     {quantiles[2]:.3f}")
        print(f"90th:     {quantiles[3]:.3f}")
        print(f"95th:     {quantiles[4]:.3f}")
        print(f"Max:      {y_prob.max():.3f}")

    # Save plots if requested
    if save_plots and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # ROC curve
        try:
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"{rationale} (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {rationale}")
            plt.legend(loc="lower right")
            plt.savefig(
                output_dir / f"{rationale}_roc_curve.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            print(f"ROC curve saved to {output_dir / f'{rationale}_roc_curve.png'}")
        except Exception as e:
            print(f"Could not generate ROC curve: {e}")

        # Calibration curve
        try:
            fig = plot_calibration_curves(y_true_df, y_prob_df, [rationale])
            fig.savefig(
                output_dir / f"{rationale}_calibration.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(
                f"Calibration curve saved to {output_dir / f'{rationale}_calibration.png'}"
            )
        except Exception as e:
            print(f"Could not generate calibration curve: {e}")

        # Probability distributions
        try:
            fig = plot_probability_distributions(y_prob_df, [rationale], y_true_df)
            fig.savefig(
                output_dir / f"{rationale}_prob_dist.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
            print(
                f"Probability distribution saved to {output_dir / f'{rationale}_prob_dist.png'}"
            )
        except Exception as e:
            print(f"Could not generate probability distribution: {e}")

    # Compile results
    eval_summary = {
        "rationale": rationale,
        "n_test": len(y_test),
        "n_positive": int(y_test.sum()),
        "positive_rate": float(y_test.mean()),
        "threshold": threshold,
        "per_label_metrics": per_label_metrics,
        "confusion_matrix": {
            "TN": int(results["confusion_matrices"][rationale][0, 0]),
            "FP": int(results["confusion_matrices"][rationale][0, 1]),
            "FN": int(results["confusion_matrices"][rationale][1, 0]),
            "TP": int(results["confusion_matrices"][rationale][1, 1]),
        },
        "timestamp": datetime.now().isoformat(),
    }

    return eval_summary


def evaluate_all_rationales(
    rationales: list,
    model_dir: Path,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
    save_plots: bool = True,
    output_dir: Path = None,
    additional_features: list = None,
) -> dict:
    """Evaluate all trained models."""
    all_results = {}

    for rationale in rationales:
        result = evaluate_single_rationale(
            rationale,
            model_dir,
            test_df,
            threshold,
            save_plots,
            output_dir,
            additional_features,
        )

        if result:
            all_results[rationale] = result

    return all_results


def evaluate_multi_label(
    rationales: list,
    model_dir: Path,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
    save_plots: bool = True,
    output_dir: Path = None,
    additional_features: list = None,
) -> dict:
    """Evaluate all models together as a multi-label system."""
    print(f"\n{'=' * 80}")
    print("MULTI-LABEL EVALUATION")
    print(f"{'=' * 80}")

    # Prepare containers
    y_true_all = {}
    y_pred_all = {}
    y_prob_all = {}

    # Load all models and make predictions
    for rationale in rationales:
        model_path = model_dir / f"{rationale}_model.pkl"
        preprocessor_path = model_dir / f"{rationale}_preprocessor.pkl"

        if not model_path.exists() or not preprocessor_path.exists():
            print(f"Skipping {rationale} - model not found")
            continue

        model, preprocessor = load_trained_model(model_path, preprocessor_path)
        X_test, y_test = prepare_test_data(
            test_df, preprocessor, rationale, additional_features
        )

        if X_test is None:
            print(f"Skipping {rationale} - no test data")
            continue

        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test, threshold=threshold)

        y_true_all[rationale] = y_test
        y_pred_all[rationale] = y_pred
        y_prob_all[rationale] = y_prob

    if not y_true_all:
        print("No models available for multi-label evaluation")
        return None

    # Find common indices (samples that have all rationales labeled)
    # For simplicity, we'll use the first rationale's length
    n_samples = len(next(iter(y_true_all.values())))

    # Create DataFrames
    y_true_df = pd.DataFrame(y_true_all)
    y_pred_df = pd.DataFrame(y_pred_all)
    y_prob_df = pd.DataFrame(y_prob_all)

    # Evaluate using multi-label metrics
    evaluator = EvaluationMetrics(rationales=list(y_true_all.keys()))
    results = evaluator.evaluate(y_true_df, y_pred_df, y_prob_df)

    # Print results
    evaluator.print_reports(results)

    # Save plots if requested
    if save_plots and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Combined ROC curves
        try:
            evaluator.plot_roc_curves(y_true_df, y_prob_df)
            plt.title("ROC Curves - All Rationales")
            plt.legend(loc="lower right")
            plt.savefig(
                output_dir / "all_rationales_roc_curves.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"Combined ROC curves saved")
        except Exception as e:
            print(f"Could not generate combined ROC curves: {e}")

        # Combined calibration curves
        try:
            fig = plot_calibration_curves(y_true_df, y_prob_df, list(y_true_all.keys()))
            fig.savefig(
                output_dir / "all_rationales_calibration.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"Combined calibration curves saved")
        except Exception as e:
            print(f"Could not generate combined calibration curves: {e}")

    # Compile summary
    summary = {
        "n_rationales": len(y_true_all),
        "n_test_samples": n_samples,
        "overall_metrics": results["overall"].to_dict("records"),
        "per_label_metrics": results["per_label"].to_dict("records"),
        "timestamp": datetime.now().isoformat(),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained voting rationale models"
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
        default="./models/trained",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--rationales",
        nargs="+",
        default=CORE_RATIONALES,
        help="Rationales to evaluate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for predictions",
    )
    parser.add_argument(
        "--min_meetings_rat", type=int, default=1, help="Minimum N_Meetings_Rat filter"
    )
    parser.add_argument(
        "--min_dissent", type=int, default=5, help="Minimum N_dissent filter"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size (should match training)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed (should match training)",
    )
    parser.add_argument(
        "--no_plots", action="store_true", help="Disable plot generation"
    )
    parser.add_argument(
        "--multi_label", action="store_true", help="Also perform multi-label evaluation"
    )
    parser.add_argument(
        "--additional_features",
        nargs="+",
        default=None,
        help="Additional features used during training (if any)",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VOTING RATIONALE MODEL EVALUATION")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Rationales: {args.rationales}")
    print(f"Threshold: {args.threshold}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 80)

    # Load data
    loader = DataLoader()
    df = loader.load_data(args.data_path)

    # Apply filters
    df = loader.apply_filters(
        min_meetings_rat=args.min_meetings_rat, min_dissent=args.min_dissent
    )

    # Split data (using same seed as training)
    train_df, test_df = loader.split_train_test(
        test_size=args.test_size,
        random_seed=args.random_seed,
        label=args.rationales,
    )

    print(f"\nTest set: {len(test_df)} samples")

    # Evaluate individual models
    print(f"\n{'=' * 80}")
    print("INDIVIDUAL MODEL EVALUATION")
    print(f"{'=' * 80}")

    individual_results = evaluate_all_rationales(
        rationales=args.rationales,
        model_dir=model_dir,
        test_df=test_df,
        threshold=args.threshold,
        save_plots=not args.no_plots,
        output_dir=output_dir,
        additional_features=args.additional_features,
    )

    # Save individual results
    if individual_results:
        results_path = output_dir / "individual_evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(individual_results, f, indent=2)
        print(f"\nIndividual evaluation results saved to {results_path}")

        # Create summary table
        summary_data = []
        for rationale, result in individual_results.items():
            metrics = result.get("per_label_metrics", {})
            summary_data.append(
                {
                    "rationale": rationale,
                    "n_test": result["n_test"],
                    "positive_rate": f"{result['positive_rate']:.2%}",
                    "accuracy": f"{metrics.get('accuracy', 0):.4f}",
                    "precision": f"{metrics.get('precision', 0):.4f}",
                    "recall": f"{metrics.get('recall', 0):.4f}",
                    "f1": f"{metrics.get('f1', 0):.4f}",
                    "roc_auc": f"{metrics.get('roc_auc', 0):.4f}",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Evaluation summary saved to {summary_path}")

        print(f"\n{'=' * 80}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 80}")
        print(summary_df.to_string(index=False))

    # Multi-label evaluation
    if args.multi_label:
        multi_label_results = evaluate_multi_label(
            rationales=args.rationales,
            model_dir=model_dir,
            test_df=test_df,
            threshold=args.threshold,
            save_plots=not args.no_plots,
            output_dir=output_dir,
            additional_features=args.additional_features,
        )

        if multi_label_results:
            ml_results_path = output_dir / "multi_label_evaluation_results.json"
            with open(ml_results_path, "w") as f:
                json.dump(multi_label_results, f, indent=2)
            print(f"\nMulti-label evaluation results saved to {ml_results_path}")

    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
