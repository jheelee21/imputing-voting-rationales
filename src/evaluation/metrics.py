import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Tuple
from src.utils.types import ALL_RATIONALES, CORE_RATIONALES
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


class EvaluationMetrics:
    def __init__(self, rationales: List[str] = CORE_RATIONALES):
        self.rationales = rationales

    def evaluate(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame, y_prob: pd.DataFrame
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        results = {}

        results["per_label"] = self._evaluate_per_label(y_true, y_pred, y_prob)
        results["overall"] = self._evaluate_overall(y_true, y_pred, y_prob)
        results["confusion_matrices"] = self._compute_confusion_matrices(y_true, y_pred)

        return results

    def _evaluate_per_label(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame, y_prob: pd.DataFrame
    ) -> pd.DataFrame:
        metrics = {}
        for rationale in self.rationales:
            y_t = y_true[rationale]
            y_p = y_pred[rationale]
            y_pr = y_prob[rationale]

            if y_t.sum() == 0:
                metrics.append(
                    {
                        "rationale": rationale,
                        "support": 0,
                        "positive_rate": 0.0,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "roc_auc": np.nan,
                        "avg_precision": np.nan,
                        "log_loss": np.nan,
                    }
                )
                continue

            try:
                roc_auc = roc_auc_score(y_t, y_prob)
            except:
                roc_auc = np.nan

            try:
                avg_prec = average_precision_score(y_t, y_prob)
            except:
                avg_prec = np.nan

            try:
                logloss = log_loss(y_t, y_prob)
            except:
                logloss = np.nan

            metrics.append(
                {
                    "rationale": rationale,
                    "support": int(y_t.sum()),
                    "positive_rate": y_t.mean(),
                    "accuracy": accuracy_score(y_t, y_p),
                    "precision": precision_score(y_t, y_p, zero_division=0),
                    "recall": recall_score(y_t, y_p, zero_division=0),
                    "f1": f1_score(y_t, y_p, zero_division=0),
                    "roc_auc": roc_auc,
                    "avg_precision": avg_prec,
                    "log_loss": logloss,
                }
            )

        return pd.DataFrame(metrics)

    def _evaluate_overall(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame, y_prob: pd.DataFrame
    ) -> pd.DataFrame:
        exact_match = (y_true == y_pred).all(axis=1).mean()

        # Hamming loss (fraction of wrong labels)
        hamming_loss = (y_true != y_pred).mean()

        # Micro-averaged metrics (treat all label-sample pairs equally)
        micro_precision = precision_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

        # Macro-averaged metrics (average across labels)
        macro_precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Weighted-averaged metrics (weighted by support)
        weighted_precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        weighted_recall = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results = pd.DataFrame(
            [
                {"metric": "exact_match_ratio", "value": exact_match},
                {"metric": "hamming_loss", "value": hamming_loss},
                {"metric": "micro_precision", "value": micro_precision},
                {"metric": "micro_recall", "value": micro_recall},
                {"metric": "micro_f1", "value": micro_f1},
                {"metric": "macro_precision", "value": macro_precision},
                {"metric": "macro_recall", "value": macro_recall},
                {"metric": "macro_f1", "value": macro_f1},
                {"metric": "weighted_precision", "value": weighted_precision},
                {"metric": "weighted_recall", "value": weighted_recall},
                {"metric": "weighted_f1", "value": weighted_f1},
            ]
        )

        return results

    def _compute_confusion_matrices(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        matrices = {}
        for rationale in self.rationales:
            matrices[rationale] = confusion_matrix(y_t, y_p)

        return matrices

    def print_reports(self, results: Dict[str, pd.DataFrame]):
        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)

        print("\n" + "-" * 80)
        print("PER-LABEL METRICS")
        print("-" * 80)
        print(results["per_label"].to_string(index=False))

        print("\n" + "-" * 80)
        print("OVERALL METRICS")
        print("-" * 80)
        print(results["overall"].to_string(index=False))

        print("\n" + "-" * 80)
        print("CONFUSION MATRICES")
        print("-" * 80)
        for rationale, cm in results["confusion_matrices"].items():
            print(f"\n{rationale}:")
            print(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
            print(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")

    def plot_roc_curves(self, y_true: pd.DataFrame, y_prob: pd.DataFrame):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        plt.figure(figsize=(10, 8))

        for rationale in self.rationales:
            y_t = y_true[rationale]
            y_pr = y_prob[rationale]

            if y_t.sum() == 0:
                continue

            fpr, tpr, _ = roc_curve(y_t, y_pr)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{rationale} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

    def plot_calibration_curves(self, y_true: pd.DataFrame, y_prob: pd.DataFrame):
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        plt.figure(figsize=(10, 8))

        for rationale in self.rationales:
            y_t = y_true[rationale]
            y_pr = y_prob[rationale]

            if y_t.sum() == 0:
                continue

            prob_true, prob_pred = calibration_curve(y_t, y_pr, n_bins=10)

            plt.plot(prob_pred, prob_true, marker="o", label=rationale)

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])


def evaluate_probabilistic_predictions(
    y_true: pd.DataFrame, y_prob: pd.DataFrame, rationales: List[str]
) -> pd.DataFrame:
    results = []

    for rationale in rationales:
        # Filter valid samples
        mask = y_true[rationale].notna()
        if mask.sum() == 0:
            continue

        y_t = y_true[rationale][mask].values
        y_p = y_prob[rationale][mask].values

        # Skip if only one class
        if len(np.unique(y_t)) < 2:
            continue

        results.append(
            {
                "rationale": rationale,
                "log_loss": log_loss(y_t, y_p),
                "roc_auc": roc_auc_score(y_t, y_p),
                "avg_precision": average_precision_score(y_t, y_p),
                "n_samples": int(mask.sum()),
                "positive_rate": float(y_t.mean()),
            }
        )

    return pd.DataFrame(results)


def plot_calibration_curves(
    y_true: pd.DataFrame,
    y_prob: pd.DataFrame,
    rationales: List[str],
    n_bins: int = 10,
    figsize: tuple = (15, 10),
):
    n_rationales = len(rationales)
    n_cols = 3
    n_rows = (n_rationales + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rationales > 1 else [axes]

    for i, rationale in enumerate(rationales):
        ax = axes[i]

        # Filter valid samples
        mask = y_true[rationale].notna()
        if mask.sum() == 0 or len(np.unique(y_true[rationale][mask])) < 2:
            ax.text(
                0.5, 0.5, f"Insufficient data\n{rationale}", ha="center", va="center"
            )
            continue

        y_t = y_true[rationale][mask].values
        y_p = y_prob[rationale][mask].values

        # Calculate calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_t, y_p, n_bins=n_bins
        )

        # Plot
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.5)
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            "o-",
            label=rationale,
            linewidth=2,
            markersize=6,
        )

        # Calculate calibration error
        cal_error = np.abs(fraction_of_positives - mean_predicted_value).mean()

        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("True Probability", fontsize=10)
        ax.set_title(f"{rationale}\n(Cal. Error: {cal_error:.3f})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # Remove extra subplots
    for i in range(n_rationales, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    return fig


def plot_probability_distributions(
    y_prob: pd.DataFrame,
    rationales: List[str],
    y_true: pd.DataFrame = None,
    figsize: tuple = (15, 10),
):
    n_rationales = len(rationales)
    n_cols = 3
    n_rows = (n_rationales + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rationales > 1 else [axes]

    for i, rationale in enumerate(rationales):
        ax = axes[i]

        if y_true is not None:
            mask = y_true[rationale].notna()
            if mask.sum() > 0:
                y_t = y_true[rationale][mask].values
                y_p = y_prob[rationale][mask].values

                ax.hist(
                    y_p[y_t == 0],
                    bins=30,
                    alpha=0.5,
                    label="Negative",
                    density=True,
                    color="blue",
                )
                ax.hist(
                    y_p[y_t == 1],
                    bins=30,
                    alpha=0.5,
                    label="Positive",
                    density=True,
                    color="red",
                )
                ax.legend()
        else:
            ax.hist(y_prob[rationale], bins=30, alpha=0.7, density=True, color="green")

        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(rationale, fontsize=11)
        ax.grid(alpha=0.3)

    for i in range(n_rationales, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    return fig


def analyze_prediction_confidence(
    y_prob: pd.DataFrame,
    rationales: List[str],
    thresholds: List[float] = [0.5, 0.7, 0.9],
) -> pd.DataFrame:
    results = []

    for rationale in rationales:
        row = {"rationale": rationale}

        for threshold in thresholds:
            high_conf = (y_prob[rationale] >= threshold).sum()
            pct = high_conf / len(y_prob) * 100
            row[f"prob_>_{threshold}"] = f"{high_conf} ({pct:.1f}%)"

        quantiles = y_prob[rationale].quantile([0.25, 0.50, 0.75, 0.90])
        row["25th_percentile"] = f"{quantiles[0.25]:.3f}"
        row["median"] = f"{quantiles[0.50]:.3f}"
        row["75th_percentile"] = f"{quantiles[0.75]:.3f}"
        row["90th_percentile"] = f"{quantiles[0.90]:.3f}"

        results.append(row)

    return pd.DataFrame(results)
