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
