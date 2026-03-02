"""
Unified model evaluator for all model types.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.base_model import BaseRationaleModel
from src.models.supervised import SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel
from src.data.data_manager import DataManager

# Optional: extended models (BNN, GP)
try:
    from src.models.bnn_model import BNNModel

    BNN_AVAILABLE = True
except ImportError:
    BNNModel = None
    BNN_AVAILABLE = False

try:
    from src.models.gaussian_process import GPModel

    GP_AVAILABLE = True
except ImportError:
    GPModel = None
    GP_AVAILABLE = False


class ModelEvaluator:
    """Unified interface for evaluating all model types."""

    def __init__(self, save_plots: bool = True):
        self.save_plots = save_plots
        self.results = {}

    def evaluate_supervised(
        self,
        model: SupervisedRationaleModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rationale: str,
    ) -> Dict[str, Any]:
        """Evaluate single supervised model."""

        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_prob, rationale)

        return {
            "rationale": rationale,
            "metrics": metrics,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def evaluate_mc_dropout(
        self,
        model: MCDropoutModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rationales: List[str],
        num_samples: int = 50,
    ) -> Dict[str, Any]:
        """Evaluate MC Dropout model with uncertainty."""

        # Get predictions with uncertainty
        mean_probs, epistemic_unc, total_unc = model.predict_with_uncertainty(
            X_test, num_samples=num_samples
        )
        y_pred = (mean_probs >= 0.5).astype(int)

        # Compute metrics per rationale
        results = {}
        for i, rationale in enumerate(rationales):
            y_true = y_test[:, i]
            y_p = y_pred[:, i]
            y_pr = mean_probs[:, i]

            metrics = self._compute_metrics(y_true, y_p, y_pr, rationale)
            metrics["avg_epistemic_unc"] = float(epistemic_unc[:, i].mean())
            metrics["avg_total_unc"] = float(total_unc[:, i].mean())

            results[rationale] = {
                "metrics": metrics,
                "epistemic_unc": epistemic_unc[:, i],
                "total_unc": total_unc[:, i],
            }

        return {
            "model_type": "mc_dropout",
            "results": results,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": mean_probs,
            "epistemic_unc": epistemic_unc,
            "total_unc": total_unc,
        }

    def _evaluate_uncertainty_model(
        self,
        model: BaseRationaleModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rationales: List[str],
        num_samples: int = 50,
        model_type_name: str = "uncertainty",
    ) -> Dict[str, Any]:
        """Evaluate any model with predict_with_uncertainty (e.g. BNN)."""
        mean_probs, epistemic_unc, total_unc = model.predict_with_uncertainty(
            X_test, num_samples=num_samples
        )
        y_pred = (mean_probs >= 0.5).astype(int)
        results = {}
        for i, rationale in enumerate(rationales):
            y_true = y_test[:, i]
            y_p = y_pred[:, i]
            y_pr = mean_probs[:, i]
            metrics = self._compute_metrics(y_true, y_p, y_pr, rationale)
            metrics["avg_epistemic_unc"] = float(epistemic_unc[:, i].mean())
            metrics["avg_total_unc"] = float(total_unc[:, i].mean())
            results[rationale] = {
                "metrics": metrics,
                "epistemic_unc": epistemic_unc[:, i],
                "total_unc": total_unc[:, i],
            }
        return {
            "model_type": model_type_name,
            "results": results,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": mean_probs,
            "epistemic_unc": epistemic_unc,
            "total_unc": total_unc,
        }

    def evaluate_gp(
        self,
        model: BaseRationaleModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rationale: str,
    ) -> Dict[str, Any]:
        """Evaluate single GP model (per-rationale, supervised-like)."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
        metrics = self._compute_metrics(y_test, y_pred, y_prob, rationale)
        return {
            "rationale": rationale,
            "metrics": metrics,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        rationale: str,
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""

        metrics = {
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "positive_rate": float(y_true.mean()),
        }

        if y_true.sum() == 0:
            return metrics

        metrics.update(
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )

        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except:
            metrics["roc_auc"] = np.nan

        try:
            metrics["avg_precision"] = average_precision_score(y_true, y_prob)
        except:
            metrics["avg_precision"] = np.nan

        try:
            metrics["log_loss"] = log_loss(y_true, y_prob)
        except:
            metrics["log_loss"] = np.nan

        try:
            metrics["brier_score"] = brier_score_loss(y_true, y_prob)
        except:
            metrics["brier_score"] = np.nan

        return metrics

    def _evaluate_model(
        self,
        model: BaseRationaleModel,
        test_df: pd.DataFrame,
        data_manager: DataManager,
        output_dir: Optional[Path] = None,
        num_samples: int = 50,
    ) -> Dict[str, Any]:
        """
        Evaluate any model type.

        Args:
            model: Trained model (any type)
            test_df: Test dataframe
            data_manager: DataManager instance
            output_dir: Directory to save results
            num_samples: MC samples for uncertainty (MC Dropout / BNN only)

        Returns:
            Evaluation results dictionary
        """

        rationales = model.rationales

        # Prepare test data
        test_clean = test_df.dropna(subset=rationales, how="all").copy()
        X_test, y_test, _ = data_manager.prepare_for_training(
            test_clean, rationales, fit=False
        )

        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {model.model_type.upper()}")
        print(f"{'=' * 80}")
        print(f"Test samples: {len(y_test):,}")

        # Evaluate based on model type
        if isinstance(model, MCDropoutModel):
            results = self.evaluate_mc_dropout(
                model, X_test, y_test, rationales, num_samples=num_samples
            )
            self._print_mc_dropout_results(results)
        elif BNN_AVAILABLE and isinstance(model, BNNModel):
            results = self._evaluate_uncertainty_model(
                model,
                X_test,
                y_test,
                rationales,
                num_samples=num_samples,
                model_type_name="bnn",
            )
            self._print_mc_dropout_results(results)
        elif GP_AVAILABLE and isinstance(model, GPModel):
            y_test_single = y_test[:, 0] if y_test.ndim > 1 else y_test
            results = self.evaluate_gp(model, X_test, y_test_single, model.rationale)
            self._print_supervised_results(results)
        elif isinstance(model, SupervisedRationaleModel):
            y_test_single = y_test[:, 0] if y_test.ndim > 1 else y_test
            results = self.evaluate_supervised(
                model, X_test, y_test_single, model.rationale
            )
            self._print_supervised_results(results)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        # Save results
        if output_dir:
            self._save_results(results, output_dir, model.model_type)

        return results

    def _print_supervised_results(self, results: Dict[str, Any]):
        """Print supervised model results."""
        metrics = results["metrics"]
        rationale = results["rationale"]

        print(f"\n{rationale}:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")
        print(f"  F1 Score:      {metrics['f1']:.4f}")
        print(f"  ROC AUC:       {metrics['roc_auc']:.4f}")
        print(f"  Log Loss:      {metrics['log_loss']:.4f}")

    def _print_mc_dropout_results(self, results: Dict[str, Any]):
        """Print MC Dropout results."""
        print(f"\nPer-Rationale Metrics:")
        print(f"{'-' * 80}")

        for rationale, res in results["results"].items():
            metrics = res["metrics"]
            print(f"\n{rationale}:")
            print(
                f"  F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}, "
                f"Unc: {metrics['avg_epistemic_unc']:.4f}"
            )

    def __save_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        model_type: str,
    ):
        """Save evaluation results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics CSV
        if model_type == "mc_dropout" or model_type == "bnn":
            metrics_list = []
            for rationale, res in results["results"].items():
                metric_dict = res["metrics"].copy()
                metric_dict["rationale"] = rationale
                metrics_list.append(metric_dict)

            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        else:
            metrics_df = pd.DataFrame([results["metrics"]])
            metrics_df["rationale"] = results["rationale"]
            metrics_df.to_csv(
                output_dir / f"{results['rationale']}_metrics.csv", index=False
            )

        print(f"\nResults saved to {output_dir}")

    def compare_models(
        self,
        models: Dict[str, BaseRationaleModel],
        test_df: pd.DataFrame,
        data_manager: DataManager,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models: Dict of {model_name: model}
            test_df: Test dataframe
            data_manager: DataManager instance
            output_dir: Directory to save comparison

        Returns:
            Comparison DataFrame
        """

        print(f"\n{'=' * 80}")
        print("MODEL COMPARISON")
        print(f"{'=' * 80}")

        all_results = {}

        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            results = self.evaluate_model(model, test_df, data_manager)
            all_results[name] = results

        # Create comparison table
        comparison_rows = []

        for name, results in all_results.items():
            if "results" in results:  # MC Dropout
                for rationale, res in results["results"].items():
                    comparison_rows.append(
                        {"model": name, "rationale": rationale, **res["metrics"]}
                    )
            else:  # Supervised
                comparison_rows.append(
                    {
                        "model": name,
                        "rationale": results["rationale"],
                        **results["metrics"],
                    }
                )

        comparison_df = pd.DataFrame(comparison_rows)

        print(f"\n{'=' * 80}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 80}")
        print(
            comparison_df[
                ["model", "rationale", "f1", "roc_auc", "log_loss"]
            ].to_string(index=False)
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
            print(f"\nComparison saved to {output_dir / 'model_comparison.csv'}")

        return comparison_df

    def evaluate_predictions(
        self,
        probs: "np.ndarray",  # shape (n_test, n_rationales)
        test_df: "pd.DataFrame",
        rationales: "List[str]",
        output_dir: "Optional[Path]" = None,
        epistemic_std: "Optional[np.ndarray]" = None,  # shape (n_test, n_rationales)
    ) -> "Dict[str, Any]":
        """
        Evaluate a hierarchical Bayesian model given pre-computed probabilities.

        This is the entry-point called by scripts/evaluate.py for the
        hierarchical model path, where feature extraction and inference have
        already been done before calling the evaluator.

        Parameters
        ----------
        probs : ndarray, shape (n_test, n_rationales)
            Posterior-mean predicted probabilities, one column per rationale.
        test_df : DataFrame
            Test set.  Must contain columns named after each rationale so
            ground-truth labels can be extracted.
        rationales : list[str]
            Ordered list of rationale names (matches columns in probs).
        output_dir : Path, optional
            Directory where metrics.csv and a summary print are saved.
        epistemic_std : ndarray, shape (n_test, n_rationales), optional
            Per-sample posterior standard deviation.  When supplied it is
            reported alongside the classification metrics.

        Returns
        -------
        dict with keys
            "model_type"  : "hierarchical"
            "results"     : {rationale: {"metrics": {...}, ...}, ...}
            "y_prob"      : probs
        """
        import numpy as np
        import pandas as pd
        from pathlib import Path

        print(f"\n{'=' * 80}")
        print("HIERARCHICAL BAYESIAN MODEL — PERFORMANCE METRICS")
        print(f"{'=' * 80}")
        print(f"Test samples (total) : {len(test_df):,}")

        results: dict = {}

        for i, rationale in enumerate(rationales):
            # ── ground-truth labels ──────────────────────────────────────────
            if rationale not in test_df.columns:
                print(f"\n  [{rationale}] column not found in test_df – skipping.")
                continue

            # Keep only rows with a non-NaN label for this rationale
            valid_mask = test_df[rationale].notna().values
            y_true = test_df.loc[valid_mask, rationale].values.astype(int)
            y_prob = probs[valid_mask, i]
            y_pred = (y_prob >= 0.5).astype(int)

            if y_true.sum() == 0:
                print(f"\n  [{rationale}] no positive labels – skipping metrics.")
                continue

            # ── classification metrics ───────────────────────────────────────
            metrics = self._compute_metrics(y_true, y_pred, y_prob, rationale)

            # ── epistemic uncertainty (Bayesian bonus) ───────────────────────
            if epistemic_std is not None:
                metrics["avg_epistemic_std"] = float(
                    epistemic_std[valid_mask, i].mean()
                )
                metrics["high_conf_pct"] = float(
                    (epistemic_std[valid_mask, i] < 0.1).mean() * 100
                )

            results[rationale] = {
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

        # ── pretty-print ─────────────────────────────────────────────────────
        self._print_hierarchical_results(results)

        # ── save ─────────────────────────────────────────────────────────────
        if output_dir:
            self._save_results(
                {"model_type": "hierarchical", "results": results},
                output_dir,
                "hierarchical",
            )

        return {
            "model_type": "hierarchical",
            "results": results,
            "y_prob": probs,
        }

    def _print_hierarchical_results(self, results: "Dict[str, Any]") -> None:
        """Pretty-print per-rationale metrics for a hierarchical model."""
        metric_order = [
            ("accuracy", "Accuracy      "),
            ("precision", "Precision     "),
            ("recall", "Recall        "),
            ("f1", "F1 Score      "),
            ("roc_auc", "ROC-AUC       "),
            ("avg_precision", "Avg Precision "),
            ("log_loss", "Log Loss      "),
            ("brier_score", "Brier Score   "),
            ("avg_epistemic_std", "Epistemic Std "),
            ("high_conf_pct", "High-Conf %   "),
        ]

        print(f"\nPer-Rationale Metrics:")
        print(f"{'-' * 80}")

        for rationale, res in results.items():
            m = res["metrics"]
            n_pos = m.get("n_positive", "?")
            n_samp = m.get("n_samples", "?")
            pos_rt = m.get("positive_rate", float("nan"))

            print(f"\n  {rationale}  (n={n_samp:,}, positives={n_pos} / {pos_rt:.1%})")
            for key, label in metric_order:
                if key in m and m[key] is not None:
                    val = m[key]
                    # percentages look better without the ×100 multiplication
                    fmt = f"{val:.1f}%" if key == "high_conf_pct" else f"{val:.4f}"
                    print(f"    {label}: {fmt}")

    def _save_results(
        self,
        results: "Dict[str, Any]",
        output_dir: "Path",
        model_type: str,
    ) -> None:
        """Save evaluation results to disk."""
        import pandas as pd
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Multi-label models (mc_dropout, bnn, hierarchical) share the same
        # nested results structure: results["results"][rationale]["metrics"]
        if model_type in ("mc_dropout", "bnn", "hierarchical"):
            metrics_list = []
            for rationale, res in results["results"].items():
                metric_dict = res["metrics"].copy()
                metric_dict["rationale"] = rationale
                metrics_list.append(metric_dict)

            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(output_dir / "metrics.csv", index=False)
            print(f"\nMetrics saved to {output_dir / 'metrics.csv'}")

        else:
            # Single-rationale supervised model
            metrics_df = pd.DataFrame([results["metrics"]])
            metrics_df["rationale"] = results["rationale"]
            out_path = output_dir / f"{results['rationale']}_metrics.csv"
            metrics_df.to_csv(out_path, index=False)
            print(f"\nResults saved to {out_path}")

    def tune_thresholds(
        self,
        results: dict,
        output_dir=None,
        metric: str = "f1",
        thresholds=None,
    ) -> dict:
        """
        Find the optimal decision threshold for each rationale by sweeping over
        candidate values and maximising `metric` (default: F1).

        Parameters
        ----------
        results : dict
            Output of evaluate_predictions() — expects results["results"][rationale]
            to contain "y_true" and "y_prob".
        output_dir : Path or str, optional
            Where to save threshold_tuning.csv.
        metric : str
            Which metric to maximise: "f1" | "precision" | "recall" | "balanced_accuracy"
        thresholds : array-like, optional
            Candidate thresholds. Defaults to np.arange(0.05, 0.70, 0.01).

        Returns
        -------
        dict  {rationale: {"threshold": float, "f1": float, "precision": float,
                            "recall": float, "balanced_accuracy": float}}
        """
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            balanced_accuracy_score,
        )

        if thresholds is None:
            thresholds = np.arange(0.05, 0.70, 0.01)

        nested = results.get("results", {})
        if not nested:
            print("tune_thresholds: no per-rationale results found.")
            return {}

        print(f"\n{'=' * 80}")
        print(f"THRESHOLD TUNING  (optimising: {metric.upper()})")
        print(f"{'=' * 80}")

        best_thresholds = {}
        rows = []

        for rationale, res in nested.items():
            y_true = res.get("y_true")
            y_prob = res.get("y_prob")

            if y_true is None or y_prob is None:
                print(f"  [{rationale}] missing y_true/y_prob — skipping.")
                continue

            if y_true.sum() == 0:
                print(f"  [{rationale}] no positives — skipping.")
                continue

            best_val, best_t = -1.0, 0.5
            sweep_rows = []

            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)

                f1 = f1_score(y_true, y_pred, zero_division=0)
                pre = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                bal = balanced_accuracy_score(y_true, y_pred)

                sweep_rows.append(
                    {
                        "rationale": rationale,
                        "threshold": round(t, 3),
                        "f1": f1,
                        "precision": pre,
                        "recall": rec,
                        "balanced_accuracy": bal,
                    }
                )

                target = {
                    "f1": f1,
                    "precision": pre,
                    "recall": rec,
                    "balanced_accuracy": bal,
                }.get(metric, f1)
                if target > best_val:
                    best_val, best_t = target, t

            rows.extend(sweep_rows)

            # Metrics at the best threshold
            y_pred_best = (y_prob >= best_t).astype(int)
            best_f1 = f1_score(y_true, y_pred_best, zero_division=0)
            best_pre = precision_score(y_true, y_pred_best, zero_division=0)
            best_rec = recall_score(y_true, y_pred_best, zero_division=0)
            best_bal = balanced_accuracy_score(y_true, y_pred_best)

            # Metrics at the default 0.5 threshold for comparison
            y_pred_05 = (y_prob >= 0.5).astype(int)
            f1_05 = f1_score(y_true, y_pred_05, zero_division=0)

            best_thresholds[rationale] = {
                "threshold": round(float(best_t), 3),
                "f1": round(float(best_f1), 4),
                "precision": round(float(best_pre), 4),
                "recall": round(float(best_rec), 4),
                "balanced_accuracy": round(float(best_bal), 4),
                "f1_at_0.5": round(float(f1_05), 4),
            }

            delta = best_f1 - f1_05
            sign = "+" if delta >= 0 else ""
            print(
                f"\n  {rationale}"
                f"\n    Best threshold : {best_t:.2f}"
                f"\n    F1 @ 0.50      : {f1_05:.4f}"
                f"\n    F1 @ {best_t:.2f}     : {best_f1:.4f}  ({sign}{delta:.4f})"
                f"\n    Precision      : {best_pre:.4f}"
                f"\n    Recall         : {best_rec:.4f}"
                f"\n    Balanced Acc   : {best_bal:.4f}"
            )

        # ── save full sweep CSV ───────────────────────────────────────────────
        if output_dir and rows:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            sweep_df = pd.DataFrame(rows)
            sweep_df.to_csv(out / "threshold_sweep.csv", index=False)

            best_df = pd.DataFrame(best_thresholds).T.reset_index()
            best_df.rename(columns={"index": "rationale"}, inplace=True)
            best_df.to_csv(out / "best_thresholds.csv", index=False)
            print(f"\n  Saved: {out / 'threshold_sweep.csv'}")
            print(f"  Saved: {out / 'best_thresholds.csv'}")

        return best_thresholds
