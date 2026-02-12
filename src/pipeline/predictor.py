"""
Unified predictor for generating predictions on unlabeled data.
Updated to support semi-supervised models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.models.base_model import BaseRationaleModel
from src.models.supervised import SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel
from src.data.data_manager import DataManager
from src.models.bnn_model import BNNModel
from configs.config import CATEGORICAL_IDS

try:
    from src.models.bayesian_hierarchial import HierarchicalModel

    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False


try:
    from src.models.gaussian_process import GPModel

    GP_AVAILABLE = True
except ImportError:
    GPModel = None
    GP_AVAILABLE = False

# Import semi-supervised models if available
try:
    from src.models.semi_supervised import (
        PseudoLabelingSemiSupervised,
        CoTrainingSemiSupervised,
    )

    SEMI_SUPERVISED_AVAILABLE = True
except ImportError:
    PseudoLabelingSemiSupervised = None
    CoTrainingSemiSupervised = None
    SEMI_SUPERVISED_AVAILABLE = False


class Predictor:
    """Unified interface for making predictions with any model type."""

    def __init__(
        self,
        id_columns: List[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.id_columns = id_columns or CATEGORICAL_IDS
        self.batch_size = batch_size

    @staticmethod
    def _align_features_to_model(
        X: np.ndarray,
        current_feature_names: List[str],
        expected_feature_names: Optional[List[str]],
    ) -> np.ndarray:
        """Align feature matrix columns to a model's expected feature order."""
        if not expected_feature_names:
            return X

        if list(current_feature_names) == list(expected_feature_names):
            return X

        name_to_idx = {name: idx for idx, name in enumerate(current_feature_names)}
        aligned = np.zeros((X.shape[0], len(expected_feature_names)), dtype=float)

        missing = []
        for j, name in enumerate(expected_feature_names):
            idx = name_to_idx.get(name)
            if idx is None:
                missing.append(name)
                continue
            aligned[:, j] = X[:, idx]

        if missing:
            print(
                f"Warning: {len(missing)} expected features missing at inference; "
                "filled with zeros. Example: "
                f"{missing[:5]}"
            )

        extra = set(current_feature_names) - set(expected_feature_names)
        if extra:
            print(
                f"Note: Dropping {len(extra)} extra inference features not used by model."
            )

        return aligned

    def predict_single_label_models(
        self,
        models: Dict[str, SupervisedRationaleModel],
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
    ) -> pd.DataFrame:
        """
        Generate predictions using supervised models (one per rationale).
        Works for both regular supervised and semi-supervised models.

        Args:
            models: Dict of {rationale: model}
            unlabeled_df: Dataframe with unlabeled observations
            data_manager: DataManager instance

        Returns:
            DataFrame with predictions
        """

        # Start with ID columns
        existing_ids = [c for c in self.id_columns if c in unlabeled_df.columns]
        predictions_df = unlabeled_df[existing_ids].copy()

        print(f"\n{'=' * 80}")
        print("GENERATING PREDICTIONS (separate models)")
        print(f"{'=' * 80}")
        print(f"Unlabeled samples: {len(unlabeled_df):,}")
        print(f"Rationales: {list(models.keys())}")
        print(f"{'=' * 80}\n")

        for rationale, model in models.items():
            print(f"Predicting {rationale}...", end=" ", flush=True)

            try:
                X, _, feature_names = data_manager.prepare_for_inference(
                    unlabeled_df, [rationale]
                )
                X = self._align_features_to_model(
                    X,
                    feature_names,
                    getattr(model, "feature_names", None),
                )

                # Predict
                y_prob = model.predict_proba(X)
                predictions_df[f"{rationale}_prob"] = y_prob

                print(f"✓ (mean: {y_prob.mean():.3f}, std: {y_prob.std():.3f})")

            except Exception as e:
                print(f"✗ Error: {e}")
                predictions_df[f"{rationale}_prob"] = np.nan

        return predictions_df

    def predict_multi_label_model(
        self,
        model,
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
        with_uncertainty: bool = False,
    ) -> pd.DataFrame:
        rationales = model.rationales
        existing_ids = [c for c in self.id_columns if c in unlabeled_df.columns]
        predictions_df = unlabeled_df[existing_ids].copy()

        print(f"\n{'=' * 80}")
        print("GENERATING PREDICTIONS (multi-label model)")
        print(f"{'=' * 80}")
        print(f"Unlabeled samples: {len(unlabeled_df):,}")
        print(f"Rationales: {rationales}")
        print(f"{'=' * 80}\n")

        X, _, _ = data_manager.prepare_for_training(unlabeled_df, rationales, fit=False)

        if with_uncertainty:
            mean_probs = model.predict_proba_with_uncertainty(X)
        else:
            mean_probs = model.predict_proba(X)

        for i, rationale in enumerate(rationales):
            predictions_df[f"{rationale}_prob"] = mean_probs[:, i]
            print(f"{rationale}: mean={mean_probs[:, i].mean():.3f}")
        return predictions_df

    def predict_mc_dropout(
        self,
        model: MCDropoutModel,
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
        include_uncertainty: bool = True,
        num_samples: int = 50,
    ) -> pd.DataFrame:
        """
        Generate predictions using MC Dropout model.

        Args:
            model: Trained MC Dropout model
            unlabeled_df: Dataframe with unlabeled observations
            data_manager: DataManager instance
            include_uncertainty: Whether to include uncertainty estimates
            num_samples: Number of MC samples

        Returns:
            DataFrame with predictions and uncertainties
        """

        rationales = model.rationales

        # Start with ID columns
        existing_ids = [c for c in self.id_columns if c in unlabeled_df.columns]
        predictions_df = unlabeled_df[existing_ids].copy()

        print(f"\n{'=' * 80}")
        print("GENERATING PREDICTIONS (MC Dropout)")
        print(f"{'=' * 80}")
        print(f"Unlabeled samples: {len(unlabeled_df):,}")
        print(f"Rationales: {rationales}")
        print(f"MC samples: {num_samples}")
        print(f"{'=' * 80}\n")

        # Prepare features
        X, _, _ = data_manager.prepare_for_training(unlabeled_df, rationales, fit=False)

        # Get predictions with uncertainty
        if include_uncertainty:
            mean_probs, epistemic_unc, total_unc = model.predict_with_uncertainty(
                X, num_samples=num_samples
            )

            # Add probabilities and uncertainties
            for i, rationale in enumerate(rationales):
                predictions_df[f"{rationale}_prob"] = mean_probs[:, i]
                predictions_df[f"{rationale}_epistemic_unc"] = epistemic_unc[:, i]
                predictions_df[f"{rationale}_total_unc"] = total_unc[:, i]

                print(f"{rationale}:")
                print(
                    f"  Prob: mean={mean_probs[:, i].mean():.3f}, std={mean_probs[:, i].std():.3f}"
                )
                print(f"  Unc:  mean={epistemic_unc[:, i].mean():.3f}")

        else:
            mean_probs = model.predict_proba(X, num_samples=num_samples)

            for i, rationale in enumerate(rationales):
                predictions_df[f"{rationale}_prob"] = mean_probs[:, i]
                print(f"{rationale}: mean={mean_probs[:, i].mean():.3f}")

        return predictions_df

    def predict_bnn(
        self,
        model: BNNModel,
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate predictions using Bayesian Neural Network model.
        """
        rationales = model.rationales

        # Start with ID columns
        existing_ids = [c for c in self.id_columns if c in unlabeled_df.columns]
        predictions_df = unlabeled_df[existing_ids].copy()

        print(f"\n{'=' * 80}")
        print("GENERATING PREDICTIONS (Bayesian Neural Network)")
        print(f"{'=' * 80}")
        print(f"Unlabeled samples: {len(unlabeled_df):,}")
        print(f"Rationales: {rationales}")
        print(f"{'=' * 80}\n")

        # Prepare features
        X, _, _ = data_manager.prepare_for_training(unlabeled_df, rationales, fit=False)

        # Get predictions with uncertainty
        mean_probs, epistemic_unc, total_unc = model.predict_with_uncertainty(X)

        # Add probabilities and uncertainties
        for i, rationale in enumerate(rationales):
            predictions_df[f"{rationale}_prob"] = mean_probs[:, i]
            predictions_df[f"{rationale}_epistemic_unc"] = epistemic_unc[:, i]
            predictions_df[f"{rationale}_total_unc"] = total_unc[:, i]

            print(f"{rationale}:")
            print(
                f"  Prob: mean={mean_probs[:, i].mean():.3f}, std={mean_probs[:, i].std():.3f}"
            )
            print(f"  Unc:  mean={epistemic_unc[:, i].mean():.3f}")

        return predictions_df

    def predict_gp(
        self,
        models: Dict[str, GPModel],
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
        include_uncertainty: bool = False,
    ) -> pd.DataFrame:
        """
        Generate predictions using per-rationale Gaussian Process models.

        Args:
            models: Dict of {rationale: GPModel}
            unlabeled_df: Dataframe with unlabeled observations
            data_manager: DataManager instance
            include_uncertainty: Whether to include GP predictive std

        Returns:
            DataFrame with probabilities (and optional predictive std columns)
        """

        existing_ids = [c for c in self.id_columns if c in unlabeled_df.columns]
        predictions_df = unlabeled_df[existing_ids].copy()

        print(f"\n{'=' * 80}")
        print("GENERATING PREDICTIONS (Gaussian Process)")
        print(f"{'=' * 80}")
        print(f"Unlabeled samples: {len(unlabeled_df):,}")
        print(f"Rationales: {list(models.keys())}")
        print(f"Include uncertainty: {include_uncertainty}")
        print(f"{'=' * 80}\n")

        for rationale, model in models.items():
            print(f"Predicting {rationale}...", end=" ", flush=True)

            try:
                X, _, _ = data_manager.prepare_for_training(
                    unlabeled_df, [rationale], fit=False
                )

                if include_uncertainty:
                    y_prob, y_std = model.predict_with_uncertainty(X)
                    predictions_df[f"{rationale}_prob"] = y_prob
                    predictions_df[f"{rationale}_pred_std"] = y_std
                    print(
                        f"✓ (mean: {y_prob.mean():.3f}, std: {y_prob.std():.3f}, "
                        f"pred_std_mean: {y_std.mean():.3f})"
                    )
                else:
                    y_prob = model.predict_proba(X)
                    predictions_df[f"{rationale}_prob"] = y_prob
                    print(f"✓ (mean: {y_prob.mean():.3f}, std: {y_prob.std():.3f})")

            except Exception as e:
                print(f"✗ Error: {e}")
                predictions_df[f"{rationale}_prob"] = np.nan
                if include_uncertainty:
                    predictions_df[f"{rationale}_pred_std"] = np.nan

        return predictions_df

    def predict_hierarchical(
        self,
        model: HierarchicalModel,
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
        num_samples: Optional[int] = None,
        include_uncertainty: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate predictions using Hierarchical Bayesian model.

        Args:
            model: Trained HierarchicalModel
            unlabeled_df: Dataframe with unlabeled observations
            data_manager: DataManager instance
            num_samples: Number of posterior samples (default: use model's num_samples)
            include_uncertainty: Whether to compute epistemic uncertainty (not yet implemented)

        Returns:
            DataFrame with predictions
        """
        if not HIERARCHICAL_AVAILABLE:
            raise ImportError("Hierarchical model not available")

        rationales = model.rationales

        # Start with ID columns
        existing_ids = [c for c in self.id_columns if c in unlabeled_df.columns]
        predictions_df = unlabeled_df[existing_ids].copy()

        print(f"\n{'=' * 80}")
        print("GENERATING PREDICTIONS (Hierarchical Bayesian)")
        print(f"{'=' * 80}")
        print(f"Unlabeled samples: {len(unlabeled_df):,}")
        print(f"Rationales: {rationales}")
        print(f"MC samples: {num_samples or model.num_samples}")
        print(f"{'=' * 80}\n")

        # If DataManager artifacts are unavailable, align to the feature space
        # expected by the saved model when feature names are present.
        if (
            not hasattr(data_manager, "_training_numerical_cols")
            and not hasattr(data_manager, "_training_categorical_cols")
            and getattr(model, "feature_names", None)
        ):
            data_manager._training_numerical_cols = [
                c for c in model.feature_names if not c.endswith("_encoded")
            ]
            data_manager._training_categorical_cols = [
                c for c in model.feature_names if c.endswith("_encoded")
            ]

        # Prepare features with hierarchical indices
        X, _, inv_idx, firm_idx, year_idx, _ = data_manager.prepare_for_hierarchical(
            unlabeled_df,
            rationales,
            fit=False,  # Use fitted encoders from training
        )

        print(f"Feature shape: {X.shape}")
        print(f"Unique investors: {np.unique(inv_idx).size}")
        print(f"Unique firms: {np.unique(firm_idx).size}")
        print(f"Unique years: {np.unique(year_idx).size}\n")

        expected_dim = getattr(getattr(model, "model", None), "input_dim", None)
        if expected_dim is not None and X.shape[1] != expected_dim:
            raise ValueError(
                "Feature dimension mismatch for hierarchical prediction: "
                f"got {X.shape[1]}, expected {expected_dim}. "
                "Ensure the training DataManager artifacts or model feature_names are available."
            )

        # Guard against out-of-range hierarchical indices when inference encoders
        # are bootstrapped from prediction data rather than training artifacts.
        expected_investors = getattr(getattr(model, "model", None), "n_investors", None)
        expected_firms = getattr(getattr(model, "model", None), "n_firms", None)
        expected_years = getattr(getattr(model, "model", None), "n_years", None)

        def remap_oob(
            indices: np.ndarray, upper_bound: Optional[int], label: str
        ) -> np.ndarray:
            if upper_bound is None:
                return indices
            oob_mask = (indices < 0) | (indices >= upper_bound)
            if np.any(oob_mask):
                print(
                    f"Warning: {oob_mask.sum():,} {label} indices out of bounds for trained model "
                    f"(size={upper_bound}); remapping to 0."
                )
                fixed = indices.copy()
                fixed[oob_mask] = 0
                return fixed
            return indices

        inv_idx = remap_oob(inv_idx, expected_investors, "investor")
        firm_idx = remap_oob(firm_idx, expected_firms, "firm")
        year_idx = remap_oob(year_idx, expected_years, "year")

        # Get predictions
        mean_probs = model.predict_proba(
            X=X,
            investor_idx=inv_idx,
            firm_idx=firm_idx,
            year_idx=year_idx,
            num_samples=num_samples,
        )

        # Add probabilities
        for i, rationale in enumerate(rationales):
            predictions_df[f"{rationale}_prob"] = mean_probs[:, i]
            print(
                f"{rationale}: mean={mean_probs[:, i].mean():.3f}, "
                f"std={mean_probs[:, i].std():.3f}, "
                f"min={mean_probs[:, i].min():.3f}, "
                f"max={mean_probs[:, i].max():.3f}"
            )

        # TODO: Add uncertainty quantification
        # For now, we only return mean predictions
        # Future: Can extract epistemic uncertainty from posterior samples
        if include_uncertainty:
            print(
                "\nNote: Uncertainty quantification not yet implemented for hierarchical model"
            )

        return predictions_df

    def predict(
        self,
        models: Dict[str, Any],
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
        include_uncertainty: bool = False,
        num_samples: int = 50,
    ) -> pd.DataFrame:
        """
        Generate predictions using any model type (including hierarchical).

        This is an updated version of predict that supports the hierarchical model.
        You can either replace your existing predict method or keep both.

        Args:
            models: Dict of trained models
            unlabeled_df: Dataframe with unlabeled observations
            data_manager: DataManager instance
            include_uncertainty: Whether to include uncertainty estimates
            num_samples: Number of MC samples for uncertainty estimation

        Returns:
            DataFrame with predictions
        """
        # Detect model type
        if len(models) == 1:
            model_key = list(models.keys())[0]
            model = models[model_key]

            # Check if it's a hierarchical model
            if HIERARCHICAL_AVAILABLE and isinstance(model, HierarchicalModel):
                return self.predict_hierarchical(
                    model=model,
                    unlabeled_df=unlabeled_df,
                    data_manager=data_manager,
                    num_samples=num_samples,
                    include_uncertainty=include_uncertainty,
                )

            # Check if it's a BNN model
            elif hasattr(model, "__class__") and "BNNModel" in model.__class__.__name__:
                return self.predict_bnn(
                    model=model,
                    unlabeled_df=unlabeled_df,
                    data_manager=data_manager,
                    include_uncertainty=include_uncertainty,
                    num_samples=num_samples,
                )

            # Check if it's an MC Dropout model
            elif (
                hasattr(model, "__class__")
                and "MCDropoutModel" in model.__class__.__name__
            ):
                return self.predict_mc_dropout(
                    model=model,
                    unlabeled_df=unlabeled_df,
                    data_manager=data_manager,
                    include_uncertainty=include_uncertainty,
                    num_samples=num_samples,
                )

            # Check if it's a GP model
            elif hasattr(model, "__class__") and "GPModel" in model.__class__.__name__:
                return self.predict_gp(
                    models={model_key: model},
                    unlabeled_df=unlabeled_df,
                    data_manager=data_manager,
                    include_uncertainty=include_uncertainty,
                )

        # Multi-rationale GP models
        if all(
            hasattr(m, "__class__") and "GPModel" in m.__class__.__name__
            for m in models.values()
        ):
            return self.predict_gp(
                models=models,
                unlabeled_df=unlabeled_df,
                data_manager=data_manager,
                include_uncertainty=include_uncertainty,
            )

        # Default: single-label models (one per rationale)
        return self.predict_single_label_models(
            models=models,
            unlabeled_df=unlabeled_df,
            data_manager=data_manager,
        )

    def _predict(
        self,
        models: Dict[str, BaseRationaleModel],
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
        include_uncertainty: bool = False,
        num_samples: int = 50,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generate predictions using any model type.

        Args:
            models: Dict of models (either {rationale: model} or {'mc_dropout': model})
            unlabeled_df: Dataframe with unlabeled observations
            data_manager: DataManager instance
            **kwargs: Additional arguments for specific model types

        Returns:
            DataFrame with predictions
        """
        # Single model: could be MC Dropout, BNN, or a single supervised/PCA model
        if len(models) == 1:
            first_model = next(iter(models.values()))

            if isinstance(first_model, MCDropoutModel):
                return self.predict_mc_dropout(
                    model=first_model,
                    unlabeled_df=unlabeled_df,
                    data_manager=data_manager,
                    include_uncertainty=include_uncertainty,
                    num_samples=num_samples,
                )

            if isinstance(first_model, BNNModel):
                return self.predict_bnn(
                    model=first_model,
                    unlabeled_df=unlabeled_df,
                    data_manager=data_manager,
                )

        elif GP_AVAILABLE and all(isinstance(m, GPModel) for m in models.values()):
            return self.predict_gp(
                models=models,
                unlabeled_df=unlabeled_df,
                data_manager=data_manager,
                include_uncertainty=kwargs.get("include_uncertainty", False),
            )

        if len(models) == 1 and isinstance(first_model, SupervisedRationaleModel):
            return self.predict_supervised(
                models=models,
                unlabeled_df=unlabeled_df,
                data_manager=data_manager,
            )

        # Multiple entries: assume per‑rationale supervised-style models
        return self.predict_single_label_models(
            models=models,
            unlabeled_df=unlabeled_df,
            data_manager=data_manager,
        )

    def analyze_predictions(
        self,
        predictions_df: pd.DataFrame,
        rationales: List[str],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze prediction statistics.

        Args:
            predictions_df: DataFrame with predictions
            rationales: List of rationales
            output_dir: Directory to save analysis

        Returns:
            Dictionary of analysis results
        """

        print(f"\n{'=' * 80}")
        print("PREDICTION ANALYSIS")
        print(f"{'=' * 80}\n")

        # Confidence statistics
        confidence_stats = []

        for rationale in rationales:
            prob_col = f"{rationale}_prob"

            if prob_col not in predictions_df.columns:
                continue

            probs = predictions_df[prob_col].dropna()

            confidence_stats.append(
                {
                    "rationale": rationale,
                    "n_predictions": len(probs),
                    "mean": probs.mean(),
                    "std": probs.std(),
                    "min": probs.min(),
                    "q25": probs.quantile(0.25),
                    "median": probs.quantile(0.50),
                    "q75": probs.quantile(0.75),
                    "max": probs.max(),
                    "pct_high_conf": (probs >= 0.7).mean() * 100,
                    "pct_medium_conf": ((probs >= 0.3) & (probs < 0.7)).mean() * 100,
                    "pct_low_conf": (probs < 0.3).mean() * 100,
                }
            )

        confidence_df = pd.DataFrame(confidence_stats)

        print("Confidence Statistics:")
        print(
            confidence_df[["rationale", "mean", "std", "median"]].to_string(index=False)
        )

        # Multi-label analysis (at 0.5 threshold)
        prob_cols = [
            f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns
        ]

        if prob_cols:
            binary_preds = (predictions_df[prob_cols] >= 0.5).astype(int)
            n_rationales_per_obs = binary_preds.sum(axis=1)

            multi_label_stats = {
                "total_observations": len(predictions_df),
            }

            for n in range(5):
                confident_obs = (
                    (
                        (predictions_df[prob_cols] >= 0.7)
                        | (predictions_df[prob_cols] <= 0.3)
                    )
                    .astype(int)
                    .sum(axis=1)
                )
                count = (n_rationales_per_obs == n).sum()
                pct = count / len(predictions_df) * 100
                multi_label_stats[f"n_with_{n}_rationales"] = int(count)
                multi_label_stats[f"pct_with_{n}_rationales"] = pct
                multi_label_stats["confident_observations"] = (
                    confident_obs.sum() / len(predictions_df) * 100
                )

            print(f"\nMulti-Label Distribution:")
            for n in range(5):
                count = multi_label_stats[f"n_with_{n}_rationales"]
                pct = multi_label_stats[f"pct_with_{n}_rationales"]
                conf_obs = multi_label_stats["confident_observations"]
                print(
                    f"  {n} rationales: {count:,} ({pct:.1f}%, confident: {conf_obs:.1f}%)"
                )

        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            confidence_df.to_csv(output_dir / "confidence_statistics.csv", index=False)
            print(f"\nStatistics saved to {output_dir}")

        print("To visualize, run:")
        print(
            f"  python scripts/visualise_predictions.py --pred_dir predictions/{output_dir.name}"
        )

        return {
            "confidence": confidence_df,
            "multi_label": multi_label_stats if prob_cols else None,
        }
