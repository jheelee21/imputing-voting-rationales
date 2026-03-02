#!/usr/bin/env python3
"""
Unified evaluation script for all model types (original + extended + PCA).

Usage:
    python evaluate.py --model_dir models/logistic
    python evaluate.py --model_dir models/mc_dropout --num_mc_samples 100
    python evaluate.py --model_dir models/bnn --num_mc_samples 100
    python evaluate.py --model_dir models/hierarchical
    python evaluate.py --model_dir models/catboost
    python evaluate.py --model_dir models/pca
"""

import argparse
import sys
import pickle
import numpy as np
from pathlib import Path

# ── sys.path FIRST so every subsequent import resolves correctly ───────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import DATA_CONFIG, RESULTS_DIR, CORE_RATIONALES
from src.data.data_manager import DataManager
from src.pipeline.evaluator import ModelEvaluator
from src.models.supervised import SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel

# Optional: extended models
try:
    from src.models.bnn_model import BNNModel

    BNN_AVAILABLE = True
except ImportError:
    BNNModel = None
    BNN_AVAILABLE = False

try:
    from src.models.calibrated_boosting import CalibratedBoostingModel

    BOOSTING_AVAILABLE = True
except ImportError:
    CalibratedBoostingModel = None
    BOOSTING_AVAILABLE = False

try:
    from src.models.gaussian_process import GPModel

    GP_AVAILABLE = True
except ImportError:
    GPModel = None
    GP_AVAILABLE = False

try:
    from src.models.pca_model import PCAModel

    PCA_AVAILABLE = True
except ImportError:
    PCAModel = None
    PCA_AVAILABLE = False

try:
    from src.models.bayesian_hierarchial import HierarchicalModel

    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HierarchicalModel = None
    HIERARCHICAL_AVAILABLE = False

# Import the rationale resolver from predict.py (same project)
try:
    from scripts.predict import _resolve_hierarchical_rationales
except ImportError:
    # Inline fallback in case of import issues
    def _resolve_hierarchical_rationales(model):
        MODEL_TYPE_NAMES = {"hierarchical", "mc_dropout", "bnn", "supervised", "pca"}
        raw = getattr(model, "rationales", None) or []
        if not raw or all(r in MODEL_TYPE_NAMES for r in raw):
            print(
                f"  Warning: model.rationales={raw!r} looks like a placeholder. "
                f"Falling back to CORE_RATIONALES={list(CORE_RATIONALES)}"
            )
            return list(CORE_RATIONALES)
        return list(raw)


# ── helpers ────────────────────────────────────────────────────────────────────


def _is_hierarchical_model(model) -> bool:
    """
    Robust hierarchical-model detection that survives class-identity mismatches
    caused by import-order differences (the previous bug: HierarchicalModel was
    imported before sys.path.insert, so isinstance could return False even for a
    genuine HierarchicalModel instance).
    """
    # 1. Ideal path: isinstance works
    if HIERARCHICAL_AVAILABLE and isinstance(model, HierarchicalModel):
        return True
    # 2. Fallback: check the model_type attribute string
    model_type_str = getattr(model, "model_type", "") or ""
    return "hierarchical" in model_type_str.lower()


def _ensure_data_loaded(data_manager: DataManager, args) -> None:
    """
    If the DataManager was freshly constructed (self.df is None), load the CSV
    and apply project-level filters so that split_data() can work.
    """
    if data_manager.df is not None:
        return  # already loaded (e.g. restored from data_manager.pkl)

    print(f"  Loading data from {args.data_path} …")
    data_manager.load_data(args.data_path)
    data_manager.apply_filters(
        min_meetings_rat=args.min_meetings_rat,
        min_dissent=args.min_dissent,
    )
    print(f"  {len(data_manager.df):,} rows after filtering.")


# ── model loader ───────────────────────────────────────────────────────────────


def load_models(model_dir: Path):
    """
    Load all models from directory.
    Returns (models dict, model_type str).
    """
    models = {}
    model_type = None

    if not model_dir.exists():
        print(f"ERROR: Directory {model_dir} does not exist!")
        return models, model_type

    # ---- MC Dropout ----
    mc_dropout_path = model_dir / "mc_dropout_model.pkl"
    if mc_dropout_path.exists():
        models["mc_dropout"] = MCDropoutModel.load(str(mc_dropout_path))
        return models, "mc_dropout"

    # ---- BNN ----
    bnn_path = model_dir / "bnn_model.pkl"
    if bnn_path.exists() and BNN_AVAILABLE:
        models["bnn"] = BNNModel.load(str(bnn_path))
        return models, "bnn"

    # ---- Hierarchical Bayesian ----
    hierarchical_path = model_dir / "hierarchical_model.pkl"
    if hierarchical_path.exists() and HIERARCHICAL_AVAILABLE:
        try:
            model = HierarchicalModel.load(str(hierarchical_path))
            models["hierarchical"] = model
            print(f"  Loaded hierarchical model: rationales={model.rationales}")
            return models, "hierarchical"
        except Exception as e:
            print(f"  HierarchicalModel.load() failed ({e}), trying raw pickle…")
            with open(hierarchical_path, "rb") as f:
                models["hierarchical"] = pickle.load(f)
            return models, "hierarchical"

    # ---- PCA ----
    pca_path = model_dir / "pca_model.pkl"
    if pca_path.exists() and PCA_AVAILABLE:
        models["pca"] = PCAModel.load(str(pca_path))
        return models, "pca"

    # ---- Per-rationale supervised / boosting / GP ----
    for model_path in sorted(model_dir.glob("*_model.pkl")):
        if model_path.stem in {
            "mc_dropout_model",
            "bnn_model",
            "hierarchical_model",
            "pca_model",
        }:
            continue

        rationale = model_path.stem.replace("_model", "")
        try:
            # GP
            if GP_AVAILABLE:
                try:
                    from scripts.predict import _load_gp_model

                    gp_model = _load_gp_model(model_path)
                    models[gp_model.rationale] = gp_model
                    model_type = "supervised"
                    continue
                except Exception:
                    pass

            # PCA dict format
            if PCA_AVAILABLE:
                try:
                    with open(model_path, "rb") as f:
                        temp = pickle.load(f)
                    if isinstance(temp, dict) and "pca" in temp:
                        models[rationale] = PCAModel.load(str(model_path))
                        model_type = "pca"
                        continue
                except Exception:
                    pass

            # Boosting
            if BOOSTING_AVAILABLE:
                try:
                    models[rationale] = CalibratedBoostingModel.load(str(model_path))
                    model_type = "supervised"
                    continue
                except Exception:
                    pass

            # Supervised fallback
            models[rationale] = SupervisedRationaleModel.load(str(model_path))
            model_type = "supervised"

        except Exception as e:
            print(f"Warning: Could not load {model_path.name}: {e}")

    return models, model_type or "supervised"


# ── main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate voting rationale prediction models (all types)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/{model_dir_name})",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_CONFIG["data_path"],
        help="Path to data CSV file",
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
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=50,
        help="Posterior samples for uncertainty estimation",
    )
    parser.add_argument(
        "--no_plots", action="store_true", help="Disable plot generation"
    )

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else RESULTS_DIR / model_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VOTING RATIONALE MODEL EVALUATION")
    print("=" * 80)
    print(f"Model dir : {model_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 80 + "\n")

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading models…")
    models, model_type = load_models(model_dir)
    if not models:
        print(f"No models found in {model_dir}")
        return
    print(f"Loaded {len(models)} model(s): {list(models.keys())} (type: {model_type})")

    # ── data manager ──────────────────────────────────────────────────────────
    data_manager_path = model_dir / "data_manager.pkl"
    if data_manager_path.exists():
        with open(data_manager_path, "rb") as f:
            dm_candidate = pickle.load(f)
        if isinstance(dm_candidate, DataManager):
            data_manager = dm_candidate
            print("Loaded data manager from training.")
        else:
            print(
                "Warning: data_manager.pkl did not contain a valid DataManager; "
                "creating a new one."
            )
            data_manager = DataManager()
    else:
        print("No data_manager.pkl found; creating a new DataManager.")
        data_manager = DataManager()

    # ── FIX #1: ensure the DataManager has data loaded ───────────────────────
    _ensure_data_loaded(data_manager, args)

    # ── resolve rationales ────────────────────────────────────────────────────
    first_model = next(iter(models.values()))

    # ── FIX #2: use string-based detection so class-identity mismatches don't
    #            bypass the resolver and let ['hierarchical'] leak through ──────
    if _is_hierarchical_model(first_model):
        rationales = _resolve_hierarchical_rationales(first_model)
        first_model.rationales = rationales  # patch in-place
    elif (
        len(models) == 1
        and hasattr(first_model, "rationales")
        and isinstance(first_model.rationales, list)
    ):
        rationales = first_model.rationales
    else:
        rationales = list(models.keys())

    print(f"Rationales: {rationales}")

    # ── test split ────────────────────────────────────────────────────────────
    _, test_df = data_manager.split_data(
        test_size=args.test_size,
        random_seed=args.random_seed,
        rationales=rationales,
    )
    print(f"Test samples: {len(test_df):,}")

    # ── evaluate ──────────────────────────────────────────────────────────────
    evaluator = ModelEvaluator(save_plots=not args.no_plots)

    if model_type in ("mc_dropout", "bnn", "hierarchical"):
        model = next(iter(models.values()))

        if model_type == "hierarchical":
            print("\nUsing hierarchical evaluation pipeline…")

            # prepare_for_hierarchical returns:
            #   (X, y, investor_idx, firm_idx, year_idx, feature_names)
            # We pass fit=False so the encoders/scalers fitted at training time
            # (stored on data_manager) are reused without re-fitting.
            # If the restored DataManager has no training artifacts (fresh fallback),
            # we seed it from the model's saved feature_names.
            if not hasattr(data_manager, "_training_numerical_cols") and getattr(
                model, "feature_names", None
            ):
                data_manager._training_numerical_cols = [
                    c for c in model.feature_names if not c.endswith("_encoded")
                ]
                data_manager._training_categorical_cols = [
                    c for c in model.feature_names if c.endswith("_encoded")
                ]

            X_test, _, investor_idx, firm_idx, year_idx, _ = (
                data_manager.prepare_for_hierarchical(
                    test_df,
                    rationales,
                    fit=False,
                )
            )

            # Clip indices to avoid out-of-bounds for unseen entities.
            # The counts live on the inner BayesianHierarchicalLogistic (model.model);
            # fall back to the max observed index + 1 if the inner model is missing.
            inner = getattr(model, "model", None)
            n_investors = getattr(inner, "n_investors", int(investor_idx.max()) + 1)
            n_firms = getattr(inner, "n_firms", int(firm_idx.max()) + 1)
            n_years = getattr(inner, "n_years", int(year_idx.max()) + 1)
            investor_idx = np.clip(investor_idx, 0, n_investors - 1)
            firm_idx = np.clip(firm_idx, 0, n_firms - 1)
            year_idx = np.clip(year_idx, 0, n_years - 1)

            # Use predict_with_uncertainty to get both mean probs and epistemic std
            mean_probs, epistemic_std = model.predict_with_uncertainty(
                X=X_test,
                investor_idx=investor_idx,
                firm_idx=firm_idx,
                year_idx=year_idx,
                num_samples=args.num_mc_samples,
            )

            results = evaluator.evaluate_predictions(
                probs=mean_probs,
                test_df=test_df,
                rationales=rationales,
                output_dir=output_dir,
                epistemic_std=epistemic_std,
            )

            # ── Threshold tuning ──────────────────────────────────────────────
            # Sweeps thresholds 0.05–0.70 and finds the value that maximises F1
            # for each rationale. Saves threshold_sweep.csv + best_thresholds.csv.
            best_thresholds = evaluator.tune_thresholds(
                results=results,
                output_dir=output_dir,
                metric="f1",
            )

        else:
            print("Using standard multi-label evaluation…")
            results = evaluator.evaluate_model(
                model=model,
                test_df=test_df,
                data_manager=data_manager,
                output_dir=output_dir,
                num_samples=args.num_mc_samples,
            )

    else:
        # Per-rationale classical models
        if len(models) == 1:
            results = evaluator.evaluate_model(
                model=first_model,
                test_df=test_df,
                data_manager=data_manager,
                output_dir=output_dir,
            )
        else:
            results = evaluator.compare_models(
                models=models,
                test_df=test_df,
                data_manager=data_manager,
                output_dir=output_dir,
            )

    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
