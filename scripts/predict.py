#!/usr/bin/env python3
"""
Unified prediction script for generating predictions on unlabeled data.

Usage:
    python predict.py --model_dir models/logistic --output_dir results/predictions
    python predict.py --model_dir models/mc_dropout --include_uncertainty
    python predict.py --model_dir models/hierarchical
    python predict.py --model_dir models/pca --output_dir predictions/pca
"""

import argparse
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import DATA_CONFIG, PROJECT_ROOT, ID_COLUMNS, CORE_RATIONALES
from src.data.data_manager import DataManager
from src.pipeline.predictor import Predictor
from src.pipeline.workflow import WorkflowConfig, load_and_filter_data
from src.models.supervised import SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel
from src.models.bnn_model import BNNModel

try:
    from gpytorch.likelihoods import BernoulliLikelihood
    from src.models.gaussian_process import (
        GPModel,
        VariationalGPClassifier,
        DeepKernelGPClassifier,
    )

    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

# ---------- hierarchical model ----------
try:
    from src.models.bayesian_hierarchial import HierarchicalModel

    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HierarchicalModel = None
    HIERARCHICAL_AVAILABLE = False

# ---------- PCA model ----------
try:
    from src.models.pca_model import PCAModel

    PCA_AVAILABLE = True
except ImportError:
    PCAModel = None
    PCA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gp_model(model_path: Path):
    """
    Robust GP loader.
    Supports both:
    1) Pickled GPModel object
    2) GP metadata checkpoint dict saved by GPModel.save(...)
    """
    if not GP_AVAILABLE:
        raise ImportError("GP dependencies unavailable")

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    # Case 1: already a serialized model object
    if isinstance(payload, GPModel):
        return payload

    # Case 2: metadata + state dict checkpoint
    required_keys = {
        "rationale",
        "gp_model_type",
        "kernel_type",
        "model_state",
        "likelihood_state",
    }
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        raise ValueError("Not a GP checkpoint payload")

    hyper = payload.get("hyperparameters", {})

    model = GPModel(
        rationale=payload["rationale"],
        model_type=payload["gp_model_type"],
        kernel_type=payload.get("kernel_type", "rbf"),
        num_inducing=hyper.get("num_inducing", 500),
        learning_rate=hyper.get("learning_rate", 0.01),
        num_epochs=hyper.get("num_epochs", 100),
        batch_size=hyper.get("batch_size", 256),
        hidden_dims=hyper.get("hidden_dims", [64, 32]),
        use_ard=hyper.get("use_ard", True),
        random_seed=hyper.get("random_seed", 21),
    )

    model.feature_names = payload.get("feature_names")
    model.training_losses = payload.get("training_losses", [])

    model_state = payload["model_state"]
    likelihood_state = payload["likelihood_state"]

    inducing_points = model_state.get("variational_strategy.inducing_points")
    if inducing_points is None:
        raise ValueError("GP checkpoint missing inducing points")
    inducing_points = inducing_points.to(model.device)

    if model.gp_model_type == "sparse_gp":
        ard_dims = inducing_points.shape[1] if model.use_ard else None
        model.model = VariationalGPClassifier(
            inducing_points=inducing_points,
            kernel_type=model.kernel_type,
            ard_dims=ard_dims,
        ).to(model.device)
    elif model.gp_model_type == "deep_kernel":
        model.model = DeepKernelGPClassifier(
            input_dim=inducing_points.shape[1],
            inducing_points=inducing_points,
            hidden_dims=model.hidden_dims,
            kernel_type=model.kernel_type,
        ).to(model.device)
    else:
        raise ValueError(f"Unknown GP model type: {model.gp_model_type}")

    model.likelihood = BernoulliLikelihood().to(model.device)
    model.model.load_state_dict(model_state)
    model.likelihood.load_state_dict(likelihood_state)
    model.is_fitted = True

    return model


def _resolve_hierarchical_rationales(model) -> list:
    """
    Extract the real rationale list from a loaded HierarchicalModel.

    The saved model's .rationales attribute is authoritative — UNLESS it was
    accidentally set to the model-type name (e.g. ['hierarchical']), which
    happens when an old pickle was stored with the dict key rather than the
    training target list.  In that case we fall back to CORE_RATIONALES.
    """
    MODEL_TYPE_NAMES = {"hierarchical", "mc_dropout", "bnn", "supervised", "pca"}

    raw = getattr(model, "rationales", None) or []

    # Detect corruption: all entries are model-type placeholders
    if raw and all(r in MODEL_TYPE_NAMES for r in raw):
        print(
            f"  Warning: model.rationales={raw} looks like a model-type placeholder. "
            f"Falling back to CORE_RATIONALES={list(CORE_RATIONALES)}"
        )
        return list(CORE_RATIONALES)

    # Also fall back if the list is empty
    if not raw:
        print(
            f"  Warning: model.rationales is empty. "
            f"Falling back to CORE_RATIONALES={list(CORE_RATIONALES)}"
        )
        return list(CORE_RATIONALES)

    return list(raw)


def load_models(model_dir: Path) -> dict:
    """Load all models from directory, including hierarchical and PCA models."""
    models = {}

    # ---- MC Dropout (single file, multi-label) ----
    mc_dropout_path = model_dir / "mc_dropout_model.pkl"
    if mc_dropout_path.exists():
        models["mc_dropout"] = MCDropoutModel.load(str(mc_dropout_path))
        return models

    # ---- BNN (single file, multi-label) ----
    bnn_path = model_dir / "bnn_model.pkl"
    if bnn_path.exists():
        models["bnn"] = BNNModel.load(str(bnn_path))
        return models

    # ---- Hierarchical Bayesian (single file, multi-label) ----
    # FIX: Use HierarchicalModel.load() instead of raw pickle.load() so the
    # Pyro guide is properly reconstructed and model.rationales is set correctly.
    hierarchical_path = model_dir / "hierarchical_model.pkl"
    if hierarchical_path.exists():
        if HIERARCHICAL_AVAILABLE:
            try:
                model = HierarchicalModel.load(str(hierarchical_path))
                models["hierarchical"] = model
                print(f"  Loaded hierarchical model: rationales={model.rationales}")
                return models
            except Exception as e:
                print(f"  HierarchicalModel.load() failed ({e}), trying raw pickle...")

        # Fallback: raw pickle (older format)
        with open(hierarchical_path, "rb") as f:
            models["hierarchical"] = pickle.load(f)
        print("  Loaded hierarchical model via raw pickle (older format).")
        return models

    # ---- Per-rationale models (supervised / boosting / GP / PCA) ----
    for model_path in sorted(model_dir.glob("*_model.pkl")):
        if model_path.stem in {"mc_dropout_model", "bnn_model", "hierarchical_model"}:
            continue

        rationale = model_path.stem.replace("_model", "")

        # Try GP first
        gp_error = None
        if GP_AVAILABLE:
            try:
                gp_model = _load_gp_model(model_path)
                models[gp_model.rationale] = gp_model
                continue
            except Exception as e:
                gp_error = e

        # Try PCA
        if PCA_AVAILABLE:
            try:
                with open(model_path, "rb") as f:
                    temp = pickle.load(f)
                if isinstance(temp, dict) and "pca" in temp:
                    models[rationale] = PCAModel.load(str(model_path))
                    continue
            except Exception:
                pass

        # Fall back to supervised
        try:
            models[rationale] = SupervisedRationaleModel.load(str(model_path))
        except Exception as supervised_error:
            print(
                f"  Warning: Could not load {model_path.name} as GP "
                f"({gp_error}) or supervised model ({supervised_error})."
            )

    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions on unlabeled data",
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
        help="Directory to save predictions (default: predictions/{model_type})",
    )
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
        "--include_uncertainty",
        action="store_true",
        help="Include uncertainty estimates (hierarchical / MC Dropout)",
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=50,
        help="Number of posterior / MC samples for uncertainty estimation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Process in batches (for large datasets)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="predictions.csv",
        help="Output filename",
    )

    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "predictions" / model_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VOTING RATIONALE PREDICTION")
    print("=" * 80)
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 80 + "\n")

    # ---- Load models ----
    print("Loading models...")
    models = load_models(model_dir)

    if not models:
        print(f"No models found in {model_dir}")
        return

    print(f"Loaded {len(models)} model(s)")

    # ---- Load data manager ----
    data_manager_path = model_dir / "data_manager.pkl"
    if data_manager_path.exists():
        with open(data_manager_path, "rb") as f:
            dm_candidate = pickle.load(f)
        if isinstance(dm_candidate, DataManager):
            data_manager = dm_candidate
            print("Loaded data manager from training")
        else:
            print(
                "Warning: data_manager.pkl did not contain a valid DataManager; "
                "creating a new DataManager."
            )
            data_manager = DataManager()
    else:
        print("Creating new data manager")
        data_manager = DataManager()

    # ---- Load and filter data ----
    print("\nLoading data...")
    workflow = WorkflowConfig(
        data_path=args.data_path,
        min_meetings_rat=args.min_meetings_rat,
        min_dissent=args.min_dissent,
    )
    load_and_filter_data(data_manager, workflow)

    # ---- Detect model type and resolve rationales ----
    first_model = next(iter(models.values()))
    model_key = next(iter(models.keys()))
    is_hierarchical = HIERARCHICAL_AVAILABLE and isinstance(
        first_model, HierarchicalModel
    )

    if is_hierarchical:
        # FIX: Use dedicated resolver that handles corrupted .rationales attrs
        rationales = _resolve_hierarchical_rationales(first_model)
        # Patch the model in-place so downstream code uses the correct list
        first_model.rationales = rationales
    elif hasattr(first_model, "rationales"):
        # Multi-label models: MC Dropout, BNN
        rationales = list(dict.fromkeys(first_model.rationales))
    elif hasattr(first_model, "rationale"):
        # Per-rationale supervised / GP / PCA models
        rationales = [m.rationale for m in models.values() if hasattr(m, "rationale")]
        rationales = list(dict.fromkeys(rationales))
    else:
        # Last resort: use dict keys (will only be right for supervised models)
        rationales = list(models.keys())

    print(f"Rationales: {rationales}")

    # ---- Get unlabeled data ----
    unlabeled_df = data_manager.get_unlabeled_data(rationales)

    if len(unlabeled_df) == 0:
        print("\n⚠️  No unlabeled data found!")
        return

    # ---- Generate predictions ----
    predictor = Predictor(id_columns=ID_COLUMNS, batch_size=args.batch_size)

    if is_hierarchical:
        # FIX: Route hierarchical models directly to predict_hierarchical()
        # instead of the generic predict() which doesn't know about hierarchical indices.
        predictions_df = predictor.predict_hierarchical(
            model=first_model,
            unlabeled_df=unlabeled_df,
            data_manager=data_manager,
            num_samples=args.num_mc_samples,
            include_uncertainty=args.include_uncertainty,
        )
    else:
        predictions_df = predictor.predict(
            models=models,
            unlabeled_df=unlabeled_df,
            data_manager=data_manager,
            include_uncertainty=args.include_uncertainty,
            num_samples=args.num_mc_samples,
        )

    # ---- Analyze ----
    predictor.analyze_predictions(
        predictions_df=predictions_df,
        rationales=rationales,
        output_dir=output_dir,
    )

    # ---- Save ----
    output_path = output_dir / args.output_filename
    predictions_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 80}")
    print("PREDICTION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Predictions saved to: {output_path}")
    print(f"Total predictions: {len(predictions_df):,}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
