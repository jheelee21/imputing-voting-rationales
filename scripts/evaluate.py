#!/usr/bin/env python3
"""
Unified evaluation script for all model types (original + extended).

Usage:
    python evaluate.py --model_dir models/logistic
    python evaluate.py --model_dir models/mc_dropout --num_mc_samples 100
    python evaluate.py --model_dir models/bnn --num_mc_samples 100
    python evaluate.py --model_dir models/catboost
    python evaluate.py --model_dir models/sparse_gp
"""

import argparse
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import DATA_CONFIG, RESULTS_DIR, CORE_RATIONALES
from src.data.data_manager import DataManager
from src.pipeline.evaluator import ModelEvaluator
from src.models.base_model import SupervisedRationaleModel
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


def load_models(model_dir: Path):
    """
    Load all models from directory.
    Returns (models dict, model_type str).
    model_type: 'mc_dropout' | 'bnn' | 'supervised' (covers logistic, rf, gb, catboost, lightgbm, xgboost, gp)
    """
    models = {}
    model_type = None

    if not model_dir.exists():
        print(f"ERROR: Directory {model_dir} does not exist!")
        return models, model_type

    # Multi-label: MC Dropout (single file)
    mc_dropout_path = model_dir / "mc_dropout_model.pkl"
    if mc_dropout_path.exists():
        models["mc_dropout"] = MCDropoutModel.load(str(mc_dropout_path))
        return models, "mc_dropout"

    # Multi-label: BNN (single file)
    bnn_path = model_dir / "bnn_model.pkl"
    if bnn_path.exists() and BNN_AVAILABLE:
        models["bnn"] = BNNModel.load(str(bnn_path))
        return models, "bnn"
    if bnn_path.exists() and not BNN_AVAILABLE:
        with open(bnn_path, "rb") as f:
            models["bnn"] = pickle.load(f)
        return models, "bnn"

    # Per-rationale: *_model.pkl (supervised, calibrated boosting, or GP)
    for model_path in sorted(model_dir.glob("*_model.pkl")):
        stem = model_path.stem
        if stem in ("mc_dropout_model", "bnn_model"):
            continue
        rationale = stem.replace("_model", "")
        try:
            if BOOSTING_AVAILABLE:
                model = CalibratedBoostingModel.load(str(model_path))
            elif GP_AVAILABLE:
                model = GPModel.load(str(model_path))
            else:
                model = SupervisedRationaleModel.load(str(model_path))
            models[rationale] = model
            model_type = "supervised"
        except Exception as e:
            try:
                model = SupervisedRationaleModel.load(str(model_path))
                models[rationale] = model
                model_type = "supervised"
            except Exception as e2:
                print(f"Warning: Could not load {model_path.name}: {e2}")

    return models, model_type or "supervised"


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
        help="Directory to save results (default: results/{model_type})",
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
        help="MC samples for uncertainty (MC Dropout/BNN only)",
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
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 80 + "\n")

    print("Loading models...")
    models, model_type = load_models(model_dir)
    if not models:
        print(f"No models found in {model_dir}")
        return
    print(f"Loaded {len(models)} model(s): {list(models.keys())} (type: {model_type})")

    # Data manager
    data_manager_path = model_dir / "data_manager.pkl"
    if data_manager_path.exists():
        with open(data_manager_path, "rb") as f:
            data_manager = pickle.load(f)
        print("Loaded data manager from training")
    else:
        data_manager = DataManager()
        print("Using new data manager")

    print("\nLoading data...")
    data_manager.load_data(args.data_path)
    data_manager.apply_filters(
        min_meetings_rat=args.min_meetings_rat, min_dissent=args.min_dissent
    )

    first_model = next(iter(models.values()))
    if (
        len(models) == 1
        and hasattr(first_model, "rationales")
        and isinstance(first_model.rationales, list)
    ):
        rationales = first_model.rationales
    else:
        rationales = list(models.keys())
    print(f"Rationales: {rationales}")

    _, test_df = data_manager.split_data(
        test_size=args.test_size,
        random_seed=args.random_seed,
        rationales=rationales,
    )

    evaluator = ModelEvaluator(save_plots=not args.no_plots)

    if model_type in ("mc_dropout", "bnn"):
        model = next(iter(models.values()))
        results = evaluator.evaluate_model(
            model=model,
            test_df=test_df,
            data_manager=data_manager,
            output_dir=output_dir,
            num_samples=args.num_mc_samples,
        )
    else:
        if len(models) == 1:
            model = next(iter(models.values()))
            results = evaluator.evaluate_model(
                model=model,
                test_df=test_df,
                data_manager=data_manager,
                output_dir=output_dir,
            )
        else:
            comparison_df = evaluator.compare_models(
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
