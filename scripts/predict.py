#!/usr/bin/env python3
"""
Unified prediction script for generating predictions on unlabeled data.

Usage:
    python predict.py --model_dir models/logistic --output_dir results/predictions
    python predict.py --model_dir models/mc_dropout --include_uncertainty
    python predict.py --model_dir models/pca --output_dir predictions/pca
"""

import argparse
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import DATA_CONFIG, PROJECT_ROOT, ID_COLUMNS
from src.data.data_manager import DataManager
from src.pipeline.predictor import Predictor
from src.models.supervised import SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel
from src.models.bnn_model import BNNModel


def load_models(model_dir: Path):
    """Load all models from directory, including PCA models."""
    models = {}

    # Check for MC Dropout model
    mc_dropout_path = model_dir / "mc_dropout_model.pkl"
    if mc_dropout_path.exists():
        models["mc_dropout"] = MCDropoutModel.load(str(mc_dropout_path))
        return models

    # Check for BNN model
    bnn_path = model_dir / "bnn_model.pkl"
    if bnn_path.exists():
        models["bnn"] = BNNModel.load(str(bnn_path))
        return models

    # Try to import PCA model
    try:
        from src.models.pca_model import PCAModel

        PCA_AVAILABLE = True
    except ImportError:
        PCA_AVAILABLE = False

    # Load per-rationale models (supervised, PCA, boosting, GP, etc.)
    for model_path in model_dir.glob("*_model.pkl"):
        if model_path.stem in ["mc_dropout_model", "bnn_model"]:
            continue

        rationale = model_path.stem.replace("_model", "")

        try:
            # Peek at the pickle to determine model type
            with open(model_path, "rb") as f:
                temp_obj = pickle.load(f)

            # Check if it's a PCA model (saved as dict with 'pca' key)
            if isinstance(temp_obj, dict) and "pca" in temp_obj:
                if PCA_AVAILABLE:
                    models[rationale] = PCAModel.load(str(model_path))
                    print(f"Loaded PCA model: {rationale}")
                else:
                    print(
                        f"Warning: PCA model found but PCAModel not available: {rationale}"
                    )
            else:
                # Try to reload as object (might have been pickled directly)
                if hasattr(temp_obj, "predict_proba"):
                    # It's already a model object
                    models[rationale] = temp_obj
                    print(f"Loaded model: {rationale}")
                else:
                    # Use SupervisedRationaleModel loader
                    models[rationale] = SupervisedRationaleModel.load(str(model_path))
                    print(f"Loaded supervised model: {rationale}")

        except Exception as e:
            print(f"Warning: Could not load {model_path.name}: {e}")
            continue

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions on unlabeled data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
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

    # Data configuration
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

    # Prediction configuration
    parser.add_argument(
        "--include_uncertainty",
        action="store_true",
        help="Include uncertainty estimates (MC Dropout only)",
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=50,
        help="Number of MC samples for uncertainty (MC Dropout only)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Process in batches (for large datasets)",
    )
    parser.add_argument(
        "--output_filename", type=str, default="predictions.csv", help="Output filename"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_type = model_dir.name
        output_dir = PROJECT_ROOT / "predictions" / model_type

    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 80)
    print("VOTING RATIONALE PREDICTION")
    print("=" * 80)
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 80 + "\n")

    # Load models
    print("Loading models...")
    models = load_models(model_dir)

    if not models:
        print(f"No models found in {model_dir}")
        return

    print(f"Loaded {len(models)} model(s)")

    # Load data manager
    data_manager_path = model_dir / "data_manager.pkl"
    if data_manager_path.exists():
        with open(data_manager_path, "rb") as f:
            data_manager = pickle.load(f)
        print("Loaded data manager from training")
    else:
        print("Creating new data manager")
        data_manager = DataManager()

    # Load and prepare data
    print("\nLoading data...")
    data_manager.load_data(args.data_path)
    data_manager.apply_filters(
        min_meetings_rat=args.min_meetings_rat, min_dissent=args.min_dissent
    )

    # Get rationales from models
    first_model = next(iter(models.values()))

    # Handle different model types
    if hasattr(first_model, "rationales"):
        # Multi-label models (MC Dropout, BNN)
        rationales = first_model.rationales
    elif hasattr(first_model, "rationale"):
        # Single-label models (Supervised, PCA, etc.)
        rationales = [m.rationale for m in models.values() if hasattr(m, "rationale")]
    else:
        # Fallback: use model keys as rationales
        rationales = list(models.keys())

    print(f"Rationales: {rationales}")

    # Get unlabeled data
    unlabeled_df = data_manager.get_unlabeled_data(rationales)

    if len(unlabeled_df) == 0:
        print("\n⚠️  No unlabeled data found!")
        return

    # Generate predictions
    predictor = Predictor(
        id_columns=ID_COLUMNS,
        batch_size=args.batch_size,
    )

    predictions_df = predictor.predict(
        models=models,
        unlabeled_df=unlabeled_df,
        data_manager=data_manager,
        include_uncertainty=args.include_uncertainty,
        num_samples=args.num_mc_samples,
    )

    # Analyze predictions
    analysis = predictor.analyze_predictions(
        predictions_df=predictions_df,
        rationales=rationales,
        output_dir=output_dir,
    )

    # Save predictions
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
