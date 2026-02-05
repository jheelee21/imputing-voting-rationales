#!/usr/bin/env python3
"""
Unified evaluation script for all model types.

Usage:
    python evaluate.py --model_dir models/logistic
    python evaluate.py --model_dir models/mc_dropout --num_mc_samples 100
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


def load_models(model_dir: Path):
    """Load all models from directory."""
    models = {}
    
    # Check for MC Dropout model
    mc_dropout_path = model_dir / "mc_dropout_model.pkl"
    if mc_dropout_path.exists():
        models['mc_dropout'] = MCDropoutModel.load(str(mc_dropout_path))
        return models
    
    # Load supervised models
    for model_path in model_dir.glob("*_model.pkl"):
        if model_path.stem != "mc_dropout_model":
            rationale = model_path.stem.replace("_model", "")
            models[rationale] = SupervisedRationaleModel.load(str(model_path))
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate voting rationale prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/{model_type})"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_CONFIG["data_path"],
        help="Path to data file"
    )
    parser.add_argument(
        "--min_meetings_rat",
        type=int,
        default=DATA_CONFIG["min_meetings_rat"],
        help="Minimum N_Meetings_Rat filter"
    )
    parser.add_argument(
        "--min_dissent",
        type=int,
        default=DATA_CONFIG["min_dissent"],
        help="Minimum N_dissent filter"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=DATA_CONFIG["test_size"],
        help="Test set proportion"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=DATA_CONFIG["random_seed"],
        help="Random seed"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=50,
        help="Number of MC samples for uncertainty (MC Dropout only)"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plot generation"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_type = model_dir.name
        output_dir = RESULTS_DIR / model_type
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("="*80)
    print("VOTING RATIONALE MODEL EVALUATION")
    print("="*80)
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")
    print("="*80 + "\n")
    
    # Load models
    print("Loading models...")
    models = load_models(model_dir)
    
    if not models:
        print(f"No models found in {model_dir}")
        return
    
    print(f"Loaded {len(models)} model(s): {list(models.keys())}")
    
    # Load data manager
    data_manager_path = model_dir / "data_manager.pkl"
    if data_manager_path.exists():
        with open(data_manager_path, 'rb') as f:
            data_manager = pickle.load(f)
        print("Loaded data manager from training")
    else:
        print("Creating new data manager")
        data_manager = DataManager()
    
    # Load and prepare data
    print("\nLoading data...")
    data_manager.load_data(args.data_path)
    data_manager.apply_filters(
        min_meetings_rat=args.min_meetings_rat,
        min_dissent=args.min_dissent
    )
    
    # Get rationales from models
    first_model = next(iter(models.values()))
    rationales = first_model.rationales
    
    _, test_df = data_manager.split_data(
        test_size=args.test_size,
        random_seed=args.random_seed,
        rationales=rationales,
    )
    
    # Evaluate models
    evaluator = ModelEvaluator(save_plots=not args.no_plots)
    
    if len(models) == 1:
        # Single model evaluation
        model = next(iter(models.values()))
        results = evaluator.evaluate_model(
            model=model,
            test_df=test_df,
            data_manager=data_manager,
            output_dir=output_dir,
        )
    else:
        # Multi-model comparison
        comparison_df = evaluator.compare_models(
            models=models,
            test_df=test_df,
            data_manager=data_manager,
            output_dir=output_dir,
        )
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()