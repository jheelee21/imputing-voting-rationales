#!/usr/bin/env python3
"""
Extended evaluation script for all model types including BNN, GP, and Calibrated Boosting.

Usage:
    python evaluate_extended.py --model_dir models/logistic
    python evaluate_extended.py --model_dir models/mc_dropout --num_mc_samples 100
    python evaluate_extended.py --model_dir models/bnn --num_mc_samples 100
    python evaluate_extended.py --model_dir models/catboost
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

# Try to import new models
try:
    from bnn_model import BNNModel
    BNN_AVAILABLE = True
except:
    BNN_AVAILABLE = False

try:
    from calibrated_boosting import CalibratedBoostingModel
    BOOSTING_AVAILABLE = True
except:
    BOOSTING_AVAILABLE = False

try:
    from gaussian_process import GPModel
    GP_AVAILABLE = True
except:
    GP_AVAILABLE = False


def load_models(model_dir: Path):
    """Load all models from directory."""
    models = {}
    model_type = None
    
    print(f"Scanning directory: {model_dir}")
    print(f"Directory exists: {model_dir.exists()}")
    
    if not model_dir.exists():
        print(f"ERROR: Directory {model_dir} does not exist!")
        return models, model_type
    
    # List all files
    all_files = list(model_dir.glob("*"))
    print(f"Files found: {[f.name for f in all_files]}")
    
    # Check for MC Dropout model
    mc_dropout_path = model_dir / "mc_dropout_model.pkl"
    if mc_dropout_path.exists():
        print(f"Loading MC Dropout model from {mc_dropout_path}")
        models['mc_dropout'] = MCDropoutModel.load(str(mc_dropout_path))
        model_type = 'mc_dropout'
        return models, model_type
    
    # Check for BNN model
    bnn_path = model_dir / "bnn_model.pkl"
    print(f"Checking for BNN at {bnn_path}, exists: {bnn_path.exists()}")
    
    if bnn_path.exists():
        print(f"Loading BNN model from {bnn_path}")
        if BNN_AVAILABLE:
            try:
                models['bnn'] = BNNModel.load(str(bnn_path))
                model_type = 'bnn'
                print("Successfully loaded BNN model")
                return models, model_type
            except Exception as e:
                print(f"Error loading BNN: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("BNN model found but BNNModel not available - trying generic load")
            import pickle
            with open(bnn_path, 'rb') as f:
                bnn_obj = pickle.load(f)
            models['bnn'] = bnn_obj
            model_type = 'bnn'
            return models, model_type
    
    # Load supervised models (including calibrated boosting and GP)
    print("Looking for supervised models...")
    model_files = list(model_dir.glob("*_model.pkl"))
    print(f"Found model files: {[f.name for f in model_files]}")
    
    for model_path in model_files:
        if model_path.stem not in ["mc_dropout_model", "bnn_model"]:
            rationale = model_path.stem.replace("_model", "")
            print(f"Loading model for rationale: {rationale}")
            
            # Try different model types
            try:
                if BOOSTING_AVAILABLE:
                    model = CalibratedBoostingModel.load(str(model_path))
                    model_type = 'calibrated_boosting'
                elif GP_AVAILABLE:
                    model = GPModel.load(str(model_path))
                    model_type = 'gp'
                else:
                    model = SupervisedRationaleModel.load(str(model_path))
                    model_type = 'supervised'
                
                models[rationale] = model
            except Exception as e:
                print(f"Error loading {rationale}: {e}")
                # Fallback to base supervised model
                try:
                    model = SupervisedRationaleModel.load(str(model_path))
                    models[rationale] = model
                    model_type = 'supervised'
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
    
    print(f"Loaded {len(models)} models")
    return models, model_type


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
        help="Number of MC samples for uncertainty (MC Dropout/BNN only)"
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
        model_type_name = model_dir.name
        output_dir = RESULTS_DIR / model_type_name
    
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
    models, model_type = load_models(model_dir)
    
    if not models:
        print(f"No models found in {model_dir}")
        return
    
    print(f"Loaded {len(models)} model(s): {list(models.keys())}")
    print(f"Model type: {model_type}")
    
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
    
    # Determine rationales based on model type
    if hasattr(first_model, 'rationales'):
        # Multi-label models (MC Dropout, BNN)
        rationales = first_model.rationales
    elif isinstance(models, dict):
        # Supervised models - rationales are the keys
        rationales = list(models.keys())
    else:
        rationales = CORE_RATIONALES
    
    print(f"Rationales: {rationales}")
    
    _, test_df = data_manager.split_data(
        test_size=args.test_size,
        random_seed=args.random_seed,
        rationales=rationales,
    )
    
    # Evaluate models
    evaluator = ModelEvaluator(save_plots=not args.no_plots)
    
    if model_type in ['mc_dropout', 'bnn']:
        # Multi-label model evaluation
        model = first_model
        
        # Prepare test data
        test_clean = test_df.dropna(subset=rationales, how='all').copy()
        X_test, y_test, _ = data_manager.prepare_for_training(
            test_clean, rationales, fit=False
        )
        
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_type.upper()}")
        print(f"{'='*80}")
        print(f"Test samples: {len(y_test):,}")
        
        # Get predictions with uncertainty
        mean_probs, epistemic_unc, total_unc = model.predict_with_uncertainty(
            X_test, num_samples=args.num_mc_samples
        )
        y_pred = (mean_probs >= 0.5).astype(int)
        
        # Compute metrics per rationale
        results = {}
        for i, rationale in enumerate(rationales):
            y_true = y_test[:, i]
            y_p = y_pred[:, i]
            y_pr = mean_probs[:, i]
            
            metrics = evaluator._compute_metrics(y_true, y_p, y_pr, rationale)
            metrics['avg_epistemic_unc'] = float(epistemic_unc[:, i].mean())
            metrics['avg_total_unc'] = float(total_unc[:, i].mean())
            
            results[rationale] = {
                'metrics': metrics,
                'epistemic_unc': epistemic_unc[:, i],
                'total_unc': total_unc[:, i],
            }
            
            print(f"\n{rationale}:")
            print(f"  F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}, "
                  f"Epistemic Unc: {metrics['avg_epistemic_unc']:.4f}")
        
        # Save results
        import pandas as pd
        metrics_list = []
        for rationale, res in results.items():
            metric_dict = res['metrics'].copy()
            metric_dict['rationale'] = rationale
            metrics_list.append(metric_dict)
        
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
        
        print(f"\nResults saved to {output_dir}")
    
    else:
        # Supervised models - evaluate each rationale separately
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_type.upper()}")
        print(f"{'='*80}")
        
        all_metrics = []
        
        for rationale, model in models.items():
            print(f"\nEvaluating {rationale}...")
            
            # Prepare test data for this rationale
            test_clean = test_df.dropna(subset=[rationale]).copy()
            X_test, y_test, _ = data_manager.prepare_for_training(
                test_clean, [rationale], fit=False
            )
            y_test_single = y_test[:, 0] if y_test.ndim > 1 else y_test
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Compute metrics
            metrics = evaluator._compute_metrics(y_test_single, y_pred, y_prob, rationale)
            metrics['rationale'] = rationale
            all_metrics.append(metrics)
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  Mean Prob: {y_prob.mean():.4f}")
        
        # Save results
        import pandas as pd
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(metrics_df[['rationale', 'f1', 'roc_auc', 'accuracy']].to_string(index=False))
        print(f"\nResults saved to {output_dir}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()