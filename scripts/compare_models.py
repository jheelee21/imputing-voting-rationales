"""
Compare different model types: Supervised, MC Dropout, and Variational Inference

This script loads multiple trained models and compares their:
- Predictive performance (accuracy, F1, AUC)
- Calibration quality
- Uncertainty estimates (for probabilistic models)
- Computational efficiency
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import DataLoader
from src.utils.types import CORE_RATIONALES
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, log_loss, brier_score_loss
)


def load_supervised_models(model_dir: Path, rationales: list):
    """Load individual supervised models for each rationale"""
    models = {}
    preprocessors = {}
    
    for rationale in rationales:
        model_path = model_dir / f"{rationale}_model.pkl"
        preprocessor_path = model_dir / f"{rationale}_preprocessor.pkl"
        
        if model_path.exists() and preprocessor_path.exists():
            with open(model_path, 'rb') as f:
                models[rationale] = pickle.load(f)
            with open(preprocessor_path, 'rb') as f:
                preprocessors[rationale] = pickle.load(f)
    
    return models, preprocessors


def evaluate_supervised(models, preprocessors, test_df, rationales):
    """Evaluate supervised models (per-rationale)"""
    
    print("\n" + "=" * 80)
    print("EVALUATING SUPERVISED MODELS")
    print("=" * 80)
    
    results = {}
    all_probs = {}
    all_preds = {}
    
    for rationale in rationales:
        if rationale not in models:
            print(f"Model for {rationale} not found, skipping...")
            continue
        
        model = models[rationale]
        preprocessor = preprocessors[rationale]
        
        # Prepare data
        test_clean = test_df.dropna(subset=[rationale]).copy()
        
        X_test_df = preprocessor.prepare_features(test_clean)
        X_test_df = preprocessor.encode_categorical(X_test_df, fit=False)
        X_test_df = preprocessor.handle_missing(X_test_df)
        
        y_test = (test_clean[rationale] == 1).astype(int).values
        
        categorical_cols = [c for c in X_test_df.columns if c.endswith("_encoded")]
        numerical_cols = [c for c in X_test_df.columns if c not in categorical_cols]
        
        X_test_scaled = preprocessor.scale_features(X_test_df[numerical_cols], fit=False)
        X_test = np.hstack([X_test_scaled, X_test_df[categorical_cols].values])
        
        # Predict
        start_time = time.time()
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        all_probs[rationale] = y_prob
        all_preds[rationale] = y_pred
        
        # Compute metrics
        if y_test.sum() > 0:
            results[rationale] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_prob),
                'avg_precision': average_precision_score(y_test, y_prob),
                'log_loss': log_loss(y_test, y_prob),
                'brier_score': brier_score_loss(y_test, y_prob),
                'inference_time': inference_time,
                'n_samples': len(y_test),
                'n_positive': int(y_test.sum()),
            }
            
            print(f"\n{rationale}:")
            print(f"  F1: {results[rationale]['f1']:.4f}, AUC: {results[rationale]['auc']:.4f}")
            print(f"  Log Loss: {results[rationale]['log_loss']:.4f}")
            print(f"  Time: {inference_time:.3f}s")
    
    return pd.DataFrame(results).T, all_probs, all_preds


def evaluate_mc_dropout(model_path, preprocessor_path, test_df, rationales, num_samples=50):
    """Evaluate MC Dropout model"""
    
    print("\n" + "=" * 80)
    print("EVALUATING MC DROPOUT MODEL")
    print("=" * 80)
    
    # Import here to avoid dependency if not needed
    from mc_dropout_model import MCDropoutModel
    
    # Load model
    model = MCDropoutModel.load_model(str(model_path))
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Prepare data
    test_clean = test_df.dropna(subset=rationales, how='all').copy()
    
    X_test_df = preprocessor.prepare_features(test_clean)
    X_test_df = preprocessor.encode_categorical(X_test_df, fit=False)
    X_test_df = preprocessor.handle_missing(X_test_df)
    
    y_test = test_clean[rationales].fillna(0).astype(int)
    
    categorical_cols = [c for c in X_test_df.columns if c.endswith("_encoded")]
    numerical_cols = [c for c in X_test_df.columns if c not in categorical_cols]
    
    X_test_scaled = preprocessor.scale_features(X_test_df[numerical_cols], fit=False)
    X_test = np.hstack([X_test_scaled, X_test_df[categorical_cols].values])
    
    # Predict
    start_time = time.time()
    mean_probs, epistemic_unc, _ = model.predict_with_uncertainty(X_test, num_samples=num_samples)
    inference_time = time.time() - start_time
    
    y_pred = (mean_probs >= 0.5).astype(int)
    
    # Compute metrics per rationale
    results = {}
    all_probs = {}
    all_preds = {}
    all_unc = {}
    
    for i, rationale in enumerate(rationales):
        y_true = y_test.values[:, i]
        y_pr = mean_probs[:, i]
        y_p = y_pred[:, i]
        unc = epistemic_unc[:, i]
        
        all_probs[rationale] = y_pr
        all_preds[rationale] = y_p
        all_unc[rationale] = unc
        
        if y_true.sum() > 0:
            results[rationale] = {
                'accuracy': accuracy_score(y_true, y_p),
                'f1': f1_score(y_true, y_p, zero_division=0),
                'auc': roc_auc_score(y_true, y_pr),
                'avg_precision': average_precision_score(y_true, y_pr),
                'log_loss': log_loss(y_true, y_pr),
                'brier_score': brier_score_loss(y_true, y_pr),
                'inference_time': inference_time / len(rationales),  # Amortized
                'avg_uncertainty': float(unc.mean()),
                'n_samples': len(y_true),
                'n_positive': int(y_true.sum()),
            }
            
            print(f"\n{rationale}:")
            print(f"  F1: {results[rationale]['f1']:.4f}, AUC: {results[rationale]['auc']:.4f}")
            print(f"  Log Loss: {results[rationale]['log_loss']:.4f}")
            print(f"  Avg Uncertainty: {results[rationale]['avg_uncertainty']:.4f}")
    
    print(f"\nTotal inference time: {inference_time:.3f}s ({num_samples} MC samples)")
    
    return pd.DataFrame(results).T, all_probs, all_preds, all_unc


def compare_models(supervised_metrics, mc_dropout_metrics, rationales, save_dir):
    """Compare metrics across model types"""
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    comparison = []
    
    for rationale in rationales:
        if rationale not in supervised_metrics.index or rationale not in mc_dropout_metrics.index:
            continue
        
        sup = supervised_metrics.loc[rationale]
        mcd = mc_dropout_metrics.loc[rationale]
        
        comparison.append({
            'rationale': rationale,
            # Supervised
            'supervised_f1': sup['f1'],
            'supervised_auc': sup['auc'],
            'supervised_logloss': sup['log_loss'],
            'supervised_brier': sup['brier_score'],
            'supervised_time': sup['inference_time'],
            # MC Dropout
            'mc_dropout_f1': mcd['f1'],
            'mc_dropout_auc': mcd['auc'],
            'mc_dropout_logloss': mcd['log_loss'],
            'mc_dropout_brier': mcd['brier_score'],
            'mc_dropout_time': mcd['inference_time'],
            'mc_dropout_uncertainty': mcd.get('avg_uncertainty', np.nan),
            # Differences
            'f1_diff': mcd['f1'] - sup['f1'],
            'auc_diff': mcd['auc'] - sup['auc'],
            'logloss_diff': sup['log_loss'] - mcd['log_loss'],  # Lower is better
        })
        
        print(f"\n{rationale}:")
        print(f"  F1:       Supervised={sup['f1']:.4f}, MC Dropout={mcd['f1']:.4f}, Δ={mcd['f1']-sup['f1']:+.4f}")
        print(f"  AUC:      Supervised={sup['auc']:.4f}, MC Dropout={mcd['auc']:.4f}, Δ={mcd['auc']-sup['auc']:+.4f}")
        print(f"  LogLoss:  Supervised={sup['log_loss']:.4f}, MC Dropout={mcd['log_loss']:.4f}, Δ={sup['log_loss']-mcd['log_loss']:+.4f}")
        print(f"  Time:     Supervised={sup['inference_time']:.3f}s, MC Dropout={mcd['inference_time']:.3f}s")
    
    comparison_df = pd.DataFrame(comparison)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    print(f"\nAverage F1:")
    print(f"  Supervised: {comparison_df['supervised_f1'].mean():.4f}")
    print(f"  MC Dropout: {comparison_df['mc_dropout_f1'].mean():.4f}")
    print(f"  Improvement: {comparison_df['f1_diff'].mean():+.4f}")
    
    print(f"\nAverage AUC:")
    print(f"  Supervised: {comparison_df['supervised_auc'].mean():.4f}")
    print(f"  MC Dropout: {comparison_df['mc_dropout_auc'].mean():.4f}")
    print(f"  Improvement: {comparison_df['auc_diff'].mean():+.4f}")
    
    print(f"\nAverage Log Loss:")
    print(f"  Supervised: {comparison_df['supervised_logloss'].mean():.4f}")
    print(f"  MC Dropout: {comparison_df['mc_dropout_logloss'].mean():.4f}")
    print(f"  Improvement: {comparison_df['logloss_diff'].mean():+.4f} (lower is better)")
    
    print(f"\nInference Time:")
    print(f"  Supervised: {comparison_df['supervised_time'].mean():.3f}s (avg per rationale)")
    print(f"  MC Dropout: {comparison_df['mc_dropout_time'].mean():.3f}s (avg per rationale)")
    
    # Visualizations
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric comparison bar plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = [
            ('f1', 'F1 Score', axes[0, 0]),
            ('auc', 'ROC AUC', axes[0, 1]),
            ('logloss', 'Log Loss', axes[1, 0]),
            ('brier', 'Brier Score', axes[1, 1]),
        ]
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        for metric, title, ax in metrics_to_plot:
            sup_vals = comparison_df[f'supervised_{metric}'].values
            mcd_vals = comparison_df[f'mc_dropout_{metric}'].values
            
            ax.bar(x - width/2, sup_vals, width, label='Supervised', alpha=0.8)
            ax.bar(x + width/2, mcd_vals, width, label='MC Dropout', alpha=0.8)
            
            ax.set_ylabel(title, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['rationale'], rotation=45, ha='right')
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
        
        plt.suptitle('Model Comparison: Supervised vs MC Dropout', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        comp_path = save_dir / "model_comparison_metrics.png"
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved comparison plot to {comp_path}")
        
        # Save comparison table
        comp_csv_path = save_dir / "model_comparison.csv"
        comparison_df.to_csv(comp_csv_path, index=False)
        print(f"Saved comparison table to {comp_csv_path}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Compare different model types")
    parser.add_argument("--data_path", type=str, default="./data/Imputing Rationales.csv")
    parser.add_argument("--supervised_dir", type=str, default="./models/trained/supervised")
    parser.add_argument("--mc_dropout_dir", type=str, default="./models/trained/mc_dropout")
    parser.add_argument("--save_dir", type=str, default="./results/comparison")
    parser.add_argument("--rationales", nargs="+", default=CORE_RATIONALES)
    parser.add_argument("--num_mc_samples", type=int, default=50)
    parser.add_argument("--min_meetings_rat", type=int, default=1)
    parser.add_argument("--min_dissent", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=21)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MULTI-MODEL COMPARISON")
    print("=" * 80)
    print(f"Comparing: Supervised vs MC Dropout")
    print(f"Rationales: {args.rationales}")
    print("=" * 80)
    
    # Load data
    loader = DataLoader()
    df = loader.load_data(args.data_path)
    df = loader.apply_filters(
        min_meetings_rat=args.min_meetings_rat,
        min_dissent=args.min_dissent
    )
    
    _, test_df = loader.split_train_test(
        test_size=args.test_size,
        random_seed=args.random_seed,
        label=args.rationales,
    )
    
    print(f"\nTest set: {len(test_df)} samples")
    
    # Evaluate supervised models
    supervised_dir = Path(args.supervised_dir)
    if supervised_dir.exists():
        sup_models, sup_preprocessors = load_supervised_models(supervised_dir, args.rationales)
        supervised_metrics, _, _ = evaluate_supervised(
            sup_models, sup_preprocessors, test_df, args.rationales
        )
    else:
        print(f"\nSupervised models not found at {supervised_dir}")
        supervised_metrics = None
    
    # Evaluate MC Dropout
    mc_dropout_dir = Path(args.mc_dropout_dir)
    mc_model_path = mc_dropout_dir / "mc_dropout_model.pkl"
    mc_preprocessor_path = mc_dropout_dir / "mc_dropout_preprocessor.pkl"
    
    if mc_model_path.exists() and mc_preprocessor_path.exists():
        mc_dropout_metrics, _, _, _ = evaluate_mc_dropout(
            mc_model_path, mc_preprocessor_path, test_df, args.rationales, args.num_mc_samples
        )
    else:
        print(f"\nMC Dropout model not found at {mc_dropout_dir}")
        mc_dropout_metrics = None
    
    # Compare models
    if supervised_metrics is not None and mc_dropout_metrics is not None:
        comparison_df = compare_models(
            supervised_metrics, mc_dropout_metrics, args.rationales, args.save_dir
        )
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        
        avg_auc_improvement = comparison_df['auc_diff'].mean()
        avg_f1_improvement = comparison_df['f1_diff'].mean()
        
        if avg_auc_improvement > 0.01 or avg_f1_improvement > 0.01:
            print("✓ MC Dropout shows meaningful improvement over supervised models")
            print(f"  Average AUC improvement: {avg_auc_improvement:+.4f}")
            print(f"  Average F1 improvement: {avg_f1_improvement:+.4f}")
            print("\n  → Recommended: Use MC Dropout for production")
        elif abs(avg_auc_improvement) < 0.005 and abs(avg_f1_improvement) < 0.005:
            print("≈ Models perform similarly")
            print("\n  → Consider: Use Supervised for speed, MC Dropout for uncertainty")
        else:
            print("! Supervised models perform better")
            print(f"  Average AUC difference: {avg_auc_improvement:+.4f}")
            print(f"  Average F1 difference: {avg_f1_improvement:+.4f}")
            print("\n  → Recommended: Use Supervised models")
    
    print(f"\n\nAll results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()