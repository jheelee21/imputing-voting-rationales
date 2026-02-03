"""
Evaluation script for MC Dropout models with uncertainty quantification.

This script evaluates MC Dropout Neural Networks and provides:
- Standard classification metrics
- Uncertainty estimates (epistemic via dropout sampling)
- Calibration analysis
- Confidence-based predictions
- Comparison with deterministic predictions
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.models.mc_dropout import MCDropoutModel
from src.utils.types import CORE_RATIONALES
from src.evaluation.metrics import (
    evaluate_probabilistic_predictions,
    plot_calibration_curves,
    plot_probability_distributions,
    analyze_prediction_confidence,
)


def evaluate_mc_dropout(
    model: MCDropoutModel,
    preprocessor: DataPreprocessor,
    test_df: pd.DataFrame,
    rationales: list,
    num_samples: int = 50,
    save_dir: Path = None,
) -> dict:
    """
    Evaluate MC Dropout model with uncertainty quantification.
    
    Args:
        model: Trained MC Dropout model
        preprocessor: Fitted preprocessor
        test_df: Test dataframe
        rationales: List of rationales
        num_samples: Number of MC samples for uncertainty estimation
        save_dir: Directory to save results
    
    Returns:
        Dictionary of results
    """
    
    print("=" * 80)
    print("EVALUATING MC DROPOUT MODEL")
    print("=" * 80)
    
    # Prepare test data
    test_df_clean = test_df.dropna(subset=rationales, how='all').copy()
    print(f"\nTest samples: {len(test_df_clean)}")
    
    X_test_df = preprocessor.prepare_features(test_df_clean)
    X_test_df = preprocessor.encode_categorical(X_test_df, fit=False)
    X_test_df = preprocessor.handle_missing(X_test_df)
    
    y_test = test_df_clean[rationales].fillna(0).astype(int)
    
    # Scale features (ensure same columns as training)
    categorical_encoded_cols = [c for c in X_test_df.columns if c.endswith("_encoded")]
    numerical_cols = [c for c in X_test_df.columns if c not in categorical_encoded_cols]
    
    X_test_scaled = preprocessor.scale_features(X_test_df[numerical_cols], fit=False)
    X_test = np.hstack([X_test_scaled, X_test_df[categorical_encoded_cols].values])
    
    print(f"Test feature matrix shape: {X_test.shape}")
    print(f"Test label matrix shape: {y_test.shape}")
    
    # Get predictions with uncertainty (MC Dropout)
    print(f"\nGenerating predictions with {num_samples} MC samples...")
    mean_probs, epistemic_unc, total_unc = model.predict_with_uncertainty(
        X_test, num_samples=num_samples
    )
    
    y_pred = (mean_probs >= 0.5).astype(int)
    
    # Also get deterministic predictions (dropout OFF) for comparison
    print("Generating deterministic predictions (no dropout)...")
    model.model.eval()  # Turn off dropout
    import torch
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(model.device)
        logits = model.model(X_tensor)
        probs_deterministic = torch.sigmoid(logits).cpu().numpy()
    
    # Convert to DataFrames
    y_test_df = pd.DataFrame(y_test.values, columns=rationales)
    y_pred_df = pd.DataFrame(y_pred, columns=rationales)
    y_prob_df = pd.DataFrame(mean_probs, columns=rationales)
    y_prob_det_df = pd.DataFrame(probs_deterministic, columns=rationales)
    epistemic_df = pd.DataFrame(epistemic_unc, columns=rationales)
    total_unc_df = pd.DataFrame(total_unc, columns=rationales)
    
    # Compute standard metrics
    print("\n" + "=" * 80)
    print("CLASSIFICATION METRICS (MC Dropout Predictions)")
    print("=" * 80)
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, log_loss
    )
    
    metrics_per_label = []
    
    for rationale in rationales:
        y_true = y_test_df[rationale].values
        y_p = y_pred_df[rationale].values
        y_pr = y_prob_df[rationale].values
        
        if y_true.sum() == 0:
            print(f"\n{rationale}: No positive samples in test set")
            continue
        
        try:
            auc = roc_auc_score(y_true, y_pr)
        except:
            auc = np.nan
        
        try:
            avg_prec = average_precision_score(y_true, y_pr)
        except:
            avg_prec = np.nan
        
        try:
            logloss = log_loss(y_true, y_pr)
        except:
            logloss = np.nan
        
        metrics = {
            'rationale': rationale,
            'n_samples': len(y_true),
            'n_positive': int(y_true.sum()),
            'positive_rate': float(y_true.mean()),
            'accuracy': accuracy_score(y_true, y_p),
            'precision': precision_score(y_true, y_p, zero_division=0),
            'recall': recall_score(y_true, y_p, zero_division=0),
            'f1': f1_score(y_true, y_p, zero_division=0),
            'roc_auc': auc,
            'avg_precision': avg_prec,
            'log_loss': logloss,
        }
        
        metrics_per_label.append(metrics)
        
        print(f"\n{rationale}:")
        print(f"  Support: {metrics['n_positive']} ({metrics['positive_rate']:.2%})")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
    
    metrics_df = pd.DataFrame(metrics_per_label)
    
    # Compare MC Dropout vs Deterministic
    print("\n" + "=" * 80)
    print("MC DROPOUT vs DETERMINISTIC COMPARISON")
    print("=" * 80)
    
    comparison = []
    for rationale in rationales:
        y_true = y_test_df[rationale].values
        y_pr_mc = y_prob_df[rationale].values
        y_pr_det = y_prob_det_df[rationale].values
        
        if y_true.sum() == 0:
            continue
        
        try:
            auc_mc = roc_auc_score(y_true, y_pr_mc)
            auc_det = roc_auc_score(y_true, y_pr_det)
        except:
            auc_mc = auc_det = np.nan
        
        try:
            ll_mc = log_loss(y_true, y_pr_mc)
            ll_det = log_loss(y_true, y_pr_det)
        except:
            ll_mc = ll_det = np.nan
        
        comparison.append({
            'rationale': rationale,
            'auc_mc_dropout': auc_mc,
            'auc_deterministic': auc_det,
            'auc_improvement': auc_mc - auc_det,
            'logloss_mc_dropout': ll_mc,
            'logloss_deterministic': ll_det,
            'logloss_improvement': ll_det - ll_mc,  # Lower is better
        })
        
        print(f"\n{rationale}:")
        print(f"  AUC - MC Dropout: {auc_mc:.4f}, Deterministic: {auc_det:.4f}, Δ: {auc_mc - auc_det:+.4f}")
        print(f"  LogLoss - MC Dropout: {ll_mc:.4f}, Deterministic: {ll_det:.4f}, Δ: {ll_det - ll_mc:+.4f}")
    
    comparison_df = pd.DataFrame(comparison)
    
    # Uncertainty analysis
    print("\n" + "=" * 80)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 80)
    
    uncertainty_stats = []
    
    for rationale in rationales:
        epistemic = epistemic_df[rationale].values
        total = total_unc_df[rationale].values
        
        stats = {
            'rationale': rationale,
            'epistemic_mean': float(epistemic.mean()),
            'epistemic_std': float(epistemic.std()),
            'epistemic_median': float(np.median(epistemic)),
            'epistemic_q25': float(np.percentile(epistemic, 25)),
            'epistemic_q75': float(np.percentile(epistemic, 75)),
            'total_unc_mean': float(total.mean()),
            'total_unc_std': float(total.std()),
            'total_unc_median': float(np.median(total)),
        }
        
        uncertainty_stats.append(stats)
        
        print(f"\n{rationale}:")
        print(f"  Epistemic uncertainty: {stats['epistemic_mean']:.4f} ± {stats['epistemic_std']:.4f}")
        print(f"  Epistemic median: {stats['epistemic_median']:.4f} (Q25={stats['epistemic_q25']:.4f}, Q75={stats['epistemic_q75']:.4f})")
        print(f"  Total uncertainty: {stats['total_unc_mean']:.4f} ± {stats['total_unc_std']:.4f}")
    
    uncertainty_df = pd.DataFrame(uncertainty_stats)
    
    # Analyze uncertainty vs correctness
    print("\n" + "=" * 80)
    print("UNCERTAINTY vs CORRECTNESS ANALYSIS")
    print("=" * 80)
    
    unc_correctness = []
    
    for rationale in rationales:
        y_true = y_test_df[rationale].values
        y_p = y_pred_df[rationale].values
        epistemic = epistemic_df[rationale].values
        
        correct = (y_true == y_p)
        incorrect = ~correct
        
        if incorrect.sum() > 0:
            unc_correct = epistemic[correct].mean()
            unc_incorrect = epistemic[incorrect].mean()
            ratio = unc_incorrect / unc_correct if unc_correct > 0 else np.nan
            
            unc_correctness.append({
                'rationale': rationale,
                'unc_correct': unc_correct,
                'unc_incorrect': unc_incorrect,
                'ratio': ratio,
                'n_correct': correct.sum(),
                'n_incorrect': incorrect.sum(),
            })
            
            print(f"\n{rationale}:")
            print(f"  Correct predictions ({correct.sum()}): Avg unc = {unc_correct:.4f}")
            print(f"  Incorrect predictions ({incorrect.sum()}): Avg unc = {unc_incorrect:.4f}")
            print(f"  Ratio (incorrect/correct): {ratio:.2f}x")
            
            if ratio > 1:
                print(f"  ✓ Good! Higher uncertainty on mistakes")
            else:
                print(f"  ⚠ Warning: Lower uncertainty on mistakes")
    
    unc_correctness_df = pd.DataFrame(unc_correctness)
    
    # Confidence analysis
    print("\n" + "=" * 80)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 80)
    
    confidence_df = analyze_prediction_confidence(
        y_prob_df, rationales, thresholds=[0.5, 0.7, 0.9]
    )
    print("\n" + confidence_df.to_string(index=False))
    
    # Uncertainty-based rejection analysis
    print("\n" + "=" * 80)
    print("UNCERTAINTY-BASED REJECTION ANALYSIS")
    print("=" * 80)
    print("(What if we only predict when uncertainty is low?)")
    
    rejection_analysis = []
    
    for unc_threshold in [0.05, 0.10, 0.15, 0.20]:
        for rationale in rationales:
            y_true = y_test_df[rationale].values
            y_p = y_pred_df[rationale].values
            epistemic = epistemic_df[rationale].values
            
            # Only evaluate predictions with low uncertainty
            low_unc_mask = epistemic < unc_threshold
            
            if low_unc_mask.sum() == 0:
                continue
            
            retained_pct = low_unc_mask.sum() / len(y_true) * 100
            accuracy = accuracy_score(y_true[low_unc_mask], y_p[low_unc_mask])
            
            rejection_analysis.append({
                'rationale': rationale,
                'unc_threshold': unc_threshold,
                'retained_pct': retained_pct,
                'accuracy': accuracy,
            })
    
    rejection_df = pd.DataFrame(rejection_analysis)
    
    if len(rejection_df) > 0:
        print("\nAccuracy when rejecting high-uncertainty predictions:")
        for rationale in rationales:
            rat_data = rejection_df[rejection_df['rationale'] == rationale]
            if len(rat_data) > 0:
                print(f"\n{rationale}:")
                for _, row in rat_data.iterrows():
                    print(f"  Unc < {row['unc_threshold']:.2f}: "
                          f"{row['retained_pct']:.1f}% retained, "
                          f"Accuracy = {row['accuracy']:.4f}")
    
    # Visualizations
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # 1. Calibration curves
        print("\nPlotting calibration curves...")
        fig = plot_calibration_curves(y_test_df, y_prob_df, rationales)
        calib_path = save_dir / "mc_dropout_calibration.png"
        fig.savefig(calib_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved to {calib_path}")
        
        # 2. Probability distributions
        print("Plotting probability distributions...")
        fig = plot_probability_distributions(y_prob_df, rationales, y_test_df)
        prob_path = save_dir / "mc_dropout_probabilities.png"
        fig.savefig(prob_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved to {prob_path}")
        
        # 3. Uncertainty distributions
        print("Plotting uncertainty distributions...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, rationale in enumerate(rationales):
            if i >= len(axes):
                break
            
            ax = axes[i]
            epistemic = epistemic_df[rationale].values
            
            ax.hist(epistemic, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(epistemic.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {epistemic.mean():.3f}')
            ax.axvline(np.median(epistemic), color='orange', linestyle=':', linewidth=2,
                      label=f'Median: {np.median(epistemic):.3f}')
            
            ax.set_xlabel('Epistemic Uncertainty', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{rationale}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        for i in range(len(rationales), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Epistemic Uncertainty Distributions (MC Dropout)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        unc_path = save_dir / "mc_dropout_uncertainty_dist.png"
        fig.savefig(unc_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved to {unc_path}")
        
        # 4. Uncertainty vs Correctness scatter
        print("Plotting uncertainty vs correctness...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, rationale in enumerate(rationales):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            y_true = y_test_df[rationale].values
            y_p = y_pred_df[rationale].values
            epistemic = epistemic_df[rationale].values
            probs = y_prob_df[rationale].values
            
            correct = (y_true == y_p)
            
            ax.scatter(probs[correct], epistemic[correct], 
                      alpha=0.3, s=15, c='green', label=f'Correct ({correct.sum()})')
            ax.scatter(probs[~correct], epistemic[~correct], 
                      alpha=0.5, s=20, c='red', marker='x', label=f'Incorrect ({(~correct).sum()})')
            
            ax.set_xlabel('Predicted Probability', fontsize=10)
            ax.set_ylabel('Epistemic Uncertainty', fontsize=10)
            ax.set_title(f'{rationale}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
        
        for i in range(len(rationales), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Uncertainty vs Correctness (MC Dropout)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        scatter_path = save_dir / "mc_dropout_unc_vs_correctness.png"
        fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved to {scatter_path}")
        
        # 5. MC Dropout vs Deterministic comparison
        print("Plotting MC Dropout vs Deterministic comparison...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, rationale in enumerate(rationales):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            probs_mc = y_prob_df[rationale].values
            probs_det = y_prob_det_df[rationale].values
            
            ax.scatter(probs_det, probs_mc, alpha=0.3, s=10, c='steelblue')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect agreement')
            
            ax.set_xlabel('Deterministic (no dropout)', fontsize=10)
            ax.set_ylabel('MC Dropout (averaged)', fontsize=10)
            ax.set_title(f'{rationale}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        for i in range(len(rationales), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('MC Dropout vs Deterministic Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        comp_path = save_dir / "mc_dropout_vs_deterministic.png"
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved to {comp_path}")
        
        # 6. Rejection curve (accuracy vs coverage)
        if len(rejection_df) > 0:
            print("Plotting rejection curves...")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for rationale in rationales:
                rat_data = rejection_df[rejection_df['rationale'] == rationale].sort_values('unc_threshold')
                if len(rat_data) > 0:
                    ax.plot(rat_data['retained_pct'], rat_data['accuracy'], 
                           marker='o', linewidth=2, label=rationale)
            
            ax.set_xlabel('Percentage of Predictions Retained (%)', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Accuracy vs Coverage (Reject High Uncertainty)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 100])
            
            plt.tight_layout()
            reject_path = save_dir / "mc_dropout_rejection_curve.png"
            fig.savefig(reject_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved to {reject_path}")
        
        # Save all results
        print("\nSaving results to CSV...")
        
        metrics_path = save_dir / "mc_dropout_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
        
        comparison_path = save_dir / "mc_dropout_vs_deterministic.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Comparison saved to {comparison_path}")
        
        uncertainty_path = save_dir / "mc_dropout_uncertainty_stats.csv"
        uncertainty_df.to_csv(uncertainty_path, index=False)
        print(f"Uncertainty stats saved to {uncertainty_path}")
        
        unc_correct_path = save_dir / "mc_dropout_unc_correctness.csv"
        unc_correctness_df.to_csv(unc_correct_path, index=False)
        print(f"Uncertainty vs correctness saved to {unc_correct_path}")
        
        confidence_path = save_dir / "mc_dropout_confidence.csv"
        confidence_df.to_csv(confidence_path, index=False)
        print(f"Confidence analysis saved to {confidence_path}")
        
        if len(rejection_df) > 0:
            rejection_path = save_dir / "mc_dropout_rejection.csv"
            rejection_df.to_csv(rejection_path, index=False)
            print(f"Rejection analysis saved to {rejection_path}")
        
        # Save predictions with uncertainty
        predictions_df = test_df_clean[['investor_id', 'pid', 'ProxySeason']].copy()
        
        for rationale in rationales:
            predictions_df[f'{rationale}_true'] = y_test_df[rationale].values
            predictions_df[f'{rationale}_pred'] = y_pred_df[rationale].values
            predictions_df[f'{rationale}_prob_mc'] = y_prob_df[rationale].values
            predictions_df[f'{rationale}_prob_det'] = y_prob_det_df[rationale].values
            predictions_df[f'{rationale}_epistemic_unc'] = epistemic_df[rationale].values
            predictions_df[f'{rationale}_total_unc'] = total_unc_df[rationale].values
        
        predictions_path = save_dir / "mc_dropout_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")
    
    results = {
        'metrics': metrics_df,
        'comparison': comparison_df,
        'uncertainty': uncertainty_df,
        'unc_correctness': unc_correctness_df,
        'confidence': confidence_df,
        'rejection': rejection_df if len(rejection_df) > 0 else pd.DataFrame(),
        'predictions': {
            'y_true': y_test_df,
            'y_pred': y_pred_df,
            'y_prob_mc': y_prob_df,
            'y_prob_deterministic': y_prob_det_df,
            'epistemic_uncertainty': epistemic_df,
            'total_uncertainty': total_unc_df,
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MC Dropout model with uncertainty quantification"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/Imputing Rationales.csv",
        help="Path to data file",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models/trained/mc_dropout",
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/mc_dropout",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--rationales",
        nargs="+",
        default=CORE_RATIONALES,
        help="Rationales to evaluate",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of MC samples for uncertainty estimation",
    )
    parser.add_argument(
        "--min_meetings_rat",
        type=int,
        default=1,
        help="Minimum N_Meetings_Rat filter",
    )
    parser.add_argument(
        "--min_dissent",
        type=int,
        default=5,
        help="Minimum N_dissent filter",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MC DROPOUT MODEL EVALUATION")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Model directory: {args.model_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Rationales: {args.rationales}")
    print(f"MC samples: {args.num_samples}")
    print("=" * 80)
    
    # Load model and preprocessor
    model_path = Path(args.model_dir) / "mc_dropout_model.pkl"
    preprocessor_path = Path(args.model_dir) / "mc_dropout_preprocessor.pkl"
    
    print(f"\nLoading model from {model_path}")
    model = MCDropoutModel.load_model(str(model_path))
    
    print(f"Loading preprocessor from {preprocessor_path}")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load data
    loader = DataLoader()
    df = loader.load_data(args.data_path)
    
    # Apply filters
    df = loader.apply_filters(
        min_meetings_rat=args.min_meetings_rat,
        min_dissent=args.min_dissent
    )
    
    # Split data (use same seed to get same test set)
    _, test_df = loader.split_train_test(
        test_size=args.test_size,
        random_seed=args.random_seed,
        label=args.rationales,
    )
    
    print(f"\nTest set: {len(test_df)} samples")
    
    # Evaluate
    results = evaluate_mc_dropout(
        model=model,
        preprocessor=preprocessor,
        test_df=test_df,
        rationales=args.rationales,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    print("\nOverall Performance Summary:")
    print(f"Average F1 Score: {results['metrics']['f1'].mean():.4f}")
    print(f"Average ROC AUC: {results['metrics']['roc_auc'].mean():.4f}")
    print(f"Average Log Loss: {results['metrics']['log_loss'].mean():.4f}")
    print(f"\nAverage Epistemic Uncertainty: {results['uncertainty']['epistemic_mean'].mean():.4f}")
    
    if len(results['unc_correctness']) > 0:
        avg_ratio = results['unc_correctness']['ratio'].mean()
        print(f"Avg Uncertainty Ratio (incorrect/correct): {avg_ratio:.2f}x")
        if avg_ratio > 1.1:
            print("✓ Model shows good uncertainty calibration (higher on mistakes)")
        else:
            print("⚠ Model uncertainty may need improvement")
    
    print(f"\nAll results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()