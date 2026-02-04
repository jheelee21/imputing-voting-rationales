"""
Unified predictor for generating predictions on unlabeled data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.base_model import BaseRationaleModel, SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel
from src.data.data_manager import DataManager


class Predictor:
    """Unified interface for making predictions with any model type."""
    
    def __init__(
        self,
        id_columns: List[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            id_columns: Columns to include in output (e.g., investor_id, meeting_id)
            batch_size: Process in batches for large datasets
        """
        self.id_columns = id_columns or ["investor_id", "pid", "ProxySeason", "meeting_id"]
        self.batch_size = batch_size
    
    def predict_supervised(
        self,
        models: Dict[str, SupervisedRationaleModel],
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
    ) -> pd.DataFrame:
        """
        Generate predictions using supervised models (one per rationale).
        
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
        
        print(f"\n{'='*80}")
        print("GENERATING PREDICTIONS (Supervised)")
        print(f"{'='*80}")
        print(f"Unlabeled samples: {len(unlabeled_df):,}")
        print(f"Rationales: {list(models.keys())}")
        print(f"{'='*80}\n")
        
        # Predict for each rationale
        for rationale, model in models.items():
            print(f"Predicting {rationale}...", end=" ", flush=True)
            
            try:
                # Prepare features
                X, _, _ = data_manager.prepare_for_training(
                    unlabeled_df, [rationale], fit=False
                )
                
                # Predict
                y_prob = model.predict_proba(X)
                predictions_df[f"{rationale}_prob"] = y_prob
                
                print(f"✓ (mean: {y_prob.mean():.3f}, std: {y_prob.std():.3f})")
            
            except Exception as e:
                print(f"✗ Error: {e}")
                predictions_df[f"{rationale}_prob"] = np.nan
        
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
        
        print(f"\n{'='*80}")
        print("GENERATING PREDICTIONS (MC Dropout)")
        print(f"{'='*80}")
        print(f"Unlabeled samples: {len(unlabeled_df):,}")
        print(f"Rationales: {rationales}")
        print(f"MC samples: {num_samples}")
        print(f"{'='*80}\n")
        
        # Prepare features
        X, _, _ = data_manager.prepare_for_training(
            unlabeled_df, rationales, fit=False
        )
        
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
                print(f"  Prob: mean={mean_probs[:, i].mean():.3f}, std={mean_probs[:, i].std():.3f}")
                print(f"  Unc:  mean={epistemic_unc[:, i].mean():.3f}")
        
        else:
            mean_probs = model.predict_proba(X, num_samples=num_samples)
            
            for i, rationale in enumerate(rationales):
                predictions_df[f"{rationale}_prob"] = mean_probs[:, i]
                print(f"{rationale}: mean={mean_probs[:, i].mean():.3f}")
        
        return predictions_df
    
    def predict(
        self,
        models: Dict[str, BaseRationaleModel],
        unlabeled_df: pd.DataFrame,
        data_manager: DataManager,
        **kwargs
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
        
        # Determine model type
        first_model = next(iter(models.values()))
        
        if isinstance(first_model, MCDropoutModel):
            return self.predict_mc_dropout(
                model=first_model,
                unlabeled_df=unlabeled_df,
                data_manager=data_manager,
                **kwargs
            )
        
        elif isinstance(first_model, SupervisedRationaleModel):
            return self.predict_supervised(
                models=models,
                unlabeled_df=unlabeled_df,
                data_manager=data_manager,
            )
        
        else:
            raise ValueError(f"Unknown model type: {type(first_model)}")
    
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
        
        print(f"\n{'='*80}")
        print("PREDICTION ANALYSIS")
        print(f"{'='*80}\n")
        
        # Confidence statistics
        confidence_stats = []
        
        for rationale in rationales:
            prob_col = f"{rationale}_prob"
            
            if prob_col not in predictions_df.columns:
                continue
            
            probs = predictions_df[prob_col].dropna()
            
            confidence_stats.append({
                'rationale': rationale,
                'n_predictions': len(probs),
                'mean': probs.mean(),
                'std': probs.std(),
                'min': probs.min(),
                'q25': probs.quantile(0.25),
                'median': probs.quantile(0.50),
                'q75': probs.quantile(0.75),
                'max': probs.max(),
                'pct_high_conf': (probs >= 0.7).mean() * 100,
                'pct_medium_conf': ((probs >= 0.3) & (probs < 0.7)).mean() * 100,
                'pct_low_conf': (probs < 0.3).mean() * 100,
            })
        
        confidence_df = pd.DataFrame(confidence_stats)
        
        print("Confidence Statistics:")
        print(confidence_df[['rationale', 'mean', 'std', 'median']].to_string(index=False))
        
        # Multi-label analysis (at 0.5 threshold)
        prob_cols = [f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns]
        
        if prob_cols:
            binary_preds = (predictions_df[prob_cols] >= 0.5).astype(int)
            n_rationales_per_obs = binary_preds.sum(axis=1)
            
            multi_label_stats = {
                'total_observations': len(predictions_df),
                'mean_rationales_per_obs': n_rationales_per_obs.mean(),
                'median_rationales_per_obs': n_rationales_per_obs.median(),
            }
            
            for n in range(5):
                count = (n_rationales_per_obs == n).sum()
                pct = count / len(predictions_df) * 100
                multi_label_stats[f'n_with_{n}_rationales'] = int(count)
                multi_label_stats[f'pct_with_{n}_rationales'] = pct
            
            print(f"\nMulti-Label Distribution (threshold=0.5):")
            print(f"  Mean rationales per observation: {multi_label_stats['mean_rationales_per_obs']:.2f}")
            for n in range(5):
                count = multi_label_stats[f'n_with_{n}_rationales']
                pct = multi_label_stats[f'pct_with_{n}_rationales']
                print(f"  {n} rationales: {count:,} ({pct:.1f}%)")
        
        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            confidence_df.to_csv(output_dir / 'confidence_statistics.csv', index=False)
            print(f"\nStatistics saved to {output_dir}")
        
        return {
            'confidence': confidence_df,
            'multi_label': multi_label_stats if prob_cols else None,
        }