"""
Semi-Supervised Model Trainer.
Extends the existing trainer to handle semi-supervised learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pickle

from src.models.semi_supervised import (
    PseudoLabelingSemiSupervised,
    CoTrainingSemiSupervised,
    MultiLabelSemiSupervised
)
from src.data.data_manager import DataManager


class SemiSupervisedTrainer:
    """
    Trainer for semi-supervised models.
    Leverages both labeled and unlabeled dissent observations.
    """
    
    def __init__(
        self,
        method: str = "pseudo_labeling",  # or "co_training"
        rationales: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize semi-supervised trainer.
        
        Args:
            method: 'pseudo_labeling' or 'co_training'
            rationales: List of target rationales
            config: Configuration dict with:
                - base_model_type: Base supervised model
                - confidence_threshold: Threshold for pseudo-labels
                - max_iterations: Max self-training iterations
                - min_pseudo_labels: Min pseudo-labels per iteration
                - other model-specific params
        """
        self.method = method
        self.rationales = rationales
        self.config = config or {}
        self.models = {}
        self.data_manager = None
        self.training_info = {}
    
    def train_single_rationale(
        self,
        train_df: pd.DataFrame,
        unlabeled_df: pd.DataFrame,
        rationale: str,
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """
        Train semi-supervised model for single rationale.
        
        Args:
            train_df: Labeled training data (dissent rows with known rationale)
            unlabeled_df: Unlabeled data (dissent rows with missing rationale)
            rationale: Target rationale
            data_manager: DataManager instance
            val_df: Validation data (optional)
            verbose: Print progress
        
        Returns:
            Trained semi-supervised model
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Semi-Supervised Training: {rationale}")
            print(f"{'='*80}")
        
        # Prepare labeled data
        train_clean = train_df.dropna(subset=[rationale]).copy()
        
        if len(train_clean) == 0:
            print(f"No labeled data for {rationale}")
            return None
        
        X_labeled, y_labeled, feature_names = data_manager.prepare_for_training(
            train_clean,
            [rationale],
            fit=True,
            drop_high_missing=self.config.get('drop_high_missing', 1.0),
            use_all_features=self.config.get('use_all_features', False),
            exclude_cols=self.config.get('exclude_cols', None),
            missing_strategy=self.config.get('missing_strategy', 'zero'),
            verbose=verbose
        )
        
        y_labeled = y_labeled[:, 0] if y_labeled.ndim > 1 else y_labeled
        
        # Prepare unlabeled data (dissent rows where this rationale is missing)
        unlabeled_clean = unlabeled_df[unlabeled_df[rationale].isna()].copy()
        
        if len(unlabeled_clean) == 0:
            if verbose:
                print(f"No unlabeled data for {rationale}. Using supervised learning.")
            # Fall back to supervised
            from src.models.base_model import SupervisedRationaleModel
            model = SupervisedRationaleModel(
                rationale=rationale,
                base_model_type=self.config.get('base_model_type', 'logistic'),
                calibrate=self.config.get('calibrate', True),
                custom_params=self.config.get('custom_params'),
                random_seed=self.config.get('random_seed', 21),
            )
            model.feature_names = feature_names
            model.fit(X_labeled, y_labeled, verbose=verbose)
            return model
        
        X_unlabeled, _, _ = data_manager.prepare_for_training(
            unlabeled_clean,
            [rationale],
            fit=False,
            missing_strategy=self.config.get('missing_strategy', 'zero'),
        )
        
        if verbose:
            print(f"Labeled: {len(y_labeled)} ({y_labeled.sum()} positive)")
            print(f"Unlabeled: {len(X_unlabeled)}")
        
        # Prepare validation data if available
        X_val, y_val = None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=[rationale]).copy()
            if len(val_clean) > 0:
                X_val, y_val_arr, _ = data_manager.prepare_for_training(
                    val_clean,
                    [rationale],
                    fit=False,
                    missing_strategy=self.config.get('missing_strategy', 'zero'),
                )
                y_val = y_val_arr[:, 0] if y_val_arr.ndim > 1 else y_val_arr
        
        # Create semi-supervised model
        if self.method == "pseudo_labeling":
            model = PseudoLabelingSemiSupervised(
                rationale=rationale,
                base_model_type=self.config.get('base_model_type', 'logistic'),
                confidence_threshold=self.config.get('confidence_threshold', 0.9),
                max_iterations=self.config.get('max_iterations', 5),
                min_pseudo_labels=self.config.get('min_pseudo_labels', 10),
                calibrate=self.config.get('calibrate', True),
                custom_params=self.config.get('custom_params'),
                random_seed=self.config.get('random_seed', 21),
            )
        elif self.method == "co_training":
            model = CoTrainingSemiSupervised(
                rationale=rationale,
                base_model_type=self.config.get('base_model_type', 'logistic'),
                agreement_threshold=self.config.get('agreement_threshold', 0.1),
                confidence_threshold=self.config.get('confidence_threshold', 0.8),
                max_iterations=self.config.get('max_iterations', 5),
                min_pseudo_labels=self.config.get('min_pseudo_labels', 10),
                feature_split_ratio=self.config.get('feature_split_ratio', 0.5),
                calibrate=self.config.get('calibrate', True),
                custom_params=self.config.get('custom_params'),
                random_seed=self.config.get('random_seed', 21),
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Train
        model.feature_names = feature_names
        model.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val, verbose=verbose)
        
        # Store training info
        self.training_info[rationale] = {
            'n_labeled': len(y_labeled),
            'n_unlabeled': len(X_unlabeled),
            'n_positive_labeled': int(y_labeled.sum()),
            'positive_rate_labeled': float(y_labeled.mean()),
            'n_features': X_labeled.shape[1],
            'method': self.method,
            'training_history': model.training_history if hasattr(model, 'training_history') else [],
        }
        
        return model
    
    def train_all(
        self,
        train_df: pd.DataFrame,
        unlabeled_df: pd.DataFrame,
        rationales: List[str],
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Train semi-supervised models for all rationales.
        
        Args:
            train_df: Labeled training data
            unlabeled_df: Unlabeled dissent data (missing rationales)
            rationales: List of rationales to predict
            data_manager: DataManager instance
            val_df: Validation data (optional)
            save_dir: Directory to save models
        
        Returns:
            Dictionary of trained models
        """
        self.data_manager = data_manager
        self.rationales = rationales
        
        for rationale in rationales:
            model = self.train_single_rationale(
                train_df=train_df,
                unlabeled_df=unlabeled_df,
                rationale=rationale,
                data_manager=data_manager,
                val_df=val_df,
            )
            
            if model is not None:
                self.models[rationale] = model
                
                # Save individual model if requested
                if save_dir:
                    save_dir = Path(save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    model.save(str(save_dir / f"{rationale}_model.pkl"))
        
        # Save data manager and training info
        if save_dir:
            with open(save_dir / "data_manager.pkl", 'wb') as f:
                pickle.dump(data_manager, f)
            
            with open(save_dir / "training_info.json", 'w') as f:
                # Convert training history to serializable format
                serializable_info = {}
                for rat, info in self.training_info.items():
                    serializable_info[rat] = {
                        k: v for k, v in info.items()
                        if k != 'training_history'
                    }
                    if 'training_history' in info:
                        serializable_info[rat]['n_iterations'] = len(info['training_history'])
                        if info['training_history']:
                            last_iter = info['training_history'][-1]
                            serializable_info[rat]['final_pseudo_labels'] = last_iter.get('n_pseudo', 0)
                
                json.dump(serializable_info, f, indent=2)
        
        # Print summary
        self._print_training_summary()
        
        return self.models
    
    def _print_training_summary(self):
        """Print training summary."""
        if not self.training_info:
            return
        
        print(f"\n{'='*80}")
        print(f"SEMI-SUPERVISED TRAINING SUMMARY ({self.method.upper()})")
        print(f"{'='*80}")
        
        summary_data = []
        for rationale, info in self.training_info.items():
            summary_data.append({
                'rationale': rationale,
                'n_labeled': info['n_labeled'],
                'n_unlabeled': info['n_unlabeled'],
                'n_positive': info['n_positive_labeled'],
                'pos_rate': info['positive_rate_labeled'],
                'n_features': info['n_features'],
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Print iteration details if available
        print(f"\n{'='*80}")
        print("PSEUDO-LABELING ITERATIONS")
        print(f"{'='*80}")
        
        for rationale, info in self.training_info.items():
            if 'training_history' in info and info['training_history']:
                print(f"\n{rationale}:")
                history_df = pd.DataFrame(info['training_history'])
                print(history_df.to_string(index=False))
        
        print(f"{'='*80}\n")


def prepare_semi_supervised_data(
    data_manager: DataManager,
    rationales: List[str],
    test_size: float = 0.2,
    random_seed: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for semi-supervised learning.
    
    Returns:
        train_df: Labeled training data (dissent with known rationales)
        unlabeled_df: Unlabeled data (dissent with missing rationales)
        test_df: Test data (for evaluation)
    """
    # Get all labeled data
    labeled_df = data_manager.get_labeled_data(rationales)
    
    # Split labeled data into train/test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        labeled_df,
        test_size=test_size,
        random_state=random_seed,
    )
    
    # Get unlabeled data (dissent rows with all rationales missing)
    unlabeled_df = data_manager.get_unlabeled_data(rationales)
    
    print(f"\n{'='*80}")
    print("SEMI-SUPERVISED DATA SPLIT")
    print(f"{'='*80}")
    print(f"Labeled train: {len(train_df):,}")
    print(f"Unlabeled (for pseudo-labeling): {len(unlabeled_df):,}")
    print(f"Test: {len(test_df):,}")
    print(f"{'='*80}\n")
    
    return train_df, unlabeled_df, test_df