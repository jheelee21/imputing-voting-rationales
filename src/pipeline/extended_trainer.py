"""
Extended model trainer supporting all probabilistic models.
Handles BNN, Calibrated Boosting, Gaussian Processes, and existing models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pickle

from src.models.base_model import SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel
from src.data.data_manager import DataManager

# Import new models
try:
    from src.models.bnn_model import BNNModel
    BNN_AVAILABLE = True
except ImportError:
    BNN_AVAILABLE = False
    print("Warning: BNN model not available")

try:
    from src.models.calibrated_boosting import CalibratedBoostingModel, MultiLabelCalibratedBoosting
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False
    print("Warning: Calibrated Boosting models not available")

try:
    from src.models.gaussian_process import GPModel, MultiLabelGP
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("Warning: Gaussian Process models not available")


class ExtendedModelTrainer:
    """
    Extended trainer for all probabilistic models.
    
    Supported model types:
    - 'logistic', 'random_forest', 'gradient_boosting': Original supervised models
    - 'mc_dropout': MC Dropout neural network
    - 'bnn': Bayesian Neural Network
    - 'catboost', 'lightgbm', 'xgboost': Calibrated boosting models
    - 'sparse_gp', 'deep_kernel_gp': Gaussian Process models
    """
    
    def __init__(
        self,
        model_type: str,
        rationales: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        self.model_type = model_type
        self.rationales = rationales
        self.config = config or {}
        self.models = {}
        self.data_manager = None
        self.training_info = {}
        
        # Validate model type
        self._validate_model_type()
    
    def _validate_model_type(self):
        """Check if requested model type is available."""
        boosting_models = ['catboost', 'lightgbm', 'xgboost']
        gp_models = ['sparse_gp', 'deep_kernel_gp']
        
        if self.model_type == 'bnn' and not BNN_AVAILABLE:
            raise ImportError("BNN not available. Install pyro-ppl")
        
        if self.model_type in boosting_models and not BOOSTING_AVAILABLE:
            raise ImportError(f"{self.model_type} not available. Install required package")
        
        if self.model_type in gp_models and not GP_AVAILABLE:
            raise ImportError("GP models not available. Install gpytorch")
    
    def train_supervised(
        self,
        train_df: pd.DataFrame,
        rationale: str,
        data_manager: DataManager,
        verbose: bool = True,
    ):
        """Train single supervised model (original implementation)."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training {self.model_type.upper()} for: {rationale}")
            print(f"{'='*80}")
        
        train_clean = train_df.dropna(subset=[rationale]).copy()
        
        if len(train_clean) == 0:
            print(f"No training data for {rationale}")
            return None
        
        X, y, feature_names = data_manager.prepare_for_training(
            train_clean, 
            [rationale], 
            fit=True,
            drop_high_missing=self.config.get('drop_high_missing', 1.0),
            use_all_features=self.config.get('use_all_features', False),
            exclude_cols=self.config.get('exclude_cols', None),
            missing_strategy=self.config.get('missing_strategy', 'zero'),
            verbose=verbose
        )
        
        y_single = y[:, 0] if y.ndim > 1 else y
        
        if verbose:
            print(f"Samples: {len(y_single)}, Positive: {y_single.sum()} ({y_single.mean():.2%})")
        
        model = SupervisedRationaleModel(
            rationale=rationale,
            base_model_type=self.model_type,
            calibrate=self.config.get('calibrate', True),
            custom_params=self.config.get('custom_params'),
            random_seed=self.config.get('random_seed', 21),
        )
        
        model.feature_names = feature_names
        model.fit(X, y_single, verbose=verbose)
        
        self.training_info[rationale] = {
            'n_train': len(y_single),
            'n_positive': int(y_single.sum()),
            'positive_rate': float(y_single.mean()),
            'n_features': X.shape[1],
            'train_accuracy': (model.predict(X) == y_single).mean() if model.is_fitted else 0.0,
        }
        
        return model
    
    def train_calibrated_boosting(
        self,
        train_df: pd.DataFrame,
        rationale: str,
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """Train calibrated boosting model for single rationale."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training {self.model_type.upper()} (Calibrated) for: {rationale}")
            print(f"{'='*80}")
        
        train_clean = train_df.dropna(subset=[rationale]).copy()
        
        if len(train_clean) == 0:
            print(f"No training data for {rationale}")
            return None
        
        X, y, feature_names = data_manager.prepare_for_training(
            train_clean, 
            [rationale], 
            fit=True,
            drop_high_missing=self.config.get('drop_high_missing', 1.0),
            use_all_features=self.config.get('use_all_features', False),
            exclude_cols=self.config.get('exclude_cols', None),
            missing_strategy=self.config.get('missing_strategy', 'zero'),
            verbose=verbose
        )
        
        y_single = y[:, 0] if y.ndim > 1 else y
        
        # Prepare validation data if available
        X_val, y_val = None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=[rationale]).copy()
            X_val, y_val_arr, _ = data_manager.prepare_for_training(
                val_clean, [rationale], fit=False,
                missing_strategy=self.config.get('missing_strategy', 'zero'),
            )
            y_val = y_val_arr[:, 0] if y_val_arr.ndim > 1 else y_val_arr
        
        model = CalibratedBoostingModel(
            rationale=rationale,
            base_model_type=self.model_type,
            calibrate=self.config.get('calibrate', True),
            calibration_method=self.config.get('calibration_method', 'isotonic'),
            custom_params=self.config.get('custom_params'),
            random_seed=self.config.get('random_seed', 21),
        )
        
        model.feature_names = feature_names
        model.fit(X, y_single, X_val, y_val, verbose=verbose)
        
        self.training_info[rationale] = {
            'n_train': len(y_single),
            'n_positive': int(y_single.sum()),
            'positive_rate': float(y_single.mean()),
            'n_features': X.shape[1],
        }
        
        return model
    
    def train_gp(
        self,
        train_df: pd.DataFrame,
        rationale: str,
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """Train Gaussian Process model for single rationale."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training {self.model_type.upper()} for: {rationale}")
            print(f"{'='*80}")
        
        train_clean = train_df.dropna(subset=[rationale]).copy()
        
        if len(train_clean) == 0:
            print(f"No training data for {rationale}")
            return None
        
        X, y, feature_names = data_manager.prepare_for_training(
            train_clean, 
            [rationale], 
            fit=True,
            drop_high_missing=self.config.get('drop_high_missing', 1.0),
            use_all_features=self.config.get('use_all_features', False),
            exclude_cols=self.config.get('exclude_cols', None),
            missing_strategy=self.config.get('missing_strategy', 'zero'),
            verbose=verbose
        )
        
        y_single = y[:, 0] if y.ndim > 1 else y
        
        # Determine GP model type
        gp_type = 'sparse_gp' if self.model_type == 'sparse_gp' else 'deep_kernel'
        
        model = GPModel(
            rationale=rationale,
            model_type=gp_type,
            kernel_type=self.config.get('kernel_type', 'rbf'),
            num_inducing=self.config.get('num_inducing', 500),
            learning_rate=self.config.get('learning_rate', 0.01),
            num_epochs=self.config.get('num_epochs', 100),
            batch_size=self.config.get('batch_size', 256),
            hidden_dims=self.config.get('hidden_dims', [64, 32]),
            use_ard=self.config.get('use_ard', True),
            random_seed=self.config.get('random_seed', 21),
        )
        
        model.feature_names = feature_names
        model.fit(X, y_single, verbose=verbose)
        
        self.training_info[rationale] = {
            'n_train': len(y_single),
            'n_positive': int(y_single.sum()),
            'positive_rate': float(y_single.mean()),
            'n_features': X.shape[1],
        }
        
        return model
    
    def train_mc_dropout(
        self,
        train_df: pd.DataFrame,
        rationales: List[str],
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """Train MC Dropout model (multi-label)."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training MC DROPOUT for: {rationales}")
            print(f"{'='*80}")
        
        train_clean = train_df.dropna(subset=rationales, how='all').copy()
        X_train, y_train, feature_names = data_manager.prepare_for_training(
            train_clean, 
            rationales, 
            fit=True,
            drop_high_missing=self.config.get('drop_high_missing', 1.0),
            use_all_features=self.config.get('use_all_features', False),
            exclude_cols=self.config.get('exclude_cols', None),
            missing_strategy=self.config.get('missing_strategy', 'zero'),
            verbose=verbose
        )
        
        # Mask for partial labels: only compute loss on observed rationales
        label_mask = train_clean[rationales].notna().values.astype(bool)
        
        X_val, y_val = None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=rationales, how='all').copy()
            X_val, y_val, _ = data_manager.prepare_for_training(
                val_clean, rationales, fit=False,
                missing_strategy=self.config.get('missing_strategy', 'zero'),
            )
        
        if verbose:
            obs_frac = label_mask.mean()
            print(f"Label coverage: {obs_frac:.1%} of rationale cells observed (masked loss)")
        
        model = MCDropoutModel(
            rationales=rationales,
            hidden_dims=self.config.get('hidden_dims', [64, 32]),
            dropout_rate=self.config.get('dropout_rate', 0.2),
            learning_rate=self.config.get('learning_rate', 0.001),
            num_epochs=self.config.get('num_epochs', 100),
            batch_size=self.config.get('batch_size', 256),
            num_samples=self.config.get('num_samples', 50),
            weight_decay=self.config.get('weight_decay', 1e-4),
            random_seed=self.config.get('random_seed', 21),
        )
        
        model.feature_names = feature_names
        model.fit(X_train, y_train, X_val, y_val, mask=label_mask, verbose=verbose)
        
        self.training_info['mc_dropout'] = {
            'n_train': len(y_train),
            'n_val': len(y_val) if y_val is not None else 0,
            'n_features': X_train.shape[1],
            'rationales': rationales,
        }
        
        return model
    
    def train_bnn(
        self,
        train_df: pd.DataFrame,
        rationales: List[str],
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """Train Bayesian Neural Network (multi-label)."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training BNN for: {rationales}")
            print(f"{'='*80}")
        
        train_clean = train_df.dropna(subset=rationales, how='all').copy()
        X_train, y_train, feature_names = data_manager.prepare_for_training(
            train_clean, 
            rationales, 
            fit=True,
            drop_high_missing=self.config.get('drop_high_missing', 1.0),
            use_all_features=self.config.get('use_all_features', False),
            exclude_cols=self.config.get('exclude_cols', None),
            missing_strategy=self.config.get('missing_strategy', 'zero'),
            verbose=verbose
        )
        
        X_val, y_val = None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=rationales, how='all').copy()
            X_val, y_val, _ = data_manager.prepare_for_training(
                val_clean, rationales, fit=False,
                missing_strategy=self.config.get('missing_strategy', 'zero'),
            )
        
        model = BNNModel(
            rationales=rationales,
            hidden_dims=self.config.get('hidden_dims', [64, 32]),
            prior_scale=self.config.get('prior_scale', 1.0),
            learning_rate=self.config.get('learning_rate', 0.01),
            num_epochs=self.config.get('num_epochs', 100),
            batch_size=self.config.get('batch_size', 256),
            num_samples=self.config.get('num_samples', 100),
            random_seed=self.config.get('random_seed', 21),
        )
        
        model.feature_names = feature_names
        model.fit(X_train, y_train, X_val, y_val, verbose=verbose)
        
        self.training_info['bnn'] = {
            'n_train': len(y_train),
            'n_val': len(y_val) if y_val is not None else 0,
            'n_features': X_train.shape[1],
            'rationales': rationales,
        }
        
        return model
    
    def train_all(
        self,
        train_df: pd.DataFrame,
        rationales: List[str],
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Train all models based on model_type."""
        self.data_manager = data_manager
        self.rationales = rationales
        
        # Multi-label models
        if self.model_type in ['mc_dropout', 'bnn']:
            if self.model_type == 'mc_dropout':
                model = self.train_mc_dropout(train_df, rationales, data_manager, val_df)
            else:  # bnn
                model = self.train_bnn(train_df, rationales, data_manager, val_df)
            
            self.models[self.model_type] = model
            
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(save_dir / f"{self.model_type}_model.pkl"))
                
                with open(save_dir / "data_manager.pkl", 'wb') as f:
                    pickle.dump(data_manager, f)
                
                with open(save_dir / "training_info.json", 'w') as f:
                    json.dump(self.training_info, f, indent=2)
        
        # Single-label models (one per rationale)
        else:
            for rationale in rationales:
                # Select appropriate training method
                if self.model_type in ['catboost', 'lightgbm', 'xgboost']:
                    model = self.train_calibrated_boosting(train_df, rationale, data_manager, val_df)
                elif self.model_type in ['sparse_gp', 'deep_kernel_gp']:
                    model = self.train_gp(train_df, rationale, data_manager, val_df)
                else:
                    model = self.train_supervised(train_df, rationale, data_manager)
                
                if model is not None:
                    self.models[rationale] = model
                    
                    if save_dir:
                        save_dir = Path(save_dir)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        model.save(str(save_dir / f"{rationale}_model.pkl"))
            
            if save_dir:
                with open(save_dir / "data_manager.pkl", 'wb') as f:
                    pickle.dump(data_manager, f)
                
                with open(save_dir / "training_info.json", 'w') as f:
                    json.dump(self.training_info, f, indent=2)
        
        self._print_training_summary()
        return self.models
    
    def _print_training_summary(self):
        """Print training summary."""
        if not self.training_info:
            return
        
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        
        if self.model_type in ['mc_dropout', 'bnn']:
            info = self.training_info[self.model_type]
            print(f"Model: {self.model_type.upper()} (multi-label)")
            print(f"Rationales: {', '.join(info['rationales'])}")
            print(f"Train samples: {info['n_train']:,}")
            print(f"Val samples: {info['n_val']:,}")
            print(f"Features: {info['n_features']}")
        else:
            summary_df = pd.DataFrame(self.training_info).T
            print(summary_df.to_string())
        
        print(f"{'='*80}\n")