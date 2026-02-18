"""
Extended model trainer supporting all probabilistic models.
Handles BNN, Calibrated Boosting, Gaussian Processes, PCA, and existing models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pickle

from src.models.supervised import SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel
from src.data.data_manager import DataManager

try:
    from src.models.bayesian_hierarchial import HierarchicalModel

    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    print("Warning: Hierarchical model not available")


try:
    from src.models.bnn_model import BNNModel

    BNN_AVAILABLE = True
except ImportError:
    BNN_AVAILABLE = False
    print("Warning: BNN model not available")

try:
    from src.models.calibrated_boosting import (
        CalibratedBoostingModel,
        MultiLabelCalibratedBoosting,
    )

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

try:
    from src.models.pca_model import PCAModel

    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
    print("Warning: PCA model not available")


class ExtendedModelTrainer:
    """
    Extended trainer for all probabilistic models.

    Supported model types:
    - 'logistic', 'random_forest', 'gradient_boosting': Original supervised models
    - 'mc_dropout': MC Dropout neural network
    - 'bnn': Bayesian Neural Network
    - 'catboost', 'lightgbm', 'xgboost': Calibrated boosting models
    - 'sparse_gp', 'deep_kernel_gp': Gaussian Process models
    - 'pca': PCA + Classifier
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
        boosting_models = ["catboost", "lightgbm", "xgboost"]
        gp_models = ["sparse_gp", "deep_kernel_gp"]

        if self.model_type == "bnn" and not BNN_AVAILABLE:
            raise ImportError("BNN not available. Install pyro-ppl")

        if self.model_type in boosting_models and not BOOSTING_AVAILABLE:
            raise ImportError(
                f"{self.model_type} not available. Install required package"
            )

        if self.model_type in gp_models and not GP_AVAILABLE:
            raise ImportError("GP models not available. Install gpytorch")

        if self.model_type == "pca" and not PCA_AVAILABLE:
            raise ImportError("PCA model not available. Check src/models/pca_model.py")

    def train_supervised(
        self,
        train_df: pd.DataFrame,
        rationale: str,
        data_manager: DataManager,
        verbose: bool = True,
    ):
        """Train single supervised model (original implementation)."""
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Training {self.model_type.upper()} for: {rationale}")
            print(f"{'=' * 80}")

        train_clean = train_df.dropna(subset=[rationale]).copy()

        if len(train_clean) == 0:
            print(f"No training data for {rationale}")
            return None

        X, y, feature_names = data_manager.prepare_for_training(
            train_clean,
            [rationale],
            fit=True,
            drop_high_missing=self.config.get("drop_high_missing", 1.0),
            use_all_features=self.config.get("use_all_features", False),
            exclude_cols=self.config.get("exclude_cols", None),
            missing_strategy=self.config.get("missing_strategy", "zero"),
            verbose=verbose,
        )

        y_single = y[:, 0] if y.ndim > 1 else y

        if verbose:
            print(
                f"Samples: {len(y_single)}, Positive: {y_single.sum()} ({y_single.mean():.2%})"
            )

        model = SupervisedRationaleModel(
            rationale=rationale,
            base_model_type=self.model_type,
            calibrate=self.config.get("calibrate", True),
            custom_params=self.config.get("custom_params"),
            random_seed=self.config.get("random_seed", 21),
        )

        model.feature_names = feature_names
        model.fit(X, y_single, verbose=verbose)

        self.training_info[rationale] = {
            "n_train": len(y_single),
            "n_positive": int(y_single.sum()),
            "positive_rate": float(y_single.mean()),
            "n_features": X.shape[1],
            "train_accuracy": (model.predict(X) == y_single).mean()
            if model.is_fitted
            else 0.0,
        }

        return model

    def train_pca(
        self,
        train_df: pd.DataFrame,
        rationale: str,
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """Train PCA model for single rationale."""
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Training PCA for: {rationale}")
            print(f"{'=' * 80}")

        train_clean = train_df.dropna(subset=[rationale]).copy()

        if len(train_clean) == 0:
            print(f"No training data for {rationale}")
            return None

        X, y, feature_names = data_manager.prepare_for_training(
            train_clean,
            [rationale],
            fit=True,
            drop_high_missing=self.config.get("drop_high_missing", 1.0),
            use_all_features=self.config.get("use_all_features", False),
            exclude_cols=self.config.get("exclude_cols", None),
            missing_strategy=self.config.get("missing_strategy", "zero"),
            verbose=verbose,
        )

        y_single = y[:, 0] if y.ndim > 1 else y

        # Prepare validation data if available
        X_val, y_val = None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=[rationale]).copy()
            if len(val_clean) > 0:
                X_val, y_val_arr, _ = data_manager.prepare_for_training(
                    val_clean,
                    [rationale],
                    fit=False,
                    missing_strategy=self.config.get("missing_strategy", "zero"),
                )
                y_val = y_val_arr[:, 0] if y_val_arr.ndim > 1 else y_val_arr

        model = PCAModel(
            rationale=rationale,
            n_components=self.config.get("n_components"),
            variance_threshold=self.config.get("variance_threshold", 0.95),
            base_model_type=self.config.get("pca_classifier", "logistic"),
            calibrate=self.config.get("calibrate", True),
            whiten=self.config.get("whiten", False),
            custom_params=self.config.get("custom_params"),
            random_seed=self.config.get("random_seed", 21),
        )

        model.feature_names = feature_names
        model.fit(X, y_single, X_val, y_val, verbose=verbose)

        self.training_info[rationale] = {
            "n_train": len(y_single),
            "n_positive": int(y_single.sum()),
            "positive_rate": float(y_single.mean()),
            "n_features_original": X.shape[1],
            "n_components": model.n_components_used_,
            "explained_variance": float(model.explained_variance_ratio_.sum()),
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
            print(f"\n{'=' * 80}")
            print(f"Training {self.model_type.upper()} (Calibrated) for: {rationale}")
            print(f"{'=' * 80}")

        train_clean = train_df.dropna(subset=[rationale]).copy()

        if len(train_clean) == 0:
            print(f"No training data for {rationale}")
            return None

        X, y, feature_names = data_manager.prepare_for_training(
            train_clean,
            [rationale],
            fit=True,
            drop_high_missing=self.config.get("drop_high_missing", 1.0),
            use_all_features=self.config.get("use_all_features", False),
            exclude_cols=self.config.get("exclude_cols", None),
            missing_strategy=self.config.get("missing_strategy", "zero"),
            verbose=verbose,
        )

        y_single = y[:, 0] if y.ndim > 1 else y

        # Prepare validation data if available
        X_val, y_val = None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=[rationale]).copy()
            X_val, y_val_arr, _ = data_manager.prepare_for_training(
                val_clean,
                [rationale],
                fit=False,
                missing_strategy=self.config.get("missing_strategy", "zero"),
            )
            y_val = y_val_arr[:, 0] if y_val_arr.ndim > 1 else y_val_arr

        model = CalibratedBoostingModel(
            rationale=rationale,
            base_model_type=self.model_type,
            calibrate=self.config.get("calibrate", True),
            calibration_method=self.config.get("calibration_method", "isotonic"),
            custom_params=self.config.get("custom_params"),
            random_seed=self.config.get("random_seed", 21),
        )

        model.feature_names = feature_names
        model.fit(X, y_single, X_val, y_val, verbose=verbose)

        self.training_info[rationale] = {
            "n_train": len(y_single),
            "n_positive": int(y_single.sum()),
            "positive_rate": float(y_single.mean()),
            "n_features": X.shape[1],
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
            print(f"\n{'=' * 80}")
            print(f"Training {self.model_type.upper()} for: {rationale}")
            print(f"{'=' * 80}")

        train_clean = train_df.dropna(subset=[rationale]).copy()

        if len(train_clean) == 0:
            print(f"No training data for {rationale}")
            return None

        X, y, feature_names = data_manager.prepare_for_training(
            train_clean,
            [rationale],
            fit=True,
            drop_high_missing=self.config.get("drop_high_missing", 1.0),
            use_all_features=self.config.get("use_all_features", False),
            exclude_cols=self.config.get("exclude_cols", None),
            missing_strategy=self.config.get("missing_strategy", "zero"),
            verbose=verbose,
        )

        y_single = y[:, 0] if y.ndim > 1 else y

        # Determine GP model type
        gp_type = "sparse_gp" if self.model_type == "sparse_gp" else "deep_kernel"

        model = GPModel(
            rationale=rationale,
            model_type=gp_type,
            kernel_type=self.config.get("kernel_type", "rbf"),
            num_inducing=self.config.get("num_inducing", 500),
            learning_rate=self.config.get("learning_rate", 0.01),
            num_epochs=self.config.get("num_epochs", 100),
            batch_size=self.config.get("batch_size", 256),
            hidden_dims=self.config.get("hidden_dims", [64, 32]),
            use_ard=self.config.get("use_ard", True),
            random_seed=self.config.get("random_seed", 21),
        )

        model.feature_names = feature_names
        model.fit(X, y_single, verbose=verbose)

        self.training_info[rationale] = {
            "n_train": len(y_single),
            "n_positive": int(y_single.sum()),
            "positive_rate": float(y_single.mean()),
            "n_features": X.shape[1],
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
            print(f"\n{'=' * 80}")
            print(f"Training MC DROPOUT for: {rationales}")
            print(f"{'=' * 80}")

        train_clean = train_df.dropna(subset=rationales, how="all").copy()
        X_train, y_train, feature_names = data_manager.prepare_for_training(
            train_clean,
            rationales,
            fit=True,
            drop_high_missing=self.config.get("drop_high_missing", 1.0),
            use_all_features=self.config.get("use_all_features", False),
            exclude_cols=self.config.get("exclude_cols", None),
            missing_strategy=self.config.get("missing_strategy", "zero"),
            verbose=verbose,
        )

        # Mask for partial labels: only compute loss on observed rationales
        label_mask = train_clean[rationales].notna().values.astype(bool)

        X_val, y_val = None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=rationales, how="all").copy()
            X_val, y_val, _ = data_manager.prepare_for_training(
                val_clean,
                rationales,
                fit=False,
                missing_strategy=self.config.get("missing_strategy", "zero"),
            )

        if verbose:
            obs_frac = label_mask.mean()
            print(
                f"Label coverage: {obs_frac:.1%} of rationale cells observed (masked loss)"
            )

        model = MCDropoutModel(
            rationales=rationales,
            hidden_dims=self.config.get("hidden_dims", [64, 32]),
            dropout_rate=self.config.get("dropout_rate", 0.2),
            learning_rate=self.config.get("learning_rate", 0.001),
            num_epochs=self.config.get("num_epochs", 100),
            batch_size=self.config.get("batch_size", 256),
            num_samples=self.config.get("num_samples", 50),
            weight_decay=self.config.get("weight_decay", 1e-4),
            random_seed=self.config.get("random_seed", 21),
        )

        model.feature_names = feature_names
        model.fit(X_train, y_train, X_val, y_val, mask=label_mask, verbose=verbose)

        self.training_info["mc_dropout"] = {
            "n_train": len(y_train),
            "n_val": len(y_val) if y_val is not None else 0,
            "n_features": X_train.shape[1],
            "rationales": rationales,
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
            print(f"\n{'=' * 80}")
            print(f"Training BNN for: {rationales}")
            print(f"{'=' * 80}")

        train_clean = train_df.dropna(subset=rationales, how="all").copy()
        X_train, y_train, feature_names = data_manager.prepare_for_training(
            train_clean,
            rationales,
            fit=True,
            drop_high_missing=self.config.get("drop_high_missing", 1.0),
            use_all_features=self.config.get("use_all_features", False),
            exclude_cols=self.config.get("exclude_cols", None),
            missing_strategy=self.config.get("missing_strategy", "zero"),
            verbose=verbose,
        )

        X_val, y_val = None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=rationales, how="all").copy()
            X_val, y_val, _ = data_manager.prepare_for_training(
                val_clean,
                rationales,
                fit=False,
                missing_strategy=self.config.get("missing_strategy", "zero"),
            )

        model = BNNModel(
            rationales=rationales,
            hidden_dims=self.config.get("hidden_dims", [64, 32]),
            prior_scale=self.config.get("prior_scale", 1.0),
            learning_rate=self.config.get("learning_rate", 0.01),
            num_epochs=self.config.get("num_epochs", 100),
            batch_size=self.config.get("batch_size", 256),
            num_samples=self.config.get("num_samples", 100),
            random_seed=self.config.get("random_seed", 21),
        )

        model.feature_names = feature_names
        model.fit(X_train, y_train, X_val, y_val, verbose=verbose)

        self.training_info["bnn"] = {
            "n_train": len(y_train),
            "n_val": len(y_val) if y_val is not None else 0,
            "n_features": X_train.shape[1],
            "rationales": rationales,
        }

        return model

    def _print_training_summary(self):
        """Print training summary."""
        if not self.training_info:
            return

        print(f"\n{'=' * 80}")
        print("TRAINING SUMMARY")
        print(f"{'=' * 80}")

        if self.model_type in ["mc_dropout", "bnn"]:
            info = self.training_info[self.model_type]
            print(f"Model: {self.model_type.upper()} (multi-label)")
            print(f"Rationales: {', '.join(info['rationales'])}")
            print(f"Train samples: {info['n_train']:,}")
            print(f"Val samples: {info['n_val']:,}")
            print(f"Features: {info['n_features']}")
        else:
            summary_df = pd.DataFrame(self.training_info).T
            print(summary_df.to_string())

        print(f"{'=' * 80}\n")

    def train_hierarchical(
        self,
        train_df: pd.DataFrame,
        rationales: List[str],
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """
        Train hierarchical Bayesian model for multiple rationales.

        Args:
            train_df: Training dataframe
            rationales: List of target rationales
            data_manager: DataManager instance
            val_df: Optional validation dataframe
            verbose: Print training progress

        Returns:
            Trained HierarchicalModel
        """
        if not HIERARCHICAL_AVAILABLE:
            raise ImportError(
                "Hierarchical model not available. Check src/models/bayesian_hierarchial.py"
            )

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Training HIERARCHICAL BAYESIAN MODEL for: {rationales}")
            print(f"{'=' * 80}")

        # Prepare training data with hierarchical indices
        train_clean = train_df.dropna(subset=rationales, how="all").copy()

        X_train, y_train, inv_train, firm_train, year_train, feature_names = (
            data_manager.prepare_for_hierarchical(
                train_clean,
                rationales,
                fit=True,
                drop_high_missing=self.config.get("drop_high_missing", 1.0),
                use_all_features=self.config.get("use_all_features", False),
                exclude_cols=self.config.get("exclude_cols", None),
                missing_strategy=self.config.get("missing_strategy", "zero"),
                verbose=verbose,
            )
        )

        # Prepare validation data if available
        X_val, y_val, inv_val, firm_val, year_val = None, None, None, None, None
        if val_df is not None:
            val_clean = val_df.dropna(subset=rationales, how="all").copy()
            if len(val_clean) > 0:
                X_val, y_val, inv_val, firm_val, year_val, _ = (
                    data_manager.prepare_for_hierarchical(
                        val_clean,
                        rationales,
                        fit=False,  # Use fitted encoders from training
                        missing_strategy=self.config.get("missing_strategy", "zero"),
                        verbose=False,
                    )
                )

        if verbose:
            print(f"\nData shapes:")
            print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"  Unique investors: {np.unique(inv_train).size}")
            print(f"  Unique firms: {np.unique(firm_train).size}")
            print(f"  Unique years: {np.unique(year_train).size}")
            if X_val is not None:
                print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")

            # Print label statistics
            print(f"\nLabel statistics (training):")
            for i, rat in enumerate(rationales):
                pos_rate = y_train[:, i].mean()
                print(
                    f"  {rat}: {pos_rate:.1%} ({int(y_train[:, i].sum())}/{len(y_train)})"
                )

        # Initialize model
        model = HierarchicalModel(
            rationales=rationales,
            prior_scale=self.config.get("prior_scale", 0.1),
            learning_rate=self.config.get("learning_rate", 0.001),
            num_epochs=self.config.get("num_epochs", 100),
            batch_size=self.config.get("batch_size", 512),
            num_samples=self.config.get("num_samples", 100),
            patience=self.config.get("patience", 10),
            min_delta=self.config.get("min_delta", 1.0),
            grad_clip=self.config.get("grad_clip", 1.0),
            random_seed=self.config.get("random_seed", 21),
        )

        # Store feature names
        model.feature_names = feature_names

        # Train model
        model.fit(
            X=X_train,
            y=y_train,
            investor_idx=inv_train,
            firm_idx=firm_train,
            year_idx=year_train,
            X_val=X_val,
            y_val=y_val,
            investor_val=inv_val,
            firm_val=firm_val,
            year_val=year_val,
            verbose=verbose,
        )

        # Store training info
        self.training_info["hierarchical"] = {
            "n_train": len(y_train),
            "n_val": len(y_val) if y_val is not None else 0,
            "n_features": X_train.shape[1],
            "n_investors": np.unique(inv_train).size,
            "n_firms": np.unique(firm_train).size,
            "n_years": np.unique(year_train).size,
            "rationales": rationales,
            "training_losses": model.training_losses
            if hasattr(model, "training_losses")
            else [],
            "validation_losses": model.validation_losses
            if hasattr(model, "validation_losses")
            else [],
        }

        if verbose:
            print(f"\n{'=' * 80}")
            print("TRAINING COMPLETE")
            print(f"{'=' * 80}")
            print(
                f"Final training loss: {model.training_losses[-1]:.4f}"
                if model.training_losses
                else "N/A"
            )
            if model.validation_losses:
                print(f"Final validation loss: {model.validation_losses[-1]:.4f}")

        return model

    def train_all(
        self,
        train_df: pd.DataFrame,
        rationales: List[str],
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {self.model_type.upper()}")
        print(f"{'=' * 80}\n")

        # Check if model type is hierarchical
        if self.model_type == "hierarchical" or self.model_type == "bhm_improved":
            model = self.train_hierarchical(
                train_df=train_df,
                rationales=rationales,
                data_manager=data_manager,
                val_df=val_df,
                verbose=True,
            )
            self.models = {"hierarchical": model}

        # Check if model type is BNN
        elif self.model_type == "bnn":
            model = self.train_bnn(
                train_df=train_df,
                rationales=rationales,
                data_manager=data_manager,
                val_df=val_df,
                verbose=True,
            )
            self.models = {"bnn": model}

        # For other model types, train one model per rationale
        else:
            for rationale in rationales:
                if self.model_type == "mc_dropout":
                    model = self.train_mc_dropout(
                        train_df, rationale, data_manager, verbose=True
                    )
                elif self.model_type == "pca":
                    model = self.train_pca(
                        train_df, rationale, data_manager, val_df, verbose=True
                    )
                elif self.model_type in ["catboost", "lightgbm", "xgboost"]:
                    model = self.train_calibrated_boosting(
                        train_df, rationale, data_manager, val_df, verbose=True
                    )
                elif self.model_type in ["sparse_gp", "deep_kernel_gp"]:
                    model = self.train_gaussian_process(
                        train_df, rationale, data_manager, val_df, verbose=True
                    )
                else:
                    # Default to supervised model
                    model = self.train_supervised(
                        train_df, rationale, data_manager, verbose=True
                    )

                if model is not None:
                    self.models[rationale] = model

        # Save models if save_dir is provided
        if save_dir:
            self.save_models(save_dir)

        return {
            "models": self.models,
            "training_info": self.training_info,
        }

    def save_models(self, save_dir: Path):
        """Save all trained models and training info."""
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model.save(str(save_dir / f"{name}_model.pkl"))

        with open(save_dir / "data_manager.pkl", "wb") as f:
            pickle.dump(self.data_manager, f)

        with open(save_dir / "training_info.json", "w") as f:
            json.dump(self.training_info, f, indent=2)
