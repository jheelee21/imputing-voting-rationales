"""
Unified model trainer for all model types.
Handles training, validation, and saving.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import pickle

from src.models.supervised import SupervisedRationaleModel
from src.models.mc_dropout import MCDropoutModel
from src.data.data_manager import DataManager


class ModelTrainer:
    """Unified interface for training all model types."""

    def __init__(
        self,
        model_type: str = "logistic",
        rationales: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        self.model_type = model_type
        self.rationales = rationales
        self.config = config or {}
        self.models = {}
        self.data_manager = None
        self.training_info = {}

    def train_supervised(
        self,
        train_df: pd.DataFrame,
        rationale: str,
        data_manager: DataManager,
        verbose: bool = True,
    ):
        """Train a single supervised model for one rationale."""

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Training {self.model_type.upper()} for: {rationale}")
            print(f"{'=' * 80}")

        # Filter to labeled samples
        train_clean = train_df.dropna(subset=[rationale]).copy()

        if len(train_clean) == 0:
            print(f"No training data for {rationale}")
            return None

        # Prepare data
        X, y, feature_names = data_manager.prepare_for_training(
            train_clean,
            [rationale],
            fit=True,
            drop_high_missing=self.config.get("drop_high_missing", 1.0),
            use_all_features=self.config.get("use_all_features", False),
            exclude_cols=self.config.get("exclude_cols", None),
            missing_strategy=self.config.get("missing_strategy", "mean"),
            verbose=verbose,
        )

        # Extract single label
        y_single = y[:, 0] if y.ndim > 1 else y

        if verbose:
            print(
                f"Samples: {len(y_single)}, Positive: {y_single.sum()} ({y_single.mean():.2%})"
            )

        # Create and train model
        model = SupervisedRationaleModel(
            rationale=rationale,
            base_model_type=self.model_type,
            calibrate=self.config.get("calibrate", True),
            custom_params=self.config.get("custom_params"),
            random_seed=self.config.get("random_seed", 21),
        )

        model.feature_names = feature_names
        model.fit(X, y_single, verbose=verbose)

        # Store training info
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

    def train_mc_dropout(
        self,
        train_df: pd.DataFrame,
        rationales: List[str],
        data_manager: DataManager,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> MCDropoutModel:
        """Train MC Dropout model (multi-label)."""

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Training MC DROPOUT for: {rationales}")
            print(f"{'=' * 80}")

        # Prepare training data (only dissent rows with at least one rationale)
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

        # Mask for partial labels: only compute loss on observed rationales (don't treat missing as negative)
        label_mask = train_clean[rationales].notna().values.astype(bool)

        # Prepare validation data
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
            print(f"Train: {len(y_train)} samples")
            obs_frac = label_mask.mean()
            print(
                f"Label coverage: {obs_frac:.1%} of rationale cells observed (masked loss)"
            )
            if X_val is not None:
                print(f"Val: {len(y_val)} samples")

        # Create and train model
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

        # Store training info
        self.training_info["mc_dropout"] = {
            "n_train": len(y_train),
            "n_val": len(y_val) if y_val is not None else 0,
            "n_features": X_train.shape[1],
            "rationales": rationales,
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
        """
        Train all models based on model_type.

        Returns:
            Dictionary of trained models
        """
        self.data_manager = data_manager
        self.rationales = rationales

        if self.model_type == "mc_dropout":
            # Train single multi-label model
            model = self.train_mc_dropout(train_df, rationales, data_manager, val_df)
            self.models["mc_dropout"] = model

            # Save if requested
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                model.save(str(save_dir / "mc_dropout_model.pkl"))

                with open(save_dir / "data_manager.pkl", "wb") as f:
                    pickle.dump(data_manager, f)

                with open(save_dir / "training_info.json", "w") as f:
                    json.dump(self.training_info, f, indent=2)

        else:
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

            for rationale in rationales:
                model = self.train_supervised(train_df, rationale, data_manager)

                if model is not None:
                    self.models[rationale] = model
                    if save_dir is not None:
                        model.save(str(save_dir / f"{rationale}_model.pkl"))

            # Save data manager and training info
            if save_dir is not None:
                with open(save_dir / "data_manager.pkl", "wb") as f:
                    pickle.dump(data_manager, f)

                with open(save_dir / "training_info.json", "w") as f:
                    json.dump(self.training_info, f, indent=2)

        # Print summary
        self._print_training_summary()

        return self.models

    def _print_training_summary(self):
        """Print training summary."""
        if not self.training_info:
            return

        print(f"\n{'=' * 80}")
        print("TRAINING SUMMARY")
        print(f"{'=' * 80}")

        if self.model_type == "mc_dropout":
            info = self.training_info["mc_dropout"]
            print(f"Model: MC Dropout (multi-label)")
            print(f"Rationales: {', '.join(info['rationales'])}")
            print(f"Train samples: {info['n_train']:,}")
            print(f"Val samples: {info['n_val']:,}")
            print(f"Features: {info['n_features']}")
        else:
            summary_df = pd.DataFrame(self.training_info).T
            print(
                summary_df[
                    ["n_train", "n_positive", "positive_rate", "train_accuracy"]
                ].to_string()
            )

        print(f"{'=' * 80}\n")
