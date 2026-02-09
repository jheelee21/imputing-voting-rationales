from src.models.base_model import BaseRationaleModel
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


class SupervisedRationaleModel(BaseRationaleModel):
    def __init__(
        self,
        rationale: str,
        random_seed: int,
        base_model_type: str = "logistic",
        calibrate: bool = True,
        custom_params: Optional[Dict] = None,
    ):
        super().__init__([rationale], f"supervised_{base_model_type}", random_seed)
        self.rationale = rationale
        self.base_model_type = base_model_type
        self.calibrate = calibrate
        self.custom_params = custom_params or {}
        self.model = None
        self.feature_importance = None

    def _get_base_model(self):
        """Initialize base sklearn model."""

        default_params = {"random_state": self.random_seed}

        if self.base_model_type == "logistic":
            default_params.update(
                {
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "C": 1.0,
                }
            )
            default_params.update(self.custom_params)
            return LogisticRegression(**default_params)

        elif self.base_model_type == "random_forest":
            default_params.update(
                {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 20,
                    "min_samples_leaf": 10,
                    "class_weight": "balanced",
                }
            )
            default_params.update(self.custom_params)
            return RandomForestClassifier(**default_params)

        elif self.base_model_type == "gradient_boosting":
            default_params.update(
                {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "min_samples_split": 20,
                    "min_samples_leaf": 10,
                }
            )
            default_params.update(self.custom_params)
            return GradientBoostingClassifier(**default_params)

        else:
            raise ValueError(f"Unsupported model type: {self.base_model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Train supervised model."""
        n_pos = y.sum()

        if n_pos == 0:
            if verbose:
                print(f"No positive samples for '{self.rationale}'. Skipping training.")
            return self

        if verbose:
            print(
                f"Training {self.base_model_type} for '{self.rationale}': "
                f"{len(y)} samples, {n_pos} positive ({n_pos / len(y):.2%})"
            )

        # Get base model
        base_model = self._get_base_model()

        # Apply calibration if requested
        if self.calibrate and n_pos >= 3:
            from sklearn.calibration import CalibratedClassifierCV

            self.model = CalibratedClassifierCV(
                base_model, cv=min(3, n_pos), method="sigmoid"
            )
        else:
            self.model = base_model

        # Train
        self.model.fit(X, y)
        self.is_fitted = True

        # Extract feature importance
        self._compute_feature_importance()

        if verbose:
            train_acc = (self.predict(X) == y).mean()
            print(f"Training accuracy: {train_acc:.4f}")

        return self

    def _compute_feature_importance(self):
        """Extract feature importance from trained model."""
        if not self.is_fitted:
            return

        if isinstance(self.model, CalibratedClassifierCV):
            base_model = self.model.calibrated_classifiers_[0].estimator
        else:
            base_model = self.model

        # Extract importance
        if hasattr(base_model, "feature_importances_"):
            self.feature_importance = base_model.feature_importances_
        elif hasattr(base_model, "coef_"):
            self.feature_importance = np.abs(base_model.coef_[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted or self.model is None:
            return np.zeros(X.shape[0])
        return self.model.predict_proba(X)[:, 1]

    def get_top_features(self, top_k: int = 10) -> pd.DataFrame:
        """Get top important features."""
        if self.feature_importance is None or self.feature_names is None:
            return pd.DataFrame()

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.feature_importance,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_k)
        )
