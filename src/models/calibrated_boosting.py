"""
Calibrated Boosting Models for voting rationale prediction.
Includes CatBoost, LightGBM, and XGBoost with proper calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import SupervisedRationaleModel

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print(
        "Warning: CatBoost not installed. Install with: pip install catboost --break-system-packages"
    )

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print(
        "Warning: LightGBM not installed. Install with: pip install lightgbm --break-system-packages"
    )

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print(
        "Warning: XGBoost not installed. Install with: pip install xgboost --break-system-packages"
    )


class CalibratedBoostingModel(SupervisedRationaleModel):
    """
    Enhanced supervised model with calibrated boosting algorithms.
    Supports CatBoost, LightGBM, and XGBoost with better probability calibration.
    """

    def __init__(
        self,
        rationale: str,
        base_model_type: str = "catboost",
        calibrate: bool = True,
        calibration_method: str = "isotonic",  # 'isotonic' or 'sigmoid'
        custom_params: Optional[Dict] = None,
        random_seed: int = 21,
    ):
        """
        Initialize calibrated boosting model.

        Args:
            rationale: Target rationale to predict
            base_model_type: One of ['catboost', 'lightgbm', 'xgboost']
            calibrate: Whether to apply post-hoc calibration
            calibration_method: 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling)
            custom_params: Model-specific hyperparameters
            random_seed: Random seed for reproducibility
        """
        # Check availability
        if base_model_type == "catboost" and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        if base_model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        if base_model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")

        super().__init__(
            rationale=rationale,
            base_model_type=base_model_type,
            calibrate=calibrate,
            custom_params=custom_params,
            random_seed=random_seed,
        )
        self.calibration_method = calibration_method
        self._calibrator = None  # Store calibration function for manual calibration

    def _get_base_model(self):
        """Initialize calibrated boosting model."""
        default_params = {"random_state": self.random_seed}

        if self.base_model_type == "catboost":
            default_params.update(
                {
                    # "iterations": 500,
                    # "depth": 6,
                    "learning_rate": 0.03,
                    "auto_class_weights": "Balanced",
                    "verbose": False,
                    "early_stopping_rounds": 50,
                    "task_type": "CPU",
                    # Calibration-friendly settings
                    "bootstrap_type": "Bayesian",
                    "bagging_temperature": 1.0,
                    # "od_type": "Iter",
                    # "od_wait": 20,
                }
            )
            default_params.update(self.custom_params)
            return CatBoostClassifier(**default_params)

        elif self.base_model_type == "lightgbm":
            default_params.update(
                {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "class_weight": "balanced",
                    "verbose": -1,
                    "min_child_samples": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    # Better calibration
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                }
            )
            default_params.update(self.custom_params)
            return LGBMClassifier(**default_params)

        elif self.base_model_type == "xgboost":
            default_params.update(
                {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.03,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 5,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "scale_pos_weight": None,  # Will be set in fit()
                    "eval_metric": "logloss",
                    "early_stopping_rounds": 50,
                    "verbose": False,
                }
            )
            default_params.update(self.custom_params)
            return XGBClassifier(**default_params)

        else:
            raise ValueError(f"Unsupported boosting model: {self.base_model_type}")

    def _apply_manual_calibration(self, base_model, X_cal, y_cal):
        """Apply manual calibration as workaround for sklearn compatibility."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        # Get predictions on calibration set
        y_pred_proba = base_model.predict_proba(X_cal)[:, 1]

        # Fit calibrator
        if self.calibration_method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(y_pred_proba, y_cal)
        else:  # sigmoid (Platt scaling)
            calibrator = LogisticRegression()
            calibrator.fit(y_pred_proba.reshape(-1, 1), y_cal)

        # Store both base model and calibrator
        self.model = base_model
        self._calibrator = calibrator

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Train calibrated boosting model."""
        n_pos = y.sum()
        n_neg = len(y) - n_pos

        if n_pos == 0:
            if verbose:
                print(f"No positive samples for '{self.rationale}'. Skipping training.")
            return self

        if verbose:
            print(
                f"Training {self.base_model_type.upper()} for '{self.rationale}': "
                f"{len(y)} samples, {n_pos} positive ({n_pos / len(y):.2%})"
            )

        # Get base model
        base_model = self._get_base_model()

        # For XGBoost, set scale_pos_weight
        if self.base_model_type == "xgboost" and n_pos > 0:
            base_model.scale_pos_weight = n_neg / n_pos

        # Fit with validation set if available (for early stopping)
        if X_val is not None and y_val is not None:
            if self.base_model_type == "catboost":
                base_model.fit(
                    X,
                    y,
                    eval_set=(X_val, y_val),
                    verbose=False,
                )
            elif self.base_model_type == "lightgbm":
                base_model.fit(
                    X,
                    y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[],
                )
            elif self.base_model_type == "xgboost":
                base_model.fit(
                    X,
                    y,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
        else:
            base_model.fit(X, y)

        # Apply post-hoc calibration if requested
        if self.calibrate and n_pos >= 3:
            if verbose:
                print(f"Applying {self.calibration_method} calibration...")

            # Use manual calibration for better compatibility
            if X_val is not None and y_val is not None:
                # Use validation set for calibration
                self._apply_manual_calibration(base_model, X_val, y_val)
            else:
                # Use CV calibration
                from sklearn.calibration import CalibratedClassifierCV

                # Need to refit from scratch for CV
                new_base_model = self._get_base_model()

                try:
                    self.model = CalibratedClassifierCV(
                        estimator=new_base_model,
                        method=self.calibration_method,
                        cv=min(3, n_pos),
                    )
                except TypeError:
                    # Older sklearn versions
                    self.model = CalibratedClassifierCV(
                        base_estimator=new_base_model,
                        method=self.calibration_method,
                        cv=min(3, n_pos),
                    )

                self.model.fit(X, y)
        else:
            self.model = base_model

        self.is_fitted = True

        # Extract feature importance
        self._compute_feature_importance()

        if verbose:
            train_acc = (self.predict(X) == y).mean()
            print(f"Training accuracy: {train_acc:.4f}")

            if X_val is not None and y_val is not None:
                val_acc = (self.predict(X_val) == y_val).mean()
                print(f"Validation accuracy: {val_acc:.4f}")

        return self

    def _compute_feature_importance(self):
        """Extract feature importance from trained boosting model."""
        if not self.is_fitted:
            return

        from sklearn.calibration import CalibratedClassifierCV

        # Get base model
        if isinstance(self.model, CalibratedClassifierCV):
            # Try different attributes for different scikit-learn versions
            if hasattr(self.model, "calibrated_classifiers_"):
                # Newer versions
                try:
                    base_model = self.model.calibrated_classifiers_[0].estimator
                except (AttributeError, IndexError):
                    base_model = self.model.calibrated_classifiers_[0].base_estimator
            elif hasattr(self.model, "base_estimator"):
                # Older versions
                base_model = self.model.base_estimator
            else:
                # Fallback
                base_model = self.model
        else:
            base_model = self.model

        # Extract importance based on model type
        if hasattr(base_model, "feature_importances_"):
            self.feature_importance = base_model.feature_importances_
        elif hasattr(base_model, "get_feature_importance"):
            # CatBoost
            try:
                self.feature_importance = base_model.get_feature_importance()
            except:
                self.feature_importance = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with manual calibration if available."""
        if not self.is_fitted or self.model is None:
            return np.zeros(X.shape[0])

        # Get base predictions
        base_proba = self.model.predict_proba(X)[:, 1]

        # Apply manual calibrator if available
        if self._calibrator is not None:
            if self.calibration_method == "isotonic":
                return self._calibrator.predict(base_proba)
            else:  # sigmoid
                return self._calibrator.predict_proba(base_proba.reshape(-1, 1))[:, 1]

        # Otherwise return base predictions
        return base_proba

    def get_calibration_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve for model evaluation.

        Returns:
            prob_true: True probabilities in each bin
            prob_pred: Mean predicted probabilities in each bin
        """
        from sklearn.calibration import calibration_curve

        y_prob = self.predict_proba(X)
        prob_true, prob_pred = calibration_curve(
            y, y_prob, n_bins=n_bins, strategy="uniform"
        )

        return prob_true, prob_pred


class MultiLabelCalibratedBoosting:
    """
    Multi-label wrapper for calibrated boosting models.
    Trains one model per rationale with shared hyperparameters.
    """

    def __init__(
        self,
        rationales: List[str],
        base_model_type: str = "catboost",
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        custom_params: Optional[Dict] = None,
        random_seed: int = 21,
    ):
        self.rationales = rationales
        self.base_model_type = base_model_type
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.custom_params = custom_params or {}
        self.random_seed = random_seed

        self.models = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Train models for all rationales."""
        for i, rationale in enumerate(self.rationales):
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Training {rationale} ({i + 1}/{len(self.rationales)})")
                print(f"{'=' * 80}")

            model = CalibratedBoostingModel(
                rationale=rationale,
                base_model_type=self.base_model_type,
                calibrate=self.calibrate,
                calibration_method=self.calibration_method,
                custom_params=self.custom_params,
                random_seed=self.random_seed,
            )

            y_single = y[:, i]
            y_val_single = y_val[:, i] if y_val is not None else None

            model.fit(X, y_single, X_val, y_val_single, verbose=verbose)
            self.models[rationale] = model

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for all rationales."""
        probs = []
        for rationale in self.rationales:
            if rationale in self.models:
                probs.append(self.models[rationale].predict_proba(X))
            else:
                probs.append(np.zeros(len(X)))

        return np.column_stack(probs)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels for all rationales."""
        return (self.predict_proba(X) >= threshold).astype(int)
