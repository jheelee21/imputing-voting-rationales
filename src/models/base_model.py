"""
Base model interface for all voting rationale prediction models.
Provides common interface and reduces code duplication.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path


class BaseRationaleModel(ABC):
    """Abstract base class for all rationale prediction models."""
    
    def __init__(
        self,
        rationales: List[str],
        model_type: str,
        random_seed: int = 21,
        **kwargs
    ):
        self.rationales = rationales
        self.model_type = model_type
        self.random_seed = random_seed
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict probabilities for each rationale."""
        pass
    
    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
        **kwargs
    ) -> np.ndarray:
        """Predict binary labels."""
        probs = self.predict_proba(X, **kwargs)
        return (probs >= threshold).astype(int)
    
    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'rationales': self.rationales,
            'model_type': self.model_type,
            'random_seed': self.random_seed,
            'is_fitted': self.is_fitted,
        }


class SupervisedRationaleModel(BaseRationaleModel):
    """
    Supervised model wrapper for scikit-learn classifiers.
    Supports single-label (per rationale) training.
    """
    
    def __init__(
        self,
        rationale: str,
        base_model_type: str = "logistic",
        calibrate: bool = True,
        custom_params: Optional[Dict] = None,
        random_seed: int = 21,
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
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        default_params = {"random_state": self.random_seed}
        
        if self.base_model_type == "logistic":
            default_params.update({
                "max_iter": 1000,
                "class_weight": "balanced",
                "C": 1.0,
            })
            default_params.update(self.custom_params)
            return LogisticRegression(**default_params)
        
        elif self.base_model_type == "random_forest":
            default_params.update({
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "class_weight": "balanced",
            })
            default_params.update(self.custom_params)
            return RandomForestClassifier(**default_params)
        
        elif self.base_model_type == "gradient_boosting":
            default_params.update({
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
            })
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
            print(f"Training {self.base_model_type} for '{self.rationale}': "
                  f"{len(y)} samples, {n_pos} positive ({n_pos/len(y):.2%})")
        
        # Get base model
        base_model = self._get_base_model()
        
        # Apply calibration if requested
        if self.calibrate and n_pos >= 3:
            from sklearn.calibration import CalibratedClassifierCV
            self.model = CalibratedClassifierCV(base_model, cv=min(3, n_pos), method="sigmoid")
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
        
        from sklearn.calibration import CalibratedClassifierCV
        
        # Get base model
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
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance,
        }).sort_values('importance', ascending=False).head(top_k)