from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path


class BaseRationaleModel(ABC):
    def __init__(
        self, rationales: List[str], model_type: str, random_seed: int = 21, **kwargs
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
        **kwargs,
    ):
        """Train the model."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict probabilities for each rationale."""
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.5, **kwargs) -> np.ndarray:
        """Predict binary labels."""
        probs = self.predict_proba(X, **kwargs)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "rationales": self.rationales,
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
        }
