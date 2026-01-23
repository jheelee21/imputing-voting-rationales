import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Tuple, Optional
import pickle
from utils.types import CORE_RATIONALES, ALL_RATIONALES


class SupervisedModel:  # to be used per rationale
    def __init__(
        self,
        rationale: str,
        calibrate: bool = True,
        base_model: str = "logistic",
        custom_params: Optional[Dict] = None,
        random_seed: int = 21,
    ):
        self.rationale = rationale
        self.base_model_type = base_model
        self.calibrate = calibrate
        self.custom_params = custom_params or {}
        self.random_state = random_seed
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def _get_base_model(self):
        default_params = {"random_state": self.random_state}

        match self.base_model_type:
            case "logistic":
                default_params.update(
                    {
                        "max_iter": 1000,
                        "class_weight": "balanced",
                        "penalty": "l2",
                        "C": 1.0,
                    }
                )
                default_params.update(self.custom_params)
                return LogisticRegression(**default_params)
            case "random_forest":
                default_params.update(
                    {
                        "n_estimators": 100,
                        "class_weight": "balanced",
                        "max_depth": 10,
                        "min_samples_split": 20,
                        "min_samples_leaf": 10,
                        "max_features": "sqrt",
                    }
                )
                default_params.update(self.custom_params)
                return RandomForestClassifier(**default_params)
            case "gradient_boosting":
                default_params.update(
                    {
                        "n_estimators": 100,
                        "max_depth": 5,
                        "learning_rate": 0.1,
                        "min_samples_split": 20,
                        "min_samples_leaf": 10,
                        "subsample": 0.8,
                    }
                )
                default_params.update(self.custom_params)
                return GradientBoostingClassifier(**default_params)
            case _:
                raise ValueError(f"Unsupported base model type: {self.base_model_type}")

    def _compute_feature_importance(self, X: np.ndarray):
        if self.model is None:
            self.feature_importance = None
            return

        if isinstance(self.model, CalibratedClassifierCV):
            base_model = self.model.calibrated_classifiers_[0].base_estimator
        else:
            base_model = self.model

        if hasattr(base_model, "feature_importances_"):
            self.feature_importance = base_model.feature_importances_
        elif hasattr(base_model, "coef_"):
            self.feature_importance = np.abs(base_model.coef_[0])
        else:
            self.feature_importance = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        pos_rate = np.mean(y) if len(y) > 0 else 0
        n_pos = y.sum()
        n_total = len(y)

        print(
            f"Training model for rationale '{self.rationale}' with {n_total} samples, positive ratio: {pos_rate:.4f}"
        )

        if n_pos == 0:
            print(
                f"No positive samples for rationale '{self.rationale}'. Skipping training."
            )
            self.model = None
            return self

        base_model = self._get_base_model()

        if self.calibrate:
            print(f"Calibrating model ...")
            self.model = CalibratedClassifierCV(
                base_model, cv=min(3, n_pos), method="sigmoid"
            )
        else:
            self.model = base_model

        self.model.fit(X, y)

        self._compute_feature_importance(X)

        print(f"Model training completed for rationale '{self.rationale}'.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(X.shape[0])
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_top_features(self, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.feature_importance is None:
            return pd.DataFrame()

        df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names
                    if self.feature_names
                    else range(len(self.feature_importance)),
                    "importance": self.feature_importance,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_k)
        )

        return df

    def save_model(self):
        filepath = f"../models/supervised_{self.rationale}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model for rationale '{self.rationale}' saved to {filepath}")
