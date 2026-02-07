"""
Semi-Supervised Learning for Voting Rationale Prediction.

Implements pseudo-labeling and self-training to leverage unlabeled dissent observations.
This is particularly useful since we have many dissent rows with missing rationales
that can help improve model performance.

Key approaches:
1. Pseudo-labeling: Train on labeled data, predict on unlabeled, add high-confidence
   predictions to training set
2. Co-training: Train multiple models on different feature subsets, use their
   agreement to label unlabeled data
3. Self-training with uncertainty: Only add pseudo-labels when model is confident
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseRationaleModel, SupervisedRationaleModel


class PseudoLabelingSemiSupervised(BaseRationaleModel):
    """
    Semi-supervised learning using pseudo-labeling with iterative refinement.

    Process:
    1. Train initial model on labeled data
    2. Predict on unlabeled data
    3. Add high-confidence predictions to training set
    4. Retrain and repeat
    """

    def __init__(
        self,
        rationale: str,
        base_model_type: str = "logistic",
        confidence_threshold: float = 0.9,
        max_iterations: int = 5,
        min_pseudo_labels: int = 10,
        calibrate: bool = True,
        custom_params: Optional[Dict] = None,
        random_seed: int = 21,
    ):
        """
        Initialize semi-supervised model.

        Args:
            rationale: Target rationale to predict
            base_model_type: Base model type (logistic, random_forest, etc.)
            confidence_threshold: Minimum probability to accept pseudo-label (0.0-1.0)
            max_iterations: Maximum self-training iterations
            min_pseudo_labels: Minimum pseudo-labels to add per iteration
            calibrate: Apply probability calibration
            custom_params: Base model hyperparameters
            random_seed: Random seed
        """
        super().__init__([rationale], f"semi_supervised_{base_model_type}", random_seed)

        self.rationale = rationale
        self.base_model_type = base_model_type
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_pseudo_labels = min_pseudo_labels
        self.calibrate = calibrate
        self.custom_params = custom_params or {}

        self.model = None
        self.training_history = []

    def _create_base_model(self) -> SupervisedRationaleModel:
        """Create base supervised model."""
        return SupervisedRationaleModel(
            rationale=self.rationale,
            base_model_type=self.base_model_type,
            calibrate=self.calibrate,
            custom_params=self.custom_params,
            random_seed=self.random_seed,
        )

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Train semi-supervised model with pseudo-labeling.

        Args:
            X_labeled: Labeled features
            y_labeled: Labeled targets
            X_unlabeled: Unlabeled features (dissent rows with missing rationale)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Semi-Supervised Training: {self.rationale}")
            print(f"{'=' * 80}")
            print(f"Labeled samples: {len(y_labeled)} ({y_labeled.sum()} positive)")
            print(f"Unlabeled samples: {len(X_unlabeled)}")
            print(f"Confidence threshold: {self.confidence_threshold}")
            print(f"Max iterations: {self.max_iterations}")

        # Start with labeled data only
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_pool = X_unlabeled.copy()  # Pool of unlabeled data

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
                print(
                    f"Training set: {len(y_train)} samples ({y_train.sum()} positive)"
                )
                print(f"Unlabeled pool: {len(X_pool)} samples")

            # Train model on current labeled set
            self.model = self._create_base_model()
            self.model.feature_names = self.feature_names
            self.model.fit(X_train, y_train, verbose=False)

            # If no unlabeled data left, stop
            if len(X_pool) == 0:
                if verbose:
                    print("No more unlabeled data. Stopping.")
                break

            # Predict on unlabeled pool
            y_pred_proba = self.model.predict_proba(X_pool)

            # Find high-confidence predictions
            high_conf_positive = y_pred_proba >= self.confidence_threshold
            high_conf_negative = y_pred_proba <= (1 - self.confidence_threshold)
            high_conf_mask = high_conf_positive | high_conf_negative

            n_pseudo = high_conf_mask.sum()

            if verbose:
                n_pos = high_conf_positive.sum()
                n_neg = high_conf_negative.sum()
                print(
                    f"High-confidence pseudo-labels: {n_pseudo} "
                    f"({n_pos} positive, {n_neg} negative)"
                )

            # Stop if too few pseudo-labels
            if n_pseudo < self.min_pseudo_labels:
                if verbose:
                    print(
                        f"Only {n_pseudo} pseudo-labels (< {self.min_pseudo_labels}). Stopping."
                    )
                break

            # Extract pseudo-labeled data
            X_pseudo = X_pool[high_conf_mask]
            y_pseudo = (y_pred_proba[high_conf_mask] >= 0.5).astype(int)

            # Add to training set
            X_train = np.vstack([X_train, X_pseudo])
            y_train = np.concatenate([y_train, y_pseudo])

            # Remove from unlabeled pool
            X_pool = X_pool[~high_conf_mask]

            # Validation performance
            if X_val is not None and y_val is not None:
                val_acc = (self.model.predict(X_val) == y_val).mean()
                val_auc = self._compute_auc(y_val, self.model.predict_proba(X_val))
                if verbose:
                    print(f"Validation: Acc={val_acc:.4f}, AUC={val_auc:.4f}")

            # Store iteration info
            self.training_history.append(
                {
                    "iteration": iteration + 1,
                    "n_labeled": len(y_labeled),
                    "n_pseudo": len(y_train) - len(y_labeled),
                    "n_unlabeled_remaining": len(X_pool),
                    "n_positive": y_train.sum(),
                }
            )

        # Final model on all accumulated data
        if verbose:
            print(f"\n{'=' * 80}")
            print("Final Training")
            print(f"{'=' * 80}")
            print(
                f"Total training samples: {len(y_train)} "
                f"({len(y_labeled)} labeled + {len(y_train) - len(y_labeled)} pseudo)"
            )

        self.model = self._create_base_model()
        self.model.feature_names = self.feature_names
        self.model.fit(X_train, y_train, verbose=verbose)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted or self.model is None:
            return np.zeros(X.shape[0])
        return self.model.predict_proba(X)

    def _compute_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute ROC AUC score."""
        try:
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(y_true, y_pred)
        except:
            return np.nan

    def get_training_history(self) -> pd.DataFrame:
        """Get training history as DataFrame."""
        return pd.DataFrame(self.training_history)


class CoTrainingSemiSupervised(BaseRationaleModel):
    """
    Co-training: Train two models on different feature subsets.
    Use their agreement on unlabeled data to expand training set.
    """

    def __init__(
        self,
        rationale: str,
        base_model_type: str = "logistic",
        agreement_threshold: float = 0.1,  # Max prob difference for agreement
        confidence_threshold: float = 0.8,
        max_iterations: int = 5,
        min_pseudo_labels: int = 10,
        feature_split_ratio: float = 0.5,
        calibrate: bool = True,
        custom_params: Optional[Dict] = None,
        random_seed: int = 21,
    ):
        """
        Initialize co-training model.

        Args:
            rationale: Target rationale
            base_model_type: Base model type
            agreement_threshold: Max probability difference for two models to agree
            confidence_threshold: Min confidence for pseudo-label
            max_iterations: Max co-training iterations
            min_pseudo_labels: Min pseudo-labels per iteration
            feature_split_ratio: Ratio of features for first model (rest go to second)
            calibrate: Apply calibration
            custom_params: Model hyperparameters
            random_seed: Random seed
        """
        super().__init__([rationale], f"cotrain_{base_model_type}", random_seed)

        self.rationale = rationale
        self.base_model_type = base_model_type
        self.agreement_threshold = agreement_threshold
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_pseudo_labels = min_pseudo_labels
        self.feature_split_ratio = feature_split_ratio
        self.calibrate = calibrate
        self.custom_params = custom_params or {}

        self.model1 = None
        self.model2 = None
        self.feature_indices1 = None
        self.feature_indices2 = None
        self.training_history = []

    def _split_features(self, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """Split features into two subsets."""
        np.random.seed(self.random_seed)
        all_indices = np.arange(n_features)
        np.random.shuffle(all_indices)

        split_point = int(n_features * self.feature_split_ratio)
        indices1 = all_indices[:split_point]
        indices2 = all_indices[split_point:]

        return indices1, indices2

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Train co-training model.

        Args:
            X_labeled: Labeled features
            y_labeled: Labeled targets
            X_unlabeled: Unlabeled features
            X_val: Validation features
            y_val: Validation labels
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Co-Training: {self.rationale}")
            print(f"{'=' * 80}")
            print(f"Labeled samples: {len(y_labeled)}")
            print(f"Unlabeled samples: {len(X_unlabeled)}")

        # Split features
        n_features = X_labeled.shape[1]
        self.feature_indices1, self.feature_indices2 = self._split_features(n_features)

        if verbose:
            print(f"Model 1: {len(self.feature_indices1)} features")
            print(f"Model 2: {len(self.feature_indices2)} features")

        # Initialize training sets
        X_train1 = X_labeled[:, self.feature_indices1].copy()
        X_train2 = X_labeled[:, self.feature_indices2].copy()
        y_train1 = y_labeled.copy()
        y_train2 = y_labeled.copy()
        X_pool = X_unlabeled.copy()

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Train both models
            self.model1 = SupervisedRationaleModel(
                self.rationale,
                self.base_model_type,
                self.calibrate,
                self.custom_params,
                self.random_seed,
            )
            self.model2 = SupervisedRationaleModel(
                self.rationale,
                self.base_model_type,
                self.calibrate,
                self.custom_params,
                self.random_seed,
            )

            self.model1.fit(X_train1, y_train1, verbose=False)
            self.model2.fit(X_train2, y_train2, verbose=False)

            if len(X_pool) == 0:
                break

            # Predict on unlabeled pool with both models
            X_pool1 = X_pool[:, self.feature_indices1]
            X_pool2 = X_pool[:, self.feature_indices2]

            y_pred1 = self.model1.predict_proba(X_pool1)
            y_pred2 = self.model2.predict_proba(X_pool2)

            # Find where models agree and are confident
            prob_diff = np.abs(y_pred1 - y_pred2)
            agree_mask = prob_diff <= self.agreement_threshold

            avg_prob = (y_pred1 + y_pred2) / 2
            confident_mask = (avg_prob >= self.confidence_threshold) | (
                avg_prob <= (1 - self.confidence_threshold)
            )

            pseudo_mask = agree_mask & confident_mask
            n_pseudo = pseudo_mask.sum()

            if verbose:
                print(f"Agreements: {agree_mask.sum()}")
                print(f"High-confidence pseudo-labels: {n_pseudo}")

            if n_pseudo < self.min_pseudo_labels:
                if verbose:
                    print(f"Too few pseudo-labels ({n_pseudo}). Stopping.")
                break

            # Add pseudo-labeled data
            X_pseudo = X_pool[pseudo_mask]
            y_pseudo = (avg_prob[pseudo_mask] >= 0.5).astype(int)

            X_train1 = np.vstack([X_train1, X_pseudo[:, self.feature_indices1]])
            X_train2 = np.vstack([X_train2, X_pseudo[:, self.feature_indices2]])
            y_train1 = np.concatenate([y_train1, y_pseudo])
            y_train2 = np.concatenate([y_train2, y_pseudo])

            # Remove from pool
            X_pool = X_pool[~pseudo_mask]

            self.training_history.append(
                {
                    "iteration": iteration + 1,
                    "n_labeled": len(y_labeled),
                    "n_pseudo": len(y_train1) - len(y_labeled),
                    "n_unlabeled_remaining": len(X_pool),
                }
            )

        # Final training on all data
        if verbose:
            print(f"\nFinal models with {len(y_train1)} samples")

        self.model1 = SupervisedRationaleModel(
            self.rationale,
            self.base_model_type,
            self.calibrate,
            self.custom_params,
            self.random_seed,
        )
        self.model2 = SupervisedRationaleModel(
            self.rationale,
            self.base_model_type,
            self.calibrate,
            self.custom_params,
            self.random_seed,
        )

        self.model1.fit(X_train1, y_train1, verbose=False)
        self.model2.fit(X_train2, y_train2, verbose=False)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict using average of both models."""
        if not self.is_fitted:
            return np.zeros(X.shape[0])

        X1 = X[:, self.feature_indices1]
        X2 = X[:, self.feature_indices2]

        y_pred1 = self.model1.predict_proba(X1)
        y_pred2 = self.model2.predict_proba(X2)

        return (y_pred1 + y_pred2) / 2


class MultiLabelSemiSupervised:
    """
    Multi-label wrapper for semi-supervised learning.
    Trains one semi-supervised model per rationale.
    """

    def __init__(
        self,
        rationales: List[str],
        method: str = "pseudo_labeling",  # or "co_training"
        base_model_type: str = "logistic",
        confidence_threshold: float = 0.9,
        max_iterations: int = 5,
        **kwargs,
    ):
        """
        Initialize multi-label semi-supervised model.

        Args:
            rationales: List of target rationales
            method: 'pseudo_labeling' or 'co_training'
            base_model_type: Base model type
            confidence_threshold: Confidence threshold for pseudo-labels
            max_iterations: Max iterations
            **kwargs: Additional arguments for base models
        """
        self.rationales = rationales
        self.method = method
        self.base_model_type = base_model_type
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.kwargs = kwargs

        self.models = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Train semi-supervised models for all rationales."""
        for i, rationale in enumerate(self.rationales):
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Training {rationale} ({i + 1}/{len(self.rationales)})")
                print(f"{'=' * 80}")

            y_single = y_labeled[:, i]
            y_val_single = y_val[:, i] if y_val is not None else None

            if self.method == "pseudo_labeling":
                model = PseudoLabelingSemiSupervised(
                    rationale=rationale,
                    base_model_type=self.base_model_type,
                    confidence_threshold=self.confidence_threshold,
                    max_iterations=self.max_iterations,
                    **self.kwargs,
                )
            elif self.method == "co_training":
                model = CoTrainingSemiSupervised(
                    rationale=rationale,
                    base_model_type=self.base_model_type,
                    confidence_threshold=self.confidence_threshold,
                    max_iterations=self.max_iterations,
                    **self.kwargs,
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")

            model.fit(
                X_labeled, y_single, X_unlabeled, X_val, y_val_single, verbose=verbose
            )

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
