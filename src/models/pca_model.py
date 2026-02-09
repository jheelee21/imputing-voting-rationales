"""
PCA-based Model for voting rationale prediction.

This model applies Principal Component Analysis (PCA) for dimensionality reduction
before training a classifier. Useful for:
- Reducing multicollinearity among features
- Handling high-dimensional feature spaces
- Extracting principal components that capture most variance

Place this file in: src/models/pca_model.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseRationaleModel


class PCAModel(BaseRationaleModel):
    """
    PCA + Classifier model for single-label classification.

    Workflow:
    1. Apply PCA to reduce feature dimensions
    2. Train classifier on principal components
    3. Optionally apply probability calibration
    """

    def __init__(
        self,
        rationale: str,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        base_model_type: str = "logistic",
        calibrate: bool = True,
        whiten: bool = False,
        custom_params: Optional[Dict] = None,
        random_seed: int = 21,
    ):
        """
        Initialize PCA model.

        Args:
            rationale: Target rationale to predict
            n_components: Number of principal components to keep.
                         If None, keep components explaining variance_threshold
            variance_threshold: Minimum explained variance to retain (0.0-1.0)
                              Only used if n_components is None
            base_model_type: Classifier to use on PC features
                           ['logistic', 'random_forest', 'gradient_boosting']
            calibrate: Whether to apply probability calibration
            whiten: Whether to whiten the data (make components uncorrelated)
            custom_params: Hyperparameters for base classifier
            random_seed: Random seed for reproducibility
        """
        super().__init__([rationale], f"pca_{base_model_type}", random_seed)

        self.rationale = rationale
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.base_model_type = base_model_type
        self.calibrate = calibrate
        self.whiten = whiten
        self.custom_params = custom_params or {}

        # Model components
        self.pca = None
        self.classifier = None
        self.explained_variance_ratio_ = None
        self.n_components_used_ = None
        self.feature_importance_pca = None  # Importance in original feature space

    def _determine_n_components(self, X: np.ndarray) -> int:
        """
        Determine optimal number of components.

        If n_components is specified, use it.
        Otherwise, use variance_threshold to determine components.
        """
        if self.n_components is not None:
            return min(self.n_components, X.shape[1], X.shape[0])

        # Fit PCA with all components to check explained variance
        pca_temp = PCA(random_state=self.random_seed)
        pca_temp.fit(X)

        # Find number of components that explain variance_threshold
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n_comp = np.argmax(cumsum >= self.variance_threshold) + 1

        return max(1, min(n_comp, X.shape[1], X.shape[0]))

    def _get_base_classifier(self):
        """Initialize base classifier."""
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
            raise ValueError(f"Unsupported classifier: {self.base_model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Train PCA + classifier model.

        Args:
            X: Training features (already scaled)
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print progress
        """
        n_pos = y.sum()

        if n_pos == 0:
            if verbose:
                print(f"No positive samples for '{self.rationale}'. Skipping training.")
            return self

        if verbose:
            print(f"\n{'=' * 80}")
            print(
                f"Training PCA + {self.base_model_type.upper()} for '{self.rationale}'"
            )
            print(f"{'=' * 80}")
            print(f"Samples: {len(y)}, Positive: {n_pos} ({n_pos / len(y):.2%})")
            print(f"Original features: {X.shape[1]}")

        # Determine number of components
        n_comp = self._determine_n_components(X)

        if verbose:
            if self.n_components is not None:
                print(f"PCA components: {n_comp} (user-specified)")
            else:
                print(
                    f"PCA components: {n_comp} (explaining {self.variance_threshold:.1%} variance)"
                )

        # Fit PCA
        self.pca = PCA(
            n_components=n_comp, whiten=self.whiten, random_state=self.random_seed
        )
        X_pca = self.pca.fit_transform(X)

        # Store variance info
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.n_components_used_ = n_comp

        if verbose:
            total_var = self.explained_variance_ratio_.sum()
            print(f"Explained variance: {total_var:.1%}")
            print(f"Reduced features: {X.shape[1]} → {n_comp}")

        # Transform validation data if provided
        X_val_pca = None
        if X_val is not None:
            X_val_pca = self.pca.transform(X_val)

        # Get base classifier
        base_classifier = self._get_base_classifier()

        # Apply calibration if requested
        if self.calibrate and n_pos >= 3:
            from sklearn.calibration import CalibratedClassifierCV

            if verbose:
                print("Applying probability calibration...")

            try:
                self.classifier = CalibratedClassifierCV(
                    estimator=base_classifier, cv=min(3, n_pos), method="sigmoid"
                )
            except TypeError:
                # Older sklearn versions
                self.classifier = CalibratedClassifierCV(
                    base_estimator=base_classifier, cv=min(3, n_pos), method="sigmoid"
                )
        else:
            self.classifier = base_classifier

        # Train classifier on PC features
        self.classifier.fit(X_pca, y)
        self.is_fitted = True

        # Compute feature importance in original space
        self._compute_feature_importance()

        # Evaluate
        if verbose:
            train_acc = (self.predict(X) == y).mean()
            print(f"Training accuracy: {train_acc:.4f}")

            if X_val is not None and y_val is not None:
                val_acc = (self.predict(X_val) == y_val).mean()
                print(f"Validation accuracy: {val_acc:.4f}")

        print(f"{'=' * 80}\n")
        return self

    def _compute_feature_importance(self):
        """
        Compute feature importance in original feature space.

        Maps PC importance back to original features using PCA loadings.
        """
        if not self.is_fitted:
            return

        from sklearn.calibration import CalibratedClassifierCV

        # Get base classifier
        if isinstance(self.classifier, CalibratedClassifierCV):
            # Try different attributes for different scikit-learn versions
            if hasattr(self.classifier, "calibrated_classifiers_"):
                try:
                    base_clf = self.classifier.calibrated_classifiers_[0].estimator
                except (AttributeError, IndexError):
                    base_clf = self.classifier.calibrated_classifiers_[0].base_estimator
            elif hasattr(self.classifier, "base_estimator"):
                base_clf = self.classifier.base_estimator
            else:
                base_clf = self.classifier
        else:
            base_clf = self.classifier

        # Get importance in PC space
        if hasattr(base_clf, "feature_importances_"):
            pc_importance = base_clf.feature_importances_
        elif hasattr(base_clf, "coef_"):
            pc_importance = np.abs(base_clf.coef_[0])
        else:
            return

        # Map back to original features using PCA components
        # PCA components shape: (n_components, n_features)
        # Each row is a PC, each column is an original feature
        components = np.abs(self.pca.components_)  # Absolute loadings

        # Weighted sum: importance of each original feature
        # = sum over PCs of (PC_importance * PC_loading_on_feature)
        self.feature_importance_pca = (pc_importance[:, np.newaxis] * components).sum(
            axis=0
        )

        # Normalize to sum to 1
        if self.feature_importance_pca.sum() > 0:
            self.feature_importance_pca /= self.feature_importance_pca.sum()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted or self.pca is None or self.classifier is None:
            return np.zeros(X.shape[0])

        # Transform to PC space
        X_pca = self.pca.transform(X)

        # Predict
        return self.classifier.predict_proba(X_pca)[:, 1]

    def get_explained_variance(self) -> pd.DataFrame:
        """
        Get explained variance by each principal component.

        Returns:
            DataFrame with PC index, variance, and cumulative variance
        """
        if not self.is_fitted or self.pca is None:
            return pd.DataFrame()

        cumsum = np.cumsum(self.explained_variance_ratio_)

        return pd.DataFrame(
            {
                "PC": np.arange(1, len(self.explained_variance_ratio_) + 1),
                "Variance": self.explained_variance_ratio_,
                "Cumulative": cumsum,
            }
        )

    def get_top_features(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get top important features in original feature space.

        Args:
            top_k: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance_pca is None or self.feature_names is None:
            return pd.DataFrame()

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.feature_importance_pca,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_k)
        )

    def get_pc_loadings(self, pc_index: int = 0) -> pd.DataFrame:
        """
        Get feature loadings for a specific principal component.

        Args:
            pc_index: Index of PC (0-based)

        Returns:
            DataFrame with features and their loadings on the PC
        """
        if not self.is_fitted or self.pca is None or self.feature_names is None:
            return pd.DataFrame()

        if pc_index >= self.n_components_used_:
            raise ValueError(
                f"PC index {pc_index} out of range (max: {self.n_components_used_ - 1})"
            )

        loadings = self.pca.components_[pc_index]

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "loading": loadings,
                "abs_loading": np.abs(loadings),
            }
        ).sort_values("abs_loading", ascending=False)

    def plot_explained_variance(self, output_path: Optional[str] = None):
        """
        Plot explained variance by principal components.

        Args:
            output_path: Path to save plot (optional)
        """
        if not self.is_fitted or self.pca is None:
            print("Model not fitted yet.")
            return

        var_df = self.get_explained_variance()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance
        ax1.bar(var_df["PC"], var_df["Variance"], alpha=0.7, color="steelblue")
        ax1.set_xlabel("Principal Component", fontsize=12)
        ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
        ax1.set_title("Variance Explained by Each PC", fontsize=13, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # Cumulative variance
        ax2.plot(
            var_df["PC"], var_df["Cumulative"], marker="o", linewidth=2, markersize=6
        )
        ax2.axhline(
            y=self.variance_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold: {self.variance_threshold:.1%}",
        )
        ax2.set_xlabel("Number of Components", fontsize=12)
        ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
        ax2.set_title("Cumulative Explained Variance", fontsize=13, fontweight="bold")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.05)

        plt.suptitle(
            f"PCA Analysis - {self.rationale}", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_pc_loadings(
        self,
        pc_indices: List[int] = [0, 1],
        top_k: int = 15,
        output_path: Optional[str] = None,
    ):
        """
        Plot feature loadings for specified principal components.

        Args:
            pc_indices: Which PCs to plot (0-based)
            top_k: Number of top features to show per PC
            output_path: Path to save plot (optional)
        """
        if not self.is_fitted or self.pca is None:
            print("Model not fitted yet.")
            return

        n_pcs = len(pc_indices)
        fig, axes = plt.subplots(1, n_pcs, figsize=(7 * n_pcs, 6))

        if n_pcs == 1:
            axes = [axes]

        for i, pc_idx in enumerate(pc_indices):
            loadings_df = self.get_pc_loadings(pc_idx).head(top_k)

            ax = axes[i]
            colors = ["green" if x > 0 else "red" for x in loadings_df["loading"]]

            ax.barh(
                range(len(loadings_df)), loadings_df["loading"], color=colors, alpha=0.7
            )
            ax.set_yticks(range(len(loadings_df)))
            ax.set_yticklabels(loadings_df["feature"], fontsize=9)
            ax.set_xlabel("Loading", fontsize=11)
            ax.set_title(
                f"PC{pc_idx + 1} (Var: {self.explained_variance_ratio_[pc_idx]:.1%})",
                fontsize=12,
                fontweight="bold",
            )
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
            ax.grid(axis="x", alpha=0.3)

        plt.suptitle(
            f"Principal Component Loadings - {self.rationale}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def save(self, filepath: str):
        """Save model to disk."""
        save_dict = {
            "rationale": self.rationale,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "pca": self.pca,
            "classifier": self.classifier,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "n_components_used_": self.n_components_used_,
            "feature_importance_pca": self.feature_importance_pca,
            "hyperparameters": {
                "n_components": self.n_components,
                "variance_threshold": self.variance_threshold,
                "base_model_type": self.base_model_type,
                "calibrate": self.calibrate,
                "whiten": self.whiten,
                "custom_params": self.custom_params,
                "random_seed": self.random_seed,
            },
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"PCA model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        # Recreate model
        model = cls(rationale=save_dict["rationale"], **save_dict["hyperparameters"])

        # Restore state
        model.feature_names = save_dict["feature_names"]
        model.pca = save_dict["pca"]
        model.classifier = save_dict["classifier"]
        model.explained_variance_ratio_ = save_dict.get("explained_variance_ratio_")
        model.n_components_used_ = save_dict.get("n_components_used_")
        model.feature_importance_pca = save_dict.get("feature_importance_pca")
        model.is_fitted = save_dict["is_fitted"]

        print(f"PCA model loaded from {filepath}")
        return model


class MultiLabelPCA:
    """
    Multi-label wrapper for PCA models.
    Trains one PCA model per rationale with shared hyperparameters.
    """

    def __init__(
        self,
        rationales: List[str],
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        base_model_type: str = "logistic",
        calibrate: bool = True,
        whiten: bool = False,
        custom_params: Optional[Dict] = None,
        random_seed: int = 21,
    ):
        self.rationales = rationales
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.base_model_type = base_model_type
        self.calibrate = calibrate
        self.whiten = whiten
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
        """Train PCA models for all rationales."""
        for i, rationale in enumerate(self.rationales):
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Training {rationale} ({i + 1}/{len(self.rationales)})")
                print(f"{'=' * 80}")

            model = PCAModel(
                rationale=rationale,
                n_components=self.n_components,
                variance_threshold=self.variance_threshold,
                base_model_type=self.base_model_type,
                calibrate=self.calibrate,
                whiten=self.whiten,
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

    def get_explained_variance_summary(self) -> pd.DataFrame:
        """Get variance summary for all rationales."""
        summaries = []
        for rationale, model in self.models.items():
            if model.is_fitted:
                summaries.append(
                    {
                        "rationale": rationale,
                        "n_components": model.n_components_used_,
                        "explained_variance": model.explained_variance_ratio_.sum(),
                    }
                )

        return pd.DataFrame(summaries)
