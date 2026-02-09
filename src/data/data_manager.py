"""
Unified data handling: loading, filtering, splitting, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from configs.config import ALL_RATIONALES

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import (
    CORE_RATIONALES,
    REQUIRED_FEATURES,
    GENERAL_FEATURES,
    CATEGORICAL_IDS,
)


class DataManager:
    """Unified data management: load, filter, split, and preprocess."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}

    # ==================== LOADING & FILTERING ====================

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df):,} records from {filepath}")
        return self.df

    def apply_filters(
        self, min_meetings_rat: int = 1, min_dissent: int = 5
    ) -> pd.DataFrame:
        """Apply filters per PDF instructions."""
        initial_count = len(self.df)

        # Filter 1: Remove non-disclosers (N_Meetings_Rat = 0)
        self.df = self.df[self.df["N_Meetings_Rat"] > 0].copy()
        print(f"After removing non-disclosers: {len(self.df):,} records")

        # Filter 2: Minimum meetings with rationales
        if min_meetings_rat > 1:
            self.df = self.df[self.df["N_Meetings_Rat"] >= min_meetings_rat].copy()
            print(
                f"After N_Meetings_Rat >= {min_meetings_rat}: {len(self.df):,} records"
            )

        # Filter 3: Minimum dissenting votes
        if min_dissent > 0:
            self.df = self.df[self.df["N_dissent"] >= min_dissent].copy()
            print(f"After N_dissent >= {min_dissent}: {len(self.df):,} records")

        print(f"Total filtered: {initial_count:,} → {len(self.df):,} records")
        return self.df

    def get_labeled_data(self, rationales: List[str]) -> pd.DataFrame:
        """Get observations with at least one labeled rationale (for training)."""
        dissent_df = self.df[self.df["ind_dissent"] == 1].copy()
        has_label = dissent_df[rationales].notna().any(axis=1)
        labeled_df = dissent_df[has_label].copy()

        print(
            f"Labeled observations: {len(labeled_df):,} / {len(dissent_df):,} dissents"
        )
        return labeled_df

    def get_unlabeled_data(self, rationales: List[str]) -> pd.DataFrame:
        """Get observations with all rationales missing (for prediction)."""
        dissent_df = self.df[self.df["ind_dissent"] == 1].copy()
        no_label = dissent_df[rationales].isna().all(axis=1)
        unlabeled_df = dissent_df[no_label].copy()

        print(
            f"Unlabeled observations: {len(unlabeled_df):,} / {len(dissent_df):,} dissents"
        )
        return unlabeled_df

    def split_data(
        self,
        test_size: float = 0.2,
        random_seed: int = 21,
        rationales: List[str] = None,
        stratify_by: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split labeled data into train/test sets."""
        rationales = rationales or CORE_RATIONALES
        labeled_df = self.get_labeled_data(rationales)

        stratify = (
            labeled_df[stratify_by]
            if stratify_by and stratify_by in labeled_df.columns
            else None
        )

        train_df, test_df = train_test_split(
            labeled_df, test_size=test_size, random_state=random_seed, stratify=stratify
        )

        print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
        return train_df, test_df

    # ==================== FEATURE PREPARATION ====================

    def get_required_features(
        self,
        rationales: List[str],
        use_all_features: bool = False,
        exclude_cols: List[str] = None,
    ) -> List[str]:
        """
        Get required features based on target rationales.

        Args:
            rationales: List of target rationales
            use_all_features: If True, use all available columns (subject to filtering)
            exclude_cols: Columns to explicitly exclude

        Returns:
            List of feature column names
        """
        exclude_cols = exclude_cols or []

        if use_all_features:
            # Start with all columns except rationales and IDs that shouldn't be features
            exclude_set = set(
                exclude_cols + ALL_RATIONALES + ["meeting_id", "ind_dissent"]
            )
            required = [col for col in self.df.columns if col not in exclude_set]
        else:
            # Use only required features per rationale
            required = set(GENERAL_FEATURES + CATEGORICAL_IDS)

            for rationale in rationales:
                if rationale in REQUIRED_FEATURES:
                    required.update(REQUIRED_FEATURES[rationale])

            required = list(required)

        return required

    def prepare_features(
        self,
        df: pd.DataFrame,
        rationales: List[str],
        additional_features: List[str] = None,
        drop_high_missing: float = 1.0,
        use_all_features: bool = False,
        exclude_cols: List[str] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Prepare feature matrix.

        Args:
            df: Input dataframe
            rationales: Target rationales
            additional_features: Extra features to include
            drop_high_missing: Drop features with missing rate > this threshold (0.0-1.0)
            use_all_features: If True, use all available columns
            exclude_cols: Columns to explicitly exclude
            verbose: Print detailed information

        Returns:
            DataFrame with selected features
        """
        df = df.copy()

        # Get required features
        feature_cols = self.get_required_features(
            rationales, use_all_features=use_all_features, exclude_cols=exclude_cols
        )

        # Add additional features
        if additional_features:
            feature_cols.extend([f for f in additional_features if f in df.columns])

        # Remove duplicates while preserving order
        seen = set()
        feature_cols = [x for x in feature_cols if x not in seen and not seen.add(x)]

        # Filter by columns that actually exist in df
        feature_cols = [f for f in feature_cols if f in df.columns]

        if verbose:
            print(f"Initial features: {len(feature_cols)}")

        # Drop features with too many missing values
        if drop_high_missing < 1.0:
            missing_rates = df[feature_cols].isna().mean()
            keep_features = missing_rates[
                missing_rates <= drop_high_missing
            ].index.tolist()
            dropped = set(feature_cols) - set(keep_features)

            if dropped:
                dropped_info = [(col, missing_rates[col]) for col in sorted(dropped)]
                if verbose:
                    print(
                        f"\nDropped {len(dropped)} high-missing features (threshold={drop_high_missing}):"
                    )
                    for col, rate in dropped_info[:10]:  # Show first 10
                        print(f"  {col}: {rate:.2%} missing")
                    if len(dropped_info) > 10:
                        print(f"  ... and {len(dropped_info) - 10} more")
                else:
                    print(
                        f"Dropped {len(dropped)} features with missing rate > {drop_high_missing:.1%}"
                    )

            feature_cols = keep_features

        if verbose:
            print(f"Final features: {len(feature_cols)}")

        return df[feature_cols]

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()

        for col in CATEGORICAL_IDS:
            if col not in df.columns:
                continue

            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                    df[col].astype(str)
                )
            else:
                le = self.label_encoders[col]
                col_values = df[col].astype(str)
                # Handle unseen categories
                mapping = {val: idx for idx, val in enumerate(le.classes_)}
                df[f"{col}_encoded"] = col_values.map(mapping).fillna(-1).astype(int)

        return df

    def handle_missing(
        self,
        df: pd.DataFrame,
        strategy: str = "zero",
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Handle missing values in numerical columns.
        For median/mean, values are fitted on training data (fit=True) and stored;
        at predict time (fit=False) stored values are used so train and inference match.
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if strategy == "median":
            if fit:
                self._imputation_values = df[numeric_cols].median().to_dict()
            vals = getattr(self, "_imputation_values", None) or {}
            for col in numeric_cols:
                fill_val = vals.get(col, 0.0) if not fit else df[col].median()
                df[col] = df[col].fillna(fill_val)
        elif strategy == "mean":
            if fit:
                self._imputation_values = df[numeric_cols].mean().to_dict()
            vals = getattr(self, "_imputation_values", None) or {}
            for col in numeric_cols:
                fill_val = vals.get(col, 0.0) if not fit else df[col].mean()
                df[col] = df[col].fillna(fill_val)
        else:  # "zero"
            df[numeric_cols] = df[numeric_cols].fillna(0)
            if fit and not hasattr(self, "_imputation_values"):
                self._imputation_values = None

        return df

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale numerical features."""
        # Convert to numpy to avoid feature name issues
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        if fit:
            return self.scaler.fit_transform(X_values)
        else:
            return self.scaler.transform(X_values)

    # ==================== COMPLETE PIPELINE ====================

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        rationales: List[str],
        additional_features: List[str] = None,
        drop_high_missing: float = 1.0,
        use_all_features: bool = False,
        exclude_cols: List[str] = None,
        missing_strategy: str = "zero",
        fit: bool = True,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Complete preprocessing pipeline for training or inference.

        Args:
            df: Input dataframe
            rationales: Target rationales to predict
            additional_features: Extra features to include
            drop_high_missing: Drop features with missing rate > threshold (0.0-1.0)
            use_all_features: If True, use all available columns (subject to filtering)
            exclude_cols: Columns to explicitly exclude
            missing_strategy: Imputation for remaining missing numerics: "zero" | "median" | "mean"
            fit: If True, fit scalers/encoders/imputation; if False, transform only
            verbose: Print detailed information

        Returns:
            X: Feature matrix (numpy array)
            y: Label matrix (numpy array or None)
            feature_names: List of feature names
        """
        # Prepare features
        X_df = self.prepare_features(
            df,
            rationales,
            additional_features,
            drop_high_missing=drop_high_missing,
            use_all_features=use_all_features,
            exclude_cols=exclude_cols,
            verbose=verbose,
        )

        # Encode categorical
        X_df = self.encode_categorical(X_df, fit=fit)

        # Handle missing (median/mean fitted on train, reused at predict)
        X_df = self.handle_missing(X_df, strategy=missing_strategy, fit=fit)

        # Separate numerical and categorical (use only _encoded for IDs; exclude raw categoricals from numerical)
        categorical_cols = [c for c in X_df.columns if c.endswith("_encoded")]
        numerical_cols = [
            c
            for c in X_df.columns
            if c not in categorical_cols and c not in CATEGORICAL_IDS
        ]

        # For inference (fit=False), ensure we use the exact same features as training
        if not fit and hasattr(self, "_training_numerical_cols"):
            # Add missing columns with zeros
            for col in self._training_numerical_cols:
                if col not in X_df.columns:
                    X_df[col] = 0
                    if verbose:
                        print(f"Warning: Added missing column '{col}' with zeros")

            # Use only the columns from training in the same order
            numerical_cols = self._training_numerical_cols
            categorical_cols = self._training_categorical_cols

            # Also ensure categorical columns exist
            for col in categorical_cols:
                if col not in X_df.columns:
                    X_df[col] = -1  # Use -1 for missing categorical
                    if verbose:
                        print(
                            f"Warning: Added missing categorical column '{col}' with -1"
                        )
        else:
            # Store for future use
            self._training_numerical_cols = numerical_cols
            self._training_categorical_cols = categorical_cols

        # Scale numerical features (convert to values to avoid feature name issues)
        numerical_values = X_df[numerical_cols].values
        X_scaled = (
            self.scaler.fit_transform(numerical_values)
            if fit
            else self.scaler.transform(numerical_values)
        )

        # Combine features
        X = np.hstack([X_scaled, X_df[categorical_cols].values])

        # Get feature names
        feature_names = list(numerical_cols) + list(categorical_cols)

        # Get labels if available
        y = None
        if all(r in df.columns for r in rationales):
            y = df[rationales].fillna(0).astype(int).values

        if verbose:
            print(f"Final shape: X={X.shape}, y={y.shape if y is not None else None}")
            print(f"Total features: {len(feature_names)}")

        return X, y, feature_names

    # all entries with missing rationales are treated as unlabeled, so we can use them for inference
    def prepare_for_inference(
        self,
        df: pd.DataFrame,
        rationales: List[str],
    ):
        """
        Lightweight wrapper around prepare_for_training for inference.

        At prediction time we typically pass in a pre-filtered dataframe
        (e.g. output of get_unlabeled_data). Here we just reuse the exact
        preprocessing pipeline but with fit=False so that scalers,
        encoders, and imputation parameters learned during training are
        applied consistently.
        """
        return self.prepare_for_training(
            df=df,
            rationales=rationales,
            fit=False,
        )
