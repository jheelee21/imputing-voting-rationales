"""
Unified data handling: loading, filtering, splitting, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import (
    CORE_RATIONALES, REQUIRED_FEATURES, GENERAL_FEATURES, CATEGORICAL_IDS
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
        self,
        min_meetings_rat: int = 1,
        min_dissent: int = 5
    ) -> pd.DataFrame:
        """Apply filters per PDF instructions."""
        initial_count = len(self.df)
        
        # Filter 1: Remove non-disclosers (N_Meetings_Rat = 0)
        self.df = self.df[self.df["N_Meetings_Rat"] > 0].copy()
        print(f"After removing non-disclosers: {len(self.df):,} records")
        
        # Filter 2: Minimum meetings with rationales
        if min_meetings_rat > 1:
            self.df = self.df[self.df["N_Meetings_Rat"] >= min_meetings_rat].copy()
            print(f"After N_Meetings_Rat >= {min_meetings_rat}: {len(self.df):,} records")
        
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
        
        print(f"Labeled observations: {len(labeled_df):,} / {len(dissent_df):,} dissents")
        return labeled_df
    
    def get_unlabeled_data(self, rationales: List[str]) -> pd.DataFrame:
        """Get observations with all rationales missing (for prediction)."""
        dissent_df = self.df[self.df["ind_dissent"] == 1].copy()
        no_label = dissent_df[rationales].isna().all(axis=1)
        unlabeled_df = dissent_df[no_label].copy()
        
        print(f"Unlabeled observations: {len(unlabeled_df):,} / {len(dissent_df):,} dissents")
        return unlabeled_df
    
    def split_data(
        self,
        test_size: float = 0.2,
        random_seed: int = 21,
        rationales: List[str] = None,
        stratify_by: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split labeled data into train/test sets."""
        from configs.config import CORE_RATIONALES
        
        rationales = rationales or CORE_RATIONALES
        labeled_df = self.get_labeled_data(rationales)
        
        stratify = labeled_df[stratify_by] if stratify_by and stratify_by in labeled_df.columns else None
        
        train_df, test_df = train_test_split(
            labeled_df, test_size=test_size, random_state=random_seed, stratify=stratify
        )
        
        print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
        return train_df, test_df
    
    # ==================== FEATURE PREPARATION ====================
    
    def get_required_features(self, rationales: List[str]) -> List[str]:
        """Get required features based on target rationales."""
        required = set(GENERAL_FEATURES + CATEGORICAL_IDS)
        
        for rationale in rationales:
            if rationale in REQUIRED_FEATURES:
                required.update(REQUIRED_FEATURES[rationale])
        
        return list(required)
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        rationales: List[str],
        additional_features: List[str] = None,
        drop_high_missing: float = 1.0,
    ) -> pd.DataFrame:
        """Prepare feature matrix."""
        df = df.copy()
        
        # Get required features
        feature_cols = self.get_required_features(rationales)
        
        # Add additional features
        if additional_features:
            feature_cols.extend([f for f in additional_features if f in df.columns])
        
        feature_cols = list(set(feature_cols))
        
        # Drop features with too many missing values
        if drop_high_missing < 1.0:
            missing_rates = df[feature_cols].isna().mean()
            keep_features = missing_rates[missing_rates <= drop_high_missing].index.tolist()
            dropped = set(feature_cols) - set(keep_features)
            if dropped:
                print(f"Dropped {len(dropped)} high-missing features: {sorted(dropped)}")
            feature_cols = keep_features
        
        return df[feature_cols]
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
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
        strategy: str = "mean"
    ) -> pd.DataFrame:
        """Handle missing values in numerical columns."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """Scale numerical features."""
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    # ==================== COMPLETE PIPELINE ====================
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        rationales: List[str],
        additional_features: List[str] = None,
        fit: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Complete preprocessing pipeline for training or inference.
        
        Returns:
            X: Feature matrix
            y: Label matrix (or None for unlabeled data)
            feature_names: List of feature names
        """
        # Prepare features
        X_df = self.prepare_features(df, rationales, additional_features)
        
        # Encode categorical
        X_df = self.encode_categorical(X_df, fit=fit)
        
        # Handle missing
        X_df = self.handle_missing(X_df)
        
        # Separate numerical and categorical
        categorical_cols = [c for c in X_df.columns if c.endswith("_encoded")]
        numerical_cols = [c for c in X_df.columns if c not in categorical_cols]
        
        # Scale numerical features
        if hasattr(self.scaler, 'feature_names_in_') and not fit:
            # Use exact column order from training
            X_scaled = self.scaler.transform(X_df[self.scaler.feature_names_in_])
        else:
            X_scaled = self.scale_features(X_df[numerical_cols], fit=fit)
        
        # Combine features
        X = np.hstack([X_scaled, X_df[categorical_cols].values])
        
        # Get feature names
        feature_names = list(numerical_cols) + list(categorical_cols)
        
        # Get labels if available
        y = None
        if all(r in df.columns for r in rationales):
            y = df[rationales].fillna(0).astype(int).values
        
        return X, y, feature_names