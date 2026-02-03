import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple

from src.utils.types import (
    CORE_RATIONALES,
    REQUIRED_FEATURES,
    GENERAL_FEATURES,
    CATEGORICAL_IDS,
)


class DataPreprocessor:
    def __init__(self, rationales: List[str] = None):
        self.rationales = rationales or CORE_RATIONALES
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names = []

    def get_required_features(self) -> List[str]:
        required = set(GENERAL_FEATURES + CATEGORICAL_IDS)

        for rationale in self.rationales:
            if rationale in REQUIRED_FEATURES:
                required.update(REQUIRED_FEATURES[rationale])

        return list(required)

    def prepare_features(
        self,
        df: pd.DataFrame,
        additional_features: List[str] = None,
        drop_high_missing: float = 1.0,
        use_all_features: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()

        if use_all_features:
            feature_cols = df.columns.tolist()
        else:
            feature_cols = self.get_required_features()

            if additional_features:
                feature_cols.extend([f for f in additional_features if f in df.columns])

            feature_cols = list(set(feature_cols))

        if drop_high_missing:
            missing_rates = df[feature_cols].isna().mean()
            keep_features = missing_rates[
                missing_rates <= drop_high_missing
            ].index.tolist()
            dropped = set(feature_cols) - set(keep_features)
            if dropped:
                print(
                    f"Dropped {len(dropped)} features with >{drop_high_missing * 100}% missing: {dropped}"
                )
            feature_cols = keep_features

        return df[feature_cols]

    def encode_categorical(
        self,
        df: pd.DataFrame,
        fit: bool = True,  # true for training, false for inference
    ) -> pd.DataFrame:
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
                # unseen categories
                le = self.label_encoders[col]
                df[f"{col}_encoded"] = (
                    df[col]
                    .astype(str)
                    .map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                )

        return df

    def handle_missing(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        match strategy:
            case "median":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            case "mean":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            case "zero":
                df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def prepare_data(
        self, df: pd.DataFrame, fit: bool = True, scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        df = self.encode_categoricals(df, fit=fit)
        df = self.handle_missing(df)

        feature_cols = [c for c in df.columns if c not in self.rationales]
        X = df[feature_cols].copy()
        y = df[self.rationales].copy()

        y = (y == 1).astype(int)

        if scale:
            X = self.scale_features(X, fit=fit)
        else:
            X = X.values

        return X, y.values
