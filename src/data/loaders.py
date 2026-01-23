import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

from src.utils.types import CORE_RATIONALES, ALL_RATIONALES, StratifyOption


class DataLoader:
    def __init__(self):
        self.df = None

    def load_data(
        self, filepath: str = "./data/Imputing Rationales.csv"
    ) -> pd.DataFrame:
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} records from {filepath}")
        return self.df

    def apply_filters(
        self, min_meetings_rat: int = 1, min_dissent: int = 5
    ) -> pd.DataFrame:
        initial_count = len(self.df)

        # Filter out non-disclosers
        if min_meetings_rat > 0:
            self.df = self.df[self.df["N_Meetings_Rat"] > 0].copy()
            print(f"After removing non-disclosers: {len(self.df)} records")

        if min_meetings_rat > 1:
            self.df = self.df[self.df["N_Meetings_Rat"] >= min_meetings_rat].copy()
            print(f"After N_Meetings_Rat >= {min_meetings_rat}: {len(self.df)} records")

        if min_dissent > 0:
            self.df = self.df[self.df["N_dissent"] >= min_dissent].copy()
            print(f"After N_dissent >= {min_dissent}: {len(self.df)} records")

        print(f"Filtered from {initial_count} to {len(self.df)} records")
        return self.df

    def get_labeled_data(self, rationales: List[str]) -> pd.DataFrame:
        dissent_df = self.df[self.df["ind_dissent"] == 1].copy()

        # Find observations with at least one non-missing rationale
        has_label = dissent_df[rationales].notna().any(axis=1)
        labeled_df = dissent_df[has_label].copy()

        print(
            f"Found {len(labeled_df)} labeled observations out of {len(dissent_df)} dissents"
        )
        return labeled_df

    def get_unlabeled_data(self, rationales: List[str]) -> pd.DataFrame:
        dissent_df = self.df[self.df["ind_dissent"] == 1].copy()

        # Find observations with all rationales missing
        no_label = dissent_df[rationales].isna().all(axis=1)
        unlabeled_df = dissent_df[no_label].copy()

        print(
            f"Found {len(unlabeled_df)} unlabeled observations out of {len(dissent_df)} dissents"
        )
        return unlabeled_df

    def split_train_test(
        self,
        test_size: float = 0.2,
        random_seed: int = 21,
        label: List[str] = CORE_RATIONALES,
        stratify_by: Optional[StratifyOption] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        labeled_df = self.get_labeled_data(label)

        if stratify_by and stratify_by in labeled_df.columns:
            stratify = labeled_df[stratify_by]
        else:
            stratify = None

        train_df, test_df = train_test_split(
            labeled_df, test_size=test_size, random_state=random_seed, stratify=stratify
        )

        print(f"Train set: {len(train_df)} observations")
        print(f"Test set: {len(test_df)} observations")

        return train_df, test_df
