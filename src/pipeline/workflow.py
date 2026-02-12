"""Shared workflow helpers for training and prediction pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from configs.config import DATA_CONFIG, FEATURE_CONFIG, CORE_RATIONALES
from src.data.data_manager import DataManager


@dataclass
class WorkflowConfig:
    """Common run-time configuration used by train/evaluate/predict scripts."""

    data_path: str = DATA_CONFIG["data_path"]
    min_meetings_rat: int = DATA_CONFIG["min_meetings_rat"]
    min_dissent: int = DATA_CONFIG["min_dissent"]
    random_seed: int = DATA_CONFIG["random_seed"]
    test_size: float = DATA_CONFIG["test_size"]


def load_and_filter_data(
    data_manager: DataManager,
    workflow: WorkflowConfig,
) -> pd.DataFrame:
    """Load raw CSV and apply project-level sample filters."""
    data_manager.load_data(workflow.data_path)
    return data_manager.apply_filters(
        min_meetings_rat=workflow.min_meetings_rat,
        min_dissent=workflow.min_dissent,
    )


def split_labeled_data(
    data_manager: DataManager,
    rationales: Sequence[str],
    workflow: WorkflowConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dissent+labeled rows into train/validation subsets."""
    return data_manager.split_data(
        test_size=workflow.test_size,
        random_seed=workflow.random_seed,
        rationales=list(rationales),
    )


def resolve_rationales(requested: Optional[Sequence[str]] = None) -> List[str]:
    """Return validated rationale list (defaults to CORE_RATIONALES)."""
    rationales = list(requested) if requested else list(CORE_RATIONALES)
    if len(rationales) == 0:
        raise ValueError("At least one rationale must be provided")
    return rationales


def build_feature_config(
    required_only: bool,
    drop_high_missing: Optional[float] = None,
    exclude_cols: Optional[Sequence[str]] = None,
    missing_strategy: Optional[str] = None,
) -> Dict:
    """Create consistent feature/preprocessing config for all trainers."""
    return {
        "use_all_features": not required_only,
        "drop_high_missing": (
            drop_high_missing
            if drop_high_missing is not None
            else FEATURE_CONFIG.get("drop_high_missing", 0.5)
        ),
        "exclude_cols": list(exclude_cols)
        if exclude_cols is not None
        else FEATURE_CONFIG.get("exclude_cols"),
        "missing_strategy": missing_strategy
        or FEATURE_CONFIG.get("missing_strategy", "median"),
    }


def resolve_save_dir(
    default_root: Path, model_type: str, save_dir: Optional[str]
) -> Path:
    """Resolve and create model output directory."""
    path = Path(save_dir) if save_dir else default_root / model_type
    path.mkdir(parents=True, exist_ok=True)
    return path
