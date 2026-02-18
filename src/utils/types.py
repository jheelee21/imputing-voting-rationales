from enum import Enum
from typing import List


class ModelType(Enum):
    SUPERVISED = "supervised"
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    MC_DROPOUT = "mc_dropout"
    BNN = "bnn"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    SPARSE_GP = "sparse_gp"
    DEEP_KERNEL_GP = "deep_kernel_gp"
    SEMI_SUPERVISED = "semi_supervised"
    GAUSSIAN_PROCESS = "gaussian_process"
    PCA = "pca"
    HIERARCHICAL = "hierarchical"
    BAYESIAN_HIERARCHICAL = "bayesian_hierarchical"
    BHM_IMPROVED = "bhm_improved"


def get_trainable_model_types() -> List[str]:
    """Model type strings accepted by scripts/train.py."""
    return list(model.value for model in ModelType)


def get_extended_model_types() -> List[str]:
    """Subset that requires ExtendedModelTrainer."""
    return [
        ModelType.BNN.value,
        ModelType.CATBOOST.value,
        ModelType.LIGHTGBM.value,
        ModelType.XGBOOST.value,
        ModelType.SPARSE_GP.value,
        ModelType.DEEP_KERNEL_GP.value,
        ModelType.PCA.value,
        ModelType.HIERARCHICAL.value,
        ModelType.BHM_IMPROVED.value,
    ]
