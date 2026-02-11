from enum import Enum


class ModelType(Enum):
    SUPERVISED = "supervised"
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"
    MC_DROPOUT = "mc_dropout"
    BNN = "bnn"
    CATBOOST = "catboost"
    SEMI_SUPERVISED = "semi_supervised"
    GAUSSIAN_PROCESS = "gaussian_process"
    PCA = "pca"
    BAYESIAN_HIERARCHICAL = "bayesian_hierarchical"
