from enum import Enum

CORE_RATIONALES = ["diversity", "indep", "tenure", "busyness", "combined_ceo_chairman"]
ALL_RATIONALES = [
    "diversity",
    "boardstructure",
    "indep",
    "tenure",
    "governance",
    "combined_ceo_chairman",
    "esg_csr",
    "responsiveness",
    "attendance",
    "compensation",
    "busyness",
    "norat_misc",
]
REQUIRED_FEATURES = {
    "diversity": ["Per_female", "D_Per_female"],
    "tenure": ["AvTenure", "D_AvTenure"],
    "indep": ["Per_Independent", "D_Independent"],
}

# General must-include features
GENERAL_FEATURES = ["frac_vote_against"]
CATEGORICAL_IDS = ["investor_id", "pid", "ProxySeason"]


class StratifyOption(Enum):
    INVESTOR_ID = "investor_id"
    FIRM_ID = "pid"
