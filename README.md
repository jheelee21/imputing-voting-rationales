# imputing-voting-rationales

Probability-Based Predictions for Voting Rationale Classification Tasks

The goal is to build a **multi-label classification** model to predict the specific reasons (rationales) why an investor voted **"Against"** a director, and to **infer disclosers’ missing rationales**.

## Modeling scope (strict)

- We model the **reason for dissent conditional on dissent having occurred** (`ind_dissent = 1`). We do not predict whether dissent occurred.
- **Training**: Only dissent rows with at least one labeled rationale are used. For each rationale, missing labels are either excluded (supervised: one model per rationale) or masked in the loss (MC Dropout: multi-label with partial labels).
- **Prediction (imputation)**: Targets dissent rows where **all** rationales are missing; the model infers those missing labels.