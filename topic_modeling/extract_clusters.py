
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── ID columns used across the project ───────────────────────────────────────
ID_COLUMNS = ["investor_id", "pid", "ProxySeason", "meeting_id", "id_rationale"]


# ─────────────────────────────────────────────────────────────────────────────
# Core helper  (importable from script.py)
# ─────────────────────────────────────────────────────────────────────────────

def save_clusters(
    df_sub: pd.DataFrame,
    labels: np.ndarray,
    output_path: str,
    id_columns: list[str] = ID_COLUMNS,
) -> pd.DataFrame:
    """
    Attach cluster labels to df_sub and write a slim ID + cluster file.

    Parameters
    ----------
    df_sub      : The filtered dataframe used for clustering (indep == 1).
                  Its row order must match `labels`.
    labels      : 1-D array of integer cluster assignments, aligned with df_sub.
    output_path : Destination file path (csv / dta / parquet / xlsx supported).
    id_columns  : Columns to keep alongside the cluster label.

    Returns
    -------
    DataFrame with [id_columns..., "cluster"] written to output_path.
    """
    if len(labels) != len(df_sub):
        raise ValueError(
            f"labels length ({len(labels)}) does not match df_sub rows ({len(df_sub)})."
        )

    # Keep only the ID columns that actually exist in the data
    present_ids = [c for c in id_columns if c in df_sub.columns]
    if not present_ids:
        raise ValueError(
            f"None of the expected ID columns {id_columns} were found in df_sub."
        )

    result = df_sub[present_ids].copy().reset_index(drop=True)
    result["cluster"] = labels

    _write(result, output_path)
    print(f"  Cluster assignments : {output_path}  ({len(result):,} rows, "
          f"{result['cluster'].nunique()} unique clusters)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers (mirrors script.py)
# ─────────────────────────────────────────────────────────────────────────────

def _read(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    readers = {
        ".csv":     lambda p: pd.read_csv(p, low_memory=False),
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
        ".dta":     pd.read_stata,
        ".parquet": pd.read_parquet,
        ".pkl":     pd.read_pickle,
    }
    if ext not in readers:
        raise ValueError(f"Unsupported format: {ext}")
    return readers[ext](path)


def _write(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(path).suffix.lower()
    writers = {
        ".csv":     lambda df, p: df.to_csv(p, index=False),
        ".xlsx":    lambda df, p: df.to_excel(p, index=False),
        ".xls":     lambda df, p: df.to_excel(p, index=False),
        ".dta":     lambda df, p: df.to_stata(p, write_index=False),
        ".parquet": lambda df, p: df.to_parquet(p, index=False),
    }
    writers.get(ext, lambda df, p: df.to_csv(p, index=False))(df, path)


def main():
    parser = argparse.ArgumentParser(
        description="Keep only id_rationale and cluster columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="embeddings/embedded_rationale_text.csv",
        help="Path to embedded_rationale_text.csv",
    )
    parser.add_argument(
        "--output",
        default="embeddings/clusters_only.csv",
        help="Output file path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    for col in ("id_rationale", "cluster"):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

    slim = df[["id_rationale", "cluster"]].copy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(args.output, index=False)

    print(f"Saved {len(slim):,} rows -> {args.output}")
    print(f"\nCluster distribution:")
    print(slim["cluster"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()