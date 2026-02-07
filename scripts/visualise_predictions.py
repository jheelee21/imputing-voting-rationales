import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import CORE_RATIONALES


def plot_probability_distributions(
    predictions_df: pd.DataFrame, rationales: list, output_dir: Path
):
    """Plot distribution of predicted probabilities for each rationale."""
    prob_cols = [
        f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns
    ]

    if not prob_cols:
        print("No probability columns found")
        return

    n_rationales = len(prob_cols)
    n_cols = 3
    n_rows = (n_rationales + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rationales > 1 else [axes]

    for i, prob_col in enumerate(prob_cols):
        ax = axes[i]
        rationale = prob_col.replace("_prob", "")

        probs = predictions_df[prob_col].dropna()

        # Histogram
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 25)
        bin_width = 0.025
        bins = np.arange(min(probs), max(probs) + bin_width, bin_width)
        ax.hist(probs, bins=bins, alpha=0.7, edgecolor="black", density=True)

        # Add mean line
        mean_prob = probs.mean()
        ax.axvline(
            mean_prob,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_prob:.3f}",
        )

        # # Add median line
        # median_prob = probs.median()
        # ax.axvline(
        #     median_prob,
        #     color="blue",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f"Median: {median_prob:.3f}",
        # )

        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{rationale} - Probability Distribution", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Remove extra subplots
    for i in range(n_rationales, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    output_path = output_dir / "probability_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Probability distributions saved to {output_path}")


def plot_correlation_heatmap(
    predictions_df: pd.DataFrame, rationales: list, output_dir: Path
):
    """Plot correlation heatmap between predicted probabilities."""
    prob_cols = [
        f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns
    ]

    if len(prob_cols) < 2:
        print("Need at least 2 rationales for correlation heatmap")
        return

    # Calculate correlation
    corr = predictions_df[prob_cols].corr()

    # Rename columns for better display
    corr.columns = [col.replace("_prob", "") for col in corr.columns]
    corr.index = [idx.replace("_prob", "") for idx in corr.index]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Correlation Between Predicted Probabilities", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    output_path = output_dir / "probability_correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Correlation heatmap saved to {output_path}")


def plot_confidence_comparison(
    predictions_df: pd.DataFrame, rationales: list, output_dir: Path
):
    """Plot comparison of confidence levels across rationales."""
    prob_cols = [
        f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns
    ]

    if not prob_cols:
        return

    # Calculate confidence levels
    confidence_data = []
    for prob_col in prob_cols:
        rationale = prob_col.replace("_prob", "")
        probs = predictions_df[prob_col].dropna()

        confidence_data.append(
            {
                "Rationale": rationale,
                "High (≥0.7)": (probs >= 0.7).sum(),
                "Medium (0.3-0.7)": ((probs >= 0.3) & (probs < 0.7)).sum(),
                "Low (<0.3)": (probs < 0.3).sum(),
            }
        )

    conf_df = pd.DataFrame(confidence_data)
    conf_df = conf_df.set_index("Rationale")

    # Plot stacked bar chart
    ax = conf_df.plot(
        kind="barh",
        stacked=True,
        figsize=(12, 6),
        color=["#2ecc71", "#f39c12", "#e74c3c"],
    )

    plt.xlabel("Number of Predictions", fontsize=12)
    plt.ylabel("Rationale", fontsize=12)
    plt.title(
        "Prediction Confidence Distribution by Rationale",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(title="Confidence Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_path = output_dir / "confidence_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Confidence comparison saved to {output_path}")


def plot_mean_probabilities(
    predictions_df: pd.DataFrame, rationales: list, output_dir: Path
):
    """Plot bar chart of mean predicted probabilities."""
    prob_cols = [
        f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns
    ]

    if not prob_cols:
        return

    means = []
    stds = []
    names = []

    for prob_col in prob_cols:
        rationale = prob_col.replace("_prob", "")
        probs = predictions_df[prob_col].dropna()
        means.append(probs.mean())
        stds.append(probs.std())
        names.append(rationale)

    # Sort by mean
    sorted_indices = np.argsort(means)[::-1]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    names = [names[i] for i in sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(names))

    bars = ax.barh(
        x_pos,
        means,
        xerr=stds,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            mean + std + 0.01,
            i,
            f"{mean:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(x_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rationale", fontsize=12, fontweight="bold")
    ax.set_title(
        "Mean Predicted Probabilities by Rationale\n(Error bars show ±1 std)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(means) + max(stds) + 0.1)

    plt.tight_layout()
    output_path = output_dir / "mean_probabilities.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Mean probabilities chart saved to {output_path}")


def plot_boxplots(predictions_df: pd.DataFrame, rationales: list, output_dir: Path):
    """Plot boxplots of predicted probabilities."""
    prob_cols = [
        f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns
    ]

    if not prob_cols:
        return

    # Prepare data
    data_list = []
    labels = []
    for prob_col in prob_cols:
        rationale = prob_col.replace("_prob", "")
        probs = predictions_df[prob_col].dropna()
        data_list.append(probs.values)
        labels.append(rationale)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        data_list, labels=labels, patch_artist=True, notch=True, showmeans=True
    )

    ax.set_ylim(0, 1)

    # Customize colors
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)

    for mean in bp["means"]:
        mean.set_marker("D")
        mean.set_markerfacecolor("green")
        mean.set_markersize(6)

    ax.set_ylabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_xlabel("Rationale", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribution of Predicted Probabilities\n(Red line = median, Green diamond = mean)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks()

    plt.tight_layout()
    output_path = output_dir / "probability_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Probability boxplots saved to {output_path}")


def plot_multi_label_distribution(
    predictions_df: pd.DataFrame, rationales: list, output_dir: Path
):
    """Plot distribution of number of rationales per observation."""
    prob_cols = [
        f"{r}_prob" for r in rationales if f"{r}_prob" in predictions_df.columns
    ]

    if not prob_cols:
        return

    # Count rationales at different thresholds
    thresholds = [0.3, 0.5, 0.7]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, threshold in enumerate(thresholds):
        ax = axes[i]

        # Binary predictions at threshold
        binary_preds = (predictions_df[prob_cols] >= threshold).astype(int)
        n_rationales = binary_preds.sum(axis=1)

        # Plot histogram
        counts = n_rationales.value_counts().sort_index()
        ax.bar(counts.index, counts.values, alpha=0.7, edgecolor="black", linewidth=1.5)

        ax.set_xlabel("Number of Rationales", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=10, fontweight="bold")
        ax.set_title(
            f"Threshold = {threshold}\n(Mean: {n_rationales.mean():.2f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for x, y in zip(counts.index, counts.values):
            ax.text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "Number of Predicted Rationales per Observation at Different Thresholds",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / "multi_label_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Multi-label distribution saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize probability predictions")
    parser.add_argument(
        "--pred_dir",
        type=str,
        help="Path to predictions CSV file",
    )
    parser.add_argument(
        "--rationales",
        nargs="+",
        default=CORE_RATIONALES,
        help="Rationales to visualize",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VISUALIZING PROBABILITY PREDICTIONS")
    print("=" * 80)
    print(f"Predictions directory: {args.pred_dir}")
    print(f"Rationales: {args.rationales}")
    print("=" * 80)

    # Load predictions
    print("\nLoading predictions...")

    predictions_df = pd.read_csv(Path(args.pred_dir) / "predictions.csv")

    output_dir = Path(args.pred_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Loaded {len(predictions_df):,} predictions")

    # Generate visualizations
    print("\nGenerating visualizations...")
    print(f"{'-' * 80}")

    plot_probability_distributions(predictions_df, args.rationales, output_dir)
    # plot_correlation_heatmap(predictions_df, args.rationales, output_dir)
    plot_confidence_comparison(predictions_df, args.rationales, output_dir)
    plot_mean_probabilities(predictions_df, args.rationales, output_dir)
    plot_boxplots(predictions_df, args.rationales, output_dir)
    plot_multi_label_distribution(predictions_df, args.rationales, output_dir)

    print(f"{'-' * 80}")
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"\n{'=' * 80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
