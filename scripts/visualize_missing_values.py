from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "leishmania_ml_dataset.csv"
FIGURES_PATH = ROOT / "figures"


def main() -> None:
    FIGURES_PATH.mkdir(exist_ok=True)
    model_df = pd.read_csv(DATA_PATH)

    missing_count = model_df.isna().sum()
    missing_percent = missing_count / len(model_df) * 100
    missing_cells = int(missing_count.sum())
    total_cells = int(model_df.shape[0] * model_df.shape[1])

    print(f"Dataset shape: {model_df.shape}")
    print(f"Total cells: {total_cells}")
    print(f"Missing cells: {missing_cells}")
    print(f"Data completeness: {(1 - missing_cells / total_cells) * 100:.2f}%")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    missing_percent_sorted = missing_percent.sort_values(ascending=False)
    colors = ["#be123c" if value > 0 else "#0f766e" for value in missing_percent_sorted.values]
    missing_percent_sorted.plot(
        kind="bar",
        ax=axes[0],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_title("Missing values by column")
    axes[0].set_ylabel("Missing (%)")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", alpha=0.25)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    subset_rows = min(100, len(model_df))
    missing_matrix = model_df.iloc[:subset_rows].isna().astype(int)
    sns.heatmap(
        missing_matrix,
        cmap="RdYlGn_r",
        ax=axes[1],
        xticklabels=model_df.columns,
        yticklabels=False,
        linewidths=0.1,
        linecolor="gray",
        cbar_kws={"label": "Missing = 1"},
    )
    axes[1].set_title(f"Missing-value heatmap, first {subset_rows} compounds")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Compound index")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "missing_value_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_PATH / "missing_value_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "column": model_df.columns,
            "non_null_count": [int(model_df[column].notna().sum()) for column in model_df.columns],
            "completeness_percent": [float(model_df[column].notna().mean() * 100) for column in model_df.columns],
        }
    )
    summary["status"] = summary["completeness_percent"].map(
        lambda value: "complete" if value == 100 else "missing values present"
    )
    summary.to_csv(FIGURES_PATH / "missing_value_summary_table.csv", index=False)
    print("Saved missing-value figures and summary table.")


if __name__ == "__main__":
    main()
