"""EDA: histograms, correlation heatmap, class balance, boxplots."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data import load_clean  # noqa: E402

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="deep")


def run_eda(df=None, out_dir=config.REPORTS_DIR / "eda"):
    df = load_clean() if df is None else df
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=config.TARGET, data=df, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No disease (0)", "Disease (1)"])
    ax.set_title("Class balance")
    for c in ax.containers:
        ax.bar_label(c)
    paths.append(_save(fig, out_dir / "class_balance.png"))

    # Histograms of numeric features
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC):
        sns.histplot(data=df, x=col, hue=config.TARGET, kde=True, ax=ax, bins=20)
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "histograms.png"))

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    paths.append(_save(fig, out_dir / "correlation_heatmap.png"))

    # Categorical vs target
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), config.CATEGORICAL):
        sns.countplot(x=col, hue=config.TARGET, data=df, ax=ax)
        ax.set_title(f"{col} vs target")
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "categorical_vs_target.png"))

    # Missing values summary
    missing = df.isna().sum()
    missing[missing > 0].to_csv(out_dir / "missing_values.csv")
    df.describe().T.to_csv(out_dir / "summary_stats.csv")
    log.info("EDA artifacts written to %s", out_dir)
    return paths


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_eda()
