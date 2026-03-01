from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


def plot_missingness(miss_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Plot missing data in a horizontal bar chart."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = miss_df.head(top_n).iloc[::-1]
    plt.figure()
    plt.barh(plot_df["column"], plot_df["missing_rate"])
    plt.xlabel("Missing rate")
    plt.title(f"Top {min(top_n, len(miss_df))} columns by missingness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_corr_heatmap(
    corr: pd.DataFrame,
    out_path: Path,
    missing: str = "drop",
) -> None:
    """Create a heatmap of correlations for numeric columns."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if corr.empty:
        return

    if corr.shape[0] > 20:
        print("Too many variables to annotate heatmap clearly.")

        plt.figure()
        im = plt.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
        plt.title("Correlation heatmap (numeric columns)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    plt.figure()
    im = plt.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)

    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_histograms(
    df: pd.DataFrame,
    numeric_cols: List[str],
    fig_dir: Union[str, Path],
    max_cols: int = 12,
) -> Dict[str, Any]:
    """Save histograms for up to max_cols numeric columns.

    Backward compatible:
      - existing callers pass fig_dir
    Agent-friendly:
      - returns artifact_paths + text
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for c in numeric_cols[:max_cols]:
        if c not in df.columns:
            continue
        plt.figure()
        df[c].dropna().hist(bins=30)
        plt.title(f"Histogram: {c}")
        plt.tight_layout()
        out_path = fig_dir / f"hist_{c}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        written.append(str(out_path))

    return {
        "text": f"Saved {len(written)} histogram(s) to {fig_dir}",
        "artifact_paths": written,
    }


def plot_bar_charts(
    df: pd.DataFrame,
    # Router/LLM-friendly args
    x: Optional[str] = None,  # categorical column (preferred)
    y: Optional[str] = None,  # ignored for bar charts (kept to avoid tool-call crashes)
    # Back-compat args
    cat_cols: Optional[List[str]] = None,
    column: Optional[str] = None,
    # Output control
    fig_dir: Optional[Union[str, Path]] = None,
    max_cols: int = 12,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Save bar charts of category COUNTS for categorical columns (top_k categories).

    Accepts any ONE of:
      - x="species"            (router style)
      - column="species"       (older style)
      - cat_cols=["species","island"]

    'y' is accepted for compatibility with (x, y) tool suggestions, but is not used.
    Returns artifact_paths + text (agent-friendly); safe for callers that ignore returns.
    """
    if fig_dir is None:
        fig_dir = Path("figures")
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    provided = [arg is not None for arg in (cat_cols, column, x)]
    if sum(provided) > 1:
        raise ValueError("Provide only one of: 'cat_cols', 'column', or 'x'.")

    if cat_cols is None:
        if column is not None:
            cat_cols = [column]
        elif x is not None:
            cat_cols = [x]
        else:
            raise ValueError("Provide one of: 'cat_cols', 'column', or 'x'.")

    written: List[str] = []
    for c in cat_cols[:max_cols]:
        if c not in df.columns:
            raise ValueError(f"Column not found: '{c}'")

        plt.figure()
        counts = df[c].astype("string").value_counts(dropna=True).head(top_k)
        counts.plot(kind="bar")
        plt.title(f"Top {min(top_k, len(counts))} values: {c}")
        plt.tight_layout()
        out_path = fig_dir / f"bar_{c}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        written.append(str(out_path))

    return {
        "text": f"Saved {len(written)} bar chart(s) (counts) to {fig_dir}",
        "artifact_paths": written,
    }


def plot_cat_num_boxplot(
    df: pd.DataFrame,
    categorical_column: str,
    numerical_column: str,
    # Backward compatible output controls:
    out_path: Optional[Union[str, Path]] = None,
    fig_dir: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    missing: str = "drop",
) -> Dict[str, Any]:
    """Boxplot of a numeric column grouped by a categorical column.

    Backward compatible:
      - older callers may pass out_path
      - Build2 runner tends to inject fig_dir
      - some routers may prefer out_dir

    Missing-data policy:
      - drop (default): drop rows missing either variable
      - raise: error if any missing
    """
    cat_col = categorical_column
    num_col = numerical_column

    if cat_col not in df.columns:
        raise ValueError(f"Column not found: {cat_col}")
    if num_col not in df.columns:
        raise ValueError(f"Column not found: {num_col}")

    # Resolve output directory / path
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = fig_dir or out_dir or Path("figures")
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"boxplot_{cat_col}_vs_{num_col}.png"

    sub = df[[cat_col, num_col]].copy()
    n_before = len(sub)

    if missing == "drop":
        sub = sub.dropna(subset=[cat_col, num_col])
    elif missing == "raise":
        if sub[[cat_col, num_col]].isna().any().any():
            raise ValueError(f"Missing values found in '{cat_col}' and/or '{num_col}'.")
    else:
        raise ValueError(f"Unknown missing policy: {missing}")

    n_after = len(sub)
    dropped = n_before - n_after

    if sub.empty:
        raise ValueError("No non-missing rows remain for requested columns.")

    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna(subset=[num_col])
    if sub.empty:
        raise ValueError("Numeric column could not be coerced to numeric.")

    groups: List[Any] = []
    labels: List[str] = []
    for g, grp in sub.groupby(cat_col):
        vals = grp[num_col].values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(g))

    if not groups:
        raise ValueError("No valid groups to plot after filtering.")

    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.title(f"{num_col} by {cat_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {
        "text": f"Saved boxplot to {out_path}. Dropped {dropped} row(s) due to missing values (policy='{missing}').",
        "artifact_paths": [str(out_path)],
    }
