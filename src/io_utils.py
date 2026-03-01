from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dirs(
    reports: Path,
    fig_dir: Optional[Path] = None,
    create_figures: bool = True,
) -> Optional[Path]:
    """Create output folders.

    Backward compatible defaults:
      - Previously, calling ensure_dirs(reports) always created (reports / "figures").
      - Now, that remains the default (create_figures=True).

    For Build2 (or any workflow that separates tool artifacts), call:
      ensure_dirs(reports, create_figures=False)

    Args:
        reports: Base reports directory.
        fig_dir: Optional override for the figures directory to create.
                 If None and create_figures=True, defaults to reports/"figures".
        create_figures: Whether to create a figures directory at all.

    Returns:
        The figures directory Path if created; otherwise None.
    """
    reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)

    if not create_figures:
        return None

    fig_dir = Path(fig_dir) if fig_dir is not None else (reports / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def read_data(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic error handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df
