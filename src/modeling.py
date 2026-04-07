from __future__ import annotations

from typing import Optional, List, Dict, Any
import pandas as pd
import statsmodels.formula.api as smf


def multiple_linear_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: Optional[List[str]] = None,) -> Dict[str, Any]:
    """
    Fit multiple linear regression using statsmodels OLS (formula interface)
    for numeric outcome and numeric and categorical predictors.

    - outcome must exist
    - predictors optional:
        if None -> all columns except outcome (you may want to restrict in your course)
    - drops rows with missing values in outcome/predictors
    - returns a JSON-safe dictionary of key results
    """
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found in dataframe.")

    if predictors is None:
        raise ValueError("You must specify the predictors for regression.")

    missing_preds = [p for p in predictors if p not in df.columns]
    if missing_preds:
        raise ValueError(f"Predictor(s) not found: {missing_preds}")

    # Build formula: numeric predictors as-is, non-numeric wrapped as C(...)
    terms: List[str] = []
    for p in predictors:
        if pd.api.types.is_numeric_dtype(df[p]):
            terms.append(p)
        else:
            terms.append(f"C({p})")

    if not terms:
        raise ValueError("No predictors provided after processing.")

    formula = f"{outcome} ~ " + " + ".join(terms)

    model_df = df[[outcome] + predictors].dropna()
    if model_df.shape[0] < 3:
        raise ValueError("Not enough complete rows to fit regression (need >= 3).")

    fitted = smf.ols(formula=formula, data=model_df).fit()

    out: Dict[str, Any] = {
        "outcome": str(outcome),
        "predictors": [str(p) for p in predictors],
        "n_rows_used": int(model_df.shape[0]),
        "formula": str(formula),
        "r_squared": float(fitted.rsquared),
        "adj_r_squared": float(fitted.rsquared_adj),
        "params": {str(k): float(v) for k, v in fitted.params.items()},
        "pvalues": {str(k): float(v) for k, v in fitted.pvalues.items()},
        "stderr": {str(k): float(v) for k, v in fitted.bse.items()},
    }
    return out


def rank_stocks_by_ev_ebit(
    df: pd.DataFrame,
    top_n: int = 20,
    latest_only: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Rank stocks by EV/EBIT using WRDS-style column names.

    EV = market value + long-term debt + current debt - cash
    EV/EBIT = enterprise value / EBIT

    Lower EV/EBIT indicates a cheaper stock.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with WRDS financial columns.
    top_n : int, default 20
        Number of cheapest stocks to return.
    latest_only : bool, default True
        If True, keep only the most recent observation per ticker.

    Returns
    -------
    pd.DataFrame
        Ranked dataframe of cheapest stocks by EV/EBIT.
    """

    required_columns = [
        "(tic) Ticker Symbol",
        "(datadate) Data Date",
        "(mkvalt) Market Value - Total - Fiscal",
        "(dltt) Long-Term Debt - Total",
        "(dlc) Debt in Current Liabilities - Total",
        "(che) Cash and Short-Term Investments",
        "(ebit) Earnings Before Interest and Taxes",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for EV/EBIT ranking: {missing_columns}"
        )

    working_df = df.copy()

    # Rename columns to simpler internal names
    working_df = working_df.rename(
        columns={
            "(tic) Ticker Symbol": "ticker",
            "(datadate) Data Date": "date",
            "(mkvalt) Market Value - Total - Fiscal": "market_value",
            "(dltt) Long-Term Debt - Total": "long_term_debt",
            "(dlc) Debt in Current Liabilities - Total": "current_debt",
            "(che) Cash and Short-Term Investments": "cash",
            "(ebit) Earnings Before Interest and Taxes": "ebit",
        }
    )

    # Convert date
    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")

    # Drop rows missing key fields
    working_df = working_df.dropna(
        subset=[
            "ticker",
            "date",
            "market_value",
            "long_term_debt",
            "current_debt",
            "cash",
            "ebit",
        ]
    )

    # Keep only latest observation per ticker
    if latest_only:
        working_df = working_df.sort_values(["ticker", "date"])
        working_df = working_df.groupby("ticker", as_index=False).tail(1)

    # Compute enterprise value
    working_df["enterprise_value"] = (
        working_df["market_value"]
        + working_df["long_term_debt"]
        + working_df["current_debt"]
        - working_df["cash"]
    )

    # Keep only valid rows for EV/EBIT
    working_df = working_df[
        (working_df["enterprise_value"] > 0) &
        (working_df["ebit"] > 0)
    ]

    if working_df.empty:
        raise ValueError(
            "No valid rows remain after filtering for positive enterprise value and EBIT."
        )

    # Compute EV/EBIT
    working_df["ev_ebit"] = working_df["enterprise_value"] / working_df["ebit"]

    ranked_df = (
        working_df[
            [
                "ticker",
                "date",
                "market_value",
                "long_term_debt",
                "current_debt",
                "cash",
                "enterprise_value",
                "ebit",
                "ev_ebit",
            ]
        ]
        .sort_values(by="ev_ebit", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    ranked_df.index = ranked_df.index + 1
    ranked_df.index.name = "rank"

    return ranked_df
