# tools registry for ai data analysis agent

import pandas as pd
from langfuse.openai import OpenAI  # type: ignore[reportPrivateImportUsage]
import src.checks as checks
import src.io_utils as io_utils
import src.modeling as modeling
import src.plotting as plotting
import src.profiling as profiling
import src.summaries as summaries

# src/tools.py
"""
def calc_ratio():
    # calculate the EV/EBITDA ratio

def rank_ratio():
    # rank the companies with the lowest EV/EBITDA ratio

def write():
    # write a 2 sentence blurb about each company
"""

# start

# column names
TICKER = "(tic) Ticker Symbol"
COMPANY = "(conm) Company Name"
DATE = "(datadate) Data Date"
MKTCAP = "(mkvalt) Market Value - Total - Fiscal"
DEBT_LT = "(dltt) Long-Term Debt - Total"
DEBT_CL = "(dlc) Debt in Current Liabilities - Total"
CASH = "(che) Cash and Short-Term Investments"
EBIT = "(ebit) Earnings Before Interest and Taxes"
REVENUE = "(sale) Sales/Turnover (Net)"
NI = "(ni) Net Income (Loss)"


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent fiscal year row for each company."""
    df = df.copy()
    df[DATE] = pd.to_datetime(df[DATE], dayfirst=True, errors="coerce")
    return df.sort_values(DATE).groupby(TICKER, as_index=False).last()


def _ev_ebit_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with EV and EV/EBIT calculated for each company."""
    latest = _latest(df)

    for col in [MKTCAP, DEBT_LT, DEBT_CL, CASH, EBIT]:
        latest[col] = pd.to_numeric(latest[col], errors="coerce")

    latest["EV"] = (
        latest[MKTCAP]
        + latest[DEBT_LT].fillna(0)
        + latest[DEBT_CL].fillna(0)
        - latest[CASH].fillna(0)
    )

    latest["EV_EBIT"] = latest["EV"] / latest[EBIT]

    return latest[[TICKER, COMPANY, MKTCAP, EBIT, "EV", "EV_EBIT"]].dropna(
        subset=["EV_EBIT"]
    )


def calculate_ev_ebit(df: pd.DataFrame, **kwargs):
    """Calculate EV/EBIT for each company using its most recent fiscal year."""
    result = _ev_ebit_df(df).rename(
        columns={
            TICKER: "Ticker",
            COMPANY: "Company",
            MKTCAP: "Market_Cap",
            EBIT: "EBIT",
        }
    )
    return {"text": result.to_string(index=False), "artifact_paths": []}


def rank_stocks(df: pd.DataFrame, **kwargs):
    """Rank companies from cheapest to most expensive by EV/EBIT (lower = better)."""
    result = _ev_ebit_df(df)
    result = result[result["EV_EBIT"] > 0].sort_values("EV_EBIT").reset_index(drop=True)
    result.index += 1
    result.index.name = "Rank"

    result = result.rename(
        columns={
            TICKER: "Ticker",
            COMPANY: "Company",
            MKTCAP: "Market_Cap",
            EBIT: "EBIT",
        }
    )

    result["EV_EBIT"] = result["EV_EBIT"].map(lambda x: f"{x:.2f}x")

    return {"text": result.to_string(), "artifact_paths": []}


def _fmt_money(value) -> str:
    """Format numeric values safely for prompts."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}M"


def _fmt_multiple(value) -> str:
    """Format EV/EBIT safely for prompts."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}x"


def write_company_blurbs(df: pd.DataFrame, **kwargs):
    """Write a 2-sentence analyst blurb for each company using the OpenAI API."""
    client = OpenAI()
    latest = _latest(df)

    for col in [MKTCAP, DEBT_LT, DEBT_CL, CASH, EBIT, REVENUE, NI]:
        latest[col] = pd.to_numeric(latest[col], errors="coerce")

    latest["EV_EBIT"] = (
        latest[MKTCAP]
        + latest[DEBT_LT].fillna(0)
        + latest[DEBT_CL].fillna(0)
        - latest[CASH].fillna(0)
    ) / latest[EBIT]

    blurbs = []

    for _, row in latest.iterrows():
        prompt = (
            f"Write exactly 2 sentences about {row[COMPANY]} ({row[TICKER]}). "
            f"Revenue: {_fmt_money(row[REVENUE])}, "
            f"EBIT: {_fmt_money(row[EBIT])}, "
            f"Net Income: {_fmt_money(row[NI])}, "
            f"EV/EBIT: {_fmt_multiple(row['EV_EBIT'])}. "
            f"Sentence 1: describe profitability. "
            f"Sentence 2: comment on EV/EBIT valuation "
            f"(15-25x is generally average for large-cap stocks)."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.4,
        )

        content = response.choices[0].message.content or ""
        blurb = content.strip()

        blurbs.append(f"[{row[TICKER]}] {row[COMPANY]}\n{blurb}\n")

    return {"text": "\n".join(blurbs), "artifact_paths": []}


# end

TOOLS = {
    # summaries
    "rank_stocks": rank_stocks,
    "write_company_blurbs": write_company_blurbs,
    "summarize_numeric": summaries.summarize_numeric,
    "summarize_categorical": summaries.summarize_categorical,
    "missingness_table": summaries.missingness_table,
    "pearson_correlation": summaries.pearson_correlation,
    # profiling
    "basic_profile": profiling.basic_profile,
    "split_columns": profiling.split_columns,
    # modeling
    "multiple_linear_regression": modeling.multiple_linear_regression,
    # plotting
    "plot_missingness": plotting.plot_missingness,
    "plot_corr_heatmap": plotting.plot_corr_heatmap,
    "plot_histograms": plotting.plot_histograms,
    "plot_bar_charts": plotting.plot_bar_charts,
    "plot_cat_num_boxplot": plotting.plot_cat_num_boxplot,
    # checks
    "assert_json_safe": checks.assert_json_safe,
    "target_check": checks.target_check,
    # io
    "ensure_dirs": io_utils.ensure_dirs,
    "read_data": io_utils.read_data,
}

TOOL_DESCRIPTIONS = {
    "rank_stocks": "Rank stocks by EV/EBIT multiple. Calculates Enterprise Value (EV = Market Cap + Debt - Cash) divided by EBIT, then ranks companies from lowest to highest EV/EBIT ratio.",
    "write_company_blurbs": "Write a 2-sentence analyst blurb for each company based on financial data including revenue, EBIT, net income, and EV/EBIT valuation.",
    "plot_bar_charts": "Bar chart of category counts for categorical columns.",
    "plot_cat_num_boxplot": "Boxplot of a numeric variable grouped by a categorical variable.",
}