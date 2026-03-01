# tools registry for ai data analysis agent

import src.checks as checks
import src.io_utils as io_utils
import src.modeling as modeling
import src.plotting as plotting
import src.profiling as profiling
import src.summaries as summaries

# src/tools.py

TOOLS = {
    # summaries
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

# Optional (recommended): descriptions kept separate from callables
TOOL_DESCRIPTIONS = {
    "plot_bar_charts": "Bar chart of category counts for categorical columns (NOT associations with numeric variables).",
    "plot_cat_num_boxplot": "Boxplot showing the distribution of a numeric variable grouped by a categorical variable (categorical–numeric association).",
}
