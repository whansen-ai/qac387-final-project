"""
build: Build1 "Suggest" + Build2 "CodeGen" + HITL + (Execute OR Tool-run)
+ Langfuse tracing (LangChain callbacks + observe decorator) + CLI tags

This script demonstrates a simple CLI for human-in-the-loop data analysis with LLMs,
using subprocess to run generated code and a tool-run mode that routes to your existing Build0
functions.The script is instrumented with Langfuse for tracings.

In order to complete the assignment, you will need to:
1) Download the build1_llm_assistant_assignment_2.py and the requirements.txt file, 2) Download the tools.py file,
from the src folder and place it in your project's src folder. 3) Run the requirements.txt file to
install the necessary libraries into your local project virtual environment.

4) Complete the assignment by filling in the blanks (marked with TODO) in the
build1_llm_assistant_assignment_2.py file. The main areas you need to complete are:

To run the script:
  python builds/build2_hitl_agent.py --data data/penguins.csv --report_dir reports --tags build2
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import inspect
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# always set the location of the .env file to the project root for consistent env var loading
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root (parent of /builds)
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------------------
# Optional Langfuse instrumentation
# --------------------------------------------------------------------------------------
LANGFUSE_AVAILABLE = False
try:
    from langfuse import observe, propagate_attributes  # type: ignore
    from langfuse.langchain import CallbackHandler  # type: ignore

    LANGFUSE_AVAILABLE = True
except Exception:
    LANGFUSE_AVAILABLE = False

    def observe(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap

    class propagate_attributes:  # type: ignore
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

# --------------------------------------------------------------------------------------
# helper function to make sure metadata is JSON-serializable for Langfuse
#  (e.g., for unknown tool args)
# --------------------------------------------------------------------------------------


def make_metadata_safe(metadata: dict) -> dict:
    safe = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        else:
            # dicts, lists, Paths, numpy types, etc.
            safe[k] = json.dumps(v, default=str)
    return safe


# --------------------------------------------------------------------------------------
# Import Build0 utilities (adjust the import path as needed based on your project structure)
# --------------------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import ensure_dirs, read_data, basic_profile  # noqa: E402


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
# This function takes a dataset profile dictionary (with keys
# like 'n_rows', 'n_cols', 'columns', and 'dtypes') and converts it into a string
# format that can be included in a prompt for the LLM. It lists the number of
# rows and columns, and then enumerates each column along with its data type.
def profile_to_schema_text(profile: dict) -> str:
    lines = [
        f"Rows: {profile.get('n_rows')}",
        f"Columns: {profile.get('n_cols')}",
        "",
        "Columns and dtypes:",
    ]
    for col in profile["columns"]:
        lines.append(f"- {col}: {profile['dtypes'].get(col)}")
    return "\n".join(lines)


# Regular expressions to extract code and JSON blocks from text
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(text or "")
    return m.group(1).strip() if m else None


def split_sections(text: str) -> Tuple[str, str, str]:
    if not text:
        return "", "", ""
    up = text.upper()
    i_plan = up.find("PLAN:")
    i_code = up.find("CODE:")
    i_ver = up.find("VERIFY:")
    if i_plan == -1 or i_code == -1 or i_ver == -1:
        return text.strip(), "", ""
    return text[i_plan:i_code].strip(), text[i_code:i_ver].strip(), text[i_ver:].strip()


def invoke_chain_text(
    chain,
    inputs: Dict[str, Any],
    config: Dict[str, Any],
    stream: bool,
    print_output: bool = True,
) -> str:
    if stream:
        chunks = []
        for chunk in chain.stream(inputs, config=config):
            if print_output:
                print(chunk, end="", flush=True)
            chunks.append(chunk)
        if print_output:
            print("\n")
        return "".join(chunks)

    out = chain.invoke(inputs, config=config)
    if print_output:
        print("\n" + out + "\n")
    return out


# This function tries to parse the tool plan output as JSON directly, and if that fails,
# it looks for a JSON block within the text. It returns a dictionary of the parsed JSON
# or an empty dictionary if parsing fails.
def parse_tool_plan(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_RE.search(raw)
    if match:
        try:
            obj = json.loads(match.group(1))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


# This function recursively walks through the tool arguments to find any string
# values that are expected to be column names (based on key hints) and checks if
# they are in the known columns set. It collects any unknown column names it finds
# and returns them as a set.
def find_unknown_columns(args_obj: Any, known_columns: set[str]) -> set[str]:
    expected_column_keys = {
        "column",
        "columns",
        "col",
        "cols",
        "x",
        "y",
        "outcome",
        "predictor",
        "predictors",
        "feature",
        "features",
        "target",
        "groupby",
    }
    unknown: set[str] = set()

    def walk(obj: Any, key_hint: Optional[str] = None) -> None:
        key_l = (key_hint or "").lower()
        expects_column = (
            key_l in expected_column_keys
            or key_l.endswith("_col")
            or key_l.endswith("_cols")
        )

        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, str(k))
            return

        if isinstance(obj, list):
            for item in obj:
                walk(item, key_hint)
            return

        if isinstance(obj, str) and expects_column and obj not in known_columns:
            unknown.add(obj)

    walk(args_obj)
    return unknown


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# This function takes a comma-separated string of tags, splits it into individual tags,
# and returns a list of cleaned tags. It handles cases where the input might be empty or
# contain extra whitespace.
def parse_tags(tags_csv: str) -> list[str]:
    return [t.strip() for t in (tags_csv or "").split(",") if t.strip()]


# This function creates a configuration dictionary for Langfuse tracing.
# It includes the session ID and tags in the metadata, and if Langfuse is available,
# it also adds a CallbackHandler to the callbacks list. This configuration can be
# passed to chain invocations to ensure that the traces are properly tagged and
# associated with the correct session in Langfuse.
def make_langfuse_config(session_id: str, tags: list[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"configurable": {"session_id": session_id}}

    if LANGFUSE_AVAILABLE:
        # v3: handler takes no session/tags kwargs; attach them via metadata
        cfg["callbacks"] = [CallbackHandler()]  # type: ignore

        cfg["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_tags": tags,
        }

    return cfg


# --------------------------------------------------------------------------------------
# Build0 tool registry loader
# --------------------------------------------------------------------------------------
ToolFn = Callable[..., Any]


def load_tools() -> Dict[str, ToolFn]:
    """
    Load TOOLS registry from your Build0 codebase.

    Search order (edit to match your repo if you like):
      1) src.tools: TOOLS
      2) src.build0_tools: TOOLS
      3) src: TOOLS  (exported in src/__init__.py)

    Returns:
      dict[str, callable]
    Raises:
      RuntimeError with clear instructions if not found.
    """
    candidates = [
        ("src.tools", "TOOLS"),
        ("src.build0_tools", "TOOLS"),
        ("src", "TOOLS"),
    ]

    for module_name, attr in candidates:
        try:
            mod = importlib.import_module(module_name)
            tools = getattr(mod, attr)
            if isinstance(tools, dict) and tools:
                return tools
        except Exception:
            continue

    raise RuntimeError(
        "Could not import a TOOLS registry.\n\n"
        "Create src/tools.py with something like:\n\n"
        "  from src.some_build0_module import describe_numeric, freq_table, simple_ols\n"
        "  TOOLS = {\n"
        "      'describe_numeric': describe_numeric,\n"
        "      'freq_table': freq_table,\n"
        "      'simple_ols': simple_ols,\n"
        "  }\n\n"
        "Then rerun this script."
    )


@dataclass
class ToolResult:
    name: str
    artifact_paths: list[str]
    text: str


def normalize_tool_return(tool_name: str, result: Any) -> ToolResult:
    """
    Normalize tool returns into ToolResult.

    Accepted return types:
      - str
      - dict with 'text' and optional 'artifact_paths'
      - tuple(text, artifact_paths)
    """
    if isinstance(result, ToolResult):
        return result

    if isinstance(result, str):
        return ToolResult(name=tool_name, artifact_paths=[], text=result)

    if isinstance(result, dict):
        text = str(result.get("text", ""))
        artifact_paths = result.get("artifact_paths", []) or []
        if not isinstance(artifact_paths, list):
            artifact_paths = [str(artifact_paths)]
        return ToolResult(
            name=tool_name, artifact_paths=[str(p) for p in artifact_paths], text=text
        )

    if isinstance(result, tuple) and len(result) == 2:
        text, artifacts = result
        if artifacts is None:
            artifacts = []
        if not isinstance(artifacts, list):
            artifacts = [artifacts]
        return ToolResult(
            name=tool_name, artifact_paths=[str(p) for p in artifacts], text=str(text)
        )

    # fallback
    return ToolResult(name=tool_name, artifact_paths=[], text=str(result))


# --------------------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------------------
def build_suggest_chain(
    model: str, temperature: float = 0.2, stream: bool = False, memory: bool = False
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    system_text = (
        "You are a data analysis assistant.\n"
        "You ONLY see the dataset schema (columns + dtypes). Do NOT invent columns.\n\n"
        "Return:\n"
        "1) 2-3 plausible research questions (bulleted)\n"
        "2) For each: outcome(s), predictor(s), and suggested analysis type\n"
        "3) 5-7 clarifying questions\n"
    )

    if memory:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                ("human", "Dataset schema:\n{schema_text}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "User question:\n{user_query}"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                (
                    "human",
                    "Dataset schema:\n{schema_text}\n\nUser question:\n{user_query}\n",
                ),
            ]
        )

    base_chain = prompt | llm | StrOutputParser()
    if not memory:
        return base_chain

    history = InMemoryChatMessageHistory()
    return RunnableWithMessageHistory(
        base_chain,
        lambda _session_id: history,
        input_messages_key="user_query",
        history_messages_key="history",
    )


def build_codegen_chain(
    model: str, temperature: float = 0.2, stream: bool = False, memory: bool = False
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    system_text = (
        "You are a careful Python data analysis code generator.\n"
        "IMPORTANT RULES:\n"
        "- You ONLY know the dataset schema. Do NOT invent columns.\n"
        "- Produce ONE Python script that can run as a standalone file.\n"
        "- The script MUST:\n"
        "  (1) use argparse with --data and --report_dir\n"
        "  (2) read the CSV at --data with pandas\n"
        "  (3) handle missing values explicitly\n"
        "  (4) save at least ONE artifact into --report_dir\n"
        "  (5) validate referenced columns exist (exit nonzero if not)\n\n"
        "OUTPUT FORMAT (exactly):\n"
        "PLAN:\n"
        "- ...\n\n"
        "CODE:\n"
        "```python\n"
        "# full script\n"
        "```\n\n"
        "VERIFY:\n"
        "- ...\n"
    )

    if memory:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                ("human", "Dataset schema:\n{schema_text}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "User request:\n{user_request}"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                (
                    "human",
                    "Dataset schema:\n{schema_text}\n\nUser request:\n{user_request}\n",
                ),
            ]
        )

    base_chain = prompt | llm | StrOutputParser()
    if not memory:
        return base_chain

    history = InMemoryChatMessageHistory()
    return RunnableWithMessageHistory(
        base_chain,
        lambda _session_id: history,
        input_messages_key="user_request",
        history_messages_key="history",
    )


def build_toolplan_chain(
    model: str, allowed_tools: list[str], temperature: float = 0.0, stream: bool = False
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    allow_str = "\n".join([f"- {t}" for t in allowed_tools])

    system_text = (
        "You are a routing assistant.\n"
        "You MUST choose exactly one tool from the allow-list and output ONLY valid JSON.\n\n"
        f"Allow-list tools:\n{allow_str}\n\n"
        "JSON output schema:\n"
        "{{\n"
        '  "tool": "<one of the allow-list tool names>",\n'
        '  "args": {{ ... }},\n'
        '  "note": "one sentence explaining why this tool fits"\n'
        "}}\n\n"
        "Rules:\n"
        "- Use ONLY columns in the schema.\n"
        "- Argument names MUST match the tool function signature exactly.\n"
        '- For categorical–numeric association tools, use keys: "categorical_column" and "numerical_column".\n'
        '- For categorical-only count bar charts, use key "x" (or "column") and do NOT include a numeric y.\n'
        "Tool Selection Rules:\n\n"
        '1. If the user asks for the association between two numeric variables, Pearson correlation, correlation coefficient, r, r², p-value, or confidence interval between two numeric variables → use "pearson_correlation" with arguments:\n'
        '   {{ "x": "<numeric_column_1>", "y": "<numeric_column_2>" }}\n\n'
        '2. If the user asks for a full correlation matrix across multiple numeric variables → use "correlations".\n\n'
        "3. If the user asks for a visualization of association between two numeric variables → use a numeric–numeric plotting tool (e.g., scatterplot).\n\n"
        '4. If one variable is categorical and one is numeric → use "plot_cat_num_boxplot".\n'
        "5. If both variables are categorical → use categorical frequency or bar chart tools.\n\n"
        "Never invent grouping variables that were not requested.\n"
        "- If the request is unclear, pick the closest tool and set args conservatively."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            (
                "human",
                "Dataset schema:\n{schema_text}\n\nUser request:\n{user_request}\n",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_results_summarizer_chain(
    model: str, temperature: float = 0.2, stream: bool = False
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    system_text = (
        "You are an expert at explaining data analysis results.\n"
        "Given a user request and tool outputs, do:\n"
        "1) What we ran (1-2 sentences)\n"
        "2) Key results (bullets)\n"
        "3) Interpretation (plain language)\n"
        "4) Caveats/assumptions (bullets)\n"
        "5) Next steps (2-3 suggestions)\n"
        "Do NOT invent results; use only what is provided.\n"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", "User request:\n{user_request}\n\nTool output:\n{tool_output}\n"),
        ]
    )
    return prompt | llm | StrOutputParser()


# --------------------------------------------------------------------------------------
# Execution (subprocess, not exec)
# --------------------------------------------------------------------------------------
@observe(name="execute-generated-script", as_type="span", capture_output=False)
def run_generated_script(
    script_path: Path, data_path: Path, report_dir: Path, timeout_s: int = 60
) -> subprocess.CompletedProcess:
    with propagate_attributes(tags=["build", "execute"]):
        cmd = [
            sys.executable,
            str(script_path),
            "--data",
            str(data_path),
            "--report_dir",
            str(report_dir),
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


HELP_TEXT = """Commands:
  help                         Show this help text
  schema                       Print dataset schema
  suggest <question>           Build1-style suggestions (LLM)
  code <request>               Generate Python code (LLM) + approve to save
  run                          Execute last approved script via subprocess
  tool <request>               Tool-run: LLM chooses one tool + args (JSON), run tool, summarize example: tool run a frequency table for sex
  exit                         Quit
"""


@observe(name="build-suggest", as_type="span")
def traced_suggest(
    suggest_chain,
    schema_text: str,
    question: str,
    config: Dict[str, Any],
    stream: bool,
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "suggest"]):
        return invoke_chain_text(
            suggest_chain,
            {"schema_text": schema_text, "user_query": question},
            config=config,
            stream=stream,
        )


@observe(name="build-codegen", as_type="generation")
def traced_codegen(
    codegen_chain,
    schema_text: str,
    request: str,
    config: Dict[str, Any],
    stream: bool,
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "codegen"]):
        return invoke_chain_text(
            codegen_chain,
            {"schema_text": schema_text, "user_request": request},
            config=config,
            stream=stream,
        )


@observe(name="build-toolplan", as_type="generation")
def traced_toolplan(
    toolplan_chain,
    schema_text: str,
    request: str,
    config: Dict[str, Any],
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "toolplan"]):
        return toolplan_chain.invoke(
            {"schema_text": schema_text, "user_request": request}, config=config
        )


@observe(name="build-summarize", as_type="generation")
def traced_summarize(
    summarize_chain,
    request: str,
    tool_output: str,
    config: Dict[str, Any],
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "summarize"]):
        return summarize_chain.invoke(
            {"user_request": request, "tool_output": tool_output}, config=config
        )


@observe(name="build-run-tool", as_type="span", capture_output=False)
def traced_run_tool(
    tool_name: str,
    tool_fn: ToolFn,
    df: pd.DataFrame,
    tool_dir: Path,
    tool_args: Dict[str, Any],
    tags: list[str],
) -> ToolResult:
    with propagate_attributes(
        tags=tags + ["build", "toolrun"],
        metadata=make_metadata_safe({"tool": tool_name, "args": tool_args}),
    ):
        # Inject fig_dir default if tool supports it
        sig = inspect.signature(tool_fn)
        tool_args2 = dict(tool_args)

        if "fig_dir" in sig.parameters and "fig_dir" not in tool_args2:
            tool_args2["fig_dir"] = tool_dir

        try:
            result = tool_fn(df, report_dir=tool_dir, **tool_args2)
        except TypeError:
            result = tool_fn(df, **tool_args2)

        return normalize_tool_return(tool_name, result)


def main() -> None:

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="build: Suggest + CodeGen + HITL + Execute/Tool-run + Langfuse"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--report_dir", type=str, default="reports")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--out_file", type=str, default="generated_analysis.py")
    parser.add_argument("--timeout_s", type=int, default=60)
    parser.add_argument("--session_id", type=str, default="cli-session")
    parser.add_argument(
        "--tags", type=str, default="build", help="Comma-separated Langfuse tags"
    )
    args = parser.parse_args()

    tag_list = parse_tags(args.tags)

    data_path = Path(args.data)

    # ensure_dirs is for report structure; mkdir is for specific folders
    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir, create_figures=False)

    base_tool_dir = report_dir / "tool_figures"
    base_tool_dir.mkdir(parents=True, exist_ok=True)

    session_tool_dir = base_tool_dir / f"session_{args.session_id}"
    session_tool_dir.mkdir(parents=True, exist_ok=True)

    base_tool_out_dir = report_dir / "tool_outputs"
    base_tool_out_dir.mkdir(parents=True, exist_ok=True)
    session_tool_out_dir = base_tool_out_dir / f"session_{args.session_id}"
    session_tool_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tool artifacts will be saved to: {session_tool_dir}\n")

    # Load data + schema
    df = read_data(data_path)
    schema_text = profile_to_schema_text(basic_profile(df))

    # Load Build0 tools registry
    tools = load_tools()
    allowed_tools = sorted(tools.keys())

    suggest_chain = build_suggest_chain(
        args.model, args.temperature, args.stream, args.memory
    )
    codegen_chain = build_codegen_chain(
        args.model, args.temperature, args.stream, args.memory
    )
    toolplan_chain = build_toolplan_chain(
        args.model, allowed_tools=allowed_tools, temperature=0.0, stream=args.stream
    )
    summarize_chain = build_results_summarizer_chain(
        args.model, args.temperature, args.stream
    )

    base_config = make_langfuse_config(session_id=args.session_id, tags=tag_list)

    script_path = report_dir / args.out_file
    run_log_path = report_dir / "run_log.txt"

    print("\n=== build2: Suggest + CodeGen + HITL + Execute/Tool-run ===\n")
    print(f"Tags: {tag_list}")
    print(f"Build0 tools loaded: {', '.join(allowed_tools)}\n")

    if LANGFUSE_AVAILABLE:
        print("Langfuse: ENABLED (CallbackHandler + observe decorator)\n")
    else:
        print("Langfuse: not installed or not available (running without tracing)\n")

    print("Type 'help' for commands. Type 'exit' to quit.\n")

    state: Dict[str, Any] = {"code_approved": None}
    # Creates continuous command-line interface (CLI) loop that:
    # Waits for user input, processes it, and responds to special commands (exit, quit, help)
    # Loop repeats until the user exits
    while True:
        user_in = input("> ").strip()
        if not user_in:
            continue
        low = user_in.lower()

        if low in {"exit", "quit"}:
            print("Goodbye!")
            break

        if low == "help":
            print("\n" + HELP_TEXT + "\n")
            continue

        if low == "schema":
            print("\n=== DATASET SCHEMA ===")
            print(schema_text + "\n")
            continue

        if low.startswith("suggest "):
            q = user_in[len("suggest ") :].strip()
            _ = traced_suggest(
                suggest_chain, schema_text, q, base_config, args.stream, tag_list
            )
            continue

        if low.startswith("code "):
            req = user_in[len("code ") :].strip()
            out = traced_codegen(
                codegen_chain, schema_text, req, base_config, args.stream, tag_list
            )
            candidate = extract_python_code(out)

            if not candidate:
                print(
                    "WARNING: No fenced ```python code block found. Ask again and require it.\n"
                )
                continue

            _, _, verify = split_sections(out)
            print("=== HUMAN VERIFICATION CHECKLIST (from model) ===")
            print((verify + "\n") if verify else "(No VERIFY section found.)\n")

            approve = input("Approve and save this code? (y/n) ").strip().lower()
            if approve != "y":
                print("\nCode not approved.\n")
                continue

            state["code_approved"] = candidate
            save_text(script_path, candidate)
            print(f"\nApproved and saved to: {script_path}\n")
            print(
                "Next: type 'run' to execute, or 'tool <request>' to run a Build0 tool.\n"
            )
            continue

        if low == "run":
            if not state.get("code_approved") or not script_path.exists():
                print("\nNo approved script found yet. Use: code <request>\n")
                continue

            confirm = input(f"Execute {script_path.name} now? (y/n) ").strip().lower()
            if confirm != "y":
                print("\nExecution not approved.\n")
                continue

            print("\nRunning generated script...\n")
            try:
                result = run_generated_script(
                    script_path, data_path, report_dir, timeout_s=args.timeout_s
                )
            except subprocess.TimeoutExpired:
                msg = f"ERROR: Script timed out after {args.timeout_s} seconds.\n"
                save_text(run_log_path, msg)
                print(msg)
                continue

            log = []
            log.append("=== COMMAND ===\n")
            log.append(
                f"{sys.executable} {script_path} --data {data_path} --report_dir {report_dir}\n\n"
            )
            log.append("=== STDOUT ===\n")
            log.append(result.stdout or "(empty)\n")
            log.append("\n=== STDERR ===\n")
            log.append(result.stderr or "(empty)\n")
            log.append(f"\n=== RETURN CODE ===\n{result.returncode}\n")
            save_text(run_log_path, "".join(log))

            print(f"Finished. Return code: {result.returncode}")
            print(f"Saved execution log to: {run_log_path}\n")
            continue

        # --- TOOL MODE ---
        # Accept either:
        #   tool
        #   tool <request>
        if low == "tool":
            req = input(
                "Tool request (e.g., 'run a frequency table for sex'): "
            ).strip()
            if not req:
                print("\nUsage: tool <analysis request>\n")
                continue

        elif low.startswith("tool "):
            req = user_in[len("tool ") :].strip()
            if not req:
                print("\nUsage: tool <analysis request>\n")
                continue

        else:
            req = None

        if req is not None:
            toolplan_raw = traced_toolplan(
                toolplan_chain, schema_text, req, base_config, tag_list
            )

            plan = parse_tool_plan(toolplan_raw)
            if not plan:
                print("\nERROR: Tool planner did not return valid JSON. Try again.\n")
                print("Raw output was:\n", toolplan_raw, "\n")
                continue

            tool_name = plan.get("tool")
            tool_args = plan.get("args", {}) or {}
            note = plan.get("note", "")

            print("\n=== TOOL PLAN ===")
            print(json.dumps(plan, indent=2))
            if note:
                print(f"\nNote: {note}")
            print()

            if tool_name not in tools:
                print(
                    f"\nERROR: Proposed tool '{tool_name}' is not in TOOLS registry.\n"
                )
                print(f"Available tools: {', '.join(allowed_tools)}\n")
                continue

            unknown_cols = find_unknown_columns(tool_args, set(df.columns))
            if unknown_cols:
                print("\nERROR: Tool args reference unknown columns.\n")
                print("Unknown columns:", ", ".join(sorted(unknown_cols)), "\n")
                continue

            confirm = input(f"Run tool '{tool_name}' now? (y/n) ").strip().lower()
            if confirm != "y":
                print("\nTool execution not approved.\n")
                continue

            try:
                res = traced_run_tool(
                    tool_name,
                    tools[tool_name],
                    df,
                    session_tool_dir,
                    tool_args,
                    tag_list,
                )

            except Exception as e:
                print(f"\nERROR running tool: {e}\n")
                continue

            # Combine tool text + artifact paths so the summary step can find the files
            artifact_lines = "\n".join(f"- {p}" for p in (res.artifact_paths or []))
            combined = (res.text or "").strip()
            if artifact_lines:
                combined += "\n\n=== ARTIFACT PATHS ===\n" + artifact_lines
            if not combined and artifact_lines:
                combined = f"Tool '{tool_name}' produced artifacts:\n\n=== ARTIFACT PATHS ===\n{artifact_lines}"

            out_txt = session_tool_out_dir / f"{tool_name}_output.txt"
            save_text(out_txt, combined)
            print(f"\nSaved tool output to: {out_txt}\n")

            summary = traced_summarize(
                summarize_chain, req, combined, base_config, tag_list
            )
            print("\n=== INTERPRETATION & SUMMARY ===\n")
            print(summary + "\n")
            continue


if __name__ == "__main__":
    main()
