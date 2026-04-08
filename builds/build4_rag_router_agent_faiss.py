"""
build4: HITL + Router + RAG (FAISS-loaded index + Optional CodeGen/Execute)
+ Langfuse tracing (LangChain callbacks + observe decorator)

THis build adds a top-level ROUTER that decides whether to:
    (A) run one of the Build0 tools, OR
    (B) fall back to CodeGen + optional Execute (subprocess).

It adds a single main command:
    ask <request>   (router decides tool vs codegen)

Keeps power-user commands:
    tool <request>  (force tool mode)
    code <request>  (force codegen mode)
    run             (execute last approved code)

You will need the expected Build0 tool registry (tools.py in the updated src folder)

Each tool function should accept (df, report_dir, **kwargs) and return either:
- a string, OR
- a dict with a "text" field (+ optional "artifact_paths"), OR
- a tuple (text, artifact_paths)

To run this script, you will need to make sure you have the most updated src and requirements.txt file
from the course repository.

# You will also need to have built the RAG index by running the build_rag_index.py script before running the agent,

To run the script:
python builds/build4_rag_router_agent_faiss.py --data data/penguins.csv --report_dir reports --knowledge_dir knowledge --session_id cli-session --memory

To stream LLM output, add the --stream flag to the command above

To interact with the agent, use the following commands:
    help                         Show this help text
    schema                       Print dataset schema
    suggest <question>           Questions about the dataset or analysis (LLM)
    ask <request>                ROUTER decides: tool-run OR codegen (HITL)
    tool <request>               Force tool-run: choose one Build0 tool + args (HITL)
    code <request>               Force code generation (HITL) + approve to save
    run                          Execute last approved script via subprocess (HITL)
    exit                         Quit
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List
from textwrap import dedent

import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# import Build0 tools and RAG helpers
# find path to src for imports, allows the script to be run from project root or builds folder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import ensure_dirs, read_data, basic_profile
from src.rag_faiss_utils_pdf import (
    load_faiss_index,
    retrieve_chunks,
    format_rag_context,
)

# always set the location of the .env file to the project root for consistent env var loading
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root (parent of /builds)
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------------------
# Langfuse instrumentation
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
# Minimal RAG helpers (Build4: retrieval added to codegen path)
# --------------------------------------------------------------------------------------


@dataclass
class RagIndex:
    """Container for a prebuilt FAISS-backed RAG index."""

    index: Any
    chunks: list[Any]
    knowledge_dir: Path
    embedding_model: str


def load_saved_rag_index(knowledge_dir: str | Path) -> RagIndex:
    """Load a previously built FAISS index and its chunk metadata."""
    knowledge_dir = Path(knowledge_dir)
    index, chunks, embedding_model = load_faiss_index(knowledge_dir)
    return RagIndex(
        index=index,
        chunks=chunks,
        knowledge_dir=knowledge_dir,
        embedding_model=embedding_model,
    )


def prepare_codegen_request_with_rag(
    req: str,
    schema_text: str,
    rag_index: Optional[RagIndex],
    rag_k: int = 4,
) -> tuple[str, Optional[str]]:
    """Augment the codegen request with retrieved context from a saved FAISS index.

    Returns:
        (augmented_request, pretty_context_or_none)
    """
    if rag_index is None:
        return req, None

    retrieval_query = f"User request: {req}\n\nDataset schema:\n{schema_text}"
    results = retrieve_chunks(
        query=retrieval_query,
        index=rag_index.index,
        chunks=rag_index.chunks,
        k=rag_k,
        embedding_model=rag_index.embedding_model,
    )
    rag_context = format_rag_context(results)

    augmented_request = dedent(
        """
        Retrieved reference material:
        {rag_context}

        Original user request:
        {req}

        Use the retrieved material when it is relevant, but only reference dataset columns 
        that actually appear in the schema.
        """
    ).strip()
    return augmented_request, rag_context


# notification helper to print RAG status in the CLI at startup and after loading the index
def print_rag_status(rag_index):
    print("\nRAG STATUS")
    print("----------")

    if rag_index is None:
        print("RAG disabled")
        print("(no knowledge_dir provided)\n")
        return

    print("RAG enabled")
    print(f"knowledge_dir  : {rag_index.knowledge_dir}")
    print(f"chunks loaded  : {len(rag_index.chunks)}")
    print(f"embedding model: {rag_index.embedding_model}\n")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def setup_artifact_dirs(report_dir: Path) -> tuple[Path, Path]:
    """Create and return standardized artifact directories."""
    tool_output_dir = report_dir / "tool_outputs"
    tool_figure_dir = report_dir / "tool_figures"
    tool_output_dir.mkdir(parents=True, exist_ok=True)
    tool_figure_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== ARTIFACT DIRECTORIES ===")
    print("tool_outputs :", tool_output_dir)
    print("tool_figures :", tool_figure_dir)
    print()
    return tool_output_dir, tool_figure_dir


def sanitize_session_id(session_id: str) -> str:
    """Make a session id safe for folder names while keeping it readable."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", session_id.strip())
    return safe or "cli-session"


def get_session_artifact_dirs(report_dir: Path, session_id: str) -> dict[str, Path]:
    """Return a consistent session-based artifact directory structure."""
    safe_session = sanitize_session_id(session_id)

    dirs = {
        "tool_outputs_root": report_dir / "tool_outputs",
        "tool_figures_root": report_dir / "tool_figures",
        "tool_output_session": report_dir / "tool_outputs" / f"session_{safe_session}",
        "tool_figure_session": report_dir
        / "tool_figures"
        / f"session_{safe_session}"
        / "figures",
        "generated_session": report_dir / f"session_{safe_session}",
        "generated_figures": report_dir / f"session_{safe_session}" / "figures",
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    return dirs


def inject_artifact_paths(
    tool_fn,
    tool_name: str,
    args: Dict[str, Any],
    tool_output_dir: Path,
    tool_figure_dir: Path,
) -> Dict[str, Any]:
    """
    Inject standard artifact directories into tool arguments if the tool supports them
    and they weren't explicitly provided by the router/user.
    """
    sig = inspect.signature(tool_fn)
    params = sig.parameters

    # Common directory-style params across tools
    dir_param_candidates = {
        "fig_dir": tool_figure_dir,
        "plot_dir": tool_figure_dir,
        "plots_dir": tool_figure_dir,
        "figure_dir": tool_figure_dir,
        "figures_dir": tool_figure_dir,
        "out_dir": tool_output_dir,
        "output_dir": tool_output_dir,
        "artifact_dir": tool_output_dir,
        "report_dir": tool_output_dir,
    }

    for p, default_dir in dir_param_candidates.items():
        if p in params and p not in args:
            args[p] = default_dir

    # Common single-file “output path” parameters (for tables, single figures, etc.)
    # Only set if the tool takes it AND it wasn't provided.
    file_param_candidates = ["out_path", "output_path", "save_path"]
    for p in file_param_candidates:
        if p in params and p not in args:
            # Choose an extension that won’t break most tools; many plotting funcs accept .png
            # and table funcs often accept .csv/.json. If tool needs a specific ext, it should
            # set its own default or the router can pass it explicitly.
            default_path = tool_output_dir / f"{tool_name}_output"
            args[p] = default_path

    return args


def print_artifact_summary(tool_output_dir: Path, tool_figure_dir: Path) -> None:
    """Nice CLI printout; handy now and later for a UI."""
    print("\n=== ARTIFACT LOCATIONS ===")
    print(f"Tool outputs : {tool_output_dir}")
    print(f"Tool figures : {tool_figure_dir}\n")


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


# Regexes to extract fenced code blocks and JSON blocks from LLM output (best-effort, for flexibility in formatting)
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(text or "")
    return m.group(1).strip() if m else None


# This is a best-effort split of the LLM response into PLAN / CODE / VERIFY sections,
# based on simple substring searches.
def split_sections(text: str) -> Tuple[str, str, str]:
    """Split an LLM response into PLAN / CODE / VERIFY sections (best-effort)."""
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


def parse_json_object(raw: str) -> Dict[str, Any]:
    """
    Parse a JSON object from:
      - raw JSON text
      - or a fenced ```json block
    Returns {} on failure.
    """
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
            pass

    # Fallback: parse between the first '{' and last '}'.
    # This helps with model outputs that include extra pre/post text.
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(raw[i : j + 1])
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


def find_unknown_columns(args_obj: Any, known_columns: set[str]) -> set[str]:
    """
    Walk the tool args and identify unknown column references in common keys
    (column/columns/x/y/outcome/predictors/etc.). This prevents hallucinated columns.
    """
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


def coerce_tool_args(raw_args: Any) -> Dict[str, Any]:
    """Ensure tool args are a dict so **kwargs calls are safe."""  # kwargs (short for "keyword arguments") in the tool functions expect a dict; this coercion allows for more flexible LLM output formats while keeping the router robust.
    if isinstance(raw_args, dict):
        return raw_args
    return {}


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def parse_tags(tags_csv: str) -> list[str]:
    return [t.strip() for t in (tags_csv or "").split(",") if t.strip()]


def make_langfuse_config(session_id: str, tags: list[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"configurable": {"session_id": session_id}}

    if LANGFUSE_AVAILABLE:
        try:
            cfg["callbacks"] = [CallbackHandler(sessionId=session_id, tags=tags)]  # type: ignore
        except TypeError:
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

    Search order:
      1) src.tools: TOOLS
      2) src.build0_tools: TOOLS
      3) src: TOOLS  (exported in src/__init__.py)
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


# Load tool descriptions for better LLM guidance and print them in the CLI at startup
# Helps with LLM tool selection and also serves as documentation for users of the CLI.
# The descriptions are optional but recommended for better LLM performance and user experience.
# Even if descriptions are missing, the router will still work based on tool names and arg signatures.
# Helpful even with RAG, as it guides the model on how to use the retrieved context and
# which tool to pick for which kind of request.
def load_tool_descriptions() -> Dict[str, str]:
    """Best-effort load of optional TOOL_DESCRIPTIONS from src.tools."""
    try:
        mod = importlib.import_module("src.tools")
        raw = getattr(mod, "TOOL_DESCRIPTIONS", {})
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
    except Exception:
        pass
    return {}


def format_capability_hints(
    allowed_tools: list[str], descriptions: Dict[str, str]
) -> str:
    lines = []
    for tool in allowed_tools:
        desc = descriptions.get(tool, "")
        if desc:
            lines.append(f"- {tool}: {desc}")
        else:
            lines.append(f"- {tool}")
    return "\n".join(lines)


def format_tool_arg_hints(tools: Dict[str, ToolFn], allowed_tools: list[str]) -> str:
    """
    Build argument-name guidance from real tool signatures.

    Excludes framework/runtime params like df/report_dir and variadic params.
    """
    lines: list[str] = []
    for tool_name in allowed_tools:
        fn = tools.get(tool_name)
        if fn is None:
            continue

        required: list[str] = []
        optional: list[str] = []
        try:
            sig = inspect.signature(fn)
            for p in sig.parameters.values():
                if p.name in {"df", "report_dir"}:
                    continue
                if p.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if p.default is inspect.Parameter.empty:
                    required.append(p.name)
                else:
                    optional.append(p.name)
        except (TypeError, ValueError):
            lines.append(f"- {tool_name}: args unknown (could not inspect signature)")
            continue

        if required and optional:
            lines.append(f"- {tool_name}: required={required}; optional={optional}")
        elif required:
            lines.append(f"- {tool_name}: required={required}; optional=[]")
        elif optional:
            lines.append(f"- {tool_name}: required=[]; optional={optional}")
        else:
            lines.append(f"- {tool_name}: required=[]; optional=[]")

    return "\n".join(lines)


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
        "\n"
        "Your job is to generate ONE standalone Python script that satisfies the user's exact request.\n"
        "\n"
        "CORE PRIORITY:\n"
        "- Satisfy the user's exact request.\n"
        "- Do not substitute profiling, generic cleaning, exports, or summaries unless explicitly requested.\n"
        "- First determine the requested output type: plot, table, model, summary, or transformation.\n"
        "- The generated code must produce that output type.\n"
        "\n"
        "IMPORTANT RULES:\n"
        "- You ONLY know the dataset schema and any retrieved reference context. Do NOT invent columns.\n"
        "- Use only the columns needed for the requested task.\n"
        "- Validate that all referenced columns exist, and exit nonzero if required columns are missing.\n"
        "- Handle missing values explicitly, but only for the columns needed for the requested task.\n"
        "- If missing-data handling is not specified, use a simple reasonable default for the required columns only.\n"
        "- Save at least one relevant artifact into --report_dir.\n"
        "\n"
        "GENERAL VS SPECIFIC REQUESTS:\n"
        "- If the user's request is broad or exploratory, such as 'analyze the data', "
        "'what should I do first', or 'what analyses make sense', prefer broad workflow guidance.\n"
        "- For broad exploratory requests, use high-level analysis reasoning and dataset overview logic.\n"
        "- If the user's request names a specific analysis, variable, model, table, or plot, treat it as specific.\n"
        "- For specific requests, do not fall back to broad workflow guidance unless explicitly asked.\n"
        "\n"
        "SPECIAL RULES BY REQUEST TYPE:\n"
        "- If the user asks for a plot, chart, graph, figure, or visualization:\n"
        "  * generate plotting code\n"
        "  * use the requested variables\n"
        "  * save the figure into --report_dir\n"
        "  * do NOT replace the task with profiling, cleaning, or dataset export\n"
        "  * if the request is numeric by category, choose a reasonable grouped plot such as a boxplot unless the user specifies otherwise\n"
        "- If the user asks for a model:\n"
        "  * fit the requested model\n"
        "  * print and/or save interpretable results\n"
        "- If the user asks for a table or summary:\n"
        "  * generate that table or summary directly\n"
        "  * save it if appropriate\n"
        "\n"
        "SCRIPT REQUIREMENTS:\n"
        "- Produce ONE Python script that can run as a standalone file.\n"
        "- The script MUST:\n"
        "  (1) use argparse with --data and --report_dir\n"
        "  (2) read the CSV at --data with pandas\n"
        "  (3) validate referenced columns exist\n"
        "  (4) create the requested analysis output\n"
        "  (5) save at least one relevant artifact into --report_dir\n"
        "\n"
        "OUTPUT FORMAT (exactly):\n"
        "PLAN:\n"
        "- Requested task: ...\n"
        "- Output type: ...\n"
        "- Required columns: ...\n"
        "- Missing-data handling: ...\n"
        "- Saved artifact: ...\n"
        "\n"
        "CODE:\n"
        "```python\n"
        "# full script\n"
        "```\n"
        "\n"
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
    model: str,
    allowed_tools: list[str],
    tool_descriptions: Dict[str, str],
    tool_arg_hints: str,
    temperature: float = 0.0,
    stream: bool = False,
):
    """Pick one tool + args ONLY (JSON)."""
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    allow_str = format_capability_hints(allowed_tools, tool_descriptions)

    system_text = dedent(
        """
        You are a routing assistant. You pick the single BEST tool to satisfy
        a user request from an allow-list of tools.

        You see:
        - Dataset schema (columns + dtypes)
        - Allow-list tools + tool signatures
        - User request

        Allow-list tools:
        {allow_str}

        Tool argument names by signature:
        {tool_arg_hints}

        Return ONLY valid JSON in exactly this form:
        ```json
        {{
          "tool": "<one of the allow-list tool names>",
          "args": {{ ... }},
          "note": "one sentence explaining why this tool fits"
        }}
        ```

        Rules:
        - Use ONLY columns in the schema.
        - args keys MUST use valid parameter names for the selected tool signature above.
        - Do NOT use generic keys like 'column' unless that exact parameter exists.
        - IMPORTANT: If the selected tool requires an input column, args MUST include it.
        - Never output an empty args object for summarize_categorical.
        - For summarize_categorical:
          - If the user requests one column, use args {{"column": "<col>"}}
          - If the user requests multiple columns, use args {{"cat_cols": ["<col1>", "<col2>"]}}
        - Filesystem paths, report directories, and session folders are handled by the runtime.
        - Do not include markdown outside the JSON.
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            (
                "human",
                "Dataset schema:\n{schema_text}\n\nUser request:\n{user_request}\n",
            ),
        ]
    ).partial(allow_str=allow_str, tool_arg_hints=tool_arg_hints)
    return prompt | llm | StrOutputParser()


def build_router_chain(
    model: str,
    allowed_tools: list[str],
    tool_descriptions: Dict[str, str],
    tool_arg_hints: str,
    temperature: float = 0.0,
    stream: bool = False,
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    allow_str = format_capability_hints(allowed_tools, tool_descriptions)

    system_text = dedent(
        """
        You are a routing assistant for a Python data analysis agent.

        You must choose exactly one mode:
        - tool: use an existing tool when a suitable tool already exists
        - codegen: generate custom Python code when no suitable tool exists or the request is too specific
        - answer: only for simple explanatory questions that do not require running analysis

        You see:
        - Dataset schema (columns + dtypes)
        - Allow-list tools + descriptions
        - Tool argument names by signature
        - User request

        Allow-list tools:
        {allow_str}

        Tool argument names by signature:
        {tool_arg_hints}

        GENERAL VS SPECIFIC REQUESTS:
        - If the user's request is broad or exploratory, such as "analyze the data",
          "what should I do first", or "what analyses make sense", prefer broad workflow guidance.
        - If the user's request names a specific analysis, variable, model, table, or plot, treat it as specific.
        - For specific requests, do not fall back to broad workflow guidance unless explicitly asked.

        CORE ROUTING RULES:
        - Prefer tool mode for standard, repeatable analyses that match an available tool.
        - Prefer codegen mode for custom analysis requests, unusual combinations, or outputs not covered by tools.
        - Do not choose codegen if a tool clearly fits the request.
        - Do not choose tool if no available tool can reasonably satisfy the request.
        - Satisfy the user's exact request.

        PLOT-SPECIFIC RULES:
        - If an existing plotting tool clearly matches the request, choose tool mode.
        - If no plotting tool clearly matches, choose codegen mode.
        - For a request like "plot numeric_var by category_var", use a grouped plotting tool if available; otherwise choose codegen.

        ARGUMENT RULES:
        - Only use argument names that match the actual tool signature and tool description.
        - Never invent argument names.
        - Never pass precomputed objects unless the tool explicitly expects them.
        - For plot_corr_heatmap, pass numeric_cols, not corr.
        - Keep arguments minimal and directly tied to the request.

        Return exactly one JSON object.

        Tool mode example:
        {{
          "mode": "tool",
          "tool": "summarize_categorical",
          "args": {{"column": "sex"}},
          "note": "A frequency table is a categorical summary."
        }}

        Codegen mode example:
        {{
          "mode": "codegen",
          "code_request": "Create a boxplot of flipper_length_mm by species and save it.",
          "note": "This is a specific grouped plot not clearly covered by one tool."
        }}

        Answer mode example:
        {{
          "mode": "answer",
          "note": "This is a conceptual question that does not require running analysis."
        }}
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            (
                "human",
                "Dataset schema:\n{schema_text}\n\nUser request:\n{user_request}\n",
            ),
        ]
    ).partial(allow_str=allow_str, tool_arg_hints=tool_arg_hints)

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
  ask <request>                ROUTER decides: tool-run OR codegen (HITL)
  tool <request>               Force tool-run: choose one Build0 tool + args (HITL)
  code <request>               Force code generation (HITL) + approve to save
  run                          Execute last approved script via subprocess (HITL)
  exit                         Quit

Examples:
  ask run a frequency table for sex
  ask fit a regression of bill_length_mm on flipper_length_mm and sex
  tool run a correlation heatmap for numeric columns
  code create a plot of bill_length_mm by species and save it
"""


# --------------------------------------------------------------------------------------
# Traced wrappers
# --------------------------------------------------------------------------------------
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


@observe(name="build-router", as_type="generation")
def traced_router(
    router_chain,
    schema_text: str,
    request: str,
    config: Dict[str, Any],
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "router"]):
        return router_chain.invoke(
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
    report_dir: Path,
    tool_args: Dict[str, Any],
    tags: list[str],
    session_id: str,
) -> ToolResult:
    # --- Standard artifact folders (always present) ---
    artifact_dirs = get_session_artifact_dirs(report_dir, session_id)
    tool_output_dir = artifact_dirs["tool_output_session"]
    tool_figure_dir = artifact_dirs["tool_figure_session"]

    # --- Signature inspection (once) ---
    try:
        sig = inspect.signature(tool_fn)
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        sig = None
        params = {}
        accepts_kwargs = True  # safest fallback

    # --- Inject dirs into tool_args if the tool supports them and they're missing ---
    # Your plotting tools in src/plotting.py typically take fig_dir; keep this flexible.
    dir_defaults = {
        # figures
        "fig_dir": tool_figure_dir,
        "plot_dir": tool_figure_dir,
        "plots_dir": tool_figure_dir,
        "figure_dir": tool_figure_dir,
        "figures_dir": tool_figure_dir,
        # outputs
        "out_dir": tool_output_dir,
        "output_dir": tool_output_dir,
        "artifact_dir": tool_output_dir,
        # NOTE: we still pass report_dir separately below for tools that support it
    }

    for k, default_path in dir_defaults.items():
        if k not in tool_args and (k in params or accepts_kwargs):
            tool_args[k] = default_path

    # Normalize string paths → Path objects (helps if router emitted strings)
    for k in list(tool_args.keys()):
        if k in dir_defaults and isinstance(tool_args[k], str):
            tool_args[k] = Path(tool_args[k])

    # --- Trace + execute ---
    with propagate_attributes(
        tags=tags + ["build", "toolrun"],
        metadata={
            "tool": tool_name,
            "args": json.dumps(tool_args, ensure_ascii=False, default=str),
        },
    ):
        # Preserve your existing "report_dir if supported" behavior
        supports_report_dir = ("report_dir" in params) or accepts_kwargs

        if supports_report_dir:
            result = tool_fn(df, report_dir=report_dir, **tool_args)
        else:
            result = tool_fn(df, **tool_args)

        print("\n=== TOOL COMPLETE ===")
        print("Tool:", tool_name)
        print("Outputs saved to:", tool_output_dir)
        print("Figures saved to:", tool_figure_dir)
        print()

        return normalize_tool_return(tool_name, result)


# --------------------------------------------------------------------------------------
# Core routines (HITL)
# --------------------------------------------------------------------------------------
def do_tool_run(
    *,
    req: str,
    toolplan_chain,
    summarize_chain,
    tools: Dict[str, ToolFn],
    allowed_tools: list[str],
    df: pd.DataFrame,
    df_columns: set[str],
    report_dir: Path,
    schema_text: str,
    base_config: Dict[str, Any],
    tags: list[str],
    session_id: str,
) -> None:
    """Tool planner -> HITL approve -> run tool -> save output -> summarize."""
    toolplan_raw = traced_toolplan(toolplan_chain, schema_text, req, base_config, tags)
    plan = parse_json_object(toolplan_raw)
    if not plan:
        print("\nERROR: Tool planner did not return valid JSON. Try again.\n")
        print("Raw output was:\n", toolplan_raw, "\n")
        return

    do_tool_run_from_plan(
        req=req,
        plan=plan,
        summarize_chain=summarize_chain,
        tools=tools,
        allowed_tools=allowed_tools,
        df=df,
        df_columns=df_columns,
        report_dir=report_dir,
        base_config=base_config,
        tags=tags,
        session_id=session_id,
        title="TOOL PLAN",
    )


def do_tool_run_from_plan(
    *,
    req: str,
    plan: Dict[str, Any],
    summarize_chain,
    tools: Dict[str, ToolFn],
    allowed_tools: list[str],
    df: pd.DataFrame,
    df_columns: set[str],
    report_dir: Path,
    base_config: Dict[str, Any],
    tags: list[str],
    session_id: str,
    title: str = "TOOL PLAN",
) -> None:
    """Run a validated tool plan directly (used by router to avoid a second LLM plan call)."""
    tool_name = plan.get("tool")
    tool_args = coerce_tool_args(plan.get("args", {}))
    note = plan.get("note", "")

    print(f"\n=== {title} ===")
    print(json.dumps(plan, indent=2))
    if note:
        print(f"\nNote: {note}")
    print()

    if tool_name not in tools:
        print(f"\nERROR: Proposed tool '{tool_name}' is not in TOOLS registry.\n")
        print(f"Available tools: {', '.join(allowed_tools)}\n")
        return

    unknown_cols = find_unknown_columns(tool_args, df_columns)
    if unknown_cols:
        print("\nERROR: Tool args reference unknown columns.\n")
        print("Unknown columns:", ", ".join(sorted(unknown_cols)), "\n")
        return

    confirm = input(f"Run tool '{tool_name}' now? (y/n) ").strip().lower()
    if confirm != "y":
        print("\nTool execution not approved.\n")
        return

    try:
        res = traced_run_tool(
            tool_name, tools[tool_name], df, report_dir, tool_args, tags, session_id
        )
    except Exception as e:
        print(f"\nERROR running tool: {e}\n")
        return

    artifact_dirs = get_session_artifact_dirs(report_dir, session_id)
    out_txt = artifact_dirs["tool_output_session"] / f"{tool_name}_output.txt"
    save_text(out_txt, res.text)
    print(f"\nSaved tool output to: {out_txt}\n")

    summary = traced_summarize(summarize_chain, req, res.text, base_config, tags)
    print("\n=== INTERPRETATION & SUMMARY ===\n")
    print(summary + "\n")


def do_codegen(
    *,
    req: str,
    codegen_chain,
    schema_text: str,
    base_config: Dict[str, Any],
    stream: bool,
    tags: list[str],
    script_path: Path,
    state: Dict[str, Any],
    rag_index: Optional[RagIndex] = None,
    rag_k: int = 4,
) -> None:
    codegen_request, rag_context = prepare_codegen_request_with_rag(
        req=req,
        schema_text=schema_text,
        rag_index=rag_index,
        rag_k=rag_k,
    )

    if rag_context:
        print("\n=== RAG CONTEXT RETRIEVED FOR CODEGEN ===\n")
        print(rag_context + "\n")

    out = traced_codegen(
        codegen_chain, schema_text, codegen_request, base_config, stream, tags
    )
    candidate = extract_python_code(out)

    if not candidate:
        print(
            "WARNING: No fenced ```python code block found. Ask again and require it.\n"
        )
        return

    _, _, verify = split_sections(out)
    print("=== HUMAN VERIFICATION CHECKLIST (from model) ===")
    print((verify + "\n") if verify else "(No VERIFY section found.)\n")

    approve = input("Approve and save this code? (y/n) ").strip().lower()
    if approve != "y":
        print("\nCode not approved.\n")
        return

    state["code_approved"] = candidate
    save_text(script_path, candidate)
    print(f"\nApproved and saved to: {script_path}\n")
    print("Next: type 'run' to execute, or 'ask <request>' to route another request.\n")


def do_execute(
    *,
    script_path: Path,
    data_path: Path,
    report_dir: Path,
    timeout_s: int,
    state: Dict[str, Any],
    session_id: str,
) -> None:
    if not state.get("code_approved") or not script_path.exists():
        print(
            "\nNo approved script found yet. Use: code <request> (or ask <request> that routes to codegen)\n"
        )
        return

    confirm = input(f"Execute {script_path.name} now? (y/n) ").strip().lower()
    if confirm != "y":
        print("\nExecution not approved.\n")
        return

    print("\nRunning generated script...\n")
    artifact_dirs = get_session_artifact_dirs(report_dir, session_id)
    generated_report_dir = artifact_dirs["generated_session"]
    run_log_path = generated_report_dir / "run_log.txt"
    try:
        result = run_generated_script(
            script_path, data_path, generated_report_dir, timeout_s=timeout_s
        )
    except subprocess.TimeoutExpired:
        msg = f"ERROR: Script timed out after {timeout_s} seconds.\n"
        save_text(run_log_path, msg)
        print(msg)
        return

    log = []
    log.append("=== COMMAND ===\n")
    log.append(
        f"{sys.executable} {script_path} --data {data_path} --report_dir {generated_report_dir}\n\n"
    )
    log.append("=== STDOUT ===\n")
    log.append(result.stdout or "(empty)\n")
    log.append("\n=== STDERR ===\n")
    log.append(result.stderr or "(empty)\n")
    log.append(f"\n=== RETURN CODE ===\n{result.returncode}\n")
    save_text(run_log_path, "".join(log))

    print(f"Finished. Return code: {result.returncode}")
    print(f"Saved execution log to: {run_log_path}\n")


def do_router(
    *,
    req: str,
    router_chain,
    codegen_chain,
    summarize_chain,
    tools: Dict[str, ToolFn],
    allowed_tools: list[str],
    df: pd.DataFrame,
    df_columns: set[str],
    report_dir: Path,
    schema_text: str,
    base_config: Dict[str, Any],
    stream: bool,
    tags: list[str],
    script_path: Path,
    state: Dict[str, Any],
    session_id: str,
    rag_index: Optional[RagIndex] = None,
    rag_k: int = 4,
) -> None:
    """
    Router -> (tool-run OR codegen).
    If router selects tool mode but no matching tool exists in TOOLS,
    fall back to code generation.
    """
    raw = traced_router(router_chain, schema_text, req, base_config, tags)
    plan = parse_json_object(raw)
    if not plan:
        print("\nERROR: Router did not return valid JSON. Try again.\n")
        print("Raw output was:\n", raw, "\n")
        return

    mode = (plan.get("mode") or "").strip().lower()
    note = plan.get("note", "")

    print("\n=== ROUTER DECISION ===")
    print(json.dumps(plan, indent=2))
    if note:
        print(f"\nNote: {note}")
    print()

    if mode == "tool":
        router_tool = str(plan.get("tool") or "").strip()
        if router_tool not in tools:
            print(
                "Router fallback: no matching tool is available in TOOLS. "
                "Falling back to code generation.\n"
            )
            do_codegen(
                req=req,
                codegen_chain=codegen_chain,
                schema_text=schema_text,
                base_config=base_config,
                stream=stream,
                tags=tags,
                script_path=script_path,
                state=state,
                rag_index=rag_index,
                rag_k=rag_k,
            )
            return

        do_tool_run_from_plan(
            req=req,
            plan=plan,
            summarize_chain=summarize_chain,
            tools=tools,
            allowed_tools=allowed_tools,
            df=df,
            df_columns=df_columns,
            report_dir=report_dir,
            base_config=base_config,
            tags=tags,
            session_id=session_id,
            title="TOOL PLAN (from router)",
        )
        return

    if mode == "codegen":
        code_req = (plan.get("code_request") or "").strip()
        if not code_req:
            # fallback: just use the original request
            code_req = req
        do_codegen(
            req=code_req,
            codegen_chain=codegen_chain,
            schema_text=schema_text,
            base_config=base_config,
            stream=stream,
            tags=tags,
            script_path=script_path,
            state=state,
            rag_index=rag_index,
            rag_k=rag_k,
        )
        return

    print("\nERROR: Router 'mode' must be 'tool' or 'codegen'. Try again.\n")
    print("Raw output was:\n", raw, "\n")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="build4a: HITL + Router (Tool Routing + Optional CodeGen/Execute) + Langfuse"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--report_dir", type=str, default="reports")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--out_file", type=str, default="agent_generated_analysis.py")
    parser.add_argument("--timeout_s", type=int, default=60)
    parser.add_argument(
        "--knowledge_dir",
        type=str,
        default=None,
        help="Optional path to a knowledge/ folder containing a prebuilt FAISS RAG index and chunk metadata",
    )
    parser.add_argument(
        "--rag_k",
        type=int,
        default=4,
        help="Number of retrieved chunks to inject into codegen",
    )
    parser.add_argument("--session_id", type=str, default="cli-session")
    parser.add_argument(
        "--tags", type=str, default="build4", help="Comma-separated Langfuse tags"
    )
    args = parser.parse_args()

    tag_list = parse_tags(args.tags)

    data_path = Path(args.data)
    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir)
    ensure_dirs(report_dir / "tool_outputs")
    ensure_dirs(report_dir / "tool_figures")
    session_dirs = get_session_artifact_dirs(report_dir, args.session_id)

    # Load data + schema
    df = read_data(data_path)
    df_columns = set(df.columns)
    schema_text = profile_to_schema_text(basic_profile(df))

    # Load Build0 tools registry
    tools = load_tools()
    allowed_tools = sorted(tools.keys())
    tool_descriptions = load_tool_descriptions()
    tool_arg_hints = format_tool_arg_hints(tools, allowed_tools)

    # Optional Build4A RAG index (used on the codegen path only)
    rag_index: Optional[RagIndex] = None
    if args.knowledge_dir:
        knowledge_dir = Path(args.knowledge_dir)
        if not knowledge_dir.exists():
            raise FileNotFoundError(f"knowledge_dir does not exist: {knowledge_dir}")
        print(f"\nLoading saved FAISS RAG index from: {knowledge_dir}")
        try:
            rag_index = load_saved_rag_index(knowledge_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}\n\nBuild the RAG index first with:\n"
                f"  python build_rag_index.py --knowledge_dir {knowledge_dir}"
            ) from e
        print(
            f"RAG ready: {len(rag_index.chunks)} chunks loaded using {rag_index.embedding_model}.\n"
        )

    # Chains
    suggest_chain = build_suggest_chain(
        args.model, args.temperature, args.stream, args.memory
    )
    codegen_chain = build_codegen_chain(
        args.model, args.temperature, args.stream, args.memory
    )
    toolplan_chain = build_toolplan_chain(
        args.model,
        allowed_tools=allowed_tools,
        tool_descriptions=tool_descriptions,
        tool_arg_hints=tool_arg_hints,
        temperature=0.0,
        stream=args.stream,
    )
    router_chain = build_router_chain(
        args.model,
        allowed_tools=allowed_tools,
        tool_descriptions=tool_descriptions,
        tool_arg_hints=tool_arg_hints,
        temperature=0.0,
        stream=args.stream,
    )
    summarize_chain = build_results_summarizer_chain(
        args.model, args.temperature, args.stream
    )

    base_config = make_langfuse_config(session_id=args.session_id, tags=tag_list)

    script_path = session_dirs["generated_session"] / args.out_file

    print("\n=== build4a: HITL + Router + RAG ===\n")
    print(f"Tags: {tag_list}")
    print(f"Session artifact folder: {session_dirs['generated_session']}")
    print(f"Tool output folder   : {session_dirs['tool_output_session']}")
    print(f"Tool figure folder   : {session_dirs['tool_figure_session']}")
    print(f"Build0 tools loaded: {', '.join(allowed_tools)}\n")
    if rag_index is not None:
        print(
            f"RAG: ENABLED ({len(rag_index.chunks)} chunks from {rag_index.knowledge_dir})\n"
        )
    else:
        print(
            "RAG: disabled (pass --knowledge_dir to enable Build4A retrieval on codegen)\n"
        )

    if LANGFUSE_AVAILABLE:
        print("Langfuse: ENABLED (CallbackHandler + observe decorator)\n")
    else:
        print("Langfuse: not installed or not available (running without tracing)\n")

    print("Type 'help' for commands. Type 'exit' to quit.\n")

    state: Dict[str, Any] = {"code_approved": None}

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
            if not q:
                print("\nUsage: suggest <question>\n")
                continue
            _ = traced_suggest(
                suggest_chain, schema_text, q, base_config, args.stream, tag_list
            )
            continue

        if low.startswith("ask "):
            req = user_in[len("ask ") :].strip()
            if not req:
                print("\nUsage: ask <analysis request>\n")
                continue
            do_router(
                req=req,
                router_chain=router_chain,
                codegen_chain=codegen_chain,
                summarize_chain=summarize_chain,
                tools=tools,
                allowed_tools=allowed_tools,
                df=df,
                df_columns=df_columns,
                report_dir=report_dir,
                schema_text=schema_text,
                base_config=base_config,
                stream=args.stream,
                tags=tag_list,
                script_path=script_path,
                state=state,
                session_id=args.session_id,
                rag_index=rag_index,
                rag_k=args.rag_k,
            )
            continue

        if low.startswith("tool "):
            req = user_in[len("tool ") :].strip()
            if not req:
                print("\nUsage: tool <analysis request>\n")
                continue
            do_tool_run(
                req=req,
                toolplan_chain=toolplan_chain,
                summarize_chain=summarize_chain,
                tools=tools,
                allowed_tools=allowed_tools,
                df=df,
                df_columns=df_columns,
                report_dir=report_dir,
                schema_text=schema_text,
                base_config=base_config,
                tags=tag_list,
                session_id=args.session_id,
            )
            continue

        if low.startswith("code "):
            req = user_in[len("code ") :].strip()
            if not req:
                print("\nUsage: code <analysis request>\n")
                continue
            do_codegen(
                req=req,
                codegen_chain=codegen_chain,
                schema_text=schema_text,
                base_config=base_config,
                stream=args.stream,
                tags=tag_list,
                script_path=script_path,
                state=state,
                rag_index=rag_index,
                rag_k=args.rag_k,
            )
            continue

        if low == "run":
            do_execute(
                script_path=script_path,
                data_path=data_path,
                report_dir=report_dir,
                timeout_s=args.timeout_s,
                state=state,
                session_id=args.session_id,
            )
            continue

        print("\nUnrecognized command. Type 'help' for options.\n")


if __name__ == "__main__":
    main()
