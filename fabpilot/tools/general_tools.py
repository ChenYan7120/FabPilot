"""General tools: restricted Python analysis, chart and report generation."""
from __future__ import annotations

import ast
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fabpilot.tools.registry import ToolContext, tool
from fabpilot.tools.schemas import AnalysisResult, ChartResult
from fabpilot.utils import ToolValidationError

# --------------------------------------------------------------------------- #
# Restricted Python analysis
# --------------------------------------------------------------------------- #

_ALLOWED = {"numpy": np, "np": np, "pandas": pd, "pd": pd, "math": math, "statistics": statistics}
_FORBIDDEN = ("__", "import", "open", "eval", "exec", "compile", "globals", "locals",
              "getattr", "setattr", "delattr", "subprocess", "os", "sys", "input", "breakpoint")


class PythonAnalysisInput(BaseModel):
    code: str = Field(description="A single Python expression, e.g. 'np.mean(v)'.")
    variables: dict[str, Any] = Field(
        default_factory=dict, description="Named variables available to the expression."
    )


def _assert_safe(code: str) -> None:
    lowered = code.lower()
    for bad in _FORBIDDEN:
        if bad in lowered:
            raise ToolValidationError(f"expression contains forbidden token '{bad}'")
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as exc:
        raise ToolValidationError(f"invalid expression: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if any(part.startswith("_") for part in node.attr.split(".")):
                raise ToolValidationError("attribute access to private members is not allowed")


@tool(
    name="python_analysis",
    description=(
        "Evaluate a restricted single-expression Python calculation (numpy/pandas/math "
        "available). No imports, I/O or attribute access to private members."
    ),
    input_model=PythonAnalysisInput,
    tags=["general"],
)
def python_analysis(ctx: ToolContext, code: str, variables: dict[str, Any] | None = None) -> AnalysisResult:
    _assert_safe(code)
    namespace = dict(_ALLOWED)
    namespace.update(variables or {})
    result = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 — restricted namespace
    if isinstance(result, (np.ndarray, pd.Series)):
        result = result.tolist()
    try:
        json.dumps(result)
    except TypeError:
        result = str(result)
    return AnalysisResult(expression=code, result=result)


# --------------------------------------------------------------------------- #
# Chart generation
# --------------------------------------------------------------------------- #


class ChartSeries(BaseModel):
    name: str
    x: list[Any]
    y: list[Any]


class ChartInput(BaseModel):
    kind: str = Field(default="line", description="'line', 'scatter' or 'hist'.")
    title: str = Field(description="Chart title.")
    series: list[ChartSeries] = Field(description="One or more series to plot.")
    filename: str | None = Field(default=None, description="Output filename (auto-generated if omitted).")


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()
    return slug or "chart"


@tool(
    name="chart",
    description="Render a line / scatter / histogram chart and save it as a PNG file.",
    input_model=ChartInput,
    tags=["general"],
)
def chart(
    ctx: ToolContext, kind: str, title: str, series: list[ChartSeries],
    filename: str | None = None,
) -> ChartResult:
    if kind not in ("line", "scatter", "hist"):
        raise ToolValidationError(f"unknown chart kind '{kind}'")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ToolValidationError(f"matplotlib not installed: {exc}")

    out_dir = Path(ctx.chart_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (filename or _safe_slug(title)) + ".png"
    path = out_dir / fname

    plt.figure(figsize=(10, 5))
    for s in series:
        if kind == "hist":
            plt.hist(s.y, bins=40, alpha=0.6, label=s.name)
        elif kind == "scatter":
            plt.scatter(s.x, s.y, s=12, alpha=0.6, label=s.name)
        else:
            plt.plot(s.x, s.y, linewidth=1.5, label=s.name)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()

    return ChartResult(kind=kind, file_path=str(path), title=title, n_series=len(series))


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #


class ReportInput(BaseModel):
    title: str = Field(description="Report title.")
    content: str = Field(description="Markdown body of the report.")
    filename: str | None = Field(default=None, description="Output filename (auto-generated if omitted).")


@tool(
    name="report",
    description="Write a Markdown report to the reports directory and return its path.",
    input_model=ReportInput,
    tags=["general"],
)
def report(ctx: ToolContext, title: str, content: str, filename: str | None = None) -> dict:
    out_dir = Path(ctx.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (filename or _safe_slug(title)) + ".md"
    path = out_dir / fname
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return {"title": title, "file_path": str(path)}
