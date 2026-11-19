"""FabPilot 评测框架。

对外接口：
  * :mod:`fabpilot.eval.tasks` —— 带真值标签和脚本化路由的场景。
  * :mod:`fabpilot.eval.scripted` —— 确定性的 ``ScriptedLLM`` 测试替身。
  * :mod:`fabpilot.eval.metrics` —— 基于真实输出计算的九类指标。
  * :mod:`fabpilot.eval.runner` —— 运行每一对 (config, scenario) 组合。
  * :mod:`fabpilot.eval.report` —— 渲染 JSON / Markdown 报告。
"""
from fabpilot.eval.metrics import NUMERIC_METRICS, compute_metrics
from fabpilot.eval.report import render_json, render_markdown, write_report
from fabpilot.eval.runner import CONFIGS, EvalConfig, Harness, build_harness, run_evaluation, run_scenario
from fabpilot.eval.tasks import SCENARIOS, GroundTruth, Scenario

__all__ = [
    "CONFIGS",
    "EvalConfig",
    "GroundTruth",
    "Harness",
    "NUMERIC_METRICS",
    "SCENARIOS",
    "Scenario",
    "build_harness",
    "compute_metrics",
    "render_json",
    "render_markdown",
    "run_evaluation",
    "run_scenario",
    "write_report",
]
