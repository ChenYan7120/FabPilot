"""FabPilot 的命令行入口。

子命令：
  fabpilot init-db        创建业务数据库表。
  fabpilot seed           生成合成工业数据并写入数据库。
  fabpilot run TASK       通过层级式 agent 系统运行一个任务。
  fabpilot api            启动 FastAPI 服务器。
  fabpilot ui             启动 Streamlit UI。
  fabpilot eval           运行评测 harness 并写出报告。
"""
from __future__ import annotations

import argparse
import sys


def _cmd_init_db(args: argparse.Namespace) -> int:
    from fabpilot.config import get_settings
    from fabpilot.data.database import build_session_scope

    settings = get_settings()
    with build_session_scope(settings.database_url, create_tables=True) as session:
        session.commit()
    print(f"Created tables for {settings.database_url}")
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    from fabpilot.config import get_settings
    from fabpilot.data.seed import seed

    seed(seed=args.seed, drop=args.drop, n_equipment=args.n_equipment, days=args.days)
    print(f"Seeded database (seed={args.seed}, n_equipment={args.n_equipment}, days={args.days})")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from fabpilot.runtime import build_runtime, run_task

    runtime = build_runtime()
    state = run_task(runtime, args.task, session_id=args.session_id, thread_id=args.thread_id)
    print(f"thread_id: {state.get('thread_id')}")
    print(f"status:    {state.get('status')}")
    delegated = [r.get("agent") for r in state.get("delegation_records", [])]
    if delegated:
        print(f"agents:    {', '.join(delegated)}")
    print(f"tool calls: {len(state.get('tool_history', []))}")
    print()
    print(state.get("final_report", ""))
    return 0


def _cmd_api(args: argparse.Namespace) -> int:
    import uvicorn

    from fabpilot.api.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    from fabpilot.config import get_settings

    get_settings()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "fabpilot/ui/app.py",
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    return subprocess.call(cmd)


def _cmd_eval(args: argparse.Namespace) -> int:
    from pathlib import Path

    from fabpilot.config import get_settings
    from fabpilot.eval import render_markdown, run_evaluation, write_report

    payload = run_evaluation(seed=args.seed)
    out = get_settings().data_path / "reports"
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "evaluation.md"
    json_path = out / "evaluation.json"
    write_report(payload, str(md_path), str(json_path))
    print(render_markdown(payload))
    print(f"\nWrote {md_path} and {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fabpilot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Create database tables")
    p_init.set_defaults(func=_cmd_init_db)

    p_seed = sub.add_parser("seed", help="Generate and seed synthetic data")
    p_seed.add_argument("--seed", type=int, default=42)
    p_seed.add_argument("--drop", action="store_true")
    p_seed.add_argument("--n-equipment", type=int, default=5)
    p_seed.add_argument("--days", type=int, default=7)
    p_seed.set_defaults(func=_cmd_seed)

    p_run = sub.add_parser("run", help="Run a task through the agent system")
    p_run.add_argument("task")
    p_run.add_argument("--session-id", default=None)
    p_run.add_argument("--thread-id", default=None)
    p_run.set_defaults(func=_cmd_run)

    p_api = sub.add_parser("api", help="Start FastAPI server")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.set_defaults(func=_cmd_api)

    p_ui = sub.add_parser("ui", help="Start Streamlit UI")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.set_defaults(func=_cmd_ui)

    p_eval = sub.add_parser("eval", help="Run the evaluation harness")
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
