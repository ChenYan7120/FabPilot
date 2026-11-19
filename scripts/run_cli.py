#!/usr/bin/env python3
"""从命令行通过 agent 系统运行单个任务。"""
import sys

from fabpilot.main import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
