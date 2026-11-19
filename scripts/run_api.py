#!/usr/bin/env python3
"""启动 FastAPI 服务器。"""
from fabpilot.main import main

if __name__ == "__main__":
    raise SystemExit(main(["api"]))
