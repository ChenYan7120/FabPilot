#!/usr/bin/env python3
"""Run a single task through the agent system from the command line."""
import sys

from fabpilot.main import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
