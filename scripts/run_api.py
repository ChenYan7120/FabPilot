#!/usr/bin/env python3
"""Start the FastAPI server."""
from fabpilot.main import main

if __name__ == "__main__":
    raise SystemExit(main(["api"]))
