#!/usr/bin/env python3
"""Create the business database tables."""
from fabpilot.main import main

if __name__ == "__main__":
    raise SystemExit(main(["init-db"]))
