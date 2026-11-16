#!/usr/bin/env python3
"""Foreground entry point for the DeepUplift Autonomous Research Runner."""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

from deepuplift.orchestration.autonomous_runner import AutonomousResearchRunner, SingleInstanceLock
from deepuplift.orchestration.hermes_codex_sdk import RunLedger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cycles", type=int, default=1, help="0 means run until stopped")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()
    runner = AutonomousResearchRunner(
        run_id=args.run_id,
        repo_root=args.repo_root,
        ledger=RunLedger(args.ledger),
        board=args.board,
        state_root=args.state_root,
        poll_interval=args.poll_interval,
    )
    if args.plan or args.plan_only:
        runner.plan()
    if args.plan_only:
        return 0

    def stop(_signum, _frame):
        runner.stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    runner.run(max_cycles=args.cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

