"""Unified CLI entrypoint for all experiments.

Dispatches to each method folder's own main.py, keeping each package
independently runnable (``python -m baseline main train-base ...``).

Commands:
  baseline:       train-base, evaluate
  pacer:          finetune, train-base, evaluate
  ssl_methods:    mean-teacher, fixmatch-pose, flexmatch, freematch
  sota:           stld-hrnet
  hrnet:          pacer-hrnet
"""
import argparse
import os
import subprocess
import sys
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))

COMMAND_TO_FOLDER = {
    "train-base": "baseline",
    "evaluate": "baseline",
    "finetune": "pacer",
    "mean-teacher": "ssl_methods",
    "fixmatch-pose": "ssl_methods",
    "flexmatch": "ssl_methods",
    "freematch": "ssl_methods",
    "stld-hrnet": "sota",
    "pacer-hrnet": "hrnet",
}


def _run(command: str, passthrough_args: List[str]) -> int:
    folder = COMMAND_TO_FOLDER[command]
    main_py = os.path.join(HERE, folder, "main.py")
    cmd = [sys.executable, main_py, command] + passthrough_args
    completed = subprocess.run(cmd)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PACER: Percentile-Adaptive Curriculum on Expected Radial Error (self-training for few-shot landmark detection)",
    )
    parser.add_argument("command", choices=sorted(COMMAND_TO_FOLDER))

    args, passthrough = parser.parse_known_args()
    return _run(args.command, passthrough)


if __name__ == "__main__":
    sys.exit(main())
