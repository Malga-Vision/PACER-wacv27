"""SSL baseline experiments: Mean Teacher, FixMatch, FlexMatch, FreeMatch."""
import os
import subprocess
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description="SSL baseline experiments")
    ap.add_argument("command", choices=["mean-teacher", "fixmatch-pose", "flexmatch", "freematch"])
    args, passthrough = ap.parse_known_args()

    script_map = {
        "mean-teacher": "ssl_methods/mean_teacher.py",
        "fixmatch-pose": "ssl_methods/fixmatch_pose.py",
        "flexmatch": "ssl_methods/flexmatch.py",
        "freematch": "ssl_methods/freematch.py",
    }
    script = os.path.join(os.path.dirname(HERE), script_map[args.command])
    cmd = [sys.executable, script] + passthrough
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
