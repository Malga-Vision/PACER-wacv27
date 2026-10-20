"""HRNet experiments: PACER-HRNet."""
import os
import subprocess
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description="HRNet experiments")
    ap.add_argument("command", choices=["pacer-hrnet"])
    args, passthrough = ap.parse_known_args()

    script_map = {
        "pacer-hrnet": "hrnet/pacer_hrnet.py",
    }
    script = os.path.join(os.path.dirname(HERE), script_map[args.command])
    cmd = [sys.executable, script] + passthrough
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
