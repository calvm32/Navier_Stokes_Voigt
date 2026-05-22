import argparse
import sys
import yaml
import os
import subprocess
from pathlib import Path

def build_parser():
    parser = argparse.ArgumentParser(description="Run solver")
    parser.add_argument("path", help="Save directory")
    parser.add_argument("--np", type=int, help="MPI processes")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    save_dir = Path(args.path).resolve()
    run_info_path = save_dir / "run_info.yaml"

    if not run_info_path.exists():
        print("run_info.yaml not found. Are you inside a save directory?")
        sys.exit(1)

    with open(run_info_path) as f:
        run_info = yaml.safe_load(f)

    solver_module = run_info["solver_module"]
    print(f"Launching solver module: {solver_module}")
    print(f"Save directory: {save_dir}")

    python_exe = sys.executable  # absolute path to venv python
    python_cmd = [python_exe, "-m", solver_module, str(save_dir)]

    if args.np:
        mpirun = "/usr/bin/mpirun"  # absolute path — no PATH lookup ambiguity
        cmd = [mpirun, "-np", str(args.np), "--bind-to", "none"] + python_cmd
    else:
        cmd = python_cmd

    #print(f"DEBUG cmd: {cmd}", flush=True)

    result = subprocess.run(cmd, env=os.environ.copy())
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()