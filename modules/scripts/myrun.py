import argparse
import sys
import yaml
import os
from pathlib import Path
import mpi4py

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

    # Build command
    python_cmd = [sys.executable, "-m", solver_module, str(save_dir)]

    if args.np:
        cmd = ["mpirun", "-np", str(args.np)] + python_cmd
    else:
        cmd = python_cmd

    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()