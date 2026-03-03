import argparse
import subprocess
import sys
import yaml
import os


def build_parser():
    parser = argparse.ArgumentParser(description="Run NSV solver")
    parser.add_argument("--np", type=int, help="MPI processes")
    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    run_info_path = "run_info.yaml"

    if not os.path.exists(run_info_path):
        print("run_info.yaml not found. Are you inside a save directory?")
        sys.exit(1)

    with open(run_info_path) as f:
        run_info = yaml.safe_load(f)

    solver_module = run_info["solver_module"]

    print(f"Launching solver module: {solver_module}")
    print(f"Working directory: {os.getcwd()}")

    cmd = [sys.executable, "-m", solver_module]

    if args.np:
        cmd = ["mpirun", "-np", str(args.np)] + cmd

    subprocess.run(cmd)
    

if __name__ == "__main__":
    main()