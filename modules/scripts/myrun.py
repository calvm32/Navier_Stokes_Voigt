import argparse
import sys
import yaml
import os
from pathlib import Path

def set_headless_hpc_env():
    """
    Prevents:
    - X11 authorization attempts
    - Mesa / OpenGL GPU probing (/dev/dri/renderD128)
    - VTK offscreen renderer GPU fallback
    - Qt GUI backend initialization
    - ROCm SMI initialization noise
    """

    os.environ.setdefault("DISPLAY", "")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # matplotlib safety (if ever imported indirectly)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # force software rendering for OpenGL (prevents /dev/dri access)
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")

    # VTK headless mode (critical for Firedrake ecosystems)
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    # ROCm / AMD noise suppression (safe even if not present)
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    os.environ.setdefault("ROCR_VISIBLE_DEVICES", "")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "")


def build_parser():
    parser = argparse.ArgumentParser(description="Run solver")
    parser.add_argument("path", help="Save directory")
    parser.add_argument("--np", type=int, help="MPI processes")
    return parser


def main():
    set_headless_hpc_env()

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