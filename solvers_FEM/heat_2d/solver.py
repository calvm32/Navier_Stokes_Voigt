from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import sys

from solvers_FEM.timesteppers import *
from .make_weak_form import *
from processing.printoff import blue
from processing.config_setup import *
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

def main(save_dir):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg, solver_parameters = load_run_configs(save_dir)

    plot_path = Path(save_dir) / "plots"
    plot_path.mkdir(exist_ok=True)

    # ------------------
    # Configure settings
    # ------------------

    # Extract settings
    t0 = cfg["t0"]
    T = cfg["T"]
    dt = cfg["dt"]
    theta = cfg["theta"]
    N = cfg["N"]
    solver = cfg["solver"]

    vtkfile_name = "Soln"

    # ------------
    # Setup spaces
    # ------------

    blue(f"\n*** Starting solve ***", spaced=True)

    # mesh and measures
    H = 1
    L = 1

    mesh = UnitSquareMesh(N, N)
    x, y = SpatialCoordinate(mesh)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    # declare function space and interpolate functions
    V = FunctionSpace(mesh, "CG", 1)

    # -------------------
    # Configure functions
    # -------------------

    # initialize t for later
    t = Constant(t0)

    namespace = {
        "as_vector": as_vector,
        "Constant": Constant,
        "x": x,
        "y": y,
        "pi": pi,
        "sin": sin,
        "cos": cos,
        "exp": exp,
        "t": t,
    }

    ufl_cfg = load_run_ufls(save_dir, namespace)

    # ------------------
    # Allocate functions
    # ------------------

    # time dependant
    def get_data(t_curr):

        t.assign(t_curr)

        return {
            "ufl_u0": ufl_cfg["ufl_u0"],
            "ufl_f": ufl_cfg["ufl_f"],
            "ufl_g": ufl_cfg["ufl_g"]
        }

    # ----------
    # Run solver
    # ----------

    if solver == "CN":
        u_error_list = timestepper_CN(get_data, 
            V, dx, ds, 
            t0, T, dt, theta=theta,
            sample_length=L, sample_height=H,
            make_weak_form=make_weak_form_CN,
            solver_parameters=solver_parameters,
            vtkfile_name=vtkfile_name)
    elif solver == "BDF2":
        u_error_list = timestepper_BDF2(get_data, 
            V, dx, ds, 
            t0, T, dt,
            sample_length=L, sample_height=H,
            make_weak_form_BDF2=make_weak_form_BDF2,
            make_weak_form_CN=make_weak_form_CN,
            solver_parameters=solver_parameters,
            vtkfile_name=vtkfile_name)

if __name__ == "__main__":

    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])