import yaml
from pathlib import Path
import os
import shutil
import csv
import sys

from processing.printoff import blue
from processing.config_setup import *
import matplotlib.pyplot as plt
import numpy as np

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
    gamma = cfg["gamma"]
    Re = cfg["Re"]
    alpha = cfg["alpha"]
    G = cfg["G"]
    P = cfg["P"]
    solver = cfg["solver"]
    elements = cfg["elements"]

    vtkfile_name = "Soln"

    # --------------
    # Configure mesh
    # --------------

    HERE = os.path.dirname(os.path.abspath(__file__))
    MESH_PATH = os.path.join(HERE, "meshes/mms", "channel.msh")

    # get height H
    y_coords = mesh.coordinates.dat.data[:, 1]

    local_ymin = y_coords.min()
    local_ymax = y_coords.max()

    global_ymin = comm.allreduce(local_ymin, op=MPI.MIN)
    global_ymax = comm.allreduce(local_ymax, op=MPI.MAX)

    H = global_ymax - global_ymin

    # get length L
    x_coords = mesh.coordinates.dat.data[:, 0]

    local_xmin = x_coords.min()
    local_xmax = x_coords.max()

    global_xmin = comm.allreduce(local_xmin, op=MPI.MIN)
    global_xmax = comm.allreduce(local_xmax, op=MPI.MAX)

    L = global_xmax - global_xmin

    # ------------
    # Setup spaces
    # ------------

    blue(f"\n*** Starting solve ***", spaced=True)

    if rank == 0:
        print("\n--- Degrees of Freedom ---")
        print(f"// V Total DoFs: {V.dof_count}")
        print(f"// W Total DoFs: {W.dof_count}\n")

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
        "H": H,
        "L": L,
        "G": G,
        "P": P,
        "Re": Re,
        "pi": pi,
        "sin": sin,
        "cos": cos,
        "exp": exp,
        "t": t,
    }

    ufl_cfg = load_run_ufls(save_dir, namespace)

    # -------------------
    # Boundary conditions
    # -------------------

    ufl_inflow = ufl_cfg["ufl_inflow"]

    # ------------------
    # Allocate functions
    # ------------------

    def get_data(t_curr):

        t.assign(t_curr)

        return {
            "ufl_v0": ufl_cfg["ufl_v0"],
            "ufl_p0": ufl_cfg["ufl_p0"],
            "ufl_f": ufl_cfg["ufl_f"],
            "ufl_g": ufl_cfg["ufl_g"]
        }

    # ----------
    # Run solver
    # ----------

    v_error_list, p_error_list = timestepper_RK4(get_data, 
        Z, dx, ds, 
        t0, T, dt, 
        theta=theta, gamma=gamma, Re=Re, alpha=alpha,
        sample_length=L, sample_height=H,
        make_weak_form=make_weak_form_CN, 
        bcs=bcs, nullspace=nullspace,
        solver_parameters=solver_parameters,
        appctx=appctx, vtkfile_name=vtkfile_name)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])