from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import csv
import sys

from modules.solvers_FEM.timesteppers import *
from .make_weak_form import *
from modules.processing.printoff import blue
from modules.processing.config_setup import *
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
    gamma = cfg["gamma"]
    Re = cfg["Re"]
    G = cfg["G"]
    P = cfg["P"]
    solver = cfg["solver"]
    elements = cfg["elements"]
    views = cfg["views"]

    # Build appctx
    appctx = {
        "Re": Re,
        "gamma": gamma,
        "velocity_space": 0
    }

    # views = news for solver param debugging
    if views == "Full":
        solver_parameters.update({
            'ksp_view': None, 
            'pc_view': None,
            'snes_view': None, 
            'pc_fieldsplit_view': None,
            'firedrake_ksp_view': None,
            'firedrake_pc_view': None,
            'firedrake_ksp_view': None,
            'firedrake_pc_view': None,
        })
    elif views == "Some":
        solver_parameters.update({
            'ksp_monitor_true_residual': None, 
            'snes_monitor': None,
        })

    vtkfile_name = "Soln"

    # --------------
    # Configure mesh
    # --------------

    HERE = os.path.dirname(os.path.abspath(__file__))
    MESH_PATH = os.path.join(HERE, "meshes", "channel.msh")

    # ------------
    # Setup spaces
    # ------------

    blue(f"\n*** Starting solve ***", spaced=True)

    mesh = Mesh(MESH_PATH)
    x, y, z = SpatialCoordinate(mesh)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

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

    # get width W
    z_coords = mesh.coordinates.dat.data[:, 2]

    local_xmin = z_coords.min()
    local_xmax = z_coords.max()

    global_zmin = comm.allreduce(local_zmin, op=MPI.MIN)
    global_zmax = comm.allreduce(local_zmax, op=MPI.MAX)

    W = global_zmax - global_zmin

    if elements == "SV":
        k = 3  # or higher for stability on arbitrary triangles
        V = VectorFunctionSpace(mesh, "CG", k)
        W = FunctionSpace(mesh, "DG", k-1)
        Z = V * W
    elif elements == "TH":
        V = VectorFunctionSpace(mesh, "CG", 2)
        W = FunctionSpace(mesh, "CG", 1)
        Z = V * W

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
        "z": z,
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

    bc_inflow = DirichletBC(Z.sub(0), ufl_inflow, (1,3))
    bc_walls = DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), (3,4))

    bcs = [bc_walls, bc_inflow]
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=Z.mesh().comm)])

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

    if solver == "CN":
            v_error_list, p_error_list = timestepper_CN(get_data, 
                Z, dx, ds, 
                t0, T, dt, 
                theta=theta, gamma=gamma, Re=Re,
                sample_xmax=L, sample_ymax=H, sample_zmax=W,
                make_weak_form=make_weak_form_CN, 
                bcs=bcs, nullspace=nullspace,
                solver_parameters=solver_parameters,
                appctx=appctx, vtkfile_name=vtkfile_name)

    elif solver == "BDF2":
            v_error_list, p_error_list = timestepper_BDF2(get_data, 
                Z, dx, ds, 
                t0, T, dt, 
                gamma=gamma, Re=Re, 
                sample_xmax=L, sample_ymax=H, sample_zmax=W,
                make_weak_form_BDF2=make_weak_form_BDF2,
                make_weak_form_CN=make_weak_form_CN,
                bcs=bcs, nullspace=nullspace,
                solver_parameters=solver_parameters,
                appctx=appctx, vtkfile_name=vtkfile_name)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])