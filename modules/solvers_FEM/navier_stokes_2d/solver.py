from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import csv
import sys
import numpy as np
from mpi4py import MPI

from modules.solvers_FEM.timesteppers import *
from .make_weak_form import *
from modules.processing.printoff import blue
from modules.processing.config_setup import *
import matplotlib.pyplot as plt
from modules.processing.post_processing.ns import plot_ns

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
    char_length = cfg["char_length"]
    probes = cfg["probes"]

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

    run_info_path = Path(save_dir) / "run_info.yaml"

    with open(run_info_path, "r") as f:
        run_info = yaml.safe_load(f)

    mesh_name = run_info.get("mesh_name")

    HERE = Path(__file__).resolve()

    if mesh_name is not None:
        MESH_PATH = os.path.join(save_dir, mesh_name)
    else:
        MESH_PATH = os.path.join(HERE.parents[2], f"settings/meshes/channel.msh")

    # ------------
    # Setup spaces
    # ------------

    blue(f"\n*** Starting solve ***", spaced=True)

    mesh = Mesh(MESH_PATH)
    x, y = SpatialCoordinate(mesh)

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
        print("\n--- Degrees of Freedom (on node 0) ---")
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
    ufl_vel_bc1 = ufl_cfg["ufl_vel_bc1"]
    ufl_vel_bc2 = ufl_cfg["ufl_vel_bc2"]
    ufl_vel_bc3 = ufl_cfg["ufl_vel_bc3"]
    ufl_vel_bc4 = ufl_cfg["ufl_vel_bc4"]
    ufl_vel_bc6 = ufl_cfg["ufl_vel_bc6"]

    # velocity BC configuration
    vel_bc1 = DirichletBC(Z.sub(0), ufl_vel_bc1, (1))
    vel_bc2 = DirichletBC(Z.sub(0), ufl_vel_bc2, (2))
    vel_bc3 = DirichletBC(Z.sub(0), ufl_vel_bc3, (3))
    vel_bc4 = DirichletBC(Z.sub(0), ufl_vel_bc4, (4))

    # only apply 6th BC if relevant
    mesh = Z.mesh().meshes[-1] if hasattr(Z.mesh(), "meshes") else Z.mesh()

    base_mesh = mesh.meshes[0]
    markers = getattr(base_mesh.exterior_facets, "unique_markers", [])
    have_interior_body = 6 in markers

    if have_interior_body:
        vel_bc6 = DirichletBC(Z.sub(0), ufl_vel_bc6, (6))
        bcs = [vel_bc1, vel_bc2, vel_bc3, vel_bc4, vel_bc6]
    else:
        bcs = [vel_bc1, vel_bc2, vel_bc3, vel_bc4]

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
            "ufl_g": as_vector([0.0, 0.0]),
        }

    # get max
    V0 = FunctionSpace(mesh, "DG", 0)

    u_in = Function(V).interpolate(ufl_inflow)
    u_mag = Function(V0).project(sqrt(dot(u_in, u_in)))

    Umax = mesh.comm.allreduce(u_mag.dat.data_ro.max(), op=MPI.MAX)

    # ----------
    # Run solver
    # ----------

    if solver == "CN":
            v_error_list, p_error_list = timestepper_CN(get_data, 
                Z, dx, ds, 
                t0, T, dt, 
                theta=theta, gamma=gamma, Re=Re,
                sample_xmax=L, sample_ymax=H,
                make_weak_form=make_weak_form_CN, 
                bcs=bcs, nullspace=nullspace,
                solver_parameters=solver_parameters,
                appctx=appctx, vtkfile_name=vtkfile_name, 
                Umax=Umax, char_length=char_length, probes=probes)

    elif solver == "BDF2":
            v_error_list, p_error_list = timestepper_BDF2(get_data, 
                Z, dx, ds, 
                t0, T, dt, 
                gamma=gamma, Re=Re,
                sample_xmax=L, sample_ymax=H,
                make_weak_form_BDF2=make_weak_form_BDF2,
                make_weak_form_CN=make_weak_form_CN,
                bcs=bcs, nullspace=nullspace,
                solver_parameters=solver_parameters,
                appctx=appctx, vtkfile_name=vtkfile_name, 
                Umax=Umax, char_length=char_length, probes=probes)

    comm.Barrier()
    if rank == 0:
        plot_ns(Path("plot_final_data.npz"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])