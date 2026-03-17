from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import csv
import sys
import time

from solvers.timesteppers import *
from .make_weak_form import *
from solvers.processing.printoff import blue
from solvers.processing.config_setup import *
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

# ---------
# Get paths
# ---------

def load_run_configs(save_dir):
    save_dir = Path(save_dir)
    cfg_path = save_dir / "settings.yaml"
    solver_params_path = save_dir / "solver_params.yaml"

    cfg = load_config(cfg_path)
    solver_parameters = load_solver_parameters(solver_params_path)

    return cfg, solver_parameters

def load_run_ufls(save_dir, namespace):
    save_dir = Path(save_dir)
    ufl_path = save_dir / "ufl_expr.yaml"
    
    ufl_cfg = load_ufl_expressions(ufl_path, namespace=namespace)

    return ufl_cfg

def main(save_dir):

    gamma_list = []
    cpu_times = []

    for gamma in range(50):

        gamma_list.append(gamma)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        cfg, solver_parameters = load_run_configs(save_dir)

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
        MESH_PATH = os.path.join(HERE, "meshes/mms", "channel_bary1.msh")

        # print(f"[solver.py] Loading mesh from: {MESH_PATH}")

        # ------------
        # Setup spaces
        # ------------

        blue(f"\n*** Starting solve ***", spaced=True)

        # Load the mesh
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

        print(f"// V Total DoFs: {V.dof_count}")
        print(f"// W Total DoFs: {W.dof_count}")

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

        u_inflow = ufl_cfg["u_inflow"]

        bc_inflow = DirichletBC(Z.sub(0), u_inflow, (1,2))
        bc_walls = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3,4))

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

        start = time.process_time()

        if solver == "CN":
            v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals, velocity_y_vals, omega_vals, r_vals, S2, energy_spec_probe, cpu_time, div_list = timestepper_CN(get_data, 
                    Z, dx, ds, 
                    t0, T, dt,
                    sample_length=L, sample_height=H,
                    make_weak_form=make_weak_form_CN, 
                    bcs=bcs, nullspace=nullspace,
                    solver_parameters=solver_parameters,
                    appctx=appctx, vtkfile_name=vtkfile_name)

        elif solver == "BDF2":
            v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals, velocity_y_vals, omega_vals, r_vals, S2, energy_spec_probe, cpu_time, div_list = timestepper_BDF2(get_data, 
                    Z, dx, ds, 
                    t0, T, dt, 
                    sample_length=L, sample_height=H,
                    make_weak_form_BDF2=make_weak_form_BDF2,
                    make_weak_form_CN=make_weak_form_CN,
                    bcs=bcs, nullspace=nullspace,
                    solver_parameters=solver_parameters,
                    appctx=appctx, vtkfile_name=vtkfile_name)

        endstart = time.process_time()
        cpu_time = (end - start) / 60
        cpu_times.append(cpu_time)

    # Data logging dict
    plot_data = {}

    plot_path = Path(save_dir) / "plots"
    plot_path.mkdir(exist_ok=True)

    if rank == 0:

    # gamma_list = [0, 0.1, 1.0, 10.0, 100.0]
    # original gamma_times = [4.5036, 4.5477, 4.5370, 4.5119, 4.5255]
    # u0 = [x,y] gamma_times = [9.7168, 8.0605, 8.1203, 8.0842, 8.0802]
    # f = [x,y] gamma_times = [6.1373, 6.1222, 6.1946, 6.1460, 6.1618]

        # ---------------
        # Plot divergence
        # ---------------

        plt.semilogy(gamma_list, cpu_time_list, "-o")
        plt.xlabel("gamma")
        plt.ylabel("CPU time to simulate")
        plt.title('Gamma vs. CPU Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "div_plot.png", dpi=200, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])