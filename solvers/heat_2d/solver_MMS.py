from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import sys

import matplotlib.pyplot as plt
from solvers.timesteppers import timestepper_BDF2
from .make_weak_form import make_weak_form_BDF2
from solvers.processing.printoff import blue, green
from solvers.processing.config_setup import *

# ---------
# Get paths
# ---------

def load_run_configs(save_dir):
    save_dir = Path(save_dir)
    cfg_path = save_dir / "settings.yaml"
    solver_params_path = save_dir / "solver_params.yaml"
    ufl_path = save_dir / "ufl_expr.yaml"

    cfg = load_config(cfg_path)
    solver_parameters = load_solver_parameters(solver_params_path)

    # optionally load UFL expressions
    namespace = {"Constant": Constant, "as_vector": as_vector}  # extend as needed
    ufl_cfg = load_ufl_expressions(ufl_path, namespace=namespace)

    return cfg, solver_parameters, ufl_cfg

def main(save_dir):
    cfg, solver_parameters, ufl_cfg = load_run_configs(save_dir)

    # -----------------
    # MMS Configuration
    # -----------------

    # extract settings
    t0 = cfg["t0"]
    T = cfg["T"]
    dt = cfg["dt"]
    theta = cfg["theta"]
    solver = cfg["solver"]

    vtkfile_name = "Soln"

    # -------------
    # Start solving
    # -------------

    # MMS loops over mesh resolutions in this list
    N_list = []
    for n in range(1, 6):
        N = 2**n
        N_list.append(N)

    # calculate error as mesh size increases
    final_error_list = [] 

    for N in N_list:

        blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True) # report mesh size
        new_vtkfile_name = f"{vtkfile_name}_N{N}" # write to new file

        # ------------
        # Setup spaces
        # ------------

        # mesh and measures
        mesh = UnitSquareMesh(N, N)
        x, y = SpatialCoordinate(mesh)

        dx = Measure("dx", domain=mesh)
        ds = Measure("ds", domain=mesh)

        # declare function space and interpolate functions
        V = FunctionSpace(mesh, "CG", 1)

        # -------------------
        # Configure functions
        # -------------------

        namespace = {
            "as_vector": as_vector,
            "Constant": Constant,
            "x": x,
            "y": y,
            "pi": pi,
            "sin": sin,
            "cos": cos,
            "exp": exp,
        }

        # ------------------
        # Allocate functions
        # ------------------

        # time dependant
        def get_data(t):

            namespace.update({
                "t": t,
            })

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
            u_error_list, energy_list, all_time_list = timestepper_CN(get_data, 
                V, dx, ds, 
                t0, T, dt, theta=theta,
                make_weak_form=make_weak_form_CN,
                solver_parameters=solver_parameters,
                vtkfile_name=vtkfile_name)
        elif solver == "BDF2":
            u_error_list, energy_list, all_time_list = timestepper_BDF2(get_data, 
                V, dx, ds, 
                t0, T, dt,
                make_weak_form=make_weak_form_BDF2,
                solver_parameters=solver_parameters,
                vtkfile_name=vtkfile_name)

        final_error = 0
        for err in u_error_list:
            final_error += err
        
        final_error_list.append(sqrt(final_error))
        
        green(f"Final L2 Error (temperature) = {final_error:0.8e}", spaced=True)

    # ---------------
    # Heat error plot
    # ---------------

    plt.figure()
    plt.semilogy(N_list, v_final_error_list, "-o")
    plt.xlabel("mesh size")
    plt.ylabel("heat error")
    plt.grid(True)
    plt.tight_layout()
    if rank == 0:
        plt.savefig(plot_path / "heat_convergence_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])