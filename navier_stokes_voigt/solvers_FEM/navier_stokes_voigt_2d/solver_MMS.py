from firedrake import *
import yaml
import os
from pathlib import Path
import shutil
import sys

from navier_stokes_voigt.solvers_FEM.timesteppers import *
from .make_weak_form import *
from navier_stokes_voigt.processing.printoff import blue, green
from navier_stokes_voigt.processing.config_setup import *
import matplotlib.pyplot as plt

def main(save_dir):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg, solver_parameters = load_run_configs(save_dir)

    plot_path = Path(save_dir) / "plots"
    plot_path.mkdir(exist_ok=True)

    # ------------------
    # Configure settings
    # ------------------

    t0 = cfg["t0"]
    T = cfg["T"]
    theta = cfg["theta"]
    gamma = cfg["gamma"]
    Re = cfg["Re"]
    alpha = cfg["alpha"]
    G = cfg["G"]
    P = cfg["P"]

    solver = cfg["solver"]
    elements = cfg["elements"]
    views = cfg["views"]

    vtkfile_name = "Soln"

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

    # -------------
    # Start solving
    # -------------

    HERE = os.path.dirname(os.path.abspath(__file__))

    # calculate error as mesh size increases
    v_final_error_list = []
    p_final_error_list = []

    # Loop over mesh resolutions
    N_list = []
    cpu_times = []

    for n in range(2, 6):

        N = 2**n
        N_list.append(N)

        blue(f"\n*** Mesh size N = {N:0d} ***", spaced=True) # report mesh size
        new_vtkfile_name = f"{vtkfile_name}_N{N}" # write to new file

        dt = 0.1 / N**2

        # Build appctx
        appctx = {
            "Re": Re,
            "gamma": gamma,
            "velocity_space": 0
        }

        # --------------
        # Configure mesh
        # --------------

        H = 1.0
        L = 4.0

        if elements == "SV":
            MESH_PATH = os.path.join(HERE, f"meshes/mms/channel_bary{n}.msh")
            mesh = Mesh(MESH_PATH)
        elif elements == "TH":
            mesh = RectangleMesh(int(L*N), int(H*N), L, H)

        x, y = SpatialCoordinate(mesh)

        # ------------
        # Setup spaces
        # ------------

        dx = Measure("dx", domain=mesh)
        ds = Measure("ds", domain=mesh)

        if elements == "SV":
            k = 3  # or higher for stability on arbitrary triangles
            V = VectorFunctionSpace(mesh, "CG", k)
            W = FunctionSpace(mesh, "DG", k-1)
            Z = V * W
        elif elements == "TH":
            V = VectorFunctionSpace(mesh, "CG", 2)
            W = FunctionSpace(mesh, "CG", 1)
            Z = V * W

        # --------------------
        # Compute mesh spacing
        # --------------------

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
            "alpha": alpha
        }

        ufl_cfg = load_run_ufls(save_dir, namespace)

        # -------------------
        # Boundary conditions
        # -------------------

        ufl_inflow = ufl_cfg["ufl_v0"]

        bc_inflow = DirichletBC(Z.sub(0), ufl_inflow, (1,2))
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

        if solver == "CN":
                v_error_list, p_error_list = timestepper_CN(get_data, 
                    Z, dx, ds, 
                    t0, T, dt, 
                    theta=theta, gamma=gamma, Re=Re, alpha=alpha,
                    sample_length=L, sample_height=H,
                    make_weak_form=make_weak_form_CN, 
                    bcs=bcs, nullspace=nullspace,
                    solver_parameters=solver_parameters,
                    appctx=appctx, vtkfile_name=vtkfile_name)

        elif solver == "BDF2":
                v_error_list, p_error_list = timestepper_BDF2(get_data, 
                    Z, dx, ds, 
                    t0, T, dt, 
                    gamma=0, Re=Re, alpha=alpha,
                    sample_length=L, sample_height=H,
                    make_weak_form_BDF2=make_weak_form_BDF2,
                    make_weak_form_CN=make_weak_form_CN,
                    bcs=bcs, nullspace=nullspace,
                    solver_parameters=solver_parameters,
                    appctx=appctx, vtkfile_name=vtkfile_name)

        #cpu_times.append(cpu_time)

        v_final_error = 0
        for err in v_error_list:
            v_final_error += err
        
        v_final_error_list.append(sqrt(v_final_error))

        p_final_error = 0
        for err in p_error_list:
            p_final_error += err

        p_final_error_list.append(sqrt(p_final_error))

        green(f"Final L2 Error (velocity) = {v_final_error:0.8e}", spaced=True)
        green(f"Final L2 Error (pressure) = {p_final_error:0.8e}", spaced=True)

    plot_path = Path(save_dir) / "plots"
    plot_path.mkdir(exist_ok=True)

    # -------------
    # Velocity plot
    # -------------

    plt.figure()
    plt.semilogy(N_list, v_final_error_list, "-o")
    plt.xlabel("mesh size")
    plt.ylabel("velocity error")
    plt.grid(True)
    plt.tight_layout()
    if rank == 0:
        plt.savefig(plot_path / "velocity_convergence_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

    # --------------
    # Presssure plot
    # --------------

    plt.figure()
    plt.semilogy(N_list, p_final_error_list, "-o")
    plt.xlabel("mesh size")
    plt.ylabel("pressure error")
    plt.grid(True)
    plt.tight_layout()
    if rank == 0:
        plt.savefig(plot_path / "pressure_convergencede_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])