from firedrake import *
import yaml
import os
from pathlib import Path
import shutil
import sys

from modules.solvers_FEM.timesteppers import *
from .make_weak_form import *
from modules.processing.printoff import blue, green
from modules.processing.config_setup import *
import matplotlib.pyplot as plt

from modules.solvers_FEM.navier_stokes_2d.make_weak_form import *

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
    theta = 0.5
    gamma = cfg["gamma"]
    Re = cfg["Re"]
    alpha = cfg["alpha"]
    G = cfg["G"]
    P = cfg["P"]

    solver = cfg["solver"]
    elements = cfg["elements"]
    views = cfg["views"]
    char_length = 1

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

    HERE = Path(__file__).resolve()

    # calculate error as mesh size increases
    v_final_error_list = []
    p_final_error_list = []

    v_time_error_list = []
    p_time_error_list = []

    # Loop over mesh resolutions
    h_list = []
    cpu_times = []

    for n in range(1, 10):

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

        MESH_PATH = os.path.join(HERE.parents[2], f"settings/meshes/mms/bary_channel{n}.msh")
        mesh = Mesh(MESH_PATH)

        x, y = SpatialCoordinate(mesh)

        # compute mesh size for convergence plotting
        h = mesh.cell_sizes.dat.data_ro.max()
        h_list.append(float(h))

        blue(f"\n*** Mesh size h = {h:8f} ***", spaced=True) # report mesh size
        new_vtkfile_name = f"{vtkfile_name}_h{h:4f}" # write to new file

        dt = 0.1*h**2

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
        # Normalize functions
        # -------------------

        ufl_v0 = ufl_cfg["ufl_v0"]

        # get max
        V0 = FunctionSpace(mesh, "DG", 0)

        u_in = Function(V).interpolate(ufl_v0)
        u_mag = Function(V0).project(sqrt(dot(u_in, u_in)))

        Umax = mesh.comm.allreduce(u_mag.dat.data_ro.max(), op=MPI.MAX)

        scale = 0.001

        ufl_v0_normalized = scale*ufl_cfg["ufl_v0"]/Umax
        ufl_p0_normalized = scale*ufl_cfg["ufl_p0"]/Umax
        ufl_f_normalized = scale*ufl_cfg["ufl_f"]/Umax

        # -------------------
        # Boundary conditions
        # -------------------

        bcs = [DirichletBC(Z.sub(0), ufl_v0_normalized, (1,2,3,4))]
        nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=Z.mesh().comm)])

        # ------------------
        # Allocate functions
        # ------------------

        def get_data(t_curr):
            
            t.assign(t_curr)

            return {
                "ufl_v0": ufl_v0_normalized,
                "ufl_p0": ufl_p0_normalized,
                "ufl_f": ufl_f_normalized,
                "ufl_g": as_vector([0.0, 0.0]),
            }

        Umax = 1 # b/c of normalization

        # ----------
        # Run solver
        # ----------

        if solver == "CN":
                v_error_list, p_error_list = timestepper_CN(get_data, 
                    Z, dx, ds, 
                    t0, T, dt, 
                    theta=theta, gamma=gamma, Re=Re, alpha=alpha,
                    sample_xmax=L, sample_ymax=H,
                    make_weak_form=make_weak_form_CN, 
                    bcs=bcs, nullspace=nullspace,
                    solver_parameters=solver_parameters,
                    appctx=appctx, vtkfile_name=vtkfile_name, 
                    Umax=Umax, char_length=char_length)

        elif solver == "BDF2":
                v_error_list, p_error_list = timestepper_BDF2(get_data, 
                    Z, dx, ds, 
                    t0, T, dt, 
                    gamma=0, Re=Re, alpha=alpha,
                    sample_xmax=L, sample_ymax=H,
                    make_weak_form_BDF2=make_weak_form_BDF2,
                    make_weak_form_CN=make_weak_form_CN,
                    bcs=bcs, nullspace=nullspace,
                    solver_parameters=solver_parameters,
                    appctx=appctx, vtkfile_name=vtkfile_name, 
                    Umax=Umax, char_length=char_length)

        #cpu_times.append(cpu_time)

        v_time_error = 0
        for err in v_error_list:
            v_time_error += err**2
        v_final_error = v_error_list[-1]
        
        v_time_error_list.append(sqrt(v_time_error))
        v_final_error_list.append(v_final_error)

        p_time_error = 0
        for err in p_error_list:
            p_time_error += err**2
        p_final_error = p_error_list[-1]

        p_time_error_list.append(sqrt(p_time_error))
        p_time_error_list.append(p_final_error)

        green(f"Final L2 Error (velocity) = {v_final_error:0.8e}", spaced=True)
        green(f"Final H1 Error (pressure) = {p_final_error:0.8e}", spaced=True)        
        green(f"l2 Time Norm of L2 Error (velocity) = {v_time_error:0.8e}", spaced=True)
        green(f"l2 Time Norm of H1 Error (pressure) = {p_time_error:0.8e}", spaced=True)

    plot_path = Path(save_dir) / "plots"
    plot_path.mkdir(exist_ok=True)

    # -------------
    # Velocity plot
    # -------------

    plt.figure()
    plt.loglog(h_list, v_final_error_list, "-o")
    plt.gca().invert_xaxis()
    plt.xlabel(r"log(Mesh Size $h$)")
    plt.ylabel("log(Velocity L2 Error)")
    plt.title("Convergence of Velocity L2 Error")
    plt.grid(True)
    plt.tight_layout()
    if rank == 0:
        plt.savefig(plot_path / "velocity_convergence_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

    # --------------
    # Presssure plot
    # --------------

    plt.figure()
    plt.loglog(h_list, p_final_error_list, "-o")
    plt.gca().invert_xaxis()
    plt.xlabel(r"log(Mesh Size $h$)")
    plt.ylabel("log(Pressure H1 Error)")
    plt.title("Convergence of Pressure H1 Error")
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