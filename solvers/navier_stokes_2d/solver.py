from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import csv
import sys

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
        "velocity_x_space": 0
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
    MESH_PATH = os.path.join(HERE, "meshes", "step_big_h0.9.msh")

    # print(f"[solver.py] Loading mesh from: {MESH_PATH}")

    # ------------
    # Setup spaces
    # ------------

    blue(f"\n*** Starting solve ***\n", spaced=True)

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

    # print(f"// V Total DoFs: {V.dof_count}")
    # print(f"// W Total DoFs: {W.dof_count}")

    # -------------------
    # Configure functions
    # -------------------

    # initialize t for later
    t = t0

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
        v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals, velocity_y_vals, omega_vals, r_vals, S2, energy_spec_probe = timestepper_CN(get_data, 
                Z, dx, ds, 
                t0, T, dt,
                sample_length=L, sample_height=H,
                make_weak_form=make_weak_form_CN, 
                bcs=bcs, nullspace=nullspace,
                solver_parameters=solver_parameters,
                appctx=appctx, vtkfile_name=vtkfile_name)

    elif solver == "BDF2":
        v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals, velocity_y_vals, omega_vals, r_vals, S2, energy_spec_probe = timestepper_BDF2(get_data, 
                Z, dx, ds, 
                t0, T, dt, 
                sample_length=L, sample_height=H,
                make_weak_form_BDF2=make_weak_form_BDF2,
                make_weak_form_CN=make_weak_form_CN,
                bcs=bcs, nullspace=nullspace,
                solver_parameters=solver_parameters,
                appctx=appctx, vtkfile_name=vtkfile_name)

    # Data logging dict
    plot_data = {}

    plot_path = Path(save_dir) / "plots"
    plot_path.mkdir(exist_ok=True)

    if rank == 0:

        # -----------------
        # Plot palinstrophy
        # -----------------

        plot_data["palinstrophy"] = (every_time_list, palinstrophy_list)
        plt.semilogy(every_time_list, palinstrophy_list, "-o")
        plt.xlabel("time")
        plt.ylabel("palinstrophy")
        plt.title('Palinstrophy')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "palinstrophy_plot.png", dpi=200, bbox_inches='tight')
        plt.close()

        # --------------------
        # Plot stream function
        # --------------------

        plot_data["stream_func"] = (every_time_list, stream_func_list)
        plt.semilogy(every_time_list, stream_func_list, "-o")
        plt.xlabel("time")
        plt.ylabel("stream function")
        plt.title('Stream Function')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "stream_func_plot.png", dpi=200, bbox_inches='tight')
        plt.close()

        # --------------
        # Plot Enstrophy
        # --------------

        plot_data["enstrophy"] = (every_time_list, enstrophy_list)
        plt.semilogy(every_time_list, enstrophy_list, "-o")
        plt.xlabel("time")
        plt.ylabel("enstrophy")
        plt.title('Enstrophy')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "enstrophy_plot.png", dpi=200, bbox_inches='tight')
        plt.close()

        # -----------
        # Plot Energy
        # -----------

        # pop first values 
        all_time_list_del = all_time_list[1:]
        energy_list_del = energy_list[1:]

        plot_data["energy"] = (all_time_list, energy_list)
        plt.semilogy(all_time_list_del, energy_list_del, "-o")
        plt.xlabel("time")
        plt.ylabel("energy")
        plt.title('Total Kinetic Energy')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "energy_plot.png", dpi=200, bbox_inches='tight')
        plt.close()

        # ------------
        # Velocity PDF
        # ------------

        plot_data["velocity_x_pdf"] = (np.arange(len(velocity_x_vals)), velocity_x_vals)
        plt.hist(velocity_x_vals, bins=100, density=True)
        plt.xlabel("samples")
        plt.ylabel("x-velocity")
        plt.title('x-Velocity Probabiility Density Function')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "velocity_x_PDF.png", dpi=200, bbox_inches='tight')
        plt.close()

        plot_data["velocity_y_pdf"] = (np.arange(len(velocity_y_vals)), velocity_y_vals)
        plt.hist(velocity_y_vals, bins=100, density=True)
        plt.xlabel("samples")
        plt.ylabel("y-velocity")
        plt.title('y-Velocity Probabiility Density Function')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "velocity_y_PDF.png", dpi=200, bbox_inches='tight')
        plt.close()

        # -------------
        # Vorticity PDF
        # -------------

        plot_data["vorticity_pdf"] = (np.arange(len(omega_vals)), omega_vals)
        plt.hist(omega_vals, bins=100, density=True)
        plt.xlabel("samples")
        plt.ylabel("vorticity")
        plt.title('Vorticity Probability Density Function')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "vorticity_PDF.png", dpi=200, bbox_inches='tight')
        plt.close()

        # --------------
        # structure func
        # --------------

        plot_data["structure_function"] = (r_vals, S2)
        plt.plot(r_vals, S2, "-o")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$S_2(r)$")
        plt.title('2nd-Order Longitudinal Structure Function')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "structure_function.png", dpi=200)
        plt.close()

        # --------------------
        # Plot Energy spectrum 
        # --------------------

        # based on Taylor's frozen flow hypothesis (at point, vary times)

        probe_values = np.array(energy_spec_probe)
        time_values = np.array(every_time_list)

        dt = time_values[1] - time_values[0]

        N = len(time_values)
        T_total = N * dt
        ux = probe_values[:, 0]
        ux = ux - np.mean(ux) # remove mean

        # FFT
        u_hat = np.fft.fft(ux)

        # Frequencies
        f = np.fft.fftfreq(N, d=dt)

        # keep positive frequencies only
        pos = f > 0
        f = f[pos]
        u_hat = u_hat[pos]

        # One-sided temporal spectrum
        E_f = 2 * (dt / N) * np.abs(u_hat)**2

        # Taylor hypothesis
        U_mean = np.mean(probe_values[:,0])
        k = 2*np.pi*f / U_mean
        E_k = E_f * U_mean / (2*np.pi)

        plt.figure()

        mask = (k > 0) & (E_k > 0) & np.isfinite(E_k)
        k_plot = k[mask]
        E_plot = E_k[mask]
        plt.loglog(k_plot, E_plot, label="Energy spectrum")

        # Reference slope
        if len(k_plot) > 6:
            C = E_plot[5] * k_plot[5]**(5/3)
            plt.loglog(k_plot, C * k_plot**(-5/3), '--', label=r"$k^{-5/3}$")

        plot_data["energy_spec_probe"] = (k, E_k)
        plt.xlabel("Wavenumber k")
        plt.ylabel("E(k)")
        plt.title('Time-Averaged Energy Spectrum at Single Pt.')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path / "energy_spec_FF2.png", dpi=200)
        plt.close()

        # --------------------
        # Save all data to CSV
        # --------------------

        with open(plot_path / "all_plot_data.csv", "w", newline="") as f:
            writer = csv.writer(f)

            for key, value in plot_data.items():
                writer.writerow([f"# {key}"])

                if isinstance(value, tuple) and len(value) == 2:
                    x_vals, y_vals = value
                    writer.writerow(["x", "y"])
                    for x, y in zip(x_vals, y_vals):
                        writer.writerow([x, y])

                else:
                    writer.writerow(["value"])
                    for v in value:
                        writer.writerow([v])

                writer.writerow([])

        #print("[solver.py] All plot data saved to 'all_plot_data.csv'")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])