from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import csv
import sys
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import PillowWriter
from matplotlib.colors import TwoSlopeNorm

from modules.processing.printoff import blue, green
from modules.processing.config_setup import *
from modules.solvers_spectral import *
from .make_rhs import *

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

    vtkfile_name = "Soln"

    # --------------------
    # Setup spectral space
    # --------------------

    H = 10
    L = 40

    # MMS loops over mesh resolutions in this list
    N_list = []
    for n in range(1, 5):
        N = 2**n
        N_list.append(N)

    # calculate error as mesh size increases
    final_error_list = [] 

    for N in N_list:

        blue(f"\n*** Grid size N = {N:0d} ***", spaced=True) # report mesh size

        Nx = 2*int(N*L/2)
        Ny = 2*int(N*H/2)

        dx = L/Nx
        dy = H/Ny

        # enforce CFL
        dt = cfg["dt"]
        dt = min(dt, 0.1 * min(dx, dy)**2 / Re)

        # Grid (periodic, endpoint excluded)
        x = np.linspace(0,L,Nx,endpoint = False)
        y = np.linspace(0,H,Ny,endpoint = False)
        X,Y = np.meshgrid(x,y,indexing = "ij")

        # wavenumbers
        kx = 2.0*np.pi*np.fft.fftfreq(Nx,d = dx)
        ky = 2.0*np.pi*np.fft.fftfreq(Ny,d = dy)
        ksq = kx[:,None]**2 + ky[None,:]**2

        # -------------------
        # Configure functions
        # -------------------

        # initialize t for later
        t = t0

        namespace = {
            "x": X,
            "y": Y,
            "L": L,
            "H": H,
            "t": t
        }

        numpy_cfg = load_run_numpy(save_dir, namespace)

        # initial value
        u0 = numpy_cfg["numpy_u0"](X, Y, t0)
        u_hat_0 = np.fft.fftn(u0)
            
        # ----------
        # Run solver
        # ----------

        rhs = make_rhs(kx, ky)

        # setup forcing func
        def f_func(t):
            f = numpy_cfg["numpy_f"](X, Y, t)
            return f
        def f_hat_func(t):
            f = numpy_cfg["numpy_f"](X, Y, t)
            return np.fft.fftn(f)

        u_hat, times = timestepper_RK4(rhs, u_hat_0, f_hat_func, t0, T, dt)

        u_num_final = np.real(np.fft.ifftn(u_hat[..., -1]))
        u_exact_final = numpy_cfg["numpy_u0"](X, Y, T)

        final_error = np.sqrt(np.mean((u_num_final - u_exact_final)**2))
        final_error_list.append(final_error)
        
        green(f"Final L2 Error (temperature) = {final_error:0.8e}", spaced=True)

    # ----------------
    # Temperature plot
    # ----------------

    plt.figure()
    plt.semilogy(N_list, final_error_list, "-o")
    plt.xlabel("mesh size")
    plt.ylabel("temperature error")
    plt.grid(True)
    plt.tight_layout()
    if rank == 0:
        plt.savefig(plot_path / "MMS_convergence_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ----------------
    # now plot things!
    # ----------------


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])