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

from navier_stokes_voigt.processing.printoff import blue, green
from navier_stokes_voigt.processing.config_setup import *
from navier_stokes_voigt.solvers_spectral import *
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
    dt = cfg["dt"]
    Re = cfg["Re"]

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
        if dt > Re*min(dx, dy)**2:
            dt = 0.5*Re*min(dx, dy)**2

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
            "Re": Re,
        }

        numpy_cfg = load_run_numpy(save_dir, namespace)

        # initial value
        psi0 = numpy_cfg["numpy_psi0"](X, Y, t0)
        psi_hat_0 = np.fft.fftn(psi0)

        # 2/3 dealiasing
        mask = np.ones((Nx, Ny))
        def dealias(u_hat):
            Nx, Ny = u_hat.shape
            kx_cut = Nx // 3
            ky_cut = Ny // 3
            mask[kx_cut:-kx_cut, :] = 0
            mask[:, ky_cut:-ky_cut] = 0

            return u_hat * mask
            
        # ----------
        # Run solver
        # ----------

        rhs = make_rhs(kx, ky, Re)

        # setup forcing func
        def f_func(t):
            f = numpy_cfg["numpy_f"](X, Y, t)
            return f
        def f_hat_func(t):
            f = numpy_cfg["numpy_f"](X, Y, t)
            return np.fft.fftn(f)

        psi_hat, times = timestepper_intfactor_RK4(rhs, psi_hat_0, f_hat_func, t0, T, dt, Re, ksq)

        psi_num_final = np.real(np.fft.ifftn(psi_hat[..., -1]))
        psi_exact_final = numpy_cfg["numpy_psi0"](X, Y, T)

        final_error = np.sqrt(np.mean((psi_num_final - psi_exact_final)**2))
        final_error_list.append(final_error)
        
        green(f"Final L2 Error (vorticity) = {final_error:0.8e}", spaced=True)

    # --------------
    # Vorticity plot
    # --------------

    plt.figure()
    plt.semilogy(N_list, final_error_list, "-o")
    plt.xlabel("mesh size")
    plt.ylabel("vorticity error")
    plt.grid(True)
    plt.tight_layout()
    if rank == 0:
        plt.savefig(plot_path / "MMS_convergence_plot.png", dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])