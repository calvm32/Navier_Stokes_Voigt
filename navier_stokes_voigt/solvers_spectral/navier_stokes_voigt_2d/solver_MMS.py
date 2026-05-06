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
    alpha = cfg["alpha"]

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
        dt = min(cfg["dt"], 0.5*Re*min(dx, dy)**2)

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
            "alpha": alpha,
        }

        numpy_cfg = load_run_numpy(save_dir, namespace)

        # initial value
        psi0 = numpy_cfg["numpy_psi0"](X, Y, t0)
        psi_hat_0 = np.fft.fftn(psi0)

        rhs = make_rhs(kx, ky, Re, alpha)

        # linear term for intfactor
        L_hat = -ksq/Re

        # -------------------------------------
        # compute actual forcing based on exact
        # -------------------------------------

        # setup forcing func
        # def f_func(t):
        #     f = numpy_cfg["numpy_f"](X, Y, t)
        #     return f
        # def f_hat_func(t):
        #     f = numpy_cfg["numpy_f"](X, Y, t)
        #     return np.fft.fftn(f)

        def f_hat_func(t):
            psi_exact = numpy_cfg["numpy_psi0"](X, Y, t)
            psi_hat = np.fft.fftn(psi_exact)

            psi_t = numpy_cfg["numpy_psi_t"](X, Y, t)
            psi_t_hat = np.fft.fftn(psi_t)

            zero_hat = np.zeros_like(psi_hat)

            rhs_eval = rhs(psi_hat, zero_hat)

            # multiply BACK by (1 + alpha^2 k^2)
            f_hat = (1.0 + alpha**2 * ksq) * psi_t_hat - (rhs_eval * (1.0 + alpha**2 * ksq))

            return f_hat

        # ----------
        # Run solver
        # ----------

        #psi_hat, times = timestepper_intfactor_RK4(rhs, psi_hat_0, f_hat_func, t0, T, dt, L_hat)
        psi_hat, times = timestepper_RK4(rhs, psi_hat_0, f_hat_func, t0, T, dt)
        for i in range(len(times)):
            psi_num_now = np.real(np.fft.ifftn(psi_hat[..., i]))
            psi_exact_now = numpy_cfg["numpy_psi0"](X, Y, times[i])
            diff = np.sqrt(np.mean((psi_num_now - psi_exact_now)**2))
            # print(f"RMS diff at {i} = {diff}")

        psi_num_final = np.real(np.fft.ifftn(psi_hat[..., -1]))
        psi_exact_final = numpy_cfg["numpy_psi0"](X, Y, T)

        error_sq = (psi_num_final - psi_exact_final)**2
        final_error = np.sqrt(np.sum(error_sq) * dx * dy)
        final_error_list.append(final_error)

        # print("RMS final =", np.sqrt(np.mean(error_sq)))
        # print("L2 final =", np.sqrt(np.sum(error_sq) * dx * dy))
        # print("sum(error_sq) =", np.sum(error_sq))
        # print("dx*dy =", dx*dy)
        
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