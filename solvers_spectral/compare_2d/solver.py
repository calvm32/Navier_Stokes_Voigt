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
from scipy.optimize import curve_fit

from processing.printoff import blue
from processing.config_setup import *
from solvers_spectral import *
from .make_rhs import *
from .curve_fitter import *

def main(save_dir):

    num_comparisons = 100
    comparison_type = "rat_power" #valid: power, exp, iter, sat_exp, log, log_sat, log_power, rat_power

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
    dof = 1e4

    N = sqrt(dof/(H*L)) # number of subdivisions per unit length

    Nx = 2*int(N*L/2)
    Ny = 2*int(N*H/2)

    dx = L/Nx
    dy = H/Ny

    # enforce CFL
    if dt > Re*min(dx, dy)**2:
        dt = 0.5*Re*min(dx, dy)**2

    # Grid (periodic, endpoint excluded)
    x = np.linspace(0,L,Nx,endpoint = False)
    y = np.linspace(0,H,Ny,endpoint = False)
    X,Y = np.meshgrid(x,y,indexing = "ij")

    # Standard NumPy wavenumbers:
    # fftfreq gives cycles per unit length; multiply by 2*pi for angular wavenumbers.
    kx = 2.0*np.pi*np.fft.fftfreq(Nx,d = dx)
    ky = 2.0*np.pi*np.fft.fftfreq(Ny,d = dy)
    Laplacian_k = -kx[:,None]**2 - ky[None,:]**2

    # setup for Laplacian terms
    ksq = kx[:,None]**2 + ky[None,:]**2
    inv_lap = np.zeros_like(ksq) # array of zeroes, then keep 0 node = 0
    for i in range(ksq.shape[0]): # go through and set stuff, but avoid dividing by 0
        for j in range(ksq.shape[1]):
            if ksq[i, j] != 0:
                inv_lap[i, j] = -1.0 / ksq[i, j]

    # -------------------
    # Configure functions
    # -------------------

    # initialize t for later
    t = Constant(t0)

    namespace = {
        "x": X,
        "y": Y,
        "L": L,
        "H": H,
    }

    numpy_cfg = load_run_numpy(save_dir, namespace)

    # initial value
    numpy_psi0 = numpy_cfg["numpy_psi0"]
    psi0_func = numpy_psi0*np.ones_like(X) 
    psi_hat_0 = np.fft.fftn(psi0_func)

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

    rhs_NSE, rhs_NSV = make_rhs(kx, ky, dealias, Re, inv_lap)

    # setup forcing func
    numpy_f = numpy_cfg["numpy_f"]
    f_x_func, f_y_func = numpy_f

    f_x = f_x_func * np.ones_like(X)
    f_y = f_y_func * np.ones_like(X)

    # FFT each component and put back
    f_x_hat = np.fft.fftn(f_x)
    f_y_hat = np.fft.fftn(f_y)
    f_hat = (f_x_hat, f_y_hat)

    alpha_list = np.linspace(-5,4,num_comparisons)
    for i in range(len(alpha_list)):
        alpha_list[i] = exp(alpha_list[i])
    omega_l2_list = np.zeros_like(alpha_list)

    for i in range(len(alpha_list)):
        alpha = alpha_list[i]
        psi_hat_diff, times = timestepper_intfactor_compare_RK4(rhs_NSE, rhs_NSV, 
                                                                psi_hat_0, f_hat, t0, T, 
                                                                dt, ksq, Re, alpha)

        # initial vorticity
        psi0_hat = psi_hat_diff[..., 0]
        omega0_hat = -(kx[:, None]**2 + ky[None, :]**2) * psi0_hat
        omega0 = np.fft.ifftn(omega0_hat).real

        omega_intime = []

        # convert to vorticity
        for n in range(len(times) - 1):
            #print(f"{n}/{len(times) - 1}")

            omega_hat = -(kx[:, None]**2 + ky[None, :]**2) * psi_hat_diff[..., n]
            omega = np.fft.ifftn(omega_hat).real

            omega_intime.append(omega)

        omega_l2_list[i] = np.linalg.norm(omega_intime)
        print(f"solved {i+1}/{len(alpha_list)}")

    # ----------------
    # now plot things!
    # ----------------

    curve_fitter(alpha_list, omega_l2_list, comparison_type)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])