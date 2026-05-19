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

from modules.processing.printoff import blue
from modules.processing.config_setup import *
from modules.solvers_spectral import *
from .make_rhs import *
from .curve_fitter import *

def main(save_dir):

    num_comparisons = 20 
    comparison_type = "sat_exp" #valid: power, exp, iter, sat_exp, log, log_sat, log_power, logistic
    # iter

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
    dof = 1e4

    N = sqrt(dof/(H*L)) # number of subdivisions per unit length

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
    t = Constant(t0)

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

    # setup forcing func
    def f_func(t):
        f = numpy_cfg["numpy_f"](X, Y, t)
        return f
    def f_hat_func(t):
        f = numpy_cfg["numpy_f"](X, Y, t)
        return np.fft.fftn(f)

    # linear term for intfactor
    L_hat = -ksq/Re

    # ----------
    # Run solver
    # ----------

    alpha_list = np.linspace(-5,4,num_comparisons)
    for i in range(len(alpha_list)):
        alpha_list[i] = exp(alpha_list[i])

    omega_l2_list = np.zeros_like(alpha_list)
    velocity_l2_list = np.zeros_like(alpha_list)

    for i in range(len(alpha_list)):
        alpha = alpha_list[i]


        rhs_NSE, rhs_NSV = make_rhs(kx, ky, Re, alpha)
        psi_hat_diff, times = timestepper_compare_RK4(rhs_NSE, rhs_NSV, psi_hat_0, f_hat_func, t0, T, dt)

        # initial vorticity
        psi0_hat = psi_hat_diff[..., 0]
        omega0_hat = -(kx[:, None]**2 + ky[None, :]**2) * psi0_hat
        omega0 = np.fft.ifftn(omega0_hat).real

        omega_intime = []
        velocity_intime = []

        # convert to vorticity AND VELOCITY
        for n in range(len(times) - 1):
            #print(f"{n}/{len(times) - 1}")

            # convert vorticity
            omega_hat = -(kx[:, None]**2 + ky[None, :]**2) * psi_hat_diff[..., n]
            omega = np.fft.ifftn(omega_hat).real

            omega_intime.append(omega)

            # convert velocity
            u_hat = 1j*ky[None, :] * psi_hat_diff[..., n]
            v_hat = -1j*kx[:, None] * psi_hat_diff[..., n]

            u = np.fft.ifftn(u_hat).real
            v = np.fft.ifftn(v_hat).real
            velocity_intime.append(np.sqrt(np.sum(u**2 + v**2)))

        omega_l2_list[i] = np.linalg.norm(omega_intime)
        velocity_l2_list[i] = np.linalg.norm(velocity_intime)
        print(f"solved {i+1}/{len(alpha_list)}")

    # ----------------
    # now plot things!
    # ----------------

    curve_fitter(alpha_list, omega_l2_list, comparison_type, yaxislabel=r"$||vorticity diff||_{L^2}$")
    curve_fitter(alpha_list, velocity_l2_list, comparison_type, yaxislabel=r"$||velocity diff||_{L^2}$")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])