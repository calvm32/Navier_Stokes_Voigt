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

from navier_stokes_voigt.processing.printoff import blue
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
    numpy_u0 = numpy_cfg["numpy_u0"]
    u0_func = numpy_u0*np.ones_like(X) 
    u_hat_0 = np.fft.fftn(u0_func)
        
    # ----------
    # Run solver
    # ----------

    rhs = make_rhs(kx, ky)

    # setup forcing func
    numpy_f = numpy_cfg["numpy_f"]
    f_x_func, f_y_func = numpy_f

    f_x = f_x_func * np.ones_like(X)
    f_y = f_y_func * np.ones_like(X)

    # FFT each component and put back
    f_x_hat = np.fft.fftn(f_x)
    f_y_hat = np.fft.fftn(f_y)
    f_hat = (f_x_hat, f_y_hat)

    u_hat, times = timestepper_intfactor_RK4(rhs, u_hat_0, f_hat, t0, T, dt, ksq)

    # ----------------
    # now plot things!
    # ----------------


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])