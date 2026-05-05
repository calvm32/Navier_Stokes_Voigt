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
    if dt > min(dx, dy)**2:
        dt = 0.5*min(dx, dy)**2

    # Grid (periodic, endpoint excluded)
    x = np.linspace(0,L,Nx,endpoint = False)
    y = np.linspace(0,H,Ny,endpoint = False)
    X,Y = np.meshgrid(x,y,indexing = "ij")

    # wavenumbers
    kx = 2.0*np.pi*np.fft.fftfreq(Nx,d = dx)
    ky = 2.0*np.pi*np.fft.fftfreq(Ny,d = dy)

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

    # ----------------
    # now plot things!
    # ----------------

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    writer = PillowWriter(fps=20)
    writer.setup(fig, "heat_solution.gif", dpi=120)

    # initial condition (stable scaling)
    umax = np.max(np.abs(u0)) + 1e-12

    im = ax1.imshow(
        u0,
        extent=[0, L, 0, H],
        origin="lower",
        cmap="RdBu_r",
        vmin=-umax,
        vmax=umax,
        interpolation="bicubic",
        aspect="auto"
    )

    cbar = fig.colorbar(im, ax=ax1, label="u")

    ax1.set_xlim(0, L)
    ax1.set_ylim(0, H)
    ax1.set_title(f"t = {t0:.3f}")

    # ---------------
    # energy tracking
    # ---------------
    energy_prev = 0.5 * np.sum(u0**2) * dx * dy

    times_list = []
    dE_list = []
    diss_list = []

    line1, = ax2.plot([], [], linestyle='--', label=r"dE/dt")
    line2, = ax2.plot([], [], linewidth=2.5, label=r"-κ ∫|∇u|² dx")

    ax2.set_xlim(t0, T)
    ax2.set_ylim(-1e-6, 1e-6)
    ax2.legend()
    ax2.set_title("Energy Balance (Heat Equation)")

    writer.grab_frame()

    # animation loop
    for n in range(len(times) - 1):
        print(f"{n}/{len(times) - 1}")

        # solution
        u = np.fft.ifftn(u_hat[..., n]).real
        im.set_data(u)
        ax1.set_title(f"t = {times[n]:.3f}")

        # energy
        energy = 0.5 * np.sum(u**2) * dx * dy

        grad_u_sq = (kx[:, None]**2 + ky[None, :]**2) * np.abs(u_hat[..., n])**2
        dissipation = np.sum(grad_u_sq) * dx * dy

        # time derivative
        if n > 0:
            dE_dt = (energy - energy_prev) / dt
        else:
            dE_dt = 0.0

        energy_prev = energy

        times_list.append(times[n])
        dE_list.append(dE_dt)
        diss_list.append(-dissipation)

        line1.set_data(times_list, dE_list)
        line2.set_data(times_list, diss_list)

        # FIXED: no autoscale (prevents collapse/flicker)
        ymin = min(min(dE_list), min(diss_list))
        ymax = max(max(dE_list), max(diss_list))
        pad = 0.1 * (ymax - ymin + 1e-12)

        ax2.set_ylim(ymin - pad, ymax + pad)

        writer.grab_frame()

    writer.finish()
    plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])