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
    t = t0

    namespace = {
        "x": X,
        "y": Y,
        "L": L,
        "H": H,
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

    rhs = make_rhs(kx, ky, dealias, Re, inv_lap)

    # setup forcing func
    def f_func(t):
        f = numpy_cfg["numpy_f"](X, Y, t)
        return f
    def f_hat_func(t):
        f = numpy_cfg["numpy_f"](X, Y, t)
        return np.fft.fftn(f)

    psi_hat, times = timestepper_intfactor_RK4(rhs, psi_hat_0, f_hat_func, t0, T, dt, Re, alpha)

    # ----------------
    # now plot things!
    # ----------------

    print("done")
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    writer = PillowWriter(fps=20)
    writer.setup(fig, "navier_stokes_vorticity.gif", dpi=120)

    # initial vorticity
    psi0_hat = psi_hat[..., 0]
    omega0_hat = -(kx[:, None]**2 + ky[None, :]**2) * psi0_hat
    omega0 = np.fft.ifftn(omega0_hat).real

    # --------------------------------
    # compute stats for final project:
    # --------------------------------

    enstrophy = (1/2)*np.sum(omega0**2)*dx*dy
    grad_omega_sq_hat = (kx[:, None]**2 + ky[None, :]**2)* np.abs(omega0_hat)**2
    dissipation = (1/Re)*(1/(Nx * Ny))*np.sum(grad_omega_sq_hat)

    enstrophy_prev = enstrophy

    times_list = []
    dE_list = []
    diss_list = []

    line1, = ax2.plot([], [], linestyle='--', label=r"dE/dt")
    line2, = ax2.plot([], [], linewidth=3.0, label=r"-nu int(nabla w)^2 dx")

    ax2.set_xlim(t0, T)
    ax2.set_ylim(-1, 1)
    ax2.legend()
    ax2.set_title("Enstrophy Balance")

    # -----------------
    # continue plotting
    # -----------------

    # center color map at zero ig
    norm = TwoSlopeNorm(vmin=omega0.min(), vcenter=0.0, vmax=omega0.max())

    im = ax1.imshow(
        omega0, 
        extent=[0, L, 0, H],
        origin="lower",
        cmap="RdBu_r", 
        norm=norm,
        interpolation="bicubic", 
        aspect="auto"
    )

    cbar = fig.colorbar(im, ax=ax1, label="Vorticity ω")

    ax1.set_xlim(0, L)
    ax1.set_ylim(0, H)
    ax1.set_title(f"t = {t0:.3f}")

    writer.grab_frame()

    # animation loop
    for n in range(len(times) - 1):
        print(f"{n}/{len(times) - 1}")

        # -----------------
        # actual sol'n plot
        # -----------------
        omega_hat = -(kx[:, None]**2 + ky[None, :]**2) *psi_hat[..., n]
        omega = np.fft.ifftn(omega_hat).real

        im.set_data(omega) # update normalization
        ax1.set_title(f"t = {times[n]:.3f}")

        # -------------------
        # energy balance plot   
        # -------------------
        enstrophy = (1/2)*np.sum(omega**2)*dx*dy
        grad_omega_sq_hat = (kx[:, None]**2 + ky[None, :]**2)*np.abs(omega_hat)**2
        dissipation = (1/Re)*np.sum(grad_omega_sq_hat) *dx*dy / (Nx*Ny)

        # v simple derivative approx
        if n > 0:
            dE_dt = (enstrophy - enstrophy_prev)/dt
        else:
            dE_dt = 0.0

        enstrophy_prev = enstrophy

        times_list.append(times[n])
        dE_list.append(dE_dt)
        diss_list.append(-dissipation)

        line1.set_data(times_list, dE_list)
        line2.set_data(times_list, diss_list)

        ax2.relim()
        ax2.autoscale_view()

        writer.grab_frame()

    writer.finish()
    plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])