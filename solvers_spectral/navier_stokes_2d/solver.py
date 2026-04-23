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

from processing.printoff import blue
from processing.config_setup import *
from solvers_spectral import *

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
    theta = cfg["theta"]
    Re = cfg["Re"]
    G = cfg["G"]
    P = cfg["P"]

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

    mask = np.ones((Nx, Ny))

    # 2/3 dealiasing
    def dealias(u_hat):
        Nx, Ny = u_hat.shape
        kx_cut = Nx // 3
        ky_cut = Ny // 3
        mask[kx_cut:-kx_cut, :] = 0
        mask[:, ky_cut:-ky_cut] = 0

        return u_hat * mask
    
    # --------
    # make RHS
    # --------

    def rhs(psi_hat, f_hat, ksq):

        # laplacian
        lap_psi_hat = -ksq*psi_hat

        # gradients
        psi_x = np.fft.ifftn(1j*kx[:,None]*psi_hat)
        psi_y = np.fft.ifftn(1j*ky[None,:]*psi_hat)

        lap_psi_x = np.fft.ifftn(1j*kx[:,None]*lap_psi_hat)
        lap_psi_y = np.fft.ifftn(1j*ky[None,:]*lap_psi_hat)

        # nonlinear Jacobian
        J = psi_x*lap_psi_y - psi_y*lap_psi_x
        J_hat = np.fft.fftn(J)
        J_hat = dealias(J_hat)

        nonlinear_hat = -inv_lap * J_hat

        # viscous term
        viscous_hat = (1.0/Re)* lap_psi_hat

        # forcing term
        f_x_hat, f_y_hat = f_hat

        curl_f_hat = 1j*kx[:,None]*f_y_hat - 1j*ky[None,:]*f_x_hat
        forcing_hat = inv_lap*curl_f_hat

        return (viscous_hat + nonlinear_hat + forcing_hat) 

    # initial value
    numpy_psi0 = numpy_cfg["numpy_psi0"]
    psi0_func = numpy_psi0*np.ones_like(X) 
    psi_hat_0 = np.fft.fftn(psi0_func)

    # ----------
    # Run solver
    # ----------

    # setup forcing func
    numpy_f = numpy_cfg["numpy_f"]
    f_x_func, f_y_func = numpy_f

    f_x = f_x_func * np.ones_like(X)
    f_y = f_y_func * np.ones_like(X)

    # FFT each component and put back
    f_x_hat = np.fft.fftn(f_x)
    f_y_hat = np.fft.fftn(f_y)
    f_hat = (f_x_hat, f_y_hat)

    psi_hat, times = timestepper_intfactor_RK4(rhs, psi_hat_0, f_hat, t0, T, dt, ksq, Re)

    # ----------------
    # now plot things!
    # ----------------

    print("done")
 
    fig, ax = plt.subplots(figsize=(7, 5))

    writer = PillowWriter(fps=20)
    writer.setup(fig, "navier_stokes_vorticity.gif", dpi=120)

    # initial vorticity
    psi0_hat = psi_hat[..., 0]
    omega0_hat = -(kx[:, None]**2 + ky[None, :]**2) * psi0_hat
    omega0 = np.fft.ifftn(omega0_hat).real

    # center color map at zero ig
    norm = TwoSlopeNorm(vmin=omega0.min(), vcenter=0.0, vmax=omega0.max())

    im = ax.imshow(
        omega0, 
        extent=[0, L, 0, H],
        origin="lower",
        cmap="RdBu_r", 
        norm=norm,
        interpolation="bicubic", 
        aspect="auto"
    )

    cbar = fig.colorbar(im, ax=ax, label="Vorticity ω")

    ax.set_xlim(0, L)
    ax.set_ylim(0, H)
    ax.set_title(f"t = {t0:.3f}")

    writer.grab_frame()

    # animation loop
    for n in range(len(times) - 1):
        print(f"{n}/{len(times) - 1}")
        psi_hat_n = psi_hat[..., n]

        omega_hat = -(kx[:, None]**2 + ky[None, :]**2) * psi_hat_n
        omega = np.fft.ifftn(omega_hat).real

        im.set_data(omega) # update normalization
        ax.set_title(f"t = {times[n]:.3f}")

        writer.grab_frame()

    writer.finish()
    plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])