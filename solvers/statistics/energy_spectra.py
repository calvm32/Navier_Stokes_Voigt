import numpy as np
from mpi4py import MPI
from firedrake import *

class energy_spectra:
    """
    2D isotropic shell-averaged kinetic energy spectrum
    - Projects velocity onto structured DG0 grid
    - Performs 2D FFT
    - Radially bins in |k|
    """

    def __init__(self, u, mesh, nbins=50, Nx=128, Ny=128):
        self.u = u
        self.mesh = mesh
        self.nbins = nbins
        self.Nx = Nx
        self.Ny = Ny

        # Structured sampling space
        self.Vs = VectorFunctionSpace(mesh, "DG", 0)
        self.u_proj = Function(self.Vs)

    def compute(self):

        comm = self.mesh.comm

        # -----------------------------
        # Project velocity (fast solve)
        # -----------------------------
        self.u_proj.project(self.u)

        uvals = self.u_proj.dat.data_ro

        # Gather everything to rank 0
        u_all = comm.gather(uvals, root=0)

        if comm.rank != 0:
            return None, None

        u_all = np.vstack(u_all)

        # Extract components
        ux = u_all[:, 0]
        uy = u_all[:, 1]

        # Remove mean
        ux -= np.mean(ux)
        uy -= np.mean(uy)

        # ---------------------
        # Determine domain size
        # ---------------------
        coords = self.mesh.coordinates.dat.data_ro
        xmin = coords[:, 0].min()
        xmax = coords[:, 0].max()
        ymin = coords[:, 1].min()
        ymax = coords[:, 1].max()

        Lx = xmax - xmin
        Ly = ymax - ymin

        # ---------------------------
        # Interpolate to uniform grid
        # ---------------------------
        Nx = self.Nx
        Ny = self.Ny

        ux = ux[:Nx*Ny]
        uy = uy[:Nx*Ny]

        ux_grid = ux.reshape(Nx, Ny)
        uy_grid = uy.reshape(Nx, Ny)

        # ------
        # 2D FFT
        # ------
        ux_hat = np.fft.fft2(ux_grid)
        uy_hat = np.fft.fft2(uy_grid)

        norm = Nx * Ny
        E_hat = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2) / norm**2

        # -----------
        # Wavenumbers
        # -----------
        kx = 2*np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
        ky = 2*np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)

        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2)

        # -------------
        # Shell binning
        # -------------
        K_flat = K.ravel()
        E_flat = E_hat.ravel()

        k_bins = np.linspace(0, K_flat.max(), self.nbins + 1)

        E = np.zeros(self.nbins)
        counts = np.zeros(self.nbins)

        inds = np.digitize(K_flat, k_bins) - 1

        for i in range(len(E_flat)):
            b = inds[i]
            if 0 <= b < self.nbins:
                E[b] += E_flat[i]
                counts[b] += 1

        counts[counts == 0] = 1
        E /= counts

        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        return k_centers, E