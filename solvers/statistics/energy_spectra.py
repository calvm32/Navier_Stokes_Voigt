import numpy as np
from mpi4py import MPI

class energy_spectra:
    """
    kinetic energy spectrum using DOF
    mostly works for rectangles
    """

    def __init__(self, u, mesh, nbins=50):
        self.u = u
        self.mesh = mesh
        self.nbins = nbins

        # DOF coordinates
        self.coords = mesh.coordinates.dat.data_ro.copy()

    def compute(self):

        comm = self.mesh.comm

        coords = self.mesh.coordinates.dat.data_ro
        uvals = self.u.dat.data_ro

        # Only work on rank 0
        coords_all = comm.gather(coords, root=0)
        uvals_all = comm.gather(uvals, root=0)

        if comm.rank != 0:
            return None, None

        coords_all = np.vstack(coords_all)
        uvals_all = np.vstack(uvals_all)

        # Separate velocity components
        ux = uvals_all[:, 0]
        uy = uvals_all[:, 1]

        # Remove global mean
        ux -= np.mean(ux)
        uy -= np.mean(uy)

        # Sort DOFs lexicographically (y fastest)
        sort_inds = np.lexsort((coords_all[:,1], coords_all[:,0]))
        coords_sorted = coords_all[sort_inds]
        ux_sorted = ux[sort_inds]
        uy_sorted = uy[sort_inds]

        # Determine grid size
        x_unique = np.unique(coords_sorted[:,0])
        y_unique = np.unique(coords_sorted[:,1])

        Nx = len(x_unique)
        Ny = len(y_unique)

        # Reshape into 2D grid
        ux_grid = ux_sorted.reshape(Nx, Ny)
        uy_grid = uy_sorted.reshape(Nx, Ny)

        # Domain lengths
        Lx = x_unique.max() - x_unique.min()
        Ly = y_unique.max() - y_unique.min()

        # 2D FFT
        ux_hat = np.fft.fft2(ux_grid)
        uy_hat = np.fft.fft2(uy_grid)

        norm = (Nx * Ny)
        E_hat = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2) / norm**2

        # Wavenumbers
        kx = 2*np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
        ky = 2*np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)

        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2)

        # Radial binning
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

        k_centers = 0.5*(k_bins[:-1] + k_bins[1:])

        return k_centers, E