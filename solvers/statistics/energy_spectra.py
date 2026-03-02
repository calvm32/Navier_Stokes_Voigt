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

        if comm.rank != 0:
            return None, None

        # Domain bounds
        coords = self.mesh.coordinates.dat.data_ro
        xmin = coords[:,0].min()
        xmax = coords[:,0].max()
        ymin = coords[:,1].min()
        ymax = coords[:,1].max()

        Lx = xmax - xmin
        Ly = ymax - ymin

        Nx = 128
        Ny = 128

        x = np.linspace(xmin, xmax, Nx, endpoint=False)
        y = np.linspace(ymin, ymax, Ny, endpoint=False)

        X, Y = np.meshgrid(x, y, indexing="ij")
        points = np.vstack([X.ravel(), Y.ravel()]).T

        # Sample velocity
        u_sample = np.array([self.u.at(pt) for pt in points])
        u_sample = u_sample.reshape(Nx, Ny, 2)

        ux = u_sample[:,:,0]
        uy = u_sample[:,:,1]

        # Remove mean
        ux -= np.mean(ux)
        uy -= np.mean(uy)

        # 2D FFT
        ux_hat = np.fft.fft2(ux)
        uy_hat = np.fft.fft2(uy)

        norm = Nx * Ny
        E_hat = 0.5*(np.abs(ux_hat)**2 + np.abs(uy_hat)**2) / norm**2

        # Wavenumbers
        kx = 2*np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
        ky = 2*np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)

        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2)

        # Radial binning
        K_flat = K.ravel()
        E_flat = E_hat.ravel()

        nbins = self.nbins
        k_bins = np.linspace(0, K_flat.max(), nbins+1)

        E = np.zeros(nbins)
        counts = np.zeros(nbins)

        inds = np.digitize(K_flat, k_bins) - 1

        for i in range(len(E_flat)):
            b = inds[i]
            if 0 <= b < nbins:
                E[b] += E_flat[i]
                counts[b] += 1

        counts[counts == 0] = 1
        E /= counts

        k_centers = 0.5*(k_bins[:-1] + k_bins[1:])

        return k_centers, E