from firedrake import *
from mpi4py import MPI
import numpy as np

class FFT_energy_spectrum_2d:

    def __init__(self, u, Nx=512, Ny=512, nbins=80):

        self.u = u
        self.mesh = u.function_space().mesh()

        self.Nx = Nx
        self.Ny = Ny
        self.nbins = nbins

    def compute(self):

        comm = self.mesh.comm

        # only rank 0 computes FFT
        if comm.rank != 0:
            return None, None

        coords = self.mesh.coordinates.dat.data_ro

        xmin = coords[:,0].min()
        xmax = coords[:,0].max()

        ymin = coords[:,1].min()
        ymax = coords[:,1].max()

        Lx = xmax - xmin
        Ly = ymax - ymin

        x = np.linspace(xmin, xmax, self.Nx)
        y = np.linspace(ymin, ymax, self.Ny)

        X, Y = np.meshgrid(x, y)

        pts = np.column_stack([
            X.ravel(),
            Y.ravel()
        ])

        # ---------------
        # sample velocity
        # ---------------
        ux = np.full(len(pts), np.nan)
        uy = np.full(len(pts), np.nan)

        for i, p in enumerate(pts):

            try:
                val = self.u.at(p)

                ux[i] = val[0]
                uy[i] = val[1]

            except PointNotInDomainError:
                pass

        ux = ux.reshape(self.Ny, self.Nx)
        uy = uy.reshape(self.Ny, self.Nx)

        # mask airfoil / outside region
        mask = np.isfinite(ux) & np.isfinite(uy)

        ux[~mask] = 0.0
        uy[~mask] = 0.0

        # remove mean
        ux_mean = np.mean(ux[mask])
        uy_mean = np.mean(uy[mask])

        ux -= ux_mean
        uy -= uy_mean

        # FFT
        dx = Lx / (self.Nx - 1)
        dy = Ly / (self.Ny - 1)

        ux_hat = np.fft.fft2(ux)
        uy_hat = np.fft.fft2(uy)

        ux_hat *= dx * dy
        uy_hat *= dx * dy

        # wavenumbers
        kx = 2*np.pi*np.fft.fftfreq(self.Nx, d=dx)
        ky = 2*np.pi*np.fft.fftfreq(self.Ny, d=dy)

        KX, KY = np.meshgrid(kx, ky)

        kmag = np.sqrt(KX**2 + KY**2)

        # modal energy
        E2D = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2)

        # isotropic shell averaging
        kmax = kmag.max()
        bins = np.linspace(0.0, kmax, self.nbins + 1)
        E = np.zeros(self.nbins)

        for i in range(self.nbins):

            shell = ((kmag >= bins[i]) & (kmag < bins[i+1]))
            E[i] = np.sum(E2D[shell])

        dk = bins[1] - bins[0]

        E /= dk

        k = 0.5*(bins[:-1] + bins[1:])

        # Parseval check
        E_phys = (0.5 * np.mean(
                ux[mask]**2 +
                uy[mask]**2
            ) *Lx *Ly
        )

        E_spec = np.sum(E * dk)

        print(
            "Parseval relative error:",
            abs(E_phys - E_spec) / E_phys
        )

        return k, E