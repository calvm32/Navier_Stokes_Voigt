import numpy as np
from mpi4py import MPI
from firedrake import *
import cmath


class energy_spectra:
    """
    2D isotropic shell-averaged kinetic energy spectrum
    computed by direct Fourier projection.

    Works on arbitrary unstructured meshes.
    No structured grid required.
    """

    def __init__(self, u, mesh, nbins=40, kmax=40):
        self.u = u
        self.mesh = mesh
        self.nbins = nbins
        self.kmax = kmax

    def compute(self):

        comm = self.mesh.comm

        # Remove mean velocity (important!)
        mean_u = assemble(self.u * dx) / assemble(1.0 * dx(domain=self.mesh))
        u_fluct = self.u - mean_u

        # Domain size estimate (for wavenumber scaling)
        coords = self.mesh.coordinates.dat.data_ro
        xmin = coords[:, 0].min()
        xmax = coords[:, 0].max()
        ymin = coords[:, 1].min()
        ymax = coords[:, 1].max()

        Lx = xmax - xmin
        Ly = ymax - ymin

        # Define wavenumber grid
        kx_vals = np.arange(-self.kmax, self.kmax + 1)
        ky_vals = np.arange(-self.kmax, self.kmax + 1)

        energies = []
        kmags = []

        for kx_i in kx_vals:
            for ky_i in ky_vals:

                if kx_i == 0 and ky_i == 0:
                    continue

                kx = 2 * np.pi * kx_i / Lx
                ky = 2 * np.pi * ky_i / Ly

                kvec = as_vector([kx, ky])
                x = SpatialCoordinate(self.mesh)

                phase = exp(-1j * (kx * x[0] + ky * x[1]))

                # Fourier coefficient (complex)
                uhat_x = assemble(u_fluct[0] * phase * dx)
                uhat_y = assemble(u_fluct[1] * phase * dx)

                # Combine MPI contributions
                uhat_x = comm.allreduce(uhat_x, op=MPI.SUM)
                uhat_y = comm.allreduce(uhat_y, op=MPI.SUM)

                energy = 0.5 * (abs(uhat_x)**2 + abs(uhat_y)**2)

                energies.append(energy)
                kmags.append(np.sqrt(kx**2 + ky**2))

        energies = np.array(energies)
        kmags = np.array(kmags)

        if comm.rank != 0:
            return None, None

        # Shell binning
        k_bins = np.linspace(0, kmags.max(), self.nbins + 1)

        E = np.zeros(self.nbins)
        counts = np.zeros(self.nbins)

        inds = np.digitize(kmags, k_bins) - 1

        for i in range(len(energies)):
            b = inds[i]
            if 0 <= b < self.nbins:
                E[b] += energies[i]
                counts[b] += 1

        counts[counts == 0] = 1
        E /= counts

        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        return k_centers, E