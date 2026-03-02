import numpy as np
from mpi4py import MPI
from firedrake import *


class energy_spectra:
    """
    2D isotropic shell-averaged kinetic energy spectrum
    computed by direct Fourier projection on unstructured meshes.
    """

    def __init__(self, u, mesh, nbins=40, kmax=40):
        self.u = u
        self.mesh = mesh
        self.nbins = nbins
        self.kmax = kmax

    def compute(self):

        comm = self.mesh.comm
        dxm = Measure("dx", domain=self.mesh)

        # -------------------------------------------------
        # Domain area
        # -------------------------------------------------
        area = assemble(1.0 * dxm)

        # -------------------------------------------------
        # Remove mean velocity (componentwise!)
        # -------------------------------------------------
        mean_u0 = assemble(self.u[0] * dxm) / area
        mean_u1 = assemble(self.u[1] * dxm) / area

        u_fluct = as_vector([
            self.u[0] - mean_u0,
            self.u[1] - mean_u1
        ])

        # -------------------------------------------------
        # Domain size (global min/max across MPI)
        # -------------------------------------------------
        coords = self.mesh.coordinates.dat.data_ro

        xmin = comm.allreduce(coords[:, 0].min(), op=MPI.MIN)
        xmax = comm.allreduce(coords[:, 0].max(), op=MPI.MAX)
        ymin = comm.allreduce(coords[:, 1].min(), op=MPI.MIN)
        ymax = comm.allreduce(coords[:, 1].max(), op=MPI.MAX)

        Lx = xmax - xmin
        Ly = ymax - ymin

        # -------------------------------------------------
        # Wavenumber grid
        # -------------------------------------------------
        kx_vals = np.arange(-self.kmax, self.kmax + 1)
        ky_vals = np.arange(-self.kmax, self.kmax + 1)

        energies = []
        kmags = []

        x = SpatialCoordinate(self.mesh)

        for kx_i in kx_vals:
            for ky_i in ky_vals:

                if kx_i == 0 and ky_i == 0:
                    continue

                kx = 2.0 * np.pi * kx_i / Lx
                ky = 2.0 * np.pi * ky_i / Ly

                phase = exp(-1j * (kx * x[0] + ky * x[1]))

                # Fourier coefficients (Firedrake already MPI-reduced)
                uhat_x = assemble(u_fluct[0] * phase * dxm)
                uhat_y = assemble(u_fluct[1] * phase * dxm)

                # Normalize by domain area
                uhat_x /= area
                uhat_y /= area

                energy = 0.5 * (abs(uhat_x)**2 + abs(uhat_y)**2)

                energies.append(energy)
                kmags.append(np.sqrt(kx**2 + ky**2))

        energies = np.array(energies)
        kmags = np.array(kmags)

        # Only rank 0 performs binning
        if comm.rank != 0:
            return None, None

        # -------------------------------------------------
        # Shell averaging
        # -------------------------------------------------
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