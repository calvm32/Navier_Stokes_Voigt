import numpy as np
from mpi4py import MPI
from firedrake import *

class energy_spectra_3d:
    """
    3D isotropic shell-averaged kinetic energy spectrum
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
        area = assemble(1.0 * dxm)

        # --------------------
        # Remove mean velocity
        # --------------------
        mean_ux = assemble(self.u[0] * dxm) / area
        mean_uy = assemble(self.u[1] * dxm) / area
        mean_uz = assemble(self.u[2] * dxm) / area

        u_fluct = as_vector([
            self.u[0] - mean_ux,
            self.u[1] - mean_uy,
            self.u[2] - mean_uz
        ])

        # ------------------
        # Global domain size
        # ------------------
        if hasattr(mesh, 'coordinates'):
            coords = mesh.coordinates.dat.data_ro
        elif hasattr(mesh, 'meshes'):
            coords = mesh.meshes[0].coordinates.dat.data_ro
        else:
            raise AttributeError(f"Cannot extract coordinates from mesh of type {type(mesh)}")

        xmin = comm.allreduce(coords[:, 0].min(), op=MPI.MIN)
        xmax = comm.allreduce(coords[:, 0].max(), op=MPI.MAX)
        ymin = comm.allreduce(coords[:, 1].min(), op=MPI.MIN)
        ymax = comm.allreduce(coords[:, 1].max(), op=MPI.MAX)
        zmin = comm.allreduce(coords[:, 2].min(), op=MPI.MIN)
        zmax = comm.allreduce(coords[:, 2].max(), op=MPI.MAX)

        Lx = xmax - xmin
        Ly = ymax - ymin
        Lz = zmax - zmin

        # ---------------
        # Wavenumber grid
        # ---------------
        kx_vals = np.arange(-self.kmax, self.kmax + 1)
        ky_vals = np.arange(-self.kmax, self.kmax + 1)
        kz_vals = np.arange(-self.kmax, self.kmax + 1)

        energies = []
        kmags = []

        x = SpatialCoordinate(self.mesh)

        for kx_i in kx_vals:
            for ky_i in ky_vals:
                for kz_i in kz_vals:

                    if kx_i == 0 and ky_i == 0 and kz_i == 0:
                        continue

                    kx = 2.0 * np.pi * kx_i / Lx
                    ky = 2.0 * np.pi * ky_i / Ly
                    kz = 2.0 * np.pi * kz_i / Lz

                    theta = kx * x[0] + ky * x[1] + kz * x[2]

                    cos_phase = cos(theta)
                    sin_phase = sin(theta)

                    # Fourier coefficients (real + imaginary)
                    uhat_x_real = assemble(u_fluct[0] * cos_phase * dxm) / area
                    uhat_x_imag = -assemble(u_fluct[0] * sin_phase * dxm) / area

                    uhat_y_real = assemble(u_fluct[1] * cos_phase * dxm) / area
                    uhat_y_imag = -assemble(u_fluct[1] * sin_phase * dxm) / area

                    uhat_z_real = assemble(u_fluct[2] * cos_phase * dxm) / area
                    uhat_z_imag = -assemble(u_fluct[2] * sin_phase * dxm) / area

                    energy = 0.5 * (
                        uhat_x_real**2 + uhat_x_imag**2 +
                        uhat_y_real**2 + uhat_y_imag**2 +
                        uhat_z_real**2 + uhat_z_imag**2
                    )

                    energies.append(energy)
                    kmags.append(np.sqrt(kx**2 + ky**2 + kz**2))

        energies = np.array(energies)
        kmags = np.array(kmags)

        # Only rank 0 does binning
        if comm.rank != 0:
            return None, None

        # ---------------
        # Shell averaging
        # ---------------
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
        # E /= counts # kolmogorov energy spectrum based on sums not avgs
        
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        return k_centers, E