from mpi4py import MPI
from firedrake import *
import numpy as np

class MPI_energy_spectra_2d:
    """
    MPI-safe kinetic energy spectrum using distributed quadrature
    (no global grid, no .at(), no FFT over full domain).
    """

    def __init__(self, u, nbins=50):
        self.u = u
        self.mesh = u.function_space().mesh()
        self.nbins = nbins

    def compute(self):

        comm = self.mesh.comm
        dx = Measure("dx", domain=self.mesh)

        # remove mean flow
        area = assemble(Constant(1.0) * dx)

        u0 = assemble(self.u[0] * dx) / area
        u1 = assemble(self.u[1] * dx) / area

        u_fluc = as_vector([self.u[0] - u0,
                            self.u[1] - u1])

        # domain size (MPI-safe)
        coords = self.mesh.coordinates.dat.data_ro

        xmin = comm.allreduce(coords[:,0].min(), op=MPI.MIN)
        xmax = comm.allreduce(coords[:,0].max(), op=MPI.MAX)
        ymin = comm.allreduce(coords[:,1].min(), op=MPI.MIN)
        ymax = comm.allreduce(coords[:,1].max(), op=MPI.MAX)

        Lx, Ly = xmax-xmin, ymax-ymin

        # k-space resolution tied to mesh
        hmin = comm.allreduce(np.min(np.linalg.norm(np.diff(coords, axis=0), axis=1)), op=MPI.MIN)
        kmax = np.pi / max(hmin, 1e-12)

        nb = self.nbins
        bins = np.logspace(np.log10(1.0), np.log10(kmax), nb+1)

        E_local = np.zeros(nb)
        counts = np.zeros(nb)

        x = SpatialCoordinate(self.mesh)

        # -------------------------------
        # distributed spectral projection
        # -------------------------------
        for i in range(nb*3):  # oversampling shells
            for j in range(nb*3):

                kx = 2*np.pi * (i - nb*1.5) / Lx
                ky = 2*np.pi * (j - nb*1.5) / Ly

                if kx == 0 and ky == 0:
                    continue

                k = np.sqrt(kx**2 + ky**2)

                phase = kx*x[0] + ky*x[1]

                cos_p = cos(phase)
                sin_p = sin(phase)

                ux_hat = assemble(u_fluc[0]*cos_p*dx) - 1j*assemble(u_fluc[0]*sin_p*dx)
                uy_hat = assemble(u_fluc[1]*cos_p*dx) - 1j*assemble(u_fluc[1]*sin_p*dx)

                E = 0.5*(abs(ux_hat)**2 + abs(uy_hat)**2)

                b = np.searchsorted(bins, k) - 1

                if 0 <= b < nb:
                    E_local[b] += E
                    counts[b] += 1

        # -------------
        # MPI reduction
        # -------------
        E_global = np.zeros_like(E_local)
        count_global = np.zeros_like(counts)

        comm.Allreduce(E_local, E_global, op=MPI.SUM)
        comm.Allreduce(counts, count_global, op=MPI.SUM)

        count_global[count_global == 0] = 1

        dk = np.diff(bins)

        E_global /= count_global
        E_global /= dk

        k_centers = 0.5*(bins[:-1] + bins[1:])

        energy_phys = assemble(0.5*inner(u_fluc,u_fluc)*dx)
        energy_spec = np.sum(E_k*np.diff(k_vals))

        print("Parseval error:",
            abs(energy_phys-energy_spec)/energy_phys)

        return k_centers, E_global