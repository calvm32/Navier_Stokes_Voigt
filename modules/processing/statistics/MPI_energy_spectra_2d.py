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
        print("1.1")
        self.mesh = u.function_space().mesh()
        print("1.2")
        self.nbins = nbins

    def compute(self):

        print("0")

        comm = self.mesh.comm
        mesh = self.mesh
        dx = Measure("dx", domain=self.mesh)

        print(f"rank {comm.rank}: entered compute", flush=True)

        print("comm =", comm, flush=True)
        print("rank =", comm.rank, flush=True)
        print("size =", comm.size, flush=True)

        # remove mean flow
        area = assemble(Constant(1.0) * dx)

        print(comm.rank, "0.1", flush=True)
        
        self.u.sub(0).dat.vec.ghostUpdate()
        self.u.sub(1).dat.vec.ghostUpdate()

        u0_local = self.u.sub(0).dat.data_ro.mean()
        u1_local = self.u.sub(1).dat.data_ro.mean()

        u0 = comm.allreduce(u0_local, op=MPI.SUM) / comm.size
        u1 = comm.allreduce(u1_local, op=MPI.SUM) / comm.size

        u_fluc = as_vector([self.u[0] - u0, self.u[1] - u1])

        print("0.2")

        # domain size
        if hasattr(mesh, 'coordinates'):
            coords = mesh.coordinates.dat.data_ro
        elif hasattr(mesh, 'meshes'):
            coords = mesh.meshes[0].coordinates.dat.data_ro
        else:
            raise AttributeError(f"Cannot extract coordinates from mesh of type {type(mesh)}")

        print("1")

        xmin = comm.allreduce(coords[:,0].min(), op=MPI.MIN)
        xmax = comm.allreduce(coords[:,0].max(), op=MPI.MAX)
        ymin = comm.allreduce(coords[:,1].min(), op=MPI.MIN)
        ymax = comm.allreduce(coords[:,1].max(), op=MPI.MAX)

        print("2")

        Lx, Ly = xmax-xmin, ymax-ymin

        # k-space resolution tied to mesh
        hmin = comm.allreduce(np.min(np.linalg.norm(np.diff(coords, axis=0), axis=1)), op=MPI.MIN)
        kmax = np.pi / max(hmin, 1e-12)

        print("3")

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

        print("4")

        # -------------
        # MPI reduction
        # -------------
        E_global = np.zeros_like(E_local)
        count_global = np.zeros_like(counts)

        comm.Allreduce(E_local, E_global, op=MPI.SUM)
        comm.Allreduce(counts, count_global, op=MPI.SUM)

        print("5")

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