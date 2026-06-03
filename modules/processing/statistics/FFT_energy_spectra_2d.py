from firedrake import *
from mpi4py import MPI
import numpy as np
from firedrake import PointNotInDomainError


class FFT_energy_spectra_2d:
    """
    MPI-safe FFT energy spectrum for Firedrake fields on arbitrary (nonperiodic) meshes.

    Key idea:
    - Build structured Cartesian grid on rank 0
    - Evaluate solution safely via distributed-aware point evaluation
    - Avoid VertexOnlyMesh (not suitable for FFT grids on cut domains)
    """

    def __init__(self, u, Nx=256, Ny=256, nbins=80):
        self.u = u
        self.mesh = u.function_space().mesh()
        self.Nx = Nx
        self.Ny = Ny
        self.nbins = nbins

    def compute(self):

        mesh = self.mesh
        comm = mesh.comm

        # compute domain bounds
        coords = mesh.coordinates.dat.data_ro

        xmin = comm.allreduce(coords[:, 0].min(), op=MPI.MIN)
        xmax = comm.allreduce(coords[:, 0].max(), op=MPI.MAX)
        ymin = comm.allreduce(coords[:, 1].min(), op=MPI.MIN)
        ymax = comm.allreduce(coords[:, 1].max(), op=MPI.MAX)

        Lx = xmax - xmin
        Ly = ymax - ymin

        print("0")

        # ONLY rank 0 builds sampling grid
        if comm.rank != 0:
            return None, None

        x = np.linspace(xmin, xmax, self.Nx)
        y = np.linspace(ymin, ymax, self.Ny)

        X, Y = np.meshgrid(x, y)

        pts = np.column_stack([X.ravel(), Y.ravel()])

        print("1")

        # safe interpolation
        ux = np.full(len(pts), np.nan)
        uy = np.full(len(pts), np.nan)
        
        print("starting evaluation", flush=True)

        fft_mesh = RectangleMesh(
            self.Nx - 1, self.Ny - 1,
            Lx, Ly,
            origin=(xmin, ymin),
            comm=COMM_SELF
        )

        Vfft = VectorFunctionSpace(fft_mesh, "CG", 1)

        print("1.1")

        u_fft = Function(Vfft)
        u_fft.interpolate(self.u)

        print("1.2")

        vals = u_fft.dat.data_ro

        ux = vals[:, 0].reshape(self.Ny, self.Nx)
        uy = vals[:, 1].reshape(self.Ny, self.Nx)

        print("2")

        # reshape ONLY after masking validity
        mask = np.isfinite(ux) & np.isfinite(uy)

        ux[~mask] = 0.0
        uy[~mask] = 0.0

        ux -= np.mean(ux[mask])
        uy -= np.mean(uy[mask])

        print("3")

        # FFT
        dx = Lx / (self.Nx - 1)
        dy = Ly / (self.Ny - 1)

        ux_hat = np.fft.fft2(ux) * dx * dy
        uy_hat = np.fft.fft2(uy) * dx * dy

        # wavenumbers
        kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=dy)

        KX, KY = np.meshgrid(kx, ky)

        kmag = np.sqrt(KX**2 + KY**2)

        print("4")

        # spectral energy
        E2D = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2)

        kmax = kmag.max()
        bins = np.linspace(0.0, kmax, self.nbins + 1)

        E = np.zeros(self.nbins)

        for i in range(self.nbins):
            sel = (kmag >= bins[i]) & (kmag < bins[i + 1])
            E[i] = np.sum(E2D[sel])

        print("5")

        dk = bins[1] - bins[0]
        E /= dk

        k = 0.5 * (bins[:-1] + bins[1:])

        # Parseval check
        E_phys = 0.5 * np.mean(ux**2 + uy**2) * Lx * Ly
        E_spec = np.sum(E * dk)

        print("6")

        print("Parseval error:",
              abs(E_phys - E_spec) / max(E_phys, 1e-15))

        return k, E