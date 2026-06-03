from firedrake import *
from mpi4py import MPI
import numpy as np


class FFT_energy_spectra_2d:

    """
    FFT-based isotropic kinetic energy spectrum for a
    distributed Firedrake velocity field.

    Uses VertexOnlyMesh sampling instead of repeated
    point evaluation.

    Usage
    -----

    spectrum = FFT_energy_spectra_2d(
        u_old.sub(0),
        Nx=256,
        Ny=256,
        nbins=80
    )

    k, E = spectrum.compute()

    if mesh.comm.rank == 0:
        plt.loglog(k, E)
    """

    def __init__(
        self,
        u,
        Nx=256,
        Ny=256,
        nbins=80
    ):

        self.u = u
        self.mesh = u.function_space().mesh()

        self.Nx = Nx
        self.Ny = Ny
        self.nbins = nbins

    def compute(self):

        comm = self.mesh.comm

        coords = self.mesh.coordinates.dat.data_ro

        xmin = comm.allreduce(
            np.min(coords[:, 0]),
            op=MPI.MIN
        )

        xmax = comm.allreduce(
            np.max(coords[:, 0]),
            op=MPI.MAX
        )

        ymin = comm.allreduce(
            np.min(coords[:, 1]),
            op=MPI.MIN
        )

        ymax = comm.allreduce(
            np.max(coords[:, 1]),
            op=MPI.MAX
        )

        Lx = xmax - xmin
        Ly = ymax - ymin

        # ------------------------------------
        # Cartesian sampling grid
        # ------------------------------------

        x = np.linspace(xmin, xmax, self.Nx)
        y = np.linspace(ymin, ymax, self.Ny)

        X, Y = np.meshgrid(x, y)

        pts = np.column_stack(
            [X.ravel(), Y.ravel()]
        )
        
        print("requested:", pts.shape)
        print("actual:", vom.coordinates.dat.data_ro.shape)

        # ------------------------------------
        # VertexOnlyMesh sampling
        # ------------------------------------

        vom = VertexOnlyMesh(
            self.mesh,
            pts,
            redundant=True,
            missing_points_behaviour=MissingPointsBehaviour.IGNORE
        )

        Vvom = VectorFunctionSpace(
            vom,
            "DG",
            0
        )

        u_sample = Function(Vvom)

        u_sample.interpolate(self.u)

        vals = u_sample.dat.data_ro

        # vals.shape should be (Nx*Ny, 2)

        ux = vals[:, 0].reshape(
            self.Ny,
            self.Nx
        )

        uy = vals[:, 1].reshape(
            self.Ny,
            self.Nx
        )

        # ------------------------------------
        # Only rank 0 proceeds
        # ------------------------------------

        if comm.rank != 0:
            return None, None

        # ------------------------------------
        # Remove mean
        # ------------------------------------

        mask = (
            np.isfinite(ux)
            &
            np.isfinite(uy)
        )

        ux[~mask] = 0.0
        uy[~mask] = 0.0

        ux -= np.mean(ux[mask])
        uy -= np.mean(uy[mask])

        # ------------------------------------
        # FFT
        # ------------------------------------

        dx = Lx / (self.Nx - 1)
        dy = Ly / (self.Ny - 1)

        ux_hat = np.fft.fft2(ux)
        uy_hat = np.fft.fft2(uy)

        # Parseval-consistent normalization

        ux_hat *= dx * dy
        uy_hat *= dx * dy

        # ------------------------------------
        # Wavenumbers
        # ------------------------------------

        kx = 2.0 * np.pi * np.fft.fftfreq(
            self.Nx,
            d=dx
        )

        ky = 2.0 * np.pi * np.fft.fftfreq(
            self.Ny,
            d=dy
        )

        KX, KY = np.meshgrid(
            kx,
            ky
        )

        kmag = np.sqrt(
            KX**2 +
            KY**2
        )

        # ------------------------------------
        # Modal energy
        # ------------------------------------

        E2D = 0.5 * (
            np.abs(ux_hat)**2 +
            np.abs(uy_hat)**2
        )

        # ------------------------------------
        # Isotropic shell averaging
        # ------------------------------------

        kmax = np.max(kmag)

        bins = np.linspace(
            0.0,
            kmax,
            self.nbins + 1
        )

        dk = bins[1] - bins[0]

        E = np.zeros(self.nbins)

        for i in range(self.nbins):

            shell = (
                (kmag >= bins[i])
                &
                (kmag < bins[i + 1])
            )

            if np.any(shell):
                E[i] = np.sum(E2D[shell])

        E /= dk

        k = 0.5 * (
            bins[:-1] +
            bins[1:]
        )

        # ------------------------------------
        # Parseval diagnostic
        # ------------------------------------

        E_phys = (
            0.5 *
            np.mean(
                ux**2 +
                uy**2
            ) *
            Lx *
            Ly
        )

        E_spec = np.sum(E * dk)

        relerr = abs(
            E_phys - E_spec
        ) / max(E_phys, 1e-15)

        print(
            f"Parseval relative error = {relerr:.3e}"
        )

        return k, E