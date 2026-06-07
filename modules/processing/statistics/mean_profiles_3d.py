import numpy as np
from firedrake import *
from mpi4py import MPI

class mean_profiles_3d:
    """
    Used for Log Law of the Wall, which does't exist in 2D,
    so this isn't currently used

    untested & may have errors
    """
    def __init__(self, V, nu, dsN, wall_id):
        self.V = V
        self.mesh = V.mesh()
        self.Re = Re
        self.dsN = dsN
        self.wall_id = wall_id

        self.u_sum = Function(V)
        self.num_samples = 0

    def sample(self, u):
        self.u_sum += u
        self.num_samples += 1

    def finalize(self, nbins=100, H=1.0):
        if self.num_samples == 0:
            raise RuntimeError("mean_profiles: no samples collected")

        u_mean = Function(self.V)
        u_mean.assign(self.u_sum / self.num_samples)

        # -----------------
        # wall shear stress
        # -----------------
        tau_w = nu * assemble(
            grad(u_mean)[0, 1] * self.dsN(self.wall_id)
        ) / assemble(
            1.0 * self.dsN(self.wall_id)
        )

        u_tau = np.sqrt(abs(tau_w))

        # -------------------
        # wall-normal binning
        # -------------------
        if hasattr(mesh, 'coordinates'):
            coords = mesh.coordinates.dat.data_ro
        elif hasattr(mesh, 'meshes'):
            coords = mesh.meshes[0].coordinates.dat.data_ro
        else:
            raise AttributeError(f"Cannot extract coordinates from mesh of type {type(mesh)}")
        yvals = coords[:, 1]
        uvals = u_mean.dat.data[:, 0]

        bins = np.linspace(0, H, nbins + 1)
        comm = self.mesh.comm
        u_bin = comm.allreduce(u_bin, op=MPI.SUM)
        count = comm.allreduce(count, op=MPI.SUM)

        for y, u in zip(yvals, uvals):
            j = np.searchsorted(bins, y) - 1
            if 0 <= j < nbins:
                u_bin[j] += u
                count[j] += 1

        u_bin /= np.maximum(count, 1)
        y_bin = 0.5 * (bins[:-1] + bins[1:])

        u_plus = u_bin / u_tau
        y_plus = y_bin * u_tau / nu

        return y_plus, u_plus
