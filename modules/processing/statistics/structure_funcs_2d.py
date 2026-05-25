import numpy as np
from mpi4py import MPI


class structure_funcs_2d:
    """
    Monte-Carlo estimate of the longitudinal 2nd-order
    structure function:

        S2(r) = < ((u(x+r)-u(x)) · rhat)^2 >

    for homogeneous/isotropic turbulence.

    MPI-safe and scalable.
    """

    def __init__(
        self,
        u,
        mesh,
        r_max=None,
        nbins=40,
        seed=None,
    ):

        self.u = u
        self.mesh = mesh
        self.comm = mesh.comm

        rng = np.random.default_rng(seed)
        self.rng = rng

        # --------------------------------
        # coordinates + local velocity DOFs
        # --------------------------------

        coords = mesh.coordinates.dat.data_ro
        values = u.dat.data_ro

        self.coords = coords
        self.values = values

        self.ndofs = coords.shape[0]

        # ------------------------
        # radial domain
        # ------------------------

        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)

        Lx = xmax - xmin
        Ly = ymax - ymin

        if r_max is None:
            r_max = 0.25 * min(Lx, Ly)

        self.r_max = r_max

        self.r_edges = np.linspace(0.0, r_max, nbins + 1)
        self.r_centers = 0.5 * (
            self.r_edges[:-1] + self.r_edges[1:]
        )

        # accumulators
        self.S2_accum = np.zeros(nbins, dtype=np.float64)
        self.counts = np.zeros(nbins, dtype=np.int64)

    def sample(self, nsamples=10000):
        """
        Draw random DOF pairs and accumulate longitudinal increments.
        """

        coords = self.coords
        values = self.values

        for _ in range(nsamples):

            # random pair
            i = self.rng.integers(0, self.ndofs)
            j = self.rng.integers(0, self.ndofs)

            if i == j:
                continue

            xi = coords[i]
            xj = coords[j]

            dx = xj - xi

            r = np.linalg.norm(dx)

            # avoid singular pair
            if r <= 1e-14:
                continue

            # outside target range
            if r >= self.r_max:
                continue

            # longitudinal direction
            rhat = dx / r

            ui = values[i]
            uj = values[j]

            du = uj - ui

            # longitudinal increment
            du_long = np.dot(du, rhat)

            # radial bin
            bin_idx = np.searchsorted(
                self.r_edges,
                r,
                side="right"
            ) - 1

            if 0 <= bin_idx < len(self.S2_accum):

                self.S2_accum[bin_idx] += du_long**2
                self.counts[bin_idx] += 1

    def finalize(self):
        """
        MPI-safe reduction and averaging.
        Must be called on ALL ranks.
        """

        S2_global = self.comm.allreduce(
            self.S2_accum,
            op=MPI.SUM
        )

        counts_global = self.comm.allreduce(
            self.counts,
            op=MPI.SUM
        )

        S2 = np.zeros_like(S2_global)

        mask = counts_global > 0

        S2[mask] = (
            S2_global[mask]
            / counts_global[mask]
        )

        return (
            self.r_centers.copy(),
            S2,
            counts_global.copy(),
        )