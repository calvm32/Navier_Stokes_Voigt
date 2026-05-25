import numpy as np
from mpi4py import MPI


class structure_funcs_2d:

    def __init__(
        self,
        u,
        mesh,
        r_max,
        nbins=32,
        seed=0,
    ):

        self.comm = mesh.comm

        self.coords = mesh.coordinates.dat.data_ro
        self.values = u.dat.data_ro

        self.ndofs = self.coords.shape[0]

        self.r_max = r_max

        self.r_edges = np.geomspace(
            r_max / 500.0,
            r_max,
            nbins + 1
        )

        self.r_centers = np.sqrt(
            self.r_edges[:-1]
            * self.r_edges[1:]
        )

        # GLOBAL synchronized accumulators
        self.S2 = np.zeros(nbins)
        self.counts = np.zeros(nbins)

        self.rng = np.random.default_rng(
            seed + self.comm.rank
        )

    def sample(self, nsamples=20000):

        rng = self.rng

        # -------------------------
        # local vectorized sampling
        # -------------------------

        i = rng.integers(
            0,
            self.ndofs,
            size=nsamples
        )

        j = rng.integers(
            0,
            self.ndofs,
            size=nsamples
        )

        mask = i != j

        i = i[mask]
        j = j[mask]

        xi = self.coords[i]
        xj = self.coords[j]

        dx = xj - xi

        r = np.linalg.norm(dx, axis=1)

        mask = (
            (r > 1e-14)
            & (r < self.r_max)
        )

        if not np.any(mask):
            return

        i = i[mask]
        j = j[mask]
        dx = dx[mask]
        r = r[mask]

        rhat = dx / r[:, None]

        ui = self.values[i]
        uj = self.values[j]

        du = uj - ui

        du_long = np.sum(
            du * rhat,
            axis=1
        )

        vals = du_long**2

        bins = np.searchsorted(
            self.r_edges,
            r,
            side="right"
        ) - 1

        valid = (
            (bins >= 0)
            & (bins < len(self.S2))
        )

        bins = bins[valid]
        vals = vals[valid]

        # ----------------------
        # local temporary arrays
        # ----------------------

        local_S2 = np.zeros_like(self.S2)
        local_counts = np.zeros_like(self.counts)

        np.add.at(local_S2, bins, vals)
        np.add.at(local_counts, bins, 1)

        # immediate synchronization
        global_S2 = np.zeros_like(local_S2)
        global_counts = np.zeros_like(local_counts)

        self.comm.Allreduce(
            local_S2,
            global_S2,
            op=MPI.SUM
        )

        self.comm.Allreduce(
            local_counts,
            global_counts,
            op=MPI.SUM
        )

        # accumulate globally synchronized values
        self.S2 += global_S2
        self.counts += global_counts

    def compute(self):

        out = np.zeros_like(self.S2)

        mask = self.counts > 0

        out[mask] = (
            self.S2[mask]
            / self.counts[mask]
        )

        return self.r_centers, out