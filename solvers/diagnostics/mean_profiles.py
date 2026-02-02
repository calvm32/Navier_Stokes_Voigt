# solvers/diagnostics/mean_profiles.py
import numpy as np
from firedrake import *

class MeanProfiles:
    def __init__(self, V, Re, dsN, wall_id):
        """
        V       : velocity FunctionSpace
        Re      : Reynolds number
        dsN     : boundary measure (already bound to mesh)
        wall_id : boundary id
        """
        self.V = V
        self.mesh = V.mesh()
        self.Re = Re
        self.dsN = dsN
        self.wall_id = wall_id

        self.u_sum = Function(V)
        self.nsamples = 0

    def sample(self, u):
        self.u_sum += u
        self.nsamples += 1

    def finalize(self, nbins=100, H=1.0):
        if self.nsamples == 0:
            raise RuntimeError("MeanProfiles: no samples collected")

        u_mean = Function(self.V)
        u_mean.assign(self.u_sum / self.nsamples)

        # ---- wall shear stress (correct) ----
        nu = 1.0 / self.Re
        tau_w = nu * assemble(
            grad(u_mean)[0, 1] * self.dsN(self.wall_id)
        ) / assemble(
            1.0 * self.dsN(self.wall_id)
        )

        u_tau = np.sqrt(abs(tau_w))

        # ---- wall-normal binning ----
        coords = self.mesh.coordinates.dat.data
        yvals = coords[:, 1]
        uvals = u_mean.dat.data[:, 0]

        bins = np.linspace(0, H, nbins + 1)
        u_bin = np.zeros(nbins)
        count = np.zeros(nbins)

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
