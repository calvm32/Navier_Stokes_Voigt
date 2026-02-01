from firedrake import *
import yaml
from pathlib import Path

from solvers.config_setup import *

# -------------
# Configuration
# -------------

CFG_PATH1 = Path(__file__).parent / "configs" / "USER_constants.yaml"
cfg = load_config(CFG_PATH1)

theta = cfg["theta"]

# ---------
# Weak form
# ---------

def make_weak_form(idt, f, g, u_old, u_older, dx, dsN):
    """
    Bilinear and linear forms for heat equation
      -> BDF2 time stepping
      -> bilinear, linear
    """

    def forms(u, v):

        # -------------
        # Bilinear form
        # -------------
        a = (
            (3.0/2.0) * idt * u * v * dx
            + inner(grad(u), grad(v)) * dx
        )

        # -----------
        # Linear form
        # -----------
        L = (
            idt * (2.0 * u_old - 0.5 * u_older) * v * dx
            + f * v * dx
            + g * v * dsN
        )

        return a, L

    return forms