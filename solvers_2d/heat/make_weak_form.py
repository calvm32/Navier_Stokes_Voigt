from firedrake import *

def make_weak_form(theta, idt, f_n, f_np1, g_n, g_np1, dsN):
    """
    Returns func F(u, u_old, v), which builds weak form
    using external coefficients
    """

    def F(u, u_old, v, *args):
        return (
            idt * (u - u_old) * v * dx
            + inner(grad(theta * u + (1 - theta) * u_old), grad(v)) * dx
            - (theta * f_np1 + (1 - theta) * f_n) * v * dx
            - (theta * g_np1 + (1 - theta) * g_n) * v * dsN
        )

    return F