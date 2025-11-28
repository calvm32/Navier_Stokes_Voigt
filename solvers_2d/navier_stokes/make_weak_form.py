from firedrake import *

def make_weak_form(theta, idt, f_n, f_np1, g_n, g_np1, dsN):
    """
    Returns func F(u, u_old, p, q, v), 
    which builds weak form
    using external coefficients
    """

    def F(u, p, u_old, p_old, v, q):
        u_mid = theta * u + (1 - theta) * u_old

        return (
            idt * inner(u - u_old, v) * dx
            + 1.0 / Re * inner(grad(u_mid), grad(v)) * dx +
            inner(dot(grad(u_mid), u_mid), v) * dx -
            p * div(v) * dx +
            div(u_mid) * q * dx
            - inner((theta * f_np1 + (1 - theta) * f_n), v) * dx
        )

    return F