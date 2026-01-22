from firedrake import *

def make_weak_form(idt, f, f_old, g, g_old, u_old, dx, dsN):
    """
    Bilinear and linear forms for heat equation
      -> Crank-Nicolson
      -> bilinear, linear
    """
    
    # Midpoints
    f_mid = theta*f + (1.0 - theta)*f_old
    g_mid = theta*g + (1.0 - theta)*g_old

    def forms(u, v):
        
        # Bilinear form a(u,v)
        a = (
            idt * u * v * dx
            + theta * inner(grad(u), grad(v)) * dx
        )

        # Linear form L(v)
        L = (
            idt * u_old * v * dx
            - (1.0 - theta) * inner(grad(u_old), grad(v)) * dx
            + f_mid * v * dx
            + g_mid * v * dsN
        )

        return a, L

    return forms
