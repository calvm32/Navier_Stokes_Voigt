from firedrake import *

def make_weak_form(theta, idt, f, f_old, g, g_old, dx , dsN):
    """
    Weak form for heat equation using CN
    """
    
    f_mid = theta * f + (1-theta) * f_old
    g_mid = theta * g + (1-theta) * g_old

    def F(u_new, u_old, v):
        u_mid = theta*u_new + (1-theta)*u_old

        return ( idt*(u_new - u_old)*v*dx
            + inner(grad(u_mid), grad(v))*dx
            - f_mid*v*dx - g_mid*v*dsN
        )

    return F