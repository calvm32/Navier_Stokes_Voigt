from firedrake import *

def make_weak_form(theta, idt, f, f_old, g, g_old, dx, dsN, Re=1.0):
    """
    Weak form for Navier-Stokes equations using CN
    """

    def F(u, u_old, v):
        u_new, p_new = split(u) 
        v_u, v_p = split(v)

        u_old_v = u_old.sub(0)
        u_mid = theta*u_new + (1-theta)*u_old_v
        f_mid = theta*f.sub(0) + (1-theta)*f_old.sub(0)

        # Momentum equation
        F_mom = (
            idt*inner(u_new - u_old_v, v_u)*dx
            + inner(grad(u_mid)*u_mid, v_u)*dx
            + (1.0/Re)*inner(grad(u_mid), grad(v_u))*dx
            - div(v_u)*p_new*dx
            - inner(f_mid, v_u)*dx
        )

        # Continuity equation
        F_cont = -v_p*div(u_mid)*dx

        return F_mom + F_cont

    return F
