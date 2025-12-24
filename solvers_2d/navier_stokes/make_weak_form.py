from firedrake import *

def make_weak_form(theta, idt, f_new, f_old, g_new, g_old, dx, dsN, Re=1.0):
    """
    Weak form for Navier-Stokes equations using CN
    """

    def F(u, u_old, v):
        u_new, p_new = split(u)   # u is the trial function / Function (will be replaced by u_new in solver)
        v_u, v_p = split(v)       # v is the TestFunction

        # Velocity midpoint for theta scheme
        u_old_v = u_old.sub(0)
        u_mid = theta*u_new + (1-theta)*u_old_v

        # Source term midpoint
        f_mid = theta*f_new.sub(0) + (1-theta)*f_old.sub(0)

        nu = Constant(1.0/Re)

        # Momentum equation
        F_mom = (
            idt*inner(u_new - u_old_v, v_u)*dx
            + inner(grad(u_mid)*u_mid, v_u)*dx
            + nu*inner(grad(u_mid), grad(v_u))*dx
            - div(v_u)*p_new*dx
            - inner(f_mid, v_u)*dx
        )

        # Continuity equation
        F_cont = -v_p*div(u_mid)*dx

        return F_mom + F_cont

    return F
