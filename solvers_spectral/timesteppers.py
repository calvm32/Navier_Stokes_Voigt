import time
from pathlib import Path
import numpy as np

from processing.printoff import iter_info_verbose, text, green
from processing.statistics.pdf_sampler import pdf_sampler
from processing.statistics.structure_funcs import structure_funcs
from processing.post_processing import *

def timestepper_RK4(rhs, u_hat_0, f_hat, t0, T, dt, ksq):
    """
    Solve the ODE y' = f(t,y) on the interval [t0,T] with y(t0) = y0
    using the Runge-Kutta-4 approximation method 
    """
    
    N = int(np.floor((T - t0) / dt) + 1) # fixed number of steps
    t = t0+dt*np.arange(N)

    u_hat = np.zeros(u_hat_0.shape + (N,), dtype=complex) # append time axis N
    u_hat[..., 0] = u_hat_0  # Set initial Fourier coefficients

    for n in range(0, N - 1):
        k1 = rhs(u_hat[...,n], f_hat, ksq)
        k2 = rhs(u_hat[...,n] + dt/2 * k1, f_hat, ksq)
        k3 = rhs(u_hat[...,n] + dt/2 * k2, f_hat, ksq)
        k4 = rhs(u_hat[...,n] + dt * k3, f_hat, ksq)

        u_hat[...,n + 1] = u_hat[...,n] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return u_hat, t


def timestepper_intfactor_RK4(rhs, u_hat_0, f_hat, t0, T, dt, ksq, Re):
    """
    Solve the ODE y' = f(t,y) on the interval [t0,T] with y(t0) = y0
    using the Runge-Kutta-4 approximation method,
    specifically with integrating factor of e^power applied
    """
    
    N = int(np.floor((T - t0) / dt) + 1) # fixed number of steps
    t = t0+dt*np.arange(N)

    u_hat = np.zeros(u_hat_0.shape + (N,), dtype=complex) # append time axis N
    u_hat[..., 0] = u_hat_0  # Set initial Fourier coefficients

    # integrating factor
    E = np.exp(-1*(ksq/Re)*dt)

    for n in range(0, N - 1):
        k1 = rhs(u_hat[...,n], f_hat, ksq)
        k2 = rhs(np.sqrt(E)*(u_hat[...,n] + dt/2 * k1), f_hat, ksq)
        k3 = rhs(np.sqrt(E)*(u_hat[...,n] + dt/2 * k2), f_hat, ksq)
        k4 = rhs(np.sqrt(E)*(u_hat[...,n] + dt * k3), f_hat, ksq)

        u_hat[...,n + 1] = E*u_hat[...,n] + E*(dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return u_hat, t