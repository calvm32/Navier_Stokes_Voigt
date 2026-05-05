import time
from pathlib import Path
import numpy as np

from navier_stokes_voigt.processing.printoff import iter_info_verbose, text, green
from navier_stokes_voigt.processing.statistics.pdf_sampler import pdf_sampler
from navier_stokes_voigt.processing.statistics.structure_funcs import structure_funcs
from navier_stokes_voigt.processing.post_processing import *

def timestepper_RK4(rhs, u_hat_0, f_hat_func, t0, T, dt):
    """
    Solve the ODE y' = f(t,y) on the interval [t0,T] with y(t0) = y0
    using the Runge-Kutta-4 approximation method 
    """
    
    N = int(np.floor((T - t0) / dt) + 1) # fixed number of steps
    times = t0+dt*np.arange(N)

    u_hat = np.zeros(u_hat_0.shape + (N,), dtype=complex) # append time axis N
    u_hat[..., 0] = u_hat_0  # Set initial Fourier coefficients

    for n in range(0, N - 1):
        t = times[n]
        k1 = rhs(u_hat[...,n], f_hat_func(t))
        k2 = rhs(u_hat[...,n] + dt/2 * k1, f_hat_func(t+dt/2))
        k3 = rhs(u_hat[...,n] + dt/2 * k2, f_hat_func(t+dt/2))
        k4 = rhs(u_hat[...,n] + dt * k3, f_hat_func(t+dt))

        u_hat[...,n + 1] = u_hat[...,n] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return u_hat, times


def timestepper_intfactor_RK4(rhs, u_hat_0, f_hat_func, t0, T, dt, ksq, Re=0):
    """
    Solve the ODE y' = f(t,y) on the interval [t0,T] with y(t0) = y0
    using the Runge-Kutta-4 approximation method,
    specifically with integrating factor of e^power applied
    """
    
    N = int(np.floor((T - t0) / dt) + 1) # fixed number of steps
    times = t0+dt*np.arange(N)

    u_hat = np.zeros(u_hat_0.shape + (N,), dtype=complex) # append time axis N
    u_hat[..., 0] = u_hat_0  # Set initial Fourier coefficients

    # integrating factor
    Efull = np.exp(-1*(ksq/Re)*dt)
    Ehalf = np.exp(-1*(ksq/Re)*dt / 2)

    for n in range(0, N - 1):
        t = times[n]
        k1 = rhs(u_hat[...,n], f_hat_func(t))
        k2 = rhs(Ehalf*(u_hat[...,n] + dt/2 * k1), f_hat_func(t+dt/2))
        k3 = rhs(Ehalf*(u_hat[...,n]) + dt/2 * k2, f_hat_func(t+dt/2))
        k4 = rhs(Efull*(u_hat[...,n]+ dt * k3), f_hat_func(t+dt))

        u_hat[...,n + 1] = Efull*u_hat[...,n] + (dt/6)*(Efull*k1 + Ehalf*2*(k2 + k3) + k4)

    return u_hat, times

def timestepper_intfactor_compare_RK4(rhs1, rhs2, u_hat_0, f_hat_func, t0, T, dt, ksq, Re):
    """
    Solve the ODE y' = f(t,y) on the interval [t0,T] with y(t0) = y0
    using the Runge-Kutta-4 approximation method,
    specifically with integrating factor of e^power applied
    """
    
    N = int(np.floor((T - t0) / dt) + 1) # fixed number of steps
    times = t0+dt*np.arange(N)

    u_hat_diff = np.zeros(u_hat_0.shape + (N,), dtype=complex) # append time axis N
    u_hat1 = np.zeros(u_hat_0.shape + (N,), dtype=complex) # append time axis N
    u_hat2 = np.zeros(u_hat_0.shape + (N,), dtype=complex) # append time axis N

    u_hat1[..., 0] = u_hat_0  # Set initial Fourier coefficients
    u_hat2[..., 0] = u_hat_0  # Set initial Fourier coefficients

    # integrating factor
    Efull = np.exp(-1*(ksq/Re)*dt)
    Ehalf = np.exp(-1*(ksq/Re)*dt / 2)

    for n in range(0, N - 1):
        t = times[n]
        
        k1 = rhs1(u_hat1[...,n], f_hat_func(t))
        k2 = rhs1(Ehalf*(u_hat1[...,n] + dt/2 * k1), f_hat_func(t+dt/2))
        k3 = rhs1(Ehalf*(u_hat1[...,n]) + dt/2 * k2, f_hat_func(t+dt/2))
        k4 = rhs1(Efull*(u_hat1[...,n] + dt * k3), f_hat_func(t+dt))

        u_hat1[...,n + 1] = Efull*u_hat1[...,n] + (dt/6)*(Efull*k1 + Efull*2*(k2 + k3) + k4)

        k1 = rhs2(u_hat2[...,n], f_hat_func(t))
        k2 = rhs2(Ehalf*(u_hat2[...,n] + dt/2 * k1), f_hat_func(t+dt/2))
        k3 = rhs2(Ehalf*(u_hat2[...,n]) + dt/2 * k2, f_hat_func(t+dt/2))
        k4 = rhs2(Efull*(u_hat2[...,n] + dt * k3), f_hat_func(t+dt))

        u_hat2[...,n + 1] = Efull*u_hat2[...,n] + (dt/6)*(Efull*k1 + Efull*2*(k2 + k3) + k4)

        u_hat_diff[...,n + 1] = u_hat1[...,n + 1] - u_hat2[...,n + 1]

    return u_hat_diff, times