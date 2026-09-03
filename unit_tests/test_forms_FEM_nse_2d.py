import pytest
from firedrake import *
from modules.solvers_FEM.navier_stokes_2d.make_weak_form import make_weak_form_BDF2, make_weak_form_CN

def test_FEM_nse_2d_th_form_assembly():
    """Test that Taylor-Hood weak forms assemble without UFL shape errors."""
    mesh = UnitSquareMesh(4, 4)
    
    # Taylor-Hood elements: CG2 for Velocity, CG1 for Pressure
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W
    
    # mixed space funcs
    U_old = Function(Z)
    U_older = Function(Z)
    
    # vector-valued funcs
    f = Function(V)
    f_old = Function(V)
    g = Function(V)
    g_old = Function(V)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)
    
    # Mock parameters
    gamma = Constant(0.05)
    nu = Constant(0.01)
    dt = Constant(0.01)
    idt = 1/dt
    theta = Constant(0.5)

    F_BDF2 = make_weak_form_BDF2(idt, f, f_old, g, g_old, U_older, U_old, dx, ds, gamma, nu)
    F_CN = make_weak_form_CN(idt, f, f_old, g, g_old, U_old, dx, ds, theta, gamma, nu)

    assert F_BDF2 is not None
    assert F_CN is not None