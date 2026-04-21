from firedrake import *
import pytest

from solvers_FEM.navier_stokes_2d.make_weak_form import *

def _setup_ns_problem():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 2)   # velocity
    W = FunctionSpace(mesh, "CG", 1)         # pressure
    Z = V * W

    u = Function(Z)
    v = TestFunction(Z)
    u_old = Function(Z)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    def get_data(t):
        return {
            "ufl_v0": as_vector([Constant(0), Constant(0)]),
            "ufl_p0": Constant(0),
            "ufl_f": as_vector([0.0, 0.0]),
            "ufl_g": as_vector([0.0, 0.0])
        }

    return get_data, u_old, u, v, dx, ds

@pytest.mark.unit
def test_ns_weak_form_cn_assembles():
    get_data, u_old, u, v, dx, ds = _setup_ns_problem()
    F = make_weak_form_CN(
        get_data=get_data,
        u_old=u_old,
        u=u,
        v=v,
        dx=dx,
        ds=ds,
        t=0.0,
        dt=0.1
    )
    assemble(F)

@pytest.mark.unit
def test_ns_weak_form_bdf2_assembles():
    get_data, u_old, u, v, dx, ds = _setup_ns_problem()
    F = make_weak_form_BDF2(
        get_data=get_data,
        u_old=u_old,
        u=u,
        v=v,
        dx=dx,
        ds=ds,
        t=0.0,
        dt=0.1
    )
    assemble(F)
