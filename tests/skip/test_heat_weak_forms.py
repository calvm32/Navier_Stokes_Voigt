from firedrake import *
import pytest

from navier_stokes_voigt.solvers_FEM.heat_2d.make_weak_form import *

def _setup_heat_problem():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    u_old = Function(V)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    def get_data(t):
        return {
            "ufl_u0": Constant(0),
            "ufl_f": Constant(1),
            "ufl_g": Constant(0),
        }

    return get_data, u_old, u, v, dx, ds

@pytest.mark.unit
def test_heat_weak_form_cn_assembles():
    get_data, u_old, u, v, dx, ds = _setup_heat_problem()

    F = make_weak_form_CN(
        get_data=get_data,
        u_old=u_old,
        u=u,
        v=v,
        dx=dx,
        ds=ds,
        t=0.0,
        dt=0.1,
    )

    assemble(F)

@pytest.mark.unit
def test_heat_weak_form_bdf2_assembles():
    get_data, u_old, u, v, dx, ds = _setup_heat_problem()

    F = make_weak_form_BDF2(
        get_data=get_data,
        u_old=u_old,
        u=u,
        v=v,
        dx=dx,
        ds=ds,
        t=0.0,
        dt=0.1,
    )

    assemble(F)
