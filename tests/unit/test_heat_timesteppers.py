from firedrake import *
import pytest

from solvers.timesteppers import *
from solvers.heat_2d.make_weak_form import *

def _setup_heat_timestepper():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    def get_data(t):
        return {
            "ufl_u0": Constant(0),
            "ufl_f": Constant(1),
            "ufl_g": Constant(0),
        }

    return V, dx, ds, get_data


@pytest.mark.unit
def test_bdf2_timestepper_runs_two_steps():
    V, dx, ds, get_data = _setup_heat_problem()

    errors, energy, times = timestepper_BDF2(
        get_data=get_data,
        V=V,
        dx=dx,
        ds=ds,
        t0=0.0,
        T=0.2,
        dt=0.1,
        make_weak_form=make_weak_form_BDF2,
        solver_parameters={},
        vtkfile_name=None,
    )

    # sanity checks
    assert len(errors) == 2
    assert len(times) == 2
    for e in errors:
        assert e >= 0

@pytest.mark.unit
def test_cn_timestepper_runs_two_steps():
    V, dx, ds, get_data = _setup_heat_problem()

    errors, energy, times = timestepper_CN(
        get_data=get_data,
        V=V,
        dx=dx,
        ds=ds,
        t0=0.0,
        T=0.2,
        dt=0.1,
        make_weak_form=make_weak_form_CN,
        solver_parameters={},
        vtkfile_name=None,
    )

    # sanity checks
    assert len(errors) == 2
    assert len(times) == 2
    for e in errors:
        assert e >= 0