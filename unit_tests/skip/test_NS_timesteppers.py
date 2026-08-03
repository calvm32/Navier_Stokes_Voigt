from firedrake import *
import pytest

from modules.solvers_FEM.navier_stokes_2d.make_weak_form import *
from modules.solvers_FEM.timesteppers import *

def _setup_nse_timestepper():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    # minimal boundary condition (velocity zero)
    bcs = [DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 2, 3, 4))]

    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

    appctx = {"Re": 1.0, "gamma": 0.0, "velocity_space": 0}

    def get_data(t):
        return {
            "ufl_v0": as_vector([Constant(0), Constant(0)]),
            "ufl_p0": Constant(0),
            "ufl_f": as_vector([0.0, 0.0]),
            "ufl_g": as_vector([0.0, 0.0])
        }

    return Z, dx, ds, bcs, nullspace, appctx, get_data

@pytest.mark.unit
def test_nse_bdf2_timestepper_runs_one_step():
    Z, dx, ds, bcs, nullspace, appctx, get_data = _setup_nse_timestepper()
    errors_v, errors_p, *_ = timestepper_BDF2(
        get_data=get_data,
        Z=Z,
        dx=dx,
        ds=ds,
        t0=0.0,
        T=0.1,
        dt=0.1,
        make_weak_form=make_weak_form_BDF2,
        bcs=bcs,
        nullspace=nullspace,
        solver_parameters={},
        appctx=appctx,
        vtkfile_name=None
    )
    assert len(errors_v) == 1
    assert len(errors_p) == 1

@pytest.mark.unit
def test_nse_cn_timestepper_runs_one_step():
    Z, dx, ds, bcs, nullspace, appctx, get_data = _setup_nse_timestepper()
    errors_v, errors_p, *_ = timestepper(
        get_data=get_data,
        Z=Z,
        dx=dx,
        ds=ds,
        t0=0.0,
        T=0.1,
        dt=0.1,
        make_weak_form=make_weak_form_CN,
        bcs=bcs,
        nullspace=nullspace,
        solver_parameters={},
        appctx=appctx,
        vtkfile_name=None
    )
    assert len(errors_v) == 1
    assert len(errors_p) == 1
