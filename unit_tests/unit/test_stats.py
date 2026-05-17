import numpy as np
import pytest
from firedrake import *

from navier_stokes_voigt.processing.statistics.pdf_sampler_2d import pdf_sampler_2d
from navier_stokes_voigt.processing.statistics.structure_funcs_2d import structure_funcs_2d
from navier_stokes_voigt.processing.statistics.pdf_sampler_2d import pdf_sampler_2d
from navier_stokes_voigt.processing.statistics.structure_funcs_2d import structure_funcs_2d

@pytest.fixture
def mesh():
    return UnitSquareMesh(32, 32)

def test_constant_field(mesh):
    """
    test contant velocity field to find the 
    accuracy of pdf_sampler_2d (probability distribution function sampler)
    """
    V = VectorFunctionSpace(mesh, "CG", 2)

    u = Function(V).interpolate(as_vector((1.0, 0.0)))

    sampler = pdf_sampler_2d(mesh)
    sampler.sample_velocity_y(u, npoints=1000)

    vel, vort = sampler.finalize()

    assert len(vel) > 0
    assert np.allclose(vel, 1.0, atol=1e-10)

def test_vorticity_sampling(mesh):
    """
    test vorticity using linear velocity to find the
    accuracy of the vorticity computation, omega
    """
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)

    u = Function(V).interpolate(as_vector((y, 0)))
    omega = Function(Q).interpolate(curl(u))

    sampler = pdf_sampler_2d(mesh)
    sampler.sample_vorticity(omega, npoints=1000)

    vel, vort = sampler.finalize()

    assert len(vort) > 0
    assert np.allclose(vort, -1.0, atol=1e-10)

def test_structure_constant(mesh):
    """
    test the structure function of a constant field to find that
    accuracy of structure func computation and sampling
    """
    V = VectorFunctionSpace(mesh, "CG", 2)
    u = Function(V).interpolate(as_vector((2.0, 0.0)))

    struct = structure_funcs_2d(u, mesh, nbins=10)
    struct.sample(nsamples_per_bin=20)

    r, S2 = struct.compute()

    assert np.allclose(S2, 0.0, atol=1e-10)