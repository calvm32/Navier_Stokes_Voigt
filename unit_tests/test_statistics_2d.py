import pytest
from firedrake import *

from modules.processing.statistics.pdf_sampler_2d import pdf_sampler_2d
from modules.processing.statistics.structure_funcs_2d import structure_funcs_2d

def test_statistics_samplers():
    """Test statistical post-processing functions using a known synthetic Firedrake signal."""
    
    sample_xmax, sample_ymax = 1.0, 1.0
    mesh = UnitSquareMesh(10, 10, sample_xmax, sample_ymax)
    V = VectorFunctionSpace(mesh, "CG", 2)
    
    u_old = Function(V)
    x, y = SpatialCoordinate(mesh)
    
    u_old.interpolate(as_vector([sin(2*pi*x)*cos(2*pi*y), -cos(2*pi*x)*sin(2*pi*y)]))
    pdfs = pdf_sampler_2d(mesh) 
    
    struct_func = structure_funcs_2d(
        u_old.sub(0),
        mesh,
        r_max=0.25 * min(sample_xmax, sample_ymax),
        nbins=30
    )
    
    assert pdfs is not None
    assert len(struct_func) > 0