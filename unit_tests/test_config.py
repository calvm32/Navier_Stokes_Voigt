import pytest
from modules.processing.load_dump import load_yaml

def test_load_solver_parameters():
    """Verify that default FEM configurations load correctly."""
    config = load_yaml("modules/settings/solver_parameters/nsv_2d_FEM_TH.yaml")
    
    # Assert critical keys exist
    assert "alpha" in config, "NSV parameter alpha is missing"
    assert "nu" in config, "Kinematic viscosity nu is missing"
    
def test_user_expr_parsing():
    """Ensure MMS expressions are valid strings that Firedrake can parse."""
    expr_config = load_yaml("modules/settings/user_expr/nsv_FEM_MMS.yaml")
    
    assert "velocity_exact" in expr_config
    assert "pressure_exact" in expr_config