from firedrake import *
import pytest

from pathlib import Path
from navier_stokes_voigt.processing.config_setup import *

def test_settings_yaml_loads():
    cfg = load_config(
        Path("navier_stokes_voigt/templatessettings/heat_MMS.yaml")
    )

    required_keys = ["t0", "T", "dt", "theta"]
    for k in required_keys:
        assert k in cfg

def test_solver_params_yaml_loads():
    params = load_solver_parameters(
        Path("navier_stokes_voigt/templatessolver_parameters/heat_MMS.yaml")
    )

    assert isinstance(params, dict)
