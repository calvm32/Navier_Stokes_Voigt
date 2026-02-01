from firedrake import *
import pytest

from pathlib import Path
from solvers.config_setup import load_config, load_solver_parameters

def test_constants_yaml_loads():
    cfg = load_config(
        Path("templates/constants/heat_MMS.yaml")
    )

    required_keys = ["t0", "T", "dt", "theta"]
    for k in required_keys:
        assert k in cfg

def test_solver_params_yaml_loads():
    params = load_solver_parameters(
        Path("templates/solver_parameters/heat_MMS.yaml")
    )

    assert isinstance(params, dict)
