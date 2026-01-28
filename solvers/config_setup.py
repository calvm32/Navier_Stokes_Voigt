import yaml
from pathlib import Path

dt=0

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
        
def load_solver_parameters(path, *, dt=dt):
    with open(path) as f:
        params = yaml.safe_load(f)["solver_parameters"]

    # Substitute runtime values
    if dt != 0:
        for k, v in params.items():
            if isinstance(v, str) and v == "{dt}":
                params[k] = dt

    return params