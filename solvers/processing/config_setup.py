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

def load_ufl_expressions(path, *, namespace=None):
    """
    Load UFL expressions from a YAML file where entries are stored as strings.
    """

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    namespace = dict(namespace or {})
    results = {}

    for key, value in data.items():
        if isinstance(value, str):
            results[key] = eval(value, {}, namespace)
            namespace[key] = results[key]
        else:
            results[key] = value
            namespace[key] = value

    return results

def load_run_configs(save_dir):
    save_dir = Path(save_dir)
    cfg_path = save_dir / "settings.yaml"
    solver_params_path = save_dir / "solver_params.yaml"

    cfg = load_config(cfg_path)
    solver_parameters = load_solver_parameters(solver_params_path)

    return cfg, solver_parameters

def load_run_ufls(save_dir, namespace):
    save_dir = Path(save_dir)
    ufl_path = save_dir / "ufl_expr.yaml"
    
    ufl_cfg = load_ufl_expressions(ufl_path, namespace=namespace)

    return ufl_cfg