import yaml
from pathlib import Path
import numpy as np
import types

dt=0

# ------------------
# Core config loader
# ------------------

def load_expressions(path, *, namespace=None, backend="ufl"):

    # used for numpy funcs
    def make_callable(expr, env):
        return lambda x, y, t: eval(expr, {"__builtins__": {}}, env | {"x": x, "y": y, "t": t})


    if namespace is None:
        raise ValueError("namespace must be provided explicitly")

    path = Path(path)

    with path.open("r") as f:
        data = yaml.safe_load(f) or {}

    # base environment
    if backend == "numpy":
        env = {
            **namespace,
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "pi": np.pi,
        }
    elif backend == "ufl":
        env = dict(namespace)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # inject constants FIRST
    for k, v in data.items():
        if not isinstance(v, str):
            env[k] = v

    # compile expressions
    compiled = {}
    for k, v in data.items():
        if isinstance(v, str):
            compiled[k] = compile(v, f"<expr:{k}>", "eval")

    # evaluate expressions
    results = dict(env)

    if backend == "numpy":
        def make_callable(expr):
            return lambda x, y, t: eval(expr, {"__builtins__": {}}, {
                **env,
                "x": x,
                "y": y,
                "t": t
            })

        for key, code in compiled.items():
            expr_str = data[key]
            results[key] = make_callable(expr_str)
    elif backend == "ufl":
        for key, code in compiled.items():
            results[key] = eval(code, {"__builtins__": {}}, results)

    return results


# --------------
# Config loaders
# --------------

def load_run_configs(save_dir):
    save_dir = Path(save_dir)

    cfg = load_config(save_dir / "user_settings.yaml")
    solver_parameters = load_solver_parameters(save_dir / "solver_params.yaml")

    return cfg, solver_parameters


# ---------------
# UFL expressions
# ---------------

def load_run_ufls(save_dir, namespace):
    save_dir = Path(save_dir)
    ufl_path = save_dir / "user_expr.yaml"

    return load_expressions(
        ufl_path,
        namespace=namespace,
        backend="ufl",
    )


# -----------------
# NumPy expressions
# -----------------

def load_run_numpy(save_dir, namespace):
    save_dir = Path(save_dir)
    numpy_path = save_dir / "user_expr.yaml"

    return load_expressions(
        numpy_path,
        namespace=namespace,
        backend="numpy",
    )


# force scientific notation to be written as a number
def coerce_numeric(v):
    """Convert string-encoded numbers (e.g. '1e-07') to float/int."""
    if not isinstance(v, str):
        return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


# ---------------------
# Load configs entirely
# ---------------------

def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return {k: coerce_numeric(v) for k, v in raw.items()}

def load_solver_parameters(path, *, dt=dt):
    with open(path) as f:
        params = yaml.safe_load(f)["solver_parameters"]
    params = {k: coerce_numeric(v) for k, v in params.items()}
    if dt != 0:
        for k, v in params.items():
            if isinstance(v, str) and v == "{dt}":
                params[k] = dt
    return params