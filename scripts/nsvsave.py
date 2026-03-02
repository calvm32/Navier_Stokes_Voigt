import argparse
import os
import sys
from dataclasses import dataclass
from solvers.processing.load_dump import load_txt, dump_txt

@dataclass
class RunConfig:
    problem: str
    mms: bool
    elements: str | None

# -----------------
# template resolver
# -----------------

class TemplateResolver:

    BASE = "templates"

    @staticmethod
    def resolve(cfg: RunConfig):

        if cfg.problem == "h2":
            return TemplateResolver._resolve_heat(cfg)
        elif cfg.problem == "ns2":
            return TemplateResolver._resolve_ns(cfg)
        else:
            raise ValueError("Unknown problem type")

    @staticmethod
    def _resolve_heat(cfg: RunConfig):

        if cfg.mms:
            return {
                "settings": f"{TemplateResolver.BASE}/settings/heat_MMS.yaml",
                "solver": f"{TemplateResolver.BASE}/solver_parameters/heat_MMS.yaml",
                "ufl": f"{TemplateResolver.BASE}/ufl_expr/heat_MMS.yaml",
                "solver_path": "solvers/heat_2d/solver_MMS",
            }

        return {
            "settings": f"{TemplateResolver.BASE}/settings/heat.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/heat.yaml",
            "ufl": f"{TemplateResolver.BASE}/ufl_expr/heat_expr.yaml",
            "solver_path": "solvers/heat_2d/solver",
        }

    @staticmethod
    def _resolve_ns(cfg: RunConfig):

        if cfg.mms:
            return {
                "settings": f"{TemplateResolver.BASE}/settings/NS_MMS.yaml",
                "solver": f"{TemplateResolver.BASE}/solver_parameters/NS_MMS.yaml",
                "ufl": f"{TemplateResolver.BASE}/ufl_expr/NS_MMS.yaml",
                "solver_path": "solvers/navier_stokes_2d/solver_MMS",
            }
        
        if cfg.elements is None:
            raise ValueError("NS requires element type")

        return {

            "settings": f"{TemplateResolver.BASE}/settings/NS_MMS.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/NS_{cfg.elements.upper()}.yaml",
            "ufl": f"{TemplateResolver.BASE}/ufl_expr/NS_expr.yaml",
            "solver_path": "solvers/navier_stokes_2d/solver",
        }

# ------------
# save builder
# ------------

class SaveManager:

    @staticmethod
    def create(save_path: str, templates: dict):

        if os.path.exists(save_path):
            raise FileExistsError(f"Save already exists: {save_path}")

        # write the solver path for nsvrun
        def write_solver_metadata(save_path, solver_module):
            with open(f"{save_path}/run_info.yaml", "w") as f:
                yaml.dump({"solver_module": solver_module}, f)

        os.makedirs(save_path)
        os.makedirs(f"{save_path}/data")
        os.makedirs(f"{save_path}/vis")

        dump_txt(load_txt(templates["settings"]),
                 f"{save_path}/settings.yaml")

        dump_txt(load_txt(templates["solver"]),
                 f"{save_path}/solver_params.yaml")

        dump_txt(load_txt(templates["ufl"]),
                 f"{save_path}/ufl_expr.yaml")

        print(f"Created save at: {save_path}")
        print(f"Solver path: {templates['solver_path']}")

def build_parser():
    parser = argparse.ArgumentParser(description="nsvsave research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build = sub.add_parser("build")
    build.add_argument("problem", choices=["h2", "ns2"])
    build.add_argument("save_path")
    build.add_argument("--mms", action="store_true")
    build.add_argument("--elements", choices=["sv", "th"])

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":

        cfg = RunConfig(
            problem=args.problem,
            mms=args.mms,
            elements=args.elements,
        )

        templates = TemplateResolver.resolve(cfg)
        SaveManager.create(args.save_path, templates)
        write_solver_metadata(save_path, templates["solver_path"])

if __name__ == "__main__":
    main()