import argparse
import os
import sys
from dataclasses import dataclass
from modules.processing.load_dump import load_txt, dump_txt
from pathlib import Path
import yaml
import shutil

@dataclass
class RunConfig:
    problem: str
    mms: bool
    elements: str | None

# -----------------
# template resolver
# -----------------

class TemplateResolver:

    BASE = Path(__file__).resolve().parents[1] / "user_settings"

    @staticmethod
    def resolve(cfg: RunConfig):

        if cfg.problem == "h2_FEM":
            return TemplateResolver._resolve_heat_FEM(cfg)
        elif cfg.problem == "ns2_FEM":
            return TemplateResolver._resolve_ns_FEM(cfg)
        elif cfg.problem == "nsv2_FEM":
            return TemplateResolver._resolve_nsv_FEM(cfg)
        if cfg.problem == "h2_spec":
            return TemplateResolver._resolve_heat_spec(cfg)
        elif cfg.problem == "ns2_spec":
            return TemplateResolver._resolve_ns_spec(cfg)
        elif cfg.problem == "nsv2_spec":
            return TemplateResolver._resolve_nsv_spec(cfg)
        elif cfg.problem == "comp_spec":
            return TemplateResolver._resolve_compare_spec(cfg)
        elif cfg.problem == "comp_FEM":
            return TemplateResolver._resolve_compare_FEM(cfg)
        else:
            raise ValueError("Unknown problem type")


    @staticmethod
    def _resolve_heat_FEM(cfg: RunConfig):

        if cfg.mms:
            return {
                "settings": f"{TemplateResolver.BASE}/settings/heat_FEM.yaml",
                "solver": f"{TemplateResolver.BASE}/solver_parameters/heat_FEM.yaml",
                "ufl": f"{TemplateResolver.BASE}/user_expr/heat_FEM_MMS.yaml",
                "solver_path": "modules.solvers_FEM.heat_2d.solver_MMS",
            }

        return {
            "settings": f"{TemplateResolver.BASE}/settings/heat_FEM.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/heat_FEM.yaml",
            "ufl": f"{TemplateResolver.BASE}/user_expr/heat_FEM.yaml",
            "solver_path": "modules.solvers_FEM.heat_2d.solver",
        }


    @staticmethod
    def _resolve_heat_spec(cfg: RunConfig):

        if cfg.mms:
            return {
                "settings": f"{TemplateResolver.BASE}/settings/heat_spec.yaml",
                "solver": f"{TemplateResolver.BASE}/solver_parameters/any_spec.yaml",
                "ufl": f"{TemplateResolver.BASE}/user_expr/heat_spec_MMS.yaml",
                "solver_path": "modules.solvers_spectral.heat_2d.solver_MMS",
            }

        return {
            "settings": f"{TemplateResolver.BASE}/settings/heat_spec.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/any_spec.yaml",
            "ufl": f"{TemplateResolver.BASE}/user_expr/heat_spec.yaml",
            "solver_path": "modules.solvers_spectral.heat_2d.solver",
        }


    @staticmethod
    def _resolve_ns_FEM(cfg: RunConfig):

        if cfg.mms:
            return {
                "settings": f"{TemplateResolver.BASE}/settings/ns_FEM_{cfg.elements.upper()}.yaml",
                "solver": f"{TemplateResolver.BASE}/solver_parameters/ns_FEM_{cfg.elements.upper()}.yaml",
                "ufl": f"{TemplateResolver.BASE}/user_expr/ns_FEM_MMS.yaml",
                "solver_path": "modules.solvers_FEM.navier_stokes_2d.solver_MMS",
            }
        
        if cfg.elements is None:
            raise ValueError("This problem requires element type")

        return {

            "settings": f"{TemplateResolver.BASE}/settings/ns_FEM_{cfg.elements.upper()}.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/ns_FEM_{cfg.elements.upper()}.yaml",
            "ufl": f"{TemplateResolver.BASE}/user_expr/ns_FEM.yaml",
            "solver_path": "modules.solvers_FEM.navier_stokes_2d.solver",
        }


    @staticmethod
    def _resolve_ns_spec(cfg: RunConfig):

        if cfg.mms:
            return {
                "settings": f"{TemplateResolver.BASE}/settings/ns_spec.yaml",
                "solver": f"{TemplateResolver.BASE}/solver_parameters/any_spec.yaml",
                "ufl": f"{TemplateResolver.BASE}/user_expr/ns_spec_MMS.yaml",
                "solver_path": "modules.solvers_spectral.navier_stokes_2d.solver_MMS",
            }

        return {
            "settings": f"{TemplateResolver.BASE}/settings/ns_spec.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/any_spec.yaml",
            "ufl": f"{TemplateResolver.BASE}/user_expr/ns_spec.yaml",
            "solver_path": "modules.solvers_spectral.navier_stokes_2d.solver",
        }


    def _resolve_nsv_FEM(cfg: RunConfig):

        if cfg.mms:
            return {
                "settings": f"{TemplateResolver.BASE}/settings/nsv_FEM_{cfg.elements.upper()}.yaml",
                "solver": f"{TemplateResolver.BASE}/solver_parameters/nsv_FEM_{cfg.elements.upper()}.yaml",
                "ufl": f"{TemplateResolver.BASE}/user_expr/nsv_FEM_MMS.yaml",
                "solver_path": "modules.solvers_FEM.navier_stokes_voigt_2d.solver_MMS",
            }
        
        if cfg.elements is None:
            raise ValueError("This problem requires element type")

        return {

            "settings": f"{TemplateResolver.BASE}/settings/nsv_FEM_{cfg.elements.upper()}.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/nsv_FEM_{cfg.elements.upper()}.yaml",
            "ufl": f"{TemplateResolver.BASE}/user_expr/nsv_FEM.yaml",
            "solver_path": "modules.solvers_FEM.navier_stokes_voigt_2d.solver",
        }


    def _resolve_nsv_spec(cfg: RunConfig):

        if cfg.mms:
            return {
                "settings": f"{TemplateResolver.BASE}/settings/nsv_spec.yaml",
                "solver": f"{TemplateResolver.BASE}/solver_parameters/any_spec.yaml",
                "ufl": f"{TemplateResolver.BASE}/user_expr/nsv_spec_MMS.yaml",
                "solver_path": "modules.solvers_spectral.navier_stokes_voigt_2d.solver_MMS",
            }

        return {
            "settings": f"{TemplateResolver.BASE}/settings/nsv_spec.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/any_spec.yaml",
            "ufl": f"{TemplateResolver.BASE}/user_expr/nsv_spec.yaml",
            "solver_path": "modules.solvers_spectral.navier_stokes_voigt_2d.solver",
        }


    def _resolve_compare_spec(cfg: RunConfig):

        return {
            "settings": f"{TemplateResolver.BASE}/settings/nsv_spec.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/any_spec.yaml",
            "ufl": f"{TemplateResolver.BASE}/user_expr/ns_spec.yaml",
            "solver_path": "modules.solvers_spectral.compare_2d.solver",
        }

    def _resolve_compare_FEM(cfg: RunConfig):
        if cfg.elements is None:
            raise ValueError("This problem requires element type")

        return {

            "settings": f"{TemplateResolver.BASE}/settings/nsv_FEM_{cfg.elements.upper()}.yaml",
            "solver": f"{TemplateResolver.BASE}/solver_parameters/ns_FEM_{cfg.elements.upper()}.yaml",
            "ufl": f"{TemplateResolver.BASE}/user_expr/ns_FEM.yaml",
            "solver_path": "modules.solvers_FEM.compare_2d.solver",
        }

# ------------
# save builder
# ------------

class SaveManager:

    @staticmethod
    def create(save_path: str, user_settings: dict, mesh_name: str | None = None):

        if os.path.exists(save_path):
            raise FileExistsError(f"Save already exists: {save_path}")

        os.makedirs(save_path)
        os.makedirs(f"{save_path}/vis")

        dump_txt(load_txt(user_settings["settings"]),
                 f"{save_path}/settings.yaml")

        dump_txt(load_txt(user_settings["solver"]),
                 f"{save_path}/solver_params.yaml")

        dump_txt(load_txt(user_settings["ufl"]),
                 f"{save_path}/user_expr.yaml")

        run_info = {
            "solver_module": user_settings["solver_path"]
        }

        # ----------------
        # Optional mesh
        # ----------------

        if mesh_name is not None:

            mesh_source = (
                TemplateResolver.BASE / "meshes" / mesh_name
            )

            if not mesh_source.exists():
                raise FileNotFoundError(
                    f"Mesh file does not exist: {mesh_source}"
                )

            mesh_dest = Path(save_path) / mesh_name

            shutil.copy(mesh_source, mesh_dest)

            run_info["mesh_name"] = mesh_name

        with open(f"{save_path}/run_info.yaml", "w") as f:
            yaml.dump(run_info, f)

        print(f"Created save at: {save_path}")
        print(f"Solver path: {user_settings['solver_path']}")

        if mesh_name is not None:
            print(f"Mesh copied: {mesh_name}")

def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a save directory"
    )

    parser.add_argument(
        "save_path",
        help="Name of the save directory to create"
    )

    parser.add_argument(
        "--problem",
        required=True,
        choices=["h2_FEM", "h3_FEM", "ns2_FEM", "ns3_FEM", "nsv2_FEM", "nsv3_FEM",
                 "h2_spec", "ns2_spec", "nsv2_spec", "comp_spec", "comp_FEM"],
        help="Problem type"
    )

    parser.add_argument(
        "--mms",
        action="store_true",
        help="Use method of manufactured solutions"
    )

    parser.add_argument(
        "--elements",
        choices=["sv", "th"],
        help="Element type"
    )

    parser.add_argument(
        "--mesh",
        help="Optional name of mesh [with corresp. file extension]"
    )

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = RunConfig(
        problem=args.problem,
        mms=args.mms,
        elements=args.elements,
    )

    user_settings = TemplateResolver.resolve(cfg)
    SaveManager.create(args.save_path, user_settings, args.mesh)

if __name__ == "__main__":
    main()