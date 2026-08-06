import argparse
import os
import sys
import numpy as np

def build_parser():
    parser = argparse.ArgumentParser(description="Calculate physical mesh resolution (h) for Firedrake.")
    parser.add_argument("mesh_file", help="Path to the .msh file")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.mesh_file):
        print(f"Error: Could not find file '{args.mesh_file}' in the current directory.")
        sys.exit(1)

    print(f"Loading mesh from: {args.mesh_file} ...")

    try:
        import firedrake as fd
    except ImportError:
        print("Error: Firedrake not found. Make sure you are running this inside the Apptainer container!")
        sys.exit(1)

    # Load the Gmsh file
    mesh = fd.Mesh(args.mesh_file)

    # Create a Discontinuous Galerkin (degree 0) function space
    V = fd.FunctionSpace(mesh, "DG", 0)

    # Calculate max edge length of each triangle
    h_func = fd.project(fd.CellDiameter(mesh), V)
    h_data = h_func.dat.data_ro

    print("\n" + "="*40)
    print("MESH RESOLUTION REPORT")
    print("="*40)
    print(f"Total Elements: {len(h_data):,}")
    print(f"Minimum h:      {np.min(h_data):.6f}  <-- Use this for smallest alpha")
    print(f"Maximum h:      {np.max(h_data):.6f}")
    print(f"Average h:      {np.mean(h_data):.6f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()