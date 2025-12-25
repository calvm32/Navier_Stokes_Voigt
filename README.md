# Fluid Mechanics Solvers

## Description

Crank-Nicolson solver for:
- 2D Heat equation
- 2D Navier-Stokes equation

## How to run
This repository uses Firedrake, which currently requires a lot of luck to install. 

### For local runs,

Start with *https://www.firedrakeproject.org/install.html*, then run `source /path-to/venv-firedrake/bin/activate`.

Alternatively, copy the code into something like Google CoLab and include
```
try:
    import firedrake
except ImportError:
    !wget "https://fem-on-colab.github.io/releases/firedrake-install-release-real.sh" -O "/tmp/firedrake-install.sh" && bash "/tmp/firedrake-install.sh"
    import firedrake
```

To see solutions, run the output folder and corresponding .pvd file in Paraview.

### For cluster runs,

tmux
cd /data/$USER
. venv-firedrake/bin/activate
cd Fluid_Mechanics

python -m solvers.heat.heat_eqn
