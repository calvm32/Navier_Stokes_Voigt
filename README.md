# Fluid Mechanics DEs

## For smaller runs,

Start with *https://www.firedrakeproject.org/install.html*, then run `source /path-to/venv-firedrake/bin/activate`.

Alternatively, copy the code into somehting like Google CoLabs and include
```
try:
    import firedrake
except ImportError:
    !wget "https://fem-on-colab.github.io/releases/firedrake-install-release-real.sh" -O "/tmp/firedrake-install.sh" && bash "/tmp/firedrake-install.sh"
    import firedrake
```

To see solutions, run the output volder and corresponding .pvd file in Paraview.

## For larger runs,

(later)
