# Fluid Mechanics Solvers

This repository provides a FEM Crank-Nicolson and BDF2 solver for:
- 2D heat equation
- 2D Navier-Stokes equations
- 2D Navier-Stokes-Voigt equations

Aditionally, this repository provides a spectral RK4 solver for:
- 2D Navier-Stokes-Voigt equations

## Set up on your device

This repository uses Firedrake, which currently requires a lot of luck to install. See [the bottom of this page](#firedrake-install) for my recommended workflow. Once you have Firedrake in a virtual environment, activate it and run the following to get all my personal commands and some required libraries:
```
pip install -e .
pip install -r requirements.txt
```

**NOTE** if running in VSCODE or a similar program, you need to make sure that the Python interpreter points to the correct location. For VSCODE users, press `Ctrl+Shift+P` and then select the virtual environment you just installed for Firedrake.

Finally, all users should run the following:
```
export PYTHONPATH=/path/to/project/Navier_Stokes_Voigt:$PYTHONPATH
```

## Run on your device
To create a directory, use `mysave <directory_name>` followed by the options:

1. `--problem` then `h2` for 2D heat equation, `ns2` for 2D Navier-Stokes equations, or `nsv2` for 2D Navier-Stokes-Voigt equations

2. `--elements` then `sv` for Scott Vogelius or `th` for Taylor Hood elements

3. `--mms` for testing the method of manufactured solutions

4. `--spec` for implementing spectral methods instead of FEM (for now: only works on NSV)

Now to run the problem, use `myrun <relative_path_to_directory>` followed by `--np <N>` to run using MPI parallel processing on N processors

## Firedrake install

Please run the following, exchanging the directory `~` for your desired directory:

```
wget https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-configure`
python3 firedrake-configure --show-petsc-version --no-package-manager`
git clone --branch $(python3 firedrake-configure --show-petsc-version --no-package-manager) https://gitlab.com/petsc/petsc.git

python3 ~/firedrake-configure --no-package-manager --show-petsc-configure-options | xargs -L1 ./configure
```

OR if that last line doesn't work, something like

```
python3 ~/firedrake-configure --no-package-manager --show-petsc-configure-options | xargs -L1 -I {} ./configure {} --with-bison=0 --download-fblaslapack=1
```

continuing, PETSC will provide some sort of instructions that you should follow exactly, which look like this:

```
make PETSC_DIR=~/petsc PETSC_ARCH=arch-firedrake-default all
```

### Interpreting data

## Method of manufactured solutions (MMS) run

Your error plots will save automatically to the subfolder *plots* upon completion. If you want any of the other data, or your run did not finish, you will need to call the relevant file in `solvers/processing/post_processing` yourself.

## Regular solver run

All of the information will save automatically to the subfolder *plots* upon completion. If your run did not finish, you will need to call the relevant file in `solvers/processing/post_processing` yourself.