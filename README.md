# Fluid Mechanics Solvers

This repository provides a FEM Crank-Nicolson and BDF2 solver for:
- 2D heat equation
- 2D Navier-Stokes equations (NSE)
- 2D Navier-Stokes-Voigt equations (NSV)
- 2D difference between NSE and NSV

Aditionally, this repository provides a spectral RK4 solver for:
- 2D heat equation
- 2D Navier-Stokes equations (NSE)
- 2D Navier-Stokes-Voigt equations (NSV)
- 2D difference between NSE and NSV

---

# How do I run this?

You can run small runs locally, but I suggest to run everything else on computer with larger processing capabilities. For a computer cluster, follow the same directions as [locally](#set-up-on-your-device). For SLURM runs, follow the [other directions below]()

## Set up on your device

This repository uses Firedrake, which currently requires a lot of luck to install. See [the bottom of this page](#firedrake-install) for my recommended workflow. Once you have Firedrake in a virtual environment, activate it and run the following to get all my personal commands and some required libraries:
```
pip install -e .
pip install -r requirements.txt
```
Depending on Firedrake version, more libraries may need to be installed.

**NOTE** if running in VSCODE or a similar program, you need to make sure that the Python interpreter points to the correct location. For VSCODE users, press `Ctrl+Shift+P` and then select the virtual environment you just installed for Firedrake.

## Run on your device
To create a directory, use `mysave <directory_name>` followed by the options:

1. `--problem` then a string with (1.1) `h`, `ns`, or `nsv` for the problem type, (1.2) `2` or  `3` for dimension, and finally (1.3) `_FEM` or `_spec` for solver type

2. `--elements` then `sv` for Scott Vogelius or `th` for Taylor Hood elements (only viable for ns or nsv problems)

3. `--mms` for testing the method of manufactured solutions

4. `--mesh` followed by the name of a usable mesh file, e.g. `example.msh`

Now to run the problem, use `myrun <relative_path_to_directory>` followed by `--np N` to run using MPI parallel processing on N processors

5. `--set` followed by one of the user input files and a valid key with its desired value, e.g. `user_settings.T=20`

6. `--list-settings` after defining problem, elements, mms flag, etc. this lists the settings that will be applied

## Run on a SLURM device

To prevent large memory being used on the startup node, first run
`apptainer pull firedrake_2025.10.4.sif docker://firedrakeproject/firedrake:2025.10.4` 
after setting up the directory.

Next, make a copy of the file `parallel_job.sh`, and edit as needed.

---

# Interpreting data

### Method of manufactured solutions (MMS) run

Your error plots will save automatically to the subfolder *plots* upon completion. If you want any of the other data, or your run did not finish, you will need to call the relevant file in `solvers/processing/post_processing` yourself.

### Regular solver run

All of the information will save automatically to the subfolder *plots* upon completion. If your run did not finish, you will need to call the relevant file in `solvers/processing/post_processing` yourself.

---

# Help with errors

If you come across an error, please email me at [calum.heldt@gmail.com](mailto:calum.heldt@gmail.com)

---

# Firedrake install

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

# Other Info.

## Parameter alpha vs. DOFs

For any mesh used, one needs to know the total DOFs of the combined velocity and pressure spaces V + W in order to scale alpha correctly. Values for builtin meshes are kept track of in the table below.

| Mesh Name | Mesh Description | Total DOFs |
| --------- | ---------------- | ---------- |
| fine_bluff_body_chord1 | a fine mesh of a bluff body with chord 1 | 262416 |