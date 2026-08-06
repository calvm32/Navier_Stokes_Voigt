---
---
# Fluid Mechanics Solvers

This repository provides the following solvers: 
- FEM Crank-Nicolson (CN),
- FEM 2nd-order backwards differentiation formula (BDF2), and
- spectral methods 4th-order Runga-Kutta (RK4)

for the following PDEs:
- 2D heat equation
- 2D Navier-Stokes equations (NSE)
- 2D Navier-Stokes-Voigt equations (NSV)
- 2D difference between NSE and NSV

---
---
# How do I run a solver?

You can solve small problems locally, but I suggest to run everything else on computer with larger processing capabilities. For a computer cluster, follow the [local run directions](#local-run-setup). For SLURM runs, follow the [SLURM run directions](#slurm-run-directions)

## Local run directions

### Setup
This repository uses Firedrake, which, depending on your OS, may require some luck to install. See [the bottom of this page](#firedrake-install) for my recommended workflow. Once you have a Firedrake virtual environment, activate it and run the following terminal commands to get this repository's custom commands and some required libraries:
```
pip install -e .
pip install -r requirements.txt
```
Depending on Firedrake version, more libraries may need to be installed.

**NOTE:** if running in VSCODE or a similar code editor, you need to make sure that the Python interpreter points to the correct location. For VSCODE users, press `Ctrl+Shift+P` and then select the virtual environment you just installed for Firedrake.

### Running solvers
To create a directory, use the custom command `mysave <run_directory_name>` followed by the options:

1. `--problem` then a string with (1.1) `h`, `nse`, or `nsv` for the problem type, (1.2) `2` or  `3` for dimension, and finally (1.3) `-fem` or `-spec` for solver type (e.g. `nsv2-fem` for 2D NSV FEM),

2. `--elements` then `sv` for Scott Vogelius or `th` for Taylor Hood elements (only viable for `nse` or `nsv` problems),

3. `--mms` for testing the method of manufactured solutions,

4. `--mesh` followed by the name of a usable mesh file (e.g. `example.msh`),

5. `--set` followed by one of the user input files and a valid key with its desired value (e.g. `user_settings.T=20`),

6. `--list-settings` after defining problem, elements, mms flag, etc. this lists the settings that will be applied.

Note that you may also change the user settings, solver parameters, and initial/boundary conditions (called user expressions) by simply navigating to `<run_directory_name>` and editing the files `user_settings`, `solver_params`, and `user_expr`, respectively.

Now to solve the given problem, use `myrun <relative_path_to/run_directory_name>`, optionally followed by `--np N` to run using MPI parallel processing on N processors. 

## SLURM run directions

### Setup
To prevent large memory being used on the startup node, first run
`apptainer pull firedrake_2025.10.4.sif docker://firedrakeproject/firedrake:2025.10.4` 
after setting up the directory.

### Running solvers
Next, make a copy of the file `parallel_job.sh`, and edit as needed.

---
---
# Interpreting data

### Method of manufactured solutions (MMS) run

Your error plots will save automatically to the subfolder *plots* upon completion. If you want any of the other data, or your run did not finish, you will need to call the relevant file in `solvers/processing/post_processing` yourself.

### Regular solver run

All of the information will save automatically to the subfolder *plots* upon completion. If your run did not finish, you will need to call the relevant file in `solvers/processing/post_processing` yourself.

---
---
# Other Info.

## Best choice of parameter alpha for a given mesh

For any mesh used, one needs to know the total DOFs of the combined velocity and pressure spaces V + W in order to scale alpha correctly. Values for built-in meshes are kept in the table below. Notice that any built-in mesh follows the naming convention of:
***brief descriptor*** + "\_" + ***chord length if applicable*** + "\_" + ***mesh size "h" BEFORE barycentric refinement*** + "\_" + ***"bary" IF barycentrically-refined***.

| Mesh Name | Description | Total Mixed Space DOFs | Total Node DOFs | Min. Mesh Size h | Alpha Value Used |
| --------- | ----------- | ---------------------- | --------------- | ---------------- | ---------------- |
| bluf_body_chord1_h0.5_bary    | coarse bluff body with chord 1    | 2.6x10^5 | 17,448  | 0.068812 | 0.07  |
| bluf_body_chord1_h0.1_bary    | fine bluff body with chord 1      | 6.5x10^6 | 431,496 | 0.012868 | 0.014 |
| channel_h2.0_bary.msh         | coarse channel w/ no obstructions |          | 1,048   | 0.779723 | 0.8   |
| ... |

If you need to find the approximate h for your own mesh, simply run the following custom command: `mymesh relative/path/to/my_mesh_name.msh`. It is suggested to set alpha slightly larger than the smallest h and absolutely no smaller.

## Benchmark testing for high-performance computing

To get an estimate of how many wall-clock hours and core-hours a run will take, run the `mybenchmark` custom command. To do so, you'll first need to configure the exact same options as in a `mysave` command:

1. `--problem` then a string with (1.1) `h`, `nse`, or `nsv` for the problem type, (1.2) `2` or  `3` for dimension, and finally (1.3) `-fem` or `-spec` for solver type (e.g. `nsv2-fem` for 2D NSV FEM),

2. `--elements` then `sv` for Scott Vogelius or `th` for Taylor Hood elements (only viable for `nse` or `nsv` problems),

3. `--mms` for testing the method of manufactured solutions,

4. `--mesh` followed by the name of a usable mesh file (e.g. `example.msh`),

5. `--set` followed by one of the user input files and a valid key with its desired value (e.g. `user_settings.T=20`),

6. `--list-settings` after defining problem, elements, mms flag, etc. this lists the settings that will be applied.

THEN, you will also need additional options:

7. `--cores` for how many cores you would like to run the test with,

8. `--test-steps` exactly how many steps you would like to use to approximate the run (more steps = greater accuracy),

9. `--final-time` the final time of the run you're looking to approximate.

---
---
# Help with errors

If you come across an error, please email me at [calum.heldt@gmail.com](mailto:calum.heldt@gmail.com) or submit an issue on this repository.

---
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

7. `--cores` for how many cores you would like to run the test with

8. `--test-steps` exactly how many steps you would like to use to approximate the run (more steps = greater accuracy)

9. `--final-time` the final time of the run you're looking to approximate
