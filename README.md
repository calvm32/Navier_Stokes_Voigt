# Fluid Mechanics Solvers

## Description

Crank-Nicolson and BDF2 solver for:
- 2D heat equation
- 2D Navier-Stokes equations

## Set up on your device
This repository uses Firedrake, which currently requires a lot of luck to install. Once you have it in a virtual environment, activate it and run **pip install -e .**

## Run on your device
To create a directory, use **nsvsave <directory_name>** followed by the options:
    1. **--problem** then **h2** for 2D heat equation or **ns2** for 2D Navier-Stokes equations
    2. **--elements** then **sv** for Scott Vogelius or **th** for Taylor Hood elements
    3. **--mms** for testing the method of manufactured solutions

Now to run the problem, use **nsvrun <relative_path_to_directory>** followed by
    1. **--np <N>** to run using MPI parallel processing on N processors