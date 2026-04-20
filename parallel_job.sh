#!/bin/bash
#SBATCH --job-name=nsv2_firedrake
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --time=00:01:00

module purge
module load apptainer compiler/gcc/11 openmpi/4.1

cd test_run || exit 1

mpirun -np $SLURM_NTASKS \
    apptainer exec --bind $PWD:$PWD docker://firedrakeproject/firedrake:2025.10.4 \
    myrun .