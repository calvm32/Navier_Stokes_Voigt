#!/bin/bash
#SBATCH --job-name=nsv2_firedrake
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --time=01:00:00
#SBATCH --partition=compute

module purge
module load apptainer compiler/gcc/11 openmpi/4.1

set -e

# -------------
# USER SETTINGS
# -------------

PROJECT_DIR=$HOME/projects/Navier_Stokes_Voigt

RUN_NAME=test_run
PROBLEM=nsv2_FEM
ELEMENTS=sv # th or sv (ONLY IF ns OR nsv FEM)
MMS=no # yes or no

# -----------------
# ENVIRONMENT SETUP
# -----------------

SOLUTIONS_DIR=$PROJECT_DIR/solutions
RUN_DIR=$SOLUTIONS_DIR/$RUN_NAME

cd $PROJECT_DIR || exit 1

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export OMPI_MCA_opal_cuda_support=false

mkdir -p "$SOLUTIONS_DIR"

# --------------------
# CREATE RUN DIRECTORY
# --------------------

MMS_FLAG=""

if [ "$MMS" = "yes" ]; then
    MMS_FLAG="--mms"
fi


if [ ! -d "$RUN_DIR" ]; then

    CREATE_CMD="mysave solutions/$RUN_NAME \
        --problem $PROBLEM"

    # FEM NS/NSV problems require element choice
    if [[ "$PROBLEM" == *"FEM"* ]] && \
       [[ "$PROBLEM" == ns2_* || "$PROBLEM" == nsv2_* ]]; then
        CREATE_CMD="$CREATE_CMD --elements $ELEMENTS"
    fi

    CREATE_CMD="$CREATE_CMD $MMS_FLAG"

    echo "Creating run directory:"
    echo "$CREATE_CMD"

    apptainer exec \
        --bind $PROJECT_DIR:$PROJECT_DIR \
        --pwd $PROJECT_DIR \
        docker://firedrakeproject/firedrake:2025.10.4 \
        bash -lc "$CREATE_CMD"

    echo "After mysave:"
    ls -lah "$SOLUTIONS_DIR"

    if [ ! -d "$RUN_DIR" ]; then
        echo "ERROR: Run directory was not created."
        exit 1
fi

# --------------
# RUN SIMULATION
# --------------

cd "$RUN_DIR" || exit 1

apptainer exec \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --pwd "$RUN_DIR" \
    docker://firedrakeproject/firedrake:2025.10.4 \
    myrun --np $SLURM_NTASKS .