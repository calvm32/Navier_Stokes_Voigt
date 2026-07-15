#!/bin/bash
#SBATCH --job-name=nsv2_firedrake
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --time=40:00:00
#SBATCH --mem=64G

module purge
module load apptainer compiler/gcc/11 openmpi/4.1

set -e

# -------------
# USER SETTINGS
# -------------

PROJECT_DIR=$HOME/projects/Navier_Stokes_Voigt

# set up run
RUN_NAME=test_run
PROBLEM=nsv2_FEM
ELEMENTS=sv # th or sv (ONLY IF ns OR nsv FEM)
MMS=no # yes or no
MESH=fine_bluff_body_chord1.msh

# override .yaml settings
SETS=(
    user_settings.alpha=0.5
    # ex: "user_settings.T=10.0"
    # ex: "user_settings.Re=1000"
    # ex: "solver_params.ksp_rtol=1e-8"
)
LIST_SETTINGS=no # yes or no

# -----------------
# ENVIRONMENT SETUP
# -----------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="test_run_${TIMESTAMP}"

IMAGE=$PROJECT_DIR/firedrake_2025.10.4.sif
SOLUTIONS_DIR=$PROJECT_DIR/solutions
RUN_DIR=$SOLUTIONS_DIR/$RUN_NAME

cd $PROJECT_DIR || exit 1

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export OMPI_MCA_opal_cuda_support=false

mkdir -p "$SOLUTIONS_DIR"

# ------------------------
# INSTALL PROJECT PACKAGE
# ------------------------

apptainer exec \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --pwd $PROJECT_DIR \
    $IMAGE \
    bash -c "
        pip install --user -e .
        pip install --user -r requirements.txt
        export PATH=\$HOME/.local/bin:\$PATH
        which mysave
        which myrun
    "

# --------------------
# CREATE RUN DIRECTORY
# --------------------


if [ ! -d "$RUN_DIR" ]; then

    CREATE_CMD="mysave solutions/$RUN_NAME \
        --problem $PROBLEM"

    # FEM NS/NSV problems require element choice
    if [[ "$PROBLEM" == *"FEM"* ]] && \
       [[ "$PROBLEM" == ns2_* || "$PROBLEM" == nsv2_* ]]; then
        CREATE_CMD="$CREATE_CMD --elements $ELEMENTS"
    fi

    # include mesh if defined
    if [ -n "$MESH" ]; then
        CREATE_CMD="$CREATE_CMD --mesh $MESH"
    fi

    # include MMS flag if declared
    MMS_FLAG=""

    if [ "$MMS" = "yes" ]; then
        MMS_FLAG="--mms"
    fi

    CREATE_CMD="$CREATE_CMD $MMS_FLAG"

    for setting in "${SETS[@]}"; do
        CREATE_CMD="$CREATE_CMD --set $setting"
    done

    if [ "$LIST_SETTINGS" = "yes" ]; then
        apptainer exec \
            --bind $PROJECT_DIR:$PROJECT_DIR \
            --pwd $PROJECT_DIR \
            $IMAGE \
            bash -c "
                export PATH=\$HOME/.local/bin:\$PATH
                mysave dummy \
                    --problem $PROBLEM \
                    ${ELEMENTS:+--elements $ELEMENTS} \
                    --list-settings
            "
        exit 0
    fi

    echo "Creating run directory:"
    echo "$CREATE_CMD"
	
    apptainer exec \
    	--bind $PROJECT_DIR:$PROJECT_DIR \
    	--pwd $PROJECT_DIR \
    	$IMAGE \
    	bash -c "
    	    export PATH=\$HOME/.local/bin:\$PATH
       	    $CREATE_CMD
    	"
fi

# --------------
# RUN SIMULATION
# --------------

cd "$RUN_DIR" || exit 1

srun apptainer exec \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --pwd "$RUN_DIR" \
    $IMAGE \
    bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        myrun .
    "
