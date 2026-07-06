#!/bin/bash -l
#SBATCH --job-name=openfoam_test
#SBATCH --partition=q48
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=out_slurm-%j.out
#SBATCH --error=err_slurm-%j.err

#set -eo pipefail
set -o pipefail 

echo "========= Job started at $(date) =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Running on node(s):"
scontrol show hostnames "$SLURM_JOB_NODELIST"

cd "$SLURM_SUBMIT_DIR"

echo "Loading modules..."

module purge
echo "module purge status: $?"

module load gcc/13.2.0
echo "gcc load status: $?"

module load zlib/1.2.11
echo "zlib load status: $?"

export FOAM_INST_DIR="$HOME/OpenFOAM"
export WM_MPLIB=OPENMPI

echo "FOAM_INST_DIR=$FOAM_INST_DIR"
echo "About to source OpenFOAM bashrc..."

set +u
source "$FOAM_INST_DIR/openfoam/etc/bashrc"
echo "OpenFOAM source status: $?"

set -o pipefail

echo "WM_PROJECT_DIR = ${WM_PROJECT_DIR:-NOT SET}"
echo "FOAM_USER_APPBIN = ${FOAM_USER_APPBIN:-NOT SET}"
echo "pimpleFoamReact path:"
which pimpleFoamReact || true

echo "Sourcing RunFunctions..."
. "${WM_PROJECT_DIR:?}/bin/tools/RunFunctions"

export OMP_NUM_THREADS=1

SCRATCH_CASE="/scratch/$SLURM_JOB_ID/case"

echo "Copying case to scratch: $SCRATCH_CASE"
mkdir -p "$SCRATCH_CASE"

rsync -a \
    --exclude 'processor*' \
    "$SLURM_SUBMIT_DIR"/ "$SCRATCH_CASE"/

cd "$SCRATCH_CASE"

nProcs="${SLURM_NTASKS}"

echo "Using nProcs = $nProcs"

runApplication blockMesh
runApplication checkMesh
runApplication decomposePar

mpirun -np "$SLURM_NTASKS" pimpleFoamReact -parallel \
    2>&1 | tee log.pimpleFoamReact

runApplication reconstructPar

postProcess -func writeCellCentres -time 0 \
    2>&1 | tee log.writeCellCentres

echo "Copying results back to submit directory"

rsync -a \
    --exclude 'processor*' \
    "$SCRATCH_CASE"/ "$SLURM_SUBMIT_DIR"/

echo "========= Job finished at $(date) =========="
