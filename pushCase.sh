#!/bin/bash

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

REMOTE_USER="thtd"
REMOTE_HOST="grendel.cscaa.dk"

LOCAL_CASE_DIR="$HOME/OpenFOAM/au672287-v2512/run/projects"
CLUSTER_OUTPUT_DIR="/home/thtd/OpenFOAM/thtd-v2512/run/projects"

# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------

if [ $# -ne 1 ]; then
    echo "Usage: $0 <case_name>"
    exit 1
fi

CASE_NAME="$1"

LOCAL_CASE="${LOCAL_CASE_DIR}/${CASE_NAME}/"
REMOTE_CASE="${REMOTE_USER}@${REMOTE_HOST}:${CLUSTER_OUTPUT_DIR}/${CASE_NAME}/"

if [ ! -d "$LOCAL_CASE" ]; then
    echo "ERROR: Local case directory does not exist:"
    echo "$LOCAL_CASE"
    exit 1
fi

echo "========================================"
echo "Uploading case: $CASE_NAME"
echo "From: $LOCAL_CASE"
echo "To:   $REMOTE_CASE"
echo "========================================"

rsync -avz --progress \
    --exclude 'processor*' \
    --exclude 'log.*' \
    --exclude 'postProcessing' \
    --exclude 'out_slurm-*.out' \
    --exclude 'err_slurm-*.err' \
    "$LOCAL_CASE" \
    "$REMOTE_CASE"

echo
echo "Upload complete."
