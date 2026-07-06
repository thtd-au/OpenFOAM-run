#!/bin/bash

set -eo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

REMOTE_USER="thtd"
REMOTE_HOST="grendel.cscaa.dk"

CLUSTER_CASE_DIR="/home/thtd/OpenFOAM/thtd-v2512/run/projects"
LOCAL_OUTPUT_DIR="$HOME/OpenFOAM/au672287-v2512/run/projects"

# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------

if [ $# -ne 1 ]; then
    echo "Usage: $0 <case_name>"
    exit 1
fi

CASE_NAME="$1"

REMOTE_CASE="${REMOTE_USER}@${REMOTE_HOST}:${CLUSTER_CASE_DIR}/${CASE_NAME}/"
LOCAL_CASE="${LOCAL_OUTPUT_DIR}/${CASE_NAME}/"

mkdir -p "$LOCAL_CASE"

echo "========================================"
echo "Fetching case: $CASE_NAME"
echo "From: $REMOTE_CASE"
echo "To:   $LOCAL_CASE"
echo "========================================"

rsync -avz --progress \
    --exclude 'processor*' \
    --exclude 'out_slurm-*.out' \
    --exclude 'err_slurm-*.err' \
    "$REMOTE_CASE" \
    "$LOCAL_CASE"

echo
echo "Transfer complete."
