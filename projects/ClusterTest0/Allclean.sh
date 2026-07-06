#!/bin/bash
#SBATCH --job-name=AllClean
#SBATCH --output=AllClean.out
#SBATCH --error=AllClean.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1

set -eo pipefail

echo "Cleaning directory: $(pwd)"

for item in * .*; do
    # Skip current/parent directories
    [[ "$item" == "." || "$item" == ".." ]] && continue

    case "$item" in
        0|system|constant|Allrun|BatchJob.sbatch|Allclean.sh)
            echo "Keeping: $item"
            ;;
        *)
            echo "Removing: $item"
            rm -rf -- "$item"
            ;;
    esac
done

echo "Done."
