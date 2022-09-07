#!/bin/bash

#SBATCH --job-name=select-views
#SBATCH --output=/trinity/home/a.artemov/tmp/sharpf-stats/%A.out
#SBATCH --error=/trinity/home/a.artemov/tmp/sharpf-stats/%A.err
#SBATCH --array=1-1
#SBATCH --time=00:10:00
#SBATCH --partition=submit
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2g

__usage="
Usage: $0 -d data_dir [-v] <INPUT_FILE

  -d: 	input directory
  -v:   if set, verbose mode is activated (more output from the script generally)
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "vd:i:" opt
do
    case ${opt} in
        d) DATA_DIR=$OPTARG;;
        i) INPUT_FILENAME=$OPTARG;;
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

count=0
while IFS=' ' read -r scene room type; do
    (( count++ ))
    if (( count == SLURM_ARRAY_TASK_ID )); then
        break
    fi
done <"${INPUT_FILENAME:-/dev/stdin}"

conda activate frustum_chunk_intersection-3.8.6

SCRIPT="${REPO}/scripts/select_views.py"

$SCRIPT \
  $VERBOSE_ARG \
  --data-dir "${DATA_DIR}" \
  --scene "${scene}" \
  --room "${room}" \
  --type "${type}" \
  --chunk "*" \
  --overlap 0.01 \
  --sdf-thr 0.01 \
  --output-dir "${DATA_DIR}"/output \
  --output-fraction