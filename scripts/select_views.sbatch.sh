#!/bin/bash

#SBATCH --job-name=select-views
#SBATCH --output=select-views-logs/%A_%a.out
#SBATCH --error=select-views-logs/%A_%a.err
#SBATCH --array=1-1644
#SBATCH --time=10:00:00
#SBATCH --partition=submit
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4g
#SBATCH --oversubscribe

__usage="
Usage: $0 -d data_dir -o output_dir -c chunk_subdir [-v] <INPUT_FILE

  -d: 	input directory
  -o: 	output directory
  -c:   sub-directory containing chunks (e.g. scannet_chunk_64 / scannet_chunk_128)
  -v:   if set, verbose mode is activated (more output from the script generally)
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
set -x
VERBOSE=false
CHUNK_SUBDIR=scannet_chunk_64
while getopts "vd:i:o:c:" opt
do
    case ${opt} in
        d) DATA_DIR=$OPTARG;;
        i) INPUT_FILENAME=$OPTARG;;
        o) OUTPUT_DIR=$OPTARG;;
        c) CHUNK_SUBDIR=$OPTARG;;
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

if [[ ! ${OUTPUT_DIR} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi
mkdir -p "${OUTPUT_DIR}"

count=0
while IFS=' ' read -r scene room type; do
    (( count++ ))
    if (( count == SLURM_ARRAY_TASK_ID )); then
        break
    fi
done <"${INPUT_FILENAME:-/dev/stdin}"

# conda init bash
source /rhome/aartemov/miniconda3/bin/activate /rhome/aartemov/miniconda3/envs/py38_dev

REPO=/rhome/aartemov/repos/frustum_chunk_intersection
SCRIPT="${REPO}/scripts/select_views.py"

MATTERPORT_SDF_THR=0.5
MATTERPORT_MAX_DISTANCE_THR=0.06

SCANNET_SDF_THR=0.02
SCANNET_MAX_DISTANCE_THR=0.01
#  --association-file "${DATA_DIR}/association/${scene}_room${room}.txt" \

$SCRIPT \
  $VERBOSE_ARG \
  --data-dir "${DATA_DIR}" \
  --scene "${scene}" \
  --room "${room}" \
  --type "${type}" \
  --chunk "*" \
  --chunk-subdir "${CHUNK_SUBDIR}" \
  --overlap 0.01 \
  --sdf-thr ${SCANNET_SDF_THR} \
  --max-distance-thr ${SCANNET_MAX_DISTANCE_THR} \
  --output-dir "${OUTPUT_DIR}" \
  --output-fraction
