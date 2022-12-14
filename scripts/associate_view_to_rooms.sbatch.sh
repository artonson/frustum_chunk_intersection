#!/bin/bash

#SBATCH --job-name=associate-views
#SBATCH --output=associate-views-logs/%A_%a.out
#SBATCH --error=associate-views-logs/%A_%a.err
#SBATCH --array=1-1
#SBATCH --time=00:20:00
#SBATCH --partition=submit
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64g
#SBATCH --oversubscribe

__usage="
Usage: $0 -d data_dir -o output_dir -y data_type [-v] <INPUT_FILE

  -d: 	input directory
  -o: 	output directory
  -y:   data type [matterport3d or scannet]
  -v:   if set, verbose mode is activated (more output from the script generally)
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
set -x
VERBOSE=false
DATA_TYPE=matterport3d
while getopts "vd:i:o:y:" opt
do
    case ${opt} in
        d) DATA_DIR=$OPTARG;;
        y) DATA_TYPE=$OPTARG;;
        i) INPUT_FILENAME=$OPTARG;;
        o) OUTPUT_DIR=$OPTARG;;
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
while IFS=' ' read -r scene room ; do
    (( count+=1 ))
    if (( count == SLURM_ARRAY_TASK_ID )); then
        break
    fi
done <"${INPUT_FILENAME:-/dev/stdin}"

# conda init bash
source /rhome/aartemov/miniconda3/bin/activate /rhome/aartemov/miniconda3/envs/py38_dev

REPO=/rhome/aartemov/repos/frustum_chunk_intersection
SCRIPT="${REPO}/scripts/associate_views_to_rooms.py"

$SCRIPT \
  $VERBOSE_ARG \
  --data-dir "${DATA_DIR}" \
  --data-type "${DATA_TYPE}" \
  --scene "${scene}" \
  --room "${room}" \
  --sdf-thr 1.0 \
  --output-dir "${OUTPUT_DIR}"
