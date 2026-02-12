#!/bin/bash
#SBATCH --job-name=ddp_testdata_extract
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=./test_data/slurm_extract_%j.out
#SBATCH --error=./test_data/slurm_extract_%j.err

set -euo pipefail

# ----------------------------------------------
# Generate repository-local test parquet datasets
# from 2Cohort_TZ_config.json for internal testing.
# ----------------------------------------------

REPO_DIR="${REPO_DIR:-$PWD}"
SOURCE_CONFIG="${SOURCE_CONFIG:-2Cohort_TZ_config.json}"
OUTPUT_CONFIG="${OUTPUT_CONFIG:-2Cohort_TZ_test_config.json}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-test_data/2cohort_tz}"
MIN_ROWS_PER_GROUP="${MIN_ROWS_PER_GROUP:-5}"
MAX_ROWS_PER_DATASET="${MAX_ROWS_PER_DATASET:-250}"
SEED="${SEED:-42}"

mkdir -p "${REPO_DIR}/test_data"
cd "${REPO_DIR}"

echo "================================================"
echo "DNA Damage test dataset extraction"
echo "Job ID:                ${SLURM_JOB_ID:-N/A}"
echo "Host:                  $(hostname)"
echo "Repository:            ${REPO_DIR}"
echo "Source config:         ${SOURCE_CONFIG}"
echo "Output config:         ${OUTPUT_CONFIG}"
echo "Output data dir:       ${OUTPUT_DATA_DIR}"
echo "min_rows_per_group:    ${MIN_ROWS_PER_GROUP}"
echo "max_rows_per_dataset:  ${MAX_ROWS_PER_DATASET}"
echo "seed:                  ${SEED}"
echo "================================================"

python --version
python -c "import pandas, pyarrow; print('pandas:', pandas.__version__)"

python create_test_dataset_from_config.py \
  --source-config "${SOURCE_CONFIG}" \
  --output-config "${OUTPUT_CONFIG}" \
  --output-data-dir "${OUTPUT_DATA_DIR}" \
  --min-rows-per-group "${MIN_ROWS_PER_GROUP}" \
  --max-rows-per-dataset "${MAX_ROWS_PER_DATASET}" \
  --seed "${SEED}"

echo ""
echo "Generated files:"
find "${OUTPUT_DATA_DIR}" -maxdepth 1 -type f -name "*.parquet" | sort

echo "Done: $(date)"
