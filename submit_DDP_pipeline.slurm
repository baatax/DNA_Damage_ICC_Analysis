#!/bin/bash
#SBATCH --job-name=talazoparib_dna_damage
#SBATCH --partition=batch
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/tufts/levinlab/adebaa01/talazoparib_analysis_output/slurm_%j.out
#SBATCH --error=/cluster/tufts/levinlab/adebaa01/talazoparib_analysis_output/slurm_%j.err
# #SBATCH --mail-user=your.email@institution.edu
# #SBATCH --mail-type=END,FAIL

# ============================================================================
# DNA Damage Panel Analysis - Talazoparib
# ============================================================================
# 
# Genotypes: WT-CTRL, KO-CTRL, WT-P1, KO-P1
# Drug: Talazoparib (PARP inhibitor)
# 
# EC50 values:
#   - WT-CTRL: 333 nM (0.333 µM)
#   - KO-CTRL: 333 nM (0.333 µM)
#   - WT-P1:   33 µM
#   - KO-P1:   33 µM
# ============================================================================

# --- Configuration ---
CONFIG_FILE="/cluster/tufts/levinlab/adebaa01/DNA_Damage_analysis/2Cohort_TZ_config.json"
OUTPUT_DIR="/cluster/tufts/levinlab/adebaa01/talazoparib_analysis_output"
N_WORKERS=8

# --- Environment Setup ---
# Uncomment and modify based on your cluster setup

# Option 1: Load modules
# module purge
# module load python/3.10
# module load anaconda3

# Option 2: Activate conda environment
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate dna_damage_env

# Option 3: Activate virtual environment
# source /path/to/venv/bin/activate

# --- Ensure output directory exists ---
mkdir -p ${OUTPUT_DIR}

# --- Print job information ---
echo "=============================================="
echo "DNA Damage Panel Analysis Pipeline"
echo "=============================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Job Name:     ${SLURM_JOB_NAME}"
echo "Node:         $(hostname)"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Memory:       ${SLURM_MEM_PER_NODE}"
echo "Start Time:   $(date)"
echo "----------------------------------------------"
echo "Config:       ${CONFIG_FILE}"
echo "Output:       ${OUTPUT_DIR}"
echo "Workers:      ${N_WORKERS}"
echo "=============================================="

module purge
module load gcc/11.2.0
module load cmake/3.23_gui
source /cluster/tufts/levinlab/adebaa01/kk_venv/bin/activate

# --- Check dependencies ---
echo ""
echo "Checking Python environment..."
python --version
which python

echo ""
echo "Checking required packages..."
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import scipy; print(f'scipy: {scipy.__version__}')"
python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')"

# --- Verify config file ---
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo ""
echo "Config file contents:"
echo "----------------------------------------------"
cat ${CONFIG_FILE}
echo ""
echo "----------------------------------------------"

# --- Run the pipeline ---
echo ""
echo "Starting pipeline..."
echo "=============================================="

python /cluster/tufts/levinlab/adebaa01/DNA_Damage_analysis/run_dna_damage_pipeline.py \
    ${CONFIG_FILE} \
    --output ${OUTPUT_DIR} \
    --workers ${N_WORKERS} \
    --resume \
    --log-level INFO

PIPELINE_EXIT_CODE=$?

# --- Print completion status ---
echo ""
echo "=============================================="
if [ ${PIPELINE_EXIT_CODE} -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline failed with exit code: ${PIPELINE_EXIT_CODE}"
fi
echo "End Time:     $(date)"
echo "=============================================="

# --- List output files ---
echo ""
echo "Output files:"
echo "----------------------------------------------"
find ${OUTPUT_DIR} -type f -name "*.csv" -o -name "*.json" -o -name "*.parquet" | head -50

# --- Print disk usage ---
echo ""
echo "Disk usage:"
du -sh ${OUTPUT_DIR}

exit ${PIPELINE_EXIT_CODE}
