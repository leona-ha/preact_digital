#!/bin/bash
#SBATCH --job-name=preact_pipeline
#SBATCH --output=logs/%x-%j.log
#SBATCH --partition=compute
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

# Make sure the logs folder exists before submitting:
# mkdir -p logs
#
# Submit with command:
# sbatch slurm/run_all_scripts.sbatch

# Exit on any error
set -e

echo "=== Started full PREACT pipeline at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Node: $SLURM_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Activate Conda environment
CONDA_ENV_NAME=${CONDA_ENV_NAME:-preact}
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate "$CONDA_ENV_NAME" 2>/dev/null || source activate "$CONDA_ENV_NAME" 2>/dev/null || true

echo "Conda environment: ${CONDA_DEFAULT_ENV:-$CONDA_ENV_NAME} ($CONDA_PREFIX)"
echo "Python binary: $(which python)"

# Step 1: Data Preprocessing & Cleaning
echo ""
echo "=== 1. Running Data Preprocessing (01_Data_Preprocess.py) ==="
python src/scripts/01_Data_Preprocess.py

# Step 2: Daily Passive Data Aggregation
echo ""
echo "=== 2. Running Passive Data Aggregation (02_Aggregate_All_Passive.py) ==="
python src/scripts/02_Aggregate_All_Passive.py

# Step 3: Daily GPS Feature Extraction
echo ""
echo "=== 3. Running GPS Data Aggregation (02gps_Aggregate_GPS.py) ==="
python src/scripts/02gps_Aggregate_GPS.py

# Step 4: ECG Preprocessing & HRV Extraction
echo ""
echo "=== 4. Running ECG Preprocessing (06_ECG_Preprocess.py) ==="
python src/scripts/06_ECG_Preprocess.py

echo ""
echo "=== Finished full PREACT pipeline at $(date '+%Y-%m-%d %H:%M:%S') ==="
