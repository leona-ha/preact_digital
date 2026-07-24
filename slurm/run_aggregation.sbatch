#!/bin/bash
#SBATCH --job-name=passive_agg
#SBATCH --output=logs/%x-%j.log
#SBATCH --partition=compute
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

# make sure the logs folder exists before submitting, for example:
# mkdir logs
#
# submit with command
# sbatch run_aggregation.sbatch

# Exit on any error
set -e

echo "=== Started passive aggregation pipeline at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Node: $SLURM_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Activate Conda environment
CONDA_ENV_NAME=${CONDA_ENV_NAME:-preact}
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate "$CONDA_ENV_NAME" 2>/dev/null || source activate "$CONDA_ENV_NAME" 2>/dev/null || true

# Run passive data aggregation
echo ""
echo "=== 1. Running Passive Data Aggregation (02_Aggregate_All_Passive.py) ==="
python src/scripts/02_Aggregate_All_Passive.py

# Run GPS data aggregation
echo ""
echo "=== 2. Running GPS Data Aggregation (02gps_Aggregate_GPS.py) ==="
python src/scripts/02gps_Aggregate_GPS.py

echo ""
echo "=== Finished passive aggregation pipeline at $(date '+%Y-%m-%d %H:%M:%S') ==="
