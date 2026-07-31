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

# Set environment variable to disable debug mode (run on full dataset)
export DEBUG=False

# Run passive data aggregation
echo ""
echo "=== 1. Running Passive Data Aggregation (02_Aggregate_All_Passive.py) ==="
/home/leha18/.conda/envs/py39-env/bin/python src/scripts/02_Aggregate_All_Passive.py --nodebug

# Run GPS data aggregation
echo ""
echo "=== 2. Running GPS Data Aggregation (02gps_Aggregate_GPS.py) ==="
/home/leha18/.conda/envs/py39-env/bin/python src/scripts/02gps_Aggregate_GPS.py --nodebug
echo ""
echo "=== Finished passive aggregation pipeline at $(date '+%Y-%m-%d %H:%M:%S') ==="
