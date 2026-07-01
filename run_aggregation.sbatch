#!/bin/bash
#SBATCH --job-name=passive_agg
#SBATCH --output=%x-%j.log
#SBATCH --partition=compute
#SBATCH --mem=384G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00

# Exit on any error
set -e

echo "=== Started passive aggregation pipeline at $(date) ==="
echo "Node: $SLURM_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory requested: 384G"

# Set environment variable to disable debug mode (run on full dataset)
export DEBUG=False

# Run passive data aggregation
echo ""
echo "=== 1. Running Passive Data Aggregation (02_Aggregate_All_Passive.py) ==="
/home/milu10/.conda/envs/tiki13/bin/python notebooks/02_Aggregate_All_Passive.py --nodebug

# Run GPS data aggregation
echo ""
echo "=== 2. Running GPS Data Aggregation (02gps_Aggregate_GPS.py) ==="
/home/milu10/.conda/envs/tiki13/bin/python notebooks/02gps_Aggregate_GPS.py --nodebug

echo ""
echo "=== Finished passive aggregation pipeline at $(date) ==="
