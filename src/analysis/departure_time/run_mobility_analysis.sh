#!/bin/bash
#SBATCH --job-name=departure-time-analysis
#SBATCH --account=
#SBATCH --partition=
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00

set -e

module load anaconda3/2024.02-1-11.4
source activate polars_lbs

# cd "$(dirname "$0")"

if [ -z "$CONFIG" ]; then
    echo ""
    echo "No configuration specified. Running all analyses..."
    echo "To run a specific configuration, use:"
    echo "  sbatch --export=CONFIG=HW_informed submit_analysis.slurm"
    echo ""
    python run_analysis.py --all
else
    echo ""
    echo "Running configuration: $CONFIG"
    echo ""
    python run_analysis.py --config "$CONFIG"
fi
