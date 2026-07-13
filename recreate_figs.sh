#!/bin/bash
#SBATCH --job-name=lengnick
#SBATCH --array=0-29
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/seed_%a.out

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate legnick
pip install -r requirements.txt

MASTER_SEED=20260710
N_SEEDS=30

SEED=$(python3 -c "
import random
print(random.Random(${MASTER_SEED}).sample(range(1, 10**7), k=${N_SEEDS})[${SLURM_ARRAY_TASK_ID}])
")

python run_baseline.py --seed $SEED --months 7000 \
    --out diagnostics/run_seed${SEED}.csv \
    --firm-snapshots diagnostics/firm_snapshots_seed${SEED}.csv