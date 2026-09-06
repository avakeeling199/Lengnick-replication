#!/bin/bash
#SBATCH --account gpu_mills.prj
#SBATCH --partition gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --time 60:00:00
#SBATCH --output logs/full_run_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ava.keeling@stx.ox.ac.uk

mkdir -p logs

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate legnick

export PATH=/well/mills/users/tej036/opt/bin:$PATH
export LD_LIBRARY_PATH=/well/mills/users/tej036/opt/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/well/mills/users/tej036/ollama_models
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NUM_PARALLEL=12

ollama serve > ollama_run.log 2>&1 &
sleep 10

# Resuming from the post-fix smoke-test checkpoint (month 60, hard price
# bands + reframed history already applied) instead of starting from month 0.
cp diagnostics/checkpoint_seed9001.pkl diagnostics/checkpoint_seed9001_month60.pkl

echo "=== sanity check: resume from month-60 checkpoint + 1 month ==="
python main.py --seed 9001 --months 61 --resume-from diagnostics/checkpoint_seed9001.pkl --pricing-mode llm \
    --out diagnostics/sanity_resume_check_seed9001.csv \
    --firm-snapshots diagnostics/sanity_resume_snap_seed9001.csv

if [ $? -ne 0 ]; then
    echo "=== SANITY CHECK FAILED — aborting before full run ==="
    exit 1
fi

echo "=== Sanity check passed, resuming full run to month 7000 ==="
python main.py --seed 9001 --months 7000 --resume-from diagnostics/checkpoint_seed9001.pkl --pricing-mode llm \
    --out diagnostics/run_llm_seed9001.csv \
    --firm-snapshots diagnostics/firm_snapshots_llm_seed9001.csv