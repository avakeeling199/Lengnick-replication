#!/bin/bash
#SBATCH --account gpu_mills.prj
#SBATCH --partition gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --time 60:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ava.keeling@stx.ox.ac.uk

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

cp diagnostics/checkpoint_seed42.pkl diagnostics/checkpoint_seed42_month4500.pkl
cp diagnostics/run_llm_seed42.csv diagnostics/run_llm_seed42_month4500.csv
cp diagnostics/firm_snapshots_llm_seed42.csv diagnostics/firm_snapshots_llm_seed42_month4500.csv

echo "=== sanity check: resume + 1 month ==="
python main.py --seed 42 --months 4501 --resume-from diagnostics/checkpoint_seed42.pkl --pricing-mode llm \
    --out diagnostics/sanity_resume_check.csv \
    --firm-snapshots diagnostics/sanity_resume_snap.csv

if [ $? -ne 0 ]; then
    echo "=== SANITY CHECK FAILED — aborting before full resume ==="
    exit 1
fi

echo "=== sanity check passed, resuming full run ==="
python main.py --seed 42 --months 7000 --resume-from diagnostics/checkpoint_seed42.pkl --pricing-mode llm \
    --out diagnostics/run_llm_seed42.csv \
    --firm-snapshots diagnostics/firm_snapshots_llm_seed42.csv