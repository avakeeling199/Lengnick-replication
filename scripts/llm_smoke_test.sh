#!/bin/bash
#SBATCH --account gpu_mills.prj
#SBATCH --partition gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --time 03:45:00
#SBATCH --qos gpu_bmrc_4hr
#SBATCH --output logs/smoke_test_%j.out
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

ollama serve > ollama_smoke_test.log 2>&1 &
sleep 10

# Uses seed 9001 (not 42) so this does NOT touch checkpoint_seed42.pkl or the
# seed-42 run/snapshot CSVs -- those are the "before the fix" evidence for
# the report and must stay untouched.
echo "=== smoke test: fresh LLM-pricing run, seed 9001, 60 months ==="
python main.py --seed 9001 --months 60 --pricing-mode llm \
    --out diagnostics/smoke_test_seed9001.csv \
    --firm-snapshots diagnostics/smoke_test_firm_snapshots_seed9001.csv

echo "=== zombie-firm check ==="
python scripts/check_zombie_firms.py diagnostics/smoke_test_firm_snapshots_seed9001.csv

echo "=== aggregate health check ==="
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv('diagnostics/smoke_test_seed9001.csv')
monthly = df.iloc[20::21].reset_index(drop=True)
cols = ['Employment', 'AvgPrice', 'AvgWage', 'FirmLiquidity', 'HHLiquidity']
print(monthly[cols].iloc[::10].to_string())
print("--- last 10 months ---")
print(monthly[cols].tail(10).to_string())
EOF

echo "=== smoke test complete ==="
kill %1
