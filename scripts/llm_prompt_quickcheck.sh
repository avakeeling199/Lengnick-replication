#!/bin/bash
#SBATCH --account gpu_mills.prj
#SBATCH --partition gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --time 00:30:00
#SBATCH --qos gpu_bmrc_4hr
#SBATCH --output logs/prompt_quickcheck_%j.out
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

ollama serve > ollama_prompt_quickcheck.log 2>&1 &
sleep 10

# Short run purely to check whether the inventory-direction fix in prompts.py
# stops the model from citing rising/high inventory as a reason to raise price.
# Uses seed 9002 so it doesn't touch the seed 9001 smoke-test evidence.
echo "=== prompt quick-check: 8-month LLM-pricing run, seed 9002 ==="
python main.py --seed 9002 --months 8 --pricing-mode llm \
    --out diagnostics/quickcheck_seed9002.csv \
    --firm-snapshots diagnostics/quickcheck_firm_snapshots_seed9002.csv

echo "=== inversion check: raises that cite rising/high inventory ==="
python3 - <<'EOF'
import pandas as pd, re

fs = pd.read_csv('diagnostics/quickcheck_firm_snapshots_seed9002.csv')
fs = fs.sort_values(['firm_id', 'month'])
fs['prev_price'] = fs.groupby('firm_id')['price'].shift(1)
fs['action'] = pd.cut(fs['price'] - fs['prev_price'],
                       bins=[-1e9, -0.001, 0.001, 1e9],
                       labels=['lower', 'hold', 'raise'])
fs = fs.dropna(subset=['action'])

raises = fs[fs['action'] == 'raise']
pattern = re.compile(r'inventor\w* (?:level\w* )?(?:have been |has been |is |are )?(?:increas|ris|high|grow)', re.I)
inverted = raises[raises['llm_reasoning'].str.contains(pattern)]

n_inv, n_raise = len(inverted), len(raises)
rate = n_inv / n_raise if n_raise else float('nan')
print(f"raises citing rising/high inventory as justification: {n_inv} / {n_raise} ({rate:.1%})")
print("(seed 9001 smoke test baseline was 236/3775 = 6.3%)")

print()
print("action counts:", fs['action'].value_counts().to_dict())

if n_inv:
    print()
    print("=== sample of remaining inversions (if any) ===")
    for _, r in inverted.head(5).iterrows():
        print(f"month={r.month} firm={r.firm_id} inv={r.llm_input_inventory} demand={r.llm_input_demand} "
              f"price {r.prev_price}->{r.price}")
        print(f"  reasoning: {r.llm_reasoning}")
EOF

echo "=== prompt quick-check complete ==="
kill %1
