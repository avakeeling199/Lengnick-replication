#!/bin/bash
#SBATCH --account gpu_mills.prj
#SBATCH --partition gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --time 00:30:00
#SBATCH --qos gpu_bmrc_4hr
#SBATCH --output logs/test_parallel_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ava.keeling@stx.ox.ac.uk

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate legnick

export PATH=/well/mills/users/tej036/opt/bin:$PATH
export LD_LIBRARY_PATH=/well/mills/users/tej036/opt/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/well/mills/users/tej036/ollama_models
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NUM_PARALLEL=16

ollama serve > ollama_test.log 2>&1 &
sleep 10

echo "=== Testing NUM_PARALLEL=16 with llama3.3:70b, 2 months ==="
time python main.py --seed 42 --months 2 --pricing-mode llm \
    --out diagnostics/test_parallel6.csv \
    --firm-snapshots diagnostics/test_parallel6_snap.csv

echo "=== Test complete ==="