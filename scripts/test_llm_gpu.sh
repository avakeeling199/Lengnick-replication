#!/bin/bash
#SBATCH --account gpu_mills.prj
#SBATCH --partition gpu_rtx8000_48gb,gpu_a100_40gb,gpu_v100_32gb,gpu_p100_16gb
#SBATCH --gres gpu:1
#SBATCH --time 00:30:00
#SBATCH --qos gpu_bmrc_4hr
#SBATCH --output logs/test_llm_gpu_%j.out

mkdir -p logs

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate legnick

export PATH=/well/mills/users/tej036/opt/bin:$PATH
export LD_LIBRARY_PATH=/well/mills/users/tej036/opt/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/well/mills/users/tej036/ollama_models

OLLAMA_NUM_PARALLEL=4 ollama serve > ollama.log 2>&1 &
sleep 10

echo "=== ollama list ==="
ollama list

echo "=== starting simulation ==="
python main.py --seed 42 --months 2 --pricing-mode llm \
    --out diagnostics/test_gpu_llm.csv \
    --firm-snapshots diagnostics/test_gpu_llm_snap.csv

echo "=== done ==="
kill %1