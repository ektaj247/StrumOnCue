#!/bin/bash
#SBATCH --job-name=fret-train
#SBATCH --account=seita_1951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Train the chord-fretting policy on the ten training chord changes.
# Submit from the project root. Optional overrides:
#   sbatch --export=ALL,ITERATIONS=20,NUM_ENVS=1024,RUN_NAME=smoke slurm/fret_train.sbatch

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit from the project root}"
PY="$PROJECT_DIR/external/mjlab_hand/.venv/bin/python"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$PROJECT_DIR/outputs/wandb"
mkdir -p "$WANDB_DIR"

cd "$PROJECT_DIR"
echo "host: $(hostname)   commit: $(git rev-parse --short HEAD)   start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Stop here if the contact bookkeeping is broken on this machine.
"$PY" scripts/rl_selfcheck.py

"$PY" scripts/rl_train.py Strum-Fret-Shadow \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERATIONS:-3000}" \
  --agent.seed "${SEED:-42}" \
  --agent.run-name "${RUN_NAME:-pilot}"

echo "end: $(date)"
