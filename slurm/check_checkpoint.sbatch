#!/bin/bash
#SBATCH --job-name=strum-play
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit from the project root}"
HAND_DIR="$PROJECT_DIR/external/mjlab_hand"
PY="$HAND_DIR/.venv/bin/python"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export WANDB_MODE=offline

cd "$HAND_DIR"

"$PY" "$PROJECT_DIR/scripts/check_checkpoint.py" \
  --checkpoint "$HAND_DIR/logs/rsl_rl/shadow_grasp/2026-10-22_03-26-41/model_9.pt" \
  --output "$PROJECT_DIR/outputs/checkpoint-${SLURM_JOB_ID}" \
  --steps 200
