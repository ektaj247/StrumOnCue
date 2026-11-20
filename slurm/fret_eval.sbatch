#!/bin/bash
#SBATCH --job-name=fret-eval
#SBATCH --account=seita_1951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Score a checkpoint on one split. Submit from the project root:
#   sbatch --export=ALL,CHECKPOINT=logs/rsl_rl/strum_fret/<run>/model_2999.pt,SPLIT=train slurm/fret_eval.sbatch
# SPLIT=heldout runs the two chord changes never seen in training. Run it
# only after the training setup is frozen.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit from the project root}"
PY="$PROJECT_DIR/external/mjlab_hand/.venv/bin/python"
SPLIT="${SPLIT:-train}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

cd "$PROJECT_DIR"
echo "host: $(hostname)   commit: $(git rev-parse --short HEAD)   checkpoint: ${CHECKPOINT:?Set CHECKPOINT}"

"$PY" scripts/rl_eval.py --checkpoint "$CHECKPOINT" --split "$SPLIT" \
  --output "outputs/rl-eval-$SPLIT-$SLURM_JOB_ID"
