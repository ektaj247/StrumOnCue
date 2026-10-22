#!/bin/bash
#SBATCH --job-name=strum-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit from the project root}"
HAND_DIR="$PROJECT_DIR/external/mjlab_hand"
PY="$HAND_DIR/.venv/bin/python"
TRAIN="$HAND_DIR/.venv/bin/train"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export WANDB_MODE=offline
export WANDB_DIR="$PROJECT_DIR/outputs/wandb"

mkdir -p "$WANDB_DIR"

cd "$PROJECT_DIR"

date
hostname
git rev-parse HEAD
git submodule status

nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv

"$PY" - <<'PY'
import sys
from importlib.metadata import version

import torch
import warp as wp

print("Python:", sys.version.split()[0])

for package in ("mjlab", "mujoco", "torch", "warp-lang"):
    print(f"{package}: {version(package)}")

print("PyTorch CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"

print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones(1024, device="cuda")
print("CUDA calculation:", x.sum().item())

wp.init()
print("Warp devices:", wp.get_devices())
assert wp.is_cuda_available(), "Warp cannot access CUDA"
PY

cd "$HAND_DIR"

"$TRAIN" Grasp-Shadow \
  --env.scene.num-envs 64 \
  --agent.max-iterations 10

date
