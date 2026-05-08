#!/bin/bash

# Training script for lung patch virtual staining (frozen -> H&E) using OT-CFM.
# Usage:
#   ./scripts/train_lung_patches.sh                          # fresh run
#   ./scripts/train_lung_patches.sh path/to/checkpoint.ckpt  # resume from checkpoint
#   change cuda_visible_devices, --nproc_per_node and trainer.devices to use different GPU(s)

CKPT_PATH=$1

CMD="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 runner/src/train.py \
  experiment=lung_patches \
  trainer.devices=1"

if [ -n "$CKPT_PATH" ]; then
    CMD="$CMD ckpt_path=$CKPT_PATH"
fi

eval $CMD
