#!/bin/bash
# Launch SynDiff training: frozen section -> H&E image translation
#
# Usage:
#   bash run_train.sh                # fresh run
#   bash run_train.sh --resume       # resume from OUTPUT_PATH/EXP_NAME/content.pth
#
# Prerequisites:
#   - Create the output directory before running:
#       mkdir -p /path/to/output
#   - Compile custom CUDA ops on first run (happens automatically via ninja JIT).
#     If it fails, install: sudo apt install python3.11-dev ninja-build
#
# Dataset / step math (1.4M images, batch_size=4, 6 GPUs):
#   - Per-step throughput across all GPUs: 4 * 6 = 24 images/step
#   - DistributedSampler shards: 1,400,000 / 6 ≈ 233,333 images per process per epoch
#   - Steps per epoch (per process): 233,333 / 4 ≈ 58,333
#   - Estimated wall time per epoch: ~60-90 hours (depends on I/O + GPU)
#
# Resume safety:
#   --save_content_every_steps 2000 writes content.pth every 2,000 steps (≈ once
#   every 2-3 hours). If the run dies mid-epoch, restart with --resume and you
#   lose at most ~2,000 steps of work. content.pth is written atomically
#   (temp file + rename) so a crash mid-write won't corrupt the checkpoint.
#
# Estimated RAM: ~180 GB system RAM (6 DDP processes x ~30 GB JPEG bytes)

INPUT_PATH="${INPUT_PATH:-/localdata/jparkbf/pathology/lung_img_patches}"    # directory containing train_A/, train_B/, val_A/, val_B/
OUTPUT_PATH="${OUTPUT_PATH:-$(dirname "$0")/output}"                         # saves to SynDiff/output/ alongside this script
EXP_NAME="${EXP_NAME:-syndiff_frozen_he}"

# Pin the CUDA toolkit used for ninja JIT compilation of custom ops.
# Default matches the local cu118 PyTorch install; the Docker image overrides
# this to /usr/local/cuda (12.8) via ENV.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Pre-compile the custom CUDA ops in a single process so all DDP ranks find
# the prebuilt .so files in the cache instead of racing on the same compile.
echo "Pre-compiling CUDA extensions..."
python3 -c "from utils.op import upfirdn2d, fused_act; print('CUDA ops ready')"

python3 train.py \
    --image_size 256 \
    --exp ${EXP_NAME} \
    --num_channels 6 \
    --num_channels_dae 32 \
    --ch_mult 1 1 2 2 4 4 \
    --num_timesteps 4 \
    --num_res_blocks 2 \
    --batch_size 4 \
    --contrast1 frozen \
    --contrast2 HE \
    --num_epoch 2 \
    --ngf 48 \
    --embedding_type positional \
    --use_ema \
    --ema_decay 0.999 \
    --r1_gamma 1.0 \
    --z_emb_dim 256 \
    --lr_d 1e-4 \
    --lr_g 1.6e-4 \
    --lazy_reg 10 \
    --lambda_l1_loss 0.5 \
    --save_content \
    --save_content_every 1 \
    --save_content_every_steps 5000 \
    --save_ckpt_every 1 \
    --max_steps 89000 \
    --val_steps 5000 \
    --num_process_per_node 4 \
    --local_rank 0 \
    --port_num 6021 \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH} \
    "$@"

# Tuning notes:
#   - --save_content_every_steps 2000  : worst-case work lost on crash. Lower = safer
#                                        but more disk I/O (content.pth is ~1-2 GB).
#   - --num_epoch 10                   : outer-loop bound. Must be > steps/epoch needed
#                                        or training stops before --max_steps fires.
#   - --max_steps 300000               : ~5 epochs of headroom at 58,333 steps/epoch.
#   - To run on fewer GPUs (e.g., 1), change --num_process_per_node 1.
#   - If you hit GPU OOM, drop --batch_size first, then --num_channels_dae.

