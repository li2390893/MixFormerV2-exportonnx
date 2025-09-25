#!/bin/bash
# Training script for custom dataset

# Stage1 Training with custom dataset
# Make sure to download pretrained MAE models first:
# wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -P ./pretrained_models/

echo "Starting training with custom dataset..."

python tracking/train.py \
    --script mixformer2_vit \
    --config custom_dataset \
    --save_dir ./checkpoints_custom \
    --mode multiple \
    --nproc_per_node 1 \
    --use_wandb 0

echo "Training completed!"
echo "Checkpoints saved in ./checkpoints_custom"