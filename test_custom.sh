#!/bin/bash
# Testing script for custom dataset

echo "Testing tracker on custom dataset..."

# Make sure you have a trained model checkpoint
MODEL_PATH="./checkpoints_custom/mixformer2_vit/custom_dataset/MixFormerV2_Custom_ep0300.pth.tar"

python tracking/test.py \
    mixformer2_vit \
    custom_dataset \
    --runid 300 \
    --dataset_name custom \
    --threads 1

echo "Testing completed!"
echo "Results saved in test/tracking_results/"