#!/bin/bash

# Quick Start Guide for Ablation Study
# All models use the SAME UNet architecture

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/cgasf_ablation}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT_DIR}/ckpt}"

echo "==========================================="
echo "Ablation Study - Quick Start"
echo "==========================================="
echo ""
echo "All three models (Diffusion, VAE, GAN) now use"
echo "the EXACT SAME UNet architecture:"
echo "  - Model Channels: 128"
echo "  - Channel Mult: (1,2,4,8)"
echo "  - Attention Type: triple"
echo "  - ResBlocks: 3"
echo "  - Image Size: 256x256"
echo ""
echo "==========================================="
echo ""

# Get GPU ID from command line or use default
GPU_ID=${1:-4}

echo "Using GPU: $GPU_ID"
echo ""
echo "Choose a model to train:"
echo "  1) VAE"
echo "  2) GAN"
echo "  3) Diffusion (original)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Starting VAE training..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/vae_train.py \
        --data_dir "${DATA_DIR}" \
        --image_size 256 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --attention_resolutions 16,8 \
        --channel_mult 1,2,4,8 \
        --attention_type triple \
        --lr 5e-4 \
        --batch_size 32 \
        --save_dir "${CKPT_ROOT}/cgasf_ablation_vae" \
        --special_weight 1.0 \
        --save_interval 5000 \
        --num_steps 200000
        ;;
    2)
        echo "Starting GAN training..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/gan_train.py \
        --data_dir "${DATA_DIR}" \
        --image_size 256 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --attention_resolutions 16,8 \
        --channel_mult 1,2,4,8 \
        --attention_type triple \
        --lr 5e-4 \
        --batch_size 32 \
        --save_dir "${CKPT_ROOT}/cgasf_ablation_gan" \
        --special_weight 0.0 \
        --save_interval 5000 \
        --num_steps 200000
        ;;
    3)
        echo "Starting Diffusion training..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python train_midiff.py \
        --data_dir "${DATA_DIR}" \
        --image_size 256 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --diffusion_steps 1000 \
        --noise_schedule cosine \
        --learn_sigma True \
        --class_cond False \
        --rescale_learned_sigmas False \
        --rescale_timesteps False \
        --lr 5e-4 \
        --batch_size 32 \
        --save_dir "${CKPT_ROOT}/cgasf_ablation_triple" \
        --special_weight 1.0 \
        --attention_type triple \
        --save_interval 5000
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
