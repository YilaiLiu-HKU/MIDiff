#!/bin/bash

# Ablation Study Training Scripts
# Based on the original diffusion training command

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Common parameters
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/cgasf_ablation}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT_DIR}/ckpt}"
IMAGE_SIZE=256
NUM_CHANNELS=128
LR=5e-4
BATCH_SIZE=32
SPECIAL_WEIGHT=1.0
SAVE_INTERVAL=5000
GPU_ID=4

echo "========================================="
echo "Ablation Study Training Scripts"
echo "========================================="
echo ""
echo "Original Diffusion Model Command:"
echo "CUDA_VISIBLE_DEVICES=4 python train_midiff.py --data_dir ${DATA_DIR} --image_size 256 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule cosine --learn_sigma True --class_cond False --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-4 --batch_size 32 --save_dir ${CKPT_ROOT}/cgasf_ablation_triple --special_weight 1.0 --attention_type triple --save_interval 5000"
echo ""
echo "========================================="
echo ""

# VAE Training Command
echo "1. VAE Training Command (Using Same UNet Architecture):"
echo "----------------------------------------"
VAE_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/vae_train.py \
--data_dir ${DATA_DIR} \
--image_size ${IMAGE_SIZE} \
--num_channels ${NUM_CHANNELS} \
--num_res_blocks 3 \
--attention_resolutions 16,8 \
--channel_mult 1,2,4,8 \
--attention_type triple \
--lr ${LR} \
--batch_size ${BATCH_SIZE} \
--save_dir ${CKPT_ROOT}/cgasf_ablation_vae \
--special_weight ${SPECIAL_WEIGHT} \
--save_interval ${SAVE_INTERVAL} \
--num_steps 200000"

echo "$VAE_CMD"
echo ""

# GAN Training Command  
echo "2. GAN Training Command (Using Same UNet Architecture):"
echo "----------------------------------------"
GAN_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/gan_train.py \
--data_dir ${DATA_DIR} \
--image_size ${IMAGE_SIZE} \
--num_channels ${NUM_CHANNELS} \
--num_res_blocks 3 \
--attention_resolutions 16,8 \
--channel_mult 1,2,4,8 \
--attention_type triple \
--lr ${LR} \
--batch_size ${BATCH_SIZE} \
--save_dir ${CKPT_ROOT}/cgasf_ablation_gan \
--special_weight 0.0 \
--save_interval ${SAVE_INTERVAL} \
--num_steps 200000"

echo "$GAN_CMD"
echo ""

# Original Diffusion Training Command
echo "3. Diffusion Training Command (Original):"
echo "----------------------------------------"
DIFF_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python train_midiff.py \
--data_dir ${DATA_DIR} \
--image_size ${IMAGE_SIZE} \
--num_channels ${NUM_CHANNELS} \
--num_res_blocks 3 \
--diffusion_steps 1000 \
--noise_schedule cosine \
--learn_sigma True \
--class_cond False \
--rescale_learned_sigmas False \
--rescale_timesteps False \
--lr ${LR} \
--batch_size ${BATCH_SIZE} \
--save_dir ${CKPT_ROOT}/cgasf_ablation_triple \
--special_weight ${SPECIAL_WEIGHT} \
--attention_type triple \
--save_interval ${SAVE_INTERVAL}"

echo "$DIFF_CMD"
echo ""

echo "========================================="
echo "To run any of these commands, copy and paste them into your terminal"
echo "Or uncomment one of the lines below and run this script"
echo "========================================="

# Uncomment one of these to run directly:
# eval "$VAE_CMD"
# eval "$GAN_CMD"
# eval "$DIFF_CMD"
