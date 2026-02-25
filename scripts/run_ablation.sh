#!/bin/bash

# Ablation Study Training Scripts
# Based on the original diffusion training command

# Common parameters
DATA_DIR="/home/yilai/projects/poster/NetDiffus/tiff_log_thr"
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
echo "CUDA_VISIBLE_DEVICES=4 python scripts/image_train.py --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log_thr --image_size 256 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule cosine --learn_sigma True --class_cond False --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-4 --batch_size 32 --save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_triple --special_weight 1.0 --cos_weight 0.01 --attention_type triple --save_interval 5000"
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
--save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_vae \
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
--save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_gan \
--special_weight 0.0 \
--save_interval ${SAVE_INTERVAL} \
--num_steps 200000"

echo "$GAN_CMD"
echo ""

# Original Diffusion Training Command
echo "3. Diffusion Training Command (Original):"
echo "----------------------------------------"
DIFF_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/image_train.py \
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
--save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_triple \
--special_weight ${SPECIAL_WEIGHT} \
--cos_weight 0.01 \
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
