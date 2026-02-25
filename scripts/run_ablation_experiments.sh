#!/bin/bash
# 批量运行消融实验脚本

# 设置基础参数
DATA_DIR="/home/yilai/projects/poster/NetDiffus/tiff_log"
BASE_SAVE_DIR="ablation_experiments"
IMAGE_SIZE=256
NUM_CHANNELS=128
NUM_RES_BLOCKS=3
DIFFUSION_STEPS=1000
NOISE_SCHEDULE="cosine"
LEARN_SIGMA="True"
BATCH_SIZE=32
LR=5e-4
SAVE_INTERVAL=1000
SPECIAL_WEIGHT=1.0
COS_WEIGHT=0.01

# 消融实验配置列表
ABLATION_CONFIGS=(
    "baseline"
    "triplet_replace"
    "triplet_add"
    "hybrid"
)

# TripletAttention版本
TRIPLET_VERSIONS=("v1" "v2")

# 是否使用空间注意力
TRIPLET_NO_SPATIAL="True"

echo "开始运行消融实验..."
echo "数据目录: $DATA_DIR"
echo "保存目录: $BASE_SAVE_DIR"
echo "实验配置数量: ${#ABLATION_CONFIGS[@]}"

# 遍历所有配置
for config in "${ABLATION_CONFIGS[@]}"; do
    for triplet_version in "${TRIPLET_VERSIONS[@]}"; do
        echo ""
        echo "=========================================="
        echo "运行实验: $config (TripletAttention版本: $triplet_version)"
        echo "=========================================="
        
        # 构建命令
        CMD="CUDA_VISIBLE_DEVICES=7 python scripts/ablation_experiment.py \
            --data_dir $DATA_DIR \
            --image_size $IMAGE_SIZE \
            --num_channels $NUM_CHANNELS \
            --num_res_blocks $NUM_RES_BLOCKS \
            --diffusion_steps $DIFFUSION_STEPS \
            --noise_schedule $NOISE_SCHEDULE \
            --learn_sigma $LEARN_SIGMA \
            --class_cond False \
            --rescale_learned_sigmas False \
            --rescale_timesteps False \
            --lr $LR \
            --batch_size $BATCH_SIZE \
            --save_dir $BASE_SAVE_DIR \
            --special_weight $SPECIAL_WEIGHT \
            --cos_weight $COS_WEIGHT \
            --save_interval $SAVE_INTERVAL \
            --ablation_config $config \
            --triplet_version $triplet_version \
            --triplet_no_spatial $TRIPLET_NO_SPATIAL"
        
        echo "执行命令:"
        echo "$CMD"
        echo ""
        
        # 执行命令
        eval $CMD
        
        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo "✓ 实验 $config ($triplet_version) 完成"
        else
            echo "✗ 实验 $config ($triplet_version) 失败"
            # 可以选择继续或退出
            # exit 1
        fi
        
        echo ""
        sleep 5  # 等待一下再运行下一个实验
    done
done

echo ""
echo "=========================================="
echo "所有消融实验完成！"
echo "=========================================="
