import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import mode
from multiprocessing import Pool, cpu_count

# --- 全局数据加载与预计算 ---

try:
    file = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/all_users_data_with6cluster.npz"), allow_pickle=True)
    app_traffics = file['Category_ID_Traffic (Byte)']
    
    num_features = app_traffics.shape[2]
    feature_maxes = np.max(app_traffics, axis=(0, 1))
    
    zero_mask = feature_maxes == 0
    feature_maxes[zero_mask] = 1
    
    log_feature_maxes = np.log1p(feature_maxes)
    
    feature_mines = np.min(
        np.where(app_traffics != 0, app_traffics, np.inf),
        axis=(0, 1)
    )
    feature_mines[np.isinf(feature_mines)] = 0

    print("全局统计信息加载完毕。")

except FileNotFoundError:
    print("错误：数据集 'dataset/all_users_data_with6cluster.npz' 未找到。请确保路径正确。")
    exit()

# --- 辅助与逆变换函数 (来自您的代码) ---

def inverse_transform_value(value, feature_max_for_app):
    if value <= 0:
        return 0
    log_val = value * np.log1p(feature_max_for_app)
    return np.expm1(log_val)

def inverse_transform_with_max_compensation(app_trace, feature_maxes_local, max_val_local=None):
    non_zero_mask = app_trace != 0
    app_trace = app_trace.astype(np.float64)
    
    app_trace = app_trace * log_feature_maxes[None, :]
    app_trace[non_zero_mask] = np.expm1(app_trace[non_zero_mask])
    
    min_threshold = feature_mines[None, :] * 0.9
    app_trace = np.where(
        app_trace < min_threshold,
        0,
        app_trace
    )
    return app_trace.astype(np.float32)

def center_crop(img, target_h, target_w):
    h, w = img.shape[-3], img.shape[-2]
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return img[..., start_h:start_h + target_h, start_w:start_w + target_w, :]

# --- 核心处理函数 (来自您的代码) ---

def process_single_image(args):
    image_data, MEAN_FACTOR, sensitivity_factor = args
    
    T = 192
    app_n = 20
    poi_n = image_data.shape[1] // app_n

    # 关键修正：确保reshape维度顺序与后续处理一致
    # mean(axis=1) 是对每个POI块求均值，因此POI应是第一个维度
    gasf_full = image_data.reshape(T, app_n,poi_n) 
    
    app_trace = np.zeros((T, app_n), dtype=np.float32)
    poi_trace = np.zeros((T, poi_n), dtype=np.uint8)
    
    for t in range(T):
        blocks = gasf_full[t] # shape: (poi_n, app_n)
        
        # 修正权重计算
        block_medians = np.median(blocks, axis=1)
        block_p90s = np.percentile(blocks, 60, axis=1)
        combined_scores = (1 - MEAN_FACTOR) * block_medians + MEAN_FACTOR * block_p90s
        
        # 找到得分最高的POI块
        primary_poi_idx = np.argmax(combined_scores)
        
        # 找到最主要的app候选
        first_pos_by_app = np.argmax(blocks, axis=0) # shape: (app_n,)
        vals, cnts = np.unique(first_pos_by_app, return_counts=True)
        # 确保vals中有内容
        if vals.size == 0:
            poi_trace[t, 0] = 1
            continue

        top_poi_candidate = vals[np.argmax(cnts)]

        # 检查最主要的候选POI是否就是得分最高的POI
        if top_poi_candidate != primary_poi_idx:
            poi_trace[t, 0] = 1
            continue

        # 获取获胜block中的最大值和对应的app
        block_max = np.max(blocks[primary_poi_idx, :])
        app_pos = np.argmax(blocks[primary_poi_idx, :])

        # 使用动态阈值进行过滤
        transformed_max = inverse_transform_value(block_max, feature_maxes[app_pos])
        dynamic_threshold = feature_mines[app_pos] * sensitivity_factor
        
        if transformed_max < dynamic_threshold:
            poi_trace[t, 0] = 1
            continue
            
        app_trace[t, app_pos] = block_max
        poi_trace[t, primary_poi_idx] = 1
            
    app_trace = inverse_transform_with_max_compensation(app_trace, feature_maxes)
    return app_trace, poi_trace

# --- 主恢复与统计函数 ---

def recover_traces(MEAN_FACTOR, strict_sensitivity, lenient_sensitivity):
    
    try:
        image_list = np.load('/home/yilai/projects/poster/NetDiffus/ckpt/tiff_log/model003000.pt_samples_3000x256x160x1.npz')['arr_0']
    except FileNotFoundError:
        print("错误：生成的图像文件未找到。")
        return

    image_list = center_crop(image_list, 192, 140)
    num_images = len(image_list)
    
    # --- 分组策略 ---
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    split_point = num_images // 2
    strict_indices = indices[:split_point]
    
    # --- 准备参数列表 ---
    args_list = []
    for i in range(num_images):
        img_data = image_list[i].squeeze(-1)
        sensitivity = strict_sensitivity if i in strict_indices else lenient_sensitivity
        args_list.append((img_data, MEAN_FACTOR, sensitivity))
    
    print(f"开始并行处理 {num_images} 张图像...")
    print(f"策略 -> 严格组因子: {strict_sensitivity}, 宽松组因子: {lenient_sensitivity}")
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_single_image, args_list)
        
    all_app_traces, all_poi_traces = [], []
    removed = 0
    for app_trace, poi_trace in results:
        if np.any(app_trace != 0):
            all_app_traces.append(app_trace)
            all_poi_traces.append(poi_trace)
        else:
            removed += 1
    
    print(f"处理完成。过滤了 {removed} 个空样本。")
    if not all_poi_traces:
        print("警告：没有样本通过过滤，无法统计。")
        return

    # --- 最终统计 ---
    poi_traces_arr = np.array(all_poi_traces, dtype=np.uint8)
    has_poi_rows_per_image = np.any(poi_traces_arr[:, :, 1:] == 1, axis=2)
    num_row_per_image = has_poi_rows_per_image.sum(axis=1)
    
    if num_row_per_image.size == 0:
        print("警告：过滤后没有包含有效POI的样本，无法统计。")
        return

    mean_num_row = np.mean(num_row_per_image)
    std_num_row  = np.std(num_row_per_image)
    
    print("\n" + "="*30)
    print("最终统计结果")
    print("="*30)
    print(f"有效样本数: {len(num_row_per_image)}")
    print(f"每个样本的平均活动行数 (Mean): {mean_num_row:.4f}")
    print(f"每个样本活动行数的标准差 (Std): {std_num_row:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    # --- 在这里调整关键参数 ---
    
    # 权重因子，平衡中位数和高分位数的重要性
    MEAN_FACTOR = 0.5 
    
    # 严格组的敏感度因子 (>1.0)
    STRICT_SENSITIVITY = 1.1
    
    # 宽松组的敏感度因子 (<1.0)
    LENIENT_SENSITIVITY = 0.7
    
    recover_traces(
        MEAN_FACTOR=MEAN_FACTOR,
        strict_sensitivity=STRICT_SENSITIVITY,
        lenient_sensitivity=LENIENT_SENSITIVITY
    )