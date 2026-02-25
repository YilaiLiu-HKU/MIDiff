from scipy.stats import mode
import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from multiprocessing import Pool, cpu_count

T = 192
app_n = 20
file = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/all_users_data_with6cluster.npz"), allow_pickle=True)
print(file.files)
app_traffics = file['Category_ID_Traffic (Byte)']
pois = file["poi_labels"]
num_features = app_traffics.shape[2]
feature_max = np.max(app_traffics)  # 全局最大值，单个标量
feature_min = np.min(np.where(app_traffics != 0, app_traffics, np.inf))  # 全局非零最小值
print(f"Global maximum traffic value: {feature_max}")
print(f"Global minimum non-zero traffic value: {feature_min}")

def process_single_image(args):
    image_path, MEAN_FACTOR, pred_maxes, image_idx = args

    poi_n = image_path.shape[1]//app_n

    gasf_full = image_path.reshape(T, app_n, poi_n)
    app_trace = np.zeros((T, app_n), dtype=np.float32)
    poi_trace = np.zeros((T, poi_n), dtype=np.uint8)
    
    for t in range(T):
        blocks = gasf_full[t]
        block_means = blocks.mean(axis=1)
        total_mean = block_means.sum()
        others_mean = (total_mean - block_means) / (app_n - 1)
        
        is_significant = block_means > (others_mean * MEAN_FACTOR)
        candidates = np.where(is_significant)[0]

        first_pos = np.argmax(blocks, axis=0)
        vals, cnts = np.unique(first_pos, return_counts=True)
        top2_idx = np.argsort(cnts)[::-1][:2]
        top2_pos = vals[top2_idx]

        if top2_pos[0] == 0:
            if len(top2_pos) == 1:
                poi_trace[t, 0] = 1
                continue
            else:
                app_pos = top2_pos[1]
        else:
            app_pos = top2_pos[0]

        poi_pos_by_block = np.argmax(blocks, axis=1)
        mode_pos = mode(poi_pos_by_block)[0]

        # 使用预测的最大值
        predicted_max = pred_maxes[image_idx, t]  # 直接使用图像行和时间步列
        post_traffic = np.clip(predicted_max, 0, 1)
        if post_traffic == 0:
            poi_trace[t, 0] = 1
            continue
        else:
            app_trace[t, app_pos] = post_traffic
            poi_trace[t, mode_pos] = 1

    app_trace = inverse_transform_global(app_trace, poi_trace)
    return app_trace, poi_trace

def inverse_transform_global(app_trace, poi_trace):
    """使用全局最大值进行反归一化，并处理最小值过滤"""
    non_zero_mask = app_trace != 0
    app_trace = app_trace.astype(np.float64)
    
    max_val = app_trace.max()
    row_idx = np.where(app_trace == max_val)[0][0]
    
    # gamma 逆变换
    gamma = 0.25
    print(f'Before gamma correction max:{app_trace.max()}')
    app_trace[non_zero_mask] = np.power(np.abs(app_trace[non_zero_mask]), 1/gamma) * np.sign(app_trace[non_zero_mask])
    print(f'After gamma correction max:{app_trace.max()}')
    
    # 使用全局最大值进行缩放
    app_trace = app_trace * feature_max
    
    # 最小值过滤
    min_threshold = feature_min * 0.9
    for t in range(len(app_trace)):
        if np.all(app_trace[t] < min_threshold):
            app_trace[t] = 0.0  # 设置该时刻所有app流量为0
            poi_trace[t] = 0    # 清零该时刻的POI轨迹
            poi_trace[t, 0] = 1  # 设置第一个POI为1
    
    print("Original max value:", max_val)
    print("After global scaling max:", app_trace.max())
    print("Shape:", app_trace.shape)
    print("Non-zero min value:", np.min(app_trace[app_trace != 0]))
    print("Max value row index:", row_idx)
    print("Minimum threshold:", min_threshold)
    
    return app_trace.astype(np.float32)

def recover_traces(MEAN_FACTOR=1.2, output_suffix=""):
    # 加载预测的最大值
    with open('/home/yilai/projects/poster/NetDiffus/ckpt/post_traffic/model078000.pt_max_predictions.csv', 'r') as f:
        lines = f.readlines()[1:]  # 跳过标题行
    
    pred_maxes = []
    for line in lines:
        # 分割行，跳过image_index
        values = line.strip().split(',')[1:]
        # 转换为浮点数
        values = [float(v) for v in values]
        pred_maxes.append(values)
    
    pred_maxes = np.array(pred_maxes)  # 形状为 (n_images, 256)
    print(f"Original predictions shape: {pred_maxes.shape}")
    
    # 计算中心裁剪的起始和结束位置
    total_len = pred_maxes.shape[1]
    start_idx = (total_len - T) // 2
    end_idx = start_idx + T
    
    # 执行中心裁剪
    pred_maxes = pred_maxes[:, start_idx:end_idx]  # 现在形状应为 (n_images, T)
    print(f"After center crop shape: {pred_maxes.shape}")
    print(f"Sample predictions: {pred_maxes[0, :5]}...")

    image_list = np.load('/home/yilai/projects/poster/NetDiffus/ckpt/post_traffic/model078000.pt_samples_3000x256x160x1.npz')['arr_0']
    print(f"Loaded images shape: {image_list.shape}")
    
    def center_crop(img, target_h, target_w):
        h, w = img.shape[-3], img.shape[-2]
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return img[..., start_h:start_h + target_h, start_w:start_w + target_w, :]

    image_list = center_crop(image_list, 192, 140)
    print(f"Processing {len(image_list)} images with shape {image_list[0].shape}")

    # 测试单个图像处理
    process_single_image([image_list[0], 1.5, pred_maxes, 0])

    with Pool(processes=cpu_count()) as pool:
        args_list = [(img_path, MEAN_FACTOR, pred_maxes, idx) 
                    for idx, img_path in enumerate(image_list)]
        results = pool.map(process_single_image, args_list)

    all_app_traces = []
    all_poi_traces = []
    for app_trace, poi_trace in results:
        all_app_traces.append(app_trace)
        all_poi_traces.append(poi_trace)

    output_path = f"/home/yilai/projects/poster/NetDiffus/ckpt/post_traffic/recover_global.npz"
    print(f"POI trace unique values: {np.unique(poi_trace)}")
    np.savez(
        output_path,
        app_traces=np.array(all_app_traces),
        poi_traces=np.array(all_poi_traces),
        MEAN_FACTOR=MEAN_FACTOR
    )
    print(f"Saved to {output_path}")

for mean_factor in [1.15]:
    recover_traces(MEAN_FACTOR=mean_factor, output_suffix="")