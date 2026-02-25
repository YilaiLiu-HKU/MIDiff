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
feature_maxes = np.max(app_traffics, axis=(0, 1))  # 形状 (num_features,)
zero_mask= feature_maxes==0
print(zero_mask)
feature_maxes[zero_mask]=0
feature_mines = np.min(
    np.where(app_traffics != 0, app_traffics, np.inf),
    axis=(0, 1)  # 在第 0 维和第 1 维上计算最小值
)
print(feature_mines.shape)
print(feature_maxes)
print(feature_mines)

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

                # 使用预测的最大值而不是blocks.max()
        predicted_max = pred_maxes[image_idx, t]  # 直接使用图像行和时间步列
        post_traffic=np.clip(predicted_max, 0, 1) 
        if post_traffic==0:
            poi_trace[t, 0] = 1
            continue
        else:
      
            #import pdb; pdb.set_trace()
            app_trace[t, app_pos] = post_traffic  # 将值限制在0和1之间
            poi_trace[t, mode_pos] = 1

    app_trace = inverse_transform_with_max_compensation(app_trace, feature_maxes)
    return app_trace, poi_trace

def inverse_transform_with_max_compensation(app_trace, feature_maxes, max=None):
    non_zero_mask = app_trace != 0
    app_trace = app_trace.astype(np.float64)
    
    max_val = app_trace.max()
    row_idx = np.where(app_trace == max_val)[0][0]
    
    # pred_maxes 已经在 [-1,1] 范围内，不需要再映射
    gamma = 0.25
    print(f'max:{app_trace.max()}')
    app_trace[non_zero_mask] = np.power(np.abs(app_trace[non_zero_mask]), 1/gamma) * np.sign(app_trace[non_zero_mask])
    print(app_trace.max())
    
    app_trace = app_trace * feature_maxes[None, :]
    
    print("max value:", max_val)
    print("max value after scale:", app_trace.max())
    print(app_trace.shape)
    print("Min value :", np.min(app_trace[app_trace!= 0]))
    print("row index:", row_idx)
    
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
    #import pdb; pdb.set_trace()
    image_list = np.load('/home/yilai/projects/poster/NetDiffus/ckpt/post_traffic/model078000.pt_samples_3000x256x160x1.npz')['arr_0']
    print(image_list.shape)
    
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

    output_path = f"/home/yilai/projects/poster/NetDiffus/ckpt/post_traffic/recover.npz"
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