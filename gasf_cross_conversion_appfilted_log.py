import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import tifffile
import matplotlib.ticker as mticker
from collections import Counter
save_dir = "tiff_log_app15/"  # directory to store GASF converted data as PNG
top_n = 15
os.makedirs(save_dir,exist_ok=True)  # number of videos
import numpy as np
import matplotlib.pyplot as plt
file=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"dataset/all_users_data_with6cluster.npz"),allow_pickle=True)
print(file.files)
app_traffics=file['Category_ID_Traffic (Byte)']
pois=file["poi_labels"]
#import pdb;pdb.set_trace()
records = []
processed_max_values = [] # 用于存储每个处理后图像的最大像素值
pixels_gt_0_5 = [] # 用于存储所有>0.5的像素值
app_traffic_max_values = [] # 用于存储每个app_traffic样本的最大值
app_traffic_non_zero_values = [] # 用于存储app_traffics中所有非零值

# ===== 新增：为图表3初始化列表 =====
normalized_traffic_max_values = [] # 用于存储每张归一化图像的最大值
normalized_traffic_non_zero_values = [] # 用于存储所有归一化后的非零值
# ====================================

num_features = app_traffics.shape[2]
feature_maxes = np.max(app_traffics, axis=(0, 1))  # 形状 (num_features,)
log_app_traffics = np.log1p(app_traffics)

# 2. (修改) 在对数变换后的数据上计算新的最大值
log_feature_maxes = np.max(log_app_traffics, axis=(0, 1))
#import pdb;pdb.set_trace()
zero_mask= feature_maxes==0
log_feature_maxes[zero_mask]=1
# 保存特征最大值到CSV
feature_max_df = pd.DataFrame({
    'feature_index': range(num_features),
    'max_value': feature_maxes
})

feature_max_df.to_csv(os.path.join(save_dir, "feature_max_values.csv"), index=False)
print(f"Saved feature max values to {save_dir}feature_max_values.csv")

# 目标图像尺寸
TARGET_H, TARGET_W =256, 160
# 1. 统计每个 app 类别在所有样本、所有时间步中的出现频次
#    app_traffics.shape = (N, T, C)
N, T, C = app_traffics.shape
freq = np.zeros(C, dtype=int)

def minmax_scale(ts):
    return (ts - np.min(ts)) / (np.max(ts) - np.min(ts))

def to_gasf_cross(x, y):
    x_phi = np.arccos(x)
    y_phi = np.arccos(y)
    gasf_xy = np.cos(x_phi[:, None] + y_phi[None, :])
    return gasf_xy
for sample in app_traffics:
    # 每个时间步只有一个非 0 索引，用 argmax 取出即可
    active_idx = np.argmax(sample, axis=1)   # shape (T,)
    mask = np.any(sample != 0, axis=1)       # 该时间步是否有 app
    active_idx = active_idx[mask]
    for idx in active_idx:
        freq[idx] += 1

# 2. 选出 top_n 个类别

top_indices = np.argsort(freq)[-top_n:]          # 出现最多的 top_n 类别
print("Top app classes:", top_indices)
print("Their frequencies:", freq[top_indices])

# 3. 过滤样本
ratio_threshold = top_n / C
filtered_app = []
filtered_poi = []

for app, poi in zip(app_traffics, pois):
    # 哪些时间步有 app
    mask_step = np.any(app != 0, axis=1)          # (T,)
    if not np.any(mask_step):
        continue  # 全 0 样本直接丢弃

    active_idx = np.argmax(app, axis=1)[mask_step]
    top_mask = np.isin(active_idx, top_indices)
    ratio = np.sum(top_mask) / mask_step.sum()

    if ratio < ratio_threshold:
        continue  # 不满足要求，丢弃

    # 保留样本，修改非 top_n 类别的时间步
    new_app = np.zeros_like(app)      # (T, C)
    new_poi = np.zeros_like(poi)      # (T, P)
    new_poi[:, 0] = 1                 # 默认第 0 个索引为 1

    for t in np.where(mask_step)[0]:
        idx = np.argmax(app[t])
        if idx in top_indices:
            new_app[t] = app[t]       # 保留 top_n 类别
            new_poi[t] = poi[t]       # 保留原 poi
        # 否则 app[t] 已经是全 0，poi[t] 已经是 [1,0,0,...]

    filtered_app.append(new_app)
    filtered_poi.append(new_poi)

# 替换原始变量
app_traffics = np.stack(filtered_app, axis=0)
pois         = np.stack(filtered_poi, axis=0)
def check_app_only_top(app_data, top_indices):
    """
    app_data: (N, T, C)
    top_indices: list/array of allowed app indices
    返回 True 表示全部非 0 类别都在 top_indices 内
    """
    # 找出所有非 0 的时间步
    mask_step = np.any(app_data != 0, axis=2)            # (N, T)
    active_idx = np.argmax(app_data, axis=2)             # (N, T)
    # 只保留有 app 的时间步
    active_idx = active_idx[mask_step]
    # 判断是否有超出 top_indices 的类别
    illegal = np.setdiff1d(active_idx, top_indices)
    if len(illegal) > 0:
        print("❌ 发现非法 app 类别：", illegal)
        return False
    else:
        print("✅ 所有非 0 app 类别均在 top_indices 内")
        return True

# 调用
if not check_app_only_top(app_traffics, top_indices):
    raise ValueError("预处理结果存在未被允许的 app 类别！")
print(f"Filtered samples: {len(app_traffics)} / {N}")


for i, (app_traffic, poi) in enumerate(zip(app_traffics, pois)):
    if np.all(app_traffic == 0):
        print(f"信息: 跳过样本 {i}，因为其流量数据完全为零。")
        continue
    # 为每个样本创建保存目录
    path_to_save = os.path.abspath(os.path.join(save_dir, str(i + 1)))
    os.makedirs(path_to_save, exist_ok=True)
    
    # 独立归一化每个特征维度
    # 使用之前计算好的每个特征的最大值
    #import pdb;pdb.set_trace()
    orig_max=np.max(app_traffic)
    
    # 收集原始app_traffic的统计数据 (图表2)
    app_traffic_max_values.append(orig_max)
    app_traffic_non_zero_values.extend(app_traffic[app_traffic > 0].flatten())
    
    log_app_traffic_sample = np.log1p(app_traffic)
    normalized_traffic = log_app_traffic_sample / log_feature_maxes[None, :]
    
    # ===== 新增：收集归一化后，进入to_gasf_cross前的数据 (图表3) =====
    normalized_traffic_max_values.append(np.max(normalized_traffic))
    normalized_traffic_non_zero_values.extend(normalized_traffic[normalized_traffic > 0].flatten())
    # =================================================================

    print("norm_traffic max:", np.max(normalized_traffic))
    #app_traffic=fast_near_min_log_normalization(app_traffic,global_max)
    #import pdb;pdb.set_trace()
    T, app_dim = normalized_traffic.shape
    _, poi_dim = poi.shape
    path_to_save = os.path.abspath(save_dir + str(i + 1))
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Compute Gramian angular fields
    flattened_list = []
    #import pdb;pdb.set_trace()
    for t in range(192):
        
        gasf = to_gasf_cross(normalized_traffic[t],poi[t])
        flattened = gasf.flatten()      
        flattened_list.append(flattened)

    image = np.stack(flattened_list, axis=0)
    image[np.isclose(image, 0)] = 0
    #import pdb;pdb.set_trace()
    """gamma = 0.25
    image = np.power(np.abs(image), gamma) * np.sign(image)
    image=(image+1)/2"""
    #import pdb;pdb.set_trace()
    # Resize the image to 128x128
    h, w = image.shape
    pad_h = max(0, TARGET_H - h)
    pad_w = max(0, TARGET_W - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    #image = (image - image.min()) / (image.max() - image.min())
    #image=np.log1p(image)
    padded_img = np.pad(image,
                    pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=-1)
    #import pdb;pdb.set_trace()
    mask = np.pad(np.ones_like(image, dtype=np.uint8),
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0)
    records.append({"index": i + 1, "orig_app_max": orig_max})
    #log_img=np.log1p(padded_img)
    """padded_img_uint8=(padded_img*255).astype(np.uint8)
    Image.fromarray(padded_img_uint8).save(os.path.join(path_to_save, f"cross_gasf_{i+1}.png"))
"""
    # 保存TIFF文件
    tifffile.imwrite(
        os.path.join(path_to_save, f"cross_gasf_{i+1}.tiff"),
        padded_img,
        photometric='minisblack'  # 对于单通道图像
    )
    
    # 保存对应的PNG可视化版本
    plt.imsave(os.path.join(path_to_save, f"visualized_{i+1}.png"), padded_img, cmap='gray')
    
    # 保存mask
    plt.imsave(os.path.join(path_to_save, f"mask_{i+1}.png"), mask, cmap='gray')
    # 检查最大值是否接近原始数据的最大值
    saved_img = tifffile.imread(os.path.join(path_to_save, f"cross_gasf_{i+1}.tiff"))
    print("traffic max:", np.max(orig_max))
    
    print("gasf max:", np.max(flattened_list))
    print("padding img:", np.max(image))    
    print("padding img min:", np.min(image))  
    print("saved img:", np.max(saved_img))  
    print(f"Saved sample {i+1}: image shape={padded_img.shape}, original shape={image.shape}")
    # 收集最终处理后图像的统计数据 (图表1)
    processed_max_values.append(np.max(padded_img))
    pixels_gt_0_5.extend(padded_img[padded_img > 0].flatten())
    # ... (省略了保存和打印信息的代码，以保持简洁)

# ===== 图表1：生成并保存处理后图像像素值的分布图 =====

if processed_max_values and pixels_gt_0_5:
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制所有>0.5像素值的分布
    weights_gt_0_5 = np.ones(len(pixels_gt_0_5)) / len(pixels_gt_0_5)
    ax.hist(pixels_gt_0_5, bins=50, color='skyblue', edgecolor='k', alpha=0.7, 
            weights=weights_gt_0_5, label='All Pixels > 0')

    # 绘制最大像素值的分布
    weights_max = np.ones(len(processed_max_values)) / len(processed_max_values)
    ax.hist(processed_max_values, bins=50, color='salmon', edgecolor='k', alpha=0.7, 
            weights=weights_max, label='Max Pixel Value per Image')

    ax.set_title('Distribution of Pixel Values in Processed Images', fontsize=16)
    ax.set_xlabel('Pixel Value', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plot_save_path = os.path.join(save_dir, "pixel_value_distribution.png")
    plt.savefig(plot_save_path)
    plt.close()

    print(f"\nDistribution plot of processed image pixel values has been saved to: {plot_save_path}")
else:
    print("\nNo data collected for pixel distribution plot.")

# ===== 图表2：生成并保存原始 app_traffics 数据的分布图 =====

if app_traffic_max_values and app_traffic_non_zero_values:
    fig, ax = plt.subplots(figsize=(12, 7))

    # 由于数据范围广，使用对数分桶(log bins)能更好地展示分布
    log_bins = np.logspace(np.log10(min(app_traffic_non_zero_values)), 
                            np.log10(max(app_traffic_max_values)), 
                            num=50)

    # 绘制所有非零流量值的分布
    weights_non_zero = np.ones(len(app_traffic_non_zero_values)) / len(app_traffic_non_zero_values)
    ax.hist(app_traffic_non_zero_values, bins=log_bins, color='limegreen', edgecolor='k', alpha=0.7, 
            weights=weights_non_zero, label='All Non-Zero Traffic Values')

    # 绘制每个样本最大流量值的分布
    weights_max_traffic = np.ones(len(app_traffic_max_values)) / len(app_traffic_max_values)
    ax.hist(app_traffic_max_values, bins=log_bins, color='gold', edgecolor='k', alpha=0.7, 
            weights=weights_max_traffic, label='Max Traffic Value per Sample')

    # 设置X轴为对数刻度
    ax.set_xscale('log')
    
    ax.set_title('Distribution of Original App Traffic Values (Log Scale)', fontsize=16)
    ax.set_xlabel('Traffic (Bytes) - Log Scale', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 保存图像
    plot_save_path_2 = os.path.join(save_dir, "app_traffic_value_distribution.png")
    plt.savefig(plot_save_path_2)
    plt.close()

    print(f"Distribution plot of original app traffic values has been saved to: {plot_save_path_2}")
else:
    print("\nNo data collected for app traffic distribution plot.")

# ===== 新增图表3：生成并保存进入 to_gasf_cross 前的数据分布图 =====
if normalized_traffic_max_values and normalized_traffic_non_zero_values:
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制所有非零值的分布
    weights_non_zero_norm = np.ones(len(normalized_traffic_non_zero_values)) / len(normalized_traffic_non_zero_values)
    ax.hist(normalized_traffic_non_zero_values, bins=50, range=(0,1), color='teal', edgecolor='k', alpha=0.7, 
            weights=weights_non_zero_norm, label='All Non-Zero Normalized Values')

    # 绘制每个样本最大值的分布
    weights_max_norm = np.ones(len(normalized_traffic_max_values)) / len(normalized_traffic_max_values)
    ax.hist(normalized_traffic_max_values, bins=50, range=(0,1), color='purple', edgecolor='k', alpha=0.7, 
            weights=weights_max_norm, label='Max Normalized Value per Sample')

    ax.set_title('Distribution of Normalized Traffic Values (Input to to_gasf_cross)', fontsize=16)
    ax.set_xlabel('Normalized Value', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 保存图像
    plot_save_path_3 = os.path.join(save_dir, "normalized_traffic_distribution.png")
    plt.savefig(plot_save_path_3)
    plt.close()

    print(f"\nDistribution plot of normalized traffic values has been saved to: {plot_save_path_3}")
else:
    print("\nNo data collected for normalized traffic distribution plot.")