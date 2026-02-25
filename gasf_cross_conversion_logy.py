import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import tifffile
import matplotlib.ticker as mticker # 导入用于格式化坐标轴的库

#data_dir = "Youtube/vid"  # directory where 1D traces are stored (Format : Youtube/vid*)
save_dir = "tiff_log/"  # directory to store GASF converted data as PNG
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
T=192
def minmax_scale(ts):
    return (ts - np.min(ts)) / (np.max(ts) - np.min(ts))

def to_gasf_cross(x, y):
    x_phi = np.arccos(x)
    y_phi = np.arccos(y)
    gasf_xy = np.cos(x_phi[:, None] + y_phi[None, :])
    return gasf_xy
def is_strict_one_hot(poi_vector):
    """
    检查一个向量是否为严格的独热编码格式 (有且仅有一个1, 其余为0).
    """
    # 转换为numpy数组以进行高效计算
    vec = np.asarray(poi_vector)
    
    # 条件1: 向量中所有元素的和必须精确等于1
    is_sum_one = np.isclose(vec.sum(), 1.0)
    
    # 条件2: 向量中只能包含0和1
    # np.all()会检查vec中所有元素是否都满足括号内的条件
    # (vec == 0) | (vec == 1) 会生成一个布尔数组, 如 [True, True, False]
    contains_only_0_and_1 = np.all((vec == 0) | (vec == 1))
    
    return is_sum_one and contains_only_0_and_1



####create norm img
app_dim = app_traffics.shape[2]
poi_dim = pois.shape[2]

# 构造全0流量；log1p(0)=0，除以 log_feature_maxes 仍为0
zero_app = np.zeros((T, app_dim), dtype=app_traffics.dtype)
norm_zero = zero_app  # 这里就是 0

# 构造 POI：每个时间步独热在第0位
poi_zero = np.zeros((T, poi_dim), dtype=pois.dtype)
poi_zero[:, 0] = 1

# 生成 GASF cross
flat = []
for t in range(T):
    g = to_gasf_cross(norm_zero[t], poi_zero[t])
    flat.append(g.flatten())
image = np.stack(flat, axis=0)

# 与主流程保持一致的 padding + 保存
h, w = image.shape
pad_h = max(0, TARGET_H - h); pad_w = max(0, TARGET_W - w)
pad_top = pad_h // 2; pad_bottom = pad_h - pad_top
pad_left = pad_w // 2; pad_right = pad_w - pad_left
padded_img = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0)

extra_dir = os.path.abspath(os.path.join(save_dir, "_norm"))
os.makedirs(extra_dir, exist_ok=True)
tifffile.imwrite(os.path.join(extra_dir, "norm_img.tiff"),
                 padded_img, photometric='minisblack')
plt.imsave(os.path.join(extra_dir, f"norm_visualized.png"), padded_img, cmap='gray')

for i, (app_traffic, poi) in enumerate(zip(app_traffics, pois)):
    if np.all(app_traffic == 0):
        print(f"信息: 跳过样本 {i}，因为其流量数据完全为零。")
        continue
    # 为每个样本创建保存目录
    path_to_save = os.path.abspath(os.path.join(save_dir, str(i + 1)))
    os.makedirs(path_to_save, exist_ok=True)
    print(is_strict_one_hot(poi[0]))
    
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