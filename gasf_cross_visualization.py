import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_rgb
# 加载数据
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/all_users_data_with6cluster.npz")
file = np.load(file_path, allow_pickle=True)
app_traffics = file['Category_ID_Traffic (Byte)']
pois = file["poi_labels"]

# 创建保存目录
save_dir = "cross_gasf_classification/"
os.makedirs(save_dir, exist_ok=True)

# 定义颜色映射
# 0: 背景 (黑色) - 填充区域
# 1: app=0 & poi=0 (红色)
# 2: app=0 & poi=1 (绿色)
# 3: app!=0 & poi=1 (蓝色)
# 4: app!=0 & poi=0 (黄色)
color_dict =  {
    0: "#000000",  # 黑色 - 背景/填充
    1: "#a8dde1",  # 红色 - app=0 & poi=0
    2: "#75b5dc",  # 绿色 - app=0 & poi=1
    3: "#313772",  # 蓝色 - app!=0 & poi=1
    4:  "#326db6"  # 黄色 - app!=0 & poi=0
}

# 创建图例
def create_legend(save_path):
    fig, ax = plt.subplots(figsize=(8, 1.5))
    fig.patch.set_facecolor('white')
    categories = [
    ("app=0 & poi=0", '#4CAF50'),
    ("app=0 & poi=1", '#F0776D'),
    ("app!=0 & poi=1", '#2196F3'),
    ("app!=0 & poi=0", '#FFEB3B'),
    ("Background/Padding", '#000000')
]
    
    for i, (label, color) in enumerate(categories):
        ax.bar(0, 0, color=color, label=label)
    
    ax.legend(loc='center', ncol=3, frameon=False)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "legend.png"), bbox_inches='tight', dpi=150)
    plt.close()

# 创建全局图例
create_legend(save_dir)

# 目标图像尺寸
TARGET_H, TARGET_W = 256, 160

# 交叉变换和分类函数
def classify_cross_pixels(app, poi):
    """
    对app和poi向量进行交叉分类
    返回分类矩阵 (app_dim, poi_dim)
    """
    app_dim = len(app)
    poi_dim = len(poi)
    
    # 创建分类矩阵
    class_matrix = np.zeros((app_dim, poi_dim), dtype=np.uint8)
    
    # 创建布尔掩码
    app_zero = app == 0
    app_nonzero = ~app_zero
    poi_zero = poi == 0
    poi_nonzero = ~poi_zero
    
    # 分类逻辑
    # 1. app=0 & poi=0
    class_matrix[app_zero[:, None] & poi_zero[None, :]] = 1
    
    # 2. app=0 & poi=1
    class_matrix[app_zero[:, None] & poi_nonzero[None, :]] = 2
    
    # 3. app!=0 & poi=1
    class_matrix[app_nonzero[:, None] & poi_nonzero[None, :]] = 3
    
    # 4. app!=0 & poi=0
    class_matrix[app_nonzero[:, None] & poi_zero[None, :]] = 4
    
    return class_matrix

# 处理每个样本
for i, (app_traffic, poi_labels) in enumerate(zip(app_traffics, pois)):
    print(f"Processing sample {i+1}/{len(app_traffics)}")
    
    # 创建样本保存目录
    sample_dir = os.path.join(save_dir, f"sample_{i+1}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 存储每个时间步的分类图像
    classification_images = []
    
    # 处理每个时间步
    for t in range(app_traffic.shape[0]):
        app_vector = app_traffic[t]
        poi_vector = poi_labels[t]
        
        # 获取分类矩阵
        class_matrix = classify_cross_pixels(app_vector, poi_vector)
        classification_images.append(class_matrix)
    
    # 堆叠所有时间步
    class_image_stack = np.stack(classification_images, axis=0)
    
    # 保存原始分类堆栈 (用于后续分析)
    np.save(os.path.join(sample_dir, f"class_stack_{i+1}.npy"), class_image_stack)
    
    # 转换为彩色图像
    h, w = class_image_stack.shape[0], class_image_stack.shape[1] * class_image_stack.shape[2]
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 展平后两个维度 (app和poi)
    flat_class = class_image_stack.reshape(h, -1)
    
    # 应用颜色映射
    for class_val, hex_color in color_dict.items():
        rgb_color = (np.array(to_rgb(hex_color)) * 255).astype(np.uint8)
        mask = flat_class == class_val
        color_image[mask] = rgb_color
    
    # 调整图像大小
    current_h, current_w = color_image.shape[:2]
    
    # 计算填充
    pad_h = max(0, TARGET_H - current_h)
    pad_w = max(0, TARGET_W - current_w)
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # 填充图像 (使用黑色背景)

    
    # 保存图像
    img = Image.fromarray(color_image)
    img.save(os.path.join(sample_dir, f"cross_classification_{i+1}.png"))
    
    # 创建并保存该样本的图例
    create_legend(sample_dir)
    
    # 保存统计信息
    unique, counts = np.unique(flat_class, return_counts=True)
    stats = {class_val: count for class_val, count in zip(unique, counts)}
    
    with open(os.path.join(sample_dir, f"stats_{i+1}.txt"), 'w') as f:
        f.write("Pixel Classification Statistics:\n")
        f.write(f"Total pixels: {flat_class.size}\n")
        for class_val in sorted(stats.keys()):
            percentage = stats[class_val] / flat_class.size * 100
            f.write(f"Class {class_val}: {stats[class_val]} pixels ({percentage:.2f}%)\n")

# 收集映射前的原始数据进行统计
all_values = []
total_values = 0

# 收集所有样本的原始值信息
for i in range(len(app_traffics)):
    # 直接使用app_traffic和poi_labels的原始数据
    app_vector = app_traffics[i]  # 第i个样本的所有时间步的app数据
    poi_vector = pois[i]          # 第i个样本的所有时间步的poi数据
    
    # 收集所有时间步的数据
    for t in range(app_vector.shape[0]):
        app = app_vector[t]  # 当前时间步的app数据
        all_values.extend(app[app > 0])  # 只收集非零值
        total_values += len(app[app > 0])

pixel_values = np.array(all_values)

# 创建统计分析
def analyze_pixel_distribution(values, n_bins=10):
    """分析原始数据值的分布情况"""
    if len(values) > 0:
        min_val = values.min()
        max_val = values.max()
        
        # 先统计零值的数量（虽然我们没有收集零值，但记录一下总体情况）
        zero_count = sum(app_traffic <= 0 for app_traffic in app_traffics)
        
        # 为非零值创建区间
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        hist, edges = np.histogram(values, bins=bin_edges)
        total_count = len(values)
        
        # 创建可视化
        plt.figure(figsize=(15, 6))
        
        # 直方图
        plt.subplot(1, 2, 1)
        plt.bar(range(len(hist)), (hist/total_count)*100)
        plt.title('Original Value Distribution (Non-zero Values)')
        plt.xlabel('Value Range')
        plt.ylabel('Percentage')
        
        # 添加值标签
        # 使用科学记数法表示大数值的区间
        percentage_ranges = [f"{edges[i]:.2e}-{edges[i+1]:.2e}" for i in range(len(edges)-1)]
        plt.xticks(range(len(hist)), percentage_ranges, rotation=45)
        
        for i, v in enumerate(hist):
            percentage = (v/total_count)*100
            plt.text(i, percentage, f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 累积分布图
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(hist)
        plt.plot(range(len(cumsum)), (cumsum/total_count)*100, 'bo-')
        plt.title('Cumulative Distribution')
        plt.xlabel('Value Range')
        plt.ylabel('Cumulative Percentage')
        plt.xticks(range(len(hist)), percentage_ranges, rotation=45)
        
        for i, v in enumerate(cumsum):
            percentage = (v/total_count)*100
            plt.text(i, percentage, f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'original_value_distribution.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 保存统计信息到文件
        with open(os.path.join(save_dir, 'original_value_statistics.txt'), 'w') as f:
            f.write("Original Value Distribution Statistics:\n")
            f.write(f"Total non-zero values analyzed: {total_count}\n")
            f.write(f"Value range: {min_val:.2e} to {max_val:.2e}\n\n")
            
            # 计算一些基本统计量
            f.write("Basic Statistics:\n")
            f.write(f"Mean: {np.mean(values):.2e}\n")
            f.write(f"Median: {np.median(values):.2e}\n")
            f.write(f"Std Dev: {np.std(values):.2e}\n")
            f.write(f"25th Percentile: {np.percentile(values, 25):.2e}\n")
            f.write(f"75th Percentile: {np.percentile(values, 75):.2e}\n\n")
            
            f.write("Distribution by ranges:\n")
            for i in range(len(hist)):
                f.write(f"Range {percentage_ranges[i]}:\n")
                f.write(f"  Count: {hist[i]}\n")
                f.write(f"  Percentage: {(hist[i]/total_count)*100:.2f}%\n")
                f.write(f"  Cumulative: {(cumsum[i]/total_count)*100:.2f}%\n\n")

# 执行分析
analyze_pixel_distribution(pixel_values)

print(f"Processing completed. Results saved to {save_dir}")
print(f"Global statistics have been saved to:")
print(f"1. {save_dir}/global_statistics.txt")
print(f"2. {save_dir}/global_statistics.png")