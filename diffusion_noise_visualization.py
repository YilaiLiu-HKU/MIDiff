import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pathlib import Path
import math

# =============================================================================
# Diffusion 加噪相关函数
# =============================================================================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    获取预定义的 beta 调度
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    创建 beta 调度
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normalize_to_minus_one_one(data):
    """
    将数据归一化到 [-1, 1] 范围
    支持任意输入范围的数据
    """
    # 找到数据的最大值和最小值
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        # 如果所有值都相同，返回零数组
        return np.zeros_like(data, dtype=np.float64)
    
    # 如果数据已经在 [-1, 1] 范围内（允许小的误差），直接返回
    if data_min >= -1.01 and data_max <= 1.01:
        return data.astype(np.float64)
    
    # 归一化到 [0, 1]，然后转换到 [-1, 1]
    normalized = (data - data_min) / (data_max - data_min)  # [0, 1]
    normalized = 2.0 * normalized - 1.0  # [-1, 1]
    
    return normalized.astype(np.float64)


def denormalize_from_minus_one_one(normalized_data, original_min, original_max):
    """
    将 [-1, 1] 范围的数据反归一化回原始范围
    """
    # 从 [-1, 1] 转换到 [0, 1]
    zero_one = (normalized_data + 1.0) / 2.0
    
    # 转换回原始范围
    denormalized = zero_one * (original_max - original_min) + original_min
    
    return denormalized


def add_noise_to_data(x_start, t, betas, noise=None, normalize=True):
    """
    对数据添加噪声（diffusion 前向过程）
    
    Args:
        x_start: 原始数据 [T, ...] 或 [B, T, ...]
        t: 时间步（标量）
        betas: beta 调度数组
        noise: 可选的噪声，如果为 None 则随机生成
        normalize: 是否先归一化到 [-1, 1] 范围
    
    Returns:
        加噪后的数据（与 x_start 相同的形状和范围）
    """
    # 确保 t 是标量
    t = int(np.clip(t, 0, len(betas) - 1))
    
    # 保存原始范围（用于反归一化）
    if normalize:
        original_min = np.min(x_start)
        original_max = np.max(x_start)
        # 归一化到 [-1, 1]
        x_normalized = normalize_to_minus_one_one(x_start)
    else:
        x_normalized = x_start.astype(np.float64)
        original_min = None
        original_max = None
    
    # 计算 alpha 相关参数
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    # 提取对应时间步的参数（标量）
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod[t])
    
    # 生成噪声（在 [-1, 1] 范围内）
    if noise is None:
        noise = np.random.randn(*x_normalized.shape).astype(np.float64)
    
    # 应用加噪公式: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    # 注意：噪声也是在归一化空间中的
    x_t_normalized = sqrt_alphas_cumprod * x_normalized + sqrt_one_minus_alphas_cumprod * noise
    
    # 反归一化回原始范围
    if normalize:
        x_t = denormalize_from_minus_one_one(x_t_normalized, original_min, original_max)
    else:
        x_t = x_t_normalized
    
    return x_t


# =============================================================================
# 可视化相关函数（从 gasf_cross_visualization.py 复用）
# =============================================================================

# 定义颜色映射
color_dict = {
    0: "#000000",  # 黑色 - 背景/填充
    1: "#a8dde1",  # app=0 & poi=0
    2: "#75b5dc",  # app=0 & poi=1
    3: "#313772",  # app!=0 & poi=1
    4: "#326db6"   # app!=0 & poi=0
}


def classify_cross_pixels(app, poi, threshold=None):
    """
    对app和poi向量进行交叉分类
    返回分类矩阵 (app_dim, poi_dim)
    
    Args:
        app: app 向量
        poi: poi 向量
        threshold: 判断"零值"的阈值。如果为 None，则自动计算（使用中位数的一半）
    """
    app_dim = len(app)
    poi_dim = len(poi)
    
    # 创建分类矩阵
    class_matrix = np.zeros((app_dim, poi_dim), dtype=np.uint8)
    
    # 自动确定阈值（如果未提供）
    app_abs = np.abs(app)
    poi_abs = np.abs(poi)
    
    if threshold is None:
        # 对于原始数据，使用绝对值阈值
        # 对于加噪数据，使用相对阈值（基于数据分布）
        
        # 如果数据看起来是离散的（很多精确的0），使用小的阈值
        if np.sum(app_abs < 1e-6) > len(app) * 0.1:  # 如果超过10%的值接近0
            app_threshold = 1e-6
        else:
            app_threshold = np.median(app_abs[app_abs > 0]) * 0.1 if np.any(app_abs > 0) else 1e-6
        
        if np.sum(poi_abs < 1e-6) > len(poi) * 0.1:
            poi_threshold = 1e-6
        else:
            poi_threshold = np.median(poi_abs[poi_abs > 0]) * 0.1 if np.any(poi_abs > 0) else 1e-6
    else:
        app_threshold = threshold
        poi_threshold = threshold
    
    # 创建布尔掩码
    app_zero = app_abs < app_threshold
    app_nonzero = ~app_zero
    poi_zero = np.abs(poi) < poi_threshold
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


def create_classification_rgb_image(app_data, poi_data):
    """
    使用分类矩阵和颜色映射创建RGB图像（像 gasf_cross_visualization.py 那样）
    
    Args:
        app_data: app数据 [T, app_dim]
        poi_data: poi数据 [T, poi_dim]
    
    Returns:
        RGB图像 [T, app_dim * poi_dim, 3]，值在 [0, 255] 范围
    """
    T, app_dim = app_data.shape
    _, poi_dim = poi_data.shape
    
    # 存储每个时间步的分类图像
    classification_images = []
    
    # 处理每个时间步
    for t in range(T):
        app_vector = app_data[t]
        poi_vector = poi_data[t]
        
        # 获取分类矩阵
        class_matrix = classify_cross_pixels(app_vector, poi_vector)
        classification_images.append(class_matrix)
    
    # 堆叠所有时间步
    class_image_stack = np.stack(classification_images, axis=0)  # [T, app_dim, poi_dim]
    
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
    
    return color_image


def add_noise_to_rgb_image(rgb_image, t, betas, noise=None):
    """
    对RGB图像添加噪声（在像素值空间）
    
    Args:
        rgb_image: RGB图像 [H, W, 3]，值在 [0, 255] 范围
        t: 时间步（标量）
        betas: beta 调度数组
        noise: 可选的噪声，如果为 None 则随机生成
    
    Returns:
        加噪后的RGB图像 [H, W, 3]，值在 [0, 255] 范围
    """
    # 将RGB图像归一化到 [-1, 1] 范围
    rgb_normalized = (rgb_image.astype(np.float64) / 127.5) - 1.0  # [0, 255] -> [-1, 1]
    
    # 确保 t 是标量
    t = int(np.clip(t, 0, len(betas) - 1))
    
    # 计算 alpha 相关参数
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    # 提取对应时间步的参数（标量）
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod[t])
    
    # 生成噪声（在 [-1, 1] 范围内）
    if noise is None:
        noise = np.random.randn(*rgb_normalized.shape).astype(np.float64)
    
    # 应用加噪公式: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    rgb_noisy_normalized = sqrt_alphas_cumprod * rgb_normalized + sqrt_one_minus_alphas_cumprod * noise
    
    # 反归一化回 [0, 255] 范围
    rgb_noisy = ((rgb_noisy_normalized + 1.0) / 2.0 * 255.0).astype(np.uint8)
    rgb_noisy = np.clip(rgb_noisy, 0, 255)  # 确保值在有效范围内
    
    return rgb_noisy


def visualize_rgb_image(rgb_image, save_path, sample_idx, timestep=None):
    """
    保存RGB图像
    
    Args:
        rgb_image: RGB图像 [H, W, 3]
        save_path: 保存路径
        sample_idx: 样本索引
        timestep: 时间步（可选）
    """
    img = Image.fromarray(rgb_image)
    if timestep is not None:
        filename = f"diffusion_t{timestep:04d}_sample_{sample_idx}.png"
    else:
        filename = f"rgb_image_sample_{sample_idx}.png"
    img.save(os.path.join(save_path, filename))


def create_noise_progression_visualization(app_traffic, poi_labels, sample_idx, 
                                          output_dir, num_diffusion_timesteps=1000,
                                          num_visualization_steps=10):
    """
    创建 diffusion 加噪过程的可视化
    
    Args:
        app_traffic: app 流量数据 [T, app_dim]
        poi_labels: poi 标签数据 [T, poi_dim]
        sample_idx: 样本索引
        output_dir: 输出目录
        num_diffusion_timesteps: diffusion 总时间步数
        num_visualization_steps: 要可视化的时间步数量
    """
    print(f"  创建样本 {sample_idx} 的 diffusion 加噪可视化...")
    
    # 创建样本保存目录
    sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 初始化 diffusion 参数
    betas = get_named_beta_schedule("linear", num_diffusion_timesteps)
    
    # 选择要可视化的时间步（特定时间步）
    visualization_timesteps = np.array([0, 5, 10, 50, 100, 200, 500, 999], dtype=int)
    # 确保时间步不超过最大值
    visualization_timesteps = np.clip(visualization_timesteps, 0, num_diffusion_timesteps - 1)
    
    # 先使用分类矩阵和颜色映射生成原始RGB图像
    print(f"    生成原始分类RGB图像 (t=0)...")
    original_rgb_image = create_classification_rgb_image(
        app_traffic.astype(np.float64),
        poi_labels.astype(np.float64)
    )
    
    # 保存原始图像（t=0）
    visualize_rgb_image(original_rgb_image, sample_dir, sample_idx, timestep=0)
    
    # 对每个可视化时间步进行加噪（在RGB图像空间）
    for vis_t in visualization_timesteps[1:]:  # 跳过 t=0（已处理）
        print(f"    处理 t={vis_t}/{num_diffusion_timesteps-1}...")
        
        # 对RGB图像进行加噪（在像素值空间）
        noisy_rgb_image = add_noise_to_rgb_image(
            original_rgb_image,
            vis_t,
            betas,
            noise=None
        )
        
        # 保存加噪后的RGB图像
        visualize_rgb_image(noisy_rgb_image, sample_dir, sample_idx, timestep=vis_t)
    
    # 创建对比图（将所有时间步放在一起）
    create_comparison_grid(sample_dir, sample_idx, visualization_timesteps, 
                          num_diffusion_timesteps)


def create_comparison_grid(sample_dir, sample_idx, timesteps, num_diffusion_timesteps):
    """
    创建所有时间步的对比网格图
    """
    num_steps = len(timesteps)
    cols = min(5, num_steps)
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes[None, :] if cols > 1 else axes[None, None]
    elif cols == 1:
        axes = axes[:, None]
    
    axes = axes.flatten()
    
    for idx, t in enumerate(timesteps):
        img_path = os.path.join(sample_dir, f"diffusion_t{t:04d}_sample_{sample_idx}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"t={t}\n({t/num_diffusion_timesteps*100:.1f}%)", 
                               fontsize=10)
            axes[idx].axis('off')
        else:
            axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(timesteps), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, f"noise_progression_sample_{sample_idx}.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    # 配置参数
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "dataset/all_users_data_with6cluster.npz")
    output_dir = "diffusion_noise_visualization/"
    num_samples = 5  # 要处理的样本数量
    num_diffusion_timesteps = 1000  # diffusion 总时间步数
    num_visualization_steps = 10  # 要可视化的时间步数量
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    if not os.path.exists(data_file):
        print(f"错误：找不到数据文件 {data_file}")
        print("请确保数据文件存在，或修改 data_file 路径")
        exit(1)
    
    print(f"正在加载数据从: {data_file}")
    file = np.load(data_file, allow_pickle=True)
    app_traffics = file['Category_ID_Traffic (Byte)']
    pois = file["poi_labels"]
    
    print(f"数据加载完成。样本数量: {len(app_traffics)}")
    print(f"将处理前 {num_samples} 个样本")
    print(f"Diffusion 时间步数: {num_diffusion_timesteps}")
    print(f"可视化时间步数: {num_visualization_steps}")
    print("-" * 60)
    
    # 处理每个样本
    for i in range(min(num_samples, len(app_traffics))):
        print(f"\n处理样本 {i+1}/{min(num_samples, len(app_traffics))}")
        app_traffic = app_traffics[i]
        poi_labels = pois[i]
        
        create_noise_progression_visualization(
            app_traffic=app_traffic,
            poi_labels=poi_labels,
            sample_idx=i+1,
            output_dir=output_dir,
            num_diffusion_timesteps=num_diffusion_timesteps,
            num_visualization_steps=num_visualization_steps
        )
    
    print("\n" + "=" * 60)
    print(f"处理完成！结果已保存到: {output_dir}")
    print(f"每个样本包含:")
    print(f"  - 各个时间步的单独图像 (diffusion_tXXXX_sample_X.png)")
    print(f"  - 时间步对比网格图 (noise_progression_sample_X.png)")
