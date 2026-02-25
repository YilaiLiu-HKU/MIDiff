import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

def to_gasf_cross(x, y):
    """Cross GASF transformation"""
    x_phi = np.arccos(x)
    y_phi = np.arccos(y)
    gasf_xy = np.cos(x_phi[:, None] + y_phi[None, :])
    return gasf_xy

def analyze_gasf_distribution(values, n_bins=10, save_dir="analysis_results"):
    """
    Analyze the distribution of GASF values before padding.
    Specifically handle -1 values separately and analyze with high precision (0.001%).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if len(values) > 0:
        # 分离-1值
        neg_one_mask = values == -1
        neg_one_count = np.sum(neg_one_mask)
        non_neg_one_values = values[~neg_one_mask]
        
        # 为非-1值创建区间
        import pdb;pdb.set_trace()
        min_val = non_neg_one_values.min()
        max_val = non_neg_one_values.max()
        
        # 创建区间（从-1开始）
        bin_edges = np.concatenate(([-1], np.linspace(min_val, max_val, n_bins)))
        hist, edges = np.histogram(values, bins=bin_edges)
        total_count = len(values)
        
        # 创建可视化
        plt.figure(figsize=(20, 10))
        
        # 主直方图（线性刻度）
        plt.subplot(2, 2, 1)
        plt.bar(range(len(hist)), (hist/total_count)*100)
        plt.title('GASF Value Distribution (Linear Scale)')
        plt.xlabel('Value Range')
        plt.ylabel('Percentage')
        
        # 使用精确的区间标注
        percentage_ranges = []
        for i in range(len(edges)-1):
            if i == 0 and edges[i] == -1:
                percentage_ranges.append("[-1]")
            else:
                percentage_ranges.append(f"({edges[i]:.4f}, {edges[i+1]:.4f}]")
        
        plt.xticks(range(len(hist)), percentage_ranges, rotation=45)
        
        for i, v in enumerate(hist):
            percentage = (v/total_count)*100
            if percentage >= 0.001:  # 只显示大于0.001%的标签
                plt.text(i, percentage, f'{percentage:.3f}%', ha='center', va='bottom')
        
        # 对数刻度直方图
        plt.subplot(2, 2, 2)
        plt.bar(range(len(hist)), (hist/total_count)*100)
        plt.yscale('log')
        plt.title('GASF Value Distribution (Log Scale)')
        plt.xlabel('Value Range')
        plt.ylabel('Percentage (log scale)')
        plt.xticks(range(len(hist)), percentage_ranges, rotation=45)
        
        for i, v in enumerate(hist):
            percentage = (v/total_count)*100
            if percentage > 0:  # 避免log(0)
                plt.text(i, percentage, f'{percentage:.3f}%', ha='center', va='bottom')
        
        # 累积分布图（线性刻度）
        plt.subplot(2, 2, 3)
        cumsum = np.cumsum(hist)
        plt.plot(range(len(cumsum)), (cumsum/total_count)*100, 'bo-')
        plt.title('Cumulative Distribution (Linear Scale)')
        plt.xlabel('Value Range')
        plt.ylabel('Cumulative Percentage')
        plt.xticks(range(len(hist)), percentage_ranges, rotation=45)
        
        for i, v in enumerate(cumsum):
            percentage = (v/total_count)*100
            plt.text(i, percentage, f'{percentage:.3f}%', ha='center', va='bottom')
        
        # 累积分布图（对数刻度）
        plt.subplot(2, 2, 4)
        plt.plot(range(len(cumsum)), (cumsum/total_count)*100, 'bo-')
        plt.yscale('log')
        plt.title('Cumulative Distribution (Log Scale)')
        plt.xlabel('Value Range')
        plt.ylabel('Cumulative Percentage (log scale)')
        plt.xticks(range(len(hist)), percentage_ranges, rotation=45)
        
        for i, v in enumerate(cumsum):
            percentage = (v/total_count)*100
            if percentage > 0:  # 避免log(0)
                plt.text(i, percentage, f'{percentage:.3f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gasf_value_distribution.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 保存统计信息
        with open(os.path.join(save_dir, 'gasf_value_statistics.txt'), 'w') as f:
            f.write("GASF Value Distribution Statistics (Before Padding):\n")
            f.write(f"Total values analyzed: {total_count}\n")
            f.write(f"Value range: [{min_val:.6f}, {max_val:.6f}]\n\n")
            
            # 基本统计量
            f.write("Basic Statistics:\n")
            f.write(f"Mean: {np.mean(values):.6f}\n")
            f.write(f"Median: {np.median(values):.6f}\n")
            f.write(f"Std Dev: {np.std(values):.6f}\n")
            
            # 计算详细的百分位数
            percentiles = [0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99, 99.999]
            f.write("\nDetailed Percentiles:\n")
            for p in percentiles:
                f.write(f"{p:>7}th percentile: {np.percentile(values, p):.6f}\n")
            
            # 计算峰度和偏度
            f.write("\nAdditional Statistics:\n")
            f.write(f"Skewness: {float(pd.Series(values).skew()):.6f}\n")
            f.write(f"Kurtosis: {float(pd.Series(values).kurtosis()):.6f}\n")
            
            # -1值的特别统计
            neg_one_mask = values == -1
            neg_one_count = np.sum(neg_one_mask)
            neg_one_percentage = (neg_one_count / total_count) * 100
            f.write(f"\nNegative One (-1) Statistics:\n")
            f.write(f"Count: {neg_one_count}\n")
            f.write(f"Percentage: {neg_one_percentage:.3f}%\n\n")
            
            f.write("Distribution by ranges:\n")
            for i in range(len(hist)):
                f.write(f"Range {percentage_ranges[i]}:\n")
                f.write(f"  Count: {hist[i]}\n")
                f.write(f"  Percentage: {(hist[i]/total_count)*100:.3f}%\n")
                # 计算非常小的比例
                tiny_percentage = (hist[i]/total_count)*100000
                if tiny_percentage < 1:
                    f.write(f"  Small Scale: {tiny_percentage:.6f}‰\n")
                f.write(f"  Cumulative: {(cumsum[i]/total_count)*100:.3f}%\n\n")

def main():
    # 加载数据
    save_dir = "gasf_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "dataset/all_users_data_with6cluster.npz")
    file = np.load(file_path, allow_pickle=True)
    app_traffics = file['Category_ID_Traffic (Byte)']
    pois = file["poi_labels"]
    
    # 计算特征最大值
    num_features = app_traffics.shape[2]
    feature_maxes = np.max(app_traffics, axis=(0, 1))
    zero_mask = feature_maxes == 0
    feature_maxes[zero_mask] = 1
    
    # 收集所有GASF值（在padding之前）
    all_gasf_values = []
    gamma = 0.25  # 与原始代码保持一致的gamma值
    
    for i, (app_traffic, poi) in enumerate(zip(app_traffics, pois)):
        print(f"Processing sample {i+1}/{len(app_traffics)}")
        
        # 归一化
        app_traffic = app_traffic / feature_maxes[None, :]
        
        # 对每个时间步计算GASF
        for t in range(192):
            gasf = to_gasf_cross(app_traffic[t], poi[t])
            # 应用gamma变换和归一化
            transformed = np.power(np.abs(gasf), gamma) * np.sign(gasf)
            normalized = (transformed + 1) / 2
            all_gasf_values.extend(normalized.flatten())
    
    # 转换为numpy数组
    all_gasf_values = np.array(all_gasf_values)
    
    # 分析分布
    analyze_gasf_distribution(all_gasf_values, n_bins=10, save_dir=save_dir)
    
    print(f"Analysis completed. Results saved to {save_dir}/")
    print(f"Total number of GASF values analyzed: {len(all_gasf_values)}")
    print(f"Value range: [{all_gasf_values.min():.6f}, {all_gasf_values.max():.6f}]")

if __name__ == "__main__":
    main()