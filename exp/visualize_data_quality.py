import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from umap import UMAP
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Termes", "Times New Roman", "Liberation Serif", "Times"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 颜色到深色版本的映射
DARK_COLOR_MAP = {
    'blue': 'darkblue',
    'green': 'darkgreen',
    'orange': 'darkorange',
    'purple': 'indigo',  # 使用indigo代替darkpurple
    'brown': 'saddlebrown',
    'pink': 'deeppink',
    'gray': 'dimgray',
    'olive': 'darkolivegreen',
    'red': 'darkred'
}

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_csv_data(data_path):
    """
    从CSV文件加载数据
    CSV格式：每行是一个样本，每3列为一组(ch1, ch2, ch3)，共192个时间步
    所以每行应该有192*3=576列
    """
    df = np.loadtxt(data_path, delimiter=',')
    B = df.shape[0]  # 样本数
    T = 192  # 时间步数
    
    # 重塑数据：(B, 576) -> (B, 192, 3)
    data = df.reshape(B, T, 3)
    
    # 提取三个通道
    ch1 = data[:, :, 0].copy()
    ch2 = data[:, :, 1].copy()
    ch3 = data[:, :, 2].copy()
    
    # 数据预处理
    ch1 = np.maximum(ch1, 0)  # 负数转0
    
    # 计算ch1的最大值（用于clip）
    ch1_max = np.max(ch1) if len(ch1) > 0 else 1.0
    ch1 = np.clip(ch1, 0, ch1_max)
    
    ch2 = np.clip(ch2, 0, 19).astype(np.int64)  # 转0-19
    ch3 = np.clip(ch3, 0, 6).astype(np.int64)  # 转0-6
    
    data = np.stack([ch1, ch2, ch3], axis=2)
    return data

def extract_features(data):
    """
    从时间序列数据中提取特征用于可视化
    数据格式: (B, 192, 3)
    返回: (B, feature_dim)
    """
    B, T, C = data.shape
    features = []
    
    for i in range(B):
        sample = data[i]  # (192, 3)
        sample_features = []
        
        # 对每个通道提取统计特征
        for c in range(C):
            channel_data = sample[:, c]
            
            # 基本统计量
            sample_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.median(channel_data),
            ])
            
            # 分位数
            sample_features.extend([
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
            ])
            
            # 非零值的统计
            nonzero = channel_data[channel_data != 0]
            if len(nonzero) > 0:
                sample_features.extend([
                    len(nonzero) / T,  # 非零比例
                    np.mean(nonzero),
                    np.std(nonzero),
                ])
            else:
                sample_features.extend([0.0, 0.0, 0.0])
            
            # 时间序列特征（如果通道是ch1，考虑趋势）
            if c == 0:  # ch1是连续值
                # 一阶差分
                diff = np.diff(channel_data)
                sample_features.extend([
                    np.mean(diff),
                    np.std(diff),
                ])
            else:  # ch2和ch3是类别
                # 类别分布
                unique, counts = np.unique(channel_data, return_counts=True)
                # 最常见的类别及其频率
                if len(unique) > 0:
                    most_common_idx = np.argmax(counts)
                    sample_features.extend([
                        unique[most_common_idx],
                        counts[most_common_idx] / T,
                    ])
                else:
                    sample_features.extend([0.0, 0.0])
        
        features.append(sample_features)
    
    return np.array(features)


def _safe_name(name):
    return os.path.splitext(os.path.basename(name))[0]


def _build_group_features(real_features, synth_features, synth_names):
    groups = [{
        'label': 'Real Data',
        'features': real_features,
        'is_real': True,
    }]
    for name, feat in zip(synth_names, synth_features):
        groups.append({
            'label': f"Synthetic: {_safe_name(name)}",
            'features': feat,
            'is_real': False,
        })
    return groups


def _split_embeddings_by_group(embeddings, group_features):
    out = []
    start = 0
    for g in group_features:
        n = len(g['features'])
        out.append({
            'label': g['label'],
            'features': g['features'],
            'embedding': embeddings[start:start+n],
            'is_real': g['is_real'],
        })
        start += n
    return out


def _add_confidence_ellipse(ax, points, color, linewidth=2.0):
    if points.shape[0] < 3:
        return
    cov = np.cov(points, rowvar=False)
    if np.isnan(cov).any() or np.isinf(cov).any():
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-9, None)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    chi2_95 = 5.991  # 95% quantile for 2D Gaussian
    width, height = 2.0 * np.sqrt(vals * chi2_95)
    mean = points.mean(axis=0)
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle='--',
        alpha=0.9,
    )
    ax.add_patch(ellipse)


def compute_embeddings_for_styleb(real_features, synth_features, synth_names):
    group_features = _build_group_features(real_features, synth_features, synth_names)
    all_features = np.vstack([g['features'] for g in group_features])

    # Manifold-oriented settings (user-preferred): smoother global geometry.
    n_total = len(all_features)
    perplexity = max(12, min(50, (n_total - 1) // 4))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        n_iter=2500,
        learning_rate='auto',
        init='pca',
        early_exaggeration=12.0,
        metric='euclidean',
    )
    emb_tsne = tsne.fit_transform(all_features)

    umap_model = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=45,
        min_dist=0.35,
        metric='cosine',
    )
    emb_umap = umap_model.fit_transform(all_features)

    tsne_groups = _split_embeddings_by_group(emb_tsne, group_features)
    umap_groups = _split_embeddings_by_group(emb_umap, group_features)
    params = {
        'tsne': {'perplexity': perplexity, 'n_iter': 2500, 'init': 'pca'},
        'umap': {'n_neighbors': 45, 'min_dist': 0.35, 'metric': 'cosine'},
    }
    return tsne_groups, umap_groups, params


def _apply_soft_grid(ax, grid_n=6):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xticks(np.linspace(x0, x1, grid_n))
    ax.set_yticks(np.linspace(y0, y1, grid_n))
    ax.grid(True, alpha=0.22, linestyle='-', linewidth=0.7, color='#9f9f9f')
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    ax.set_facecolor('#fbfbfb')


def _darken_color(color, factor=0.78):
    rgb = np.array(mcolors.to_rgb(color))
    dark_rgb = np.clip(rgb * factor, 0.0, 1.0)
    return tuple(dark_rgb.tolist())


def _plot_styleb_panel(ax, groups, colors, title=""):
    for color, g in zip(colors, groups):
        points = g['embedding']
        ax.scatter(points[:, 0], points[:, 1], s=9, c=[color], alpha=0.58, linewidths=0)
        _add_confidence_ellipse(ax, points, color=color, linewidth=1.7)

    if title:
        ax.set_title(title, fontsize=15, fontweight='normal')
    _apply_soft_grid(ax)


def _add_group_legend(ax, labels, colors, fontsize=10, markerscale=1.0, loc='lower right'):
    handles = []
    for label, color in zip(labels, colors):
        h = plt.Line2D([0], [0], marker='o', color='w', label=label,
                       markerfacecolor=color, markersize=7, alpha=0.9)
        handles.append(h)
    legend = ax.legend(
        handles=handles,
        loc=loc,
        frameon=True,
        fontsize=fontsize,
        markerscale=markerscale,
        borderpad=0.35,
        labelspacing=0.28,
        handletextpad=0.35,
    )
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(0.86)


def visualize_styleb_base(real_features, synth_features, synth_names, output_dir):
    """
    Final style version: keep only styleB_base and output
    - t-SNE single figure
    - UMAP single figure
    - combined t-SNE + UMAP figure
    """
    print("\nGenerating styleB_base visualizations...")
    tsne_groups, umap_groups, _ = compute_embeddings_for_styleb(real_features, synth_features, synth_names)

    labels = [g['label'] for g in tsne_groups]
    # High-saturation palette as requested.
    palette = ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A', '#F0027F', '#A65628', '#B15928', '#17BECF']
    colors = [palette[i % len(palette)] for i in range(len(labels))]
    dark_colors = [_darken_color(c, factor=0.78) for c in colors]

    # styleB t-SNE single
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 10))
    _plot_styleb_panel(ax1, tsne_groups, dark_colors, title="")
    _add_group_legend(ax1, labels, dark_colors, fontsize=18, markerscale=1.5)
    plt.tight_layout()
    out_tsne = os.path.join(output_dir, 'styleB_base_tsne_visualization.pdf')
    plt.savefig(out_tsne, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_tsne}")

    # styleB UMAP single
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 10))
    _plot_styleb_panel(ax2, umap_groups, dark_colors, title="")
    _add_group_legend(ax2, labels, dark_colors, fontsize=18, markerscale=1.5, loc='upper left')
    plt.tight_layout()
    out_umap = os.path.join(output_dir, 'styleB_base_umap_visualization.pdf')
    plt.savefig(out_umap, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_umap}")

    # styleB combined
    fig3, axes = plt.subplots(1, 2, figsize=(22, 6))
    _plot_styleb_panel(axes[0], tsne_groups, dark_colors, 't-SNE')
    _plot_styleb_panel(axes[1], umap_groups, dark_colors, 'UMAP')
    _add_group_legend(axes[1], labels, dark_colors, fontsize=9, markerscale=1.2, loc='upper left')
    fig3.suptitle('StyleB Base: t-SNE + UMAP', fontsize=16, fontweight='normal', y=1.06)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_combined = os.path.join(output_dir, 'styleB_base_combined_visualization.pdf')
    plt.savefig(out_combined, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_combined}")

def visualize_with_tsne(real_features, synth_features, synth_names, output_dir):
    """
    使用t-SNE进行可视化
    """
    print("正在运行t-SNE...")
    
    # 合并数据（synth_features是一个列表，需要先合并所有合成数据）
    synth_features_combined = np.vstack(synth_features) if len(synth_features) > 0 else np.array([])
    all_features = np.vstack([real_features, synth_features_combined])
    
    # 运行t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=1000)
    embeddings = tsne.fit_transform(all_features)
    
    # 分离real和synthetic的嵌入
    n_real = len(real_features)
    real_emb = embeddings[:n_real]
    synth_embs = []
    start_idx = n_real
    for i, name in enumerate(synth_names):
        n_samples = len(synth_features[i])
        synth_embs.append(embeddings[start_idx:start_idx+n_samples])
        start_idx += n_samples
    
    # 绘制可视化图（扁平化：宽高比约 2:1）
    plt.figure(figsize=(16, 10))
    
    # 绘制real data
    plt.scatter(real_emb[:, 0], real_emb[:, 1], 
                c='red', label='Real Data', alpha=0.6, s=30, edgecolors='darkred', linewidths=0.5)
    
    # 绘制各个合成数据
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, (name, synth_emb) in enumerate(zip(synth_names, synth_embs)):
        color = colors[i % len(colors)]
        plt.scatter(synth_emb[:, 0], synth_emb[:, 1], 
                   c=color, label=f'Synthetic: {os.path.splitext(os.path.basename(name))[0]}', 
                   alpha=0.6, s=30, edgecolors=DARK_COLOR_MAP.get(color, 'black'), linewidths=0.5)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=30)
    plt.ylabel('t-SNE Dimension 2', fontsize=30)# 坐标轴刻度字体大小
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.legend(loc='best', fontsize=25, markerscale=2.0)  # markerscale增大图例中点的大小
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'tsne_visualization.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"t-SNE可视化结果已保存到: {output_path}")
    plt.close()

def visualize_with_umap(real_features, synth_features, synth_names, output_dir):
    """
    使用UMAP进行可视化
    """
    print("正在运行UMAP...")
    
    # 合并数据（synth_features是一个列表，需要先合并所有合成数据）
    synth_features_combined = np.vstack(synth_features) if len(synth_features) > 0 else np.array([])
    all_features = np.vstack([real_features, synth_features_combined])
    
    # 运行UMAP
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings = umap_model.fit_transform(all_features)
    
    # 分离real和synthetic的嵌入
    n_real = len(real_features)
    real_emb = embeddings[:n_real]
    synth_embs = []
    start_idx = n_real
    for i, name in enumerate(synth_names):
        n_samples = len(synth_features[i])
        synth_embs.append(embeddings[start_idx:start_idx+n_samples])
        start_idx += n_samples
    
    # 绘制可视化图（扁平化：宽高比约 2:1）
    plt.figure(figsize=(16, 10))
    
    # 绘制real data
    plt.scatter(real_emb[:, 0], real_emb[:, 1], 
                c='red', label='Real Data', alpha=0.6, s=30, edgecolors='darkred', linewidths=0.5)
    
    # 绘制各个合成数据
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, (name, synth_emb) in enumerate(zip(synth_names, synth_embs)):
        color = colors[i % len(colors)]
        plt.scatter(synth_emb[:, 0], synth_emb[:, 1], 
                   c=color, label=f'Synthetic: {os.path.splitext(os.path.basename(name))[0]}', 
                   alpha=0.6, s=30, edgecolors=DARK_COLOR_MAP.get(color, 'black'), linewidths=0.5)
    plt.xlabel('UMAP Dimension 1', fontsize=30)
    plt.ylabel('UMAP Dimension 2', fontsize=30)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.legend(loc='best', fontsize=25, markerscale=2.0)  # markerscale增大图例中点的大小
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'umap_visualization.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"UMAP可视化结果已保存到: {output_path}")
    plt.close()

def visualize_combined(real_features, synth_features, synth_names, output_dir):
    """
    在同一张图中显示t-SNE和UMAP的结果（子图形式）
    """
    print("正在生成组合可视化图...")
    
    # t-SNE
    print("  运行t-SNE...")
    synth_features_combined = np.vstack(synth_features) if len(synth_features) > 0 else np.array([])
    all_features_tsne = np.vstack([real_features, synth_features_combined])
    tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=15,        # 降低！尤其当总样本 < 2000
    n_iter=1500,          # 增加迭代次数确保收敛
    learning_rate='auto', # 自动选择学习率（sklearn 1.2+）
    init='pca',           # 用 PCA 初始化，更稳定
    early_exaggeration=12.0,
    metric='euclidean'
)
    embeddings_tsne = tsne.fit_transform(all_features_tsne)
    
    n_real = len(real_features)
    real_emb_tsne = embeddings_tsne[:n_real]
    synth_embs_tsne = []
    start_idx = n_real
    for i, name in enumerate(synth_names):
        n_samples = len(synth_features[i])
        synth_embs_tsne.append(embeddings_tsne[start_idx:start_idx+n_samples])
        start_idx += n_samples
    
    # UMAP
    print("  运行UMAP...")
    all_features_umap = np.vstack([real_features, synth_features_combined])
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_umap = umap_model.fit_transform(all_features_umap)
    
    real_emb_umap = embeddings_umap[:n_real]
    synth_embs_umap = []
    start_idx = n_real
    for i, name in enumerate(synth_names):
        n_samples = len(synth_features[i])
        synth_embs_umap.append(embeddings_umap[start_idx:start_idx+n_samples])
        start_idx += n_samples
    
    # 绘制组合图（扁平化：单行两列，整体更扁）
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # t-SNE子图
    ax1 = axes[0]
    ax1.scatter(real_emb_tsne[:, 0], real_emb_tsne[:, 1], 
                c='red', label='Real Data', alpha=0.6, s=20, edgecolors='darkred', linewidths=0.5)
    for i, (name, synth_emb) in enumerate(zip(synth_names, synth_embs_tsne)):
        color = colors[i % len(colors)]
        ax1.scatter(synth_emb[:, 0], synth_emb[:, 1], 
                   c=color, label=f'Synthetic: {os.path.splitext(os.path.basename(name))[0]}', 
                   alpha=0.6, s=20, edgecolors=DARK_COLOR_MAP.get(color, 'black'), linewidths=0.5)
    ax1.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.legend(loc='best', fontsize=9, markerscale=1.5)  # markerscale增大图例中点的大小
    ax1.grid(True, alpha=0.3)

    # UMAP子图
    ax2 = axes[1]
    ax2.scatter(real_emb_umap[:, 0], real_emb_umap[:, 1], 
                c='red', label='Real Data', alpha=0.6, s=20, edgecolors='darkred', linewidths=0.5)
    for i, (name, synth_emb) in enumerate(zip(synth_names, synth_embs_umap)):
        color = colors[i % len(colors)]
        ax2.scatter(synth_emb[:, 0], synth_emb[:, 1], 
                   c=color, label=f'Synthetic: {os.path.splitext(os.path.basename(name))[0]}', 
                   alpha=0.6, s=20, edgecolors=DARK_COLOR_MAP.get(color, 'black'), linewidths=0.5)
    ax2.set_title('UMAP Visualization', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.legend(loc='best', fontsize=9, markerscale=1.5)  # markerscale增大图例中点的大小
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Data Quality Evaluation: Real Data vs Synthetic Data', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'combined_visualization.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"组合可视化结果已保存到: {output_path}")
    plt.close()

def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate t-SNE/UMAP data quality visualizations.")
    parser.add_argument("--real-file", default=str(repo_root / "data" / "our.csv"))
    parser.add_argument(
        "--synth-files",
        nargs="+",
        default=[
            str(repo_root / "exp" / "results" / "MIDiff.csv"),
        ],
    )
    parser.add_argument("--output-dir", default=str(repo_root / "exp" / "results" / "manifold"))
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--styleb-only",
        action="store_true",
        help="Only export the paper-used styleB_base t-SNE/UMAP figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    real_path = args.real_file
    synth_path_list = args.synth_files
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("数据质量评估：Real Data vs Synthetic Data")
    print("=" * 60)
    
    # 加载real data
    print(f"\n正在加载real data: {real_path}")
    if not os.path.exists(real_path):
        print(f"错误: 找不到real data文件: {real_path}")
        return
    
    real_data = load_csv_data(real_path)
    print(f"  Real data shape: {real_data.shape}")
    
    # 加载合成数据
    synth_data_list = []
    synth_names = []
    for synth_path in synth_path_list:
        if os.path.exists(synth_path):
            print(f"\n正在加载合成数据: {synth_path}")
            synth_data = load_csv_data(synth_path)
            print(f"  Synthetic data shape: {synth_data.shape}")
            synth_data_list.append(synth_data)
            synth_names.append(synth_path)
        else:
            print(f"警告: 找不到合成数据文件: {synth_path}，跳过")
    
    if len(synth_data_list) == 0:
        print("错误: 没有找到任何合成数据文件")
        return
    
    # 提取特征
    print("\n正在提取特征...")
    print("  提取real data特征...")
    real_features = extract_features(real_data)
    synth_features_list = [extract_features(data) for data in synth_data_list]
    
    # === 新增：特征标准化（关键步骤！） ===
    print("\n" + "="*60)
    print("正在标准化特征...")
    print("="*60)
    
    # 1. 合并所有特征（real + 所有synthetic）
    all_features = [real_features] + synth_features_list
    combined_features = np.vstack(all_features)
    
    # 2. 应用标准化（使用real data的分布统计量！）
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    
    # 3. 拆分回原始结构
    n_real = len(real_features)
    real_scaled = scaled_features[:n_real]
    
    synth_scaled = []
    start_idx = n_real
    for i, feat in enumerate(synth_features_list):
        end_idx = start_idx + len(feat)
        synth_scaled.append(scaled_features[start_idx:end_idx])
        start_idx = end_idx
    
    # 4. 用标准化后的特征替换原始特征（重要！）
    real_features = real_scaled
    synth_features_list = synth_scaled
    
    print(f"标准化后特征维度: {real_features.shape[1]}")
    print(f"标准化后real data样本数: {len(real_features)}")
    for i, name in enumerate(synth_names):
        print(f"标准化后{os.path.basename(name)}样本数: {len(synth_features_list[i])}")
    
    # 为了可视化，可能需要采样（如果数据量太大）
    max_samples = args.max_samples
    
    if len(real_features) > max_samples:
        print(f"\nReal data样本数({len(real_features)})超过{max_samples}，进行随机采样...")
        indices = np.random.choice(len(real_features), max_samples, replace=False)
        real_features = real_features[indices]
        print(f"  采样后shape: {real_features.shape}")
    
    for i in range(len(synth_features_list)):
        if len(synth_features_list[i]) > max_samples:
            print(f"\n合成数据 {synth_names[i]} 样本数({len(synth_features_list[i])})超过{max_samples}，进行随机采样...")
            indices = np.random.choice(len(synth_features_list[i]), max_samples, replace=False)
            synth_features_list[i] = synth_features_list[i][indices]
            print(f"  采样后shape: {synth_features_list[i].shape}")
    
    # 可视化
    print("\n" + "=" * 60)
    print("开始可视化...")
    print("=" * 60)
    
    if args.styleb_only:
        visualize_styleb_base(real_features, synth_features_list, synth_names, output_dir)
    else:
        visualize_with_tsne(real_features, synth_features_list, synth_names, output_dir)
        visualize_with_umap(real_features, synth_features_list, synth_names, output_dir)
        visualize_combined(real_features, synth_features_list, synth_names, output_dir)
        visualize_styleb_base(real_features, synth_features_list, synth_names, output_dir)
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"\n所有结果已保存到: {output_dir}/")
    print("  - tsne_visualization.pdf: t-SNE可视化结果")
    print("  - umap_visualization.pdf: UMAP可视化结果")
    print("  - combined_visualization.pdf: 组合可视化结果（包含t-SNE和UMAP）")
    print("  - styleB_base_tsne_visualization.pdf")
    print("  - styleB_base_umap_visualization.pdf")
    print("  - styleB_base_combined_visualization.pdf")

if __name__ == "__main__":
    main()
