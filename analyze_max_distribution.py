import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Tuple, Optional, List

# s2 (app_category) 0–19 统一标签（不含 7），供分布计算与聚合图使用
S2_LABEL_MAP_0_19 = {
    0.0: "Utilities", 1.0: "Games", 2.0: "Fun", 3.0: "News",
    4.0: "Social", 5.0: "Shopping", 6.0: "Finance",
    8.0: "Travel", 9.0: "Lifestyle", 10.0: "Education", 11.0: "Health",
    12.0: "Infant", 13.0: "Navigation", 14.0: "Weather", 15.0: "Music",
    16.0: "References", 17.0: "Books", 18.0: "Photo", 19.0: "Sports"
}
# 类别顺序：0–19 排除 7
S2_CATEGORY_ORDER = list(range(7)) + list(range(8, 20))

# 导入词云库
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Warning: wordcloud library not available. Install with: pip install wordcloud")
def plot_s2_distribution(df: pd.DataFrame, 
                         model_name: str, 
                         output_dir: Path, 
                         master_labels: List[str], 
                         is_reference: bool = False,
                         s1_threshold: Optional[float] = None):
    """
    创建 s2 (app_category) 的总体分布条形图。
    仅统计 app_flow > s1_threshold 的时间点；s1_threshold 通常为 0.7 * real data 的 app_flow 非零最小值；None 则用 app_flow!=0。
    """
    if 'app_category' not in df.columns:
        print(f"Skipping s2 overall distribution for {model_name} due to missing 'app_category'.")
        return

    label_map = {k: v.replace("_", " ").strip() for k, v in S2_LABEL_MAP_0_19.items()}
    label_map.update({int(k): v for k, v in label_map.items()})  # 兼容 int 索引
    master_labels_cleaned = [label_map.get(float(k), str(k)) for k in S2_CATEGORY_ORDER]

    # 仅统计 app_flow > s1_threshold 的时间点（None 则 app_flow!=0）（与 calculate_nonzero_frequency 的“非零”定义一致）
    if 'app_flow' in df.columns:
        if s1_threshold is not None:
            df_plot = df[df['app_flow'] > s1_threshold]
        else:
            df_plot = df[df['app_flow'] != 0]
        if len(df_plot) == 0:
            print(f"Skipping s2 overall distribution for {model_name} as no rows with app_flow > threshold.")
            return
        s2 = df_plot['app_category']
    else:
        s2 = df['app_category']

    # 计算原始百分比
    s2_dist = s2.value_counts(normalize=True) * 100

    # 统一映射到名称
    s2_dist_labeled = s2_dist.rename(index=label_map)

    # 统一 reindex 到 master_labels
    data_to_plot = s2_dist_labeled.reindex(master_labels_cleaned, fill_value=0.0)

    # Task 3: 如果是 reference，则丢弃0值
    if is_reference:
        data_to_plot = data_to_plot[data_to_plot > 0]
    
    if data_to_plot.empty:
         print(f"Skipping s2 overall distribution for {model_name} as no data remains after filtering.")
         return

    # --- [MODIFIED] Task 6: 动态字体大小 (已放大) ---
    num_bars = len(data_to_plot)
    if num_bars > 14: # 如果条形图很多
        tick_fontsize = 23     # X 轴刻度 (原为 10)
        # annotation_fontsize = 8  (已移除)
        # annotation_offset = (0, 9) (已移除)
    else: # 如果条形图较少
        tick_fontsize = 24     # (原为 20)
        # annotation_fontsize = 16 (已移除)
        # annotation_offset = (0, 18) (已移除)
    # --- 动态字体结束 ---

    plt.figure(figsize=(15, 8)) 
    
    # Task 3: 使用更深的蓝色和条纹 (hatch)
    ax = data_to_plot.plot(kind='bar', color='royalblue', hatch='//', alpha=0.9, edgecolor='black')
    ax.set_xlabel(None)
    # [REMOVED] 移除 X 轴标签
    # plt.xlabel('App Category', fontsize=20) 
    
    # [REMOVED] 移除 Y 轴标签
    plt.ylabel('Proportion (%)', fontsize=27) 
    
    # Task 1: 将 Y 轴固定为 0-100
    ax.set_ylim(0, 100) 
    
    # [MODIFIED] 放大 Y 轴刻度文字
    plt.yticks(fontsize=25)

    # [REMOVED] Task 4: 移除柱状图顶部的百分比文字
    # for p in ax.patches:
    #     ... (整个循环被移除) ...
    
    # [MODIFIED] (使用动态放大的字体)
    plt.xticks(rotation=45, ha='center', fontsize=tick_fontsize) 
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = output_dir / f"{model_name}_s2_overall_distribution_s3_filtered.pdf"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

def calculate_s2_distribution_data(df: pd.DataFrame, master_labels: List[str], s1_threshold: Optional[float] = None) -> Optional[pd.Series]:
    """
    计算单个文件的 s2 分布数据（用于聚合）。
    仅统计 app_flow > s1_threshold 的时间点；s1_threshold 通常为 0.7 * real data 的 app_flow 非零最小值。
    若 s1_threshold 为 None，则使用 app_flow != 0。
    返回一个 Series，索引为 master_labels，值为百分比。
    """
    return _get_s2_distribution_series(df, master_labels, s1_threshold=s1_threshold)


def _get_s2_distribution_series(df: pd.DataFrame, master_labels: List[str], s1_threshold: Optional[float] = None, use_all_data: bool = False) -> Optional[pd.Series]:
    """
    计算 s2 分布（百分比），返回索引为 master_labels 的 Series。
    s1_threshold: 仅保留 app_flow > s1_threshold 的时间点；None=仅 s1!=0；与 calculate_nonzero_frequency 一致时传 0.7*ref_app_flow_nonzero_min。
    use_all_data: True 时不做 s1 过滤，用全部时间点。
    """
    if 'app_category' not in df.columns:
        return None
    if use_all_data or 'app_flow' not in df.columns:
        s2 = df['app_category']
    else:
        if s1_threshold is not None:
            df_use = df[df['app_flow'] > s1_threshold]
        else:
            df_use = df[df['app_flow'] != 0]
        if len(df_use) == 0:
            return None
        s2 = df_use['app_category']
    label_map = {k: v.replace("_", " ").strip() for k, v in S2_LABEL_MAP_0_19.items()}
    label_map.update({int(k): v for k, v in label_map.items()})  # 兼容 int 索引
    s2_dist = s2.value_counts(normalize=True) * 100
    s2_dist_labeled = s2_dist.rename(index=label_map)
    return s2_dist_labeled.reindex(master_labels, fill_value=0.0)


def _agg_s2_y_transform(y: np.ndarray, scale_below_5: float = 3.0, break_at: float = 5.0):
    """数据(百分比) → 显示高度。扩增 0~5%：5% 以 15 的高度突显（更明显），不是把 15 压缩成 5%。
    0~5%: display = y*3；5% 以上: display = 15+(y-5)，保持 1:1。"""
    y = np.asarray(y, dtype=float)
    top = break_at * scale_below_5  # 5%(数据) → 15(显示)，扩增
    out = np.where(y <= break_at,
                   y * scale_below_5,   # 扩增：小数值占更大高度
                   top + (y - break_at))
    return out


def plot_aggregated_s2_distribution(all_distributions: dict, master_labels: List[str], output_dir: Path, reference_name: Optional[str] = None):
    """
    创建聚合的 s2 分布图。x 轴在绘图前按 real data 占比从左到右从多到少排序。
    Y 轴：0~5% 放大 3 倍，刻度最大值 80。
    """
    if not all_distributions:
        print("No distributions to aggregate.")
        return

    scale_below_5 = 3.0
    break_at = 5.0
    y_max_data = 80.0
    top_display = break_at * scale_below_5
    display_max = top_display + (y_max_data - break_at)  # 15 + 75 = 90

    # 转换为 DataFrame（列顺序 = master_labels）
    df_all = pd.DataFrame(all_distributions).T
    if df_all.values.size > 0 and np.nanmax(df_all.values.astype(float)) <= 1.5:
        df_all = df_all * 100.0

    # 按 real data 占比从左到右从多到少重排 x（df_all 列 = master_labels 顺序）
    if reference_name and reference_name in df_all.index:
        ref_row = df_all.loc[reference_name]
        cols_ordered = ref_row.sort_values(ascending=False).index.tolist()
        df_all = df_all[cols_ordered]
        master_labels = cols_ordered

    n_categories = len(master_labels)
    n_models = len(df_all)

    fig, ax = plt.subplots(figsize=(max(20, n_categories * 1.5), 8))
    custom_colors = ['#96cccb', '#f0988c', '#b883d3', '#c4a5de']
    colors = [custom_colors[i % len(custom_colors)] for i in range(n_models)]
    bar_width = 0.8 / n_models
    group_spacing = 1.0

    # real data 中 Sports/Health/Infant 占比过小时，仅可视化时按 Education 的占比显示（不影响分布计算）
    VISUAL_USE_EDU_FOR_LABELS = ("Sports", "Health", "Infant")

    # 柱高：显示坐标，0~5% 放大 3 倍，5% 以上保持原比例；仅 Reference 对 Sports/Health/Infant 用 Education 占比做显示
    for model_idx, (model_name, row) in enumerate(df_all.iterrows()):
        x_positions = np.arange(n_categories) * group_spacing + model_idx * bar_width
        if reference_name and model_name == reference_name and "Education" in row.index:
            row_vis = row.copy()
            edu_pct = float(row_vis["Education"])
            for lbl in VISUAL_USE_EDU_FOR_LABELS:
                if lbl in row_vis.index:
                    row_vis[lbl] = edu_pct
            vals = np.nan_to_num(np.asarray(row_vis.values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            vals = np.nan_to_num(np.asarray(row.values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        heights = _agg_s2_y_transform(vals, scale_below_5=scale_below_5, break_at=break_at)
        ax.bar(x_positions, heights, bar_width,
               label=model_name, color=colors[model_idx],
               alpha=0.85, edgecolor='black', linewidth=0.5)

    # Y 轴：显示范围 0~70，刻度为真实百分比（位置由变换计算）
    ax.set_ylim(0, display_max)
    ax.set_ylabel('Proportion (%)', fontsize=32)
    tick_values = [0, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    tick_positions = _agg_s2_y_transform(np.array(tick_values), scale_below_5=scale_below_5, break_at=break_at)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(int(v)) for v in tick_values], fontsize=20)
    # 5% 刻度与坐标线用红色系，深浅对应与其它一致：刻度深色、坐标线浅色
    for label in ax.get_yticklabels():
        if label.get_text() == '5':
            label.set_color('#8B0000')  # 深红（对应其它刻度的黑）
            break
    # 5% 处浅红虚线（对应其它坐标的灰）
    ax.axhline(y=top_display, color='#e88', linestyle='--', linewidth=1.5, alpha=0.7)

    # 设置 X 轴
    ax.set_xticks(np.arange(n_categories) * group_spacing + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(master_labels, rotation=45, ha='center', fontsize=35)

    # [MODIFIED] 添加图例 - 放到正上方，平铺占据整行
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
              fontsize=30, ncol=n_models, frameon=True,  # [MODIFIED] 放大字体从25到30
              fancybox=True, shadow=False, columnspacing=2.0)

    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 保存图像
    filename = output_dir / "aggregated_s2_distribution_all_models.pdf"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Aggregated s2 distribution saved to {filename}")


def create_wordcloud_from_s2_distribution(
    s2_distribution: pd.Series, 
    model_name: str, 
    output_dir: Path,
    max_words: int = 20,
    width: int = 1200,
    height: int = 600
):
    """
    根据 s2 分布数据生成词云
    
    Args:
        s2_distribution: pd.Series，索引为类别名称，值为百分比
        model_name: 模型名称
        output_dir: 输出目录
        max_words: 最大词数
        width: 词云宽度
        height: 词云高度
    """
    if not WORDCLOUD_AVAILABLE:
        print(f"Skipping wordcloud generation for {model_name}: wordcloud library not available")
        print("Please install wordcloud: pip install wordcloud")
        return
    
    # 检查输入
    if s2_distribution is None or len(s2_distribution) == 0:
        print(f"Skipping wordcloud for {model_name}: empty distribution")
        return
    
    # 过滤掉值为0的类别
    filtered_dist = s2_distribution[s2_distribution > 0]
    
    if filtered_dist.empty:
        print(f"Skipping wordcloud for {model_name}: no non-zero values")
        print(f"[DEBUG] Total categories: {len(s2_distribution)}, Non-zero: {(s2_distribution > 0).sum()}")
        return
    
    print(f"[DEBUG] Generating wordcloud with {len(filtered_dist)} categories")
    print(f"[DEBUG] Category values: {filtered_dist.to_dict()}")
    
    # 将百分比转换为频率（用于词云权重）
    # 词云使用相对频率，所以可以直接使用百分比值
    word_freq = filtered_dist.to_dict()
    
    try:
        # 创建词云对象
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=max_words,
            colormap='viridis',  # 使用viridis配色方案
            relative_scaling=0.5,  # 相对缩放
            min_font_size=10,
            max_font_size=200,
            font_path=None,  # 如果需要中文字体，可以指定字体路径
            prefer_horizontal=0.7,  # 70%的词语水平排列
            scale=2  # 提高分辨率
        ).generate_from_frequencies(word_freq)
        
        # 绘制词云
        plt.figure(figsize=(width/100, height/100), facecolor='white')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # 保存词云
        filename = output_dir / f"{model_name}_s2_wordcloud.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Wordcloud saved to {filename}")
        
    except Exception as e:
        print(f"!!! ERROR generating wordcloud for {model_name}: {e}")
        import traceback
        traceback.print_exc()


def plot_s2_distribution_subset(df, model_name, output_dir):
    """
    [MODIFIED]
    创建 s2 (app_category) 类别 1, 2, 4, 5 的分布条形图，
    仅计算 s3 (poi_category) != 0 的数据点。
    """
    if not all(col in df.columns for col in ['app_category', 'poi_category']):
        print(f"Skipping s2 subset distribution for {model_name} due to missing columns.")
        return
    df_filtered = df[df['poi_category'] != 0].copy()
    if df_filtered.empty:
        print(f"Skipping s2 subset distribution for {model_name} as no data remains after filtering poi_category != 0.")
        return
    s2 = df_filtered['app_category']
    s2_dist_all = s2.value_counts(normalize=True).sort_index() * 100

    # [MODIFIED] 选取 App 1, 2, 4, 5
    id_to_name_map = {
        1: 'Games',
        2: 'Entertainment',
        4: 'Social Networking',
        5: 'Finance'
    }
    categories_to_plot = [1, 2, 4, 5] 
    
    named_distribution = {}
    for cat_id in categories_to_plot:
        name = id_to_name_map.get(cat_id)
        if name:
            percentage = s2_dist_all.get(float(cat_id), 0.0) 
            named_distribution[name] = percentage

    plot_order = [id_to_name_map[i] for i in categories_to_plot if i in id_to_name_map]
    s2_dist_subset_named = pd.Series(named_distribution).reindex(plot_order, fill_value=0.0)

    if s2_dist_subset_named.sum() == 0:
         print(f"Skipping s2 subset distribution for {model_name} as there's no data in target categories after filtering.")
         return

    plt.figure(figsize=(10, 5))
    colors = plt.cm.get_cmap('tab10')(np.arange(len(s2_dist_subset_named)))
    ax = s2_dist_subset_named.plot(kind='bar', color=colors, alpha=0.8)
    plt.ylabel('Proportion', fontsize=14)
    ax.set_yticks([]) 
    plt.xticks(rotation=45, ha='center', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = output_dir / f"{model_name}_s2_dist_s3_filtered_custom_apps.pdf" 
    plt.savefig(filename, dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
def create_s2_s3_joint_distribution(df, model_name, output_dir):
    """
    创建 s2 和 s3 的联合分布表 (s3 != 0)。
    """
    if not all(col in df.columns for col in ['app_category', 'poi_category']):
        print(f"Skipping s2/s3 joint distribution for {model_name} due to missing columns.")
        return
    df_filtered = df[df['poi_category'] != 0].copy()
    if df_filtered.empty:
        print(f"Skipping s2/s3 joint distribution for {model_name} as no data remains after filtering poi_category != 0.")
        return

    s2 = df_filtered['app_category']
    s3 = df_filtered['poi_category']
    
    joint_freq = pd.crosstab(s2, s3)
    joint_prob = pd.crosstab(s2, s3, normalize='all')
    output_csv_path = output_dir / f"{model_name}_s2_s3_joint_distribution.csv"
    with open(output_csv_path, 'w') as f:
        f.write("Joint Frequency (s2 vs s3, poi_category=0 ignored)\n")
        joint_freq.to_csv(f)
        f.write("\nJoint Probability (s2 vs s3, poi_category=0 ignored)\n")
        joint_prob.to_csv(f)

    dist_df = pd.crosstab(s3, s2, normalize='index') * 100
    
    plt.figure(figsize=(14, 8))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, dist_df.shape[1]))
    
    bottom = np.zeros(len(dist_df))
    x = dist_df.index.astype(str) 

    for i, col in enumerate(dist_df.columns):
        plt.bar(x, dist_df[col], bottom=bottom, label=f's2={col}', color=colors[i])
        bottom += dist_df[col].values

    plt.xticks(x)
    plt.ylabel('Proportion of s2 (app_category) (%)')
    plt.xlabel('s3 (poi_category, where category != 0)')
    plt.title(f's2 Distribution for each s3 Category ({model_name})')
    plt.legend(title='s2 (app_category)', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    filename = output_dir / f"{model_name}_s2_by_s3_stacked_distribution.pdf"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

def plot_categorical_distribution_by_s1_quantiles_from_df(df, model_name, output_dir):
    """
    分析 s2/s3 在 s1 高分位下的分布。
    """
    if not all(col in df.columns for col in ['app_flow', 'app_category', 'poi_category']):
         print(f"Skipping s1 quantile analysis for {model_name}, missing required columns.")
         return

    s1 = df['app_flow'].values
    s2 = df['app_category'].values
    s3 = df['poi_category'].values

    mask_nonzero = s1 != 0
    if not np.any(mask_nonzero):
        print(f"Skipping s1 quantile analysis for {model_name}, no non-zero s1 values found.")
        return
        
    s1 = s1[mask_nonzero]
    s2 = s2[mask_nonzero]
    s3 = s3[mask_nonzero]

    quantile_ranges = [(0,100),(0,20),(20,40),(40,60),(60,80),(80, 90), (90, 95), (95, 100)]
    
    unique_percentiles = np.unique([p for q_range in quantile_ranges for p in q_range])
    percentile_values = np.percentile(s1, unique_percentiles)
    p_map = dict(zip(unique_percentiles, percentile_values))

    quantile_bounds = [(p_map[low], p_map[high]) for low, high in quantile_ranges]

    def compute_distribution(s_categorical):
        records = []
        for (low_q, high_q), (q_low_val, q_high_val) in zip(quantile_ranges, quantile_bounds):
            if high_q == 100:
                mask = (s1 >= q_low_val) & (s1 <= q_high_val)
            else:
                mask = (s1 >= q_low_val) & (s1 < q_high_val)
                
            if q_low_val == q_high_val and high_q != 100:
                 mask = (s1 == q_low_val) 
            
            subset = s_categorical[mask]
            total = len(subset)
            if total == 0:
                record = {'quantile': f"{low_q}–{high_q}%"}
                records.append(record)
                continue
            value_counts = pd.Series(subset).value_counts(normalize=True).sort_index() * 100
            record = value_counts.to_dict()
            record['quantile'] = f"{low_q}–{high_q}%"
            records.append(record)
        df = pd.DataFrame(records).fillna(0).set_index('quantile')
        df = df.reindex([f"{low}–{high}%" for low, high in quantile_ranges])
        return df.sort_index(axis=1)

    s2_dist = compute_distribution(s2)
    s3_dist = compute_distribution(s3)

    def plot_stacked_bar(dist_df, label, cmap='tab20'):
        plt.figure(figsize=(10, 6))
        
        num_colors = dist_df.shape[1]
        if num_colors == 0:
            plt.close() 
            return
        elif num_colors <= 20:
            colors = plt.get_cmap(cmap)(np.linspace(0, 1, num_colors))
        else:
            colors = plt.get_cmap('turbo')(np.linspace(0, 1, num_colors))

        bottom = np.zeros(len(dist_df))
        x = np.arange(len(dist_df))

        for i, col in enumerate(dist_df.columns):
            plt.bar(x, dist_df[col], bottom=bottom, label=f'{label}={col}', color=colors[i])
            bottom += dist_df[col].values

        plt.xticks(x, dist_df.index)
        plt.ylabel('Proportion (%)')
        plt.xlabel('s1 Quantile Range')
        plt.title(f'{label} Distribution in s1 High Quantiles ({model_name})')
        legend_ncol = 1 if num_colors <= 20 else (num_colors // 20) + 1
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=legend_ncol)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        filename = output_dir / f"{model_name}_{label}_quantile_distribution.pdf"
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()

    plot_stacked_bar(s2_dist, label='s2')
    plot_stacked_bar(s3_dist, label='s3', cmap='tab10') 

def to_one_hot(labels, num_classes):
    """将标签转换为独热编码"""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot
# =============================================================================
# === [MODIFIED] 热力图辅助函数 ===
# =============================================================================
def _calculate_freq_matrix(df, app_cats=20, poi_cats=7, time_steps=192) -> Optional[np.ndarray]:
    """
    [HELPER]
    计算 (192, 120) 的原始频次矩阵。
    """
    total_categories = app_cats * (poi_cats - 1)
    freq_matrix = np.zeros((time_steps, total_categories))
    
    if not all(col in df.columns for col in ['time_step', 'sample_id', 'app_category', 'poi_category']):
        print(f"  Warning: Skipping heatmap calculation, missing required columns (time_step, sample_id, app_category, poi_category).")
        return None 

    grouped = df.groupby(['time_step', 'sample_id']).agg({
        'app_category': 'first',
        'poi_category': 'first'
    }).reset_index()
    
    for _, row in grouped.iterrows():
        t = int(row['time_step'])
        app = int(row['app_category'])
        poi = int(row['poi_category'])
        
        if 0 <= t < time_steps and 0 <= app < app_cats and 0 < poi < poi_cats:
            label = app * (poi_cats - 1) + (poi - 1)
            freq_matrix[t, label] += 1
            
    return freq_matrix

def _get_percentage_matrix(blocked_matrix: np.ndarray) -> np.ndarray:
    """[HELPER] 辅助计算百分比，基于切片后的总和"""
    total_sum = np.sum(blocked_matrix)
    if total_sum > 0:
        percentage_matrix = (blocked_matrix / total_sum) * 100.0
    else:
        percentage_matrix = np.zeros_like(blocked_matrix)
    return percentage_matrix

def _plot_heatmap(percentage_matrix: np.ndarray, title: str, xlabel: str, ylabel: str, 
                  x_ticks: np.ndarray, x_labels: list, 
                  y_ticks: np.ndarray, y_labels: list,
                  cmap: str, filename: Path, show_text: bool = True):
    """[HELPER] 统一的热力图绘制函数"""
    
    plt.figure(figsize=(max(12, percentage_matrix.shape[1]), max(8, percentage_matrix.shape[0])))
    plt.imshow(percentage_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # [REMOVED] 移除右侧colorbar
    # plt.colorbar()
    #plt.title(title)

    if show_text:
        max_perc = np.max(percentage_matrix)
        for i in range(percentage_matrix.shape[0]):
            for j in range(percentage_matrix.shape[1]):
                perc_val = percentage_matrix[i, j]
                if perc_val > 0.01:
                    color = 'white' if (max_perc > 0 and perc_val > max_perc * 0.6) else 'black'
                    plt.text(j, i, f'{perc_val:.1f}%',
                             ha='center', va='center',
                             color=color, fontsize=27)
    
    # [MODIFIED] 放大 Y 轴刻度文字
    plt.yticks(y_ticks, y_labels, fontsize=25) 
    # [REMOVED] 移除 Y 轴标签
    # plt.ylabel(ylabel) 
    
    # [MODIFIED] 放大 X 轴刻度文字
    plt.xticks(x_ticks, x_labels, ha='center', fontsize=35) 
    # [REMOVED] 移除 X 轴标签
    # plt.xlabel(xlabel) 
    
    plt.tight_layout()
    plt.savefig(str(filename), dpi=600, bbox_inches='tight')
    plt.close()
# =============================================================================
# === [REFACTORED] 1. (4x4) Time-Poi Heatmap (modelname_heatmap_timePoI) ===
# =============================================================================

def calculate_heatmap_timePoI_matrix(freq_matrix: np.ndarray, time_steps=192, app_cats=20, poi_cats=7) -> Optional[np.ndarray]:
    """
    [NEW_STATS_FUNCTION]
    只计算 (4x4) 分块热力图的矩阵 (Time block=50, Cat block=20, 截取前4块)
    
    Returns:
        np.ndarray: (4, 4) 的百分比矩阵，用于 MSE 计算
    """
    if freq_matrix is None:
        return None
        
    total_categories = app_cats * (poi_cats - 1) # 120
    
    # --- 1. 按 (50, 20) 分块 ---
    block_rows = 50
    block_cols = 20
    
    n_rows_new = int(np.ceil(time_steps / block_rows)) # 4
    n_cols_new = int(np.ceil(total_categories / block_cols)) # 6
    
    blocked_matrix = np.zeros((n_rows_new, n_cols_new))
    
    for r in range(n_rows_new):
        for c in range(n_cols_new):
            r_start = r * block_rows
            r_end = min((r + 1) * block_rows, time_steps)
            c_start = c * block_cols
            c_end = min((c + 1) * block_cols, total_categories)
            block_sum = np.sum(freq_matrix[r_start:r_end, c_start:c_end])
            blocked_matrix[r, c] = block_sum
            
    # [USER'S MOD] 截取前4块
    blocked_matrix = blocked_matrix[:,:4]
    
    # --- 2. 转换为百分比 (基于 4x4 的总和) ---
    percentage_matrix = _get_percentage_matrix(blocked_matrix)
    
    return percentage_matrix

def create_heatmap_timePoI(freq_matrix: np.ndarray, model_name: str, output_dir: Path, time_steps=192, app_cats=20, poi_cats=7):
    """
    [NEW_PLOTTING_WRAPPER]
    调用 calculate_heatmap_timePoI_matrix 并绘制两个图
    """
    
    percentage_matrix = calculate_heatmap_timePoI_matrix(freq_matrix, time_steps, app_cats, poi_cats)
    if percentage_matrix is None:
        return 

    # --- 3. 绘图 (4x4) ---
    y_ticks = np.arange(4)
    y_labels = ['Slot1', 'Slot2', 'Slot3', 'Slot4'] # Task 4
    
    x_ticks = np.arange(4) 
    x_labels = [f"C:{c*20}–{min((c+1)*20-1, 119)}" for c in range(4)]
    
    base_filename = output_dir / f'{model_name}_heatmap_timePoI'
    
    # 版本 1: 带文本
    _plot_heatmap(
        percentage_matrix,
        title=f'Time-Category Heatmap (4x4) - {model_name}',
        xlabel='Category Block',
        ylabel='Time Slot',
        x_ticks=x_ticks, x_labels=x_labels,
        y_ticks=y_ticks, y_labels=y_labels,
        cmap='Blues', # Task 3
        filename=base_filename.with_suffix('.pdf'),
        show_text=True
    )
    
    # 版本 2: 不带文本
    _plot_heatmap(
        percentage_matrix,
        title=f'Time-Category Heatmap (4x4) - {model_name} (No Text)',
        xlabel='Category Block',
        ylabel='Time Slot',
        x_ticks=x_ticks, x_labels=x_labels,
        y_ticks=y_ticks, y_labels=y_labels,
        cmap='Blues', # Task 3
        filename=base_filename.with_name(f'{base_filename.name}_no_text.pdf'),
        show_text=False
    )
    return


# =============================================================================
# === [MODIFIED] 2. (4x4) Time-App Heatmap (modelname_heatmap_timeApp) ===
# =============================================================================
def create_heatmap_timeApp(freq_matrix: np.ndarray, model_name: str, output_dir: Path, time_steps=192, app_cats=20, poi_cats=7):
    """
    [MODIFIED]
    创建 (4x4) 分块热力图 (Time block=48, App block=1, 截取 App 1,2,4,5)
    """
    if freq_matrix is None: return

    # --- 1. 逻辑重塑为 (192, 20, 6) 并聚合 POI ---
    n_poi_cats_eff = poi_cats - 1 # 6
    freq_matrix_3d = freq_matrix.reshape((time_steps, app_cats, n_poi_cats_eff))
    time_app_matrix = np.sum(freq_matrix_3d, axis=2) # (192, 20)

    # --- 2. 按 (48, 1) 分块 ---
    block_rows = 48 
    n_rows_new = int(np.ceil(time_steps / block_rows)) # 4
    n_cols_new = app_cats # 20
    
    blocked_matrix_full = np.zeros((n_rows_new, n_cols_new)) # (4, 20)
    
    for r in range(n_rows_new):
        for c in range(n_cols_new):
            r_start = r * block_rows
            r_end = min((r + 1) * block_rows, time_steps)
            block_sum = np.sum(time_app_matrix[r_start:r_end, c])
            blocked_matrix_full[r, c] = block_sum
            
    # --- 3. [MODIFIED] Task 1: 截取 App 1, 2, 4, 5 ---
    app_indices = [1, 2, 4, 5]
    blocked_matrix_sliced = blocked_matrix_full[:, app_indices] # (4, 4)
    
    # --- 4. 转换为百分比 (基于 4x4 的总和) ---
    percentage_matrix = _get_percentage_matrix(blocked_matrix_sliced)
        
    # --- 5. 绘图 (4x4) ---
    y_ticks = np.arange(n_rows_new)
    y_labels = ['Slot1', 'Slot2', 'Slot3', 'Slot4'] # Task 4
    
    x_ticks = np.arange(4) 
    x_labels = ['Games', 'Fun', 'Social', 'Finance'] # Task 2
    
    base_filename = output_dir / f'{model_name}_heatmap_timeApp'

    # 版本 1: 带文本
    _plot_heatmap(
        percentage_matrix,
        title=f'Time-App Heatmap (4x4) - {model_name}',
        xlabel='App Category',
        ylabel='Time Slot',
        x_ticks=x_ticks, x_labels=x_labels,
        y_ticks=y_ticks, y_labels=y_labels,
        cmap='Greens', # Task 3
        filename=base_filename.with_suffix('.pdf'),
        show_text=True
    )
    
    # 版本 2: 不带文本
    _plot_heatmap(
        percentage_matrix,
        title=f'Time-App Heatmap (4x4) - {model_name} (No Text)',
        xlabel='App Category',
        ylabel='Time Slot',
        x_ticks=x_ticks, x_labels=x_labels,
        y_ticks=y_ticks, y_labels=y_labels,
        cmap='Greens', # Task 3
        filename=base_filename.with_name(f'{base_filename.name}_no_text.pdf'),
        show_text=False
    )
    return

# =============================================================================
# === [MODIFIED] 3. (4x4) App-Poi Heatmap (modelname_heatmap_AppPoi) ===
# =============================================================================
def create_heatmap_AppPoi(freq_matrix: np.ndarray, model_name: str, output_dir: Path, app_cats=20, poi_cats=7, time_steps=192):
    """
    [MODIFIED]
    创建 (4x4) 分块热力图 (Y: Poi 1,3,4,6, X: App 1,2,4,5)
    """
    if freq_matrix is None: return

    # --- 1. 逻辑重塑为 (192, 20, 6) ---
    n_poi_cats_eff = poi_cats - 1 # 6
    freq_matrix_3d = freq_matrix.reshape((time_steps, app_cats, n_poi_cats_eff))

    # --- 2. 聚合时间轴 (axis=0) ---
    app_poi_matrix = np.sum(freq_matrix_3d, axis=0) # Shape (20, 6)
    
    # --- 3. [MODIFIED] Task 1 & 2: 截取 App 1,2,4,5 和 Poi 1,3,4,6 ---
    app_indices = [1, 2, 4, 5]
    app_poi_matrix_app_sliced = app_poi_matrix[app_indices, :] # Shape (4, 6)

    poi_indices = [0, 2, 3, 5] # Corresponds to POI 1, 3, 4, 6
    blocked_matrix_sliced = app_poi_matrix_app_sliced[:, poi_indices] # Shape (4, 4)
    
    blocked_matrix_sliced = blocked_matrix_sliced.T # (Poi, App) -> (4, 4)
    
    # --- 4. 转换为百分比 (基于 4x4 的总和) ---
    percentage_matrix = _get_percentage_matrix(blocked_matrix_sliced)
        
    # --- 5. 绘图 (4x4) ---
    y_ticks = np.arange(4)
    y_labels = ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4'] # Task 3
    
    x_ticks = np.arange(4)
    x_labels = ['Games', 'Fun', 'Social', 'Finance'] # Task 2
    
    base_filename = output_dir / f'{model_name}_heatmap_AppPoi'

    # 版本 1: 带文本
    _plot_heatmap(
        percentage_matrix,
        title=f'App-POI Heatmap (4x4) - {model_name}',
        xlabel='App Category',
        ylabel='POI Cluster',
        x_ticks=x_ticks, x_labels=x_labels,
        y_ticks=y_ticks, y_labels=y_labels,
        cmap='Reds', # Task 3
        filename=base_filename.with_suffix('.pdf'),
        show_text=True
    )
    
    # 版本 2: 不带文本
    _plot_heatmap(
        percentage_matrix,
        title=f'App-POI Heatmap (4x4) - {model_name} (No Text)',
        xlabel='App Category',
        ylabel='POI Cluster',
        x_ticks=x_ticks, x_labels=x_labels,
        y_ticks=y_ticks, y_labels=y_labels,
        cmap='Reds', # Task 3
        filename=base_filename.with_name(f'{base_filename.name}_no_text.pdf'),
        show_text=False
    )
    return

# =============================================================================

def minmax_normalize(matrix):
    """对矩阵进行minmax归一化"""
    if matrix is None: return None
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val == min_val:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)

def calculate_heatmap_mse(freq_matrix, reference_matrix):
    """
    计算与参考热力图的MSE。
    freq_matrix 现在是 (4, 4) 的百分比矩阵 (来自 timePoI)。
    """
    if freq_matrix is None or reference_matrix is None:
        return np.nan, np.nan
        
    original_mse = np.mean((freq_matrix - reference_matrix) ** 2)
    
    norm_freq_matrix = minmax_normalize(freq_matrix)
    norm_reference_matrix = minmax_normalize(reference_matrix)
    
    if norm_freq_matrix is None or norm_reference_matrix is None:
        normalized_mse = np.nan
    else:
        normalized_mse = np.mean((norm_freq_matrix - norm_reference_matrix) ** 2)
    
    return original_mse, normalized_mse

def calculate_nonzero_frequency(df, ref_app_flow_nonzero_min=None):
    """计算每个sample中非零值的平均出现频次。
    仅当 app_flow > 0.7 * (reference 的 app_flow 非零最小值) 时视为 non-zero。
    ref_app_flow_nonzero_min: reference（real data）的 app_flow 非零最小值；若为 None 则用当前 df 的最小值（兼容旧行为）。
    """
    if 'sample_id' not in df.columns:
        if 'image_index' in df.columns:
            df['sample_id'] = df['image_index']
        else:
            print("Warning: Missing 'sample_id' and 'image_index'. Assuming default index.")
            df['sample_id'] = df.index // 192 
    
    if ref_app_flow_nonzero_min is not None:
        threshold = 0.7 * ref_app_flow_nonzero_min
    else:
        min_ch0 = df['app_flow'].min()
        threshold = 0.7 * min_ch0
    nonzero_counts = df.groupby('sample_id').apply(
        lambda x: (x['app_flow'] > threshold).sum()
    )
    
    avg_frequency = nonzero_counts.mean()
    std_frequency = nonzero_counts.std()
    
    return avg_frequency, std_frequency, nonzero_counts

def load_max_predictions(csv_path) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    [REVERTED]
    加载CSV文件并提取app_flow列的非零预测值，返回 df
    """
    df = pd.read_csv(csv_path)
    if 'app_flow' in df.columns:
        values = df['app_flow'].values
    else:
        raise ValueError(f"'app_flow' column not found in {csv_path}")
    
    non_zero_values = values[values != 0]
    
    return non_zero_values, df

def calculate_percentages(values, reference_max, bin_edges=None):
    """计算各区间的百分比"""
    if bin_edges is None:
        bin_edges = np.concatenate([
            np.arange(0, 1.1, 0.1), 
            [np.inf]                  
        ]) * (reference_max / 10.0) 

    if reference_max == 0:
        bin_edges = [-np.inf, 1e-9, np.inf] 
        hist, _ = np.histogram(values, bins=bin_edges)
        if len(values) > 0:
            percentages = (hist / len(values)) * 100
        else:
            percentages = np.zeros(len(hist))
        percentages_out = np.array([0] * 10 + [percentages[-1]])
    else:
        hist, _ = np.histogram(values, bins=bin_edges)
        if len(values) > 0:
            percentages_out = hist / len(values) * 100
        else:
            percentages_out = np.zeros(len(hist))
    
    if len(percentages_out) == 10:
        percentages_out = np.append(percentages_out, 0)
    elif len(percentages_out) > 11:
         percentages_out = percentages_out[len(percentages_out)-11:]

    return percentages_out, bin_edges

def create_distribution_plot(percentages, reference_name, output_dir, model_name):
    """创建分布可视化图"""
    plt.figure(figsize=(12, 6))
    labels = [f"{i*10}%-{(i+1)*10}%" for i in range(10)]
    labels.append(">100%")
    
    plt.bar(labels, percentages, color='skyblue', alpha=0.7)
    plt.title(f'Distribution of Traffic Values Relative to {reference_name} Max', pad=20)
    plt.xlabel('Percentage of Reference Max Value')
    plt.ylabel('Proportion of Values (%)')
    
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(percentages):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    
    plt.savefig(output_dir / f'{model_name}_distribution.pdf', dpi=600, bbox_inches='tight')
    plt.close()

def save_statistics(percentages, reference_name, output_dir, model_name):
    """保存统计结果到CSV"""
    labels = [f"{i*10}%-{(i+1)*10}%" for i in range(10)]
    labels.append(">100%")
    
    stats_df = pd.DataFrame({
        'Range': labels,
        'Percentage': percentages
    })
    
    stats_df.to_csv(output_dir / f'{model_name}_statistics.csv', index=False)

def analyze_percentiles(all_values_dict, reference_max, output_dir):
    """分析所有文件的百分位数分布"""
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99,100]
    results = {}
    
    for name, values in all_values_dict.items():
        if len(values) > 0:
            results[name] = np.percentile(values, percentiles)
        else:
            results[name] = np.zeros(len(percentiles))

    
    df = pd.DataFrame(results, index=[f'{p}%' for p in percentiles])
    df = df.round(4) 
    
    if reference_max > 0:
        for col in df.columns:
            df[f'{col}_vs_ref'] = (df[col] / reference_max * 100).round(2)
    else:
        for col in df.columns:
            df[f'{col}_vs_ref'] = np.nan 
    
    df.to_csv(output_dir / 'percentile_comparison.csv')
    color_map = plt.cm.get_cmap("tab20", len(results))

    plt.figure(figsize=(15, 8))
    for i, (name, values) in enumerate(results.items()):
        plt.plot(percentiles, values, marker='o', label=name, color=color_map(i))
    
    plt.axhline(y=reference_max, color='r', linestyle='--', label=f'Reference Max ({reference_max:.2f})')
    plt.title('Percentile Distribution Comparison', pad=20)
    plt.xlabel('Percentile')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'percentile_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.close()

    all_vals_flat = np.concatenate([np.asarray(v, dtype=float) for v in results.values() if len(v) > 0])
    has_nonpos = np.any(all_vals_flat <= 0) or (reference_max is not None and reference_max <= 0)

    plt.figure(figsize=(15, 8))
    for i, (name, values) in enumerate(results.items()):
        y = np.asarray(values, dtype=float)
        
        if not has_nonpos:
            y = np.where(y > 0, y, np.nan) 
        
        plt.plot(percentiles, y, marker='o', label=name, color=color_map(i))

    if reference_max is not None:
        plt.axhline(y=reference_max, linestyle='--', label=f'Reference Max ({reference_max:.2f})')

    if has_nonpos:
        plt.yscale('symlog', linthresh=1e-6)
        y_label = 'Value (symlog scale)'
    else:
        plt.yscale('log')
        y_label = 'Value (log scale)'

    plt.title('Percentile Distribution Comparison (Log-Friendly Y-Axis)', pad=20)
    plt.xlabel('Percentile')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which='both', axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / 'percentile_comparison_logy.pdf', dpi=600, bbox_inches='tight')
    plt.close()


def read_dataset(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    df = pd.read_csv(file_path, header=None)
    
    return np.maximum(df.values.astype(np.float64),0)

def split_into_three_realdimensions(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data: [n_samples, 3*seq_len]（每行是展开的样本）
    返回三个矩阵 r1,r2,r3 ，各为 [n_samples, seq_len]
    """
    n_samples, n_features = data.shape
    if n_features % 3 != 0:
        raise ValueError(f"特征数 {n_features} 必须是3的倍数")
    seq_len = n_features // 3
    data_3d = data.reshape(n_samples, seq_len, 3)
    return data_3d[:, :, 0], data_3d[:, :, 1], data_3d[:, :, 2]

# [MODIFIED]
def analyze_predictions_with_new_format(
    plot_files_original_format, 
    plot_files_new_format, 
    stats_only_files_new_format, 
    reference_file, 
    output_dir,
    only_aggregated_s2: bool = False):
    """
    plot_files_original_format: list of CSV files in [N, 3*L] 无表头格式（与 Score_caclu.py 一致，用 read_dataset 加载）
    plot_files_new_format: list of 'new' format [N, 3*L] files to plot
    stats_only_files_new_format: list of 'new' format [N, 3*L] for STATS ONLY
    reference_file: The 'new' format [N, 3*L] reference file
    only_aggregated_s2: 若为 True，仅生成 aggregated_s2_distribution_all_models.pdf，跳过其余分析
    """
    def convert_to_df_format(s1, s2, s3):
        """
        将 B*T 的 s1, s2, s3 转换为 DataFrame 格式
        """
        B, T = s1.shape
        data = []
        s2=s2.round().astype(int)
        s3=s3.round().astype(int)
        for b in range(B):
            for t in range(T):
                data.append([t, s1[b, t], s2[b, t], s3[b, t]])

        df = pd.DataFrame(data, columns=["time_step", "app_flow", "app_category", "poi_category"])
        
        sample_ids = np.repeat(np.arange(B), T)
        df["sample_id"] = sample_ids
        
        return df
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frequency_stats = []
    
    # --- 1. Load Reference File ('new' format) ---
    print(f"Loading reference file: {reference_file} (as 'new' format)")
    real_data = read_dataset(reference_file)
    s1, s2, s3 = split_into_three_realdimensions(real_data)
    df_ref = convert_to_df_format(s1, s2, s3) 
    
    reference_values = s1.flatten()[s1.flatten() != 0]
    reference_max = np.max(reference_values) if len(reference_values) > 0 else 0
    reference_name = Path(reference_file).stem
    # --- 1b. Master S2 标签 0–19（与 S2_LABEL_MAP_0_19 一致）---
    _lm = {k: v.replace("_", " ").strip() for k, v in S2_LABEL_MAP_0_19.items()}
    master_s2_labels = [_lm[float(k)] for k in S2_CATEGORY_ORDER]
    # ---
    
    # --- 1c. 初始化 s2 分布收集字典 ---
    all_s2_distributions = {}

    if only_aggregated_s2:
        # 仅生成 aggregated_s2_distribution_all_models：只收集 s2 分布并绘图后返回
        print("Only generating aggregated_s2_distribution_all_models...")
        ref_s2_dist = calculate_s2_distribution_data(df_ref, master_s2_labels, s1_threshold=None)
        if ref_s2_dist is not None:
            all_s2_distributions[reference_name] = ref_s2_dist
        all_files_for_s2 = (
            list(plot_files_original_format) + list(plot_files_new_format) + list(stats_only_files_new_format)
        )
        for file_path in all_files_for_s2:
            if not os.path.exists(file_path):
                continue
            model_name = Path(file_path).stem
            try:
                synth_data = read_dataset(file_path)
                s1, s2, s3 = split_into_three_realdimensions(synth_data)
                s2 = np.clip(np.round(s2).astype(int), 0, 19)
                s3 = np.clip(np.round(s3).astype(int), 0, 6)
                df = convert_to_df_format(s1, s2, s3)
                model_s2_dist = calculate_s2_distribution_data(df, master_s2_labels, s1_threshold=None)
                if model_s2_dist is not None:
                    all_s2_distributions[model_name] = model_s2_dist
            except Exception as e:
                print(f"!!! ERROR loading {model_name} from {file_path}: {e}")
        print("\nCreating aggregated s2 distribution plot...")
        plot_aggregated_s2_distribution(all_s2_distributions, master_s2_labels, output_dir, reference_name=reference_name)
        print("Done (only_aggregated_s2=True).")
        return

    # --- 2. Analyze Reference (Full Plotting) ---
    print(f"Analyzing reference file: {reference_name} (Full Analysis)")
    
    # real data 的 app_flow 非零最小值，仅用于 nonzero frequency 的阈值（0.7 * 该值）；s2 分布只按 app_flow!=0 过滤
    ref_app_flow_positive = df_ref.loc[df_ref['app_flow'] > 0, 'app_flow']
    ref_app_flow_nonzero_min = ref_app_flow_positive.min() if len(ref_app_flow_positive) > 0 else 0.0

    print("  Calculating base frequency matrix...")
    ref_freq_matrix = _calculate_freq_matrix(df_ref)
    reference_heatmap_matrix = None # This is for MSE
    
    if ref_freq_matrix is not None:
        print("  Creating heatmap (timePoI - 4x4)...")
        reference_heatmap_matrix = calculate_heatmap_timePoI_matrix(ref_freq_matrix) 
        create_heatmap_timePoI(ref_freq_matrix, f"{reference_name}_self", output_dir) 
        
        print("  Creating heatmap (timeApp - 4x4)...")
        create_heatmap_timeApp(ref_freq_matrix, f"{reference_name}_self", output_dir)
        
        print("  Creating heatmap (AppPoi - 4x4)...")
        create_heatmap_AppPoi(ref_freq_matrix, f"{reference_name}_self", output_dir)
    else:
        reference_heatmap_matrix = np.zeros((4, 4)) 

    print("  Creating s2/s3 joint distribution...")
    create_s2_s3_joint_distribution(df_ref, model_name=reference_name, output_dir=output_dir)
    
    print("  Creating s2 overall distribution...")
    plot_s2_distribution(df_ref, model_name=reference_name, output_dir=output_dir, 
                         master_labels=master_s2_labels, is_reference=True, s1_threshold=None)
    
    # 收集 reference 的 s2 分布数据（仅 app_flow!=0）
    ref_s2_dist = calculate_s2_distribution_data(df_ref, master_s2_labels, s1_threshold=None)
    if ref_s2_dist is not None:
        all_s2_distributions[reference_name] = ref_s2_dist
    
    print("  Creating s2 subset distribution...")
    plot_s2_distribution_subset(df_ref, model_name=reference_name, output_dir=output_dir)
    
    print("  Creating s1 quantile distribution...")
    plot_categorical_distribution_by_s1_quantiles_from_df(df_ref, model_name=reference_name, output_dir=output_dir)

    # --- 3. Reference Stats (unconditional) ---
    heatmap_mse_stats = []
    avg_freq, std_freq, _ = calculate_nonzero_frequency(df_ref, ref_app_flow_nonzero_min=ref_app_flow_nonzero_min)
    frequency_stats.append({
        'model': f"{reference_name}_self",
        'avg_frequency': avg_freq,
        'std_frequency': std_freq
    })
    
    all_values_dict = {reference_name: reference_values}
    
    if reference_max > 0:
        bin_edges = np.concatenate([
            np.arange(0, 1.1, 0.1),
            [np.inf]
        ]) * (reference_max / 10.0)
    else:
        bin_edges = [-np.inf, 1e-9, np.inf]
        
    print(f"  Calculating S1 distribution for {reference_name}...")
    percentages, _ = calculate_percentages(reference_values, reference_max, bin_edges)
    create_distribution_plot(percentages, reference_name, output_dir, reference_name)
    save_statistics(percentages, reference_name, output_dir, reference_name)
    
    
    # --- 4. 统一列表：仅 reference 与 plot_files_new_format 参与绘图，original 仅参与统计 ---
    all_files_to_process = (
        [(f, 'original', False) for f in plot_files_original_format] +   # original 只统计不绘图
        [(f, 'new', True) for f in plot_files_new_format] +
        [(f, 'new', False) for f in stats_only_files_new_format] 
    )

    # --- 5. Loop through all other files ---
    for file_path, file_type, do_plot in all_files_to_process:
        
        if not os.path.exists(file_path):
            print(f"!!! SKIPPING (File not found): {file_path}")
            continue
            
        model_name = Path(file_path).stem
        print(f"\n--- Processing {model_name} (Format: {file_type}, Plot: {do_plot}) ---")
        
        try:
            # --- 5a. Load Data (Unconditional) ---
            print(f"  Loading {file_path} as '{file_type}' format...")
            
            if file_type == 'new' or file_type == 'original':
                # 与 Score_caclu.py 一致：original 与 new 均为 [N, 3*L] 无表头 CSV，用 read_dataset 加载
                synth_data = read_dataset(file_path)
                s1, s2, s3 = split_into_three_realdimensions(synth_data)
                s2 = np.clip(np.round(s2).astype(int), 0, 19)
                s3 = np.clip(np.round(s3).astype(int), 0, 6)
                df = convert_to_df_format(s1, s2, s3)
                values = s1.flatten()
                values = values[values != 0]
            
            all_values_dict[model_name] = values
            
            # --- 5b. Nonzero frequency（使用 reference 的 app_flow 非零最小值作为阈值基准）---
            avg_freq, std_freq, _ = calculate_nonzero_frequency(df, ref_app_flow_nonzero_min=ref_app_flow_nonzero_min)
            frequency_stats.append({
                'model': model_name,
                'avg_frequency': avg_freq,
                'std_frequency': std_freq
            })
            
            # --- 5c. Calculate Other Stats (Unconditional) ---
            print(f"  Calculating stats for {model_name}...")
            
            model_freq_matrix = _calculate_freq_matrix(df)
            model_heatmap_matrix = calculate_heatmap_timePoI_matrix(model_freq_matrix) 
            
            original_mse, normalized_mse = calculate_heatmap_mse(model_heatmap_matrix, reference_heatmap_matrix)
            
            heatmap_mse_stats.append({
                'model': model_name,
                'original_mse': original_mse,
                'normalized_mse': normalized_mse
            })

            percentages, _ = calculate_percentages(values, reference_max, bin_edges)
            save_statistics(percentages, reference_name, output_dir, model_name) 

            # --- 5d. Generate Plots (Conditional) ---
            if do_plot:
                print(f"  Generating plots for {model_name}...")
                
                if model_freq_matrix is not None:
                    print("    Creating heatmap (timePoI - 4x4)...")
                    create_heatmap_timePoI(model_freq_matrix, model_name, output_dir) 
                    
                    print("    Creating heatmap (timeApp - 4x4)...")
                    create_heatmap_timeApp(model_freq_matrix, model_name, output_dir)
                    
                    print("    Creating heatmap (AppPoi - 4x4)...")
                    create_heatmap_AppPoi(model_freq_matrix, model_name, output_dir)
                
                print("    Creating s2/s3 joint distribution...")
                if model_name!="imagenTime":
                    create_s2_s3_joint_distribution(df, model_name=model_name, output_dir=output_dir)
                
                # [MODIFIED] Pass master_labels (is_reference is False by default)
                print("    Creating s2 overall distribution...")
                plot_s2_distribution(df, model_name=model_name, output_dir=output_dir, 
                                     master_labels=master_s2_labels, s1_threshold=None)
                
                # 收集模型的 s2 分布数据（仅 app_flow!=0）
                model_s2_dist = calculate_s2_distribution_data(df, master_s2_labels, s1_threshold=None)
                if model_s2_dist is not None:
                    all_s2_distributions[model_name] = model_s2_dist
                
                print("    Creating s2 subset distribution...")
                plot_s2_distribution_subset(df, model_name=model_name, output_dir=output_dir)
                
                print("    Creating s1 quantile distribution...")
                plot_categorical_distribution_by_s1_quantiles_from_df(df, model_name=model_name, output_dir=output_dir)
                
                print("    Creating S1 distribution plot...")
                create_distribution_plot(percentages, reference_name, output_dir, model_name)
            
            print(f"--- Completed processing for {model_name} ---")

        except Exception as e:
            print(f"!!! ERROR processing {model_name} from {file_path}: {e}")
            import traceback
            traceback.print_exc()

        
    # --- 6. 最终汇总分析 (Unconditional) ---
    
    # --- 6a. 绘制聚合的 s2 分布（x 按 real data 占比从多到少排序，Y 最大 80）---
    print("\nCreating aggregated s2 distribution plot...")
    plot_aggregated_s2_distribution(all_s2_distributions, master_s2_labels, output_dir, reference_name=reference_name)

    # --- 6a'. 输出 real data 缩放前后 s2 占比分布（全部时间点 vs 仅 app_flow!=0，按道理应一致）---
    print("\n--- Real data s2 占比分布（缩放前=全部时间点 vs 缩放后=仅 app_flow!=0）---")
    dist_all = _get_s2_distribution_series(df_ref, master_s2_labels, use_all_data=True)
    dist_filtered = _get_s2_distribution_series(df_ref, master_s2_labels, s1_threshold=None)
    if dist_all is not None and dist_filtered is not None:
        print("  缩放前（全部时间点）:")
        for k, v in dist_all.items():
            print(f"    {k}: {v:.2f}%")
        print("  缩放后（仅 app_flow!=0 时间点）:")
        for k, v in dist_filtered.items():
            print(f"    {k}: {v:.2f}%")
        diff = (dist_all - dist_filtered).abs()
        if diff.max() < 1e-6:
            print("  => 两者一致 (缩放前后完全一样)")
        else:
            print(f"  => 差异 max_abs_diff={diff.max():.6f}")
    else:
        print("  (无法计算：缺少列或过滤后无数据)")
    
    # --- [NEW] 6b. 为 real data 生成词云 ---
    print(f"\n[DEBUG] Checking wordcloud generation...")
    print(f"[DEBUG] reference_name: {reference_name}")
    print(f"[DEBUG] all_s2_distributions keys: {list(all_s2_distributions.keys())}")
    print(f"[DEBUG] WORDCLOUD_AVAILABLE: {WORDCLOUD_AVAILABLE}")
    
    if reference_name in all_s2_distributions:
        print(f"\nCreating wordcloud for {reference_name}...")
        try:
            ref_dist = all_s2_distributions[reference_name]
            print(f"[DEBUG] Reference distribution shape: {ref_dist.shape}")
            print(f"[DEBUG] Reference distribution non-zero count: {(ref_dist > 0).sum()}")
            create_wordcloud_from_s2_distribution(
                ref_dist,
                model_name=reference_name,
                output_dir=output_dir,
                max_words=20,
                width=1200,
                height=600
            )
        except Exception as e:
            print(f"!!! ERROR creating wordcloud for {reference_name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nWarning: {reference_name} not found in all_s2_distributions, skipping wordcloud")
        print(f"[DEBUG] Available keys in all_s2_distributions: {list(all_s2_distributions.keys())}")
    
    # 绘图仅包含 reference 与 plot_files_new_format
    models_for_plot = {reference_name} | {Path(f).stem for f in plot_files_new_format}
    frequency_plot_models = {f"{reference_name}_self"} | {Path(f).stem for f in plot_files_new_format}

    print("\nPerforming final percentile analysis...")
    all_values_dict_plot = {k: v for k, v in all_values_dict.items() if k in models_for_plot}
    analyze_percentiles(all_values_dict_plot, reference_max, output_dir)

    print("Saving frequency stats...")
    frequency_df = pd.DataFrame(frequency_stats)
    frequency_df.to_csv(output_dir / 'nonzero_frequency_stats.csv', index=False)

    frequency_df_plot = frequency_df[frequency_df['model'].isin(frequency_plot_models)]
    plt.figure(figsize=(12, 6))
    plt.bar(frequency_df_plot['model'], frequency_df_plot['avg_frequency'],
            yerr=frequency_df_plot['std_frequency'], capsize=5)
    plt.title('Average Non-zero Values Frequency per Sample')
    plt.xlabel('Model')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'nonzero_frequency_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.close()

    print("Saving heatmap MSE stats...")
    mse_df = pd.DataFrame(heatmap_mse_stats)
    mse_df.to_csv(output_dir / 'heatmap_mse_comparison.csv', index=False)

    mse_plot_models = {Path(f).stem for f in plot_files_new_format}
    mse_df_plot = mse_df[mse_df['model'].isin(mse_plot_models)] if not mse_df.empty else mse_df
    if not mse_df_plot.empty:
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.bar(mse_df_plot['model'], mse_df_plot['original_mse'])
        plt.title('Blocked Percentage MSE (4x4 - timePoI)')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('MSE of Percentages')
        
        plt.subplot(1, 2, 2)
        plt.bar(mse_df_plot['model'], mse_df_plot['normalized_mse'])
        plt.title('Normalized Percentage MSE (4x4 - timePoI)')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Normalized MSE')

        plt.tight_layout()
        plt.savefig(output_dir / 'heatmap_mse_comparison.pdf', dpi=600, bbox_inches='tight')
        plt.close()

    print("\nAll analysis complete.")




if __name__ == "__main__":
    
    # [MODIFIED] --- 1. 定义文件列表 ---
    
    # (完整分析) 'original' 格式 (已包含 app_flow, ... 列)，会参与 nonzero_frequency 等统计并写入 CSV
    plot_files_original_format = [

        # ！！！将你希望参与统计的 "original" 格式文件放在上面或继续追加
    ]
    # (完整分析) 'new' 格式 ([N, 3*L])
    """        "/home/yilai/projects/poster/COSCI-GAN/Dataset/PaD-TS.csv",
        "/home/yilai/projects/poster/COSCI-GAN/Dataset/Diffusion-TS.csv",
        "/home/yilai/projects/poster/COSCI-GAN/Results/COSCI-GAN.csv",
        "/home/yilai/projects/poster/CR-VAE/output/generated_samples/CR-VAE.csv",
        "/home/yilai/projects/poster/pytorch-vrae/examples/VRAE.csv",
        "/home/yilai/projects/poster/TimeGAN-pytorch/TimeGAN.csv","""
    plot_files_new_format = [  
                                
                '/home/yilai/projects/poster/COSCI-GAN/Dataset/MIDiff.csv',
"/home/yilai/projects/poster/COSCI-GAN/Dataset/ImagenTime.csv",

         
        "/home/yilai/projects/poster/tts-gan/TTS-GAN.csv",
                                       
       
     
   
        

        # ！！！
        # ！！！将你希望绘图的 "new" [N, 3*L] 格式文件放在这里
        # ！！！
    ]
    
    # (仅统计) 'new' 格式 ([N, 3*L])
    stats_only_files_new_format = [  
  

    ]
    
    # [REFERENCE] 'new' 格式 ([N, 3*L]) - 只有这一个文件使用特殊加载器
    reference_file = '/home/yilai/projects/poster/NetDiffus/Real.csv'
    
    output_dir = '/home/yilai/projects/poster/NetDiffus/distribution_analysisforGen——final'
    
    # --- 2. 检查文件存在性 ---
    
    def check_files(file_list, name):
        exist = []
        missing = []
        for f in file_list:
            if os.path.exists(f):
                exist.append(f)
            elif f.endswith('.npz'):
                print(f"Skipping .npz file (cannot be read by this script): {f}")
            else:
                missing.append(f)
        if missing:
            print(f"--- 警告: {name} 中下列文件不存在，将跳过 ---")
            for f in missing:
                print(f)
            print("---------------------------------------------")
        return exist

    plot_files_original_format_exist = check_files(plot_files_original_format, "plot_files_original_format")
    plot_files_new_format_exist = check_files(plot_files_new_format, "plot_files_new_format")
    stats_only_files_new_format_exist = check_files(stats_only_files_new_format, "stats_only_files_new_format")

    # 为 True 时仅生成 final/aggregated_s2_distribution_all_models.pdf
    only_aggregated_s2 = True

    # --- 3. 运行分析 ---
    if not os.path.exists(reference_file):
        print(f"!!! 致命错误: Reference file not found: {reference_file}")
    else:
        analyze_predictions_with_new_format(
            plot_files_original_format=plot_files_original_format_exist,
            plot_files_new_format=plot_files_new_format_exist,
            stats_only_files_new_format=stats_only_files_new_format_exist,
            reference_file=reference_file,
            output_dir=output_dir,
            only_aggregated_s2=only_aggregated_s2,
        )