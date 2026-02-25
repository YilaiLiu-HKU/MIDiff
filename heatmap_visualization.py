import numpy as np
import os
import matplotlib.pyplot as plt
import imageio

# === 加载数据 ===
base_dir = os.path.dirname(os.path.abspath(__file__))
file = np.load(os.path.join(base_dir, "dataset/all_users_data_with6cluster.npz"), allow_pickle=True)
app_traffics = file['Category_ID_Traffic (Byte)']  # (B, 288, n1)
pois          = file['poi_labels']                 # (B, 288, n2)

B, T, n1 = app_traffics.shape
_, _, n2  = pois.shape

# 你定义的 app 类别名称
app_names = ["Books", "Education", "Entertainment", "Finance", "Games",
             "Health&Fitness", "Lifestyle", "Music", "Navigation", "News",
             "Photo&Video", "References", "Shopping", "Social_Networking",
             "Travel", "Utilities", "Weather", "Infant&Mom", "Sports", "Other"]

assert n1 == len(app_names), f"Mismatch: n1={n1}, but got {len(app_names)} app names"

# === 可调参数：时间分段数 ===
N_SEGMENTS = 4
segments = np.array_split(np.arange(T), N_SEGMENTS)

# === 分段统计: 生成 (N_SEGMENTS, n1) 的均值和计数矩阵 ===
means  = np.zeros((N_SEGMENTS, n1), dtype=np.float32)
counts = np.zeros((N_SEGMENTS, n1), dtype=np.int32)

for seg_idx, idxs in enumerate(segments):
    block = app_traffics[:, idxs, :]  # (B, len(idxs), n1)
    flat  = block.reshape(-1, n1)
    nz_mask = flat > 0
    counts[seg_idx] = nz_mask.sum(axis=0)
    means[seg_idx] = np.divide(
        flat.sum(axis=0),
        counts[seg_idx],
        out=np.zeros_like(counts[seg_idx], dtype=np.float32),
        where=counts[seg_idx] != 0
    )
    means[seg_idx] = np.log1p(means[seg_idx])

# === 绘图改为更直观的二维热图：counts 作为 alpha，means 映射颜色 ===
fig, ax = plt.subplots(figsize=(12, 4))
# 映射颜色为均值，透明度为 counts
norm_means = (means - means.min()) / (means.max() - means.min()) if means.max() > means.min() else np.zeros_like(means)
def time_str(index):
    minutes = index * 5
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

segment_labels = []
for seg in segments:
    start_time = time_str(seg[0])
    end_time   = time_str(seg[-1])
    segment_labels.append(f"{start_time}–{end_time}")

# === 绘图 ===
fig, ax = plt.subplots(figsize=(12, 4))

# 直接绘制均值图，不使用透明度映射
im = ax.imshow(means, aspect='auto', cmap='viridis', alpha=1.0)

# 设置标签
ax.set_xticks(np.arange(n1))
ax.set_xticklabels(app_names, rotation=45, ha='right')
ax.set_yticks(np.arange(N_SEGMENTS))
ax.set_yticklabels(segment_labels)
ax.set_xlabel("App Category")
ax.set_ylabel("Time Period")

# colorbar 显示实际均值值
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Mean Traffic (non-zero values)")

plt.title("App Usage Over Time Segments")
plt.tight_layout()

save_path = os.path.join(base_dir, "app_traffic_stats_heatmap_noalpha.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved heatmap without alpha: {save_path}")



# =====app次数统计====
# ==== App 使用次数图 ====
plt.figure(figsize=(12, 4))
plt.imshow(counts, aspect='auto', cmap='Blues')
plt.colorbar(label="Usage Count (non-zero entries)")
plt.xticks(ticks=np.arange(n1), labels=app_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(N_SEGMENTS), labels=segment_labels)
plt.xlabel("App Category")
plt.ylabel("Time Segments")
plt.title("App Usage Count Over Time Segments")
plt.tight_layout()

app_count_path = os.path.join(base_dir, "app_usage_count_heatmap.png")
plt.savefig(app_count_path, dpi=300)
plt.close()

print(f"Saved app usage count heatmap: {app_count_path}")

# ==== POI统计 ====
segment_count = 4  # 你可以根据需要调整


poi_counts = np.zeros((segment_count, n2), dtype=np.int32)
time_per_seg = T // segment_count

for seg_idx in range(segment_count):
    start = seg_idx * time_per_seg
    end = (seg_idx + 1) * time_per_seg if seg_idx < segment_count - 1 else T
    seg_data = pois[:, start:end, :]  # shape: (B, segment_len, n2)

    # 统计值为1的次数
    seg_flat = seg_data.reshape(-1, n2)  # (B*segment_len, n2)
    poi_counts[seg_idx] = seg_flat.sum(axis=0)
# ==== 可视化 POI 热力图 ====
plt.figure(figsize=(12, 4))
plt.imshow(poi_counts[:, 1:], aspect='auto', cmap='YlGnBu')
plt.colorbar(label="Count (POI activated)")
plt.xticks(ticks=np.arange(n2 - 1), rotation=90)
segment_labels = [f"{h:02d}:{m:02f}" for h, m in zip(
    np.linspace(0, 24, segment_count + 1, endpoint=True)[:-1].astype(int),
    np.linspace(0, 24, segment_count + 1, endpoint=True)[:-1] % 1 * 60
)]
plt.yticks(ticks=np.arange(segment_count), labels=segment_labels)
plt.xlabel("POI Categories")
plt.ylabel("Time Segments")
plt.title("POI Occurrence Heatmap by Time Segment")
plt.tight_layout()

poi_save_path = os.path.join(base_dir, "poi_occurrence_heatmap.png")
plt.savefig(poi_save_path, dpi=300)
plt.close()

print(f"Saved POI occurrence heatmap: {poi_save_path}")
# app_traffics: (B, 288, n)
is_nonzero = np.any(app_traffics > 0, axis=(0, 2))  # (288,) 每个时间点是否有非零值

# 找第一个为 False 的 index，表示全为 0
first_all_zero_index = np.argmax(~is_nonzero)

# 如果全为非零，np.argmax(~is_nonzero) 会返回 0，但这不对
if is_nonzero.all():
    print("所有时间段都有非零值")
else:
    print(f"从时间索引 {first_all_zero_index} 开始，app_traffics 全为 0")
    minutes = first_all_zero_index * 5
    h = minutes // 60
    m = minutes % 60
    print(f"对应时间为 {h:02d}:{m:02d}")
