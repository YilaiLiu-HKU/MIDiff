import numpy as np
import os

# ---------------- 配置 ----------------
src_npz   = 'dataset/all_users_data_with6cluster.npz'   # 原始数据
top_n     = 15    
dst_npz   = f'dataset/all_users_data_with6cluster_top{top_n}.npz'  # 过滤后保存路径
    # 与之前脚本保持一致
 # 如需动态可改为：top_n / C
# ------------------------------------

# 1. 读取原始文件
data = np.load(src_npz, allow_pickle=True)
data_allow_pickle = dict(data)   # 先全部拷出来，后面只替换两个键

app_traffics = data['Category_ID_Traffic (Byte)']
pois         = data['poi_labels']
C = app_traffics.shape[2]
ratio_threshold = top_n / C 
# 2. 统计每个 app 的出现频次
freq = np.zeros(C, dtype=int)
for sample in app_traffics:
    active_idx = np.argmax(sample, axis=1)
    mask       = np.any(sample != 0, axis=1)
    active_idx = active_idx[mask]
    for idx in active_idx:
        freq[idx] += 1

# 3. 选出 Top-N
top_indices = np.argsort(freq)[-top_n:]
print('Top app classes:', top_indices)
print('Frequencies    :', freq[top_indices])

# 4. 过滤样本
filtered_app, filtered_poi = [], []
for app, poi in zip(app_traffics, pois):
    mask_step = np.any(app != 0, axis=1)
    if not np.any(mask_step):
        continue

    active_idx = np.argmax(app, axis=1)[mask_step]
    top_mask   = np.isin(active_idx, top_indices)
    ratio      = np.sum(top_mask) / mask_step.sum()

    if ratio < ratio_threshold:
        continue

    # 构造新样本
    new_app = np.zeros_like(app)
    new_poi = np.zeros_like(poi)
    new_poi[:, 0] = 1   # 默认全 0 的 poi 设为 [1,0,...]

    for t in np.where(mask_step)[0]:
        idx = np.argmax(app[t])
        if idx in top_indices:
            new_app[t] = app[t]
            new_poi[t] = poi[t]

    filtered_app.append(new_app)
    filtered_poi.append(new_poi)

# 5. 检查合法性
filtered_app = np.stack(filtered_app, axis=0)
filtered_poi = np.stack(filtered_poi, axis=0)

illegal = np.setdiff1d(
    np.argmax(filtered_app, axis=2)[np.any(filtered_app != 0, axis=2)],
    top_indices
)
if len(illegal) > 0:
    raise ValueError(f'❌ 发现非法类别：{illegal}')
print(f'✅ 过滤后样本数：{len(filtered_app)} / {len(app_traffics)}')
print(filtered_app)
# 6. 覆盖键并保存
data_allow_pickle['app_traces'] = filtered_app
data_allow_pickle['poi_traces']= filtered_poi

np.savez(dst_npz, **data_allow_pickle)
print(f'已保存过滤后的数据集：{dst_npz}')