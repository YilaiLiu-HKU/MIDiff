import numpy as np
import os
import matplotlib.pyplot as plt

# === 加载数据 ===
file = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/all_users_data_with6cluster.npz"), allow_pickle=True)
print(file.files)

app_traffics = file['Category_ID_Traffic (Byte)'][:,:192,:]  # shape: (B, 288, n1)
pois = file["poi_labels"][:,:192,:]                          # shape: (B, 288, n2)

B, T, n1 = app_traffics.shape
_, _, n2 = pois.shape

# === 第一步：外积 ===
# 扩展维度后相乘，相当于每一行做外积
# 结果形状：(B, 288, n1, n2)
A_expand = app_traffics[:, :, :, np.newaxis]  # (B, 288, n1, 1)
B_expand = pois[:, :, np.newaxis, :]          # (B, 288, 1, n2)

C = A_expand * B_expand                       # (B, 288, n1, n2)
C = C.reshape(B, T, n1 * n2)                  # (B, 288, n1*n2)

# === 第二步：将非零值置为1 ===
C = (C > 0).astype(np.float32)                # (B, 288, n1*n2)

# === 第三步：在 batch 维度上求和 ===
heatmap = C.sum(axis=0)                       # (288, n1*n2)

# === 保存数据 ===
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmap.npy")
np.save(save_path, heatmap)
print(f"Heatmap data saved to {save_path}")

# === 可视化 ===
plt.figure(figsize=(12, 6))
plt.style.use('seaborn')  # 使用seaborn样式提升整体美观度
plt.imshow(heatmap, aspect='auto', cmap='Blues')  # 使用YlGnBu配色方案
colorbar = plt.colorbar()
colorbar.set_label('Co-activation Count', fontsize=10)

plt.title("Heatmap of POI-APP Co-activation Patterns", fontsize=12, pad=15)
plt.xlabel("POI x APP Combinations", fontsize=10)
plt.ylabel("Time Slots (24 hours)", fontsize=10)

# 添加时间刻度
hour_ticks = np.linspace(0, 191, 13)  # 12个小时刻度点
hour_labels = [f'{int(h/12):02d}:00' for h in range(0, 25, 2)]  # 每2小时标记一次
plt.yticks(hour_ticks, hour_labels)

# 保存高质量图像
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmap.png"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
