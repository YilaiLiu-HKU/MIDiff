import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm

# 加载数据
save_dir=r'/home/yilai/poster/NetDiffus/ckpt/orginal'
file='emamodel008000.pt_samples_3000x320x160x3.npz'
data = np.load(os.path.join(save_dir,file),allow_pickle=True)  # 替换为你的 npz 文件路径
arr = data['arr_0']  # 如果你的 npz 中的键不是 'arr'，请替换

# 创建保存目录
output_dir =os.path.join(save_dir,file.split('.')[0])
os.makedirs(output_dir, exist_ok=True)

crop_h, crop_w = 288, 140
H, W = arr.shape[1], arr.shape[2]
start_h = (H - crop_h) // 2
start_w = (W - crop_w) // 2

# 遍历裁剪并保存
for i in range(arr.shape[0]):
    img = arr[i]  # shape: (320, 320, 3)

    # 中心裁剪
    cropped = img[start_h:start_h + crop_h, start_w:start_w + crop_w, :]
    
    # 将 RGB 转为灰度图像（取平均）
    gray_img = cropped.mean(axis=2)  # shape: (320, 320)
    ####对之前错误归一化的采样用以下转为正确归一化
   
    gray_img =((gray_img.astype(np.float32) - 1) * (255 / 252)).clip(0, 255).astype(np.uint8)
    
    plt.imsave(
        os.path.join(output_dir, f'image_{i:04d}.png'),
        gray_img,
        cmap='viridis'
    )

print(f"共保存 {arr.shape[0]} 张图像到 {output_dir}/，使用 colormap = viridis")
