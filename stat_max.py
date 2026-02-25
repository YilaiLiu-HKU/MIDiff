import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

root_dir = "/root/autodl-tmp/NetDiffus/dataset_png"
max_values = []
max_max_value=0
# 遍历所有子文件夹
for subdir in tqdm(os.listdir(root_dir)):
    subpath = os.path.join(root_dir, subdir)
    if not os.path.isdir(subpath):
        continue
    # 匹配 cross_gasf_*.png
    for img_path in glob.glob(os.path.join(subpath, "cross_gasf_*.png")):
        try:
            img = np.array(Image.open(img_path))
            img = img / 255.0 
            max_val = img.max()
            max_values.append(max_val)
            if max_max_value<max_val:
                max_max_value=max_val
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

max_values = np.array(max_values)
print(f"共统计图片数: {len(max_values)}")
print(f"最大值均值: {max_values.mean()}")
print(f"最大值方差: {max_values.var()}")
print(f"最大最大值: {max_max_value}")