import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import tifffile
#data_dir = "Youtube/vid"  # directory where 1D traces are stored (Format : Youtube/vid*)
save_dir = "test/"  # directory to store GASF converted data as PNG
os.makedirs(save_dir,exist_ok=True)  # number of videos
import numpy as np
import matplotlib.pyplot as plt
file=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"dataset/all_users_data_with6cluster.npz"),allow_pickle=True)
print(file.files)
app_traffics=file['Category_ID_Traffic (Byte)']
pois=file["poi_labels"]
#import pdb;pdb.set_trace()
records = []
num_features = app_traffics.shape[2]
feature_maxes = np.max(app_traffics, axis=(0, 1))  # 形状 (num_features,)
#import pdb;pdb.set_trace()
zero_mask= feature_maxes==0
feature_maxes[zero_mask]=1
# 保存特征最大值到CSV
feature_max_df = pd.DataFrame({
    'feature_index': range(num_features),
    'max_value': feature_maxes
})

feature_max_df.to_csv(os.path.join(save_dir, "feature_max_values.csv"), index=False)
print(f"Saved feature max values to {save_dir}feature_max_values.csv")

# 目标图像尺寸
TARGET_H, TARGET_W =256, 160

def minmax_scale(ts):
    return (ts - np.min(ts)) / (np.max(ts) - np.min(ts))

def to_gasf_cross(x, y):
    x_phi = np.arccos(x)
    y_phi = np.arccos(y)
    gasf_xy = np.cos(x_phi[:, None] + y_phi[None, :])
    return gasf_xy

for i, (app_traffic, poi) in enumerate(zip(app_traffics, pois)):
    # 为每个样本创建保存目录
    path_to_save = os.path.abspath(os.path.join(save_dir, str(i + 1)))
    os.makedirs(path_to_save, exist_ok=True)
    
    # 独立归一化每个特征维度
    # 使用之前计算好的每个特征的最大值
    #import pdb;pdb.set_trace()
    orig_max=np.max(app_traffic)
    app_traffic= app_traffic / feature_maxes[None, :]
    print("norm_traffic max:", np.max(app_traffic))
    #app_traffic=fast_near_min_log_normalization(app_traffic,global_max)
    #import pdb;pdb.set_trace()
    T, app_dim = app_traffic.shape
    _, poi_dim = poi.shape
    path_to_save = os.path.abspath(save_dir + str(i + 1))
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Compute Gramian angular fields
    flattened_list = []
    for t in range(192):
        gasf = to_gasf_cross(app_traffic[t],poi[t])
        flattened = gasf.flatten()     
        flattened_list.append(flattened)
    image = np.stack(flattened_list, axis=0)
    #import pdb;pdb.set_trace()
    gamma = 0.25
    image = np.power(np.abs(image), gamma) * np.sign(image)
    image=(image+1)/2
    #import pdb;pdb.set_trace()
    # Resize the image to 128x128
    h, w = image.shape
    pad_h = max(0, TARGET_H - h)
    pad_w = max(0, TARGET_W - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    #image = (image - image.min()) / (image.max() - image.min())
    #image=np.log1p(image)
    padded_img = np.pad(image,
                    pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=0)
    
    mask = np.pad(np.ones_like(image, dtype=np.uint8),
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0)
    records.append({"index": i + 1, "orig_app_max": orig_max})
    #log_img=np.log1p(padded_img)
    """padded_img_uint8=(padded_img*255).astype(np.uint8)
    Image.fromarray(padded_img_uint8).save(os.path.join(path_to_save, f"cross_gasf_{i+1}.png"))
"""
    # 保存TIFF文件
    tifffile.imwrite(
        os.path.join(path_to_save, f"cross_gasf_{i+1}.tiff"),
        padded_img,
        photometric='minisblack'  # 对于单通道图像
    )
    
    # 保存对应的PNG可视化版本
    plt.imsave(os.path.join(path_to_save, f"visualized_{i+1}.png"), padded_img, cmap='gray')
    
    # 保存mask
    plt.imsave(os.path.join(path_to_save, f"mask_{i+1}.png"), mask, cmap='gray')
    # 检查最大值是否接近原始数据的最大值
    saved_img = tifffile.imread(os.path.join(path_to_save, f"cross_gasf_{i+1}.tiff"))
    print("traffic max:", np.max(orig_max))
    
    print("gasf max:", np.max(flattened_list))
    print("padding img:", np.max(image))    
    print("padding img min:", np.min(image))  
    print("saved img:", np.max(saved_img))  
    print(f"Saved sample {i+1}: image shape={padded_img.shape}, original shape={image.shape}")

# 生成特殊样本：零流量和第一个POI位置
special_app_traffic = np.zeros_like(app_traffics[0])  # 与原始数据相同shape的全0数组
special_poi = np.zeros_like(pois[0])  # 与原始POI相同shape的数组
special_poi[:, 0] = 1  # 将第一个POI位置设为1（独热编码）

# 为特殊样本创建保存目录
special_sample_id = "norm"  # 使用下一个索引
path_to_save = os.path.abspath(os.path.join(save_dir, str(special_sample_id)))
os.makedirs(path_to_save, exist_ok=True)

# 生成特殊样本的GASF交叉图
flattened_list = []
for t in range(192):
    gasf = to_gasf_cross(special_app_traffic[t], special_poi[t])
    flattened = gasf.flatten()
    flattened_list.append(flattened)
image = np.stack(flattened_list, axis=0)

# 应用相同的图像处理步骤
gamma = 0.25
image = np.power(np.abs(image), gamma) * np.sign(image)
image = (image+1)/2

# 填充到目标尺寸
h, w = image.shape
pad_h = max(0, TARGET_H - h)
pad_w = max(0, TARGET_W - w)
pad_top = pad_h // 2
pad_bottom = pad_h - pad_top
pad_left = pad_w // 2
pad_right = pad_w - pad_left

padded_img = np.pad(image,
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0)

mask = np.pad(np.ones_like(image, dtype=np.uint8),
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0)

# 保存特殊样本
# 保存TIFF文件
tifffile.imwrite(
    os.path.join(path_to_save, f"cross_gasf_{special_sample_id}.tiff"),
    padded_img,
    photometric='minisblack'
)

# 保存对应的PNG可视化版本
plt.imsave(os.path.join(path_to_save, f"visualized_{special_sample_id}.png"), padded_img, cmap='gray')

# 保存mask
plt.imsave(os.path.join(path_to_save, f"mask_{special_sample_id}.png"), mask, cmap='gray')

records.append({"index": special_sample_id, "orig_app_max": 0})  # 添加特殊样本记录

# 保存所有记录到CSV
df = pd.DataFrame(records)
csv_path = os.path.join(save_dir, "app_traffic_orig_max_values.csv")
df.to_csv(csv_path, index=False)
print(f"All done. Max values saved to {csv_path}")
print(f"Special sample (zero traffic, first POI) saved as sample {special_sample_id}")