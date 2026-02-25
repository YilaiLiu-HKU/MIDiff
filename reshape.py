import numpy as np
import os
import pandas as pd
import torch
def read_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    df = pd.read_csv(file_path, header=None)
    return df.values.astype(np.float64)

def load_and_preprocess_samples(torch_file_path, target_seq_len=192):
    """
    加载生成样本并进行预处理：
    1. 零填充到目标序列长度
    2. 对第三维后两位取整
    3. 重塑为2D数组 (batch*seq_len, 3)
    
    参数:
        torch_file_path: .pth文件路径
        target_seq_len: 目标序列长度(默认192)
    
    返回:
        处理后的numpy数组，形状为(batch*seq_len, 3)
    """
    # 加载原始数据
    samples = torch.load(torch_file_path)
    
    # 转换为numpy数组
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    elif isinstance(samples, list):
        samples = torch.concat(samples,dim=0).detach()
    #import pdb;pdb.set_trace()
    # 检查输入形状 (batch, seq_len, 3)
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError(f"输入形状应为(batch, seq_len, 3)，实际得到{samples.shape}")
    
    batch_size, seq_len, _ = samples.shape
    
    # 零填充到目标长度
    if seq_len < target_seq_len:
        pad_width = [(0, 0), (0, target_seq_len - seq_len), (0, 0)]
        samples = np.pad(samples, pad_width, mode='constant')
    elif seq_len > target_seq_len:
        samples = samples[:, :target_seq_len, :]
    
    # 对第三维的后两位取整
    samples[..., 1:] = np.round(samples[..., 1:])

    
    return samples
def split_into_three_realdimensions(data):
    n_samples, n_features = data.shape
    assert n_features % 3 == 0, "特征数必须是3的倍数"
    seq_len = n_features // 3
    data_3d = data.reshape(n_samples, seq_len, 3)
    return data_3d[:, :, 0], data_3d[:, :, 1], data_3d[:, :, 2]
def split_into_three_dimensions(data):
    """将每行数据三等分为三个维度"""
    if data.shape[1] % 3 != 0:
        raise ValueError("数据列数必须能被3整除")
    
    n_features = data.shape[1] // 3
    dim1 = data[:, :n_features]
    dim2 = data[:, n_features:2*n_features]
    dim3 = data[:, 2*n_features:]
    return dim1, dim2, dim3
data_path='/home/yilai/poster/tts-gan/output_flattened.csv'
use_pt=False
save_dir='/home/yilai/poster/NetDiffus/downstream/dataset/'
save_name='ttsGAN'
os.makedirs(save_dir,exist_ok=True)
if use_pt:
    synth_data=load_and_preprocess_samples(data_path)
    s1, s2, s3 = np.squeeze(np.split(synth_data,3,axis=-1))
else:
    synth_data = read_dataset(data_path)
    s1, s2, s3 = split_into_three_dimensions(synth_data)
minus0_s1=s1<0
#import pdb;pdb.set_trace()
print(s1.max())
print(s2.max())
print(s3.max())
s1[minus0_s1]=0

"""np.savez(
        '/home/yilai/poster/NetDiffus/our_traces.npz',
        app_Traffic=file['app_traces'],
        POI=file['poi_traces'],
    )"""
num_classes_s3 = 7
s3=np.clip(np.round(s3),0,num_classes_s3-1)
print(f"s3的类别数量: {num_classes_s3}")

# 创建s3的独热编码
s3_onehot = np.zeros((s3.shape[0], s3.shape[1], num_classes_s3), dtype=np.float64)

# 填充独热编码
for i in range(s3.shape[0]):
    for j in range(s3.shape[1]):
        class_idx = int(s3[i, j])
        s3_onehot[i, j, class_idx] = 1.0

# 2. 处理s2：特殊独热编码
# 确定s2中的最大类别数

num_classes_s2 = 20
print(f"s2的类别数量: {num_classes_s2}")
s2=np.clip(np.round(s2),0,num_classes_s2-1)
# 创建s2的特殊编码
s2_special = np.zeros((s2.shape[0], s2.shape[1], num_classes_s2), dtype=np.float64)

# 填充特殊编码
for i in range(s2.shape[0]):
    for j in range(s2.shape[1]):
        if s1[i, j] == 0:
            # s1为0时：全0编码
            continue  # 已经是全0，无需操作
        else:
            # s1不为0时：正常独热编码，但将1替换为s1的值
            class_idx = int(s2[i, j])
            s2_special[i, j, class_idx] = s1[i, j]

# 验证结果形状
print(f"s1形状: {s1.shape}")
print(f"s2_special形状: {s2_special.shape} (每个192维特征扩展为{num_classes_s2}维)")
print(f"s3_onehot形状: {s3_onehot.shape} (每个192维特征扩展为{num_classes_s3}维)")

# 可选：保存处理后的数据
np.savez(
        os.path.join(save_dir,save_name),
        app_Traffic=s2_special,
        POI=s3_onehot,
    )