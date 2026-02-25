from scipy.stats import mode
import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from multiprocessing import Pool, cpu_count

file = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/all_users_data_with6cluster.npz"), allow_pickle=True)
print(file.files)
app_traffics = file['Category_ID_Traffic (Byte)']
pois = file["poi_labels"]
num_features = app_traffics.shape[2]
feature_maxes = np.max(app_traffics, axis=(0, 1))  # 形状 (num_features,)
log_feature_maxes=np.log1p(feature_maxes)
#import pdb;pdb.set_trace()
zero_mask= feature_maxes==0
print(zero_mask)
feature_maxes[zero_mask]=0
feature_mines = np.min(
    np.where(app_traffics != 0, app_traffics, np.inf),
    axis=(0, 1)  # 在第 0 维和第 1 维上计算最小值
)
print(feature_mines.shape)
print(feature_maxes)
print(feature_mines)
"""# 优化1: 向量化的RGB到灰度转换
viridis_cmap = cm.get_cmap('viridis')
gray_values = np.linspace(0, 1, 256)
viridis_rgb_values = viridis_cmap(gray_values)[:, :3]"""
"""
def rgb_to_original_gray(viridis_img):
    pixels = viridis_img.reshape(-1, 3)
    # 使用更快的向量化距离计算
    diff = pixels[:, np.newaxis, :] - viridis_rgb_values[np.newaxis, :, :]
    distances = np.einsum('ijk,ijk->ij', diff, diff)
    closest_indices = np.argmin(distances, axis=1)
    return gray_values[closest_indices].reshape(viridis_img.shape[:2])"""

# 优化2: 并行处理单个图像
def process_single_image(args):
    # MEAN_FACTOR 在新逻辑中未使用，但为保持函数签名一致而保留
    image_path, MEAN_FACTOR, max = args
    #image_path = image_path.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    T = 192
    app_n = 20
    poi_n = image_path.shape[1] // app_n
    #import pdb;pdb.set_trace()
    gasf_full = image_path.reshape(T, app_n, poi_n)
    app_trace = np.zeros((T, app_n), dtype=np.float32)
    poi_trace = np.zeros((T, poi_n), dtype=np.uint8)
    #import pdb;pdb.set_trace()
    # 优化3: 块处理优化
    for t in range(T):
        blocks = gasf_full[t]  # shape (app_n, poi_n) = (20, 7)

        # ---
        # --- START: 新的启发式算法逻辑 ---
        # ---

        # 1. 找到 POI 众数 (mode_pos)
        #    poi_pos_by_block: 每个 app (行) 对应的最大 POI (列) 索引, shape (20,)
        poi_pos_by_block = np.argmax(blocks, axis=1)
        mode_pos_result = mode(poi_pos_by_block)

        # 处理 mode 可能为空的边缘情况 (例如，如果 'blocks' 在 t 时刻全为 0)
     
        mode_pos = int(mode_pos_result[0])

        # 2. 检查 mode_pos 是否为 0 (背景)
        if mode_pos == 0:
            # 如果是背景，将 POI 设为 0 并跳过此时间步
            poi_trace[t, 0] = 1
          
            # app_trace[t] 保持为 0 (在 np.zeros 初始化时已完成)
            continue

        # ---
        # 3. 如果 mode_pos != 0, 执行新的 App 搜索逻辑
        # ---

        # 首先，设置 POI trace
        poi_trace[t, mode_pos] = 1

        # 找到每个 app (行) 中第二大值的*列索引*
        # np.argsort 对每行排序，返回索引
        # [:, -2] 取倒数第二个索引 (即第二大值的索引)
        # second_largest_col_idx 的 shape 为 (app_n,)
        second_largest_col_idx = np.argsort(blocks, axis=1)[:, -2]

        # 创建一个 "keep_mask" (T, F, T, ...) 来标记用于计算平均值的元素
        # 初始化为 (app_n, poi_n) 的 True 矩阵
        keep_mask = np.ones_like(blocks, dtype=bool)

        # 排除 mode_pos 列 (对所有 app)
        keep_mask[:, mode_pos] = False

        # 排除每个 app (行) 对应的 "second_largest_col_idx"
        # 使用高级索引 (arange(app_n) 对应行, second_largest_col_idx 对应列)
        keep_mask[np.arange(app_n), second_largest_col_idx] = False

        # 计算被掩码后的和
        # np.where(keep_mask, blocks, 0.0) -> (app_n, poi_n)
        # .sum(axis=1) -> (app_n,)
        masked_sums = np.sum(np.where(keep_mask, blocks, 0.0), axis=1)

        # 计算每行剩余的元素数量
        elements_per_row_count = keep_mask.sum(axis=1)  # (app_n,)

        # 计算平均值, 并处理除以 0 的情况
        # (如果 count 为 0, 将平均值设为 -1, 这样它就不会被 argmax 选中)
        row_averages = np.divide(masked_sums, elements_per_row_count,
                                 out=np.full(app_n, -1.0, dtype=np.float64),
                                 where=elements_per_row_count != 0)

        # 找到具有最大 "特殊平均值" 的 app (行) 索引
        app_pos = np.argmax(row_averages)

        # 4. 赋值
        # 将 (app_pos, mode_pos) 处的值赋给 app_trace
        app_trace[t, app_pos] = blocks[app_pos, mode_pos]

        # (保留你原来的调试断点)
        # ---
        # --- END: 新的启发式算法逻辑 ---
        # ---

    # 反log-normalization (这部分逻辑不变)
    non_zero_mask = app_trace != 0
    
    app_trace = inverse_transform_with_max_compensation(app_trace, feature_maxes, max)
    """is_too_small = np.all(app_trace < min_threshold[None, :], axis=1)  # shape: (T,)

    app_trace[is_too_small] = 0.0
    poi_trace[is_too_small] = 0
    poi_trace[is_too_small, 0] = 1"""
    return app_trace, poi_trace
import numpy as np
import numpy as np

def inverse_transform_with_max_compensation(app_trace, feature_maxes,max=None):
    """
    带多维最大值补偿的逆变换
    目标：使最终结果尽可能接近每个特征维度的原始最大值
    
    参数：
    padded_img_uint8: 正变换后的uint8图像 (H, W, C) 或 (H, W)
    feature_maxes: 每个特征维度的最大值 (C,) 或标量
    
    返回：
    恢复后的图像，尽可能接近每个特征维度的原始最大值
    """
    # 确保输入为浮点类型
    #import pdb;pdb.set_trace()
    if np.all(app_trace == 0):
        # 如果全为 0，则没有非零值可供计算
        # 直接返回全 0 数组 (保持正确的类型)
        return app_trace.astype(np.float32)
    app_trace = app_trace.astype(np.float64)
    """app_trace[non_zero_mask] = np.where(app_trace[non_zero_mask] < 214,
                                    214,
                                    app_trace[non_zero_mask])"""
    # 步骤1: 逆量化（补偿量化误差）
            # [0,1]

    min_val = app_trace[app_trace!=0].min()

    max_val = app_trace.max()
    row_idx = np.where(app_trace == max_val)[0][0]
    app_trace=app_trace*log_feature_maxes[None, :]
    app_trace[app_trace!=0]=np.power(np.e,app_trace[app_trace!=0]-1)
    
    #scale_factor=max/max_val
    # 由于 gamma < 1，正向变换会把高值压缩，低值拉伸。
    # 为了把误差往“偏大”引，反向 gamma 后做一点上漂移
    #app_trace[non_zero_mask] = np.power(app_trace[non_zero_mask], 1.0 / gamma)              # 1e-4 人为上偏
    #print(f'max:{app_trace.max()}')
    # 同样，给一点上漂移
    #app_trace[non_zero_mask] = app_trace[non_zero_mask]
    #print(app_trace.max())
    # 补偿gamma校正损失（每个通道独立补偿）
    # 确保不超出原始范围
    #print(feature_maxes[None, :])

    
    #app_trace=app_trace*scale_factor
    # 2. 它在哪一行（返回第一个出现的位置）
    min_threshold = feature_mines[None, :] * 0.9  # 使用 1.3 倍最小值作为阈值
    app_trace = np.where(
        app_trace < min_threshold,  # 小于最小值阈值
        0,                       # 置零
        app_trace                 # 保持原值
    )
    if np.all(app_trace == 0):
        # 如果全为 0，则没有非零值可供计算
        # 直接返回全 0 数组 (保持正确的类型)
        return app_trace.astype(np.float32)

    # 3. 打印这一行
    print("#########")
    print("min_value",min_val)
    print("max value:", max_val)
    print("max value after scale:", app_trace.max())
    print("min value after scale:", app_trace[app_trace!=0].min())
    # 计算非零最小值
    print(app_trace.shape)
    print("Min value :",np.min(app_trace[app_trace!= 0]))
    print("row index:", row_idx)
    #print("that row :", app_trace[row_idx])
    """for c in range(app_trace.shape[-1]):
        app_trace[..., c] = np.clip(app_trace[..., c], 0, feature_maxes[c] * 1.05)
    """
    return app_trace.astype(np.float32) 

def recover_traces( MEAN_FACTOR=1.2, output_suffix=""):
    
    stats = {
        "app_assigned": 0,
        "app_not_assigned": 0,
    }
    
    """    image_list = [os.path.join(image_root, img) for img in os.listdir(image_root) 
                    if img.endswith(('.png', '.jpg', '.jpeg'))]"""
    
    # 并行处理所有图像
    #max_estim=pd.read_csv('/home/yilai/poster/NetDiffus/ckpt/app_8/ema_0.9999_058000.pt_max_predictions.csv')
    image_list=np.load('/data/yilai/MiDiff/ckpt/ckpt/tiff_log_attnBoth/model048500.pt_samples_3000x256x160x1.npz')['arr_0']
    #import pdb;pdb.set_trace()
    def center_crop(img, target_h, target_w):
        """img: ndarray (..., H, W, C)"""
        h, w = img.shape[-3], img.shape[-2]
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return img[..., start_h:start_h + target_h, start_w:start_w + target_w, :]

# 用法示例
    image_list = center_crop(image_list, 192, 140)
    #import pdb;pdb.set_trace()
    process_single_image([image_list[0],1.5,None])
    with Pool(processes=cpu_count()) as pool:
        args_list = [(img_path, MEAN_FACTOR,max) for img_path,max in zip(image_list,[None]*len(image_list))]
        results = pool.map(process_single_image, args_list)
        
    # 收集结果
    all_app_traces = []
    all_poi_traces = []
    for app_trace, poi_trace in results:
        if np.all(app_trace == 0):
        # 如果全为 0，则没有非零值可供计算
        # 直接返回全 0 数组 (保持正确的类型)
            continue
        all_app_traces.append(app_trace)
        all_poi_traces.append(poi_trace)
    # 统计信息需要在实际处理中收集，这里简化了
    #print(stats["app_assigned"]/stats["app_not_assigned"])
    output_path = f"/data/yilai/MiDiff/ckpt/ckpt/tiff_log_attnBoth/48500.npz"
    print(np.unique(poi_trace))
    np.savez(
        output_path,
        app_traces=np.array(all_app_traces),
        poi_traces=np.array(all_poi_traces),
        MEAN_FACTOR=MEAN_FACTOR,
        stats=stats
    )
    print(f"Saved to {output_path}")
    ####统计行为数量
    
    poi_traces_arr = np.array(all_poi_traces, dtype=np.uint8)  # 形状 (N, T, poi_n)
    
    valid_mask=np.any(np.array(all_app_traces)!=0,axis=(2))
    #import pdb;pdb.set_trace()
    # 标记每一行是否出现了列 n>0 的 1
    has_poi_rows_per_image = np.any(poi_traces_arr[:, :, 1:] == 1, axis=2)  # (N, T)
    has_poi_rows_per_image[~valid_mask]=False
    num_row_per_image = has_poi_rows_per_image.sum(axis=1).astype(np.int32)  # (N,)

    # 单独剔除 num_row == 0 的样本（不纳入均值/方差）
    valid_mask = num_row_per_image > 0
    num_row_valid = num_row_per_image[valid_mask]

    if num_row_valid.size == 0:
        print(f"[MEAN_FACTOR={MEAN_FACTOR}] num_row 统计：没有可用样本（过滤后且 num_row>0 的样本为 0）")
    else:
        mean_num_row = float(np.mean(num_row_valid))
        var_num_row  = float(np.std(num_row_valid, ddof=0))  # 总体方差
        print(f"[MEAN_FACTOR={MEAN_FACTOR}] num_row 统计：mean={mean_num_row:.6f}, var={var_num_row:.6f}, images={num_row_valid.size}")


for mean_factor in [0]:
    recover_traces(MEAN_FACTOR=mean_factor, output_suffix="")