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
#import pdb;pdb.set_trace()
zero_mask= feature_maxes==0
print(zero_mask)
feature_maxes[zero_mask]=0
log_feature_maxes=np.log1p(feature_maxes)
feature_mines = np.min(
    np.where(app_traffics != 0, app_traffics, np.inf),
    axis=(0, 1)  # 在第 0 维和第 1 维上计算最小值
)
print(feature_mines.shape)
print(feature_maxes)
print(feature_mines)
print((feature_maxes/feature_mines)[feature_mines!=np.inf].min())
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
    image_path, MEAN_FACTOR,max = args
    T = 192
    app_n = 20
    poi_n = image_path.shape[1]//app_n

    gasf_full = image_path.reshape(T, app_n, poi_n)
    app_trace = np.zeros((T, app_n), dtype=np.float32)
    poi_trace = np.zeros((T, poi_n), dtype=np.uint8)
    
    for t in range(T):
        blocks = gasf_full[t]
        block_means = blocks.mean(axis=1)
        """total_mean = block_means.sum()
        others_mean = (total_mean - block_means) / (app_n - 1)
        
        # 向量化比较
        is_significant = block_means > (others_mean * MEAN_FACTOR)
        candidates = np.where(is_significant)[0]
        if candidates is None:
            poi_trace[t, 0] = 1
            print("no candidates")
            continue"""
        blocks = gasf_full[t]
        
        # 1. 计算每个POI block的稳健代表值
        #    axis=1 表示沿着app_n维度计算

        #print(active_poi_index)
        first_pos = np.argmax(blocks, axis=0)
        

        vals, cnts = np.unique(first_pos, return_counts=True)
        top2_idx = np.argsort(cnts)[::-1][:2]
        top2_pos = vals[top2_idx]
        
        if top2_pos[0]==0:
            if len(top2_pos)==1:
                poi_trace[t, 0] = 1
                continue
            else:
                app_pos=top2_pos[1]
                #if app_pos not in top_four_indices:
                #    continue 
        else:
            #print(f"block_medians: {block_medians}, block_p90s: {block_p90s}, combined_scores: {combined_scores}, active_poi_index: {active_poi_index}, top2_pos: {top2_pos}")
            app_pos=top2_pos[0]
            if 0 in top_four_indices:
                app_pos=0
                pass
            elif app_pos not in top_four_indices:
                continue              
        poi_pos_by_block = np.argmax(blocks, axis=1)
        mode_pos = mode(poi_pos_by_block)[0]
        
        block_max = blocks.max()




        block_medians = np.median(blocks, axis=1)
        block_p90s = np.percentile(blocks, 80, axis=1)
        
        # 2. 计算每个block的综合得分
        #    我们只关心正值区域，所以可以将负分位数裁剪为0，以防它们影响排序
        #block_p90s[block_p90s < 0] = 0
        
        combined_scores = (1 - MEAN_FACTOR) * block_medians + MEAN_FACTOR * block_p90s
        
        # 3. 找到得分最高的block的索引
        sorted_indices = np.argsort(combined_scores)

        # 2. 从排序后的索引数组中，取最后四个索引
        #    这四个索引就对应了分数从第四高到第一高的block
        top_four_indices = sorted_indices[:1]
        transformed_max = inverse_transform_value(block_max, feature_maxes[app_pos])
        
        # 检查是否低于最小值阈值
        if transformed_max < feature_mines[app_pos] * 0.9:
            poi_trace[t, 0] = 1
            continue
            


        
        app_trace[t, app_pos] = block_max
        poi_trace[t, mode_pos] = 1
            
    # 反log-normalization
    non_zero_mask = app_trace != 0
    # 对非零部分执行逆变换
    
    # Debug: Check values before transformations
    
    # Apply transformations
    non_zero_mask = app_trace != 0

    """app_trace[non_zero_mask] = (app_trace[non_zero_mask]+0.3) / 255.0
    print("After /255:", app_trace.max())
    
    app_trace = np.power(np.maximum(app_trace, 1e-8), 
                       4.0 * (1 + 0.0002))
    print("After **4:", app_trace.max())
    
    app_trace[non_zero_mask] = (app_trace[non_zero_mask] - 0.5)/0.5
    print("After scaling to [-1,1]:", app_trace.max())
    print(app_trace.max())
    app_trace = app_trace * feature_maxes[None, :]
    print(app_trace.max())
    exit()"""
    app_trace=inverse_transform_with_max_compensation(app_trace,feature_maxes,max)
    #exit()

    #postprocessing, filter low noises
    min_threshold = feature_mines * 1.3  # shape: (num_features,)
    """is_too_small = np.all(app_trace < min_threshold[None, :], axis=1)  # shape: (T,)

    app_trace[is_too_small] = 0.0
    poi_trace[is_too_small] = 0
    poi_trace[is_too_small, 0] = 1"""
    return app_trace, poi_trace
import numpy as np
import numpy as np

def inverse_transform_value(value, feature_max):
    """对单个值进行逆变换计算"""
    if value == 0:
        return 0
        
    value = value * 2 - 1  # 归一化到[-1,1]
    gamma = 0.25
    value = np.power(np.abs(value), 1/gamma) * np.sign(value)
    return value * feature_max

def inverse_transform_with_max_compensation(app_trace, feature_maxes, max=None):
    """带多维最大值补偿的逆变换"""
    non_zero_mask = app_trace != 0
    app_trace = app_trace.astype(np.float64)
    
    max_val = app_trace.max()
    
    row_idx = np.where(app_trace == max_val)[0][0]
    app_trace=app_trace*log_feature_maxes[None, :]
    app_trace[app_trace!=0]=np.power(np.e,app_trace[app_trace!=0]-1)
    
    # 应用最小值阈值
    min_threshold = feature_mines[None, :] * 0.9  # 使用 1.3 倍最小值作为阈值
    app_trace = np.where(
        app_trace < min_threshold,  # 小于最小值阈值
        0,                       # 置零
        app_trace                 # 保持原值
    )
    min_val=np.min(app_trace[app_trace != 0]) if np.any(app_trace != 0) else 0
    nonzero_vals = app_trace[app_trace != 0]
    """print("#########")
    print(app_trace.shape)
    print("min value after min thresholding:", min_val)
    print("max value:", max_val)
    print("max value after scale:", app_trace.max())
    
    if nonzero_vals.size > 0:
        print("Min value after scale:", np.min(nonzero_vals))
    else:
        print("Min value: N/A (all zeros)")
    print("row index:", row_idx)"""
    
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
    image_list=np.load('/data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr/ema_0.9999_046000.pt_samples_3000x256x160x1.npz')['arr_0']
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
    removed = 0

    for app_trace, poi_trace in results:
        if np.any(app_trace != 0):
            all_app_traces.append(app_trace)
            all_poi_traces.append(poi_trace)
        else:
            removed += 1  # 这条样本被丢弃

    print(f"Filtered out {removed} empty app_traces")
    # 统计信息需要在实际处理中收集，这里简化了
    #print(stats["app_assigned"]/stats["app_not_assigned"])
    output_path = f"/data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr/ema046000_filted.npz"
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
    # 标记每一行是否出现了列 n>0 的 1
    
    has_poi_rows_per_image = np.any(poi_traces_arr[:, :, 1:] == 1, axis=2)  # (N, T)
    
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


for mean_factor in [0,0.5,1]:
    recover_traces(MEAN_FACTOR=mean_factor, output_suffix="")