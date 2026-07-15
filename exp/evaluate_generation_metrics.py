# -*- coding: utf-8 -*-
"""
批量评估脚本（CSV 版），并将结果写入一个 Excel（含 4 个 sheet）：
1) DimMetrics：逐维 MAE/RMSE/JSD
2) Table2：VDS、FDDS、DA、Predictive Score (DA 和 PredictiveScore 现为 mean/std)
3) Table3：MDD, ACD, SD, KD, ED, DTW
4) CorrSim：Correlation cosine / Frobenius diff（逐维 + overall）

说明见函数内部注释。
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict

from scipy.stats import entropy, skew, kurtosis, wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample  # 新增：用于自举采样

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_FILE = str(REPO_ROOT / "data" / "our.csv")
DEFAULT_MIDIFF_FILE = str(REPO_ROOT / "exp" / "results" / "MIDiff.csv")
DEFAULT_OUTPUT_XLSX = str(REPO_ROOT / "exp" / "results" / "eval_midiff_real.xlsx")

def transform_non_zero_to_one(data: np.ndarray) -> np.ndarray:
    """
    将所有非零元素转为1，零元素保持不变。
    """
    return np.where(data != 0, 1, 0)

# ---------- I/O 与形状 ----------
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
    assert n_features % 3 == 0, "特征数必须是3的倍数"
    seq_len = n_features // 3
    data_3d = data.reshape(n_samples, seq_len, 3)
    return data_3d[:, :, 0], data_3d[:, :, 1], data_3d[:, :, 2]

def normalize_with_max(data: np.ndarray, max_val: float) -> np.ndarray:
    """线性归一到 [0,1]，用给定最大值（若 max_val=0 或 None 则返回原值）"""
    return data / max_val if max_val not in (0, None) else data

# ---------- JS 散度与逐维指标 ----------
def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def calc_dim_metrics(real: np.ndarray, synth: np.ndarray) -> Tuple[float, float, float]:
    """
    real/synth: [n_samples, seq_len]
    汇总所有位置误差（忽略 real=synth=0 的点）；行级 JSD 取均值。
    """
    mask = ~((real == 0) & (synth == 0))
    diff = real - synth
    mae  = float(np.mean(np.abs(diff[mask]))) if np.any(mask) else 0.0
    rmse = float(np.sqrt(np.mean(diff[mask] ** 2))) if np.any(mask) else 0.0

    jsd_vals = []
    jsd_vals_01 = []
    for r, s in zip(real, synth):
        if not (np.all(r == 0) and np.all(s == 0)):
            r_shift = r - r.min() if r.min() < 0 else r
            s_shift = s - s.min() if s.min() < 0 else s
            lo = min(r_shift.min(), s_shift.min())
            hi = max(r_shift.max(), s_shift.max())
            if hi == lo:
                continue
            bins = np.linspace(lo, hi, 51)
            pr, _ = np.histogram(r_shift, bins=bins, density=True)
            ps, _ = np.histogram(s_shift, bins=bins, density=True)
            jsd = js_divergence(pr, ps)
            jsd_vals.append(jsd)

            # 01化后的 JSD
            jsd_01 = transform_non_zero_to_one(np.array([jsd]))[0]
            jsd_vals_01.append(jsd_01)

    jsd = float(np.mean(jsd_vals)) if jsd_vals else 0.0
    jsd_01 = float(np.mean(jsd_vals_01)) if jsd_vals_01 else 0.0

    return mae, rmse, jsd, jsd_01

# ---------- 相关系数分布（FDDS） ----------
def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("x 与 y 的长度必须一致")
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    sx = np.std(xm, ddof=1); sy = np.std(ym, ddof=1)
    denom = sx * sy
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(xm, ym) / ((len(x) - 1) * denom))

def corr_per_sample(dimA_2d: np.ndarray, dimB_2d: np.ndarray) -> np.ndarray:
    if dimA_2d.shape != dimB_2d.shape:
        raise ValueError("两个维度矩阵的形状必须一致")
    n = dimA_2d.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _corr_1d(dimA_2d[i], dimB_2d[i])
    return out

# ---------- Table2：VDS / FDDS / DA / Predictive ----------
def vds_score(r_dims: Tuple[np.ndarray, np.ndarray, np.ndarray],
              s_dims: Tuple[np.ndarray, np.ndarray, np.ndarray],
              bins: int = 50) -> float:
    scores = []
    for r, s in zip(r_dims, s_dims):
        r_flat = r.reshape(-1); s_flat = s.reshape(-1)
        lo = min(r_flat.min(), s_flat.min()); hi = max(r_flat.max(), s_flat.max())
        if hi == lo:
            scores.append(0.0); continue
        edges = np.linspace(lo, hi, bins + 1)
        pr, _ = np.histogram(r_flat, bins=edges, density=True)
        ps, _ = np.histogram(s_flat, bins=edges, density=True)
        scores.append(js_divergence(pr, ps))
    return float(np.mean(scores)) if scores else 0.0

def fdds_score(r_dims: Tuple[np.ndarray, np.ndarray, np.ndarray],
               s_dims: Tuple[np.ndarray, np.ndarray, np.ndarray],
               bins: int = 50) -> float:
    pairs = [(0,1),(0,2),(1,2)]
    r_list = [r_dims[0], r_dims[1], r_dims[2]]
    s_list = [s_dims[0], s_dims[1], s_dims[2]]
    scores = []
    for a,b in pairs:
        rc = corr_per_sample(r_list[a], r_list[b])
        sc = corr_per_sample(s_list[a], s_list[b])
        edges = np.linspace(-1, 1, bins + 1)
        pr, _ = np.histogram(rc[np.isfinite(rc)], bins=edges, density=True)
        ps, _ = np.histogram(sc[np.isfinite(sc)], bins=edges, density=True)
        scores.append(js_divergence(pr, ps))
    return float(np.mean(scores)) if scores else 0.0

def discriminative_accuracy_da(real_flat: np.ndarray,
                               synth_flat: np.ndarray,
                               test_size: float = 0.3,
                               n_runs: int = 5,  # 新增：运行次数
                               random_state: int = 0) -> Tuple[float, float]: # 修改：返回 (mean, std)
    """
    real_flat/synth_flat: [n_samples, 3*seq_len] 的二维矩阵（即原 CSV 行）
    返回 (mean_acc, std_acc)，在 n_runs 次不同随机切分下计算。
    """
    acc_scores = []
    for i in range(n_runs):
        # 每次运行使用不同的随机种子
        run_seed = random_state + i
        
        X = np.vstack([real_flat, synth_flat])
        y = np.concatenate([np.ones(len(real_flat)), np.zeros(len(synth_flat))])
        
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=run_seed)
        
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
        
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(Xtr_s, ytr)
        
        acc = float(clf.score(Xte_s, yte))
        acc_scores.append(acc)
    
    mean_acc = float(np.mean(acc_scores))
    std_acc = float(np.std(acc_scores))
    
    return mean_acc, std_acc

def predictive_score_mae(s_dims: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         r_dims: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         lag: int = 5,
                         n_runs: int = 5,       # 新增：运行次数
                         random_state: int = 0) -> Tuple[float, float]: # 修改：返回 (mean, std)
    """
    用合成训练、真实评估的一步预测 MAE（各维独立线性回归，最后取均值）
    通过 n_runs 次自举采样（bootstrap）合成训练集来估计均值和标准差。
    """
    run_maes = []  # 存储每次运行的平均 MAE
    rng = np.random.default_rng(random_state) # 随机数生成器
    
    for _ in range(n_runs):
        dim_maes = []  # 存储这个 run 内部各维度的 MAE
        for sd, rd in zip(s_dims, r_dims):
            def build_xy(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                X_list, y_list = [], []
                for row in mat:
                    if row.shape[0] <= lag:
                        continue
                    for t in range(lag, row.shape[0]):
                        X_list.append(row[t-lag:t])
                        y_list.append(row[t])
                if not X_list:
                    return np.empty((0, lag)), np.empty((0,))
                return np.vstack(X_list), np.array(y_list)

            Xs, ys = build_xy(sd); Xr, yr = build_xy(rd)
            
            if len(ys) == 0 or len(yr) == 0:
                dim_maes.append(np.nan); continue

            # --- 新增：从合成数据中进行自举采样 ---
            # 为 resample 生成一个种子
            boot_seed = int(rng.integers(2**31 - 1))
            Xs_boot, ys_boot = resample(Xs, ys, replace=True, n_samples=len(ys), random_state=boot_seed)
            # --- 结束 ---
            
            if len(ys_boot) == 0: # 可能采样到 0
                dim_maes.append(np.nan); continue

            scaler = StandardScaler().fit(Xs_boot) # 在自举样本上 fit
            Xs_boot_s = scaler.transform(Xs_boot)
            Xr_s = scaler.transform(Xr) # 用同样的 scaler 转换真实数据

            model = Ridge(alpha=1.0) # Ridge 不再需要 random_state
            model.fit(Xs_boot_s, ys_boot) # 在自举样本上训练
            
            yhat = model.predict(Xr_s)
            dim_maes.append(float(np.mean(np.abs(yhat - yr))))
        
        # 计算这个 run 的平均 MAE (跨维度)
        maes_for_run = [m for m in dim_maes if np.isfinite(m)]
        run_maes.append(float(np.mean(maes_for_run)) if maes_for_run else np.nan)
    
    # 计算所有 run 的最终均值和标准差
    final_maes = [m for m in run_maes if np.isfinite(m)]
    mean_mae = float(np.mean(final_maes)) if final_maes else np.nan
    std_mae = float(np.std(final_maes)) if final_maes else np.nan
    
    return mean_mae, std_mae

# ---------- Table3：MDD / ACD / SD / KD / ED / DTW ----------
def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - np.mean(x)
    denom = np.sum(x ** 2)
    if denom == 0:
        return np.zeros(max_lag)
    out = np.empty(max_lag, dtype=np.float64)
    for k in range(1, max_lag + 1):
        out[k-1] = np.dot(x[:-k], x[k:]) / denom
    return out

def avg_acf_over_samples(mat: np.ndarray, max_lag: int) -> np.ndarray:
    if mat.shape[1] <= 1:
        return np.zeros(max_lag)
    max_lag = min(max_lag, mat.shape[1]-1)
    if max_lag <= 0:
        return np.zeros(1)
    acfs = [acf_1d(row, max_lag) for row in mat]
    return np.mean(np.vstack(acfs), axis=0)

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf, dtype=np.float64)
    D[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(a[i-1] - b[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n,m])

def table3_metrics(r_dims: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   s_dims: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   ac_max_lag: int = 20) -> Dict[str, float]:
    mdds = []
    for r, s in zip(r_dims, s_dims):
        r_flat = r.reshape(-1); s_flat = s.reshape(-1)
        mdds.append(wasserstein_distance(r_flat, s_flat))
    MDD = float(np.mean(mdds)) if mdds else np.nan

    ACDs = []
    for r, s in zip(r_dims, s_dims):
        ar = avg_acf_over_samples(r, ac_max_lag)
        as_ = avg_acf_over_samples(s, ac_max_lag)
        L = min(len(ar), len(as_))
        ACDs.append(np.mean(np.abs(ar[:L] - as_[:L])))
    ACD = float(np.mean(ACDs)) if ACDs else np.nan

    SDs, KDs = [], []
    for r, s in zip(r_dims, s_dims):
        r_flat = r.reshape(-1); s_flat = s.reshape(-1)
        SDs.append(abs(skew(r_flat, bias=False) - skew(s_flat, bias=False)))
        KDs.append(abs(kurtosis(r_flat, fisher=False, bias=False) - kurtosis(s_flat, fisher=False, bias=False)))
    SD = float(np.mean(SDs)) if SDs else np.nan
    KD = float(np.mean(KDs)) if KDs else np.nan

    mu_r = np.array([np.mean(r.reshape(-1)) for r in r_dims])
    mu_s = np.array([np.mean(s.reshape(-1)) for s in s_dims])
    ED = float(np.linalg.norm(mu_r - mu_s, ord=2))

    DTWs = []
    for r, s in zip(r_dims, s_dims):
        r_mean_ts = np.mean(r, axis=0)
        s_mean_ts = np.mean(s, axis=0)
        DTWs.append(dtw_distance(r_mean_ts, s_mean_ts))
    DTW = float(np.mean(DTWs)) if DTWs else np.nan

    return {"MDD": MDD, "ACD": ACD, "SD": SD, "KD": KD, "ED": ED, "DTW": DTW}

# ---------- 新增：相关矩阵相似度（余弦 & Fro） ----------
def _corr_matrix_similarity(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    """
    A/B: [n_samples, seq_len]
    计算列间相关系数矩阵 R_A, R_B（以列=时间步为变量），返回：
    - cosine 相似度：cos( vec(R_A), vec(R_B) )，越大越好
    - Frobenius 差：||R_A - R_B||_F，越小越好
    为避免合成数据出现常量列导致 NaN，这里自动做一个极小扰动。
    """
    def _fix_const_cols(X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        std = np.std(X, axis=0)
        const = (std == 0.0)
        if np.any(const):
            X[:, const] += np.random.default_rng(42).normal(0.0, 1e-8, size=(X.shape[0], int(const.sum())))
        return X

    A = _fix_const_cols(A); B = _fix_const_cols(B)
    RA = np.corrcoef(A, rowvar=False)
    RB = np.corrcoef(B, rowvar=False)

    va = RA.ravel(); vb = RB.ravel()
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    cosine = float(np.dot(va, vb) / denom) if denom != 0 else 0.0
    fro = float(np.linalg.norm(RA - RB, ord='fro'))
    return cosine, fro

# ---------- 批处理与汇总 ----------
def evaluate_one_synth(real_file: str,
                       synth_file: str,
                       norm_max_vals: Optional[Tuple[float,float,float]] = None,
                       js_bins: int = 50,
                       fdds_bins: int = 50,
                       ac_max_lag: int = 20,
                       da_test_size: float = 0.3,
                       da_random_state: int = 0,
                       da_n_runs: int = 5,          # 新增：DA 运行次数
                       pred_lag: int = 5,
                       pred_n_runs: int = 5,         # 新增：Predictive 运行次数
                       norm_list:str = None) -> Dict[str, object]:
    """
    对单个 synth_file 与 real_file 的对比，输出三组结果：
    - Table2（VDS/FDDS/DA/Predictive）
    - Table3（MDD/ACD/SD/KD/ED/DTW）
    - CorrSim（逐维 + overall 的相关矩阵相似度）
    """
    real_data  = read_dataset(real_file)[:2693]   # [N, 3L]
    synth_data = read_dataset(synth_file)[:2693]  # [N, 3L]
    synth_data=synth_data[:real_data.shape[0]]
    if real_data.shape != synth_data.shape:
        raise ValueError(f"形状不匹配: real {real_data.shape} vs synth {synth_data.shape}")


    r1, r2, r3 = split_into_three_realdimensions(real_data)
    s1, s2, s3 = split_into_three_realdimensions(synth_data)

    # 可选归一化
    if norm_max_vals is not None:
        if synth_file in norm_list:
            m1, m2, m3 = norm_max_vals
            r1 = normalize_with_max(r1, m1); r2 = normalize_with_max(r2, m2); r3 = normalize_with_max(r3, m3)
            s1 = normalize_with_max(s1, m1); s2 = normalize_with_max(s2, m2); s3 = normalize_with_max(s3, m3)

    # ---- 1) Table2
    VDS  = vds_score((r1,r2,r3), (s1,s2,s3), bins=js_bins)
    FDDS = fdds_score((r1,r2,r3), (s1,s2,s3), bins=fdds_bins)
    
    # 修改：DA 现在返回均值和标准差
    DA_mean, DA_std = discriminative_accuracy_da(
        real_data, synth_data,
        test_size=da_test_size, 
        n_runs=da_n_runs,         # 传入 n_runs
        random_state=da_random_state
    )
    DA_mean -= 0.5
    # 修改：Predictive Score 现在返回均值和标准差
    PRED_mean, PRED_std = predictive_score_mae(
        (s1,s2,s3), (r1,r2,r3), 
        lag=pred_lag,
        n_runs=pred_n_runs,        # 传入 n_runs
        random_state=da_random_state # 重用 random_state 作为基础种子
    )

    # ---- 2) Table3
    t3 = table3_metrics((r1,r2,r3), (s1,s2,s3), ac_max_lag=ac_max_lag)

    # ---- 3) CorrSim：逐维 + overall
    cos1, fro1 = _corr_matrix_similarity(r1, s1)
    cos2, fro2 = _corr_matrix_similarity(r2, s2)
    cos3, fro3 = _corr_matrix_similarity(r3, s3)

    # overall：把三个维度还原到 [N, L, 3] 再展平时间为列做相关矩阵
    # 注意 real_data/synth_data 已经是 [N, 3L]，直接 reshape 回 [N,L,3]
    L = real_data.shape[1] // 3
    r3d = real_data.reshape(real_data.shape[0], L, 3)
    s3d = synth_data.reshape(synth_data.shape[0], L, 3)
    # 这里把 3 个通道按列拼成 [N, L*3]，以“全部时间×通道”为变量计算相关矩阵
    r_all = r3d.reshape(r3d.shape[0], L * 3)
    s_all = s3d.reshape(s3d.shape[0], L * 3)
    cos_all, fro_all = _corr_matrix_similarity(r_all, s_all)

    ###0-1表
    r2_transformed = transform_non_zero_to_one(r2)
    r3_transformed = transform_non_zero_to_one(r3)
    s2_transformed = transform_non_zero_to_one(s2)
    s3_transformed = transform_non_zero_to_one(s3)

    # ---- 4) CorrSim：逐维 + overall
    cos1_01, fro1_01 = _corr_matrix_similarity(r1, s1)
    cos2_01, fro2_01 = _corr_matrix_similarity(r2_transformed, s2_transformed)
    cos3_01, fro3_01 = _corr_matrix_similarity(r3_transformed, s3_transformed)

    # overall：把三个维度还原到 [N, L, 3] 再展平时间为列做相关矩阵
    r3d = real_data.reshape(real_data.shape[0], L, 3)
    s3d = synth_data.reshape(synth_data.shape[0], L, 3)
    r_all = r3d.reshape(r3d.shape[0], L * 3)
    s_all = s3d.reshape(s3d.shape[0], L * 3)
    cos_all_01, fro_all_01 = _corr_matrix_similarity(r_all, s_all)

    return {
        "Table2": { # 修改：更新字典键
            "VDS": VDS, 
            "FDDS": FDDS, 
            "DA_mean": DA_mean, 
            "DA_std": DA_std, 
            "PredictiveScore_mean": PRED_mean,
            "PredictiveScore_std": PRED_std
        },
        "Table3": t3,
        "CorrSim": {
            "cosine_dim1": cos1, "frobenius_dim1": fro1,
            "cosine_dim2": cos2, "frobenius_dim2": fro2,
            "cosine_dim3": cos3, "frobenius_dim3": fro3,
            "cosine_overall": cos_all, "frobenius_overall": fro_all
        },
        "CorrSim_01": {
            "cosine_dim1": cos1_01, "frobenius_dim1": fro1_01,
            "cosine_dim2": cos2_01, "frobenius_dim2": fro2_01,
            "cosine_dim3": cos3_01, "frobenius_dim3": fro3_01,
            "cosine_overall": cos_all_01, "frobenius_overall": fro_all_01
        }
    }

def _format_delta(current: float, baseline: float, metric_name: str) -> object:
    if not np.isfinite(current) or not np.isfinite(baseline):
        return np.nan

    if metric_name.startswith("DA_"):
        return float(np.clip(current - baseline, -1000.0, 1000.0))

    if baseline == 0:
        if current == 0:
            pct = 0.0
        else:
            pct = 1000.0 if current > 0 else -1000.0
    else:
        pct = ((current - baseline) / abs(baseline)) * 100.0
    pct = float(np.clip(pct, -1000.0, 1000.0))
    return f"{pct:.2f}%"

def _append_delta_columns(df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    if df.empty or "synth_file" not in df.columns:
        return df

    baseline_row = df.loc[df["synth_file"] == baseline_name]
    if baseline_row.empty:
        return df
    baseline_row = baseline_row.iloc[0]

    ordered_cols = ["synth_file"]
    for col in df.columns:
        if col == "synth_file":
            continue
        ordered_cols.append(col)
        ordered_cols.append(f"{col}_delta")
        baseline_val = baseline_row[col]
        df[f"{col}_delta"] = df[col].apply(lambda x, c=col, b=baseline_val: _format_delta(x, b, c))

    return df[ordered_cols]

def _excel_writer(out_xlsx: str):
    """
    优先用 xlsxwriter；没有则退回 openpyxl；都没有则抛出提示。
    """
    try:
        import xlsxwriter  # noqa: F401
        return pd.ExcelWriter(out_xlsx, engine="xlsxwriter")
    except Exception:
        try:
            import openpyxl  # noqa: F401
            return pd.ExcelWriter(out_xlsx, engine="openpyxl")
        except Exception:
            raise RuntimeError(
                "需要安装 Excel 写入引擎：pip install XlsxWriter  或  pip install openpyxl"
            )

def run_batch_to_excel(real_file: str,
                       synth_files: List[str],
                       out_xlsx: str,
                       baseline_synth_file: Optional[str] = None,
                       norm_max_vals: Optional[Tuple[float,float,float]] = None,
                       js_bins: int = 50,
                       fdds_bins: int = 50,
                       ac_max_lag: int = 20,
                       da_test_size: float = 0.3,
                       da_random_state: int = 0,
                       da_n_runs: int = 5,          # 新增
                       pred_lag: int = 5,
                       pred_n_runs: int = 5,         # 新增
                       norm_list=None) -> None:
    """
    批量对比多个 synth_files，并写入 4 个 sheet：
    - Table2, Table3, CorrSim, CorrSim_01
    """
    rows_t2, rows_t3, rows_corr, rows_corr_01 = [], [], [], []

    for sf in synth_files:
        res = evaluate_one_synth(
            real_file=real_file,
            synth_file=sf,
            norm_max_vals=norm_max_vals,
            js_bins=js_bins,
            fdds_bins=fdds_bins,
            ac_max_lag=ac_max_lag,
            da_test_size=da_test_size,
            da_random_state=da_random_state,
            da_n_runs=da_n_runs,         # 传入
            pred_lag=pred_lag,
            pred_n_runs=pred_n_runs,       # 传入
            norm_list=norm_list
        )
        base = os.path.basename(sf)
        rows_t2 .append({"synth_file": base, **res["Table2"]})
        rows_t3 .append({"synth_file": base, **res["Table3"]})
        rows_corr.append({"synth_file": base, **res["CorrSim"]})
        rows_corr_01.append({"synth_file": base, **res["CorrSim_01"]})

    baseline_name = os.path.basename(baseline_synth_file) if baseline_synth_file else None
    df_t2 = pd.DataFrame(rows_t2)
    df_t3 = pd.DataFrame(rows_t3)
    if baseline_name:
        df_t2 = _append_delta_columns(df_t2, baseline_name)
        df_t3 = _append_delta_columns(df_t3, baseline_name)
    df_corr = pd.DataFrame(rows_corr)
    df_corr_01 = pd.DataFrame(rows_corr_01)

    with _excel_writer(out_xlsx) as writer:
        # pd.DataFrame(rows_dim).to_excel(writer, index=False, sheet_name="DimMetrics")
        df_t2.to_excel(writer, index=False, sheet_name="Table2")
        df_t3.to_excel(writer, index=False, sheet_name="Table3")
        df_corr.to_excel(writer, index=False, sheet_name="CorrSim")
        df_corr_01.to_excel(writer, index=False, sheet_name="CorrSim_01")
        
def _parse_norm_max_vals(value: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if value is None or value.lower() in {"none", "null", ""}:
        return None
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--norm-max-vals must contain exactly three comma-separated values")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIDiff/generated CSV files against real data.")
    parser.add_argument("--real-file", default=DEFAULT_REAL_FILE)
    parser.add_argument("--synth-files", nargs="+", default=[DEFAULT_MIDIFF_FILE])
    parser.add_argument("--out-xlsx", default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--baseline-synth-file", default=DEFAULT_MIDIFF_FILE)
    parser.add_argument("--norm-max-vals", type=_parse_norm_max_vals, default=None)
    parser.add_argument("--norm-list", nargs="*", default=[])
    parser.add_argument("--js-bins", type=int, default=50)
    parser.add_argument("--fdds-bins", type=int, default=50)
    parser.add_argument("--ac-max-lag", type=int, default=20)
    parser.add_argument("--da-test-size", type=float, default=0.3)
    parser.add_argument("--da-random-state", type=int, default=0)
    parser.add_argument("--da-n-runs", type=int, default=5)
    parser.add_argument("--pred-lag", type=int, default=5)
    parser.add_argument("--pred-n-runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)
    run_batch_to_excel(
        real_file=args.real_file,
        synth_files=args.synth_files,
        out_xlsx=args.out_xlsx,
        baseline_synth_file=args.baseline_synth_file,
        norm_max_vals=args.norm_max_vals,
        js_bins=args.js_bins,
        fdds_bins=args.fdds_bins,
        ac_max_lag=args.ac_max_lag,
        da_test_size=args.da_test_size,
        da_random_state=args.da_random_state,
        da_n_runs=args.da_n_runs,
        pred_lag=args.pred_lag,
        pred_n_runs=args.pred_n_runs,
        norm_list=args.norm_list,
    )


if __name__ == "__main__":
    main()
