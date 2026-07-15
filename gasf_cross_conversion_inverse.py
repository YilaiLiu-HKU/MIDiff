import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.stats import mode


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_NPZ = str(REPO_ROOT / "data" / "dataset_original_npz" / "all_users_data_with6cluster.npz")
DEFAULT_INPUT_NPZ = str(REPO_ROOT / "ckpt" / "midiff" / "ema_0.9999_048000.pt_samples_3000x256x160x1.npz")
DEFAULT_OUTPUT_NPZ = str(REPO_ROOT / "exp" / "results" / "recovered_midiff.npz")


def load_reference_stats(dataset_npz: str) -> Dict[str, np.ndarray]:
    data = np.load(dataset_npz, allow_pickle=True)
    app_traffics = data["Category_ID_Traffic (Byte)"]

    feature_maxes = np.max(app_traffics, axis=(0, 1))
    zero_mask = feature_maxes == 0

    # Keep the historical inverse behavior: zero-max columns become 0, and
    # log_feature_maxes are derived from that vector.
    feature_maxes = feature_maxes.astype(np.float64)
    feature_maxes[zero_mask] = 0
    log_feature_maxes = np.log1p(feature_maxes)

    return {
        "feature_maxes": feature_maxes,
        "log_feature_maxes": log_feature_maxes,
    }


def inverse_transform_with_max_compensation(
    app_trace: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    app_trace = app_trace.astype(np.float64)
    non_zero_mask = app_trace != 0
    if not np.any(non_zero_mask):
        return app_trace.astype(np.float32)

    app_trace = app_trace * stats["log_feature_maxes"][None, :]
    app_trace[non_zero_mask] = np.power(np.e, app_trace[non_zero_mask] - 1)
    return app_trace.astype(np.float32)


def process_single_image(args: Tuple[np.ndarray, Dict[str, np.ndarray], float]) -> Tuple[np.ndarray, np.ndarray]:
    image, stats, mean_factor = args
    time_steps = 192
    app_n = 20
    poi_n = image.shape[1] // app_n

    gasf_full = image.reshape(time_steps, app_n, poi_n)
    app_trace = np.zeros((time_steps, app_n), dtype=np.float32)
    poi_trace = np.zeros((time_steps, poi_n), dtype=np.uint8)

    for t in range(time_steps):
        blocks = gasf_full[t]
        block_medians = np.median(blocks, axis=1)
        block_p80s = np.percentile(blocks, 80, axis=1)
        combined_scores = (1 - mean_factor) * block_medians + mean_factor * block_p80s
        top_indices = np.argsort(combined_scores)[:1]

        first_pos = np.argmax(blocks, axis=0)
        vals, cnts = np.unique(first_pos, return_counts=True)
        top2_pos = vals[np.argsort(cnts)[::-1][:2]]

        if top2_pos[0] == 0:
            if len(top2_pos) == 1:
                poi_trace[t, 0] = 1
                continue
            app_pos = int(top2_pos[1])
        else:
            app_pos = int(top2_pos[0])
            if 0 in top_indices:
                app_pos = 0
            elif app_pos not in top_indices:
                continue

        poi_pos_by_block = np.argmax(blocks, axis=1)
        mode_result = mode(poi_pos_by_block, keepdims=False)
        mode_pos = int(mode_result.mode)
        block_max = float(blocks.max())

        app_trace[t, app_pos] = block_max
        poi_trace[t, mode_pos] = 1

    app_trace = inverse_transform_with_max_compensation(
        app_trace=app_trace,
        stats=stats,
    )
    return app_trace, poi_trace


def center_crop(images: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = images.shape[-3], images.shape[-2]
    if h < target_h or w < target_w:
        raise ValueError(f"Cannot crop image shape {images.shape}; target is {target_h}x{target_w}")
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return images[..., start_h:start_h + target_h, start_w:start_w + target_w, :]


def recover_traces(
    input_npz: str,
    output_npz: str,
    dataset_npz: str,
    npz_key: str,
    mean_factor: float,
    num_workers: int,
    target_h: int,
    target_w: int,
) -> None:
    stats = load_reference_stats(dataset_npz)
    images = np.load(input_npz, allow_pickle=True)[npz_key]
    images = center_crop(images, target_h=target_h, target_w=target_w)

    worker_count = cpu_count() if num_workers <= 0 else num_workers
    worker_count = max(1, min(worker_count, len(images)))
    args = [(image, stats, mean_factor) for image in images]

    if worker_count == 1:
        results = [process_single_image(item) for item in args]
    else:
        with Pool(processes=worker_count) as pool:
            results = pool.map(process_single_image, args)

    all_app_traces = []
    all_poi_traces = []
    removed = 0
    for app_trace, poi_trace in results:
        if np.any(app_trace != 0):
            all_app_traces.append(app_trace)
            all_poi_traces.append(poi_trace)
        else:
            removed += 1

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez(
        output_npz,
        app_traces=np.asarray(all_app_traces, dtype=np.float32),
        poi_traces=np.asarray(all_poi_traces, dtype=np.uint8),
        MEAN_FACTOR=mean_factor,
        stats={
            "removed_empty_samples": removed,
            "input_samples": int(len(images)),
            "output_samples": int(len(all_app_traces)),
        },
    )
    print(f"Recovered traces saved to {output_npz}")
    print(f"input_samples={len(images)} output_samples={len(all_app_traces)} removed_empty_samples={removed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover MIDiff C-GASF samples into app/POI traces.")
    parser.add_argument("--input-npz", default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output-npz", default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--dataset-npz", default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--npz-key", default="arr_0")
    parser.add_argument("--mean-factor", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0, help="0 uses all CPU cores.")
    parser.add_argument("--target-h", type=int, default=192)
    parser.add_argument("--target-w", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recover_traces(
        input_npz=args.input_npz,
        output_npz=args.output_npz,
        dataset_npz=args.dataset_npz,
        npz_key=args.npz_key,
        mean_factor=args.mean_factor,
        num_workers=args.num_workers,
        target_h=args.target_h,
        target_w=args.target_w,
    )


if __name__ == "__main__":
    main()
