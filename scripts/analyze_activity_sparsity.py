#!/usr/bin/env python3
"""Analyze per-sample activity sparsity and app-use concentration.

The script assumes traffic arrays are shaped [sample, timestep, app].
For NetDiffus, each active [sample, timestep] row has one non-zero app flow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    REPO_ROOT
    / "dataset"
    / "dataset_original_npz"
    / "all_users_data_with6cluster.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "activity_sparsity_t192"
DEFAULT_TRAFFIC_KEY = "Category_ID_Traffic (Byte)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute activity-count bins and app concentration metrics from a "
            "NetDiffus traffic NPZ."
        )
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--traffic-key", default=DEFAULT_TRAFFIC_KEY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-steps", type=int, default=192)
    parser.add_argument("--bin-width", type=int, default=8)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="A traffic value is active if it is greater than this threshold.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def normalized_entropy(values: np.ndarray) -> np.ndarray:
    """Normalized Shannon entropy along the app axis."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D values, got shape {values.shape}")

    n_categories = values.shape[1]
    totals = values.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = totals > 0
    if not np.any(valid):
        return out

    probs = values[valid] / totals[valid, None]
    term = np.zeros_like(probs)
    mask = probs > 0
    term[mask] = probs[mask] * np.log(probs[mask])
    entropy = -term.sum(axis=1)
    out[valid] = entropy / np.log(n_categories)
    return out


def gini(values: np.ndarray) -> np.ndarray:
    """Standard Gini coefficient along the app axis.

    For 20 apps, a vector with all mass on one app has Gini 0.95. The script
    also reports a max-normalized Gini so that this extreme maps to 1.0.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D values, got shape {values.shape}")
    if np.any(values < 0):
        raise ValueError("Gini is only defined here for non-negative values.")

    n_categories = values.shape[1]
    totals = values.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = totals > 0
    if not np.any(valid):
        return out

    sorted_values = np.sort(values[valid], axis=1)
    ranks = np.arange(1, n_categories + 1, dtype=np.float64)
    coeff = (
        2.0 * (sorted_values * ranks).sum(axis=1) / (n_categories * totals[valid])
        - (n_categories + 1.0) / n_categories
    )
    out[valid] = np.maximum(coeff, 0.0)
    return out


def top_k_share(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    totals = values.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = totals > 0
    if not np.any(valid):
        return out

    k = max(1, min(k, values.shape[1]))
    top_values = np.sort(values[valid], axis=1)[:, -k:]
    out[valid] = top_values.sum(axis=1) / totals[valid]
    return out


def activity_bin_table(activity_counts: np.ndarray, bin_width: int, time_steps: int) -> pd.DataFrame:
    bin_ids = activity_counts // bin_width
    max_bin_id = time_steps // bin_width
    bin_ids = np.minimum(bin_ids, max_bin_id)

    rows = []
    n_samples = len(activity_counts)
    for bin_id in range(max_bin_id + 1):
        start = bin_id * bin_width
        end = min(start + bin_width - 1, time_steps)
        label = str(start) if start == end else f"{start}-{end}"
        count = int(np.sum(bin_ids == bin_id))
        rows.append(
            {
                "bin_id": bin_id,
                "activity_start": start,
                "activity_end": end,
                "activity_bin": label,
                "sample_count": count,
                "sample_ratio": count / n_samples,
            }
        )

    df = pd.DataFrame(rows)
    df["cumulative_count"] = df["sample_count"].cumsum()
    df["cumulative_ratio"] = df["sample_ratio"].cumsum()
    return df


def summarize_group(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    values = frame[prefix].dropna()
    if values.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_p25": float(values.quantile(0.25)),
        f"{prefix}_p75": float(values.quantile(0.75)),
    }


def concentration_by_activity_bin(
    sample_df: pd.DataFrame,
    bin_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in bin_df.itertuples(index=False):
        subset = sample_df[sample_df["activity_bin_id"] == row.bin_id]
        valid = subset[subset["activity_count"] > 0]
        item = {
            "bin_id": row.bin_id,
            "activity_bin": row.activity_bin,
            "activity_start": row.activity_start,
            "activity_end": row.activity_end,
            "sample_count": int(len(subset)),
            "sample_ratio": float(len(subset) / len(sample_df)),
            "valid_metric_samples": int(len(valid)),
            "activity_count_mean": float(subset["activity_count"].mean())
            if len(subset)
            else np.nan,
            "activity_count_median": float(subset["activity_count"].median())
            if len(subset)
            else np.nan,
        }
        metric_columns = [
            "occ_entropy_norm",
            "occ_gini",
            "occ_gini_norm",
            "occ_top1_share",
            "occ_topk_share",
            "occ_active_app_count",
            "vol_entropy_norm",
            "vol_gini",
            "vol_gini_norm",
            "vol_top1_share",
            "vol_topk_share",
            "vol_active_app_count",
        ]
        for column in metric_columns:
            item.update(summarize_group(valid, column))
        rows.append(item)
    return pd.DataFrame(rows)


def dataset_level_metrics(app_occurrence: np.ndarray, app_volume: np.ndarray, top_k: int) -> pd.DataFrame:
    occurrence = app_occurrence.sum(axis=0)
    volume = app_volume.sum(axis=0)
    rows = []
    total_occurrence = occurrence.sum()
    total_volume = volume.sum()
    for app_idx, (occ, vol) in enumerate(zip(occurrence, volume)):
        rows.append(
            {
                "app_index": app_idx,
                "occurrence_count": int(occ),
                "occurrence_share": float(occ / total_occurrence)
                if total_occurrence > 0
                else np.nan,
                "traffic_volume": float(vol),
                "traffic_volume_share": float(vol / total_volume)
                if total_volume > 0
                else np.nan,
            }
        )
    df = pd.DataFrame(rows)

    global_occ = occurrence[None, :]
    global_vol = volume[None, :]
    attrs = {
        "global_occ_entropy_norm": float(normalized_entropy(global_occ)[0]),
        "global_occ_gini": float(gini(global_occ)[0]),
        "global_occ_top1_share": float(top_k_share(global_occ, 1)[0]),
        f"global_occ_top{top_k}_share": float(top_k_share(global_occ, top_k)[0]),
        "global_vol_entropy_norm": float(normalized_entropy(global_vol)[0]),
        "global_vol_gini": float(gini(global_vol)[0]),
        "global_vol_top1_share": float(top_k_share(global_vol, 1)[0]),
        f"global_vol_top{top_k}_share": float(top_k_share(global_vol, top_k)[0]),
    }
    df.attrs.update(attrs)
    return df


def save_activity_distribution_plot(
    bin_df: pd.DataFrame,
    output_path: Path,
    bin_width: int,
    time_steps: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars = ax.bar(
        bin_df["activity_bin"],
        bin_df["sample_ratio"],
        color="#3b82f6",
        edgecolor="#1d4ed8",
        linewidth=0.5,
        width=0.82,
        label="Sample ratio",
    )
    ax.set_title(f"Per-sample Activity Count Distribution (first {time_steps} timesteps)")
    ax.set_xlabel(f"Activity count per sample, binned by width={bin_width}")
    ax.set_ylabel("Sample ratio")
    ax.set_ylim(0, max(0.05, float(bin_df["sample_ratio"].max()) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=60)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    for bar, ratio in zip(bars, bin_df["sample_ratio"]):
        if ratio >= 0.03:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{ratio:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111827",
            )

    ax2 = ax.twinx()
    ax2.plot(
        bin_df["activity_bin"],
        bin_df["cumulative_ratio"],
        color="#dc2626",
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label="Cumulative ratio",
    )
    ax2.set_ylabel("Cumulative sample ratio")
    ax2.set_ylim(0, 1.03)
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_concentration_plot(group_df: pd.DataFrame, output_path: Path, top_k: int) -> None:
    x = group_df["activity_start"] + (group_df["activity_end"] - group_df["activity_start"]) / 2.0
    valid_mask = group_df["valid_metric_samples"] > 0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(
        x[valid_mask],
        group_df.loc[valid_mask, "occ_entropy_norm_median"],
        marker="o",
        label="Occurrence entropy (median)",
        color="#2563eb",
    )
    axes[0].plot(
        x[valid_mask],
        group_df.loc[valid_mask, "vol_entropy_norm_median"],
        marker="s",
        label="Volume entropy (median)",
        color="#0891b2",
    )
    axes[0].set_ylabel("Normalized entropy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        x[valid_mask],
        group_df.loc[valid_mask, "occ_gini_norm_median"],
        marker="o",
        label="Occurrence Gini, max-normalized (median)",
        color="#dc2626",
    )
    axes[1].plot(
        x[valid_mask],
        group_df.loc[valid_mask, "occ_topk_share_median"],
        marker="^",
        label=f"Occurrence top-{top_k} share (median)",
        color="#9333ea",
    )
    axes[1].set_xlabel("Activity count per sample bin midpoint")
    axes[1].set_ylabel("Concentration")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("App-use concentration by per-sample activity-count bin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_global_app_usage_plot(global_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(global_df["app_index"], global_df["occurrence_share"], color="#0f766e")
    axes[0].set_ylabel("Occurrence share")
    axes[0].set_title("Dataset-level app usage concentration")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(global_df["app_index"], global_df["traffic_volume_share"], color="#f97316")
    axes[1].set_xlabel("App index")
    axes[1].set_ylabel("Traffic volume share")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(global_df["app_index"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def safe_float(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    traffic_shape: tuple[int, int, int],
    sample_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    group_df: pd.DataFrame,
    global_df: pd.DataFrame,
) -> None:
    active_samples = sample_df[sample_df["activity_count"] > 0]
    activity_counts = sample_df["activity_count"].to_numpy()

    threshold_rows = {}
    for threshold in [8, 16, 24, 32, 48, 64]:
        threshold_rows[f"p_activity_le_{threshold}"] = float(
            np.mean(activity_counts <= threshold)
        )

    summary = {
        "npz": str(args.npz),
        "traffic_key": args.traffic_key,
        "traffic_shape_loaded": list(traffic_shape),
        "time_steps_used": args.time_steps,
        "bin_width": args.bin_width,
        "threshold": args.threshold,
        "n_samples": int(len(sample_df)),
        "n_apps": int(traffic_shape[2]),
        "zero_activity_samples": int(np.sum(activity_counts == 0)),
        "zero_activity_sample_ratio": float(np.mean(activity_counts == 0)),
        "activity_count_mean": float(np.mean(activity_counts)),
        "activity_count_median": float(np.median(activity_counts)),
        "activity_count_p25": float(np.percentile(activity_counts, 25)),
        "activity_count_p75": float(np.percentile(activity_counts, 75)),
        "activity_count_p90": float(np.percentile(activity_counts, 90)),
        "activity_count_p99": float(np.percentile(activity_counts, 99)),
        "activity_count_max": int(np.max(activity_counts)),
        "valid_metric_samples": int(len(active_samples)),
        "occ_entropy_norm_median_active": safe_float(
            active_samples["occ_entropy_norm"].median()
        ),
        "occ_gini_norm_median_active": safe_float(active_samples["occ_gini_norm"].median()),
        "occ_top1_share_median_active": safe_float(
            active_samples["occ_top1_share"].median()
        ),
        f"occ_top{args.top_k}_share_median_active": safe_float(
            active_samples["occ_topk_share"].median()
        ),
        "vol_entropy_norm_median_active": safe_float(
            active_samples["vol_entropy_norm"].median()
        ),
        "vol_gini_norm_median_active": safe_float(active_samples["vol_gini_norm"].median()),
        "global_metrics": {k: safe_float(v) for k, v in global_df.attrs.items()},
    }
    summary.update(threshold_rows)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    top_bins = bin_df.sort_values("sample_count", ascending=False).head(5)
    lines = [
        "# Activity sparsity and app concentration summary",
        "",
        f"- Source NPZ: `{args.npz}`",
        f"- Traffic key: `{args.traffic_key}`",
        f"- Loaded traffic shape: `{traffic_shape}`",
        f"- Timesteps used: first `{args.time_steps}`",
        f"- Activity count bin width: `{args.bin_width}`",
        "",
        "## Activity-count sparsity",
        "",
        f"- Samples: `{summary['n_samples']}`",
        f"- Zero-activity samples: `{summary['zero_activity_samples']}` "
        f"({summary['zero_activity_sample_ratio']:.4%})",
        f"- Activity count mean / median: `{summary['activity_count_mean']:.3f}` / "
        f"`{summary['activity_count_median']:.3f}`",
        f"- Activity count p25 / p75 / p90 / max: "
        f"`{summary['activity_count_p25']:.3f}` / "
        f"`{summary['activity_count_p75']:.3f}` / "
        f"`{summary['activity_count_p90']:.3f}` / "
        f"`{summary['activity_count_max']}`",
        f"- P(activity <= 8/16/24/32): "
        f"`{summary['p_activity_le_8']:.4%}` / "
        f"`{summary['p_activity_le_16']:.4%}` / "
        f"`{summary['p_activity_le_24']:.4%}` / "
        f"`{summary['p_activity_le_32']:.4%}`",
        "",
        "Most populated activity bins:",
        "",
    ]
    for row in top_bins.itertuples(index=False):
        lines.append(
            f"- `{row.activity_bin}`: {row.sample_count} samples ({row.sample_ratio:.4%})"
        )

    lines.extend(
        [
            "",
            "![Activity count bin distribution](activity_count_bin_distribution.png)",
            "",
            "## App concentration among active samples",
            "",
            f"- Occurrence entropy median: `{summary['occ_entropy_norm_median_active']:.4f}`",
            f"- Occurrence Gini median, max-normalized: "
            f"`{summary['occ_gini_norm_median_active']:.4f}`",
            f"- Occurrence top-1 / top-{args.top_k} share median: "
            f"`{summary['occ_top1_share_median_active']:.4f}` / "
            f"`{summary[f'occ_top{args.top_k}_share_median_active']:.4f}`",
            f"- Volume entropy median: `{summary['vol_entropy_norm_median_active']:.4f}`",
            f"- Volume Gini median, max-normalized: "
            f"`{summary['vol_gini_norm_median_active']:.4f}`",
            "",
            "![App concentration by activity bin](app_concentration_by_activity_bin.png)",
            "",
            "## Outputs",
            "",
            "- `activity_count_bins.csv`",
            "- `per_sample_app_concentration.csv`",
            "- `app_concentration_by_activity_bin.csv`",
            "- `global_app_usage.csv`",
            "- `activity_count_bin_distribution.png`",
            "- `app_concentration_by_activity_bin.png`",
            "- `global_app_usage_distribution.png`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.time_steps <= 0:
        raise ValueError("--time-steps must be positive.")
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.npz, allow_pickle=True) as data:
        if args.traffic_key not in data.files:
            raise KeyError(
                f"Traffic key {args.traffic_key!r} not found. Available keys: {data.files}"
            )
        traffic = np.asarray(data[args.traffic_key])

    if traffic.ndim != 3:
        raise ValueError(f"Expected traffic shape [N,T,C], got {traffic.shape}")
    if args.time_steps > traffic.shape[1]:
        raise ValueError(
            f"--time-steps={args.time_steps} exceeds traffic time dimension {traffic.shape[1]}"
        )

    traffic_t = traffic[:, : args.time_steps, :]
    active = traffic_t > args.threshold
    active_by_timestep = np.any(active, axis=2)
    activity_counts = active_by_timestep.sum(axis=1).astype(np.int64)

    app_occurrence = active.sum(axis=1).astype(np.float64)
    app_volume = np.where(active, traffic_t, 0.0).sum(axis=1).astype(np.float64)

    n_apps = traffic.shape[2]
    gini_max = (n_apps - 1.0) / n_apps

    occ_entropy = normalized_entropy(app_occurrence)
    occ_gini = gini(app_occurrence)
    vol_entropy = normalized_entropy(app_volume)
    vol_gini = gini(app_volume)

    bin_ids = np.minimum(activity_counts // args.bin_width, args.time_steps // args.bin_width)
    bin_labels = [
        str(int(i * args.bin_width))
        if int(i * args.bin_width) == min(int(i * args.bin_width + args.bin_width - 1), args.time_steps)
        else f"{int(i * args.bin_width)}-{min(int(i * args.bin_width + args.bin_width - 1), args.time_steps)}"
        for i in bin_ids
    ]

    sample_df = pd.DataFrame(
        {
            "sample_id": np.arange(traffic.shape[0], dtype=np.int64),
            "activity_count": activity_counts,
            "activity_ratio": activity_counts / args.time_steps,
            "activity_bin_id": bin_ids,
            "activity_bin": bin_labels,
            "occ_entropy_norm": occ_entropy,
            "occ_gini": occ_gini,
            "occ_gini_norm": occ_gini / gini_max,
            "occ_top1_share": top_k_share(app_occurrence, 1),
            "occ_topk_share": top_k_share(app_occurrence, args.top_k),
            "occ_active_app_count": (app_occurrence > 0).sum(axis=1),
            "vol_entropy_norm": vol_entropy,
            "vol_gini": vol_gini,
            "vol_gini_norm": vol_gini / gini_max,
            "vol_top1_share": top_k_share(app_volume, 1),
            "vol_topk_share": top_k_share(app_volume, args.top_k),
            "vol_active_app_count": (app_volume > 0).sum(axis=1),
        }
    )

    bin_df = activity_bin_table(activity_counts, args.bin_width, args.time_steps)
    group_df = concentration_by_activity_bin(sample_df, bin_df)
    global_df = dataset_level_metrics(app_occurrence, app_volume, args.top_k)

    bin_df.to_csv(args.output_dir / "activity_count_bins.csv", index=False)
    sample_df.to_csv(args.output_dir / "per_sample_app_concentration.csv", index=False)
    group_df.to_csv(args.output_dir / "app_concentration_by_activity_bin.csv", index=False)
    global_df.to_csv(args.output_dir / "global_app_usage.csv", index=False)

    save_activity_distribution_plot(
        bin_df,
        args.output_dir / "activity_count_bin_distribution.png",
        args.bin_width,
        args.time_steps,
    )
    save_concentration_plot(
        group_df, args.output_dir / "app_concentration_by_activity_bin.png", args.top_k
    )
    save_global_app_usage_plot(
        global_df, args.output_dir / "global_app_usage_distribution.png"
    )
    write_summary(args.output_dir, args, traffic.shape, sample_df, bin_df, group_df, global_df)

    print(f"Saved analysis outputs to: {args.output_dir}")
    print(f"Samples: {len(sample_df)}")
    print(f"Zero-activity samples: {int(np.sum(activity_counts == 0))}")
    print(
        "Activity count mean/median/max: "
        f"{activity_counts.mean():.3f} / {np.median(activity_counts):.3f} / {activity_counts.max()}"
    )
    active_samples = sample_df[sample_df["activity_count"] > 0]
    print(
        "Active-sample occurrence entropy median: "
        f"{active_samples['occ_entropy_norm'].median():.4f}"
    )
    print(
        "Active-sample occurrence Gini median (max-normalized): "
        f"{active_samples['occ_gini_norm'].median():.4f}"
    )


if __name__ == "__main__":
    main()
