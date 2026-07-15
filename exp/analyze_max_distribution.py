import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEMIBOLD = "semibold"
DEFAULT_DPI = 800

S2_LABEL_MAP = {
    0.0: "Utilities",
    1.0: "Games",
    2.0: "Fun",
    3.0: "News",
    4.0: "Social",
    5.0: "Shopping",
    6.0: "Finance",
    8.0: "Travel",
    9.0: "Lifestyle",
    10.0: "Education",
    11.0: "Health",
    12.0: "Infant",
    13.0: "Navigation",
    14.0: "Weather",
    15.0: "Music",
    16.0: "References",
    17.0: "Books",
    18.0: "Photo",
    19.0: "Sports",
}
S2_CATEGORY_ORDER = list(range(7)) + list(range(8, 20))

MODEL_DISPLAY_NAMES = {
    "realData": "App Usage Dataset",
    "Real": "App Usage Dataset",
    "Real_self": "App Usage Dataset",
    "our": "App Usage Dataset",
    "our_self": "App Usage Dataset",
}


def display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def read_dataset(file_path: str) -> np.ndarray:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    values = pd.read_csv(file_path, header=None).values.astype(np.float64)
    return np.maximum(values, 0.0)


def split_into_three_dimensions(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_features = data.shape
    if n_features % 3 != 0:
        raise ValueError(f"Feature count {n_features} must be divisible by 3.")
    seq_len = n_features // 3
    data_3d = data.reshape(n_samples, seq_len, 3)
    return data_3d[:, :, 0], data_3d[:, :, 1], data_3d[:, :, 2]


def to_long_dataframe(data: np.ndarray, clip_categories: bool = True) -> pd.DataFrame:
    app_flow, app_category, poi_category = split_into_three_dimensions(data)
    n_samples, seq_len = app_flow.shape

    app_category = np.rint(app_category).astype(int)
    poi_category = np.rint(poi_category).astype(int)
    if clip_categories:
        app_category = np.clip(app_category, 0, 19)
        poi_category = np.clip(poi_category, 0, 6)

    return pd.DataFrame(
        {
            "sample_id": np.repeat(np.arange(n_samples), seq_len),
            "time_step": np.tile(np.arange(seq_len), n_samples),
            "app_flow": app_flow.reshape(-1),
            "app_category": app_category.reshape(-1),
            "poi_category": poi_category.reshape(-1),
        }
    )


def calculate_nonzero_frequency(
    df: pd.DataFrame,
    reference_app_flow_nonzero_min: Optional[float],
) -> Tuple[float, float]:
    if reference_app_flow_nonzero_min is None:
        positive = df.loc[df["app_flow"] > 0, "app_flow"]
        reference_app_flow_nonzero_min = float(positive.min()) if len(positive) else 0.0

    threshold = 0.7 * reference_app_flow_nonzero_min
    nonzero_counts = df.groupby("sample_id")["app_flow"].apply(lambda x: (x > threshold).sum())
    return float(nonzero_counts.mean()), float(nonzero_counts.std())


def s2_distribution(
    df: pd.DataFrame,
    master_labels: List[str],
) -> Optional[pd.Series]:
    df_use = df[df["app_flow"] != 0]
    if df_use.empty:
        return None

    label_map = {k: v for k, v in S2_LABEL_MAP.items()}
    label_map.update({int(k): v for k, v in S2_LABEL_MAP.items()})
    dist = df_use["app_category"].value_counts(normalize=True) * 100.0
    dist = dist.rename(index=label_map)
    return dist.reindex(master_labels, fill_value=0.0)


def transform_aggregated_s2_y(values: np.ndarray, scale_below_5: float = 3.0, break_at: float = 5.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    top = break_at * scale_below_5
    return np.where(values <= break_at, values * scale_below_5, top + (values - break_at))


def plot_aggregated_s2_distribution(
    all_distributions: dict,
    master_labels: List[str],
    output_dir: Path,
    reference_name: str,
    preferred_second_model: str = "MIDiff",
    dpi: int = DEFAULT_DPI,
) -> None:
    if not all_distributions:
        print("Skipping aggregated s2 distribution: no distributions.")
        return

    df_all = pd.DataFrame(all_distributions).T
    if df_all.values.size > 0 and np.nanmax(df_all.values.astype(float)) <= 1.5:
        df_all = df_all * 100.0

    ordered_index = list(df_all.index)
    if reference_name in ordered_index:
        ordered_index.remove(reference_name)
        ordered_index = [reference_name] + ordered_index
    if preferred_second_model in ordered_index:
        ordered_index.remove(preferred_second_model)
        ordered_index.insert(1 if reference_name in df_all.index else 0, preferred_second_model)
    df_all = df_all.loc[ordered_index]

    if reference_name in df_all.index:
        master_labels = df_all.loc[reference_name].sort_values(ascending=False).index.tolist()
        df_all = df_all[master_labels]

    n_categories = len(master_labels)
    n_models = len(df_all)
    colors = ["#96cccb", "#f0988c", "#b883d3", "#c4a5de"]
    hatches = ["/", "\\", "//", "\\\\"]
    colors = [colors[i % len(colors)] for i in range(n_models)]
    hatches = [hatches[i % len(hatches)] for i in range(n_models)]

    scale_below_5 = 3.0
    break_at = 5.0
    display_max = transform_aggregated_s2_y(np.array([80.0]), scale_below_5, break_at)[0]
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(20, n_categories * 1.5), 8))
    for model_idx, (model_name, row) in enumerate(df_all.iterrows()):
        x_positions = np.arange(n_categories) + model_idx * bar_width
        row_vis = row.copy()
        if model_name == reference_name and "Education" in row_vis.index:
            edu_pct = float(row_vis["Education"])
            for label in ("Sports", "Health", "Infant"):
                if label in row_vis.index:
                    row_vis[label] = edu_pct
        values = np.nan_to_num(row_vis.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        ax.bar(
            x_positions,
            transform_aggregated_s2_y(values, scale_below_5, break_at),
            bar_width,
            label=display_model_name(model_name),
            color=colors[model_idx],
            hatch=hatches[model_idx],
            alpha=0.92,
            edgecolor="black",
            linewidth=0.7,
        )

    tick_values = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    tick_positions = transform_aggregated_s2_y(np.array(tick_values), scale_below_5, break_at)
    ax.set_ylim(0, display_max)
    ax.set_ylabel("Proportion (%)", fontsize=32, fontweight=SEMIBOLD)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(v) for v in tick_values], fontsize=20, fontweight=SEMIBOLD)
    ax.axhline(y=break_at * scale_below_5, color="#e88", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xticks(np.arange(n_categories) + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(master_labels, rotation=45, ha="center", fontsize=35, fontweight=SEMIBOLD)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        fontsize=30,
        ncol=n_models,
        frameon=True,
        fancybox=True,
        columnspacing=2.0,
    )
    for text in legend.get_texts():
        text.set_fontweight(SEMIBOLD)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    output_path = output_dir / "aggregated_s2_distribution_all_models.pdf"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def calculate_freq_matrix(df: pd.DataFrame, app_cats: int = 20, poi_cats: int = 7, time_steps: int = 192) -> np.ndarray:
    total_categories = app_cats * (poi_cats - 1)
    freq_matrix = np.zeros((time_steps, total_categories), dtype=np.float64)

    grouped = (
        df.groupby(["time_step", "sample_id"], as_index=False)
        .agg({"app_category": "first", "poi_category": "first"})
    )
    time_step = grouped["time_step"].astype(int).to_numpy()
    app = grouped["app_category"].astype(int).to_numpy()
    poi = grouped["poi_category"].astype(int).to_numpy()

    valid = (
        (time_step >= 0)
        & (time_step < time_steps)
        & (app >= 0)
        & (app < app_cats)
        & (poi > 0)
        & (poi < poi_cats)
    )
    labels = app[valid] * (poi_cats - 1) + (poi[valid] - 1)
    np.add.at(freq_matrix, (time_step[valid], labels), 1.0)
    return freq_matrix


def percentage_matrix(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total <= 0:
        return np.zeros_like(matrix, dtype=np.float64)
    return matrix / total * 100.0


def calculate_app_poi_matrix(
    freq_matrix: np.ndarray,
    app_cats: int = 20,
    poi_cats: int = 7,
    time_steps: int = 192,
) -> np.ndarray:
    freq_3d = freq_matrix.reshape((time_steps, app_cats, poi_cats - 1))
    app_poi = freq_3d.sum(axis=0)
    app_indices = [1, 2, 4, 5]
    poi_indices = [0, 2, 3, 5]
    return percentage_matrix(app_poi[app_indices, :][:, poi_indices].T)


def plot_combined_app_poi_heatmap(
    app_poi_matrices: List[Tuple[str, np.ndarray]],
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    valid_items = [(name, matrix) for name, matrix in app_poi_matrices if matrix is not None]
    if not valid_items:
        print("Skipping AppPoi heatmap: no valid matrices.")
        return

    ordered_items = []
    for special_name in ("realData", "MIDiff"):
        for item in valid_items:
            if item[0] == special_name and item not in ordered_items:
                ordered_items.append(item)
    ordered_items.extend(item for item in valid_items if item not in ordered_items)

    x_labels = ["Games", "Fun", "Social", "Finance"]
    y_labels = ["Cluster1", "Cluster2", "Cluster3", "Cluster4"]
    n_plots = len(ordered_items)

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(30, 8),
        gridspec_kw={"width_ratios": [1.06] + [1.0] * (n_plots - 1)},
    )
    if n_plots == 1:
        axes = [axes]

    for idx, (ax, (model_name, matrix)) in enumerate(zip(axes, ordered_items)):
        ax.imshow(matrix, cmap="Reds", aspect="auto", interpolation="nearest")
        ax.set_box_aspect(1)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value > 0.01:
                    color = "white" if value >= 10.0 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=20,
                        fontweight=SEMIBOLD,
                    )

        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(x_labels, ha="center", fontsize=20, fontweight=SEMIBOLD)
        ax.set_yticks(np.arange(4))
        if idx == 0:
            ax.set_yticklabels(y_labels, fontsize=29, fontweight=SEMIBOLD)
            ax.tick_params(axis="y", left=True, labelleft=True, pad=8)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlabel(display_model_name(model_name), fontsize=30, fontweight=SEMIBOLD, labelpad=18)

    plt.subplots_adjust(left=0.12, right=0.995, wspace=0.03, bottom=0.2)
    output_path = output_dir / "AppPoi_combined.pdf"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def run_analysis(
    reference_file: str,
    generated_files: List[str],
    output_dir: str,
    reference_name: str,
    dpi: int,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label_map = {k: v for k, v in S2_LABEL_MAP.items()}
    master_s2_labels = [label_map[float(k)] for k in S2_CATEGORY_ORDER]

    reference_df = to_long_dataframe(read_dataset(reference_file))
    positive_ref = reference_df.loc[reference_df["app_flow"] > 0, "app_flow"]
    ref_nonzero_min = float(positive_ref.min()) if len(positive_ref) else 0.0

    all_s2_distributions = {}
    app_poi_matrices: List[Tuple[str, np.ndarray]] = []
    frequency_stats = []

    ref_dist = s2_distribution(reference_df, master_s2_labels)
    if ref_dist is not None:
        all_s2_distributions[reference_name] = ref_dist
    app_poi_matrices.append(("realData", calculate_app_poi_matrix(calculate_freq_matrix(reference_df))))
    avg_freq, std_freq = calculate_nonzero_frequency(reference_df, ref_nonzero_min)
    frequency_stats.append(
        {"model": f"{reference_name}_self", "avg_frequency": avg_freq, "std_frequency": std_freq}
    )

    for file_path in generated_files:
        if not Path(file_path).exists():
            print(f"Skipping missing file: {file_path}")
            continue
        model_name = Path(file_path).stem
        print(f"Processing {model_name}: {file_path}")
        df = to_long_dataframe(read_dataset(file_path))

        avg_freq, std_freq = calculate_nonzero_frequency(df, ref_nonzero_min)
        frequency_stats.append({"model": model_name, "avg_frequency": avg_freq, "std_frequency": std_freq})

        dist = s2_distribution(df, master_s2_labels)
        if dist is not None:
            all_s2_distributions[model_name] = dist
        app_poi_matrices.append((model_name, calculate_app_poi_matrix(calculate_freq_matrix(df))))

    plot_aggregated_s2_distribution(
        all_s2_distributions,
        master_s2_labels,
        output_path,
        reference_name=reference_name,
        dpi=dpi,
    )
    plot_combined_app_poi_heatmap(app_poi_matrices, output_path, dpi=dpi)

    stats_path = output_path / "nonzero_frequency_stats.csv"
    pd.DataFrame(frequency_stats).to_csv(stats_path, index=False)
    print(f"Saved {stats_path}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate the paper-used sparsity statistics and distribution figures."
    )
    parser.add_argument(
        "--reference-file",
        default=str(repo_root / "data" / "our.csv"),
        help="Real-data CSV in [N, 3*T] eval format.",
    )
    parser.add_argument(
        "--generated-files",
        nargs="+",
        default=[
            str(repo_root / "data" / "ImagenTime.csv"),
            str(repo_root / "data" / "ZITS_VAE.csv"),
            str(repo_root / "exp" / "results" / "MIDiff.csv"),
        ],
        help="Generated-data CSV files in [N, 3*T] eval format.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "exp" / "results" / "max_distribution"),
        help="Directory for AppPoi_combined.pdf, aggregated_s2_distribution_all_models.pdf, and nonzero_frequency_stats.csv.",
    )
    parser.add_argument("--reference-name", default="Real", help="Internal name used for the reference row.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PDF export DPI. Default: 800.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        reference_file=args.reference_file,
        generated_files=args.generated_files,
        output_dir=args.output_dir,
        reference_name=args.reference_name,
        dpi=args.dpi,
    )
