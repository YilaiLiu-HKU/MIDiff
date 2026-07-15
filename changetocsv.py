import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_NPZ = str(REPO_ROOT / "exp" / "results" / "recovered_midiff.npz")
DEFAULT_OUTPUT_CSV = str(REPO_ROOT / "exp" / "results" / "MIDiff.csv")


def traces_to_eval_array(app_traces: np.ndarray, poi_traces: np.ndarray) -> np.ndarray:
    if app_traces.ndim != 3:
        raise ValueError(f"app_traces must be [N,T,C], got {app_traces.shape}")
    if poi_traces.ndim != 3:
        raise ValueError(f"poi_traces must be [N,T,P], got {poi_traces.shape}")
    if app_traces.shape[:2] != poi_traces.shape[:2]:
        raise ValueError(f"Shape mismatch: app {app_traces.shape}, poi {poi_traces.shape}")

    app_flow = app_traces.max(axis=2)
    app_category = app_traces.argmax(axis=2).astype(np.float64)
    poi_category = poi_traces.argmax(axis=2).astype(np.float64)

    background_poi = poi_category == 0
    app_flow = np.where(background_poi, 0.0, app_flow).astype(np.float64)
    app_category = np.where((app_flow == 0) | background_poi, 0.0, app_category)
    poi_category = np.where(background_poi, 0.0, poi_category)

    return np.stack([app_flow, app_category, poi_category], axis=2).reshape(app_traces.shape[0], -1)


def convert_npz_to_eval_csv(input_npz: str, output_csv: str) -> None:
    data = np.load(input_npz, allow_pickle=True)
    app_traces = data["app_traces"]
    poi_traces = data["poi_traces"]
    eval_array = traces_to_eval_array(app_traces, poi_traces)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(eval_array).to_csv(output_csv, index=False, header=False)
    print(f"Eval CSV saved to {output_csv}")
    print(f"shape={eval_array.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert recovered MIDiff traces to the eval CSV format.")
    parser.add_argument("--input-npz", default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_npz_to_eval_csv(args.input_npz, args.output_csv)


if __name__ == "__main__":
    main()
