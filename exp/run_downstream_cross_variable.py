import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_DATA_PATH = str(REPO_ROOT / "data" / "our.csv")
DEFAULT_MIDIFF_DATA_PATH = str(REPO_ROOT / "exp" / "results" / "MIDiff.csv")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "exp" / "results" / "cross_variable")
TRAIN_SOURCES = ("real", "MIDiff")

TARGET_SPECS = {
    "ch1": {
        "target_channel": 0,
        "input_channels": [1, 2],
        "task_type": "regression",
        "output_dim": 1,
    },
    "ch2": {
        "target_channel": 1,
        "input_channels": [0, 2],
        "task_type": "classification",
        "output_dim": 20,
    },
    "ch3": {
        "target_channel": 2,
        "input_channels": [0, 1],
        "task_type": "classification",
        "output_dim": 7,
    },
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv_as_tensor(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 192 * 3:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected 576")
    return data.reshape(data.shape[0], 192, 3)


def split_indices(num_samples: int, seed: int, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    train_size = max(1, int(num_samples * train_ratio))
    train_size = min(train_size, num_samples - 1) if num_samples > 1 else 1
    return indices[:train_size], indices[train_size:]


def choose_target_index(sequence: np.ndarray, deterministic: bool) -> int:
    candidates = np.where(sequence[96:, 0] != 0)[0]
    if len(candidates) == 0:
        return 191
    if deterministic:
        return int(96 + candidates[-1])
    return int(96 + np.random.choice(candidates))


def build_sample_dict(
    sequence: np.ndarray,
    sample_index: int,
    target_name: str,
    deterministic: bool,
    flow_mean: float,
    flow_std: float,
) -> Dict[str, Any]:
    spec = TARGET_SPECS[target_name]
    target_idx = choose_target_index(sequence, deterministic=deterministic)
    start_idx = target_idx - 96
    input_channels = spec["input_channels"]
    x = sequence[start_idx:target_idx, input_channels].copy()
    y = sequence[target_idx, spec["target_channel"]].copy()

    for input_pos, channel_idx in enumerate(input_channels):
        if channel_idx == 0:
            x[:, input_pos] = (x[:, input_pos] - flow_mean) / flow_std

    if spec["task_type"] == "regression":
        y = np.float32((y - flow_mean) / flow_std)
    else:
        y = np.int64(round(float(y)))

    return {
        "x": x.astype(np.float32),
        "y": y,
        "sample_index": sample_index,
        "target_idx": target_idx,
        "target_name": target_name,
    }


def build_dataset_dict(
    data: np.ndarray,
    target_name: str,
    deterministic: bool,
    flow_mean: float,
    flow_std: float,
) -> dict[str, np.ndarray]:
    samples = [
        build_sample_dict(
            sequence=data[i],
            sample_index=i,
            target_name=target_name,
            deterministic=deterministic,
            flow_mean=flow_mean,
            flow_std=flow_std,
        )
        for i in range(len(data))
    ]
    x = np.stack([sample["x"] for sample in samples], axis=0)
    if TARGET_SPECS[target_name]["task_type"] == "regression":
        y = np.asarray([sample["y"] for sample in samples], dtype=np.float32)
    else:
        y = np.asarray([sample["y"] for sample in samples], dtype=np.int64)
    target_indices = np.asarray([sample["target_idx"] for sample in samples], dtype=np.int64)
    sample_indices = np.asarray([sample["sample_index"] for sample in samples], dtype=np.int64)
    return {
        "x": x,
        "y": y,
        "target_indices": target_indices,
        "sample_indices": sample_indices,
    }


class CrossVariableDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        target_name: str,
        deterministic: bool,
        flow_mean: float,
        flow_std: float,
    ) -> None:
        self.target_name = target_name
        self.spec = TARGET_SPECS[target_name]
        self.data = build_dataset_dict(
            data=data,
            target_name=target_name,
            deterministic=deterministic,
            flow_mean=flow_mean,
            flow_std=flow_std,
        )

    def __len__(self) -> int:
        return len(self.data["x"])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data["x"][idx], dtype=torch.float32)
        if self.spec["task_type"] == "regression":
            y = torch.tensor(self.data["y"][idx], dtype=torch.float32)
        else:
            y = torch.tensor(self.data["y"][idx], dtype=torch.long)
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MambaBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        g1, g2 = self.gate(x).chunk(2, dim=-1)
        x = torch.sigmoid(g1) * torch.tanh(g2)
        x = self.out_proj(x)
        x = self.dropout(x)
        return residual + x


class CrossVariablePredictor(nn.Module):
    def __init__(
        self,
        target_name: str,
        model_type: str,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        nhead: int,
        ch2_num_classes: int = 20,
        ch3_num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.target_name = target_name
        self.spec = TARGET_SPECS[target_name]
        self.input_channels = self.spec["input_channels"]
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.task_type = self.spec["task_type"]

        self.traffic_encoder = nn.Linear(1, 16)
        self.ch2_embedding = nn.Embedding(ch2_num_classes, 16)
        self.ch3_embedding = nn.Embedding(ch3_num_classes, 8)

        per_channel_dims = {
            0: 16,
            1: 16,
            2: 8,
        }
        input_projection_dim = sum(per_channel_dims[ch] for ch in self.input_channels)
        self.input_projection = nn.Sequential(
            nn.Linear(input_projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if model_type == "MLP":
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(96 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        elif model_type == "LSTM":
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.encoder_dropout = nn.Dropout(dropout)
        elif model_type == "Transformer":
            self.positional_encoding = PositionalEncoding(hidden_dim, dropout, max_len=96)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.encoder_dropout = nn.Dropout(dropout)
        elif model_type == "Mamba":
            self.encoder = nn.ModuleList([MambaBlock(hidden_dim, dropout) for _ in range(num_layers)])
            self.encoder_norm = nn.LayerNorm(hidden_dim)
            self.encoder_dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.heads = nn.ModuleDict(
            {
                "ch1": nn.Linear(hidden_dim, 1),
                "ch2": nn.Linear(hidden_dim, 20),
                "ch3": nn.Linear(hidden_dim, 7),
            }
        )

    def encode_inputs(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for input_pos, channel_idx in enumerate(self.input_channels):
            current = x[:, :, input_pos]
            if channel_idx == 0:
                parts.append(self.traffic_encoder(current.unsqueeze(-1)))
            elif channel_idx == 1:
                parts.append(self.ch2_embedding(torch.clamp(current.round().long(), 0, 19)))
            elif channel_idx == 2:
                parts.append(self.ch3_embedding(torch.clamp(current.round().long(), 0, 6)))
            else:
                raise ValueError(f"Unsupported channel index: {channel_idx}")
        return self.input_projection(torch.cat(parts, dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_inputs(x)
        if self.model_type == "MLP":
            encoded = self.encoder(x)
        elif self.model_type == "LSTM":
            encoded, _ = self.encoder(x)
            encoded = self.encoder_dropout(encoded[:, -1, :])
        elif self.model_type == "Transformer":
            encoded = self.positional_encoding(x)
            encoded = self.encoder(encoded)
            encoded = self.encoder_dropout(encoded[:, -1, :])
        elif self.model_type == "Mamba":
            encoded = x
            for block in self.encoder:
                encoded = block(encoded)
            encoded = self.encoder_norm(encoded)
            encoded = self.encoder_dropout(encoded[:, -1, :])
        else:
            raise RuntimeError("Unreachable model_type branch")
        return self.heads[self.target_name](encoded)


@dataclass
class DatasetBundle:
    train_dataset: CrossVariableDataset
    val_dataset: CrossVariableDataset
    train_source_name: str
    val_source_name: str
    flow_mean: float
    flow_std: float
    real_flow_max: float
    real_val_raw_targets: np.ndarray


def prepare_datasets(
    train_source_name: str,
    target_name: str,
    seed: int,
    real_data_path: str,
    source_paths: dict[str, str],
) -> DatasetBundle:
    if train_source_name not in source_paths:
        raise ValueError(f"Unsupported train source: {train_source_name}")

    real_data = load_csv_as_tensor(real_data_path)
    real_train_idx, real_val_idx = split_indices(len(real_data), seed=seed)
    real_train = real_data[real_train_idx]
    real_val = real_data[real_val_idx]

    flow_mean = float(real_train[:, :, 0].mean())
    flow_std = float(real_train[:, :, 0].std() + 1e-8)
    real_flow_max = float(real_data[:, :, 0].max())

    train_source_data = load_csv_as_tensor(source_paths[train_source_name])
    if train_source_name == "real":
        train_raw = real_train
    else:
        source_train_idx, _ = split_indices(len(train_source_data), seed=seed)
        train_raw = train_source_data[source_train_idx]

    train_dataset = CrossVariableDataset(
        data=train_raw,
        target_name=target_name,
        deterministic=False,
        flow_mean=flow_mean,
        flow_std=flow_std,
    )
    val_dataset = CrossVariableDataset(
        data=real_val,
        target_name=target_name,
        deterministic=True,
        flow_mean=flow_mean,
        flow_std=flow_std,
    )

    val_target_channel = TARGET_SPECS[target_name]["target_channel"]
    real_val_targets = np.asarray(
        [
            real_val[i, val_dataset.data["target_indices"][i], val_target_channel]
            for i in range(len(real_val))
        ]
    )

    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_source_name=train_source_name,
        val_source_name="real",
        flow_mean=flow_mean,
        flow_std=flow_std,
        real_flow_max=real_flow_max,
        real_val_raw_targets=real_val_targets,
    )


def create_loss(target_name: str, labels: np.ndarray) -> nn.Module:
    if TARGET_SPECS[target_name]["task_type"] == "regression":
        return nn.MSELoss()

    num_classes = TARGET_SPECS[target_name]["output_dim"]
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        if task_type == "regression":
            loss = criterion(outputs.squeeze(-1), batch_y)
            batch_items = batch_y.size(0)
        else:
            loss = criterion(outputs, batch_y)
            batch_items = batch_y.size(0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_items
        total_items += batch_items
    return total_loss / max(total_items, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_name: str,
    flow_mean: float,
    flow_std: float,
    real_flow_max: float,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    task_type = TARGET_SPECS[target_name]["task_type"]
    model.eval()
    total_loss = 0.0
    total_items = 0
    preds = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            if task_type == "regression":
                batch_preds = outputs.squeeze(-1)
                loss = criterion(batch_preds, batch_y)
                preds.append(batch_preds.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
                batch_items = batch_y.size(0)
            else:
                loss = criterion(outputs, batch_y)
                batch_preds = torch.argmax(outputs, dim=-1)
                preds.append(batch_preds.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
                batch_items = batch_y.size(0)
            total_loss += loss.item() * batch_items
            total_items += batch_items

    preds_np = np.concatenate(preds, axis=0)
    targets_np = np.concatenate(targets, axis=0)
    metrics: dict[str, float]
    if task_type == "regression":
        preds_denorm = preds_np * flow_std + flow_mean
        targets_denorm = targets_np * flow_std + flow_mean
        preds_maxnorm = preds_denorm / real_flow_max
        targets_maxnorm = targets_denorm / real_flow_max
        metrics = {
            "mse_raw": float(mean_squared_error(targets_denorm, preds_denorm)),
            "mse_real_max_normalized": float(mean_squared_error(targets_maxnorm, preds_maxnorm)),
            "mae_raw": float(mean_absolute_error(targets_denorm, preds_denorm)),
            "r2_raw": float(r2_score(targets_denorm, preds_denorm)),
        }
        preds_np = preds_denorm
        targets_np = targets_denorm
    else:
        metrics = {
            "accuracy": float(accuracy_score(targets_np, preds_np)),
            "macro_f1": float(f1_score(targets_np, preds_np, average="macro", zero_division=0)),
        }
    return total_loss / max(total_items, 1), metrics, preds_np, targets_np


def save_metrics(metrics_path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(metrics_path))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_summary_csv(summary_csv_path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(summary_csv_path))
    metric_items = payload["final_val_metrics"]
    row = {
        "run_name": payload["run_name"],
        "train_source": payload["train_source"],
        "val_source": payload["val_source"],
        "target_name": payload["target_name"],
        "model_type": payload["model_type"],
        "epochs": payload["epochs"],
        "batch_size": payload["batch_size"],
        "learning_rate": payload["learning_rate"],
        "hidden_dim": payload["hidden_dim"],
        "num_layers": payload["num_layers"],
        "dropout": payload["dropout"],
        "nhead": payload["nhead"],
        "seed": payload["seed"],
        "device": payload["device"],
        "train_samples": payload["train_samples"],
        "val_samples": payload["val_samples"],
        "real_flow_max": payload.get("real_flow_max"),
        "final_val_loss": payload["final_val_loss"],
    }
    for key, value in metric_items.items():
        row[key] = value

    fieldnames = [
        "run_name",
        "train_source",
        "val_source",
        "target_name",
        "model_type",
        "epochs",
        "batch_size",
        "learning_rate",
        "hidden_dim",
        "num_layers",
        "dropout",
        "nhead",
        "seed",
        "device",
        "train_samples",
        "val_samples",
        "real_flow_max",
        "final_val_loss",
        "mse_raw",
        "mse_real_max_normalized",
        "mae_raw",
        "r2_raw",
        "accuracy",
        "macro_f1",
    ]

    file_exists = os.path.exists(summary_csv_path)
    with open(summary_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", default=DEFAULT_REAL_DATA_PATH)
    parser.add_argument("--midiff-data", default=DEFAULT_MIDIFF_DATA_PATH)
    parser.add_argument("--train-source", choices=list(TRAIN_SOURCES), required=True)
    parser.add_argument("--target-name", choices=list(TARGET_SPECS.keys()), required=True)
    parser.add_argument("--model-type", choices=["MLP", "LSTM", "Transformer", "Mamba"], default="LSTM")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def maybe_slice_dataset(dataset: CrossVariableDataset, max_samples: Optional[int]) -> CrossVariableDataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    sliced = CrossVariableDataset.__new__(CrossVariableDataset)
    sliced.target_name = dataset.target_name
    sliced.spec = dataset.spec
    sliced.data = {
        key: value[:max_samples]
        for key, value in dataset.data.items()
    }
    return sliced


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    bundle = prepare_datasets(
        train_source_name=args.train_source,
        target_name=args.target_name,
        seed=args.seed,
        real_data_path=args.real_data,
        source_paths={
            "real": args.real_data,
            "MIDiff": args.midiff_data,
        },
    )
    train_dataset = maybe_slice_dataset(bundle.train_dataset, args.max_train_samples)
    val_dataset = maybe_slice_dataset(bundle.val_dataset, args.max_val_samples)

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        pin_memory=torch.cuda.is_available(),
    )

    model = CrossVariablePredictor(
        target_name=args.target_name,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        nhead=args.nhead,
    ).to(device)

    criterion = create_loss(args.target_name, train_dataset.data["y"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run_name = f"{args.train_source}_{args.target_name}_{args.model_type}"
    run_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(run_dir)

    history = []
    best_score = float("inf") if TARGET_SPECS[args.target_name]["task_type"] == "regression" else -float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            task_type=TARGET_SPECS[args.target_name]["task_type"],
        )
        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            target_name=args.target_name,
            flow_mean=bundle.flow_mean,
            flow_std=bundle.flow_std,
            real_flow_max=bundle.real_flow_max,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
        )
        print(
            f"[{run_name}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"metrics={val_metrics}"
        )

        if TARGET_SPECS[args.target_name]["task_type"] == "regression":
            current_score = val_metrics["mse_raw"]
            is_better = current_score < best_score
        else:
            current_score = val_metrics["accuracy"]
            is_better = current_score > best_score

        if is_better:
            best_score = current_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_metrics, preds, targets = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        target_name=args.target_name,
        flow_mean=bundle.flow_mean,
        flow_std=bundle.flow_std,
        real_flow_max=bundle.real_flow_max,
    )

    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
    np.save(os.path.join(run_dir, "val_preds.npy"), preds)
    np.save(os.path.join(run_dir, "val_targets.npy"), targets)

    payload = {
        "run_name": run_name,
        "train_source": args.train_source,
        "val_source": bundle.val_source_name,
        "target_name": args.target_name,
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "nhead": args.nhead,
        "seed": args.seed,
        "device": str(device),
        "real_data": args.real_data,
        "midiff_data": args.midiff_data,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "flow_mean": bundle.flow_mean,
        "flow_std": bundle.flow_std,
        "real_flow_max": bundle.real_flow_max,
        "final_val_loss": final_val_loss,
        "final_val_metrics": final_metrics,
        "history": history,
    }
    save_metrics(os.path.join(run_dir, "metrics.json"), payload)
    append_summary_csv(os.path.join(args.output_dir, "summary.csv"), payload)


if __name__ == "__main__":
    main()
