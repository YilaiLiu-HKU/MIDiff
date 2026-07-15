#!/usr/bin/env python3
"""Isolated efficiency benchmark runner for MIDiff poster experiments.

The runner writes every new artifact under:

    ./efficiency_results/<run_id>/

It intentionally avoids calling ZITS' original sample_model because that path
writes back into the training output directories.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import platform
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MIDIFF_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = MIDIFF_ROOT / "efficiency_results"
EXTERNAL_ROOT = MIDIFF_ROOT / "external"
ZITS_ROOT = EXTERNAL_ROOT / "Zero-Inflated-Time-Series-Generation"
IMAGENTIME_ROOT = EXTERNAL_ROOT / "ImagenTime"
MIDiff_SAMPLE_SCRIPT = MIDIFF_ROOT / "sample_midiff.py"

DEFAULT_PYTHON = Path(os.environ.get("PYTHON", sys.executable))
DEFAULT_DATA_PATH = MIDIFF_ROOT / "data" / "our.csv"
DEFAULT_QUALITY_PATH = MIDIFF_ROOT / "exp" / "results" / "eval_midiff_real.xlsx"
FALLBACK_QUALITY_PATH = MIDIFF_ROOT / "eval_results.xlsx"

ZITS_GAN_OUTPUT = ZITS_ROOT / "output_our_gan"
ZITS_VAE_OUTPUT = ZITS_ROOT / "output_our_vae"
ZITS_GAN_CHECKPOINT = ZITS_GAN_OUTPUT / "our_gan_generator.pth"
ZITS_VAE_CHECKPOINT = ZITS_VAE_OUTPUT / "our_vae_model.pth"
ZITS_GAN_PREPROCESSOR = ZITS_GAN_OUTPUT / "gan_preprocessor.json"
ZITS_VAE_PREPROCESSOR = ZITS_VAE_OUTPUT / "vae_preprocessor.json"

DEFAULT_MIDIFF_CHECKPOINT = MIDIFF_ROOT / "ckpt" / "midiff" / "ema_0.9999_048000.pt"
DEFAULT_MIDIFF_REFERENCE_LOG = MIDIFF_ROOT / "ckpt" / "cgasf_ablation" / "log.txt"
IMAGENTIME_CONFIG = IMAGENTIME_ROOT / "configs" / "unconditional" / "energy.yaml"

CANONICAL_MODELS = {
    "zits-gan": "ZITS-GAN",
    "zits_gan": "ZITS-GAN",
    "zitsgan": "ZITS-GAN",
    "gan": "ZITS-GAN",
    "zits-vae": "ZITS-VAE",
    "zits_vae": "ZITS-VAE",
    "zitsvae": "ZITS-VAE",
    "vae": "ZITS-VAE",
    "midiff": "MIDiff",
    "mi-diff": "MIDiff",
    "imagenTime": "ImagenTime",
    "imagentime": "ImagenTime",
    "imagen-time": "ImagenTime",
}

QUALITY_NAME_CANDIDATES = {
    "ZITS-GAN": ["ZITS_GAN.csv", "ZITS-GAN.csv", "zits-gan.csv", "zits_gan.csv"],
    "ZITS-VAE": ["ZITS_VAE.csv", "ZITS-VAE.csv", "zits-vae.csv", "zits_vae.csv"],
    "MIDiff": ["MIDiff.csv", "MiDiff.csv", "mi_diff.csv"],
    "ImagenTime": ["ImagenTime.csv", "imagenTime.csv", "imagentime.csv"],
}

QUALITY_FIELDS = ("MDD", "ACD", "DTW", "ED", "VDS", "FDDS")
PROFILES = ("latency_b1", "throughput_fixed", "throughput_max")


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def local_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, default=json_default)
        f.write("\n")


def shell_join(args: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(x)) for x in args)


def parse_csv_list(value: str, *, lower: bool = True) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [item.lower() for item in items] if lower else items


def canonicalize_models(value: str) -> List[str]:
    out: List[str] = []
    for item in parse_csv_list(value):
        if item == "all":
            return ["ZITS-GAN", "ZITS-VAE", "MIDiff", "ImagenTime"]
        if item not in CANONICAL_MODELS:
            raise ValueError(f"unknown model {item!r}; choose all,zits-gan,zits-vae,midiff,imagentime")
        canonical = CANONICAL_MODELS[item]
        if canonical not in out:
            out.append(canonical)
    return out


def parse_profiles(value: str) -> List[str]:
    out: List[str] = []
    for item in parse_csv_list(value):
        if item == "all":
            return list(PROFILES)
        if item not in PROFILES:
            raise ValueError(f"unknown profile {item!r}; choose {','.join(PROFILES)}")
        if item not in out:
            out.append(item)
    return out


def make_run_dir(results_root: Path, run_id: Optional[str], fail_if_exists: bool) -> Tuple[str, Path]:
    results_root.mkdir(parents=True, exist_ok=True)
    base_id = run_id or local_run_id()
    candidate = results_root / base_id
    if candidate.exists():
        if fail_if_exists:
            raise FileExistsError(f"run directory already exists: {candidate}")
        for idx in range(1, 1000):
            next_id = f"{base_id}_{idx:02d}"
            candidate = results_root / next_id
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return next_id, candidate
        raise FileExistsError(f"could not allocate unique run_id under {results_root} for {base_id}")
    candidate.mkdir(parents=True, exist_ok=False)
    return base_id, candidate


def ensure_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"refusing to write outside run directory: {resolved}")
    return resolved


def run_subprocess(args: Sequence[str], cwd: Optional[Path] = None, timeout: int = 30) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "args": list(args),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": time.perf_counter() - started,
        }
    except FileNotFoundError as exc:
        return {"args": list(args), "returncode": 127, "stdout": "", "stderr": str(exc), "elapsed_sec": time.perf_counter() - started}
    except subprocess.TimeoutExpired as exc:
        return {
            "args": list(args),
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout}s",
            "elapsed_sec": time.perf_counter() - started,
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path, *, hash_file: bool = True) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    stat = path.stat()
    info.update(
        {
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    )
    if hash_file:
        info["sha256"] = sha256_file(path)
    return info


def git_info(repo: Path) -> Dict[str, Any]:
    if not (repo / ".git").exists():
        return {"path": str(repo), "is_git_repo": False}
    head = run_subprocess(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=10)
    branch = run_subprocess(["git", "-C", str(repo), "branch", "--show-current"], timeout=10)
    status = run_subprocess(["git", "-C", str(repo), "status", "--short", "--branch"], timeout=20)
    return {
        "path": str(repo),
        "is_git_repo": True,
        "head": head["stdout"].strip() if head["returncode"] == 0 else None,
        "branch": branch["stdout"].strip() if branch["returncode"] == 0 else None,
        "status_short_branch": status["stdout"],
        "status_returncode": status["returncode"],
        "status_stderr": status["stderr"],
    }


def collect_runtime_env(args: argparse.Namespace, models: Sequence[str]) -> Dict[str, Any]:
    torch_info: Dict[str, Any]
    try:
        import torch  # type: ignore

        torch_info = {
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [],
        }
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                torch_info["devices"].append(
                    {
                        "visible_index": idx,
                        "name": props.name,
                        "total_memory_mb": props.total_memory / (1024 * 1024),
                        "capability": f"{props.major}.{props.minor}",
                    }
                )
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "timestamp": utc_timestamp(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "configured_python": str(args.python),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "command_line": shell_join(sys.argv),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_gpu": args.gpu,
        "models": list(models),
        "torch": torch_info,
        "git": {
            "MIDiff": git_info(MIDIFF_ROOT),
            "ZITS": git_info(ZITS_ROOT),
            "ImagenTime": git_info(IMAGENTIME_ROOT),
        },
    }


def parse_nvidia_csv(stdout: str, fields: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or "No running processes found" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(fields):
            continue
        rows.append({field: parts[idx] for idx, field in enumerate(fields)})
    return rows


def to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "[not supported]", "not supported"}:
        return None
    text = text.replace("MiB", "").replace("%", "").strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def capture_nvidia_smi(run_dir: Path, label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"label": label, "timestamp": utc_timestamp(), "available": False}
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        out["error"] = "nvidia-smi not found"
        write_json(run_dir / f"nvidia_smi_{label}.json", out)
        return out

    text = run_subprocess([nvidia_smi], timeout=20)
    (run_dir / f"nvidia_smi_{label}.txt").write_text(text["stdout"] + text["stderr"], encoding="utf-8")

    gpu_fields = ["index", "uuid", "name", "memory.used", "utilization.gpu", "driver_version"]
    gpu_query = run_subprocess(
        [
            nvidia_smi,
            f"--query-gpu={','.join(gpu_fields)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    app_fields = ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]
    app_query = run_subprocess(
        [
            nvidia_smi,
            f"--query-compute-apps={','.join(app_fields)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    out.update(
        {
            "available": True,
            "nvidia_smi_path": nvidia_smi,
            "text_returncode": text["returncode"],
            "gpu_query_returncode": gpu_query["returncode"],
            "gpu_query_stderr": gpu_query["stderr"],
            "compute_query_returncode": app_query["returncode"],
            "compute_query_stderr": app_query["stderr"],
            "gpus": parse_nvidia_csv(gpu_query["stdout"], gpu_fields) if gpu_query["returncode"] == 0 else [],
            "compute_apps": parse_nvidia_csv(app_query["stdout"], app_fields) if app_query["returncode"] == 0 else [],
        }
    )
    write_json(run_dir / f"nvidia_smi_{label}.json", out)
    return out


def check_gpu_idle(
    snapshot: Mapping[str, Any],
    gpu_index: Optional[int],
    max_memory_mb: int,
    max_utilization_pct: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "requested_gpu": gpu_index,
        "max_memory_mb": max_memory_mb,
        "max_utilization_pct": max_utilization_pct,
        "ok": True,
        "reasons": [],
        "status": "not_requested" if gpu_index is None else "checked",
    }
    if gpu_index is None:
        return result
    if not snapshot.get("available"):
        result["ok"] = False
        result["reasons"].append(snapshot.get("error", "nvidia-smi unavailable"))
        return result

    gpu_rows = list(snapshot.get("gpus") or [])
    gpu = next((row for row in gpu_rows if str(row.get("index")) == str(gpu_index)), None)
    if gpu is None:
        result["ok"] = False
        result["reasons"].append(f"GPU index {gpu_index} not found in nvidia-smi")
        result["available_gpus"] = gpu_rows
        return result

    uuid = gpu.get("uuid")
    memory_used = to_int_or_none(gpu.get("memory.used"))
    utilization = to_int_or_none(gpu.get("utilization.gpu"))
    apps = [app for app in (snapshot.get("compute_apps") or []) if app.get("gpu_uuid") == uuid]
    result.update({"gpu": gpu, "compute_apps": apps, "memory_used_mb": memory_used, "utilization_pct": utilization})
    if apps:
        result["ok"] = False
        result["reasons"].append(f"GPU {gpu_index} has compute PIDs: " + ",".join(str(app.get("pid")) for app in apps))
    if memory_used is not None and memory_used > max_memory_mb:
        result["ok"] = False
        result["reasons"].append(f"GPU {gpu_index} memory.used {memory_used} MiB exceeds threshold {max_memory_mb} MiB")
    if utilization is not None and utilization > max_utilization_pct:
        result["ok"] = False
        result["reasons"].append(
            f"GPU {gpu_index} utilization {utilization}% exceeds threshold {max_utilization_pct}%"
        )
    return result


def normalize_quality_name(value: Any) -> str:
    name = str(value).strip().lower()
    if name.endswith(".csv"):
        name = name[:-4]
    return name.replace("-", "_").replace(" ", "").replace("/", "_")


def read_quality_metrics(path: Path, models: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "source": str(path),
        "exists": path.exists(),
        "metrics_by_model": {model: {} for model in models},
        "matches": {},
        "status": "missing" if not path.exists() else "ok",
    }
    if not path.exists():
        return result
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        result["status"] = "unavailable_dependency"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        excel = pd.ExcelFile(path)
        result["sheets"] = excel.sheet_names
        wanted_by_model = {
            model: {normalize_quality_name(name) for name in QUALITY_NAME_CANDIDATES.get(model, [model])}
            for model in models
        }
        for sheet in excel.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if "synth_file" not in df.columns:
                continue
            for _, row in df.iterrows():
                row_name = normalize_quality_name(row.get("synth_file"))
                for model, wanted in wanted_by_model.items():
                    if row_name not in wanted:
                        continue
                    result["matches"].setdefault(model, []).append({"sheet": sheet, "synth_file": row.get("synth_file")})
                    for field in QUALITY_FIELDS:
                        if field in df.columns and field not in result["metrics_by_model"][model]:
                            value = row.get(field)
                            if value == value:
                                try:
                                    result["metrics_by_model"][model][field] = float(value)
                                except Exception:
                                    result["metrics_by_model"][model][field] = value
        return result
    except Exception as exc:
        result["status"] = "read_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def choose_quality_path(path_arg: Optional[Path]) -> Path:
    if path_arg:
        return path_arg
    if DEFAULT_QUALITY_PATH.exists():
        return DEFAULT_QUALITY_PATH
    return FALLBACK_QUALITY_PATH


def checkpoint_for_model(model: str, args: argparse.Namespace) -> Optional[Path]:
    if model == "ZITS-GAN":
        return args.zits_gan_checkpoint
    if model == "ZITS-VAE":
        return args.zits_vae_checkpoint
    if model == "MIDiff":
        return args.midiff_checkpoint
    if model == "ImagenTime":
        return args.imagentime_checkpoint or infer_imagentime_checkpoint()
    return None


def infer_imagentime_checkpoint() -> Path:
    name = (
        "conditional-"
        "bs=8-"
        "-lr=0.0001-"
        "ch_mult=[1, 2, 4, 4]-"
        "attn_res=[36, 18]-"
        "unet_ch=128"
        "-delay=136-192"
    )
    return IMAGENTIME_ROOT / "logs" / "energy" / name


def preprocessor_for_zits_model(model: str, args: argparse.Namespace) -> Path:
    return args.zits_gan_preprocessor if model == "ZITS-GAN" else args.zits_vae_preprocessor


def base_raw_record(
    model: str,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    env_info: Mapping[str, Any],
    quality: Mapping[str, Any],
    checkpoint_meta: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    torch_info = env_info.get("torch", {})
    device = "cuda" if isinstance(torch_info, Mapping) and torch_info.get("cuda_available") else "cpu"
    quality_metrics = (quality.get("metrics_by_model", {}) if isinstance(quality, Mapping) else {}).get(model, {})
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model": model,
        "scope": "sample_only",
        "timestamp": utc_timestamp(),
        "action": args.action,
        "command": shell_join(sys.argv),
        "python": str(args.python),
        "python_executable": sys.executable,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": args.gpu,
        "data_path": str(args.data_path),
        "checkpoint": dict(checkpoint_meta or {}),
        "checkpoint_path": str((checkpoint_meta or {}).get("path", "")),
        "checkpoint_size_mb": (checkpoint_meta or {}).get("size_mb"),
        "checkpoint_sha256": (checkpoint_meta or {}).get("sha256"),
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "warmup_runs": args.warmup_runs,
        "measured_repeats": args.measured_repeats,
        "precision": "fp32",
        "flops_per_forward": None,
        "macs_per_forward": None,
        "estimated_flops_per_sample": None,
        "estimated_total_flops": None,
        "base_model_name": "N/A",
        "param_overhead_pct": None,
        "flops_overhead_pct": None,
        "latency_overhead_pct": None,
        "memory_overhead_pct": None,
        "training_time_sec": None,
        "gpu_hours": None,
        "training_epochs_or_steps": None,
        "best_checkpoint_epoch_or_step": None,
        "quality_source": quality.get("source") if isinstance(quality, Mapping) else None,
        "quality_metrics": quality_metrics,
    }


def write_command_files(run_dir: Path, args: argparse.Namespace, run_id: str, models: Sequence[str], profiles: Sequence[str]) -> None:
    command_payload = {
        "run_id": run_id,
        "argv": sys.argv,
        "command_line": shell_join(sys.argv),
        "models": list(models),
        "profiles": list(profiles),
        "action": args.action,
    }
    write_json(run_dir / "command.json", command_payload)
    (run_dir / "command.txt").write_text(command_payload["command_line"] + "\n", encoding="utf-8")


def manifest_record_for_checkpoint(
    model: str,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    env_info: Mapping[str, Any],
    quality: Mapping[str, Any],
    checkpoint: Optional[Path],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    checkpoint_meta = file_info(checkpoint, hash_file=not args.skip_checkpoint_hash) if checkpoint else None
    record = base_raw_record(model, args, run_id, run_dir, env_info, quality, checkpoint_meta)
    record.update(
        {
            "benchmark_profile": "manifest",
            "batch_size": None,
            "status": "ready_manifest" if checkpoint_meta and checkpoint_meta.get("exists") else "missing_checkpoint",
            "notes": "",
        }
    )
    if extra:
        record.update(extra)
    return record


def build_midiff_command(args: argparse.Namespace, save_dir: Path) -> List[str]:
    command = [
        str(args.python),
        str(MIDiff_SAMPLE_SCRIPT),
        "--model_path",
        str(args.midiff_checkpoint),
        "--save_dir",
        str(save_dir),
        "--num_samples",
        str(args.num_samples),
        "--batch_size",
        str(args.midiff_batch_size or args.fixed_batch_size),
        "--image_size",
        str(args.midiff_image_size),
        "--image_size_second",
        str(args.midiff_image_size_second),
        "--diffusion_steps",
        str(args.midiff_diffusion_steps),
        "--learn_sigma",
        str(args.midiff_learn_sigma),
        "--num_res_blocks",
        str(args.midiff_num_res_blocks),
        "--attention_type",
        str(args.midiff_attention_type),
    ]
    if args.midiff_extra_args:
        command.extend(args.midiff_extra_args)
    return command


def write_midiff_manifest(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    env_info: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> Dict[str, Any]:
    model_dir = ensure_under(run_dir / "MIDiff", run_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = ensure_under(model_dir / "sample_output", run_dir)
    command = build_midiff_command(args, save_dir)
    checkpoint_meta = file_info(args.midiff_checkpoint, hash_file=not args.skip_checkpoint_hash)
    manifest = {
        "model": "MIDiff",
        "status": "pending_real_runner" if checkpoint_meta.get("exists") else "missing_checkpoint",
        "reason": (
            "real run is not implemented here because the existing sample_midiff.py path enters PNG/FID postprocessing; "
            "this manifest records a checkpoint-checked isolated command only"
        ),
        "checkpoint": checkpoint_meta,
        "command": command,
        "command_line": shell_join(command),
        "cwd": str(MIDIFF_ROOT / "scripts"),
        "save_dir": str(save_dir),
        "reference_log": file_info(args.midiff_reference_log, hash_file=False),
        "num_samples": args.num_samples,
        "batch_size": args.midiff_batch_size or args.fixed_batch_size,
        "output_isolation": str(save_dir),
    }
    write_json(model_dir / "manifest.json", manifest)
    (model_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shell_join(command) + "\n", encoding="utf-8")

    record = base_raw_record("MIDiff", args, run_id, run_dir, env_info, quality, checkpoint_meta)
    record.update(
        {
            "benchmark_profile": "manifest",
            "batch_size": args.midiff_batch_size or args.fixed_batch_size,
            "status": manifest["status"],
            "manifest_path": str(model_dir / "manifest.json"),
            "prepared_command": shell_join(command),
            "forward_passes_per_sample": args.midiff_diffusion_steps,
            "notes": manifest["reason"],
        }
    )
    return record


def write_imagentime_manifest(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    env_info: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> Dict[str, Any]:
    model_dir = ensure_under(run_dir / "ImagenTime", run_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.imagentime_checkpoint or infer_imagentime_checkpoint()
    checkpoint_meta = file_info(checkpoint, hash_file=not args.skip_checkpoint_hash)
    status = "ready_manifest" if checkpoint_meta.get("exists") else "missing_checkpoint"
    command = [
        str(args.python),
        str(IMAGENTIME_ROOT / "run_visualization.py"),
        "--config",
        str(args.imagentime_config),
        "--resume",
        "True",
        "--log_dir",
        str((checkpoint.parent.parent if checkpoint.exists() else IMAGENTIME_ROOT / "logs")),
        "--batch_size",
        str(args.fixed_batch_size),
    ]
    manifest = {
        "model": "ImagenTime",
        "status": status,
        "checkpoint": checkpoint_meta,
        "config": file_info(args.imagentime_config, hash_file=False),
        "command": command,
        "command_line": shell_join(command),
        "reason": (
            "checkpoint is required for final efficiency; this runner does not instantiate a random untrained ImagenTime model"
            if status == "missing_checkpoint"
            else "checkpoint found; real sample-only benchmark still needs a no-visualization/no-random wrapper"
        ),
    }
    write_json(model_dir / "manifest.json", manifest)
    record = base_raw_record("ImagenTime", args, run_id, run_dir, env_info, quality, checkpoint_meta)
    record.update(
        {
            "benchmark_profile": "manifest",
            "batch_size": args.fixed_batch_size,
            "status": status,
            "manifest_path": str(model_dir / "manifest.json"),
            "prepared_command": shell_join(command),
            "forward_passes_per_sample": 18,
            "notes": manifest["reason"],
        }
    )
    return record


def purge_modules_from_roots(module_names: Sequence[str], roots: Sequence[Path]) -> None:
    root_strings = [str(root.resolve()) for root in roots]
    for name in list(sys.modules):
        if not any(name == module or name.startswith(module + ".") for module in module_names):
            continue
        mod = sys.modules.get(name)
        mod_file = getattr(mod, "__file__", None)
        if mod_file and any(str(Path(mod_file).resolve()).startswith(root) for root in root_strings):
            del sys.modules[name]


def import_imagentime_modules(run_dir: Path) -> Dict[str, Any]:
    old_cwd = Path.cwd()
    inserted = False
    purge_modules_from_roots(["utils", "models", "metrics"], [ZITS_ROOT, MIDIFF_ROOT / "scripts"])
    if str(IMAGENTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(IMAGENTIME_ROOT))
        inserted = True
    try:
        os.chdir(IMAGENTIME_ROOT)
        from omegaconf import OmegaConf  # type: ignore
        from models.model import ImagenTime  # type: ignore
        from models.sampler import DiffusionProcess  # type: ignore
        from utils.utils_data import gen_dataloader  # type: ignore

        return {
            "OmegaConf": OmegaConf,
            "ImagenTime": ImagenTime,
            "DiffusionProcess": DiffusionProcess,
            "gen_dataloader": gen_dataloader,
        }
    finally:
        os.chdir(old_cwd)
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(IMAGENTIME_ROOT))


def build_imagentime_args(args: argparse.Namespace) -> Any:
    modules = import_imagentime_modules(RESULTS_ROOT)
    cfg = modules["OmegaConf"].to_object(modules["OmegaConf"].load(args.imagentime_config))
    payload = {
        "seed": args.seed,
        "num_workers": 0,
        "resume": False,
        "log_dir": str(IMAGENTIME_ROOT / "logs"),
        "neptune": False,
        "tags": ["efficiency"],
        "beta1": 1e-5,
        "betaT": 1e-2,
        "deterministic": False,
        "config": str(args.imagentime_config),
        "percent": 100,
    }
    payload.update(cfg)
    return argparse.Namespace(**payload)


def run_imagentime_benchmark(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    env_info: Mapping[str, Any],
    quality: Mapping[str, Any],
    profiles: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    checkpoint = args.imagentime_checkpoint or infer_imagentime_checkpoint()
    checkpoint_meta = file_info(checkpoint, hash_file=not args.skip_checkpoint_hash)
    if not checkpoint_meta.get("exists") and not args.imagentime_allow_random_weights:
        return [write_imagentime_manifest(args, run_id, run_dir, env_info, quality)], []
    if not torch.cuda.is_available() and not args.allow_cpu:
        record = base_raw_record("ImagenTime", args, run_id, run_dir, env_info, quality, checkpoint_meta)
        record.update({"benchmark_profile": "none", "status": "failed_no_cuda"})
        return [record], []

    modules = import_imagentime_modules(run_dir)
    it_args = build_imagentime_args(args)
    it_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    it_args.batch_size = args.fixed_batch_size
    device = torch.device(it_args.device)

    load_start = time.perf_counter()
    train_loader, _ = modules["gen_dataloader"](it_args)
    model = modules["ImagenTime"](args=it_args, device=it_args.device).to(device)
    if it_args.use_stft:
        model.init_stft_embedder(train_loader)
    else:
        _ = model.ts_to_img(next(iter(train_loader))[0].to(device))
    checkpoint_epoch = None
    scope = "architecture_only_random_weights"
    load_note = "ImagenTime checkpoint missing; random weights used only for architecture/runtime cost."
    if checkpoint_meta.get("exists"):
        loaded = torch.load(str(checkpoint), map_location=device)
        model.load_state_dict(loaded.get("model", loaded), strict=False)
        if "ema_model" in loaded and getattr(model, "model_ema", None) is not None:
            model.model_ema.load_state_dict(loaded["ema_model"])
        checkpoint_epoch = loaded.get("epoch")
        scope = "sample_only"
        load_note = "Loaded trained ImagenTime checkpoint."
    model.eval()
    process = modules["DiffusionProcess"](
        it_args,
        model.net,
        (it_args.input_channels, it_args.img_resolution, it_args.img_resolution),
    )
    cuda_synchronize(torch)
    model_load_time = time.perf_counter() - load_start
    params = count_parameters(model)
    imagen_diffusion_steps = int(it_args.diffusion_steps)
    imagen_forward_calls_per_sample = max(1, 2 * imagen_diffusion_steps - 1)

    flops_info: Dict[str, Any] = {"flops_per_forward": None, "macs_per_forward": None, "flops_notes": None}
    if args.compute_flops:
        try:
            from fvcore.nn import FlopCountAnalysis  # type: ignore

            class ImagenForwardWrapper(torch.nn.Module):
                def __init__(self, wrapped: Any):
                    super().__init__()
                    self.wrapped = wrapped

                def forward(self, x: Any, sigma: Any) -> Any:
                    return self.wrapped(x, sigma, None)

            x = torch.randn(1, it_args.input_channels, it_args.img_resolution, it_args.img_resolution, device=device)
            sigma = torch.full((1,), float(process.net.sigma_max), device=device)
            with torch.no_grad():
                flops_info["flops_per_forward"] = float(FlopCountAnalysis(ImagenForwardWrapper(model.net).eval(), (x, sigma)).total())
        except Exception as exc:
            flops_info["flops_notes"] = f"{type(exc).__name__}: {exc}"

    data_min = np.asarray(getattr(it_args, "data_min", np.zeros(3)), dtype=np.float32)
    data_max = np.asarray(getattr(it_args, "data_max", np.ones(3)), dtype=np.float32)

    def postprocess(samples_tensor: Any, out_dir: Path) -> Tuple[Dict[str, Any], float]:
        start = time.perf_counter()
        with torch.no_grad():
            x_ts = model.img_to_ts(samples_tensor).detach().cpu().numpy().astype(np.float32)
        scale = np.maximum(data_max - data_min, 1e-7)
        x_ts = x_ts * scale.reshape(1, 1, -1) + data_min.reshape(1, 1, -1)
        x_ts[:, :, 0] = np.clip(x_ts[:, :, 0], data_min[0], data_max[0])
        x_ts[:, :, 1] = np.clip(np.round(x_ts[:, :, 1]), data_min[1], data_max[1])
        x_ts[:, :, 2] = np.clip(np.round(x_ts[:, :, 2]), data_min[2], data_max[2])
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / "samples.npy"
        csv_path = out_dir / "samples.csv"
        np.save(npy_path, x_ts)
        np.savetxt(csv_path, x_ts.reshape(x_ts.shape[0], -1), delimiter=",", fmt="%.8g")
        elapsed = time.perf_counter() - start
        return {
            "npy_path": str(npy_path),
            "csv_path": str(csv_path),
            "npy_size_bytes": npy_path.stat().st_size,
            "csv_size_bytes": csv_path.stat().st_size,
            "csv_sha256": sha256_file(csv_path),
            "array_shape": list(x_ts.shape),
            "flat_shape": [x_ts.shape[0], x_ts.shape[1] * x_ts.shape[2]],
            "nan_count": int(np.isnan(x_ts).sum()),
            "inf_count": int(np.isinf(x_ts).sum()),
            "label1_min": float(np.min(x_ts[:, :, 1])) if x_ts.size else None,
            "label1_max": float(np.max(x_ts[:, :, 1])) if x_ts.size else None,
            "label2_min": float(np.min(x_ts[:, :, 2])) if x_ts.size else None,
            "label2_max": float(np.max(x_ts[:, :, 2])) if x_ts.size else None,
        }, elapsed

    if args.imagentime_per_step_only:
        raw_records: List[Dict[str, Any]] = []
        validations: List[Dict[str, Any]] = []

        def one_forward(batch_size: int) -> None:
            x = torch.randn(batch_size, it_args.input_channels, it_args.img_resolution, it_args.img_resolution, device=device)
            sigma = torch.full((batch_size,), float(process.net.sigma_max), device=device)
            with torch.no_grad(), model.ema_scope():
                _ = model.net(x, sigma, None)

        for profile in profiles:
            batch_size = 1 if profile == "latency_b1" else (args.fixed_batch_size if profile == "throughput_fixed" else args.max_batch_size)
            model_dir = ensure_under(run_dir / "ImagenTime" / profile, run_dir)
            model_dir.mkdir(parents=True, exist_ok=True)

            for _ in range(args.warmup_runs):
                cuda_synchronize(torch)
                one_forward(batch_size)
                cuda_synchronize(torch)

            repeat_times: List[float] = []
            for repeat_idx in range(args.measured_repeats):
                status = "ok"
                error = None
                memory = {"peak_cuda_memory_allocated_mb": None, "peak_cuda_memory_reserved_mb": None}
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    cuda_synchronize(torch)
                    start = time.perf_counter()
                    one_forward(batch_size)
                    cuda_synchronize(torch)
                    core_time = time.perf_counter() - start
                    if torch.cuda.is_available():
                        memory = {
                            "peak_cuda_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                            "peak_cuda_memory_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
                        }
                    repeat_times.append(core_time)
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    core_time = math.nan
                validation = {
                    "ok": status == "ok",
                    "kind": "per_step_forward_only",
                    "batch_size": batch_size,
                    "image_shape": [batch_size, it_args.input_channels, it_args.img_resolution, it_args.img_resolution],
                    "diffusion_steps": imagen_diffusion_steps,
                    "net_forward_calls_per_sample": imagen_forward_calls_per_sample,
                    "note": "Measures one ImagenTime denoising-network forward, not full sampling or image-to-time-series decoding.",
                }
                record = base_raw_record("ImagenTime", args, run_id, run_dir, env_info, quality, checkpoint_meta)
                record.update(
                    {
                        "scope": "per_step_forward" if checkpoint_meta.get("exists") else "architecture_only_per_step_random_weights",
                        "benchmark_profile": profile,
                        "batch_size": batch_size,
                        "num_samples": batch_size,
                        "repeat_idx": repeat_idx,
                        "status": status,
                        "error": error,
                        "model_load_time_sec": model_load_time,
                        "core_generation_time_sec": core_time,
                        "postprocess_time_sec": 0.0,
                        "total_command_time_sec": model_load_time + (0 if math.isnan(core_time) else core_time),
                        "samples_per_sec_core": batch_size / core_time if core_time and not math.isnan(core_time) else None,
                        "latency_ms_per_sample_core": 1000.0 * core_time / batch_size if core_time and not math.isnan(core_time) else None,
                        "runtime_mean_sec": float(np.mean(repeat_times)) if repeat_times else None,
                        "runtime_std_sec": float(np.std(repeat_times, ddof=1)) if len(repeat_times) > 1 else 0.0,
                        "peak_cuda_memory_allocated_mb": memory.get("peak_cuda_memory_allocated_mb"),
                        "peak_cuda_memory_reserved_mb": memory.get("peak_cuda_memory_reserved_mb"),
                        "forward_passes_per_sample": imagen_forward_calls_per_sample,
                        "estimated_full_sample_latency_ms_from_per_step": (
                            1000.0 * core_time * imagen_forward_calls_per_sample / batch_size
                            if core_time and not math.isnan(core_time)
                            else None
                        ),
                        "best_checkpoint_epoch_or_step": checkpoint_epoch,
                        "validation": validation,
                        "notes": f"{load_note} Per-step mode: ImagenTime uses {imagen_diffusion_steps} sampler steps and {imagen_forward_calls_per_sample} denoising-network calls per sample.",
                        **params,
                        **flops_info,
                    }
                )
                if record.get("flops_per_forward") is not None:
                    record["estimated_flops_per_sample"] = record["flops_per_forward"] * imagen_forward_calls_per_sample
                    record["estimated_total_flops"] = record["estimated_flops_per_sample"]
                raw_records.append(record)
                validations.append({"model": "ImagenTime", "profile": profile, "repeat_idx": repeat_idx, "status": status, "validation": validation, "error": error})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return raw_records, validations

    raw_records: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    for profile in profiles:
        batch_size = 1 if profile == "latency_b1" else (args.fixed_batch_size if profile == "throughput_fixed" else args.max_batch_size)
        sample_count = args.latency_samples if profile == "latency_b1" else args.num_samples
        model_dir = ensure_under(run_dir / "ImagenTime" / profile, run_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        warmup_samples = args.warmup_samples if args.warmup_samples > 0 else min(sample_count, batch_size)
        for _ in range(args.warmup_runs):
            cuda_synchronize(torch)
            with torch.no_grad(), model.ema_scope():
                warm = process.sampling(sampling_number=warmup_samples)
            cuda_synchronize(torch)
            del warm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        repeat_core_times: List[float] = []
        for repeat_idx in range(args.measured_repeats):
            out_dir = ensure_under(model_dir / f"repeat_{repeat_idx:02d}", run_dir)
            status = "ok"
            error = None
            output: Dict[str, Any] = {}
            validation: Dict[str, Any] = {}
            memory = {"peak_cuda_memory_allocated_mb": None, "peak_cuda_memory_reserved_mb": None}
            total_start = time.perf_counter()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                chunks = []
                remaining = sample_count
                cuda_synchronize(torch)
                core_start = time.perf_counter()
                with torch.no_grad(), model.ema_scope():
                    while remaining > 0:
                        bs = min(batch_size, remaining)
                        chunks.append(process.sampling(sampling_number=bs).detach())
                        remaining -= bs
                cuda_synchronize(torch)
                core_time = time.perf_counter() - core_start
                if torch.cuda.is_available():
                    memory = {
                        "peak_cuda_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                        "peak_cuda_memory_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
                    }
                generated = torch.cat(chunks, dim=0)[:sample_count]
                output, post_time = postprocess(generated, out_dir)
                del generated, chunks
                validation = validate_zits_output(output, sample_count, args.seq_len)
                if not validation["ok"]:
                    status = "validation_failed"
                repeat_core_times.append(core_time)
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                core_time = math.nan
                post_time = math.nan
            total_time = time.perf_counter() - total_start
            record = base_raw_record("ImagenTime", args, run_id, run_dir, env_info, quality, checkpoint_meta)
            record.update(
                {
                    "scope": scope,
                    "benchmark_profile": profile,
                    "batch_size": batch_size,
                    "num_samples": sample_count,
                    "repeat_idx": repeat_idx,
                    "status": status,
                    "error": error,
                    "model_load_time_sec": model_load_time,
                    "core_generation_time_sec": core_time,
                    "postprocess_time_sec": post_time,
                    "total_command_time_sec": model_load_time + total_time,
                    "samples_per_sec_core": sample_count / core_time if core_time and not math.isnan(core_time) else None,
                    "samples_per_sec_total": sample_count / (model_load_time + total_time)
                    if (model_load_time + total_time) > 0
                    else None,
                    "latency_ms_per_sample_core": 1000.0 * core_time / sample_count
                    if core_time and not math.isnan(core_time)
                    else None,
                    "latency_ms_per_sample_total": 1000.0 * (model_load_time + total_time) / sample_count,
                    "runtime_mean_sec": float(np.mean(repeat_core_times)) if repeat_core_times else None,
                    "runtime_std_sec": float(np.std(repeat_core_times, ddof=1)) if len(repeat_core_times) > 1 else 0.0,
                    "peak_cuda_memory_allocated_mb": memory.get("peak_cuda_memory_allocated_mb"),
                    "peak_cuda_memory_reserved_mb": memory.get("peak_cuda_memory_reserved_mb"),
                    "forward_passes_per_sample": imagen_forward_calls_per_sample,
                    "best_checkpoint_epoch_or_step": checkpoint_epoch,
                    "output": output,
                    "validation": validation,
                    "notes": f"{load_note} ImagenTime uses {imagen_diffusion_steps} sampler steps and {imagen_forward_calls_per_sample} denoising-network calls per sample.",
                    **params,
                    **flops_info,
                }
            )
            if record.get("flops_per_forward") is not None:
                record["estimated_flops_per_sample"] = record["flops_per_forward"] * imagen_forward_calls_per_sample
                record["estimated_total_flops"] = record["estimated_flops_per_sample"] * sample_count
            raw_records.append(record)
            validations.append({"model": "ImagenTime", "profile": profile, "repeat_idx": repeat_idx, "status": status, "output": output, "validation": validation, "error": error})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return raw_records, validations


def import_midiff_modules() -> Dict[str, Any]:
    inserted = False
    scripts_root = MIDIFF_ROOT / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
        inserted = True
    try:
        import dist_util  # type: ignore
        from script_util import create_model_and_diffusion, model_and_diffusion_defaults  # type: ignore

        return {
            "dist_util": dist_util,
            "create_model_and_diffusion": create_model_and_diffusion,
            "model_and_diffusion_defaults": model_and_diffusion_defaults,
            "inserted": inserted,
        }
    except Exception:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(scripts_root))
        raise


def run_midiff_benchmark(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    env_info: Mapping[str, Any],
    quality: Mapping[str, Any],
    profiles: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    checkpoint_meta = file_info(args.midiff_checkpoint, hash_file=not args.skip_checkpoint_hash)
    if not checkpoint_meta.get("exists"):
        return [write_midiff_manifest(args, run_id, run_dir, env_info, quality)], []
    if not torch.cuda.is_available() and not args.allow_cpu:
        record = base_raw_record("MIDiff", args, run_id, run_dir, env_info, quality, checkpoint_meta)
        record.update({"benchmark_profile": "none", "status": "failed_no_cuda"})
        return [record], []

    modules = import_midiff_modules()
    defaults = modules["model_and_diffusion_defaults"]()
    model_kwargs = dict(defaults)
    model_kwargs.update({
        "image_size": args.midiff_image_size,
        "num_channels": 128,
        "num_res_blocks": args.midiff_num_res_blocks,
        "diffusion_steps": args.midiff_diffusion_steps,
        "learn_sigma": args.midiff_learn_sigma,
        "attention_type": args.midiff_attention_type,
    })
    for item in args.midiff_extra_args or []:
        if not item.startswith("--") or "=" not in item:
            continue
        key, value = item[2:].split("=", 1)
        if key not in model_kwargs:
            continue
        old_value = model_kwargs[key]
        if isinstance(old_value, bool):
            model_kwargs[key] = value.lower() in {"1", "true", "yes"}
        elif isinstance(old_value, int):
            model_kwargs[key] = int(value)
        elif isinstance(old_value, float):
            model_kwargs[key] = float(value)
        else:
            model_kwargs[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_start = time.perf_counter()
    create_kwargs = {k: model_kwargs[k] for k in defaults.keys()}
    model, diffusion = modules["create_model_and_diffusion"](**create_kwargs)
    state = modules["dist_util"].load_state_dict(str(args.midiff_checkpoint), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    cuda_synchronize(torch)
    model_load_time = time.perf_counter() - load_start
    params = count_parameters(model)
    image_size_second = args.midiff_image_size_second

    flops_info: Dict[str, Any] = {"flops_per_forward": None, "macs_per_forward": None, "flops_notes": None}
    if args.compute_flops:
        try:
            from fvcore.nn import FlopCountAnalysis  # type: ignore

            class Wrapper(torch.nn.Module):
                def __init__(self, wrapped: Any):
                    super().__init__()
                    self.wrapped = wrapped

                def forward(self, x: Any, t: Any) -> Any:
                    return self.wrapped(x, t)

            x = torch.randn(1, 1, args.midiff_image_size, image_size_second, device=device)
            t = torch.randint(0, int(diffusion.num_timesteps), (1,), device=device)
            t = diffusion._scale_timesteps(t)
            with torch.no_grad():
                flops_info["flops_per_forward"] = float(FlopCountAnalysis(Wrapper(model).eval(), (x, t)).total())
        except Exception as exc:
            flops_info["flops_notes"] = f"{type(exc).__name__}: {exc}"

    raw_records: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    for profile in profiles:
        batch_size = 1 if profile == "latency_b1" else (args.fixed_batch_size if profile == "throughput_fixed" else args.max_batch_size)
        repeat_count = args.measured_repeats
        warmup_count = args.warmup_runs
        model_dir = ensure_under(run_dir / "MIDiff" / profile, run_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        def one_forward() -> None:
            x = torch.randn(batch_size, 1, args.midiff_image_size, image_size_second, device=device)
            t = torch.randint(0, int(diffusion.num_timesteps), (batch_size,), device=device)
            t = diffusion._scale_timesteps(t)
            with torch.no_grad():
                _ = model(x, t)

        for _ in range(warmup_count):
            cuda_synchronize(torch)
            one_forward()
            cuda_synchronize(torch)

        repeat_times: List[float] = []
        for repeat_idx in range(repeat_count):
            status = "ok"
            error = None
            memory = {"peak_cuda_memory_allocated_mb": None, "peak_cuda_memory_reserved_mb": None}
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                cuda_synchronize(torch)
                start = time.perf_counter()
                one_forward()
                cuda_synchronize(torch)
                core_time = time.perf_counter() - start
                if torch.cuda.is_available():
                    memory = {
                        "peak_cuda_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                        "peak_cuda_memory_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
                    }
                repeat_times.append(core_time)
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                core_time = math.nan
            validation = {
                "ok": status == "ok",
                "kind": "per_step_forward_only",
                "batch_size": batch_size,
                "image_shape": [batch_size, 1, args.midiff_image_size, image_size_second],
                "note": "Measures one denoising model forward, not full 1000-step sampling or image decoding.",
            }
            record = base_raw_record("MIDiff", args, run_id, run_dir, env_info, quality, checkpoint_meta)
            record.update(
                {
                    "scope": "per_step_forward",
                    "benchmark_profile": profile,
                    "batch_size": batch_size,
                    "num_samples": batch_size,
                    "repeat_idx": repeat_idx,
                    "status": status,
                    "error": error,
                    "model_load_time_sec": model_load_time,
                    "core_generation_time_sec": core_time,
                    "postprocess_time_sec": 0.0,
                    "total_command_time_sec": model_load_time + (0 if math.isnan(core_time) else core_time),
                    "samples_per_sec_core": batch_size / core_time if core_time and not math.isnan(core_time) else None,
                    "latency_ms_per_sample_core": 1000.0 * core_time / batch_size if core_time and not math.isnan(core_time) else None,
                    "runtime_mean_sec": float(np.mean(repeat_times)) if repeat_times else None,
                    "runtime_std_sec": float(np.std(repeat_times, ddof=1)) if len(repeat_times) > 1 else 0.0,
                    "peak_cuda_memory_allocated_mb": memory.get("peak_cuda_memory_allocated_mb"),
                    "peak_cuda_memory_reserved_mb": memory.get("peak_cuda_memory_reserved_mb"),
                    "forward_passes_per_sample": int(diffusion.num_timesteps),
                    "estimated_full_sample_latency_ms_from_per_step": (
                        1000.0 * core_time * int(diffusion.num_timesteps) / batch_size
                        if core_time and not math.isnan(core_time)
                        else None
                    ),
                    "load_state_missing_keys_count": len(missing),
                    "load_state_unexpected_keys_count": len(unexpected),
                    "validation": validation,
                    "notes": "Per-step forward benchmark; use alongside historical/incomplete sampling log and quality table.",
                    **params,
                    **flops_info,
                }
            )
            if record.get("flops_per_forward") is not None:
                record["estimated_flops_per_sample"] = record["flops_per_forward"] * int(diffusion.num_timesteps)
                record["estimated_total_flops"] = record["estimated_flops_per_sample"]
            raw_records.append(record)
            validations.append({"model": "MIDiff", "profile": profile, "repeat_idx": repeat_idx, "status": status, "validation": validation, "error": error})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return raw_records, validations


def import_zits(run_dir: Path) -> Any:
    import_log = run_dir / "zits_import.log"
    old_cwd = Path.cwd()
    inserted = False
    purge_modules_from_roots(["utils", "models", "metrics"], [IMAGENTIME_ROOT, MIDIFF_ROOT / "scripts"])
    if str(ZITS_ROOT) not in sys.path:
        sys.path.insert(0, str(ZITS_ROOT))
        inserted = True
    try:
        os.chdir(ZITS_ROOT)
        with import_log.open("a", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                return importlib.import_module("run_our_zits")
    finally:
        os.chdir(old_cwd)
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(ZITS_ROOT))


def count_parameters(model: Any) -> Dict[str, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": total - trainable,
    }


def cuda_synchronize(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def zits_make_model(zits: Any, model_name: str, ckpt: Mapping[str, Any], device: Any) -> Any:
    if model_name == "ZITS-VAE":
        model = zits.MultiChannelVAE(
            seq_length=ckpt["seq_len"], latent_dim=ckpt["latent_dim"], hidden_ch=ckpt["hidden_ch"]
        ).to(device)
    else:
        model = zits.MultiChannelGenerator(
            seq_length=ckpt["seq_len"], latent_dim=ckpt["latent_dim"], hidden_ch=ckpt["hidden_ch"]
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def zits_generate_batch(torch: Any, model: Any, batch_size: int, device: Any) -> Any:
    z = torch.randn(batch_size, model.latent_dim, device=device)
    outputs = model.decoder(z)
    return model.decoder.hard_sample(outputs)


def zits_core_sample(torch: Any, model: Any, num_samples: int, batch_size: int, device: Any) -> Tuple[Any, float, Dict[str, Any]]:
    chunks: List[Any] = []
    remaining = num_samples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    cuda_synchronize(torch)
    start = time.perf_counter()
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            chunks.append(zits_generate_batch(torch, model, bs, device).detach())
            remaining -= bs
    cuda_synchronize(torch)
    elapsed = time.perf_counter() - start
    memory = {
        "peak_cuda_memory_allocated_mb": None,
        "peak_cuda_memory_reserved_mb": None,
    }
    if torch.cuda.is_available():
        memory["peak_cuda_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        memory["peak_cuda_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return torch.cat(chunks, dim=0)[:num_samples], elapsed, memory


def zits_postprocess_and_save(
    samples_tensor: Any,
    preprocessor: Any,
    label1_card: int,
    label2_card: int,
    out_dir: Path,
) -> Tuple[Dict[str, Any], float]:
    import numpy as np  # type: ignore

    start = time.perf_counter()
    samples = samples_tensor.detach().cpu().numpy().astype(np.float32)
    samples[:, :, 0] = preprocessor.inverse_transform(samples[:, :, 0])
    samples[:, :, 1] = np.clip(np.round(samples[:, :, 1]), 0, label1_card - 1)
    samples[:, :, 2] = np.clip(np.round(samples[:, :, 2]), 0, label2_card - 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "samples.npy"
    csv_path = out_dir / "samples.csv"
    np.save(npy_path, samples)
    np.savetxt(csv_path, samples.reshape(samples.shape[0], -1), delimiter=",", fmt="%.8g")
    elapsed = time.perf_counter() - start
    return {
        "npy_path": str(npy_path),
        "csv_path": str(csv_path),
        "npy_size_bytes": npy_path.stat().st_size,
        "csv_size_bytes": csv_path.stat().st_size,
        "csv_sha256": sha256_file(csv_path),
        "array_shape": list(samples.shape),
        "flat_shape": [samples.shape[0], samples.shape[1] * samples.shape[2]],
        "nan_count": int(np.isnan(samples).sum()),
        "inf_count": int(np.isinf(samples).sum()),
        "label1_min": float(np.min(samples[:, :, 1])) if samples.size else None,
        "label1_max": float(np.max(samples[:, :, 1])) if samples.size else None,
        "label2_min": float(np.min(samples[:, :, 2])) if samples.size else None,
        "label2_max": float(np.max(samples[:, :, 2])) if samples.size else None,
    }, elapsed


def validate_zits_output(output: Mapping[str, Any], num_samples: int, seq_len: int) -> Dict[str, Any]:
    expected_flat = [num_samples, seq_len * 3]
    expected_array = [num_samples, seq_len, 3]
    checks = {
        "csv_exists": Path(str(output.get("csv_path", ""))).exists(),
        "npy_exists": Path(str(output.get("npy_path", ""))).exists(),
        "array_shape_ok": output.get("array_shape") == expected_array,
        "flat_shape_ok": output.get("flat_shape") == expected_flat,
        "finite": output.get("nan_count") == 0 and output.get("inf_count") == 0,
        "label1_range_ok": output.get("label1_min", -1) >= 0 and output.get("label1_max", 999) <= 19,
        "label2_range_ok": output.get("label2_min", -1) >= 0 and output.get("label2_max", 999) <= 6,
    }
    checks["ok"] = all(bool(v) for v in checks.values())
    return checks


def try_zits_flops(torch: Any, model: Any, batch_size: int, device: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"flops_per_forward": None, "macs_per_forward": None, "error": None}
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped: Any):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, z: Any) -> Any:
                outputs = self.wrapped.decoder(z)
                return self.wrapped.decoder.proxy_tensor(outputs)

        z = torch.randn(batch_size, model.latent_dim, device=device)
        wrapper = Wrapper(model).eval()
        with torch.no_grad():
            out["flops_per_forward"] = float(FlopCountAnalysis(wrapper, z).total())
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def find_zits_max_batch(torch: Any, model: Any, args: argparse.Namespace, device: Any) -> int:
    best = 0
    candidate = max(1, args.fixed_batch_size)
    while candidate <= args.max_batch_size:
        try:
            sample, _, _ = zits_core_sample(torch, model, candidate, candidate, device)
            del sample
            best = candidate
            candidate *= 2
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    if best == 0:
        return 1
    return min(best, args.max_batch_size)


def zits_profile_batch_size(profile: str, torch: Any, model: Any, args: argparse.Namespace, device: Any) -> int:
    if profile == "latency_b1":
        return 1
    if profile == "throughput_fixed":
        return args.fixed_batch_size
    if profile == "throughput_max":
        return find_zits_max_batch(torch, model, args, device)
    raise ValueError(profile)


def run_zits_benchmark(
    model_name: str,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    env_info: Mapping[str, Any],
    quality: Mapping[str, Any],
    profiles: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    if not torch.cuda.is_available() and not args.allow_cpu:
        checkpoint_meta = file_info(checkpoint_for_model(model_name, args), hash_file=not args.skip_checkpoint_hash)
        record = base_raw_record(model_name, args, run_id, run_dir, env_info, quality, checkpoint_meta)
        record.update(
            {
                "benchmark_profile": "none",
                "status": "failed_no_cuda",
                "notes": "CUDA is required unless --allow-cpu is set; CPU fallback is not mixed into benchmark tables",
            }
        )
        return [record], []

    zits = import_zits(run_dir)
    checkpoint_path = checkpoint_for_model(model_name, args)
    if checkpoint_path is None:
        raise ValueError(model_name)
    checkpoint_meta = file_info(checkpoint_path, hash_file=not args.skip_checkpoint_hash)
    preprocessor_path = preprocessor_for_zits_model(model_name, args)
    preprocessor_meta = file_info(preprocessor_path, hash_file=not args.skip_checkpoint_hash)
    if not checkpoint_meta.get("exists") or not preprocessor_meta.get("exists"):
        record = base_raw_record(model_name, args, run_id, run_dir, env_info, quality, checkpoint_meta)
        record.update(
            {
                "benchmark_profile": "manifest",
                "status": "missing_checkpoint" if not checkpoint_meta.get("exists") else "missing_preprocessor",
                "preprocessor": preprocessor_meta,
            }
        )
        return [record], []

    zits.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_start = time.perf_counter()
    real_samples = zits.load_our_csv(str(args.data_path), args.seq_len)
    _, _, preprocessor = zits.build_dataset(real_samples)
    preprocessor.load(str(preprocessor_path))
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = zits_make_model(zits, model_name, ckpt, device)
    cuda_synchronize(torch)
    model_load_time = time.perf_counter() - load_start
    params = count_parameters(model)

    raw_records: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    for profile in profiles:
        batch_size = zits_profile_batch_size(profile, torch, model, args, device)
        sample_count = args.latency_samples if profile == "latency_b1" else args.num_samples
        model_dir = ensure_under(run_dir / model_name.replace("-", "_") / profile, run_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        flops = try_zits_flops(torch, model, batch_size, device) if args.compute_flops else {}

        warmup_samples = args.warmup_samples if args.warmup_samples > 0 else min(sample_count, batch_size)
        for _ in range(args.warmup_runs):
            sample, _, _ = zits_core_sample(torch, model, warmup_samples, batch_size, device)
            del sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        repeat_core_times: List[float] = []
        for repeat_idx in range(args.measured_repeats):
            out_dir = ensure_under(model_dir / f"repeat_{repeat_idx:02d}", run_dir)
            total_start = time.perf_counter()
            status = "ok"
            error = None
            output: Dict[str, Any] = {}
            validation: Dict[str, Any] = {}
            try:
                generated, core_time, memory = zits_core_sample(torch, model, sample_count, batch_size, device)
                output, post_time = zits_postprocess_and_save(
                    generated,
                    preprocessor,
                    zits.LABEL1_CARD,
                    zits.LABEL2_CARD,
                    out_dir,
                )
                del generated
                validation = validate_zits_output(output, sample_count, args.seq_len)
                if not validation["ok"]:
                    status = "validation_failed"
                repeat_core_times.append(core_time)
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                core_time = math.nan
                post_time = math.nan
                memory = {"peak_cuda_memory_allocated_mb": None, "peak_cuda_memory_reserved_mb": None}
            total_time = time.perf_counter() - total_start
            record = base_raw_record(model_name, args, run_id, run_dir, env_info, quality, checkpoint_meta)
            record.update(
                {
                    "benchmark_profile": profile,
                    "batch_size": batch_size,
                    "num_samples": sample_count,
                    "repeat_idx": repeat_idx,
                    "status": status,
                    "error": error,
                    "preprocessor": preprocessor_meta,
                    "model_load_time_sec": model_load_time,
                    "core_generation_time_sec": core_time,
                    "postprocess_time_sec": post_time,
                    "total_command_time_sec": model_load_time + total_time,
                    "samples_per_sec_core": sample_count / core_time if core_time and not math.isnan(core_time) else None,
                    "samples_per_sec_total": sample_count / (model_load_time + total_time)
                    if (model_load_time + total_time) > 0
                    else None,
                    "latency_ms_per_sample_core": 1000.0 * core_time / sample_count
                    if core_time and not math.isnan(core_time)
                    else None,
                    "latency_ms_per_sample_total": 1000.0 * (model_load_time + total_time) / sample_count,
                    "runtime_mean_sec": float(np.mean(repeat_core_times)) if repeat_core_times else None,
                    "runtime_std_sec": float(np.std(repeat_core_times, ddof=1)) if len(repeat_core_times) > 1 else 0.0,
                    "peak_cuda_memory_allocated_mb": memory.get("peak_cuda_memory_allocated_mb"),
                    "peak_cuda_memory_reserved_mb": memory.get("peak_cuda_memory_reserved_mb"),
                    "forward_passes_per_sample": 1,
                    "output": output,
                    "validation": validation,
                    **params,
                    **{k: v for k, v in flops.items() if k in {"flops_per_forward", "macs_per_forward"}},
                    "flops_notes": flops.get("error"),
                }
            )
            if record.get("flops_per_forward") is not None:
                record["estimated_flops_per_sample"] = record["flops_per_forward"] / max(batch_size, 1)
                record["estimated_total_flops"] = record["estimated_flops_per_sample"] * sample_count
            raw_records.append(record)
            validations.append(
                {
                    "model": model_name,
                    "profile": profile,
                    "repeat_idx": repeat_idx,
                    "status": status,
                    "output": output,
                    "validation": validation,
                    "error": error,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return raw_records, validations


def mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not nums:
        return None
    return statistics.mean(nums)


def stdev(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(nums) < 2:
        return 0.0 if nums else None
    return statistics.stdev(nums)


def summary_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("model")),
            str(record.get("benchmark_profile")),
            str(record.get("status")),
        )
        groups.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for (model, profile, status), items in sorted(groups.items()):
        first = items[0]
        quality = dict(first.get("quality_metrics") or {})
        row: Dict[str, Any] = {
            "model": model,
            "status": status,
            "scope": first.get("scope"),
            "benchmark_profile": profile,
            "num_records": len(items),
            "num_samples": first.get("num_samples"),
            "seq_len": first.get("seq_len"),
            "batch_size": first.get("batch_size"),
            "checkpoint_path": first.get("checkpoint_path"),
            "checkpoint_size_mb": first.get("checkpoint_size_mb"),
            "checkpoint_sha256": first.get("checkpoint_sha256"),
            "total_parameter_count": first.get("total_parameter_count"),
            "trainable_parameter_count": first.get("trainable_parameter_count"),
            "frozen_parameter_count": first.get("frozen_parameter_count"),
            "forward_passes_per_sample": first.get("forward_passes_per_sample"),
            "flops_per_forward": first.get("flops_per_forward"),
            "estimated_flops_per_sample": first.get("estimated_flops_per_sample"),
            "model_load_time_sec": first.get("model_load_time_sec"),
            "core_generation_time_mean_sec": mean(item.get("core_generation_time_sec") for item in items),
            "core_generation_time_std_sec": stdev(item.get("core_generation_time_sec") for item in items),
            "postprocess_time_mean_sec": mean(item.get("postprocess_time_sec") for item in items),
            "total_command_time_mean_sec": mean(item.get("total_command_time_sec") for item in items),
            "samples_per_sec_core_mean": mean(item.get("samples_per_sec_core") for item in items),
            "samples_per_sec_total_mean": mean(item.get("samples_per_sec_total") for item in items),
            "latency_ms_per_sample_core_mean": mean(item.get("latency_ms_per_sample_core") for item in items),
            "latency_ms_per_sample_total_mean": mean(item.get("latency_ms_per_sample_total") for item in items),
            "estimated_full_sample_latency_ms_from_per_step_mean": mean(
                item.get("estimated_full_sample_latency_ms_from_per_step") for item in items
            ),
            "peak_cuda_memory_allocated_mb_max": max(
                [float(item["peak_cuda_memory_allocated_mb"]) for item in items if item.get("peak_cuda_memory_allocated_mb") is not None],
                default=None,
            ),
            "peak_cuda_memory_reserved_mb_max": max(
                [float(item["peak_cuda_memory_reserved_mb"]) for item in items if item.get("peak_cuda_memory_reserved_mb") is not None],
                default=None,
            ),
            "quality_source": first.get("quality_source"),
            "notes": first.get("notes"),
        }
        for field in QUALITY_FIELDS:
            row[f"quality_{field}"] = quality.get(field)
        rows.append(row)
    return rows


def write_summary_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = summary_rows(records)
    base_fields = [
        "model",
        "status",
        "scope",
        "benchmark_profile",
        "num_records",
        "num_samples",
        "seq_len",
        "batch_size",
        "checkpoint_path",
        "checkpoint_size_mb",
        "checkpoint_sha256",
        "total_parameter_count",
        "trainable_parameter_count",
        "frozen_parameter_count",
        "forward_passes_per_sample",
        "flops_per_forward",
        "estimated_flops_per_sample",
        "model_load_time_sec",
        "core_generation_time_mean_sec",
        "core_generation_time_std_sec",
        "postprocess_time_mean_sec",
        "total_command_time_mean_sec",
        "samples_per_sec_core_mean",
        "samples_per_sec_total_mean",
        "latency_ms_per_sample_core_mean",
        "latency_ms_per_sample_total_mean",
        "estimated_full_sample_latency_ms_from_per_step_mean",
        "peak_cuda_memory_allocated_mb_max",
        "peak_cuda_memory_reserved_mb_max",
        "quality_MDD",
        "quality_ACD",
        "quality_DTW",
        "quality_ED",
        "quality_VDS",
        "quality_FDDS",
        "quality_source",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated efficiency benchmark runner for ZITS-GAN, ZITS-VAE, MIDiff, and ImagenTime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--action", choices=["manifest", "benchmark"], default="manifest", help="manifest only or run ZITS benchmarks")
    parser.add_argument("--models", default="all", help="comma list: all,zits-gan,zits-vae,midiff,imagentime")
    parser.add_argument("--profiles", default="latency_b1,throughput_fixed,throughput_max", help="comma list or all")
    parser.add_argument("--run-id", default=None, help="run id under efficiency_results")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--fail-if-run-exists", action="store_true", help="fail instead of suffixing a duplicate run id")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--quality-xlsx", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--latency-samples", type=int, default=1, help="sample count for latency_b1 profiles")
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=0, help="0 means one batch for warmup")
    parser.add_argument("--measured-repeats", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=None, help="physical nvidia-smi GPU index to precheck and expose")
    parser.add_argument("--gpu-max-memory-mb", type=int, default=256)
    parser.add_argument("--gpu-max-utilization-pct", type=int, default=5)
    parser.add_argument("--allow-cpu", action="store_true", help="allow CPU records; do not mix CPU and CUDA in final tables")
    parser.add_argument("--skip-checkpoint-hash", action="store_true", help="debug escape hatch; default records checkpoint sha256")
    parser.add_argument("--compute-flops", action="store_true", help="try fvcore FLOPs for ZITS decoder/generator")

    parser.add_argument("--zits-gan-checkpoint", type=Path, default=ZITS_GAN_CHECKPOINT)
    parser.add_argument("--zits-vae-checkpoint", type=Path, default=ZITS_VAE_CHECKPOINT)
    parser.add_argument("--zits-gan-preprocessor", type=Path, default=ZITS_GAN_PREPROCESSOR)
    parser.add_argument("--zits-vae-preprocessor", type=Path, default=ZITS_VAE_PREPROCESSOR)

    parser.add_argument("--midiff-checkpoint", type=Path, default=DEFAULT_MIDIFF_CHECKPOINT)
    parser.add_argument("--midiff-reference-log", type=Path, default=DEFAULT_MIDIFF_REFERENCE_LOG)
    parser.add_argument("--midiff-batch-size", type=int, default=None)
    parser.add_argument("--midiff-image-size", type=int, default=256)
    parser.add_argument("--midiff-image-size-second", type=int, default=160)
    parser.add_argument("--midiff-diffusion-steps", type=int, default=1000)
    parser.add_argument("--midiff-learn-sigma", type=bool, default=True)
    parser.add_argument("--midiff-num-res-blocks", type=int, default=3)
    parser.add_argument("--midiff-attention-type", default="triple")
    parser.add_argument("--midiff-extra-args", nargs=argparse.REMAINDER, default=None)

    parser.add_argument("--imagentime-checkpoint", type=Path, default=None)
    parser.add_argument("--imagentime-config", type=Path, default=IMAGENTIME_CONFIG)
    parser.add_argument(
        "--imagentime-allow-random-weights",
        action="store_true",
        help="allow architecture-only ImagenTime benchmark when the trained checkpoint is missing",
    )
    parser.add_argument(
        "--imagentime-per-step-only",
        action="store_true",
        help="benchmark one ImagenTime denoising-network forward instead of full sampling",
    )

    args = parser.parse_args(argv)
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    if args.latency_samples <= 0:
        parser.error("--latency-samples must be positive")
    if args.fixed_batch_size <= 0 or args.max_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.measured_repeats <= 0:
        parser.error("--measured-repeats must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        models = canonicalize_models(args.models)
        profiles = parse_profiles(args.profiles)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_id, run_dir = make_run_dir(args.results_root, args.run_id, args.fail_if_run_exists)
    raw_path = run_dir / "efficiency_raw_runs.jsonl"
    summary_path = run_dir / "efficiency_summary.csv"
    validation_path = run_dir / "validation.json"
    validation: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "started_at": utc_timestamp(),
        "status": "started",
        "models": models,
        "profiles": profiles,
        "records": [],
        "output_validations": [],
    }
    all_records: List[Dict[str, Any]] = []
    post_snapshot: Optional[Dict[str, Any]] = None
    exit_code = 0

    try:
        write_command_files(run_dir, args, run_id, models, profiles)
        before_snapshot = capture_nvidia_smi(run_dir, "before")
        gpu_precheck = check_gpu_idle(
            before_snapshot,
            args.gpu,
            args.gpu_max_memory_mb,
            args.gpu_max_utilization_pct,
        )
        validation["gpu_precheck"] = gpu_precheck
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        env_info = collect_runtime_env(args, models)
        write_json(run_dir / "environment.json", env_info)

        quality_path = choose_quality_path(args.quality_xlsx)
        quality = read_quality_metrics(quality_path, models)
        write_json(run_dir / "quality_metrics.json", quality)

        if not gpu_precheck.get("ok", False):
            validation["status"] = "failed_gpu_precheck"
            validation["finished_at"] = utc_timestamp()
            validation["records"] = []
            write_json(validation_path, validation)
            return 1

        for model in models:
            if model in {"ZITS-GAN", "ZITS-VAE"} and args.action == "benchmark":
                records, output_validations = run_zits_benchmark(model, args, run_id, run_dir, env_info, quality, profiles)
            elif model in {"ZITS-GAN", "ZITS-VAE"}:
                ckpt = checkpoint_for_model(model, args)
                record = manifest_record_for_checkpoint(
                    model,
                    args,
                    run_id,
                    run_dir,
                    env_info,
                    quality,
                    ckpt,
                    extra={"preprocessor": file_info(preprocessor_for_zits_model(model, args), hash_file=not args.skip_checkpoint_hash)},
                )
                records = [record]
                output_validations = []
            elif model == "MIDiff" and args.action == "benchmark":
                records, output_validations = run_midiff_benchmark(args, run_id, run_dir, env_info, quality, profiles)
            elif model == "MIDiff":
                records = [write_midiff_manifest(args, run_id, run_dir, env_info, quality)]
                output_validations = []
            elif model == "ImagenTime" and args.action == "benchmark":
                records, output_validations = run_imagentime_benchmark(args, run_id, run_dir, env_info, quality, profiles)
            elif model == "ImagenTime":
                records = [write_imagentime_manifest(args, run_id, run_dir, env_info, quality)]
                output_validations = []
            else:
                raise ValueError(model)

            for record in records:
                append_jsonl(raw_path, record)
                all_records.append(record)
            validation["output_validations"].extend(output_validations)

        write_summary_csv(summary_path, all_records)
        validation["records"] = [
            {
                "model": record.get("model"),
                "profile": record.get("benchmark_profile"),
                "status": record.get("status"),
                "repeat_idx": record.get("repeat_idx"),
            }
            for record in all_records
        ]
        validation["status"] = "completed"
    except Exception as exc:
        exit_code = 1
        validation["status"] = "failed"
        validation["error"] = f"{type(exc).__name__}: {exc}"
        if all_records:
            with contextlib.suppress(Exception):
                write_summary_csv(summary_path, all_records)
        raise
    finally:
        with contextlib.suppress(Exception):
            post_snapshot = capture_nvidia_smi(run_dir, "after")
        validation["gpu_after"] = post_snapshot
        validation["finished_at"] = utc_timestamp()
        validation["raw_jsonl"] = str(raw_path)
        validation["summary_csv"] = str(summary_path)
        with contextlib.suppress(Exception):
            write_json(validation_path, validation)

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"raw_jsonl={raw_path}")
    print(f"summary_csv={summary_path}")
    print(f"validation_json={validation_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
