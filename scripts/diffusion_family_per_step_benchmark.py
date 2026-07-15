#!/usr/bin/env python3
"""Per-step FP32 efficiency benchmark for diffusion-family baselines.

This script measures a single denoising-network forward, not full sampling.
It writes one JSON record per invocation so each model can be benchmarked in a
fresh process with isolated imports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MIDIFF_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = MIDIFF_ROOT / "scripts"
RESULTS_ROOT = MIDIFF_ROOT / "efficiency_results"
EXTERNAL_ROOT = MIDIFF_ROOT / "external"
IMAGENTIME_ROOT = EXTERNAL_ROOT / "ImagenTime"
DIFFUSION_TS_ROOT = EXTERNAL_ROOT / "Diffusion-TS"
PAD_TS_ROOT = EXTERNAL_ROOT / "PaD-TS"
TIMEAUTODIFF_ROOT = EXTERNAL_ROOT / "TimeAutoDiff"

DEFAULT_MIDIFF_CKPT = MIDIFF_ROOT / "ckpt" / "midiff" / "ema_0.9999_048000.pt"
DEFAULT_DIFFUSION_TS_CONFIG = DIFFUSION_TS_ROOT / "Config" / "our.yaml"
DEFAULT_DIFFUSION_TS_CKPT = DIFFUSION_TS_ROOT / "Checkpoints_our_192" / "checkpoint-10.pt"
DEFAULT_PAD_TS_CKPT = PAD_TS_ROOT / "OUTPUT" / "our_192_MMD" / "model_010000.pt"
DEFAULT_IMAGENTIME_CONFIG = IMAGENTIME_ROOT / "configs" / "unconditional" / "energy.yaml"
DEFAULT_TIMEAUTODIFF_CKPT = TIMEAUTODIFF_ROOT / "output_our" / "timeautodiff_our.pt"

MODEL_NAMES = ("ImagenTime", "MIDiff", "Diffusion-TS", "PaD-TS", "TimeAutoDiff")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(args: Sequence[str], timeout: int = 30) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "args": list(args),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "args": list(args),
            "returncode": 1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.perf_counter() - started,
        }


def precheck_gpu(gpu: Optional[int], max_memory_mb: int, max_utilization_pct: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "requested_gpu": gpu,
        "ok": True,
        "reasons": [],
        "max_memory_mb": max_memory_mb,
        "max_utilization_pct": max_utilization_pct,
    }
    if gpu is None:
        result["status"] = "not_requested"
        return result
    q = run_cmd(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=index,name,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    a = run_cmd(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    result["gpu_query"] = q
    result["compute_query"] = a
    if q["returncode"] != 0:
        result["ok"] = False
        result["reasons"].append(q["stderr"] or "nvidia-smi gpu query failed")
        return result
    parts = [part.strip() for part in q["stdout"].strip().split(",")]
    if len(parts) < 5:
        result["ok"] = False
        result["reasons"].append(f"unexpected nvidia-smi output: {q['stdout']!r}")
        return result
    mem = int(float(parts[3]))
    util = int(float(parts[4]))
    apps = [line for line in a["stdout"].splitlines() if line.strip()]
    result.update({"gpu_index": parts[0], "gpu_name": parts[1], "gpu_uuid": parts[2], "memory_used_mb": mem, "utilization_pct": util, "compute_apps": apps})
    if apps:
        result["ok"] = False
        result["reasons"].append("compute process is already present on requested GPU")
    if mem > max_memory_mb:
        result["ok"] = False
        result["reasons"].append(f"memory.used {mem} MiB exceeds {max_memory_mb} MiB")
    if util > max_utilization_pct:
        result["ok"] = False
        result["reasons"].append(f"utilization {util}% exceeds {max_utilization_pct}%")
    return result


def count_parameters(model: Any) -> Dict[str, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": total - trainable,
    }


def sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def add_path_front(path: Path) -> None:
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def import_torch_after_cuda_env() -> Any:
    import torch  # type: ignore

    torch.set_grad_enabled(False)
    return torch


def build_midiff(args: argparse.Namespace, torch: Any, device: Any) -> Dict[str, Any]:
    add_path_front(SCRIPT_ROOT)
    import dist_util  # type: ignore
    from script_util import create_model_and_diffusion, model_and_diffusion_defaults  # type: ignore

    defaults = model_and_diffusion_defaults()
    model_kwargs = dict(defaults)
    model_kwargs.update(
        {
            "image_size": args.midiff_image_size,
            "num_channels": 128,
            "num_res_blocks": args.midiff_num_res_blocks,
            "diffusion_steps": args.midiff_diffusion_steps,
            "learn_sigma": True,
            "attention_type": args.midiff_attention_type,
        }
    )
    model, diffusion = create_model_and_diffusion(**{k: model_kwargs[k] for k in defaults.keys()})
    state = dist_util.load_state_dict(str(args.midiff_checkpoint), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).float().eval()
    image_size_second = args.midiff_image_size_second

    def forward(batch_size: int) -> Any:
        x = torch.randn(batch_size, 1, args.midiff_image_size, image_size_second, device=device, dtype=torch.float32)
        t = torch.randint(0, int(diffusion.num_timesteps), (batch_size,), device=device)
        t = diffusion._scale_timesteps(t)
        return model(x, t)

    def fvcore_inputs(batch_size: int) -> Tuple[Any, Tuple[Any, Any]]:
        x = torch.randn(batch_size, 1, args.midiff_image_size, image_size_second, device=device, dtype=torch.float32)
        t = torch.randint(0, int(diffusion.num_timesteps), (batch_size,), device=device)
        t = diffusion._scale_timesteps(t)

        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped: Any):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x_in: Any, t_in: Any) -> Any:
                return self.wrapped(x_in, t_in)

        return Wrapper(model).eval(), (x, t)

    return {
        "model": model,
        "forward": forward,
        "fvcore_inputs": fvcore_inputs,
        "checkpoint_path": args.midiff_checkpoint,
        "checkpoint_step": None,
        "scope": "per_step_forward",
        "input_shape": [args.batch_size, 1, args.midiff_image_size, image_size_second],
        "forward_passes_per_sample": int(diffusion.num_timesteps),
        "load_state_missing_keys_count": len(missing),
        "load_state_unexpected_keys_count": len(unexpected),
        "note": "MIDiff one denoising U-Net forward; triplet attention checkpoint.",
    }


def build_imagentime(args: argparse.Namespace, torch: Any, device: Any) -> Dict[str, Any]:
    import argparse as argparse_module

    add_path_front(IMAGENTIME_ROOT)
    old_cwd = Path.cwd()
    os.chdir(IMAGENTIME_ROOT)
    try:
        from omegaconf import OmegaConf  # type: ignore
        from models.model import ImagenTime  # type: ignore
        from models.sampler import DiffusionProcess  # type: ignore
        from utils.utils_data import gen_dataloader  # type: ignore
    finally:
        os.chdir(old_cwd)

    cfg = OmegaConf.to_object(OmegaConf.load(args.imagentime_config))
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
    it_args = argparse_module.Namespace(**payload)
    it_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    it_args.batch_size = args.batch_size
    train_loader, _ = gen_dataloader(it_args)
    model = ImagenTime(args=it_args, device=it_args.device).to(device).float()
    if it_args.use_stft:
        model.init_stft_embedder(train_loader)
    else:
        _ = model.ts_to_img(next(iter(train_loader))[0].to(device))
    checkpoint_path = args.imagentime_checkpoint
    checkpoint_step = None
    scope = "architecture_only_per_step_random_weights"
    if checkpoint_path and checkpoint_path.exists():
        loaded = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(loaded.get("model", loaded), strict=False)
        if "ema_model" in loaded and getattr(model, "model_ema", None) is not None:
            model.model_ema.load_state_dict(loaded["ema_model"])
        checkpoint_step = loaded.get("epoch")
        scope = "per_step_forward"
    model.eval()
    process = DiffusionProcess(
        it_args,
        model.net,
        (it_args.input_channels, it_args.img_resolution, it_args.img_resolution),
    )
    forward_calls = max(1, 2 * int(it_args.diffusion_steps) - 1)

    def forward(batch_size: int) -> Any:
        x = torch.randn(batch_size, it_args.input_channels, it_args.img_resolution, it_args.img_resolution, device=device, dtype=torch.float32)
        sigma = torch.full((batch_size,), float(process.net.sigma_max), device=device, dtype=torch.float32)
        with model.ema_scope():
            return model.net(x, sigma, None)

    def fvcore_inputs(batch_size: int) -> Tuple[Any, Tuple[Any, Any]]:
        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped: Any):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x_in: Any, sigma_in: Any) -> Any:
                return self.wrapped(x_in, sigma_in, None)

        x = torch.randn(batch_size, it_args.input_channels, it_args.img_resolution, it_args.img_resolution, device=device, dtype=torch.float32)
        sigma = torch.full((batch_size,), float(process.net.sigma_max), device=device, dtype=torch.float32)
        return Wrapper(model.net).eval(), (x, sigma)

    return {
        "model": model,
        "forward": forward,
        "fvcore_inputs": fvcore_inputs,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "scope": scope,
        "input_shape": [args.batch_size, it_args.input_channels, it_args.img_resolution, it_args.img_resolution],
        "forward_passes_per_sample": forward_calls,
        "note": "ImagenTime default architecture; checkpoint weights are optional and do not change the compute graph.",
    }


def build_diffusion_ts(args: argparse.Namespace, torch: Any, device: Any) -> Dict[str, Any]:
    add_path_front(DIFFUSION_TS_ROOT)
    from Utils.io_utils import instantiate_from_config, load_yaml_config  # type: ignore

    config = load_yaml_config(str(args.diffusion_ts_config))
    model = instantiate_from_config(config["model"])
    checkpoint = torch.load(str(args.diffusion_ts_checkpoint), map_location="cpu")
    ema_state = {
        key[len("ema_model.") :]: value
        for key, value in checkpoint.get("ema", {}).items()
        if key.startswith("ema_model.")
    }
    state_source = "ema_model" if ema_state else "model"
    missing, unexpected = model.load_state_dict(ema_state or checkpoint["model"], strict=False)
    model.to(device).float().eval()
    seq_len = int(config["model"]["params"]["seq_length"])
    feature_size = int(config["model"]["params"]["feature_size"])
    forward_calls = int(config["model"]["params"].get("sampling_timesteps") or config["model"]["params"]["timesteps"])

    def forward(batch_size: int) -> Any:
        x = torch.randn(batch_size, seq_len, feature_size, device=device, dtype=torch.float32)
        t = torch.randint(0, int(model.num_timesteps), (batch_size,), device=device)
        return model.output(x, t)

    def fvcore_inputs(batch_size: int) -> Tuple[Any, Tuple[Any, Any]]:
        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped: Any):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x_in: Any, t_in: Any) -> Any:
                return self.wrapped.output(x_in, t_in)

        x = torch.randn(batch_size, seq_len, feature_size, device=device, dtype=torch.float32)
        t = torch.randint(0, int(model.num_timesteps), (batch_size,), device=device)
        return Wrapper(model).eval(), (x, t)

    return {
        "model": model,
        "forward": forward,
        "fvcore_inputs": fvcore_inputs,
        "checkpoint_path": args.diffusion_ts_checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "scope": "per_step_forward",
        "input_shape": [args.batch_size, seq_len, feature_size],
        "forward_passes_per_sample": forward_calls,
        "load_state_missing_keys_count": len(missing),
        "load_state_unexpected_keys_count": len(unexpected),
        "note": f"Diffusion-TS one denoising Transformer forward; loaded {state_source} weights; full sampling uses configured fast sampling steps.",
    }


def build_pad_ts(args: argparse.Namespace, torch: Any, device: Any) -> Dict[str, Any]:
    add_path_front(PAD_TS_ROOT)
    from Model import PaD_TS  # type: ignore
    from configs.our_config import Diffusion_args, Model_args  # type: ignore
    from diffmodel_init import create_gaussian_diffusion  # type: ignore

    model_arg = Model_args()
    diff_arg = Diffusion_args()
    model = PaD_TS(
        hidden_size=model_arg.hidden_size,
        num_heads=model_arg.num_heads,
        n_encoder=model_arg.n_encoder,
        n_decoder=model_arg.n_decoder,
        feature_last=model_arg.feature_last,
        mlp_ratio=model_arg.mlp_ratio,
        input_shape=model_arg.input_shape,
    )
    checkpoint = torch.load(str(args.pad_ts_checkpoint), map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).float().eval()
    diffusion = create_gaussian_diffusion(
        predict_xstart=diff_arg.predict_xstart,
        diffusion_steps=diff_arg.diffusion_steps,
        noise_schedule=diff_arg.noise_schedule,
        loss=diff_arg.loss,
        rescale_timesteps=diff_arg.rescale_timesteps,
    )
    seq_len, feature_size = model_arg.input_shape

    def forward(batch_size: int) -> Any:
        x = torch.randn(batch_size, seq_len, feature_size, device=device, dtype=torch.float32)
        t = torch.randint(0, int(diffusion.num_timesteps), (batch_size,), device=device)
        return model(x, t)

    def fvcore_inputs(batch_size: int) -> Tuple[Any, Tuple[Any, Any]]:
        x = torch.randn(batch_size, seq_len, feature_size, device=device, dtype=torch.float32)
        t = torch.randint(0, int(diffusion.num_timesteps), (batch_size,), device=device)
        return model, (x, t)

    return {
        "model": model,
        "forward": forward,
        "fvcore_inputs": fvcore_inputs,
        "checkpoint_path": args.pad_ts_checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "scope": "per_step_forward",
        "input_shape": [args.batch_size, seq_len, feature_size],
        "forward_passes_per_sample": int(diffusion.num_timesteps),
        "load_state_missing_keys_count": len(missing),
        "load_state_unexpected_keys_count": len(unexpected),
        "note": "PaD-TS one denoising Transformer forward; checkpoint from OUTPUT/our_192_MMD.",
    }


def build_timeautodiff(args: argparse.Namespace, torch: Any, device: Any) -> Dict[str, Any]:
    add_path_front(TIMEAUTODIFF_ROOT)
    import DIFF as diff  # type: ignore
    import VAE as vae  # type: ignore

    checkpoint = torch.load(str(args.timeautodiff_checkpoint), map_location=device, weights_only=False)
    saved_args = checkpoint["args"]
    parser = checkpoint["parser"]
    info = parser.datatype_info()
    seq_len = int(saved_args["seq_len"])
    latent_dim = int(saved_args["lat_dim"])
    time_dim = 8
    encoded_dim = int(checkpoint["vae_state_dict"]["Emb.mlp_output.2.weight"].shape[0])

    ae = vae.DeapStack(
        saved_args["channels"],
        info["n_bins"],
        info["n_cats"],
        info["n_nums"],
        info["cards"],
        encoded_dim,
        saved_args["hidden_size"],
        saved_args["vae_num_layers"],
        saved_args["emb_dim"],
        time_dim,
        latent_dim,
    ).to(device)
    ae.load_state_dict(checkpoint["vae_state_dict"])
    ae.float().eval()

    diffusion_steps = int(saved_args["diffusion_steps"])
    diff.diffusion_steps = diffusion_steps
    diff.betas = diff.get_betas(diffusion_steps)
    diff.alphas = torch.cumprod(1 - diff.betas, dim=0)
    diff_model = diff.BiRNN_score(
        latent_dim,
        encoded_dim,
        saved_args["diff_hidden_dim"],
        saved_args["diff_num_layers"],
        diffusion_steps,
        time_dim,
        saved_args["emb_dim"],
        info["n_bins"],
        info["n_cats"],
        info["n_nums"],
        info["cards"],
    ).to(device)
    diff_model.load_state_dict(checkpoint["diff_state_dict"])
    diff_model.float().eval()

    class TimeAutoDiffSamplingModel(torch.nn.Module):
        def __init__(self, autoencoder: Any, score_model: Any):
            super().__init__()
            self.autoencoder = autoencoder
            self.score_model = score_model

        def forward(self, x_in: Any, t_in: Any, i_in: Any, time_info_in: Any, cond_in: Any, target_mask_in: Any) -> Any:
            return self.score_model(x_in, t_in, i_in, time_info_in, cond_in, target_mask_in)

    model = TimeAutoDiffSamplingModel(ae, diff_model).to(device).eval()

    def time_info_tensor(batch_size: int) -> Any:
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        periods = [seq_len, max(seq_len // 2, 1), max(seq_len // 4, 1), 24]
        features = []
        for period in periods:
            angle = 2 * math.pi * t / float(period)
            features.append(torch.sin(angle))
            features.append(torch.cos(angle))
        base = torch.stack(features, dim=1)
        return base.unsqueeze(0).repeat(batch_size, 1, 1)

    def inputs(batch_size: int) -> Tuple[Any, Any, Any, Any, Any, Any]:
        x = torch.randn(batch_size, seq_len, latent_dim, device=device, dtype=torch.float32)
        t_grid = torch.linspace(0, 1, seq_len, device=device, dtype=torch.float32).view(1, -1, 1).repeat(batch_size, 1, 1)
        i = torch.full((batch_size, seq_len, 1), diffusion_steps - 1, device=device, dtype=torch.float32)
        time_info = time_info_tensor(batch_size)
        cond = torch.zeros(batch_size, seq_len, encoded_dim, device=device, dtype=torch.float32)
        target_mask = torch.ones(batch_size, seq_len, encoded_dim, device=device, dtype=torch.float32)
        return x, t_grid, i, time_info, cond, target_mask

    def forward(batch_size: int) -> Any:
        return model(*inputs(batch_size))

    def fvcore_inputs(batch_size: int) -> Tuple[Any, Tuple[Any, Any, Any, Any, Any, Any]]:
        return model, inputs(batch_size)

    return {
        "model": model,
        "forward": forward,
        "fvcore_inputs": fvcore_inputs,
        "checkpoint_path": args.timeautodiff_checkpoint,
        "checkpoint_step": None,
        "scope": "per_step_forward",
        "input_shape": [args.batch_size, seq_len, latent_dim],
        "forward_passes_per_sample": diffusion_steps,
        "note": "TimeAutoDiff one diffusion score-model forward; VAE and diffusion modules are loaded for parameter/memory accounting, while per-step FLOPs/latency measure the denoising score call.",
    }


def measure_fvcore(torch: Any, build: Mapping[str, Any], batch_size: int) -> Dict[str, Any]:
    if not build.get("fvcore_inputs"):
        return {"flops_per_forward": None, "flops_source": None, "flops_error": "no fvcore inputs"}
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        wrapper, inputs = build["fvcore_inputs"](batch_size)
        with torch.no_grad():
            total = float(FlopCountAnalysis(wrapper.eval(), inputs).total())
        return {"flops_per_forward": total, "flops_source": "fvcore FlopCountAnalysis", "flops_error": None}
    except Exception as exc:
        return {"flops_per_forward": None, "flops_source": None, "flops_error": f"{type(exc).__name__}: {exc}"}


def measure_profiler(torch: Any, forward_fn: Callable[[int], Any], batch_size: int) -> Dict[str, Any]:
    try:
        from torch.profiler import ProfilerActivity, profile  # type: ignore

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        clear_cuda(torch)
        sync_cuda(torch)
        with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with torch.no_grad():
                out = forward_fn(batch_size)
                if isinstance(out, (tuple, list)):
                    loss = sum((item.float().sum() for item in out if hasattr(item, "float")), torch.tensor(0.0, device=out[0].device))
                elif hasattr(out, "float"):
                    loss = out.float().sum()
                else:
                    loss = torch.tensor(0.0)
                _ = loss.detach()
        sync_cuda(torch)
        total = float(sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages()))
        return {"flops_per_forward": total if total > 0 else None, "flops_source": "torch.profiler with_flops=True", "flops_error": None if total > 0 else "profiler returned zero flops"}
    except Exception as exc:
        return {"flops_per_forward": None, "flops_source": None, "flops_error": f"{type(exc).__name__}: {exc}"}


def choose_flops(torch: Any, build: Mapping[str, Any], batch_size: int, prefer_profiler: bool) -> Dict[str, Any]:
    if prefer_profiler:
        first = measure_profiler(torch, build["forward"], batch_size)
        if first.get("flops_per_forward") is not None:
            return first
        second = measure_fvcore(torch, build, batch_size)
        second["flops_error"] = f"profiler failed: {first.get('flops_error')}; fvcore: {second.get('flops_error')}"
        return second
    first = measure_fvcore(torch, build, batch_size)
    if first.get("flops_per_forward") is not None:
        return first
    second = measure_profiler(torch, build["forward"], batch_size)
    second["flops_error"] = f"fvcore failed: {first.get('flops_error')}; profiler: {second.get('flops_error')}"
    return second


def benchmark_one(args: argparse.Namespace) -> Dict[str, Any]:
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch = import_torch_after_cuda_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to mix CPU timing into GPU benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    builders = {
        "MIDiff": build_midiff,
        "ImagenTime": build_imagentime,
        "Diffusion-TS": build_diffusion_ts,
        "PaD-TS": build_pad_ts,
        "TimeAutoDiff": build_timeautodiff,
    }
    load_start = time.perf_counter()
    build = builders[args.model](args, torch, device)
    sync_cuda(torch)
    load_time = time.perf_counter() - load_start
    model = build["model"]
    params = count_parameters(model)
    forward_fn = build["forward"]

    for _ in range(args.warmup_runs):
        sync_cuda(torch)
        with torch.no_grad():
            _ = forward_fn(args.batch_size)
        sync_cuda(torch)

    repeat_times: List[float] = []
    peak_allocs: List[float] = []
    peak_reserved: List[float] = []
    for _ in range(args.repeats):
        clear_cuda(torch)
        sync_cuda(torch)
        start = time.perf_counter()
        with torch.no_grad():
            _ = forward_fn(args.batch_size)
        sync_cuda(torch)
        elapsed = time.perf_counter() - start
        repeat_times.append(elapsed)
        peak_allocs.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        peak_reserved.append(torch.cuda.max_memory_reserved() / (1024 * 1024))

    prefer_profiler = args.model == "ImagenTime"
    flops = choose_flops(torch, build, args.batch_size, prefer_profiler=prefer_profiler)
    flops_per_forward = flops.get("flops_per_forward")
    flops_per_sample = float(flops_per_forward) / args.batch_size if flops_per_forward is not None else None
    latency_sec = sum(repeat_times) / len(repeat_times)
    latency_std_sec = math.sqrt(sum((x - latency_sec) ** 2 for x in repeat_times) / (len(repeat_times) - 1)) if len(repeat_times) > 1 else 0.0

    record = {
        "model": args.model,
        "precision": "FP32",
        "batch_size": args.batch_size,
        "scope": build["scope"],
        "status": "ok",
        "device": str(device),
        "physical_gpu": args.gpu,
        "torch_version": torch.__version__,
        "checkpoint_path": str(build.get("checkpoint_path")) if build.get("checkpoint_path") else "",
        "checkpoint_exists": bool(build.get("checkpoint_path") and Path(build["checkpoint_path"]).exists()),
        "checkpoint_step": build.get("checkpoint_step"),
        "input_shape": build.get("input_shape"),
        "warmup_runs": args.warmup_runs,
        "measured_repeats": args.repeats,
        "model_load_time_sec": load_time,
        "latency_sec_per_forward_mean": latency_sec,
        "latency_sec_per_forward_std": latency_std_sec,
        "latency_ms_per_forward": latency_sec * 1000.0,
        "latency_ms_per_sample": latency_sec * 1000.0 / args.batch_size,
        "FPS": args.batch_size / latency_sec,
        "peak_memory_allocated_MB": max(peak_allocs) if peak_allocs else None,
        "peak_memory_reserved_MB": max(peak_reserved) if peak_reserved else None,
        "forward_calls_per_sample": build.get("forward_passes_per_sample"),
        "estimated_full_latency_ms_per_sample": latency_sec * 1000.0 * float(build.get("forward_passes_per_sample") or 1) / args.batch_size,
        "GFLOPs_per_forward": float(flops_per_forward) / 1e9 if flops_per_forward is not None else None,
        "GFLOPs_per_sample": float(flops_per_sample) / 1e9 if flops_per_sample is not None else None,
        "estimated_GFLOPs_per_generated_sample": float(flops_per_sample) * float(build.get("forward_passes_per_sample") or 1) / 1e9 if flops_per_sample is not None else None,
        "GFLOPs_source": flops.get("flops_source"),
        "GFLOPs_error": flops.get("flops_error"),
        "parameters_M": params["total_parameter_count"] / 1e6,
        "trainable_parameters_M": params["trainable_parameter_count"] / 1e6,
        "frozen_parameters_M": params["frozen_parameter_count"] / 1e6,
        "note": build.get("note"),
    }
    for key in ("load_state_missing_keys_count", "load_state_unexpected_keys_count"):
        if key in build:
            record[key] = build[key]
    return record


def compact_rows(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.append(
            {
                "Model": rec.get("model"),
                "Precision": rec.get("precision"),
                "Batch Size": rec.get("batch_size"),
                "Scope": rec.get("scope"),
                "Params (M)": rec.get("parameters_M"),
                "Trainable Params (M)": rec.get("trainable_parameters_M"),
                "GFLOPs": rec.get("GFLOPs_per_forward"),
                "GFLOPs / sample": rec.get("GFLOPs_per_sample"),
                "Latency (ms)": rec.get("latency_ms_per_forward"),
                "Latency / sample (ms)": rec.get("latency_ms_per_sample"),
                "FPS": rec.get("FPS"),
                "Peak Mem. (MB)": rec.get("peak_memory_allocated_MB"),
                "Forward Calls / Sample": rec.get("forward_calls_per_sample"),
                "Est. Full Latency / Sample (ms)": rec.get("estimated_full_latency_ms_per_sample"),
                "GFLOPs Source": rec.get("GFLOPs_source"),
                "Checkpoint": rec.get("checkpoint_path"),
                "Note": rec.get("note"),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(args: argparse.Namespace) -> int:
    records = [read_json(path) for path in sorted(args.run_dir.glob("*.json")) if path.name not in {"gpu_precheck.json", "aggregate.json"}]
    records = [rec for rec in records if rec.get("status") == "ok"]
    full_path = args.run_dir / "diffusion_family_per_step_full.csv"
    compact_path = args.run_dir / "diffusion_family_per_step_compact.csv"
    write_csv(full_path, records)
    write_csv(compact_path, compact_rows(records))
    write_json(args.run_dir / "aggregate.json", {"records": len(records), "full_csv": str(full_path), "compact_csv": str(compact_path)})
    print(f"full_csv={full_path}")
    print(f"compact_csv={compact_path}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="benchmark one model and write a JSON record")
    bench.add_argument("--model", choices=MODEL_NAMES, required=True)
    bench.add_argument("--batch-size", type=int, required=True)
    bench.add_argument("--gpu", type=int, default=3)
    bench.add_argument("--run-dir", type=Path, required=True)
    bench.add_argument("--output-json", type=Path, default=None)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--warmup-runs", type=int, default=3)
    bench.add_argument("--repeats", type=int, default=10)
    bench.add_argument("--gpu-max-memory-mb", type=int, default=512)
    bench.add_argument("--gpu-max-utilization-pct", type=int, default=10)
    bench.add_argument("--midiff-checkpoint", type=Path, default=DEFAULT_MIDIFF_CKPT)
    bench.add_argument("--midiff-image-size", type=int, default=256)
    bench.add_argument("--midiff-image-size-second", type=int, default=160)
    bench.add_argument("--midiff-diffusion-steps", type=int, default=1000)
    bench.add_argument("--midiff-num-res-blocks", type=int, default=3)
    bench.add_argument("--midiff-attention-type", default="triple")
    bench.add_argument("--imagentime-config", type=Path, default=DEFAULT_IMAGENTIME_CONFIG)
    bench.add_argument("--imagentime-checkpoint", type=Path, default=None)
    bench.add_argument("--diffusion-ts-config", type=Path, default=DEFAULT_DIFFUSION_TS_CONFIG)
    bench.add_argument("--diffusion-ts-checkpoint", type=Path, default=DEFAULT_DIFFUSION_TS_CKPT)
    bench.add_argument("--pad-ts-checkpoint", type=Path, default=DEFAULT_PAD_TS_CKPT)
    bench.add_argument("--timeautodiff-checkpoint", type=Path, default=DEFAULT_TIMEAUTODIFF_CKPT)

    agg = sub.add_parser("aggregate", help="aggregate JSON records under run-dir into CSV")
    agg.add_argument("--run-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "bench":
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        if args.warmup_runs < 0:
            parser.error("--warmup-runs cannot be negative")
        if args.repeats <= 0:
            parser.error("--repeats must be positive")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        if args.output_json is None:
            safe_model = args.model.lower().replace("-", "_")
            args.output_json = args.run_dir / f"{safe_model}_b{args.batch_size}.json"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "aggregate":
        return aggregate(args)

    precheck = precheck_gpu(args.gpu, args.gpu_max_memory_mb, args.gpu_max_utilization_pct)
    write_json(args.run_dir / "gpu_precheck.json", precheck)
    if not precheck.get("ok"):
        print(json.dumps(precheck, indent=2), file=sys.stderr)
        return 1

    record = benchmark_one(args)
    write_json(args.output_json, record)
    print(f"output_json={args.output_json}")
    print(json.dumps(record, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
