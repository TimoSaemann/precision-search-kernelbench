import csv
import importlib.util
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_module_from_path(file_path: str):
    spec = importlib.util.spec_from_file_location("kb_module", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unsupported dtype: {dtype}")


def name_to_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


@torch.no_grad()
def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item()


@torch.no_grad()
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@torch.no_grad()
def measure_latency_ms_cuda(
    model: nn.Module,
    example_input: torch.Tensor,
    num_warmup: int,
    num_iters: int,
) -> float:
    model.eval().cuda()
    x = example_input.detach().clone()

    for _ in range(num_warmup):
        _ = model(x)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        _ = model(x)
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iters


def convert_input_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(device="cuda", dtype=dtype)


def make_model(module) -> nn.Module:
    init_inputs = module.get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        raise TypeError("get_init_inputs() must return a list or tuple")
    return module.Model(*init_inputs)


def make_example_input(module) -> torch.Tensor:
    inputs = module.get_inputs()
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("get_inputs() must return a list or tuple")
    if len(inputs) != 1:
        raise ValueError("This harness currently supports only single-input models")

    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("Only tensor inputs are currently supported")
    if not x.is_floating_point():
        raise TypeError("Only floating-point tensor inputs are currently supported")

    return x.to("cuda")


def collect_model_files(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]

    files = sorted(str(x) for x in p.glob("*.py") if x.is_file())
    if not files:
        raise FileNotFoundError(f"No .py files found in: {path}")
    return files


def print_model_result(result) -> None:
    print(f"\n{'=' * 80}")
    print(f"Model: {result.model_name}")
    print(f"File : {result.model_file}")
    print(f"Input: {result.input_shape}")
    print(f"Eager latency (ms)              : {result.eager_latency_ms}")
    print(f"Compile FP32 latency (ms)       : {result.compile_fp32_latency_ms}")
    print(f"Best valid low-precision variant: {result.best_valid_lowp_variant}")
    print(f"Best valid lowp latency (ms)    : {result.best_valid_lowp_latency_ms}")
    print(f"Speedup over eager from lowp    : {result.speedup_over_eager_from_lowp:.4f}x")
    print(f"Speedup over compile from lowp  : {result.speedup_over_compile_from_lowp:.4f}x")
    print(f"Best valid overall variant      : {result.best_valid_overall_variant}")
    print(f"Best valid overall latency (ms) : {result.best_valid_overall_latency_ms}")

    for v in result.variants:
        print(f"\n[{v['name']}]")
        print(f"  build_ok          : {v['build_ok']}")
        print(f"  run_ok            : {v['run_ok']}")
        print(f"  valid             : {v['valid']}")
        print(f"  latency_ms        : {v['latency_ms']}")
        print(f"  speedup_vs_eager  : {v['speedup_vs_eager']}")
        print(f"  mean_abs_diff     : {v['mean_abs_diff']}")
        print(f"  max_abs_diff      : {v['max_abs_diff']}")
        print(f"  error             : {v['error']}")


def write_json(results: list[Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def write_csv(results: list[Any], output_path: str) -> None:
    rows = []
    for r in results:
        row = {
            "model_name": r.model_name,
            "model_file": r.model_file,
            "input_shape": str(r.input_shape),
            "batch_size": r.batch_size,
            "eager_latency_ms": r.eager_latency_ms,
            "compile_fp32_latency_ms": r.compile_fp32_latency_ms,
            "best_valid_lowp_variant": r.best_valid_lowp_variant,
            "best_valid_lowp_latency_ms": r.best_valid_lowp_latency_ms,
            "speedup_over_eager_from_lowp": r.speedup_over_eager_from_lowp,
            "speedup_over_compile_from_lowp": r.speedup_over_compile_from_lowp,
            "best_valid_overall_variant": r.best_valid_overall_variant,
            "best_valid_overall_latency_ms": r.best_valid_overall_latency_ms,
        }
        for v in r.variants:
            prefix = v["name"]
            row[f"{prefix}_valid"] = v["valid"]
            row[f"{prefix}_latency_ms"] = v["latency_ms"]
            row[f"{prefix}_speedup_vs_eager"] = v["speedup_vs_eager"]
            row[f"{prefix}_mean_abs_diff"] = v["mean_abs_diff"]
            row[f"{prefix}_max_abs_diff"] = v["max_abs_diff"]
            row[f"{prefix}_error"] = v["error"]
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
