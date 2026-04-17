import argparse
import copy
import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

from helpers import (
    collect_model_files,
    convert_input_dtype,
    dtype_to_name,
    load_module_from_path,
    make_example_input,
    make_model,
    max_abs_diff,
    mean_abs_diff,
    measure_latency_ms_cuda,
    name_to_dtype,
    print_model_result,
    summarize_exception,
    write_json,
)
from trt_builders import (
    TRT_VARIANT_TIMEOUT_SECONDS,
    build_trt_fp16_model,
    build_trt_bf16_model,
    build_trt_int8_model,
    build_trt_fp8_model,
)

DEFAULT_WARMUP_ITERS = 10
DEFAULT_BENCH_ITERS = 50
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2

VARIANT_NAMES = [
    "eager_fp32",
    "compile_fp32",
    "trt_fp16",
    "trt_bf16",
    "trt_fp8",
    "trt_int8",
]

TRT_VARIANT_SPECS = [
    ("trt_fp16", torch.float16),
    ("trt_bf16", torch.bfloat16),
    ("trt_fp8", torch.float32),
    ("trt_int8", torch.float32),
]

LOWP_VARIANT_NAMES = {"trt_fp16", "trt_bf16", "trt_fp8", "trt_int8"}


@dataclass
class VariantResult:
    name: str
    build_ok: bool
    run_ok: bool
    valid: bool
    latency_ms: float | None
    mean_abs_diff: float | None
    max_abs_diff: float | None
    speedup_vs_eager: float | None
    error: str | None


@dataclass
class ModelResult:
    model_file: str
    model_name: str
    input_shape: list[Any]
    batch_size: int
    eager_latency_ms: float | None
    compile_fp32_latency_ms: float | None
    best_valid_lowp_variant: str | None
    best_valid_lowp_latency_ms: float | None
    speedup_over_eager_from_lowp: float
    speedup_over_compile_from_lowp: float
    best_valid_overall_variant: str | None
    best_valid_overall_latency_ms: float | None
    variants: list[dict[str, Any]]


def make_failed_variant(name: str, error: str) -> dict[str, Any]:
    return asdict(
        VariantResult(
            name=name,
            build_ok=False,
            run_ok=False,
            valid=False,
            latency_ms=None,
            mean_abs_diff=None,
            max_abs_diff=None,
            speedup_vs_eager=1.0,
            error=error,
        )
    )


def make_failed_model_result(file_path: str, error: str) -> ModelResult:
    return ModelResult(
        model_file=file_path,
        model_name=Path(file_path).stem,
        input_shape=[],
        batch_size=0,
        eager_latency_ms=None,
        compile_fp32_latency_ms=None,
        best_valid_lowp_variant=None,
        best_valid_lowp_latency_ms=None,
        speedup_over_eager_from_lowp=1.0,
        speedup_over_compile_from_lowp=1.0,
        best_valid_overall_variant=None,
        best_valid_overall_latency_ms=None,
        variants=[make_failed_variant(name, error) for name in VARIANT_NAMES],
    )


def _evaluate_trt_variant_worker(
    file_path: str,
    name: str,
    num_warmup: int,
    num_iters: int,
    atol: float,
    rtol: float,
    calib_batch_size: int,
    num_calib_batches: int,
    run_dtype_name: str,
    queue,
):
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        module = load_module_from_path(file_path)
        fp32_model = make_model(module).eval().cuda()
        example_input = make_example_input(module)

        if example_input.ndim < 1:
            raise ValueError("Input tensor must have a batch dimension")

        input_shape_wo_batch = tuple(example_input.shape[1:])

        with torch.no_grad():
            ref_output = fp32_model(example_input)

        build_map = {
            "trt_fp16": build_trt_fp16_model,
            "trt_bf16": build_trt_bf16_model,
            "trt_fp8": build_trt_fp8_model,
            "trt_int8": build_trt_int8_model,
        }
        build_fn = build_map[name]
        run_dtype = name_to_dtype(run_dtype_name)
        batch_size = int(example_input.shape[0])

        if name in {"trt_int8", "trt_fp8"}:
            model_variant = build_fn(
                fp32_model=fp32_model,
                input_shape_wo_batch=input_shape_wo_batch,
                batch_size_eval=batch_size,
                calib_batch_size=calib_batch_size,
                num_calib_batches=num_calib_batches,
                example_input=example_input,
            )
        else:
            model_variant = build_fn(
                fp32_model=fp32_model,
                input_shape_wo_batch=input_shape_wo_batch,
                batch_size_eval=batch_size,
            )

        x = convert_input_dtype(example_input, run_dtype)

        with torch.no_grad():
            out = model_variant(x)

        out_fp32 = out.float()
        ref_fp32 = ref_output.float()

        mad = mean_abs_diff(ref_fp32, out_fp32)
        maxd = max_abs_diff(ref_fp32, out_fp32)
        valid = torch.allclose(ref_fp32, out_fp32, atol=atol, rtol=rtol)

        latency = measure_latency_ms_cuda(
            model_variant,
            x,
            num_warmup=num_warmup,
            num_iters=num_iters,
        )

        queue.put(
            {
                "ok": True,
                "valid": bool(valid),
                "latency_ms": latency,
                "mean_abs_diff": mad,
                "max_abs_diff": maxd,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "error": summarize_exception(exc),
            }
        )


def evaluate_trt_variant_with_timeout(
    file_path: str,
    name: str,
    eager_latency_ms: float,
    num_warmup: int,
    num_iters: int,
    atol: float,
    rtol: float,
    calib_batch_size: int,
    num_calib_batches: int,
    run_dtype: torch.dtype,
    timeout_seconds: int = TRT_VARIANT_TIMEOUT_SECONDS,
) -> VariantResult:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    proc = ctx.Process(
        target=_evaluate_trt_variant_worker,
        args=(
            file_path,
            name,
            num_warmup,
            num_iters,
            atol,
            rtol,
            calib_batch_size,
            num_calib_batches,
            dtype_to_name(run_dtype),
            queue,
        ),
    )

    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return VariantResult(
            name=name,
            build_ok=False,
            run_ok=False,
            valid=False,
            latency_ms=None,
            mean_abs_diff=None,
            max_abs_diff=None,
            speedup_vs_eager=1.0,
            error=f"timeout after {timeout_seconds}s",
        )

    if queue.empty():
        return VariantResult(
            name=name,
            build_ok=False,
            run_ok=False,
            valid=False,
            latency_ms=None,
            mean_abs_diff=None,
            max_abs_diff=None,
            speedup_vs_eager=1.0,
            error="worker exited without returning a result",
        )

    payload = queue.get()

    if not payload["ok"]:
        return VariantResult(
            name=name,
            build_ok=False,
            run_ok=False,
            valid=False,
            latency_ms=None,
            mean_abs_diff=None,
            max_abs_diff=None,
            speedup_vs_eager=1.0,
            error=payload["error"],
        )

    latency = payload["latency_ms"]
    return VariantResult(
        name=name,
        build_ok=True,
        run_ok=True,
        valid=payload["valid"],
        latency_ms=latency,
        mean_abs_diff=payload["mean_abs_diff"],
        max_abs_diff=payload["max_abs_diff"],
        speedup_vs_eager=(eager_latency_ms / latency) if latency else 1.0,
        error=None,
    )


def evaluate_model_file(
    file_path: str,
    num_warmup: int,
    num_iters: int,
    atol: float,
    rtol: float,
    calib_batch_size: int,
    num_calib_batches: int,
) -> ModelResult:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    module = load_module_from_path(file_path)
    model_name = Path(file_path).stem

    fp32_model = make_model(module).eval().cuda()
    compiled_fp32_model = torch.compile(copy.deepcopy(fp32_model).eval().cuda())
    example_input = make_example_input(module)

    if example_input.ndim < 1:
        raise ValueError("Input tensor must have a batch dimension")

    batch_size = int(example_input.shape[0])

    with torch.no_grad():
        ref_output = fp32_model(example_input)
        compiled_output = compiled_fp32_model(example_input)

    eager_latency_ms = measure_latency_ms_cuda(
        fp32_model,
        example_input,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    compiled_mad = mean_abs_diff(ref_output.float(), compiled_output.float())
    compiled_maxd = max_abs_diff(ref_output.float(), compiled_output.float())
    compiled_valid = torch.allclose(
        ref_output.float(),
        compiled_output.float(),
        atol=atol,
        rtol=rtol,
    )
    compiled_latency = measure_latency_ms_cuda(
        compiled_fp32_model,
        example_input,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    results: list[VariantResult] = [
        VariantResult(
            name="eager_fp32",
            build_ok=True,
            run_ok=True,
            valid=True,
            latency_ms=eager_latency_ms,
            mean_abs_diff=0.0,
            max_abs_diff=0.0,
            speedup_vs_eager=1.0,
            error=None,
        ),
        VariantResult(
            name="compile_fp32",
            build_ok=True,
            run_ok=True,
            valid=bool(compiled_valid),
            latency_ms=compiled_latency,
            mean_abs_diff=compiled_mad,
            max_abs_diff=compiled_maxd,
            speedup_vs_eager=(eager_latency_ms / compiled_latency) if compiled_latency else None,
            error=None,
        ),
    ]

    for name, run_dtype in TRT_VARIANT_SPECS:
        result = evaluate_trt_variant_with_timeout(
            file_path=file_path,
            name=name,
            eager_latency_ms=eager_latency_ms,
            num_warmup=num_warmup,
            num_iters=num_iters,
            atol=atol,
            rtol=rtol,
            calib_batch_size=calib_batch_size,
            num_calib_batches=num_calib_batches,
            run_dtype=run_dtype,
            timeout_seconds=TRT_VARIANT_TIMEOUT_SECONDS,
        )
        results.append(result)

    valid_lowp_results = [
        r for r in results
        if r.name in LOWP_VARIANT_NAMES and r.valid and r.latency_ms is not None
    ]
    best_valid_lowp = min(valid_lowp_results, key=lambda r: r.latency_ms) if valid_lowp_results else None

    compile_result = next(r for r in results if r.name == "compile_fp32")

    valid_overall_results = [r for r in results if r.valid and r.latency_ms is not None]
    best_valid_overall = min(valid_overall_results, key=lambda r: r.latency_ms) if valid_overall_results else None

    speedup_over_eager_from_lowp = (
        eager_latency_ms / best_valid_lowp.latency_ms
        if best_valid_lowp is not None and best_valid_lowp.latency_ms is not None
        else 1.0
    )

    speedup_over_compile_from_lowp = (
        compile_result.latency_ms / best_valid_lowp.latency_ms
        if (
            best_valid_lowp is not None
            and best_valid_lowp.latency_ms is not None
            and compile_result.latency_ms is not None
            and best_valid_lowp.latency_ms < compile_result.latency_ms
        )
        else 1.0
    )

    return ModelResult(
        model_file=file_path,
        model_name=model_name,
        input_shape=list(example_input.shape),
        batch_size=batch_size,
        eager_latency_ms=eager_latency_ms,
        compile_fp32_latency_ms=compile_result.latency_ms,
        best_valid_lowp_variant=best_valid_lowp.name if best_valid_lowp else None,
        best_valid_lowp_latency_ms=best_valid_lowp.latency_ms if best_valid_lowp else None,
        speedup_over_eager_from_lowp=speedup_over_eager_from_lowp,
        speedup_over_compile_from_lowp=speedup_over_compile_from_lowp,
        best_valid_overall_variant=best_valid_overall.name if best_valid_overall else None,
        best_valid_overall_latency_ms=best_valid_overall.latency_ms if best_valid_overall else None,
        variants=[asdict(r) for r in results],
    )


def print_summary(results: list[ModelResult]) -> None:
    if not results:
        return

    num_models = len(results)
    num_valid_lowp = sum(r.best_valid_lowp_variant is not None for r in results)
    num_beat_eager = sum(r.speedup_over_eager_from_lowp > 1.0 for r in results)
    num_beat_compile = sum(r.speedup_over_compile_from_lowp > 1.0 for r in results)

    avg_speedup_eager = sum(r.speedup_over_eager_from_lowp for r in results) / num_models
    avg_speedup_compile = sum(r.speedup_over_compile_from_lowp for r in results) / num_models

    gm_speedup_eager = (
        torch.tensor(
            [r.speedup_over_eager_from_lowp for r in results],
            dtype=torch.float64,
        ).log().mean().exp().item()
    )

    gm_speedup_compile = (
        torch.tensor(
            [r.speedup_over_compile_from_lowp for r in results],
            dtype=torch.float64,
        ).log().mean().exp().item()
    )

    winner_order = [
        "trt_int8",
        "trt_fp16",
        "trt_bf16",
        "trt_fp8",
        "eager_fp32",
        "compile_fp32",
        None,
    ]

    winner_to_models = {k: [] for k in winner_order}
    for r in results:
        winner_to_models.setdefault(r.best_valid_overall_variant, []).append(r.model_name)

    print(f"\n{'=' * 80}")
    print("SUMMARY OVER ALL MODELS")
    print(f"Models evaluated                    : {num_models}")
    print(f"Models with valid low-precision run : {num_valid_lowp}")
    print(f"Models where lowp beats eager       : {num_beat_eager}")
    print(f"Models where lowp beats compile     : {num_beat_compile}")
    print(f"Avg speedup over eager from lowp    : {avg_speedup_eager:.4f}x")
    print(f"Geo mean speedup over eager         : {gm_speedup_eager:.4f}x")
    print(f"Avg speedup over compile from lowp  : {avg_speedup_compile:.4f}x")
    print(f"Geo mean speedup over compile       : {gm_speedup_compile:.4f}x")

    print(f"\n{'-' * 80}")
    print("BEST VALID OVERALL VARIANT DISTRIBUTION")

    for winner in winner_order:
        model_names = sorted(winner_to_models.get(winner, []))
        label = "no valid result" if winner is None else winner
        joined = ", ".join(model_names) if model_names else "-"
        print(f"{label:<15}: {len(model_names):>2} | {joined}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated precision search for KernelBench Level 3"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a KernelBench model file or a directory containing model files",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--calib-batch-size", type=int, default=64)
    parser.add_argument("--num-calib-batches", type=int, default=64)
    parser.add_argument("--json-out", type=str, default="precision_search_results.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    files = collect_model_files(args.path)
    results: list[ModelResult] = []

    for file_path in files:
        print(f"\nRunning: {file_path}")
        try:
            result = evaluate_model_file(
                file_path=file_path,
                num_warmup=args.warmup,
                num_iters=args.iters,
                atol=args.atol,
                rtol=args.rtol,
                calib_batch_size=args.calib_batch_size,
                num_calib_batches=args.num_calib_batches,
            )
        except Exception as exc:
            error = summarize_exception(exc)
            print(f"\nFAILED: {file_path}")
            print(error)
            result = make_failed_model_result(file_path, error)

        results.append(result)
        print_model_result(result)

    print_summary(results)

    write_json(results, args.json_out)

    print(f"\nSaved JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
