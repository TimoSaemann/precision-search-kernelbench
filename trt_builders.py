import copy

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch_tensorrt
from modelopt.torch.quantization.utils import export_torch_mode

TRT_VARIANT_TIMEOUT_SECONDS = 120

def get_boundary_weighted_layers_to_disable(model: nn.Module, n_front: int = 1, n_back: int = 2) -> list[str]:
    quantizable_names = get_quantizable_module_names(model)
    selected = []
    selected.extend(quantizable_names[:n_front])
    if n_back > 0:
        selected.extend(quantizable_names[-n_back:])
    # remove duplicates while preserving order
    seen = set()
    deduped = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped

def get_quantizable_module_names(model: nn.Module) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            names.append(name)
    return names


@torch.no_grad()
def build_trt_fp16_model(
    fp32_model: nn.Module,
    input_shape_wo_batch: tuple[int, ...],
    batch_size_eval: int,
):
    model = copy.deepcopy(fp32_model).eval().cuda().half()

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, *input_shape_wo_batch),
                opt_shape=(batch_size_eval, *input_shape_wo_batch),
                max_shape=(batch_size_eval, *input_shape_wo_batch),
                dtype=torch.float16,
            )
        ],
        truncate_long_and_double=True,
    )
    return trt_model


@torch.no_grad()
def build_trt_bf16_model(
    fp32_model: nn.Module,
    input_shape_wo_batch: tuple[int, ...],
    batch_size_eval: int,
):
    model = copy.deepcopy(fp32_model).eval().cuda().bfloat16()

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, *input_shape_wo_batch),
                opt_shape=(batch_size_eval, *input_shape_wo_batch),
                max_shape=(batch_size_eval, *input_shape_wo_batch),
                dtype=torch.bfloat16,
            )
        ],
        truncate_long_and_double=True,
    )
    return trt_model


@torch.no_grad()
def build_trt_fp8_model(
    fp32_model: nn.Module,
    input_shape_wo_batch: tuple[int, ...],
    batch_size_eval: int,
    calib_batch_size: int = 32,
    num_calib_batches: int = 32,
    example_input: torch.Tensor | None = None,
):
    model = copy.deepcopy(fp32_model).eval().cuda()

    if example_input is not None:
        sample = example_input.detach().to(device="cuda", dtype=torch.float32)
        calib_mean = sample.mean()
        calib_std = sample.std().clamp_min(1e-6)
    else:
        calib_mean = None
        calib_std = None

    @torch.no_grad()
    def calibrate_loop(m: nn.Module):
        m.eval()
        for _ in range(num_calib_batches):
            if calib_mean is not None and calib_std is not None:
                x = (
                    torch.randn(
                        calib_batch_size,
                        *input_shape_wo_batch,
                        device="cuda",
                        dtype=torch.float32,
                    ) * calib_std + calib_mean
                )
            else:
                x = torch.randn(
                    calib_batch_size,
                    *input_shape_wo_batch,
                    device="cuda",
                    dtype=torch.float32,
                )
            _ = m(x)

    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    quantized_model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    quantized_model.eval().cuda()

    with export_torch_mode():
        trt_model = torch_tensorrt.compile(
            quantized_model,
            ir="dynamo",
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, *input_shape_wo_batch),
                    opt_shape=(batch_size_eval, *input_shape_wo_batch),
                    max_shape=(batch_size_eval, *input_shape_wo_batch),
                    dtype=torch.float32,
                )
            ],
            use_explicit_typing=True,
            truncate_long_and_double=True,
        )

    return trt_model

@torch.no_grad()
def build_trt_int8_model(
    fp32_model: nn.Module,
    input_shape_wo_batch: tuple[int, ...],
    batch_size_eval: int,
    calib_batch_size: int = 32,
    num_calib_batches: int = 32,
    example_input: torch.Tensor | None = None,
):
    model = copy.deepcopy(fp32_model).eval().cuda()

    if example_input is not None:
        sample = example_input.detach().to(device="cuda", dtype=torch.float32)
        calib_mean = sample.mean()
        calib_std = sample.std().clamp_min(1e-6)
    else:
        calib_mean = None
        calib_std = None

    @torch.no_grad()
    def calibrate_loop(m: nn.Module):
        m.eval()
        if example_input is not None:
            base = example_input.detach().to(device="cuda", dtype=torch.float32)
            for _ in range(num_calib_batches):
                if base.shape[0] == calib_batch_size:
                    x = base.clone()
                else:
                    reps = (calib_batch_size + base.shape[0] - 1) // base.shape[0]
                    x = base.repeat((reps,) + (1,) * (base.ndim - 1))[:calib_batch_size].clone()

                noise = 0.01 * torch.randn_like(x)
                x = x + noise
                _ = m(x)
        else:
            for _ in range(num_calib_batches):
                x = torch.randn(
                    calib_batch_size,
                    *input_shape_wo_batch,
                    device="cuda",
                    dtype=torch.float32,
                )
                _ = m(x)

    quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)

    for name in get_boundary_weighted_layers_to_disable(model, n_front=0, n_back=0):
        quant_cfg["quant_cfg"][f"*{name}*"] = {"enable": False}

    quantized_model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    quantized_model.eval().cuda()

    with export_torch_mode():
        trt_model = torch_tensorrt.compile(
            quantized_model,
            ir="dynamo",
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, *input_shape_wo_batch),
                    opt_shape=(batch_size_eval, *input_shape_wo_batch),
                    max_shape=(batch_size_eval, *input_shape_wo_batch),
                    dtype=torch.float32,
                )
            ],
            use_explicit_typing=True,
            truncate_long_and_double=True,
        )

    return trt_model