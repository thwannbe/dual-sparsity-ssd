# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels for activation-guided draft-only FFN width pruning."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _selected_gate_up_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    indices_ptr,
    output_ptr,
    M,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xh,
    stride_wn,
    stride_wh,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Compute packed [gate; up] projections for selected channels."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)

    selected_offsets = offs_n - tl.where(offs_n >= K, K, 0)
    selected = tl.load(
        indices_ptr + selected_offsets,
        mask=offs_n < 2 * K,
        other=0,
    ).to(tl.int64)
    weight_rows = selected + tl.where(offs_n >= K, INTERMEDIATE, 0)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        current_h = h_start + offs_h
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + current_h[None, :] * stride_xh,
            mask=(offs_m[:, None] < M) & (current_h[None, :] < H),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + current_h[:, None] * stride_wh
            + weight_rows[None, :] * stride_wn,
            mask=(current_h[:, None] < H) & (offs_n[None, :] < 2 * K),
            other=0.0,
        )
        accumulator = tl.dot(x, weight, accumulator)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + weight_rows,
            mask=offs_n < 2 * K,
            other=0.0,
        ).to(tl.float32)
        accumulator += bias[None, :]

    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < 2 * K),
    )


@triton.jit
def _selected_down_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    indices_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wi,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Apply the down projection without materializing compact weights."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        current_k = k_start + offs_k
        selected = tl.load(
            indices_ptr + current_k,
            mask=current_k < K,
            other=0,
        ).to(tl.int64)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + current_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (current_k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + selected[:, None] * stride_wi + offs_n[None, :] * stride_wn,
            mask=(current_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(x, weight, accumulator)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :].to(tl.float32)

    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _activation_score_kernel(
    activation_ptr,
    down_norm_ptr,
    score_ptr,
    valid_tokens_ptr,
    M: tl.constexpr,
    NUM_CHANNELS: tl.constexpr,
    stride_am,
    stride_ai,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    """Compute mean(abs(z), tokens) * down-column norm."""
    channel_offsets = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    token_offsets = tl.arange(0, BLOCK_TOKENS)
    valid_tokens = tl.minimum(tl.load(valid_tokens_ptr), M)
    accumulator = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)

    # Each activation element is read exactly once by its channel program.
    for token_start in tl.range(0, M, BLOCK_TOKENS):
        tokens = token_start + token_offsets
        values = tl.load(
            activation_ptr
            + tokens[:, None] * stride_am
            + channel_offsets[None, :] * stride_ai,
            mask=(tokens[:, None] < valid_tokens)
            & (channel_offsets[None, :] < NUM_CHANNELS),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(tl.abs(values), axis=0)

    denominator = tl.maximum(valid_tokens, 1).to(tl.float32)
    down_norm = tl.load(
        down_norm_ptr + channel_offsets,
        mask=channel_offsets < NUM_CHANNELS,
        other=0.0,
    )
    tl.store(
        score_ptr + channel_offsets,
        accumulator / denominator * down_norm,
        mask=channel_offsets < NUM_CHANNELS,
    )


def _matmul_config(num_rows: int) -> tuple[int, int, int, int]:
    if num_rows <= 16:
        return 16, 64, 4, 3
    if num_rows <= 64:
        return 32, 64, 4, 3
    return 64, 64, 8, 3


def _selected_gate_up(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor:
    hidden_size = x.shape[-1]
    local_intermediate = weight.shape[0] // 2
    kept = indices.numel()
    flat_x = x.reshape(-1, hidden_size)
    output = torch.empty((flat_x.shape[0], 2 * kept), device=x.device, dtype=x.dtype)
    block_m, block_n, num_warps, num_stages = _matmul_config(flat_x.shape[0])
    grid = (triton.cdiv(flat_x.shape[0], block_m), triton.cdiv(2 * kept, block_n))
    _selected_gate_up_kernel[grid](
        flat_x,
        weight,
        bias if bias is not None else weight,
        indices,
        output,
        flat_x.shape[0],
        H=hidden_size,
        INTERMEDIATE=local_intermediate,
        K=kept,
        stride_xm=flat_x.stride(0),
        stride_xh=flat_x.stride(1),
        stride_wn=weight.stride(0),
        stride_wh=weight.stride(1),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_H=64,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output.reshape(*x.shape[:-1], 2 * kept)


def _selected_gate_up_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], 2 * indices.numel()), device=x.device, dtype=x.dtype
    )


def _selected_down(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor:
    kept = indices.numel()
    output_size = weight.shape[0]
    flat_x = x.reshape(-1, kept)
    output = torch.empty((flat_x.shape[0], output_size), device=x.device, dtype=x.dtype)
    block_m, block_n, num_warps, num_stages = _matmul_config(flat_x.shape[0])
    grid = (
        triton.cdiv(flat_x.shape[0], block_m),
        triton.cdiv(output_size, block_n),
    )
    _selected_down_kernel[grid](
        flat_x,
        weight,
        bias if bias is not None else weight,
        indices,
        output,
        flat_x.shape[0],
        N=output_size,
        K=kept,
        stride_xm=flat_x.stride(0),
        stride_xk=flat_x.stride(1),
        stride_wn=weight.stride(0),
        stride_wi=weight.stride(1),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output.reshape(*x.shape[:-1], output_size)


def _selected_down_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=x.dtype)


def _collect_activation_score(
    activation: torch.Tensor,
    down_norm: torch.Tensor,
    score: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> None:
    local_intermediate = activation.shape[-1]
    flat_activation = activation.reshape(-1, local_intermediate)
    grid = (triton.cdiv(local_intermediate, 128),)
    _activation_score_kernel[grid](
        flat_activation,
        down_norm,
        score,
        valid_tokens,
        M=flat_activation.shape[0],
        NUM_CHANNELS=local_intermediate,
        stride_am=flat_activation.stride(0),
        stride_ai=flat_activation.stride(1),
        BLOCK_TOKENS=16,
        BLOCK_CHANNELS=128,
        num_warps=4,
    )


def _collect_activation_score_fake(
    activation: torch.Tensor,
    down_norm: torch.Tensor,
    score: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="width_pruned_gate_up",
    op_func=_selected_gate_up,
    fake_impl=_selected_gate_up_fake,
)
direct_register_custom_op(
    op_name="width_pruned_down",
    op_func=_selected_down,
    fake_impl=_selected_down_fake,
)
direct_register_custom_op(
    op_name="width_pruned_collect_activation_score",
    op_func=_collect_activation_score,
    mutates_args=["score"],
    fake_impl=_collect_activation_score_fake,
)


def selected_gate_up(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor:
    if x.is_cuda:
        return torch.ops.vllm.width_pruned_gate_up(x, weight, bias, indices)

    local_intermediate = weight.shape[0] // 2
    gate, up = weight.split(local_intermediate, dim=0)
    selected_weight = torch.cat(
        (gate.index_select(0, indices), up.index_select(0, indices)), dim=0
    )
    selected_bias = None
    if bias is not None:
        gate_bias, up_bias = bias.split(local_intermediate, dim=0)
        selected_bias = torch.cat(
            (
                gate_bias.index_select(0, indices),
                up_bias.index_select(0, indices),
            ),
            dim=0,
        )
    return torch.nn.functional.linear(x, selected_weight, selected_bias)


def selected_down(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor:
    if x.is_cuda:
        return torch.ops.vllm.width_pruned_down(x, weight, bias, indices)
    return torch.nn.functional.linear(x, weight.index_select(1, indices), bias)


def collect_activation_score(
    activation: torch.Tensor,
    down_norm: torch.Tensor,
    score: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> None:
    if activation.is_cuda:
        torch.ops.vllm.width_pruned_collect_activation_score(
            activation, down_norm, score, valid_tokens
        )
        return

    flat_activation = activation.reshape(-1, activation.shape[-1])
    count = min(int(valid_tokens.item()), flat_activation.shape[0])
    if count == 0:
        score.zero_()
    else:
        score.copy_(flat_activation[:count].float().abs().mean(dim=0) * down_norm)
