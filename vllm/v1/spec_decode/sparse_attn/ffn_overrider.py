# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-only width pruning for dense SwiGLU MLPs."""

import copy
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger

logger = init_logger(__name__)

_CHANNEL_ALIGNMENT = 256
_NORM_CHUNK_ROWS = 256
_DRAFT_GATE_WEIGHT = "_draft_ffn_gate_weight"
_DRAFT_UP_WEIGHT = "_draft_ffn_up_weight"
_DRAFT_DOWN_WEIGHT = "_draft_ffn_down_weight"
_DRAFT_GATE_BIAS = "_draft_ffn_gate_bias"
_DRAFT_UP_BIAS = "_draft_ffn_up_bias"


def build_width_pruned_mlp_overrider(
    keep_ratio: float,
) -> "WidthPrunedMLPOverrider | None":
    """Return no controller at all for the exact dense-FFN configuration."""
    if keep_ratio == 1.0:
        logger.info(
            "Draft FFN width pruning is disabled (keep ratio 1.0); "
            "preserving the original dense FFN path."
        )
        return None
    return WidthPrunedMLPOverrider(keep_ratio)


def _draft_or_dense_forward(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    """Dispatch an MLP call without changing its public forward signature."""
    if not mlp._draft_ffn_enabled:
        return mlp._draft_ffn_dense_forward(x)

    gate = F.linear(
        x,
        getattr(mlp, _DRAFT_GATE_WEIGHT),
        getattr(mlp, _DRAFT_GATE_BIAS),
    )
    up = F.linear(
        x,
        getattr(mlp, _DRAFT_UP_WEIGHT),
        getattr(mlp, _DRAFT_UP_BIAS),
    )
    intermediate = F.silu(gate) * up

    down_proj = mlp.down_proj
    # RowParallelLinear adds its replicated bias on rank zero before the TP
    # reduction. Match that behavior for the prefix-view draft projection.
    down_bias = None
    if down_proj.tp_rank == 0 and not down_proj.skip_bias_add:
        down_bias = down_proj.bias
    output = F.linear(intermediate, getattr(mlp, _DRAFT_DOWN_WEIGHT), down_bias)
    if down_proj.reduce_results and down_proj.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output)
    return output


class WidthPrunedMLPOverrider:
    """Reorder SwiGLU channels and select prefix views only while drafting.

    vLLM's dense SwiGLU layers store the local TP shard as packed
    ``[gate; up]`` rows and store the matching down-projection channels in
    columns. Each TP rank selects the same number of its locally most
    important channels, keeping GEMM shapes balanced across ranks.
    """

    def __init__(self, keep_ratio: float):
        if not 0 < keep_ratio <= 1:
            raise ValueError("draft_ffn_keep_ratio must be in (0, 1]")
        self.keep_ratio = keep_ratio
        self.in_propose = False
        self.mlps: list[Any] = []
        self.draft_model: nn.Module | None = None

    @property
    def enabled(self) -> bool:
        return self.keep_ratio < 1.0

    def register_model(self, model: nn.Module) -> None:
        if not self.enabled:
            logger.info("Draft FFN width pruning is disabled (keep ratio 1.0).")
            return

        candidates: list[Any] = []
        incompatible: list[str] = []
        for name, module in model.named_modules():
            if not self._looks_like_swiglu(module):
                continue
            try:
                self._validate_mlp(module.gate_up_proj, module.down_proj, module.act_fn)
            except ValueError as exc:
                incompatible.append(f"{name}: {exc}")
            else:
                candidates.append(module)

        if incompatible:
            details = "; ".join(incompatible[:3])
            if len(incompatible) > 3:
                details += f"; and {len(incompatible) - 3} more"
            raise ValueError(
                "Draft FFN width pruning found unsupported SwiGLU layers. " + details
            )
        if not candidates:
            raise ValueError(
                "draft_ffn_keep_ratio is below 1.0, but no compatible dense "
                "SwiGLU MLPs were found in the target model"
            )
        for module in candidates:
            self._register_mlp(module)

        self.draft_model = self._build_draft_model(model)

        first = self.mlps[0]
        local_kept = getattr(first, _DRAFT_GATE_WEIGHT).shape[0]
        local_full = first.gate_up_proj.weight.shape[0] // 2
        tp_size = first.gate_up_proj.tp_size
        logger.info(
            "Enabled draft-only FFN width pruning for %d MLPs: "
            "%d/%d global intermediate channels (keep ratio %.4f).",
            len(self.mlps),
            local_kept * tp_size,
            local_full * tp_size,
            self.keep_ratio,
        )

    def _build_draft_model(self, model: nn.Module) -> nn.Module:
        """Build a separately compiled model alias with shared weights.

        vLLM drops Dynamo guards after its first trace, so a Python draft/dense
        flag in the target model would be frozen to the verification branch.
        Drafting therefore gets its own compiled backbone. Only module
        containers are copied; parameters, buffers, and KV-cache layers remain
        shared with the target model.
        """
        from vllm.compilation.backends import set_model_tag
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        mlp_ids = {id(mlp) for mlp in self.mlps}
        compiled_backbones: list[tuple[str, nn.Module]] = []
        for name, module in model.named_modules():
            if not isinstance(module, TorchCompileWithNoGuardsWrapper):
                continue
            if getattr(module, "do_not_compile", True):
                continue
            if mlp_ids.issubset({id(child) for child in module.modules()}):
                compiled_backbones.append((name, module))

        if not compiled_backbones:
            # Eager execution reevaluates the draft flag on every call.
            return model

        # Use the deepest compiled module containing all pruned MLPs so an
        # unrelated part of a composite model is not compiled twice.
        backbone_path, backbone = max(
            compiled_backbones, key=lambda item: item[0].count(".")
        )
        draft_backbone = copy.copy(backbone)
        with set_model_tag("sparse_attn_draft"):
            TorchCompileWithNoGuardsWrapper.__init__(draft_backbone)

        logger.info(
            "Created a separate compiled draft FFN graph for %s; "
            "weights remain shared with dense verification.",
            backbone_path or type(backbone).__name__,
        )
        return self._replace_module_with_shallow_alias(
            model, backbone_path, draft_backbone
        )

    @staticmethod
    def _replace_module_with_shallow_alias(
        root: nn.Module,
        module_path: str,
        replacement: nn.Module,
    ) -> nn.Module:
        """Replace one descendant while sharing the rest of the module tree."""
        if not module_path:
            return replacement

        root_alias = copy.copy(root)
        root_alias._modules = root._modules.copy()
        source_parent = root
        alias_parent = root_alias
        path = module_path.split(".")
        for name in path[:-1]:
            source_child = source_parent._modules[name]
            alias_child = copy.copy(source_child)
            alias_child._modules = source_child._modules.copy()
            alias_parent._modules[name] = alias_child
            source_parent = source_child
            alias_parent = alias_child
        alias_parent._modules[path[-1]] = replacement
        return root_alias

    def enter_propose(self) -> None:
        if not self.enabled:
            return
        assert not self.in_propose
        self.in_propose = True
        for mlp in self.mlps:
            mlp._draft_ffn_enabled = True

    def exit_propose(self) -> None:
        if not self.enabled:
            return
        for mlp in self.mlps:
            mlp._draft_ffn_enabled = False
        self.in_propose = False

    @staticmethod
    def _looks_like_swiglu(module: Any) -> bool:
        return all(
            hasattr(module, attribute)
            for attribute in ("gate_up_proj", "down_proj", "act_fn")
        )

    def _register_mlp(self, mlp: Any) -> None:
        gate_up_proj = mlp.gate_up_proj
        down_proj = mlp.down_proj
        self._validate_mlp(gate_up_proj, down_proj, mlp.act_fn)

        gate_up_weight = gate_up_proj.weight.detach()
        down_weight = down_proj.weight.detach()
        local_intermediate = gate_up_weight.shape[0] // 2
        local_keep = self._local_keep_count(local_intermediate, gate_up_proj.tp_size)

        gate, up = gate_up_weight.split(local_intermediate, dim=0)
        # Compute norms in small FP32 chunks once at model initialization.
        # This avoids materializing an entire projection in FP32.
        gate_norm = self._row_norm_fp32(gate)
        up_norm = self._row_norm_fp32(up)
        down_norm = self._column_norm_fp32(down_weight)
        score = torch.sqrt(gate_norm * up_norm) * down_norm
        indices = torch.topk(score, local_keep, sorted=False).indices.sort().values

        # Move selected channels to a shared prefix in all three projections.
        # Dense FFN semantics are preserved because gate, up, and down are
        # permuted consistently. Draft views then share the full weights'
        # storage and allocate no persistent compact-weight copy.
        all_indices = torch.arange(
            local_intermediate, device=indices.device, dtype=indices.dtype
        )
        is_remaining = torch.ones(
            local_intermediate, device=indices.device, dtype=torch.bool
        )
        is_remaining[indices] = False
        permutation = torch.cat((indices, all_indices[is_remaining]))

        with torch.no_grad():
            self._permute_in_place(gate, permutation, dim=0)
            self._permute_in_place(up, permutation, dim=0)
            self._permute_in_place(down_weight, permutation, dim=1)

        draft_gate_bias = None
        draft_up_bias = None
        if gate_up_proj.bias is not None and not gate_up_proj.skip_bias_add:
            gate_bias, up_bias = gate_up_proj.bias.detach().split(
                local_intermediate, dim=0
            )
            with torch.no_grad():
                self._permute_in_place(gate_bias, permutation, dim=0)
                self._permute_in_place(up_bias, permutation, dim=0)
            draft_gate_bias = gate_bias[:local_keep]
            draft_up_bias = up_bias[:local_keep]

        mlp.register_buffer(_DRAFT_GATE_WEIGHT, gate[:local_keep], persistent=False)
        mlp.register_buffer(_DRAFT_UP_WEIGHT, up[:local_keep], persistent=False)
        mlp.register_buffer(
            _DRAFT_DOWN_WEIGHT,
            down_weight[:, :local_keep],
            persistent=False,
        )
        mlp.register_buffer(_DRAFT_GATE_BIAS, draft_gate_bias, persistent=False)
        mlp.register_buffer(_DRAFT_UP_BIAS, draft_up_bias, persistent=False)
        # Bypass nn.Module registration for method/state used only to dispatch.
        object.__setattr__(mlp, "_draft_ffn_dense_forward", mlp.forward)
        object.__setattr__(mlp, "_draft_ffn_enabled", False)
        object.__setattr__(mlp, "forward", MethodType(_draft_or_dense_forward, mlp))
        self.mlps.append(mlp)

    @staticmethod
    def _row_norm_fp32(weight: torch.Tensor) -> torch.Tensor:
        result = torch.empty(weight.shape[0], device=weight.device, dtype=torch.float32)
        for start in range(0, weight.shape[0], _NORM_CHUNK_ROWS):
            end = min(start + _NORM_CHUNK_ROWS, weight.shape[0])
            chunk = weight[start:end].to(dtype=torch.float32, copy=True)
            chunk.square_()
            result[start:end] = chunk.sum(dim=1).sqrt_()
        return result

    @staticmethod
    def _column_norm_fp32(weight: torch.Tensor) -> torch.Tensor:
        squared_sum = torch.zeros(
            weight.shape[1], device=weight.device, dtype=torch.float32
        )
        for start in range(0, weight.shape[0], _NORM_CHUNK_ROWS):
            end = min(start + _NORM_CHUNK_ROWS, weight.shape[0])
            chunk = weight[start:end].to(dtype=torch.float32, copy=True)
            chunk.square_()
            squared_sum.add_(chunk.sum(dim=0))
        return squared_sum.sqrt_()

    @staticmethod
    def _permute_in_place(
        weight: torch.Tensor, permutation: torch.Tensor, dim: int
    ) -> None:
        reordered = weight.index_select(dim, permutation)
        weight.copy_(reordered)

    def _local_keep_count(self, local_intermediate: int, tp_size: int) -> int:
        global_intermediate = local_intermediate * tp_size
        global_keep = int(global_intermediate * self.keep_ratio)
        global_keep = max(
            _CHANNEL_ALIGNMENT,
            (global_keep // _CHANNEL_ALIGNMENT) * _CHANNEL_ALIGNMENT,
        )
        global_keep = min(global_keep, global_intermediate)
        if global_keep % tp_size != 0:
            raise ValueError(
                f"aligned keep count {global_keep} is not divisible by TP size "
                f"{tp_size}"
            )
        return global_keep // tp_size

    @staticmethod
    def _validate_mlp(gate_up_proj: Any, down_proj: Any, act_fn: Any) -> None:
        required_gate_up = (
            "weight",
            "bias",
            "skip_bias_add",
            "tp_size",
            "gather_output",
            "quant_method",
        )
        required_down = (
            "weight",
            "bias",
            "skip_bias_add",
            "tp_rank",
            "tp_size",
            "input_is_parallel",
            "reduce_results",
            "quant_method",
        )
        if not all(hasattr(gate_up_proj, attr) for attr in required_gate_up):
            raise ValueError("gate_up_proj is not a supported vLLM linear layer")
        if not all(hasattr(down_proj, attr) for attr in required_down):
            raise ValueError("down_proj is not a supported vLLM linear layer")
        if gate_up_proj.quant_method.__class__.__name__ != "UnquantizedLinearMethod":
            raise ValueError("quantized gate/up projections are not supported")
        if down_proj.quant_method.__class__.__name__ != "UnquantizedLinearMethod":
            raise ValueError("quantized down projections are not supported")
        if gate_up_proj.gather_output:
            raise ValueError("gathered gate/up projections are not supported")
        if not down_proj.input_is_parallel:
            raise ValueError("non-parallel down-projection inputs are not supported")
        if gate_up_proj.tp_size != down_proj.tp_size:
            raise ValueError("gate/up and down projections use different TP sizes")
        if not act_fn.__class__.__name__.endswith("SiluAndMul"):
            raise ValueError("only SwiGLU/SiluAndMul activations are supported")

        gate_up_weight = gate_up_proj.weight
        down_weight = down_proj.weight
        if gate_up_weight.ndim != 2 or down_weight.ndim != 2:
            raise ValueError("pruning requires unpacked two-dimensional weights")
        if gate_up_weight.shape[0] % 2 != 0:
            raise ValueError("gate/up weight does not contain two equal shards")
        if gate_up_weight.shape[0] // 2 != down_weight.shape[1]:
            raise ValueError("gate/up rows do not match down-projection columns")
