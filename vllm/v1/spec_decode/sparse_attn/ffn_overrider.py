# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-only width pruning for dense SwiGLU MLPs."""

import copy
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import torch.nn as nn

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger

logger = init_logger(__name__)

_CHANNEL_ALIGNMENT = 256
_NORM_CHUNK_ROWS = 256
_CHANNEL_INDICES = "_draft_ffn_channel_indices"
_ACTIVATION_SCORE = "_draft_ffn_activation_score"
_DOWN_NORM = "_draft_ffn_down_norm"
_VALID_TOKENS = "_draft_ffn_valid_tokens"

_collect_activation_score_fn: Callable[..., None] | None = None
_selected_down_fn: Callable[..., torch.Tensor] | None = None
_selected_gate_up_fn: Callable[..., torch.Tensor] | None = None


def _load_width_pruned_kernels() -> None:
    """Register custom ops only when FFN pruning is actually enabled."""
    global _collect_activation_score_fn, _selected_down_fn, _selected_gate_up_fn
    if _selected_gate_up_fn is not None:
        return
    from vllm.v1.spec_decode.sparse_attn.ffn_kernels import (
        collect_activation_score,
        selected_down,
        selected_gate_up,
    )

    _collect_activation_score_fn = collect_activation_score
    _selected_down_fn = selected_down
    _selected_gate_up_fn = selected_gate_up


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

    gate_up_proj = mlp.gate_up_proj
    gate_up_bias = None
    if not gate_up_proj.skip_bias_add:
        gate_up_bias = gate_up_proj.bias
    assert _selected_gate_up_fn is not None
    gate_up = _selected_gate_up_fn(
        x,
        gate_up_proj.weight,
        gate_up_bias,
        getattr(mlp, _CHANNEL_INDICES),
    )
    # Reuse vLLM's fused SiluAndMul implementation in the draft graph.
    intermediate = mlp.act_fn(gate_up)

    down_proj = mlp.down_proj
    # RowParallelLinear adds its replicated bias on rank zero before TP reduce.
    down_bias = None
    if down_proj.tp_rank == 0 and not down_proj.skip_bias_add:
        down_bias = down_proj.bias
    assert _selected_down_fn is not None
    output = _selected_down_fn(
        intermediate,
        down_proj.weight,
        down_bias,
        getattr(mlp, _CHANNEL_INDICES),
    )
    if down_proj.reduce_results and down_proj.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output)
    return output


def _activation_and_score(act_fn: Any, x: torch.Tensor) -> torch.Tensor:
    """Collect scores only in the dense prefill/verification graph."""
    activation = act_fn._draft_ffn_dense_forward(x)
    if not act_fn._draft_ffn_enabled:
        assert _collect_activation_score_fn is not None
        _collect_activation_score_fn(
            activation,
            getattr(act_fn, _DOWN_NORM),
            getattr(act_fn, _ACTIVATION_SCORE),
            getattr(act_fn, _VALID_TOKENS),
        )
    return activation


class WidthPrunedMLPOverrider:
    """Select SwiGLU channels using the immediately preceding dense pass.

    vLLM's dense SwiGLU layers store the local TP shard as packed
    ``[gate; up]`` rows and store the matching down-projection channels in
    columns. Dense prefill/verification computes
    ``mean(abs(SiLU(gate) * up)) * ||W_down[:, j]||_2``. The next draft uses
    the top channels through indexed GEMMs, so changing the selection never
    copies or repacks the full weights.
    """

    def __init__(self, keep_ratio: float):
        if not 0 < keep_ratio <= 1:
            raise ValueError("draft_ffn_keep_ratio must be in (0, 1]")
        self.keep_ratio = keep_ratio
        self.in_propose = False
        self.mlps: list[Any] = []
        self.draft_model: nn.Module | None = None
        self.activation_scores: torch.Tensor | None = None
        self.channel_indices: torch.Tensor | None = None
        self.valid_tokens: torch.Tensor | None = None
        self._topk_values: torch.Tensor | None = None
        self._topk_indices: torch.Tensor | None = None
        self._sort_order: torch.Tensor | None = None
        if self.enabled:
            _load_width_pruned_kernels()

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

        first_gate_up = candidates[0].gate_up_proj
        local_intermediate = first_gate_up.weight.shape[0] // 2
        local_keep = self._local_keep_count(local_intermediate, first_gate_up.tp_size)
        device = first_gate_up.weight.device
        for module in candidates[1:]:
            candidate_intermediate = module.gate_up_proj.weight.shape[0] // 2
            candidate_keep = self._local_keep_count(
                candidate_intermediate, module.gate_up_proj.tp_size
            )
            if (
                candidate_intermediate != local_intermediate
                or candidate_keep != local_keep
                or module.gate_up_proj.weight.device != device
            ):
                raise ValueError(
                    "activation-guided width pruning requires uniform local "
                    "intermediate widths on one device"
                )

        num_layers = len(candidates)
        self.activation_scores = torch.empty(
            (num_layers, local_intermediate), device=device, dtype=torch.float32
        )
        self.channel_indices = torch.empty(
            (num_layers, local_keep), device=device, dtype=torch.int64
        )
        self._topk_values = torch.empty(
            (num_layers, local_keep), device=device, dtype=torch.float32
        )
        self._topk_indices = torch.empty_like(self.channel_indices)
        self._sort_order = torch.empty_like(self.channel_indices)
        self.valid_tokens = torch.tensor(
            torch.iinfo(torch.int32).max, device=device, dtype=torch.int32
        )

        # A deterministic initial value is useful during warmup; every real
        # proposal replaces it with scores from the preceding dense pass.
        self.activation_scores.zero_()
        initial_indices = torch.arange(local_keep, device=device, dtype=torch.int64)
        self.channel_indices.copy_(initial_indices.expand(num_layers, -1))

        for layer_index, module in enumerate(candidates):
            self._register_mlp(
                module,
                self.activation_scores[layer_index],
                self.channel_indices[layer_index],
                self.valid_tokens,
            )

        self.draft_model = self._build_draft_model(model)

        first = self.mlps[0]
        local_kept = getattr(first, _CHANNEL_INDICES).shape[0]
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

    def prepare_activation_collection(self, num_actual_tokens: int) -> None:
        """Set the padding boundary read by dense score-collection kernels."""
        assert self.valid_tokens is not None
        self.valid_tokens.fill_(num_actual_tokens)

    def _update_draft_indices(self) -> None:
        """Select all layers once for the next multi-token draft."""
        assert self.activation_scores is not None
        assert self.channel_indices is not None
        assert self._topk_values is not None
        assert self._topk_indices is not None
        assert self._sort_order is not None
        torch.topk(
            self.activation_scores,
            self.channel_indices.shape[1],
            dim=1,
            largest=True,
            sorted=False,
            out=(self._topk_values, self._topk_indices),
        )
        # Ascending channel order improves locality in the indexed weight loads.
        torch.sort(
            self._topk_indices,
            dim=1,
            out=(self.channel_indices, self._sort_order),
        )

    def enter_propose(self) -> None:
        if not self.enabled:
            return
        assert not self.in_propose
        self._update_draft_indices()
        self.in_propose = True
        for mlp in self.mlps:
            mlp._draft_ffn_enabled = True
            mlp.act_fn._draft_ffn_enabled = True

    def exit_propose(self) -> None:
        if not self.enabled:
            return
        for mlp in self.mlps:
            mlp._draft_ffn_enabled = False
            mlp.act_fn._draft_ffn_enabled = False
        self.in_propose = False

    @staticmethod
    def _looks_like_swiglu(module: Any) -> bool:
        return all(
            hasattr(module, attribute)
            for attribute in ("gate_up_proj", "down_proj", "act_fn")
        )

    def _register_mlp(
        self,
        mlp: Any,
        activation_score: torch.Tensor,
        channel_indices: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> None:
        gate_up_proj = mlp.gate_up_proj
        down_proj = mlp.down_proj
        self._validate_mlp(gate_up_proj, down_proj, mlp.act_fn)

        down_weight = down_proj.weight.detach()
        # Only the static down-column norm remains weight-derived. Activation
        # importance and top-k selection are refreshed from every dense pass.
        down_norm = self._column_norm_fp32(down_weight)

        mlp.register_buffer(_CHANNEL_INDICES, channel_indices, persistent=False)
        mlp.act_fn.register_buffer(
            _ACTIVATION_SCORE, activation_score, persistent=False
        )
        mlp.act_fn.register_buffer(_DOWN_NORM, down_norm, persistent=False)
        mlp.act_fn.register_buffer(_VALID_TOKENS, valid_tokens, persistent=False)
        # Bypass nn.Module registration for method/state used only to dispatch.
        object.__setattr__(mlp, "_draft_ffn_dense_forward", mlp.forward)
        object.__setattr__(mlp, "_draft_ffn_enabled", False)
        object.__setattr__(mlp, "forward", MethodType(_draft_or_dense_forward, mlp))
        object.__setattr__(mlp.act_fn, "_draft_ffn_dense_forward", mlp.act_fn.forward)
        object.__setattr__(mlp.act_fn, "_draft_ffn_enabled", False)
        object.__setattr__(
            mlp.act_fn,
            "forward",
            MethodType(_activation_and_score, mlp.act_fn),
        )
        self.mlps.append(mlp)

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
