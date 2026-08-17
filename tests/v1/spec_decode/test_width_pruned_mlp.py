# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.v1.spec_decode.sparse_attn.ffn_overrider import (
    WidthPrunedMLPOverrider,
    build_width_pruned_mlp_overrider,
)


class UnquantizedLinearMethod:
    pass


class FakeGateUpProjection(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_parameter("bias", None)
        self.skip_bias_add = False
        self.tp_size = 1
        self.gather_output = False
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight), None


class FakeDownProjection(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_parameter("bias", None)
        self.skip_bias_add = False
        self.tp_rank = 0
        self.tp_size = 1
        self.input_is_parallel = True
        self.reduce_results = True
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight), None


class FakeSiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


class FakeMLP(nn.Module):
    def __init__(self, gate_up_weight: torch.Tensor, down_weight: torch.Tensor):
        super().__init__()
        self.gate_up_proj = FakeGateUpProjection(gate_up_weight)
        self.down_proj = FakeDownProjection(down_weight)
        self.act_fn = FakeSiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        intermediate = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate)
        return output


def test_width_pruning_is_used_only_during_drafting() -> None:
    generator = torch.Generator().manual_seed(7)
    hidden_size = 16
    intermediate_size = 512
    gate_up_weight = torch.randn(
        2 * intermediate_size, hidden_size, generator=generator
    )
    down_weight = torch.randn(hidden_size, intermediate_size, generator=generator)
    original_gate_up_weight = gate_up_weight.clone()
    original_down_weight = down_weight.clone()
    model = nn.Sequential(FakeMLP(gate_up_weight, down_weight))
    mlp = model[0]
    x = torch.randn(3, hidden_size, generator=generator)
    dense_output = mlp(x)

    overrider = WidthPrunedMLPOverrider(keep_ratio=0.5)
    overrider.register_model(model)

    assert overrider.draft_model is model
    assert mlp._draft_ffn_gate_weight.shape == (256, hidden_size)
    assert mlp._draft_ffn_up_weight.shape == (256, hidden_size)
    assert mlp._draft_ffn_down_weight.shape == (hidden_size, 256)
    assert (
        mlp._draft_ffn_gate_weight.untyped_storage().data_ptr()
        == mlp.gate_up_proj.weight.untyped_storage().data_ptr()
    )
    assert (
        mlp._draft_ffn_up_weight.untyped_storage().data_ptr()
        == mlp.gate_up_proj.weight.untyped_storage().data_ptr()
    )
    assert (
        mlp._draft_ffn_down_weight.untyped_storage().data_ptr()
        == mlp.down_proj.weight.untyped_storage().data_ptr()
    )
    # Registering prefix views must not alter the default/verification path.
    # A consistent channel permutation is algebraically equivalent, though
    # GEMM reduction order can introduce small floating-point differences.
    torch.testing.assert_close(mlp(x), dense_output, rtol=1e-4, atol=2e-4)

    gate, up = original_gate_up_weight.chunk(2, dim=0)
    score = torch.sqrt(
        torch.linalg.vector_norm(gate.float(), dim=1)
        * torch.linalg.vector_norm(up.float(), dim=1)
    ) * torch.linalg.vector_norm(original_down_weight.float(), dim=0)
    indices = torch.topk(score, 256, sorted=False).indices.sort().values
    expected_gate_up = torch.cat((gate[indices], up[indices]), dim=0)
    expected = F.linear(
        F.silu(F.linear(x, expected_gate_up)[..., :256])
        * F.linear(x, expected_gate_up)[..., 256:],
        original_down_weight[:, indices],
    )

    overrider.enter_propose()
    draft_output = mlp(x)
    overrider.exit_propose()

    torch.testing.assert_close(draft_output, expected)
    torch.testing.assert_close(mlp(x), dense_output, rtol=1e-4, atol=2e-4)


def test_keep_ratio_one_does_not_copy_or_patch_weights() -> None:
    mlp = FakeMLP(torch.randn(1024, 8), torch.randn(8, 512))
    original_forward = mlp.forward
    overrider = WidthPrunedMLPOverrider(keep_ratio=1.0)

    overrider.register_model(nn.Sequential(mlp))
    overrider.enter_propose()
    overrider.exit_propose()

    assert mlp.forward == original_forward
    assert not hasattr(mlp, "_draft_ffn_gate_weight")


def test_keep_ratio_one_builds_no_runtime_controller() -> None:
    assert build_width_pruned_mlp_overrider(1.0) is None


def test_keep_count_is_aligned_globally_for_tensor_parallelism() -> None:
    overrider = WidthPrunedMLPOverrider(keep_ratio=0.75)

    # 11008 global channels at TP=2 -> floor(8256 / 256) * 256 = 8192.
    assert overrider._local_keep_count(5504, tp_size=2) == 4096


def test_requested_keep_ratio_granularities() -> None:
    for ratio, expected in ((0.75, 9216), (0.5, 6144), (0.25, 3072)):
        overrider = WidthPrunedMLPOverrider(keep_ratio=ratio)
        assert overrider._local_keep_count(12288, tp_size=1) == expected


def test_shallow_model_alias_does_not_modify_original_tree() -> None:
    original = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.ReLU()))
    replacement = nn.Linear(4, 4)

    alias = WidthPrunedMLPOverrider._replace_module_with_shallow_alias(
        original, "0.0", replacement
    )

    assert alias is not original
    assert alias[0] is not original[0]
    assert alias[0][0] is replacement
    assert alias[0][1] is original[0][1]
    assert original[0][0] is not replacement
