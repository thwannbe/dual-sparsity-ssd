# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.sparse_attn.attn_overrider import (
    BaseAttnOverrider,
    DenseAttnBypass,
    build_attention_overrider,
)


def test_ratio_one_does_not_install_attention_override() -> None:
    import vllm.v1.attention.backends.flash_attn as flash_attn

    original_attention = flash_attn.flash_attn_varlen_func
    original_override_count = BaseAttnOverrider._GLOBAL_OVERRIDER_COUNT
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            sparse_attn_algorithm="vegas",
            sparse_attn_ratio=1.0,
        )
    )

    overrider = build_attention_overrider(config, torch.device("cpu"))
    overrider.enter_propose()
    overrider.exit_propose()

    assert isinstance(overrider, DenseAttnBypass)
    assert flash_attn.flash_attn_varlen_func is original_attention
    assert original_override_count == BaseAttnOverrider._GLOBAL_OVERRIDER_COUNT


def test_dense_and_sparse_attention_use_different_compile_hashes() -> None:
    common = {"method": "sparse_attn", "num_speculative_tokens": 2}
    dense = SpeculativeConfig(**common, sparse_attn_ratio=1.0)
    sparse = SpeculativeConfig(**common, sparse_attn_ratio=0.25)

    assert dense.compute_hash() != sparse.compute_hash()
