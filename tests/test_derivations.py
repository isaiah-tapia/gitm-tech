"""
Test suite for the DeepSeek-V4-Pro decode-step derivations.

Two jobs:

  1. GOLDEN TESTS pin the numbers that appear in the design note. If a refactor
     changes any figure the note quotes, these fail loudly.

  2. REASONING TESTS pin the *arguments*, not just the results. Several findings
     in the note are claims about why a number is what it is, and a number can
     stay right while its justification quietly rots. Example: the note says FP4
     expert storage is FORCED because the FP8 reading exceeds total_size on its
     own. That is a falsifiable statement, so it gets a test.

Runs standalone (no dependencies):

    python3 tests/test_derivations.py

Also collects under pytest if you install it:

    pytest tests/ -v
"""

import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "derive"))

import weights as W      # noqa: E402
import fit as F          # noqa: E402

CKPT = os.path.join(ROOT, "context", "checkpoint")
INDEX = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
PUBLISHED_BYTES = INDEX["metadata"]["total_size"]
PUBLISHED_TENSORS = len(INDEX["weight_map"])

GB = 1e9


def census(pattern_suffix):
    """Count tensors in the real index whose name ends with the given suffix."""
    return sum(1 for k in INDEX["weight_map"] if k.endswith(pattern_suffix))


def predicted_total():
    """Rebuild the full checkpoint byte total the same way weights.py main() does."""
    total = W.bf16_linear(W.VOCAB, W.DIM)[0]                      # embed
    tensors = 1
    for lid in range(W.N_LAYERS):
        b, n = W.block_bytes(lid, W.RATIOS[lid])
        total += b
        tensors += n
    total += W.rmsnorm(W.DIM)[0]; tensors += 1                    # norm
    b, n = W.hc_head_bytes(); total += b; tensors += n            # model hc_head_*
    total += W.bf16_linear(W.VOCAB, W.DIM)[0]; tensors += 1       # head
    for i in range(W.N_MTP):
        lid = W.N_LAYERS + i
        b, n = W.block_bytes(lid, W.RATIOS[lid])
        total += b; tensors += n
        for _ in range(2):
            x, xn = W.fp8_linear(W.DIM, W.DIM); total += x; tensors += xn
        for _ in range(3):
            x, xn = W.rmsnorm(W.DIM); total += x; tensors += xn
        x, xn = W.hc_head_bytes(); total += x; tensors += xn
    return total, tensors


# =====================================================================
# 1. GOLDEN: the reconciliation everything else rests on
# =====================================================================

def test_weight_total_is_byte_exact():
    """The headline claim. Not 'within 2%'. Zero."""
    total, _ = predicted_total()
    assert total == PUBLISHED_BYTES, (
        f"predicted {total:,} != published {PUBLISHED_BYTES:,} "
        f"(delta {total - PUBLISHED_BYTES:+,} B)"
    )


def test_tensor_count_is_exact():
    """Byte totals can match by coincidence if two errors cancel. Counts catch that."""
    _, tensors = predicted_total()
    assert tensors == PUBLISHED_TENSORS, (
        f"predicted {tensors:,} tensors != published {PUBLISHED_TENSORS:,}"
    )


# =====================================================================
# 2. STRUCTURE: the layer schedules that do not tile
# =====================================================================

def test_compress_ratios_covers_layers_plus_mtp():
    """62 entries, not 61. Misreading this produced a real error earlier:
    it made layer 60 look like it had no compressor, contradicting the census."""
    assert len(W.RATIOS) == W.N_LAYERS + W.N_MTP == 62


def test_compress_ratio_distribution():
    assert W.RATIOS.count(4) == 30
    assert W.RATIOS.count(128) == 31
    assert W.RATIOS.count(0) == 1
    assert 30 + 31 + 1 == len(W.RATIOS)


def test_the_single_zero_is_the_mtp_block_not_a_backbone_layer():
    """The claim that resolved the contradiction. If the zero were a backbone
    layer, one layer would lack a compressor and the census would disagree."""
    assert W.RATIOS[W.N_LAYERS] == 0
    assert all(W.RATIOS[i] != 0 for i in range(W.N_LAYERS))


def test_hash_layers_and_gated_layers_partition_the_backbone():
    assert W.N_HASH == 3
    assert W.N_HASH + 58 == W.N_LAYERS


# =====================================================================
# 3. CENSUS: derived structure vs the real index
# =====================================================================

def test_indexer_exists_on_exactly_the_ratio4_layers():
    """model.py builds an Indexer only when compress_ratio == 4."""
    assert census("attn.indexer.weights_proj.weight") == W.RATIOS[:W.N_LAYERS].count(4) == 30


def test_compressor_exists_on_every_backbone_layer():
    assert census("attn.compressor.wkv.weight") == W.N_LAYERS == 61


def test_gate_bias_and_tid2eid_partition_exactly():
    """58 learned-gate layers + 3 hash layers = 61. No layer has both or neither."""
    bias = sum(1 for k in INDEX["weight_map"]
               if k.endswith("ffn.gate.bias") and k.startswith("layers."))
    tid = census("ffn.gate.tid2eid")
    assert tid == W.N_HASH == 3
    assert bias == 58
    assert bias + tid == W.N_LAYERS


def test_expert_tensor_count_matches_layers_times_experts():
    assert census("w1.weight") == (W.N_LAYERS + W.N_MTP) * W.N_EXPERTS + (W.N_LAYERS + W.N_MTP)


# =====================================================================
# 4. FINDINGS: the reasoning, not just the result
# =====================================================================

def test_fp4_expert_storage_is_forced_not_merely_labelled():
    """The note claims FP4 is established by arithmetic rather than taken from
    the `expert_dtype` field. That claim is true only if the FP8 reading is
    impossible, so assert the impossibility directly."""
    fp8_experts = sum(
        W.fp8_linear(o, i)[0]
        for _ in range(W.N_LAYERS)
        for _ in range(W.N_EXPERTS)
        for o, i in [(W.MOE_INTER, W.DIM), (W.DIM, W.MOE_INTER), (W.MOE_INTER, W.DIM)]
    )
    assert fp8_experts > PUBLISHED_BYTES, (
        "FP8 experts must exceed total_size on their own, else FP4 is an "
        "assumption rather than a derivation"
    )


def test_tid2eid_int32_produces_the_exact_known_residual():
    """int64 storage was DERIVED from a residual, not assumed. Pin the derivation:
    the int32 shortfall must be exactly 3 layers x vocab x n_act x 4 bytes."""
    int32_total = predicted_total()[0] - W.N_HASH * W.VOCAB * W.N_ACT * (W.B_I64 - 4)
    residual = PUBLISHED_BYTES - int32_total
    assert residual == 9_308_160
    assert residual == W.N_HASH * W.VOCAB * W.N_ACT * 4


def test_fp4_scale_overhead_is_one_sixteenth_of_weights():
    """The 48 GB nobody budgets for. FP4 scales keep full output resolution."""
    w_only = W.MOE_INTER * (W.DIM // 2)
    total = W.fp4_linear(W.MOE_INTER, W.DIM)[0]
    assert (total - w_only) / w_only == 1 / 16


def test_fp8_scale_overhead_is_negligible_by_contrast():
    """FP8 blocks are 128x128, so the scale is ~1/16384. The asymmetry is the point."""
    w_only = W.MOE_INTER * W.DIM
    total = W.fp8_linear(W.MOE_INTER, W.DIM)[0]
    ratio = (total - w_only) / w_only
    assert ratio < 1 / 10_000


def test_expert_bank_dominates_parameter_count():
    """~95% of params are experts, which is why expert_dtype drives the fit."""
    expert_params = W.N_LAYERS * W.N_EXPERTS * 3 * (W.DIM * W.MOE_INTER)
    assert expert_params > 1.5e12
    assert 0.94 < expert_params / 1.6e12 < 0.98


# =====================================================================
# 5. SHARDING: why per-GPU weights are not total/8
# =====================================================================

def test_per_gpu_weights_exceed_naive_division():
    rep, shd, _ = F.per_gpu_weight_bytes()
    naive = PUBLISHED_BYTES / F.TP
    assert rep + shd > naive


def test_replication_gap_is_about_6_46_gb():
    rep, shd, _ = F.per_gpu_weight_bytes()
    gap = (rep + shd - PUBLISHED_BYTES / F.TP) / GB
    assert 6.3 < gap < 6.6, f"replication gap {gap:.2f} GB drifted"


def test_replicated_total_is_nonzero_and_shared_experts_dominate_it():
    """shared_experts is a plain Linear, so it is replicated, and it is the
    largest such item. If it ever shards, this assumption must be revisited."""
    rep, _, _ = F.per_gpu_weight_bytes()
    shared = (W.N_LAYERS + W.N_MTP) * 3 * W.fp8_linear(W.MOE_INTER, W.DIM)[0]
    assert shared / rep > 0.5, "shared_experts should be >half of replicated bytes"


def test_weights_per_gpu_matches_note():
    rep, shd, _ = F.per_gpu_weight_bytes()
    assert 114.0 < (rep + shd) / GB < 115.0


# =====================================================================
# 6. KV CACHE: the O(1) split that is easy to get wrong
# =====================================================================

def test_kv_window_term_is_invariant_in_context():
    """The named failure mode: counting the window term as growing. It does not."""
    fixed_2k, _ = F.kv_bytes_per_seq(1, ctx=2048)
    fixed_8k, _ = F.kv_bytes_per_seq(1, ctx=8192)
    fixed_1m, _ = F.kv_bytes_per_seq(1, ctx=1_048_576)
    assert fixed_2k == fixed_8k == fixed_1m


def test_kv_growing_term_is_linear_in_context():
    _, g_2k = F.kv_bytes_per_seq(1, ctx=2048)
    _, g_8k = F.kv_bytes_per_seq(1, ctx=8192)
    assert g_8k == 4 * g_2k


def test_kv_fp8_is_exactly_half_of_bf16():
    assert sum(F.kv_bytes_per_seq(1)) * 2 == sum(F.kv_bytes_per_seq(2))


def test_kv_is_small_relative_to_weights():
    """Supports the claim that KV streaming will not lead the bound ranking."""
    rep, shd, _ = F.per_gpu_weight_bytes()
    kv = sum(F.kv_bytes_per_seq(1)) * F.BATCH
    assert kv / (rep + shd) < 0.02


def test_compression_beats_uncompressed_latent_by_about_6x():
    uncompressed = W.N_LAYERS * F.CTX * W.HEAD_DIM      # 1 B/elem at fp8
    actual = sum(F.kv_bytes_per_seq(1))
    assert 5.0 < uncompressed / actual < 6.5


# =====================================================================
# 7. FIT: the two numbers, kept apart
# =====================================================================

def test_capacity_lower_bound_fits_one_h200():
    rep, shd, _ = F.per_gpu_weight_bytes()
    lower = (rep + shd + sum(F.kv_bytes_per_seq(1)) * F.BATCH
             + F.compressor_state_bytes()) / GB
    assert lower < F.GPU_CAP_GB
    assert 116.0 < lower < 117.0


def test_fp8_expert_conversion_does_not_fit_one_node():
    """The minimal-shape argument depends on this being impossible, not merely tight.

    Compare WEIGHTS ALONE against raw capacity. Adding assumed workspace here
    would make an assumption load-bearing in an argument that does not need one:
    the weights overflow on their own.
    """
    delta = sum(
        W.fp8_linear(o, i)[0] - W.fp4_linear(o, i)[0]
        for _ in range(W.N_LAYERS + W.N_MTP)
        for _ in range(W.N_EXPERTS)
        for o, i in [(W.MOE_INTER, W.DIM), (W.DIM, W.MOE_INTER), (W.MOE_INTER, W.DIM)]
    )
    rep, shd, _ = F.per_gpu_weight_bytes()
    weights_only = (rep + shd + delta / F.TP) / GB
    assert 92.0 < delta / F.TP / GB < 92.5, "the +92.17 GB/GPU figure drifted"
    assert 206.0 < weights_only < 207.5, f"weights-only {weights_only:.2f} GB drifted"
    assert weights_only > F.GPU_CAP_GB * 1.4, "overflow must be decisive, not marginal"


def test_fp8_expert_conversion_fits_two_nodes_at_tp16():
    """The other half of the minimal-shape claim: TP16 must actually work."""
    delta = sum(
        W.fp8_linear(o, i)[0] - W.fp4_linear(o, i)[0]
        for _ in range(W.N_LAYERS + W.N_MTP)
        for _ in range(W.N_EXPERTS)
        for o, i in [(W.MOE_INTER, W.DIM), (W.DIM, W.MOE_INTER), (W.MOE_INTER, W.DIM)]
    )
    rep, shd, _ = F.per_gpu_weight_bytes()
    # replicated bytes do not shrink with more ranks; only the sharded part does
    sharded_total = shd * F.TP + delta
    at_tp16 = (rep + sharded_total / 16) / GB
    assert at_tp16 < F.GPU_CAP_GB, f"TP16 gives {at_tp16:.2f} GB/GPU, still over"
    assert 105.0 < at_tp16 < 109.0


def test_tp16_divisibility_holds_for_the_minimal_shape():
    """The two-node recommendation is only valid if every dimension divides by 16."""
    for name, value in [("experts", W.N_EXPERTS), ("heads", W.N_HEADS),
                        ("o_groups", W.O_GROUPS), ("vocab", W.VOCAB)]:
        assert value % 16 == 0, f"{name}={value} does not divide by 16"


# =====================================================================
# 8. COLLECTIVES
# =====================================================================

def test_decode_graph_has_154_collectives_not_156():
    """Speculation is disabled at the baseline, so the MTP block is RESIDENT but
    never executes. It therefore contributes 0 collectives to the decode step.
    Counting all 62 blocks gives 156 and bills a block that does not run."""
    blocks_that_run = W.N_LAYERS                      # 61, MTP excluded
    indexer = W.RATIOS[:W.N_LAYERS].count(4)          # 30
    endpoints = 2
    total = 2 * blocks_that_run + indexer + endpoints
    assert total == 154
    with_mtp = 2 * (W.N_LAYERS + W.N_MTP) + indexer + endpoints
    assert with_mtp == 156, "156 is the speculation-enabled count"


def test_indexer_adds_the_third_collective_on_half_the_layers():
    """A plain TP convention of 2 per layer predicts 124 and misses these."""
    conventional = 2 * W.N_LAYERS + 2
    actual = 2 * W.N_LAYERS + W.RATIOS[:W.N_LAYERS].count(4) + 2
    assert actual - conventional == 30


# =====================================================================
# runner
# =====================================================================

def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed, failed = 0, []
    print(f"running {len(tests)} tests\n")
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            failed.append((name, e, traceback.format_exc()))
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print()
        for name, e, tb in failed:
            print(f"--- {name} ---")
            print(tb)
        return 1
    return 0


# =====================================================================
# 9. D1 PLANNER YAML
#
# The spec is a deliverable, so it gets checked like one. A hand-written
# YAML that drifts from config.json is worse than no YAML, because it
# looks authoritative. Parsed with a deliberately small extractor rather
# than pyyaml so the suite keeps its no-dependency promise.
# =====================================================================

YAML_PATH = os.path.join(ROOT, "DELIVERABLES", "deepseek-v4-pro.yaml")


def _yaml_text():
    return open(YAML_PATH).read()


def _scalar(key):
    """Pull `  key: value` from the spec block. Ints come back as ints."""
    import re
    m = re.search(rf"^\s+{re.escape(key)}:\s*([^\s#][^#\n]*?)\s*(?:#.*)?$",
                  _yaml_text(), re.M)
    assert m, f"key {key!r} not found in {YAML_PATH}"
    v = m.group(1).strip()
    if v.lstrip("-").isdigit():
        return int(v)
    if v in ("true", "false"):
        return v == "true"
    return v


def _list(key):
    """Pull `key: [ ... ]` including the multi-line form."""
    import re
    t = _yaml_text()
    i = t.index(f"{key}: [")
    j = t.index("]", i)
    body = t[i + len(key) + 3:j]
    body = re.sub(r"#[^\n]*", "", body)          # strip trailing comments
    return [x.strip() for x in body.replace("\n", " ").split(",") if x.strip()]


def test_yaml_scalars_match_config_json():
    """Every scalar the spec restates must equal the checkpoint's own value."""
    pairs = [
        ("hidden", "hidden_size"), ("n_layers", "num_hidden_layers"),
        ("vocab", "vocab_size"), ("n_heads", "num_attention_heads"),
        ("num_kv_heads", "num_key_value_heads"), ("head_dim", "head_dim"),
        ("qk_rope_head_dim", "qk_rope_head_dim"),
        ("q_lora_rank", "q_lora_rank"), ("o_lora_rank", "o_lora_rank"),
        ("o_groups", "o_groups"), ("num_experts", "n_routed_experts"),
        ("num_experts_per_tok", "num_experts_per_tok"),
        ("n_shared_experts", "n_shared_experts"),
        ("moe_intermediate_size", "moe_intermediate_size"),
        ("n_hash_layers", "num_hash_layers"),
        ("index_n_heads", "index_n_heads"), ("index_head_dim", "index_head_dim"),
        ("index_topk", "index_topk"), ("hc_mult", "hc_mult"),
        ("hc_sinkhorn_iters", "hc_sinkhorn_iters"),
        ("sliding_window", "sliding_window"),
        ("rope_theta", "rope_theta"),
        ("compress_rope_theta", "compress_rope_theta"),
        ("max_position_embeddings", "max_position_embeddings"),
        ("mtp_num_hidden_layers", "num_nextn_predict_layers"),
    ]
    for yaml_key, cfg_key in pairs:
        assert _scalar(yaml_key) == W.cfg[cfg_key], (
            f"{yaml_key}: yaml={_scalar(yaml_key)} config={W.cfg[cfg_key]}"
        )


def test_yaml_compress_ratios_match_config_exactly():
    got = [int(x) for x in _list("compress_ratios")]
    assert got == W.RATIOS
    assert len(got) == 62


def test_yaml_layer_types_derive_from_compress_ratios():
    """The written-out schedule must be reproducible from the ratios, or one of
    the two is a transcription error."""
    lt = _list("layer_types")
    assert len(lt) == W.N_LAYERS == 61
    derived = ["sparse_indexed" if W.RATIOS[i] == 4 else "sparse_strided"
               for i in range(W.N_LAYERS)]
    assert lt == derived


def test_yaml_declared_layer_counts_match_the_schedule():
    lt = _list("layer_types")
    assert _scalar("n_sparse_indexed_layers") == lt.count("sparse_indexed") == 30
    assert _scalar("n_sparse_strided_layers") == lt.count("sparse_strided") == 31


def test_yaml_collective_count_is_internally_consistent():
    lt = _list("layer_types")
    derived = 2 * W.N_LAYERS + lt.count("sparse_indexed") + 2
    assert _scalar("collectives_per_decode_step") == derived == 154


def test_yaml_hash_layers_match_declared_count():
    hl = [int(x) for x in _list("hash_layers")]
    assert len(hl) == _scalar("n_hash_layers") == W.N_HASH == 3
    assert hl == [0, 1, 2]


def test_yaml_replicated_bytes_match_the_fit_model():
    rep, _, _ = F.per_gpu_weight_bytes()
    declared = float(_scalar("replicated_bytes_per_gpu_gb"))
    assert abs(declared - rep / GB) < 0.05


def test_yaml_uses_checkpoint_tensor_names_not_hf_names():
    """The case requires the checkpoint's own vocabulary, not renamed into
    nicer words. Comments are exempt: the spec deliberately names the HF
    conventions to say it is not using them, and that is the opposite of the
    drift this guards against."""
    import re
    t = _yaml_text()
    for verbatim in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b", "tid2eid",
                     "hc_attn_fn", "ffn.shared_experts", "attn.indexer"]:
        assert verbatim in t, f"lost the checkpoint name {verbatim!r}"
    uncommented = re.sub(r"#[^\n]*", "", t)
    for hf_name in ["q_proj", "o_proj", "gate_proj", "down_proj", "mlp."]:
        assert hf_name not in uncommented, (
            f"HF name {hf_name!r} leaked into a field, not just a comment"
        )


def test_yaml_has_no_dashes_per_house_style():
    t = _yaml_text()
    assert "—" not in t and "–" not in t


if __name__ == "__main__":
    sys.exit(main())
