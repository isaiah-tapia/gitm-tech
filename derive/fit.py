"""
DeepSeek-V4-Pro — per-GPU memory fit at the baseline workload.

Hardware: 8x H200 SXM (141 GB, 4.8 TB/s, NVLink), TP8.
  The published vendor recipe (recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro) targets
  8x MI355X 288 GB instead. We price the ASSIGNED hardware and treat the divergence
  as a finding -- see DESIGN-NOTE. The recipe's software settings are still used
  where they are hardware-independent (kv-cache-dtype fp8, gpu-memory-utilization 0.9).

Baseline: 32 sequences, 8,192 cached tokens each, 1 new token/seq/step,
          text-only, speculation disabled.

The load-bearing correctness point here is SHARDING. Per-GPU weight bytes are NOT
total/8: model.py uses plain `Linear` (replicated) for wq_a, wkv, the compressors,
and -- most expensively -- shared_experts, while only ColumnParallel/RowParallel
and the routed expert bank actually shard. Dividing the 864.70 GB total by 8 gives
108.09 GB/GPU and understates the real figure.

Run:  python3 derive/fit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weights as W  # noqa: E402

TP = 8
GPU_CAP_GB = 141.0        # H200 SXM HBM3e
GPU_UTIL = 0.9            # recipe: --gpu-memory-utilization 0.9

BATCH = 32
CTX = 8192

WINDOW = W.cfg["sliding_window"]          # 128
KV_DIM = W.HEAD_DIM                       # 512 latent, num_key_value_heads == 1
IDX_DIM = W.IDX_HD                        # 128


# --------------------------------------------------------------- weights
def per_gpu_weight_bytes():
    """Walk the same structure as weights.py, but split each tensor into
    replicated (every rank holds a full copy) vs sharded (bytes/TP).

    Classification is read directly from model.py's module choice:
      Linear                -> replicated
      ColumnParallelLinear  -> sharded on out_features
      RowParallelLinear     -> sharded on in_features
      ParallelEmbedding     -> sharded on vocab
      MoE.experts           -> sharded (n_routed_experts // world_size)
      nn.Parameter(n_local_*) -> sharded
      everything else (norms, hc_*, gate, ape) -> replicated
    """
    rep = 0   # bytes replicated on EVERY gpu
    shd = 0   # bytes summed across all gpus, to be divided by TP

    def R(b):
        nonlocal rep
        rep += b

    def S(b):
        nonlocal shd
        shd += b

    # embed (ParallelEmbedding, vocab-sharded) + head (ParallelHead, vocab-sharded)
    S(W.bf16_linear(W.VOCAB, W.DIM)[0])          # embed.weight
    S(W.bf16_linear(W.VOCAB, W.DIM)[0])          # head.weight
    R(W.rmsnorm(W.DIM)[0])                       # norm.weight
    R(W.hc_head_bytes()[0])                      # model-level hc_head_*

    def block(layer_id, ratio):
        # ---- attention
        S(W.N_HEADS * W.B_FP32)                                        # attn_sink (n_local_heads)
        R(W.fp8_linear(W.Q_LORA, W.DIM)[0])                            # wq_a       Linear -> REPLICATED
        S(W.fp8_linear(W.N_HEADS * W.HEAD_DIM, W.Q_LORA)[0])           # wq_b       ColumnParallel
        R(W.fp8_linear(W.HEAD_DIM, W.DIM)[0])                          # wkv        Linear -> REPLICATED
        S(W.fp8_linear(W.O_GROUPS * W.O_LORA,
                       W.N_HEADS * W.HEAD_DIM // W.O_GROUPS)[0])       # wo_a       ColumnParallel
        S(W.fp8_linear(W.DIM, W.O_GROUPS * W.O_LORA)[0])               # wo_b       RowParallel
        R(W.rmsnorm(W.Q_LORA)[0] + W.rmsnorm(W.HEAD_DIM)[0])           # q_norm, kv_norm
        if ratio:
            R(W.compressor_bytes(ratio, W.HEAD_DIM)[0])                # Linear+ape -> REPLICATED
            if ratio == 4:
                S(W.fp8_linear(W.IDX_HEADS * W.IDX_HD, W.Q_LORA)[0])   # indexer.wq_b        ColumnParallel
                S(W.bf16_linear(W.IDX_HEADS, W.DIM)[0])                # indexer.weights_proj ColumnParallel
                R(W.compressor_bytes(4, W.IDX_HD)[0])                  # indexer.compressor  REPLICATED
        # ---- moe
        R(W.bf16_linear(W.N_EXPERTS, W.DIM)[0])                        # gate.weight
        if layer_id < W.N_HASH:
            R(W.VOCAB * W.N_ACT * W.B_I64)                             # gate.tid2eid (int64)
        else:
            R(W.N_EXPERTS * W.B_FP32)                                  # gate.bias
        for _ in range(W.N_EXPERTS):                                   # routed experts -> SHARDED
            for o, i in [(W.MOE_INTER, W.DIM), (W.DIM, W.MOE_INTER), (W.MOE_INTER, W.DIM)]:
                S(W.fp4_linear(o, i)[0])
        for _ in range(W.N_SHARED):                                    # shared expert -> REPLICATED
            for o, i in [(W.MOE_INTER, W.DIM), (W.DIM, W.MOE_INTER), (W.MOE_INTER, W.DIM)]:
                R(W.fp8_linear(o, i)[0])
        # ---- norms + hyper-connections
        R(2 * W.rmsnorm(W.DIM)[0])
        R(W.hc_block_bytes()[0])

    for lid in range(W.N_LAYERS):
        block(lid, W.RATIOS[lid])

    # MTP block is resident even with speculation disabled (it is in the checkpoint).
    mtp_rep_before, mtp_shd_before = rep, shd
    for i in range(W.N_MTP):
        lid = W.N_LAYERS + i
        block(lid, W.RATIOS[lid])
        R(2 * W.fp8_linear(W.DIM, W.DIM)[0])    # e_proj, h_proj (plain Linear)
        R(3 * W.rmsnorm(W.DIM)[0])              # enorm, hnorm, norm
        R(W.hc_head_bytes()[0])
    mtp_bytes = (rep - mtp_rep_before) + (shd - mtp_shd_before) / TP

    return rep, shd / TP, mtp_bytes


# -------------------------------------------------------------------- kv
def kv_bytes_per_seq(elem_bytes):
    """model.py Attention: kv_cache is [max_batch, window + max_seq//ratio, head_dim].
    The window term is O(1) in context -- it never grows. Only the compressed term
    scales with context. Counting the whole cache per-token is the classic error.

    num_key_value_heads == 1 and wkv is a plain `Linear`, so the latent KV is
    REPLICATED across TP ranks, not sharded: every rank needs the full 512-dim
    latent to attend with its local query heads.
    """
    fixed = 0      # window part, independent of context length
    grow = 0       # compressed part, scales with context
    for lid in range(W.N_LAYERS):
        ratio = W.RATIOS[lid]
        fixed += WINDOW * KV_DIM
        if ratio:
            grow += (CTX // ratio) * KV_DIM
            if ratio == 4:
                grow += (CTX // ratio) * IDX_DIM        # Indexer.kv_cache
    return fixed * elem_bytes, grow * elem_bytes


def compressor_state_bytes():
    """Compressor.kv_state + score_state, both fp32, both [batch, coff*ratio, coff*head_dim].
    Sized by RATIO, not by context -- and the ratio-128 layers carry the larger
    buffer (128 slots vs 8), which is the opposite of the intuition."""
    tot = 0
    for lid in range(W.N_LAYERS):
        ratio = W.RATIOS[lid]
        if not ratio:
            continue
        coff = 1 + (ratio == 4)
        tot += 2 * BATCH * (coff * ratio) * (coff * W.HEAD_DIM) * W.B_FP32
        if ratio == 4:
            tot += 2 * BATCH * (2 * 4) * (2 * W.IDX_HD) * W.B_FP32     # indexer's compressor
    return tot


# ------------------------------------------------------------------ main
def main():
    GB = 1e9
    rep, shd, mtp = per_gpu_weight_bytes()
    wt = rep + shd

    print("=" * 74)
    print("WEIGHTS PER GPU  (TP8)")
    print("=" * 74)
    print(f"  replicated on every rank          {rep/GB:9.2f} GB")
    print(f"  sharded  (total {(shd*TP)/GB:.2f} GB / {TP})   {shd/GB:9.2f} GB")
    print(f"  {'-'*40}")
    print(f"  weights per GPU                   {wt/GB:9.2f} GB")
    print(f"  naive total/8 would say           {864.704792696/TP:9.2f} GB"
          f"   <- understates by {(wt/GB)-(864.704792696/TP):.2f} GB")
    print(f"  of which MTP block (resident,")
    print(f"    absent from decode graph)       {mtp/GB:9.2f} GB")
    print()

    for name, eb in (("bf16 (checkpoint/reference)", 2), ("fp8  (vendor recipe)", 1)):
        fixed, grow = kv_bytes_per_seq(eb)
        per_seq = fixed + grow
        print(f"KV CACHE @ {name}")
        print(f"  fixed  (window, O(1) in ctx)   {fixed/1e6:9.2f} MB/seq")
        print(f"  growing(compressed @ {CTX} tok) {grow/1e6:9.2f} MB/seq")
        print(f"  per sequence                   {per_seq/1e6:9.2f} MB")
        print(f"  x{BATCH} seqs, REPLICATED per rank {per_seq*BATCH/GB:9.2f} GB/GPU")
        print()

    kv_fp8 = sum(kv_bytes_per_seq(1)) * BATCH
    kv_bf16 = sum(kv_bytes_per_seq(2)) * BATCH
    cstate = compressor_state_bytes()

    # --- activation workspace + comm buffers: ASSUMED, not derived.
    # Decode activations are genuinely tiny (batch 32 x 1 token): the hc-expanded
    # hidden state is 32*1*4*7168*2 = 1.8 MB, logits 32*129280*4 = 16.6 MB, and the
    # all_gather of logits across TP8 is 8x that. The dominant term is whatever
    # fixed workspace the engine reserves, which the checkpoint cannot tell us.
    act_ws = 4.0 * GB          # ASSUMED
    comm_buf = 2.0 * GB        # ASSUMED (NCCL buffers, graph pools)
    reserve = 6.0 * GB         # EXPLICIT reserve: fragmentation + headroom

    usable = GPU_CAP_GB * GPU_UTIL * GB

    print("=" * 74)
    print(f"PER-GPU FIT @ H200 141 GB, util {GPU_UTIL}  ->  {usable/GB:.1f} GB usable")
    print("=" * 74)
    rows = [
        ("weights (replicated + sharded)", wt, "derived"),
        ("  incl. FP4 expert scale metadata", None, ""),
        ("KV cache, fp8 (recipe)", kv_fp8, "derived"),
        ("compressor kv_state/score_state fp32", cstate, "derived"),
        ("activation workspace", act_ws, "ASSUMED"),
        ("communication buffers", comm_buf, "ASSUMED"),
        ("reserve (fragmentation/headroom)", reserve, "explicit"),
    ]
    tot = 0
    for label, b, tag in rows:
        if b is None:
            print(f"  {label:<40}{'':>10}   {tag}")
            continue
        tot += b
        print(f"  {label:<40}{b/GB:>9.2f} GB   {tag}")
    print(f"  {'-'*40}{'-'*13}")
    print(f"  {'TOTAL':<40}{tot/GB:>9.2f} GB")
    print(f"  {'usable @ util 0.9':<40}{usable/GB:>9.2f} GB")
    print(f"  {'headroom':<40}{(usable-tot)/GB:>9.2f} GB")
    print()
    print(f"  FITS (deployable config)  -> {'YES' if tot <= usable else 'NO'}")

    lower = (wt + kv_fp8 + cstate) / GB
    print(f"  capacity LOWER BOUND (weights+KV+state only, no workspace/reserve):")
    print(f"      {lower:.2f} GB vs {GPU_CAP_GB} GB raw  -> "
          f"{'FITS' if lower <= GPU_CAP_GB else 'DOES NOT FIT'}"
          f"   ({lower/GPU_CAP_GB*100:.1f}% of raw capacity)")
    print()
    print(f"  bf16 KV instead of fp8 would add {(kv_bf16-kv_fp8)/GB:.2f} GB -> total "
          f"{(tot+kv_bf16-kv_fp8)/GB:.2f} GB "
          f"({'still fits' if tot+kv_bf16-kv_fp8 <= usable else 'DOES NOT FIT'})")

    # FP8 expert conversion (convert.py --expert-dtype fp8)
    fp8_expert_delta = 0
    for _ in range(W.N_LAYERS + W.N_MTP):
        for _ in range(W.N_EXPERTS):
            for o, i in [(W.MOE_INTER, W.DIM), (W.DIM, W.MOE_INTER), (W.MOE_INTER, W.DIM)]:
                fp8_expert_delta += W.fp8_linear(o, i)[0] - W.fp4_linear(o, i)[0]
    print(f"  FP8 expert conversion would add {fp8_expert_delta/TP/GB:.2f} GB/GPU -> total "
          f"{(tot+fp8_expert_delta/TP)/GB:.2f} GB "
          f"({'fits' if tot+fp8_expert_delta/TP <= usable else 'DOES NOT FIT'})")


if __name__ == "__main__":
    main()


def replicated_breakdown():
    """What is actually replicated, largest first -- the 6.46 GB/GPU that total/8 misses."""
    GB = 1e9
    L = W.N_LAYERS + W.N_MTP
    n4 = sum(1 for r in W.RATIOS[:W.N_LAYERS] if r == 4)
    n128 = sum(1 for r in W.RATIOS[:W.N_LAYERS] if r == 128)
    items = [
        ("shared_experts (fp8, plain Linear)",
         L * 3 * W.fp8_linear(W.MOE_INTER, W.DIM)[0]),
        ("attn.compressor wkv+wgate+ape+norm",
         n4 * W.compressor_bytes(4, W.HEAD_DIM)[0] + n128 * W.compressor_bytes(128, W.HEAD_DIM)[0]),
        ("wq_a (plain Linear)", L * W.fp8_linear(W.Q_LORA, W.DIM)[0]),
        ("hc_attn_fn + hc_ffn_fn (fp32)", L * W.hc_block_bytes()[0]),
        ("indexer.compressor", n4 * W.compressor_bytes(4, W.IDX_HD)[0]),
        ("wkv (plain Linear)", L * W.fp8_linear(W.HEAD_DIM, W.DIM)[0]),
        ("gate.weight", L * W.bf16_linear(W.N_EXPERTS, W.DIM)[0]),
        ("gate.tid2eid (3 hash layers, int64)", 3 * W.VOCAB * W.N_ACT * W.B_I64),
    ]
    print()
    print("=" * 74)
    print("WHAT IS REPLICATED  (why per-GPU != total/8)")
    print("=" * 74)
    for label, b in sorted(items, key=lambda x: -x[1]):
        print(f"  {label:<44}{b/GB:>8.3f} GB")
    print(f"  {'-'*44}{'-'*11}")
    print(f"  {'subtotal (norms/sinks/bias omitted)':<44}{sum(b for _, b in items)/GB:>8.3f} GB")


if __name__ == "__main__":
    replicated_breakdown()
