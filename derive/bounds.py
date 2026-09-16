"""
DeepSeek-V4-Pro -- per-decode-step cost model at the baseline workload.

8x H200 SXM, TP8, NVLink. 32 sequences, 8,192 cached tokens, 1 new token/seq.

Everything here is a ROOFLINE FLOOR: a lower bound on time assuming perfect
overlap and vendor peak. Real execution is slower. A predicted floor that came
out ABOVE a measured number would mean an arithmetic error, so the comparison
against PR #53709's measured decode throughput at the bottom is a real check.

Precision note that drives the whole ranking: on SM90 the MXFP4 experts run
through Marlin W4A16, so expert MATH is BF16 rate (989.5 TF/s) while expert
MEMORY traffic stays at the FP4 rate. Storage precision and compute precision
are different numbers here and both matter.

Run:  python3 derive/bounds.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weights as W   # noqa: E402
import fit as F       # noqa: E402

TP = 8
BATCH = 32
CTX = 8192

# ---- H200 SXM peaks. Dense, not "with sparsity" (datasheet doubles for that).
HBM_BW       = 4.8e12      # B/s
FP8_PEAK     = 1979e12     # FLOP/s
BF16_PEAK    = 989.5e12    # FLOP/s
FP32_PEAK    = 67e12       # FLOP/s  (CUDA cores; the router and hc live here)
NVLINK_BW    = 900e9       # B/s per GPU, bidirectional convention

# ---- launch model
GRAPH_DISPATCH = 0.5e-6    # ASSUMED s/kernel GPU-side dispatch under CUDA graph
EAGER_LAUNCH   = 5.0e-6    # ASSUMED s/kernel CPU launch, no graph

KV_B = 1                   # recipe: --kv-cache-dtype fp8
IDX_KV_B = 2               # ASSUMED bf16: Indexer.kv_cache is a separate buffer
                           # allocated at default dtype, not covered by the flag

N4 = W.RATIOS[:W.N_LAYERS].count(4)      # 30
N128 = W.RATIOS[:W.N_LAYERS].count(128)  # 31
BLOCKS = W.N_LAYERS                      # MTP excluded: speculation disabled

INDEX_TOPK = W.cfg["index_topk"]         # 1024
WINDOW = W.cfg["sliding_window"]         # 128


def gemm_flops(m, n, k):
    return 2 * m * n * k


# =====================================================================
# 1. HBM traffic per rank per step
# =====================================================================
def expected_experts_touched():
    """32 tokens x top-6 = 192 selections over 384 experts, per layer.
    Expected DISTINCT experts touched = N*(1-(1-1/N)^S). Uniform-routing
    assumption -- real imbalance only increases this. Not derivable from the
    checkpoint; see EXPERIMENT 1."""
    n, s = W.N_EXPERTS, BATCH * W.N_ACT
    return n * (1 - (1 - 1 / n) ** s)


def expert_bytes_each():
    """One routed expert: w1, w2, w3 in MXFP4 plus e8m0 scales."""
    return sum(W.fp4_linear(o, i)[0]
               for o, i in [(W.MOE_INTER, W.DIM),
                            (W.DIM, W.MOE_INTER),
                            (W.MOE_INTER, W.DIM)])


def hbm_traffic():
    """Bytes read from HBM per rank per decode step, by category."""
    rep = lambda b: b              # replicated: full bytes on every rank
    shd = lambda b: b / TP         # sharded: 1/TP of the bytes

    # ---- attention + compressor + indexer weights, read EVERY step
    attn = 0
    attn += BLOCKS * rep(W.fp8_linear(W.Q_LORA, W.DIM)[0])                     # wq_a
    attn += BLOCKS * shd(W.fp8_linear(W.N_HEADS * W.HEAD_DIM, W.Q_LORA)[0])    # wq_b
    attn += BLOCKS * rep(W.fp8_linear(W.HEAD_DIM, W.DIM)[0])                   # wkv
    attn += BLOCKS * shd(W.fp8_linear(W.O_GROUPS * W.O_LORA,
                                      W.N_HEADS * W.HEAD_DIM // W.O_GROUPS)[0])  # wo_a
    attn += BLOCKS * shd(W.fp8_linear(W.DIM, W.O_GROUPS * W.O_LORA)[0])        # wo_b
    attn += N4 * rep(W.compressor_bytes(4, W.HEAD_DIM)[0])
    attn += N128 * rep(W.compressor_bytes(128, W.HEAD_DIM)[0])
    attn += N4 * shd(W.fp8_linear(W.IDX_HEADS * W.IDX_HD, W.Q_LORA)[0])        # indexer wq_b
    attn += N4 * shd(W.bf16_linear(W.IDX_HEADS, W.DIM)[0])                     # weights_proj
    attn += N4 * rep(W.compressor_bytes(4, W.IDX_HD)[0])

    # ---- mHC + router + norms, replicated, read every step
    misc = BLOCKS * rep(W.hc_block_bytes()[0])
    misc += BLOCKS * rep(W.bf16_linear(W.N_EXPERTS, W.DIM)[0])                 # gate.weight
    misc += BLOCKS * rep(2 * W.rmsnorm(W.DIM)[0])

    # ---- shared expert: fires for EVERY token, replicated on every rank
    shared = BLOCKS * rep(sum(W.fp8_linear(o, i)[0]
                              for o, i in [(W.MOE_INTER, W.DIM),
                                           (W.DIM, W.MOE_INTER),
                                           (W.MOE_INTER, W.DIM)]))

    # ---- routed experts: only the ones actually selected
    touched_per_rank = expected_experts_touched() / TP
    routed = BLOCKS * touched_per_rank * expert_bytes_each()

    # ---- KV reads. sparse_attn gathers only selected positions.
    #      ratio-4   layers: window + index_topk (indexer picks 1024 of 2048)
    #      ratio-128 layers: window + every compressed slot (ctx/128 = 64)
    kv = 0
    kv += N4 * BATCH * (WINDOW + min(INDEX_TOPK, CTX // 4)) * W.HEAD_DIM * KV_B
    kv += N128 * BATCH * (WINDOW + CTX // 128) * W.HEAD_DIM * KV_B
    # the indexer must SCAN the whole compressed cache to pick its top-k
    idx_scan = N4 * BATCH * (CTX // 4) * W.IDX_HD * IDX_KV_B

    # ---- lm_head, read once per step for the logits GEMM
    head = shd(W.bf16_linear(W.VOCAB, W.DIM)[0])

    return {
        "routed experts (selected only)": routed,
        "shared expert (all tokens)": shared,
        "attention + compressor + indexer W": attn,
        "lm_head": head,
        "KV gather (sparse_attn)": kv,
        "KV scan (indexer top-k)": idx_scan,
        "mHC + router + norms": misc,
    }


# =====================================================================
# 2. FLOPs per rank per step, split by the precision each op runs at
# =====================================================================
def flops():
    m = BATCH
    local_heads = W.N_HEADS // TP
    local_groups = W.O_GROUPS // TP
    local_idx_heads = W.IDX_HEADS // TP

    # ---- FP8 tensor core: backbone projections
    fp8 = 0
    fp8 += BLOCKS * gemm_flops(m, W.Q_LORA, W.DIM)                              # wq_a
    fp8 += BLOCKS * gemm_flops(m, local_heads * W.HEAD_DIM, W.Q_LORA)           # wq_b
    fp8 += BLOCKS * gemm_flops(m, W.HEAD_DIM, W.DIM)                            # wkv
    fp8 += BLOCKS * gemm_flops(m, local_groups * W.O_LORA,
                               W.N_HEADS * W.HEAD_DIM // W.O_GROUPS)            # wo_a
    fp8 += BLOCKS * gemm_flops(m, W.DIM, W.O_GROUPS * W.O_LORA // TP)           # wo_b
    fp8 += N4 * gemm_flops(m, local_idx_heads * W.IDX_HD, W.Q_LORA)             # indexer wq_b
    fp8 += BLOCKS * 3 * gemm_flops(m, W.MOE_INTER, W.DIM)                       # shared expert

    # ---- BF16 tensor core: Marlin W4A16 experts, compressors, attention core, head
    bf16 = 0
    pairs_per_rank = BATCH * W.N_ACT / TP          # token-expert pairs this rank owns
    bf16 += BLOCKS * pairs_per_rank * 3 * gemm_flops(1, W.MOE_INTER, W.DIM)     # routed experts
    bf16 += N4 * 2 * gemm_flops(m, 2 * W.HEAD_DIM, W.DIM)                       # compressor wkv+wgate
    bf16 += N128 * 2 * gemm_flops(m, W.HEAD_DIM, W.DIM)
    bf16 += N4 * 2 * gemm_flops(m, 2 * W.IDX_HD, W.DIM)                         # indexer compressor
    # attention core: scores + output, over the gathered positions
    bf16 += N4 * 2 * gemm_flops(m * local_heads, WINDOW + INDEX_TOPK, W.HEAD_DIM)
    bf16 += N128 * 2 * gemm_flops(m * local_heads, WINDOW + CTX // 128, W.HEAD_DIM)
    # indexer scoring einsum over the full compressed cache
    bf16 += N4 * gemm_flops(m * local_idx_heads, CTX // 4, W.IDX_HD)
    bf16 += gemm_flops(m, W.VOCAB // TP, W.DIM)                                 # lm_head

    # ---- FP32 CUDA core: router and mHC mixing
    fp32 = 0
    fp32 += BLOCKS * gemm_flops(m, W.N_EXPERTS, W.DIM)                          # gate
    fp32 += BLOCKS * 2 * gemm_flops(m, (2 + W.HC_MULT) * W.HC_MULT,
                                    W.HC_MULT * W.DIM)                          # hc mixes
    return {"fp8": fp8, "bf16": bf16, "fp32": fp32}


# =====================================================================
# 3. Collectives
# =====================================================================
def collectives():
    """Ring all-reduce moves 2(N-1)/N x payload per rank. all_gather moves
    (N-1)/N x the full result."""
    ar = 2 * (TP - 1) / TP
    ag = (TP - 1) / TP
    items = []
    # wo_b: RowParallelLinear casts to fp32 BEFORE the all-reduce -> 4 B/elem
    items.append(("wo_b all-reduce (fp32)", BLOCKS, BATCH * W.DIM * 4, ar))
    items.append(("MoE all-reduce (fp32)", BLOCKS, BATCH * W.DIM * 4, ar))
    items.append(("indexer all-reduce (fp32)", N4, BATCH * (CTX // 4) * 4, ar))
    items.append(("embed all-reduce (bf16)", 1, BATCH * W.DIM * 2, ar))
    items.append(("logits all-gather (fp32)", 1, BATCH * W.VOCAB * 4, ag))
    rows, total_bytes, count = [], 0, 0
    for name, n, payload, factor in items:
        moved = n * payload * factor
        rows.append((name, n, payload, moved))
        total_bytes += moved
        count += n
    return rows, total_bytes, count


# =====================================================================
# 4. Launch count
# =====================================================================
def launches():
    """Kernel count per step. ASSUMED op counts read off model.py's forward,
    adjusted for vLLM's fused MoE (reference impl loops over experts in Python;
    vLLM uses a grouped GEMM). This is the softest number in the file."""
    per_block = 0
    per_block += 2 * 6      # hc_pre x2: linear, rsqrt, sinkhorn(1 fused), sum...
    per_block += 2 * 3      # hc_post x2
    per_block += 2          # attn_norm, ffn_norm
    per_block += 14         # attention core path
    per_block += 8          # compressor
    per_block += 8          # gate
    per_block += 5          # fused MoE: permute, gemm w1w3, act, gemm w2, unpermute
    per_block += 5          # shared expert
    per_block += 2          # the two all-reduces
    idx_extra = 18          # indexer path, ratio-4 layers only
    total = BLOCKS * per_block + N4 * idx_extra + 4   # +embed, head, norm, allgather
    return total, per_block


# =====================================================================
def main():
    GB, MB, ms, us = 1e9, 1e6, 1e-3, 1e-6

    traffic = hbm_traffic()
    tot_bytes = sum(traffic.values())
    fl = flops()
    rows, comm_bytes, comm_count = collectives()
    n_launch, per_block = launches()

    print("=" * 78)
    print("HBM TRAFFIC PER RANK PER DECODE STEP")
    print("=" * 78)
    for k, v in sorted(traffic.items(), key=lambda x: -x[1]):
        print(f"  {k:<38}{v/GB:>9.3f} GB{v/tot_bytes*100:>8.1f}%")
    print(f"  {'-'*38}{'-'*20}")
    print(f"  {'TOTAL':<38}{tot_bytes/GB:>9.3f} GB")
    t_mem = tot_bytes / HBM_BW
    print(f"  at {HBM_BW/1e12:.1f} TB/s  ->  {t_mem/ms:.3f} ms")
    print(f"  experts touched per rank per layer: "
          f"{expected_experts_touched()/TP:.1f} of {W.N_EXPERTS//TP}")
    print()

    print("=" * 78)
    print("COMPUTE PER RANK PER DECODE STEP")
    print("=" * 78)
    peaks = {"fp8": FP8_PEAK, "bf16": BF16_PEAK, "fp32": FP32_PEAK}
    t_comp = 0
    for k in ("fp8", "bf16", "fp32"):
        t = fl[k] / peaks[k]
        t_comp += t
        print(f"  {k:<8}{fl[k]/1e9:>10.2f} GFLOP   @ {peaks[k]/1e12:>7.1f} TF/s"
              f"  ->{t/ms:>8.4f} ms")
    print(f"  {'-'*58}")
    print(f"  {'total':<8}{sum(fl.values())/1e9:>10.2f} GFLOP"
          f"{'':>24}->{t_comp/ms:>8.4f} ms")
    ai = tot_bytes and sum(fl.values()) / tot_bytes
    print(f"  arithmetic intensity: {ai:.2f} FLOP/byte")
    print(f"  BF16 ridge = {BF16_PEAK/HBM_BW:.0f} FLOP/byte   "
          f"-> {'MEMORY' if ai < BF16_PEAK/HBM_BW else 'COMPUTE'} bound, "
          f"by {BF16_PEAK/HBM_BW/ai:.0f}x")
    print()

    print("=" * 78)
    print("COLLECTIVES PER RANK PER DECODE STEP")
    print("=" * 78)
    for name, n, payload, moved in rows:
        print(f"  {name:<30}{n:>4} x {payload/MB:>7.3f} MB  -> {moved/MB:>8.3f} MB")
    print(f"  {'-'*30}{'-'*34}")
    print(f"  {'TOTAL':<30}{comm_count:>4} ops{'':>16}{comm_bytes/MB:>8.3f} MB")
    t_comm = comm_bytes / NVLINK_BW
    print(f"  at {NVLINK_BW/1e9:.0f} GB/s NVLink  ->  {t_comm/ms:.4f} ms")
    print(f"  per-collective latency floor is NOT included; see EXPERIMENT 2")
    print()

    print("=" * 78)
    print("LAUNCH OVERHEAD PER DECODE STEP")
    print("=" * 78)
    t_graph = n_launch * GRAPH_DISPATCH
    t_eager = n_launch * EAGER_LAUNCH
    print(f"  kernels/step ~{n_launch:,}  ({per_block}/block x {BLOCKS} + "
          f"{N4} indexer paths)")
    print(f"  CUDA graph  @ {GRAPH_DISPATCH/us:.1f} us  ->{t_graph/ms:>8.3f} ms   "
          f"(recipe: FULL_DECODE_ONLY)")
    print(f"  eager       @ {EAGER_LAUNCH/us:.1f} us  ->{t_eager/ms:>8.3f} ms   "
          f"(what graphs are buying)")
    print()

    print("=" * 78)
    print("RANKING")
    print("=" * 78)
    ranked = [
        ("1  routed+shared expert weight streaming",
         (traffic["routed experts (selected only)"] + traffic["shared expert (all tokens)"]) / HBM_BW,
         "MEMORY"),
        ("2  launch/dispatch (under CUDA graph)", t_graph, "LATENCY"),
        ("3  all other HBM traffic (attn, KV, head)",
         (tot_bytes - traffic["routed experts (selected only)"]
          - traffic["shared expert (all tokens)"]) / HBM_BW, "MEMORY"),
        ("   collectives (bandwidth only)", t_comm, "COMM"),
        ("   compute (all precisions)", t_comp, "COMPUTE"),
    ]
    for name, t, label in sorted(ranked, key=lambda x: -x[1]):
        print(f"  {name:<44}{t/ms:>8.3f} ms   {label}")
    print()
    floor = max(t_mem, t_comp) + t_comm + t_graph
    print(f"  step floor (mem, + comm + launch serialized): {floor/ms:.2f} ms")
    print(f"  -> {BATCH/floor:,.0f} tok/s aggregate at batch {BATCH}")
    print(f"  PR #53709 measured 1,642-1,856 tok/s decode @ 32 concurrent.")
    print(f"  floor/measured = {(BATCH/floor)/1750:.2f}x  "
          f"(a floor ABOVE measurement is correct; below would be an error)")


if __name__ == "__main__":
    main()
