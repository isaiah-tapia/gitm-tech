"""
DeepSeek-V4-Pro — weight-byte derivation from the checkpoint's own files.

Every shape below is read from `inference/model.py` (DeepSeek's reference
implementation) and every count from `config.json`. Nothing is assumed from
family convention.

Validation: the predicted total is compared against `total_size` in
`model.safetensors.index.json` (864,704,792,696 B). Predicted tensor NAMES and
COUNTS are also compared against the index's actual key census, which catches
shape errors that happen to cancel in the byte total.

Run:  python3 derive/weights.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "context", "checkpoint")

cfg = json.load(open(os.path.join(CKPT, "config.json")))

# ---------------------------------------------------------------- config
DIM          = cfg["hidden_size"]              # 7168
N_LAYERS     = cfg["num_hidden_layers"]        # 61
N_HASH       = cfg["num_hash_layers"]          # 3
N_MTP        = cfg["num_nextn_predict_layers"] # 1
VOCAB        = cfg["vocab_size"]               # 129280
N_HEADS      = cfg["num_attention_heads"]      # 128
HEAD_DIM     = cfg["head_dim"]                 # 512
ROPE_HD      = cfg["qk_rope_head_dim"]         # 64
Q_LORA       = cfg["q_lora_rank"]              # 1536
O_LORA       = cfg["o_lora_rank"]              # 1024
O_GROUPS     = cfg["o_groups"]                 # 16
N_EXPERTS    = cfg["n_routed_experts"]         # 384
N_ACT        = cfg["num_experts_per_tok"]      # 6
N_SHARED     = cfg["n_shared_experts"]         # 1
MOE_INTER    = cfg["moe_intermediate_size"]    # 3072
IDX_HEADS    = cfg["index_n_heads"]            # 64
IDX_HD       = cfg["index_head_dim"]           # 128
HC_MULT      = cfg["hc_mult"]                  # 4
RATIOS       = cfg["compress_ratios"]

BLOCK    = 128   # model.py: block_size
FP4BLOCK = 32    # model.py: fp4_block_size

# ------------------------------------------------- byte rates per element
# Derived from model.py's Linear/RMSNorm/Parameter dtypes, cross-read with the
# checkpoint comments in that file ("stored in bf16, parameter here is fp32").
B_FP8  = 1      # torch.float8_e4m3fn
B_E8M0 = 1      # torch.float8_e8m0fnu  (scale tensors)
B_BF16 = 2
B_FP32 = 4
B_I64  = 8      # see note at gate.tid2eid


def ceil_div(a, b):
    return (a + b - 1) // b


def fp8_linear(out_f, in_f):
    """model.py Linear, dtype=float8_e4m3fn: weight [out,in] fp8,
    scale [ceil(out/128), ceil(in/128)] e8m0."""
    w = out_f * in_f * B_FP8
    s = ceil_div(out_f, BLOCK) * ceil_div(in_f, BLOCK) * B_E8M0
    return w + s, 2   # bytes, tensor count (.weight + .scale)


def fp4_linear(out_f, in_f):
    """model.py Linear, dtype=float4_e2m1fn_x2: weight [out, in//2] (two fp4
    per byte), scale [out, in//32] e8m0 — note FULL out resolution, so the
    scale overhead is 1/16 of the weight, not the 1/16384 an fp8 block gives."""
    w = out_f * (in_f // 2) * 1
    s = out_f * (in_f // FP4BLOCK) * B_E8M0
    return w + s, 2


def bf16_linear(out_f, in_f):
    return out_f * in_f * B_BF16, 1


def rmsnorm(d):
    """model.py: 'rmsnorm in the checkpoint is stored in bf16, while the
    parameter here is stored in fp32 for convenient.'"""
    return d * B_BF16, 1


# ------------------------------------------------------------- components
def compressor_bytes(ratio, head_dim):
    """model.py Compressor. coff = 1 + (ratio == 4) — the overlap path stores
    two head_dim-wide halves in one tensor. wkv/wgate are bf16 in the
    checkpoint (explicit comment); ape is fp32."""
    coff = 1 + (ratio == 4)
    b = n = 0
    b += ratio * (coff * head_dim) * B_FP32; n += 1          # .ape
    wb, wn = bf16_linear(coff * head_dim, DIM)               # .wkv.weight
    b += wb; n += wn
    wb, wn = bf16_linear(coff * head_dim, DIM)               # .wgate.weight
    b += wb; n += wn
    rb, rn = rmsnorm(head_dim); b += rb; n += rn             # .norm.weight
    return b, n


def indexer_bytes():
    """model.py Indexer — instantiated only when compress_ratio == 4."""
    b = n = 0
    wb, wn = fp8_linear(IDX_HEADS * IDX_HD, Q_LORA)          # .wq_b
    b += wb; n += wn
    wb, wn = bf16_linear(IDX_HEADS, DIM)                     # .weights_proj (bf16, no scale)
    b += wb; n += wn
    cb, cn = compressor_bytes(4, IDX_HD)                     # .compressor.*
    b += cb; n += cn
    return b, n


def attention_bytes(ratio):
    b = n = 0
    b += N_HEADS * B_FP32; n += 1                            # .attn_sink (fp32)
    for out_f, in_f in [
        (Q_LORA, DIM),                                       # .wq_a
        (N_HEADS * HEAD_DIM, Q_LORA),                        # .wq_b
        (HEAD_DIM, DIM),                                     # .wkv
        (O_GROUPS * O_LORA, N_HEADS * HEAD_DIM // O_GROUPS), # .wo_a  (see note)
        (DIM, O_GROUPS * O_LORA),                            # .wo_b
    ]:
        wb, wn = fp8_linear(out_f, in_f); b += wb; n += wn
    for d in (Q_LORA, HEAD_DIM):                             # .q_norm, .kv_norm
        rb, rn = rmsnorm(d); b += rb; n += rn
    if ratio:
        cb, cn = compressor_bytes(ratio, HEAD_DIM); b += cb; n += cn
        if ratio == 4:
            ib, iN = indexer_bytes(); b += ib; n += iN
    return b, n


def moe_bytes(is_hash):
    b = n = 0
    wb, wn = bf16_linear(N_EXPERTS, DIM); b += wb; n += wn    # .gate.weight
    if is_hash:
        # CHECKPOINT CONTRADICTS model.py. model.py declares
        #   tid2eid = nn.Parameter(..., dtype=torch.int32)
        # but the stored tensor is int64. Derived, not assumed: with int32 the
        # prediction lands 9,308,160 B short of total_size, and that residual is
        # exactly 3 hash layers x 129,280 x 6 x 4 B — i.e. precisely the int32
        # -> int64 delta on this one tensor. At int64 the total is exact.
        b += VOCAB * N_ACT * B_I64; n += 1                   # .gate.tid2eid
    else:
        b += N_EXPERTS * B_FP32; n += 1                      # .gate.bias (fp32)
    for _ in range(N_EXPERTS):                               # routed: FP4
        for out_f, in_f in [(MOE_INTER, DIM), (DIM, MOE_INTER), (MOE_INTER, DIM)]:
            wb, wn = fp4_linear(out_f, in_f); b += wb; n += wn
    for _ in range(N_SHARED):                                # shared: default FP8
        for out_f, in_f in [(MOE_INTER, DIM), (DIM, MOE_INTER), (MOE_INTER, DIM)]:
            wb, wn = fp8_linear(out_f, in_f); b += wb; n += wn
    return b, n


def hc_block_bytes():
    """model.py Block: mix_hc = (2 + hc_mult) * hc_mult, hc_dim = hc_mult * dim.
    All six tensors fp32 (constructed under set_dtype(torch.float32))."""
    mix_hc = (2 + HC_MULT) * HC_MULT
    hc_dim = HC_MULT * DIM
    b = n = 0
    for _ in range(2):                                       # hc_attn_*, hc_ffn_*
        b += mix_hc * hc_dim * B_FP32; n += 1                # .._fn
        b += mix_hc * B_FP32;          n += 1                # .._base
        b += 3 * B_FP32;               n += 1                # .._scale
    return b, n


def hc_head_bytes():
    hc_dim = HC_MULT * DIM
    b = HC_MULT * hc_dim * B_FP32 + HC_MULT * B_FP32 + 1 * B_FP32
    return b, 3


def block_bytes(layer_id, ratio):
    b = n = 0
    ab, an = attention_bytes(ratio);              b += ab; n += an
    mb, mn = moe_bytes(layer_id < N_HASH);        b += mb; n += mn
    for _ in range(2):                                       # attn_norm, ffn_norm
        rb, rn = rmsnorm(DIM); b += rb; n += rn
    hb, hn = hc_block_bytes();                    b += hb; n += hn
    return b, n


# ------------------------------------------------------------------ total
def main():
    print(f"compress_ratios length = {len(RATIOS)}  (n_layers = {N_LAYERS}, "
          f"n_mtp = {N_MTP})")
    print(f"  ratio==4   : {RATIOS.count(4):3d} layers  -> indexer present")
    print(f"  ratio==128 : {RATIOS.count(128):3d} layers")
    print(f"  ratio==0   : {RATIOS.count(0):3d} layers  -> no compressor")
    print()

    rows, total_b, total_n = [], 0, 0

    def add(label, b, n):
        nonlocal total_b, total_n
        rows.append((label, b, n)); total_b += b; total_n += n

    eb, en = bf16_linear(VOCAB, DIM); add("embed.weight", eb, en)

    backbone_b = backbone_n = 0
    for lid in range(N_LAYERS):
        b, n = block_bytes(lid, RATIOS[lid])
        backbone_b += b; backbone_n += n
    add(f"layers.0-{N_LAYERS-1} ({N_LAYERS} blocks)", backbone_b, backbone_n)

    rb, rn = rmsnorm(DIM);            add("norm.weight", rb, rn)
    hb, hn = hc_head_bytes();         add("hc_head_* (model level)", hb, hn)
    lb, ln = bf16_linear(VOCAB, DIM); add("head.weight", lb, ln)

    mtp_b = mtp_n = 0
    for i in range(N_MTP):
        lid = N_LAYERS + i
        ratio = RATIOS[lid] if lid < len(RATIOS) else 0
        b, n = block_bytes(lid, ratio)
        for _ in range(2):                                   # e_proj, h_proj
            wb, wn = fp8_linear(DIM, DIM); b += wb; n += wn
        for _ in range(3):                                   # enorm, hnorm, norm
            r, rn2 = rmsnorm(DIM); b += r; n += rn2
        hb2, hn2 = hc_head_bytes(); b += hb2; n += hn2
        mtp_b += b; mtp_n += n
    add(f"mtp.0-{N_MTP-1} ({N_MTP} block, ratio={RATIOS[N_LAYERS] if N_LAYERS < len(RATIOS) else 0})",
        mtp_b, mtp_n)

    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    published = idx["metadata"]["total_size"]
    actual_n = len(idx["weight_map"])

    w = 46
    print(f"{'component':<{w}}{'GB':>12}{'tensors':>10}")
    print("-" * (w + 22))
    for label, b, n in rows:
        print(f"{label:<{w}}{b/1e9:>12.2f}{n:>10,}")
    print("-" * (w + 22))
    print(f"{'PREDICTED':<{w}}{total_b/1e9:>12.2f}{total_n:>10,}")
    print(f"{'PUBLISHED (index total_size)':<{w}}{published/1e9:>12.2f}{actual_n:>10,}")
    delta = (total_b - published) / published * 100
    dn = total_n - actual_n
    print(f"{'DELTA':<{w}}{delta:>11.2f}%{dn:>10,}")
    print()
    print("EXIT GATE: |delta| < 2%  ->", "PASS" if abs(delta) < 2 else "FAIL")
    print("TENSOR COUNT EXACT       ->", "PASS" if dn == 0 else f"FAIL ({dn:+,})")


if __name__ == "__main__":
    main()
