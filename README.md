# DeepSeek-V4-Pro — Decode-Step Execution Prediction

Work-in-progress response to the GitM Git Machine execution-graph case: predict how
**DeepSeek-V4-Pro** executes one decode step, derived from the checkpoint's own files,
before ever seeing it run. No GPU, no traces, nothing executed.

**Status:** Phases 1–2 complete and validated. Phases 3–5 outstanding.

---

## The task

| | |
|---|---|
| **Checkpoint** | `deepseek-ai/DeepSeek-V4-Pro` @ `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| **Scale** | 1.6 T total / 49 B active, FP4 + FP8 mixed, 61 layers + 1 MTP |
| **Baseline** | 32 sequences · 8,192 cached tokens each · 1 new token/seq/step · text-only · speculation disabled |
| **Hardware** | 8×H200 SXM, NVLink, TP8 (assigned) — see *Hardware divergence* below |
| **Deliverables** | D1 planner-schema YAML · D2 design note (1–2 pages) |

Full assignment: [`context/case-deepseek-v4-pro.md`](context/case-deepseek-v4-pro.md).

---

## Repo layout

```
.agent/context.md        Working context. Every finding, decision, and open question.
                         Read this first — it is the authoritative state of the work.
derive/
  weights.py             Weight-byte derivation. Validates against the index's total_size.
  fit.py                 Per-GPU memory fit at the baseline. Sharding-aware.
context/
  case-deepseek-v4-pro.md            The assignment.
  GitM-Technical-Overview.extracted.md  Company overview (PDF text, manually decoded).
  GLM-5.2-DESIGN-NOTE.reference.md   GitM's own worked example at the same baseline.
  checkpoint/                        config.json, index, inference/*.py from the HF repo.
```

## Reproduce

```bash
python3 derive/weights.py    # weight accounting vs published total_size
python3 derive/fit.py        # per-GPU fit at 8×H200 / TP8
```

Neither needs a GPU, network, or the weights themselves — only the metadata in
`context/checkpoint/`.

---

## What is established

### Weight derivation is byte-exact

```
PREDICTED  864,704,792,696 B   145,116 tensors
PUBLISHED  864,704,792,696 B   145,116 tensors
DIFFERENCE              +0 B         0
```

Not "within 2%" — zero, on both bytes and tensor count. Every projection dim, per-layer-kind
rate, and expert bank size is confirmed. The rubric names shapes as the most common failure
mode; this closes it.

### Structure

- **61 layers + 1 MTP block.** `compress_ratios` has **62** entries (`n_layers + n_mtp_layers`),
  which resolves what first looked like an off-by-one.
- **Three layer schedules that do not tile:**
  - 30 backbone layers at compress ratio 4 → overlapping `Compressor` **+ `Indexer`**
  - 31 backbone layers at ratio 128 → `Compressor` only
  - 3 layers (0–2) route by hash table (`gate.tid2eid`) instead of a learned gate;
    the other 58 carry `gate.bias`
- **MLA-style latent KV:** `num_key_value_heads: 1`, `head_dim: 512`, LoRA on *both* the query
  (`wq_a`→`wq_b`) and the output (`wo_a`→`wo_b`, `o_groups: 16`) projections.
- **mHC residuals** (Manifold-Constrained Hyper-Connections): `hc_mult: 4`, and
  `hc_split_sinkhorn` runs **20 iterations** at two sites per layer.

### Memory, per GPU at TP8

| | GB/GPU |
|---|---:|
| weights — replicated on every rank | 7.38 |
| weights — sharded (857.33 ÷ 8) | 107.17 |
| **weights total** | **114.54** |
| KV cache @ fp8, replicated | 1.42 |
| compressor `kv_state`/`score_state` (fp32) | 0.60 |
| **derived subtotal** | **116.56** |

- **Per-GPU weights are not `total/8`.** `model.py` uses plain `Linear` — replicated — for
  `wq_a`, `wkv`, both compressors, and most expensively `shared_experts` (4.10 GB/GPU alone).
  The naive division says 108.09 GB and understates by **6.46 GB**.
- **KV is replicated, not sharded.** One KV head and a plain-`Linear` `wkv` mean every rank
  holds the full 512-dim latent.
- **KV is small**: 44.34 MB/seq @ fp8, of which only 40.34 MB grows with context — the window
  term is O(1). Uncompressed MLA would be 256 MB/seq, so the compressor buys **~5.8×**.
- **FP4 scale metadata is 48.36 GB.** FP4 scales are `[out, in/32]` — full output resolution,
  unlike FP8's `[out/128, in/128]` blocks — so they cost 1/16 of expert weight bytes, not
  1/16384. Easy to miss entirely.

### Fit verdict — stated as two numbers

```
capacity LOWER BOUND     116.56 GB / 141 GB raw   FITS (82.7%)        fully derived
deployable @ util 0.9    128.56 GB / 126.90 GB    −1.66 GB            assumption-dominated
```

The second number rests on 12 GB of assumed workspace/comm/reserve against a 1.66 GB margin,
so it is **not** presented as a fact. What is firm: 116.56 GB/GPU is derived, leaving 24.44 GB
for workspace, communication buffers, reserve, and fragmentation. Whether that suffices is an
**engine** question, not a checkpoint question.

**If experts are converted to FP8** (`convert.py --expert-dtype fp8`): +92.17 GB/GPU → 220.73
GB/GPU, impossible on one node. Minimal shape is **2 nodes at TP16** (~107 GB/GPU); divisibility
holds at 16 for experts, heads, `o_groups`, and vocab. That shape moves `wo_b`'s all-reduce and
the MoE all-reduce onto the inter-node fabric.

### Collectives in the decode graph — ≈156, not 124

Read from `model.py`, not assumed from TP convention:

| Collective | Where | Count |
|---|---|---:|
| `wo_b` all-reduce — **in fp32** (`y = y.float()` first, so 4 B/element on the wire) | every block | 62 |
| `MoE` all-reduce over the fp32 accumulator | every block | 62 |
| `Indexer` all-reduce on `index_score` | **ratio-4 layers only** | 30 |
| `ParallelEmbedding` all-reduce | input | 1 |
| `ParallelHead` all-gather of logits | output | 1 |

A TP convention of 2/layer would predict 124 and miss the indexer's third collective on half
the layers.

### Checkpoint vs implementation — three documented contradictions

1. **`gate.tid2eid` is stored `int64`; `model.py` declares `int32`.** Derived, not assumed:
   at int32 the prediction falls exactly 9,308,160 B short, which is precisely
   `3 hash layers × 129,280 × 6 × 4 B`. At int64 the total is exact.
2. **`wo_a` is FP8 in the checkpoint but loaded BF16 by the reference implementation.** The
   `Linear` is built `dtype=torch.bfloat16`, yet `attn.wo_a.scale` exists 61× in the index, and
   `model.py:539` says so outright. Storage precision ≠ runtime precision, vendor-stated.
3. **No `trust_remote_code` path exists.** No `auto_map`, no `modeling_*.py`. `inference/` is a
   standalone reference implementation requiring `convert.py` + `torchrun`, not an HF
   integration. vLLM runs its own in-tree implementation, so bound analysis must cite that —
   never `inference/model.py`.

### Hardware divergence

`recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro` publishes a vendor recipe, and the assignment says
a published recipe supersedes the 8×H200 default. But the recipe targets **8× AMD MI355X
(288 GB each)**, where the fit question is nearly trivial.

Decision: **price the assigned 8×H200 and treat the divergence as a finding.** The brief states
the fit section carries the weight for V4-Pro, and that is only true at 141 GB. The recipe's
hardware-independent settings (`--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.9`) are
adopted and cited.

---

## Where this is going

### Phase 3 — Engine assumptions *(next)*

Pin an exact vLLM tag (recipe requires 0.20.0+) and read source for what the checkpoint cannot
determine: which ops fuse, what runs under CUDA graphs, the attention backend, the NCCL
collective implementation, and where runtime precision departs from storage precision. Each item
tagged *read from source* / *read from docs* / *assumed*.

Two answers already come free from the recipe rather than assumption:
`cudagraph_mode: FULL_DECODE_ONLY` and `--kv-cache-dtype fp8`.

**The open question that could reorder everything:** H200 is SM90 and has no native FP4 tensor
path — that arrives with Blackwell. If vLLM dequantizes the FP4 expert bank at load time, the
memory figures above are wrong. If it dequantizes per-op at runtime, the bound label changes
instead. This is the sharpest unresolved item in the case.

### Phase 4 — Bound hypotheses

Roofline ridges per precision present on H200 (FP8 1,979 TFLOP/s dense, BF16 989.5, FP32 67,
HBM3e 4.8 TB/s, NVLink 900 GB/s), then the three largest time consumers ranked and labeled
compute- / memory- / communication-bound. Each gets one controlled experiment with an explicit
rejection threshold, plus a statement of what that experiment *cannot* establish.

Communication must be *defended into* its rank, not assumed to lead or trail — the fp32
all-reduce and the indexer's extra 30 collectives both push against the usual intuition.

### Phase 5 — Deliverables

- **D1** — model spec YAML in the planner schema, following
  `gitm/planner/models/mimo-v2.5.yaml`: layer schedules written out where they do not tile,
  per-op precision noted, checkpoint names kept verbatim, family contradictions captured in
  comments.
- **D2** — design note, 1–2 pages, matching the rigor of `docs/glm-5.2/DESIGN-NOTE.md` rather
  than its length.

### Open decisions

| # | Question | Status |
|---|---|---|
| 1 | H200 vs MI355X | **resolved** — H200, divergence documented |
| 2 | KV dtype | engine fact, cited to recipe (`fp8`), not a checkpoint fact |
| 3 | TP8-only or expert parallelism | `--enable-expert-parallel` absent from recipe; changes every collective row |
| 4 | FP4 expert handling on SM90 | **open — Phase 3 blocker** |
| 5 | Exact vLLM tag to pin | open |
| 6 | MTP resident but inactive | resident 1.92 GB/GPU, absent from decode graph; state both halves |

---

## Method note

Two evidence classes are kept separate throughout, because the case is explicit that conflating
them is the failure: **what the checkpoint establishes** (shapes, counts, storage precision,
layer schedules) versus **what depends on the engine** (fusion, launch behavior, collective
implementation, runtime precision). Every number carries its provenance, and every assumption is
labeled rather than smuggled.
