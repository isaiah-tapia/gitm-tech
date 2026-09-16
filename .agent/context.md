# .agent/context.md — working context

Living file. Claude reads this at the start of a session and updates it as work proceeds.
Everything below is either **sourced** (traceable to a file in `context/` or a public primary
source) or **assumed** (marked as such). Never promote an assumption to a fact without a citation.

---

## 1. What this repo is

| Path | What it is |
|---|---|
| `context/case-deepseek-v4-pro.md` | The assignment. Authoritative for scope, deliverables, scoring. |
| `context/GitM-Technical-Overview (7).pdf` | Company technical overview. Background on the evaluator. |
| `context/GitM-Technical-Overview.extracted.md` | Plain-text extraction of the PDF (ASCII85+Flate decoded manually; `pdftotext`/poppler is not installed on this machine). Read this instead of the PDF. |
| `.agent/context.md` | This file. |

---

## 2. The assignment (from `context/case-deepseek-v4-pro.md`)

**Task:** Predict how **DeepSeek V4 Pro** executes *one decode step*, derived from the
checkpoint's own files, before ever seeing it run. No GPU, no traces, nothing to run.

### Baseline workload (all fit/bound claims evaluated here)
- 32 active sequences, decode phase
- 8,192 cached tokens per sequence
- 1 new token per sequence per step
- text-only, speculation disabled
- hardware/precision per assignment parameters below

### Assignment parameters
- Hardware: **one 8×H200 SXM node over NVLink, TP8**
- Precision: **FP8 weights and KV where the checkpoint supports it**
- A published **vendor serving recipe supersedes** this default — cite it if one exists.
- Must state **exact repo + revision** of the checkpoint and the **pinned engine version**.
- Alternatives to any default are allowed **with stated justification**.

### Allowed sources
- ✅ Model repo on HF: `config.json`, `model.safetensors.index.json`, and the modeling
  implementation the checkpoint actually loads (the `trust_remote_code` path where present).
- ✅ Vendor serving recipes (for deployment shape).
- ✅ Serving-engine **source** and official docs — **with the version pinned** (vLLM tag/commit),
  because the checkpoint alone does not determine kernel fusion, launch behavior, collective
  implementation, or runtime precision.
- ❌ **Third-party writeups of the architecture.** The point is what you derive, not what you find.

**Hard rule that runs through the whole submission:** keep the two evidence classes separate —
*what the checkpoint establishes* vs. *what depends on the engine*.

---

## 3. Deliverables (two artifacts)

### D1 — Model spec YAML, planner schema
Follow the format and conventions of
`gitm/planner/models/mimo-v2.5.yaml` in `github.com/GitM-Labs/runtime`:
- layer schedules written out when they do not tile
- per-op precision noted
- **checkpoint's own names kept verbatim** — quantization labels, tensor and module names exactly
  as they appear in the files, not renamed into nicer vocabulary
- where config contradicts what the model family usually does → capture in a comment (that's signal)

### D2 — Design note, markdown, 1–2 pages
Reference example (fuller scope than asked): `docs/glm-5.2/DESIGN-NOTE.md`, same repo.
**Match its rigor, not its length.** Required sections:

1. **Engine assumptions** — what the pinned engine determines beyond the checkpoint: op fusion,
   what runs under CUDA graphs, attention backend, collective implementation, where runtime
   precision differs from storage precision. Mark each item: *read from engine source* /
   *read from docs* / *assumed*.
2. **Memory fit** — per-GPU accounting at the baseline: weights, quantization metadata, KV or
   recurrent state, activation workspace, communication buffers, **explicit reserve**.
   Distinguish the **capacity lower bound** from a **configuration you would actually deploy**.
   If it does not fit 8×H200, derive the **minimal shape that holds it** and name the
   communication ops that shape introduces into the decode graph.
3. **Top-3 bound hypotheses, ranked** — the three largest time consumers in the predicted decode
   step; each labeled compute- / memory- / communication-bound at the baseline. If the layout
   includes TP/EP/PP, derive the comm cost **from that layout** and **defend where it sits in the
   ranking** rather than assuming it leads or trails. Each hypothesis needs **one controlled
   experiment**: what is held fixed, what is varied, the observable recorded, and the
   **threshold or trend that would reject it**. Plus: **say what the experiment cannot establish.**

### V4-Pro-specific emphasis
It is the large sibling — **the fit section carries the weight**. Show whether weights hold on one
8×H200 node at the checkpoint's precision; if not, derive the minimal multi-node shape.
Choose TP/EP/PP **explicitly**, derive the comm ops it introduces into the decode graph, and
**price NVLink and inter-node fabric separately**.

---

## 4. Scoring rubric (verbatim intent)

- **Correctness beats coverage.** An incomplete section with documented next steps ("here's what
  I'd do with two more hours and why") scores better than a complete section on shaky arithmetic.
- **Wrong-but-specific beats vague.** A precise false claim is testable; "it depends" is not.
  Explicitly identified uncertainty scores just as well:
  *"The config establishes X; engine source is needed to determine Y."*
- **What fails:** arithmetic that doesn't follow from the files; family assumptions presented as
  facts; hypotheses without a rejection condition.
- **Shapes are where most submissions fail.** Projection dims, per-layer-kind KV byte rates, and
  expert bank sizes follow from the config by arithmetic. **Check them twice.**

---

## 5. Who's evaluating (from the technical overview)

GitM is a **GPU execution runtime** that sits between the serving/compute stack
(vLLM / SGLang / PyTorch) and the GPU runtime (CUDA/ROCm). It runs in user space, no driver
replacement or kernel module, and works from **execution telemetry** — it never ingests weights,
source, architecture files, prompts, or outputs.

Its four-stage loop:
1. **Model expected performance** — reconstruct execution structure from telemetry + measured
   GPU/memory/interconnect/topology characteristics → a workload-specific execution ceiling.
   Roofline plus dependencies, launch gaps, synchronization, communication, missed overlap.
2. **Attribute the gap** — compare observed vs. ceiling across compute utilization, concurrency,
   sync, memory movement, communication; attribute loss to runtime causes with evidence.
3. **Apply a targeted runtime change** — graph capture of repeated launch sequences, stream
   assignment/priorities, overlap, prefetch, allocation behavior, collective tuning, choosing
   among equivalent implementations. **GitM does not write or rewrite kernels.**
4. **Verify** — measure against original, retain only on confirmed improvement within latency /
   memory / stability / correctness gates; canary + automatic rollback; output equivalence
   verifiable on demand.

Published results: **+32.4% throughput byte-identical on HFT**, **+63% on KITTI**.

**Why this matters for the submission:** the case is a proxy for stage 1 + stage 2 of that loop
done *analytically*. The reasoning style they reward is the same style the product embodies —
a derived ceiling, an attributed bound, evidence separated from assumption, and a falsifiable
next step. Writing the design note in that vocabulary (ceiling / attribution / rejection
condition) is a free alignment win.

---

## 6. Source material — CONFIRMED (2026-09-15)

Everything resolves. Local copies in `context/checkpoint/` and `context/`.

### Checkpoint
**`deepseek-ai/DeepSeek-V4-Pro`**, revision **`b5968e9190ef611bbf34a7229255be88a0e937c1`**
(lastModified 2026-06-22). Sibling revisions that exist and were *not* chosen:
`-Base` (pretrained), `-0813` (adds `dspark_*` fields — speculative variant, excluded because the
baseline says speculation disabled), `-DSpark`.

- **1.6T total / 49B active**, "FP4 + FP8 Mixed" per the model card.
- `model.safetensors.index.json` → **`total_size` = 864,704,792,696 B = 864.70 GB**, 145,116 tensors.
- Ships its own `inference/` implementation (`model.py`, `kernel.py`, `config.json`).
- Tech report: arXiv 2606.19348, *"DeepSeek-V4: Towards Highly Efficient Million-Token Context
  Intelligence."* (Citable for existence; **architecture claims must still be derived from files.**)

### There is NO `trust_remote_code` path on this checkpoint — verified
The assignment says to use "the modeling implementation the checkpoint actually loads (the
`trust_remote_code` path **where present**)." Here it is **not present**:

- `config.json` has **no `auto_map`** field (grep count 0).
- The repo has **no `modeling_*.py` and no `configuration_*.py`**. The only Python is
  `encoding/` (chat templating) and `inference/`.
- ⇒ HF Transformers cannot load custom model code from this repo. `trust_remote_code` has nothing
  to bind to for the model.

`inference/` is DeepSeek's **standalone reference implementation**, not an HF integration. Per
`inference/README.md` it requires `convert.py` to reshard the HF weights first, runs under
`torchrun --nproc-per-node 8`, and reads its own `inference/config.json` with different field
names (`dim`, `n_layers`, `n_heads` vs `hidden_size`, `num_hidden_layers`, `num_attention_heads`).

**Say this explicitly in the design note.** Treating `inference/model.py` as "the trust_remote_code
path" is a category error, and naming the absence is the evidence-class discipline being graded.

### Three code paths — keep them separate everywhere
| Path | Authority for | Caveat |
|---|---|---|
| `config.json` + `model.safetensors.index.json` | shapes, counts, precisions, layer schedules | ground truth; the only true checkpoint class |
| `inference/model.py` + `kernel.py` | reference **semantics** — what each op is, module wiring | DeepSeek's own, but **not what vLLM runs** |
| vLLM in-tree DeepSeek-V4 implementation | what actually **executes** — fusion, kernels, runtime precision, collectives | the engine class; must be version-pinned |

`--trust-remote-code` *is* in the vendor recipe, but with no `auto_map` there is no model code for
it to load; `--tokenizer-mode deepseek_v4` implies vLLM has **native in-tree** support. So bound
analysis must cite vLLM's implementation, never `inference/model.py`.

### Vendor serving recipe — EXISTS, and it changes the assignment
`recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro` (HTTP 200) publishes:

```bash
export VLLM_ROCM_USE_AITER=1
vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --host localhost --port 8001 --dtype auto \
  --kv-cache-dtype fp8 --tensor-parallel-size 8 \
  --max-num-seqs 512 --max-num-batched-tokens 8192 \
  --distributed-executor-backend mp --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}'
```
Hardware: **8× AMD MI355X, 288 GB each, single node.** vLLM **0.20.0+** minimum.
Note `cudagraph_mode: FULL_DECODE_ONLY` — directly answers the engine-assumptions section.

### GitM reference files — both reachable
- `gitm/planner/models/mimo-v2.5.yaml` — full schema. Has `spec:` / `provenance:` with
  `verified` / `estimated` / `open` / `unmodelled` subsections. Comments carry the reasoning.
- `docs/glm-5.2/DESIGN-NOTE.md` — 1,141 lines. Saved to
  `context/GLM-5.2-DESIGN-NOTE.reference.md`. Same baseline (8×H200, batch 32, kv-len 8192, TP8/EP8),
  quotes the vendor recipe verbatim, prices FP8/BF16/FP32 roofline ridges separately, marks every
  label that would flip with ⚑.

## 7. What the config actually says (derived, 2026-09-15)

From `context/checkpoint/config.json` + `inf_config.json` + tensor-name census of the index.
**Not yet double-checked — the rubric says check shapes twice, and pass 2 has not happened.**

| Field | Value | Why it matters |
|---|---|---|
| `num_hidden_layers` | 61 | |
| `hidden_size` | 7168 | |
| `vocab_size` | 129280 | untied (`tie_word_embeddings: false`) |
| `num_attention_heads` | 128 | |
| `head_dim` | **512** | unusually large |
| `num_key_value_heads` | **1** | MLA-style latent cache, not GQA |
| `qk_rope_head_dim` | 64 | |
| `q_lora_rank` / `o_lora_rank` | 1536 / 1024 | `wq_a`/`wq_b`, `wo_a`/`wo_b` — **output** proj is LoRA'd too |
| `o_groups` | 16 | new vs V3 |
| `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | 384 / 6 / 1 | 23,424 expert tensors ÷ 61 = 384 ✓ |
| `moe_intermediate_size` | 3072 | |
| `sliding_window` | 128 | |
| `max_position_embeddings` | 1,048,576 | yarn, factor 16 from 65,536 |
| `rope_theta` / `compress_rope_theta` | 10,000 / **160,000** | two different thetas in one model |
| `scoring_func` | **`sqrtsoftplus`** | V3 used sigmoid — flag as family contradiction |
| `topk_method` | `noaux_tc` | same as V3 |
| `routed_scaling_factor` | 2.5 | |
| `swiglu_limit` | 10.0 | clamped SwiGLU |
| `num_nextn_predict_layers` | 1 | MTP; tensors present under `mtp.*` |
| `quantization_config` | fp8 `e4m3`, scale_fmt **`ue8m0`**, `weight_block_size [128,128]`, dynamic acts | |
| `expert_dtype` | **`fp4`** | **experts are FP4, backbone is FP8 — two storage precisions** |
| `torch_dtype` | bfloat16 | storage ≠ this for most tensors |

### Three layer schedules that do not tile — write them out
1. **`compress_ratios`** — 61 entries: `[128, 128, 4, 128, 4, 128, 4, ... , 4, 0]`. Alternates
   128/4 from index 2, **last layer is 0**. Pairs with the per-layer `attn.compressor.*` tensors.
2. **Indexer present on only 30 of 61 layers** — `attn.indexer.*` census = 30, not 61.
   (`index_n_heads` 64, `index_head_dim` 128, `index_topk` 1024 — the DSA-style sparse indexer.)
3. **Hash routing on exactly 3 layers** — `num_hash_layers: 3`; `ffn.gate.tid2eid` appears 3×,
   and `ffn.gate.bias` appears **58×** (= 61 − 3). Those 3 layers route by token-id→expert-id
   table, not a learned gate. 58 + 3 = 61 ✓.

### mHC — Manifold-Constrained Hyper-Connections
README names it: *"mHC to strengthen conventional residual connections."* Config: `hc_mult: 4`,
`hc_sinkhorn_iters: 20`, `hc_eps: 1e-6`. Tensors: `hc_attn_base`, `hc_attn_fn`, `hc_attn_scale`,
`hc_ffn_base`, `hc_ffn_fn`, `hc_ffn_scale` (61× each) plus `hc_head_*` at model level.
**20 Sinkhorn iterations per residual site is a real op in the decode graph** — two sites per
layer × 61 layers. Likely tiny FLOPs but potentially many small launches. Worth pricing, not
assuming away.

### Attention module tensor names (verbatim — use these, do not normalize)
`attn.wq_a` `attn.wq_b` `attn.q_norm` `attn.wkv` `attn.kv_norm` `attn.wo_a` `attn.wo_b`
`attn.attn_sink` `attn.compressor.{ape,wkv,wgate,norm}` `attn.indexer.{wq_b,weights_proj,compressor.*}`
FFN: `ffn.gate.{weight,bias,tid2eid}` `ffn.experts.N.{w1,w2,w3}.{weight,scale}` `ffn.shared_experts.*`
Model level: `embed.weight` `norm.weight` `head.weight`. MTP: `mtp.N.{e_proj,h_proj,enorm,hnorm,norm}`.
Note the flat `layers.N.` prefix — **no `model.` prefix**, unlike HF-style checkpoints.

### FP4 experts — CONFIRMED by arithmetic against `total_size`
`expert_dtype: "fp4"` is not a family assumption; it is forced by the published size.

```
expert params = 61 layers × 384 experts × 3 matrices × (7168 × 3072) = 1.5474 T
   @ fp4 (0.5 B/param) →   773.7 GB  → leaves 91.0 GB for the rest   ✓
   @ fp8 (1.0 B/param) → 1,547.4 GB  → exceeds total_size alone      ✗ impossible
```
Published `total_size` = 864.70 GB. Only FP4 is consistent.
**Experts are ~95% of all parameters** (1.5474 T of 1.6 T), so this single field dominates the
entire fit section. The remaining **91.0 GB** must cover attention, shared experts, embed, head,
FP8 scales, mHC tensors, and the MTP layer — that is the Phase-1 target to close.

### FP4 is a CONVERSION-TIME choice, not an architectural fact
`inference/README.md`, final line, verbatim:
> *"If you want to use fp8, just remove `"expert_dtype": "fp4"` in `config.json` and specify
> `--expert-dtype fp8` in `convert.py`."*

⇒ The question for H200 is not "can it run this checkpoint" but **"which conversion do you run,
and what does each cost."** FP8 experts would be 1,547.4 GB — which does **not** fit 8×H200
(1,128 GB) at all, and barely fits 8×MI355X (2,304 GB). This is the hinge of the fit section.
Also note `EXPERTS=384`, `MP=8` in the same README — DeepSeek's own reference shape is 8-way.

### The fit headline (first-pass, unverified)
864.70 GB of weights ÷ 8 = **108.09 GB/GPU**. H200 is 141 GB → 76.7% of capacity before KV,
activations, comm buffers, or reserve. At the recipe's `gpu-memory-utilization 0.9`
(126.9 GB usable) that leaves **~18.8 GB/GPU for everything else**. Whether the baseline's
KV (32 seqs × 8,192 tokens) fits in that is the whole fit section — and it depends on the
per-layer-kind KV rate, which the three schedules above make non-uniform.

## 8. Open questions / decisions needed

1. **Hardware: MI355X or H200?** The assignment says a published vendor recipe *supersedes* the
   8×H200 default, and one exists — but it is **AMD MI355X 8×288 GB (2,304 GB/node)**, where the
   fit question is nearly trivial. The H200 default (1,128 GB/node) is where the fit section
   has teeth, and V4-Pro's brief explicitly says *"the fit section carries the weight."*
   → **Decision needed.** See §9 D1.
2. **KV cache dtype.** Recipe says `--kv-cache-dtype fp8`. Config's `quantization_config` does not
   cover the cache. So FP8 KV is an *engine* fact, cited to the recipe, not a checkpoint fact.
3. **TP8-only or EP?** `--enable-expert-parallel` is **not** in the recipe. With 384 experts and
   TP8-only, each rank holds the full 384-expert bank sharded by width. Changes every collective row.
4. **How is FP4 expert storage handled on H200?** H200 has no native FP4 tensor path (that's
   Blackwell). Now sharpened by the `convert.py` finding above: FP4 vs FP8 experts is a conversion
   flag, and the two land at 773.7 GB vs 1,547.4 GB. On 8×H200 (1,128 GB) the FP8 conversion
   **does not fit at all**; the FP4 one fits the weights but may have no native kernel path.
   Resolve from vLLM source: does it dequantize FP4→BF16/FP8 at load, or per-op at runtime?
   The answer changes both the memory number and the bound label. **Sharpest finding in the case.**
5. **Engine version to pin.** Recipe says 0.20.0+. Need an exact tag/commit for source citations.
6. **MTP.** `num_nextn_predict_layers: 1` and `mtp.*` tensors exist, but baseline says speculation
   disabled → the MTP layer is resident in memory (counts toward fit) but absent from the decode
   graph (excluded from bound ranking). State both halves explicitly.

## 9. Decisions log

- **2026-09-15 — Work from `DeepSeek-V4-Pro` @ `b5968e9`, not `-0813` or `-DSpark`.**
  Why: `-0813` adds `dspark_block_size`, `dspark_target_layer_ids: [58,59,60]`, `dspark_markov_rank`
  — a speculative-decoding variant. The baseline workload specifies speculation disabled, so the
  clean checkpoint is the honest match. Diff recorded; both configs are otherwise identical.
- **2026-09-16 — Cite `inference/model.py` as reference semantics, never as "the trust_remote_code
  path."** Why: no `auto_map`, no `modeling_*.py` — verified. The absence is itself a finding worth
  one sentence in the note, and vLLM's in-tree implementation is what the bound analysis must cite.
- **2026-09-16 — Treat FP4 experts as established, not assumed.** Why: the FP8 reading exceeds the
  published `total_size` on the expert bank alone. Arithmetic recorded in §7; cite it rather than
  citing `expert_dtype`.
- **2026-09-15 — D1 pending:** hardware choice (§8 Q1). Not yet made.

## 10. Phase 1 — COMPLETE. Weight derivation is byte-exact.

`derive/weights.py` reconstructs every weight tensor from `inference/model.py` shapes +
`config.json` counts and validates two ways:

```
PREDICTED  864,704,792,696 B   145,116 tensors
PUBLISHED  864,704,792,696 B   145,116 tensors
DIFFERENCE              +0 B         0
```

**Exact on both bytes and tensor count.** Not "within 2%" — zero. Every projection dim, every
per-layer-kind rate, and every expert bank size is confirmed correct. Reproduce with
`python3 derive/weights.py`.

### Component breakdown
| Component | GB | tensors |
|---|---:|---:|
| `embed.weight` | 1.85 | 1 |
| `layers.0–60` (61 blocks) | 847.04 | 142,767 |
| `norm.weight` + model-level `hc_head_*` | ~0.00 | 4 |
| `head.weight` | 1.85 | 1 |
| `mtp.0` (1 block, ratio 0) | 13.96 | 2,343 |
| **total** | **864.70** | **145,116** |

### `compress_ratios` has 62 entries, not 61 — resolves the earlier contradiction
Length = `n_layers` (61) + `n_mtp_layers` (1). Index 61 is the **MTP block**, and its value is `0`.
So the census reconciles exactly:
- **30** backbone layers at ratio 4 → `Compressor(overlap=True)` **+ `Indexer`** → matches the 30 `attn.indexer.*` tensors ✓
- **31** backbone layers at ratio 128 → `Compressor(overlap=False)`, no indexer
- 30 + 31 = **61** backbone layers, all with a compressor → matches the 61 `attn.compressor.*` ✓
- the single `0` is the MTP block → no compressor there
Earlier note that "layer 60 has ratio 0" was wrong; layer 60 is a ratio-4 layer and index 61 is MTP.

### Two checkpoint-vs-implementation contradictions found — both are design-note material
1. **`gate.tid2eid` is stored `int64`; `model.py` declares `int32`.**
   Derived, not assumed: with `int32` the prediction lands **9,308,160 B** short, and that residual
   is exactly `3 hash layers × 129,280 × 6 × 4 B` — precisely the int32→int64 delta on that one
   tensor. At int64 the total is exact. 6.2 MB/layer × 3 layers = 18.6 MB resident.
2. **`wo_a` is FP8 in the checkpoint but loaded BF16 by the reference implementation.**
   `model.py:462` constructs it `dtype=torch.bfloat16`, yet `attn.wo_a.scale` exists in the index
   (61 of them) — only FP8/FP4 `Linear` registers a scale. The code's own comment at `model.py:539`
   confirms: *"NOTE: wo_a is FP8 in checkpoint; could do FP8 einsum here for better perf, but using
   BF16 for simplicity."* ⇒ **storage precision ≠ runtime precision**, stated by the vendor. Exactly
   the "where runtime precision differs from storage precision" item the design note must cover,
   and it is sourced rather than assumed.

### Shape facts now established (all confirmed by the exact match)
- `wq_a` [1536, 7168] fp8 · `wq_b` [65536, 1536] fp8 · `wkv` [512, 7168] fp8
- `wo_a` [16384, 4096] fp8 — note in-features is `n_heads*head_dim/o_groups` = 4096, **not** 65536
- `wo_b` [7168, 16384] fp8
- FP8 scale: `[ceil(out/128), ceil(in/128)]` e8m0 — negligible
- **FP4 scale: `[out, in/32]` e8m0 — full out resolution, so scales are 1/16 of expert weight bytes,
  not 1/16384.** Expert bank = 773.71 GB weights + **48.36 GB scales**. That 48 GB is the
  "quantization metadata" line in the fit table and is easy to miss entirely.
- Compressor `coff = 1 + (ratio == 4)`; `wkv`/`wgate` bf16, `ape` fp32
- Indexer: `wq_b` [8192, 1536] fp8, `weights_proj` [64, 7168] bf16 (no scale), own ratio-4 compressor
- mHC per block: `mix_hc = (2+4)*4 = 24`, `hc_dim = 4*7168 = 28,672`, `*_fn` [24, 28672] fp32
  ⇒ 2.75 MB/block × 2 sites — small in bytes, but `hc_split_sinkhorn` runs **20 iterations** per site

## 11. Phase 2 — memory fit. DONE (`derive/fit.py`).

**Decision taken 2026-09-16:** price the **assigned 8×H200 / TP8**, cite the MI355X recipe as a
divergence finding, and keep the recipe's hardware-independent settings (`--kv-cache-dtype fp8`,
`--gpu-memory-utilization 0.9`).

### Headline: per-GPU weights are NOT `total/8`
| | GB/GPU |
|---|---:|
| replicated on every rank | 7.38 |
| sharded (857.33 GB ÷ 8) | 107.17 |
| **weights per GPU** | **114.54** |
| naive `total/8` | 108.09 |
| **understatement** | **6.46** |

`model.py` uses **plain `Linear`** — i.e. replicated — for `wq_a`, `wkv`, both compressors, and
most expensively **`shared_experts`**. Only `ColumnParallelLinear` / `RowParallelLinear`, the
routed expert bank, `ParallelEmbedding` and `ParallelHead` actually shard. Replicated, largest first:

| | GB/GPU |
|---|---:|
| `shared_experts` (fp8, plain `Linear`) | 4.096 |
| `attn.compressor` wkv+wgate+ape+norm | 1.345 |
| `wq_a` | 0.683 |
| `hc_attn_fn` + `hc_ffn_fn` (fp32) | 0.341 |
| `gate.weight` | 0.341 |
| `wkv` | 0.228 |
| `indexer.compressor` | 0.220 |
| `gate.tid2eid` (3 layers, int64) | 0.019 |

MTP block = **1.92 GB/GPU resident** but **absent from the decode graph** (speculation disabled).
State both halves.

### KV cache is tiny, and it is REPLICATED not sharded
`num_key_value_heads: 1` + `wkv` is a plain `Linear` ⇒ every rank computes and holds the **full
512-dim latent**; `sparse_attn(q_local_heads, kv_full)` needs it. **No TP division on KV.**

Per sequence at 8,192 ctx, split the way the mimo note insists:
| | bf16 | fp8 (recipe) |
|---|---:|---:|
| fixed (window, **O(1) in context**) | 8.00 MB | 4.00 MB |
| growing (compressed) | 80.67 MB | 40.34 MB |
| **per sequence** | **88.67 MB** | **44.34 MB** |
| ×32 seqs, per rank | 2.84 GB | **1.42 GB** |

Uncompressed MLA at the same context would be 61 × 8192 × 512 = **256 MB/seq @ fp8** — the
compressor buys **~5.8×**. KV is a rounding error here, which is why the fit is weight-dominated
and why bound ranking should not expect KV streaming to lead.

`Compressor.kv_state`/`score_state` are **fp32** and sized by *ratio*, not context: **0.60 GB/GPU**.
Counter-intuitively the ratio-**128** layers carry the *larger* buffer (128 slots vs 8).

### The fit verdict — two numbers, as the rubric demands
```
capacity LOWER BOUND (weights + KV + compressor state, nothing else)
    116.56 GB / 141 GB raw   -> FITS, 82.7% of capacity     [fully derived]

DEPLOYABLE at the recipe's own --gpu-memory-utilization 0.9
    weights            114.54   derived
    KV fp8               1.42   derived
    compressor state     0.60   derived
    activation workspace 4.00   ASSUMED
    comm buffers         2.00   ASSUMED
    reserve              6.00   explicit
    TOTAL              128.56   vs 126.90 usable  ->  -1.66 GB   DOES NOT FIT
```
**Honesty requirement:** the −1.66 GB verdict is *assumption-dominated* — 12 GB of assumed
workspace/comm/reserve against a 1.66 GB margin. What is firm: **116.56 GB/GPU is derived**, leaving
only **24.44 GB/GPU** for workspace + comm + reserve + fragmentation. Whether that suffices is an
**engine** question, not a checkpoint question. Say exactly that; do not present −1.66 as a fact.
bf16 KV instead of fp8 adds 1.42 GB → 129.98 GB, also over.

### Minimal shape if experts are converted to FP8
`convert.py --expert-dtype fp8` adds **+92.17 GB/GPU** → 220.73 GB/GPU at TP8. **Impossible on one
H200 node** (total checkpoint ≈ 1,602 GB vs 1,128 GB node capacity). Minimal shape = **2 nodes,
TP16**: replicated 7.38 + sharded ≈ 99.7 ⇒ ~107 GB/GPU, fits. Divisibility all holds at 16
(384 experts, 128 heads, 16 `o_groups`, 129,280 vocab). **The comm op this introduces:** the
per-layer `RowParallelLinear` all-reduce in `wo_b` and the `MoE` all-reduce now cross the
**inter-node fabric** instead of NVLink — price those two fabrics separately in §5.

### Collectives in the decode graph (read from `model.py`, TP8)
- `wo_b` — `RowParallelLinear.forward` all-reduces, and **casts to fp32 first** (`y = y.float()`)
  → 4 B/element on the wire, not 2. Per layer.
- `MoE.forward` — `dist.all_reduce(y)` over the fp32 accumulator. Per layer.
- `Indexer.forward` — `dist.all_reduce(index_score)` on the **30 ratio-4 layers only**.
- `ParallelEmbedding` — one all-reduce at the input.
- `ParallelHead` — one `all_gather` of logits at the output.
⇒ **2 all-reduces × 62 blocks + 30 indexer all-reduces + 2 endpoint collectives ≈ 156 per step.**
Not the 2/layer a TP convention would assume — the indexer adds a third on half the layers.

## 12. Phase 3 — engine assumptions. FP4-on-SM90 RESOLVED.

**Engine pinned: vLLM `v0.29.0`** (latest stable; recipe requires 0.20.0+). `v0.28.0`'s notes
call out DeepSeek-V4 performance work specifically.

### The expert format is MXFP4 — OCP MX spec, not NVFP4
My Phase-1 shape derivation independently confirms it: `model.py` stores expert weights as
`float4_e2m1fn_x2` `[out, in//2]` with scale `[out, in//32]` in `float8_e8m0fnu`. **e2m1 elements
+ e8m0 scale per 32 along K** is exactly OCP MXFP4. This matters because the vLLM backend
selection for MXFP4 and NVFP4 is different code. *(checkpoint class — derived)*

### On H200 the expert GEMMs run **Marlin W4A16** — i.e. BF16 math
| Finding | Evidence class |
|---|---|
| DeepSeek-V4 MXFP4 MoE backend priority: `FLASHINFER_TRTLLM_MXFP4_MXFP8` → `DEEPGEMM_MXFP4` → `MARLIN` → `BATCHED_MARLIN` | **read from source** — `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` @ v0.29.0 |
| "On SM90 only Marlin W4A16 passes" | **read from PR** — [#53709](https://github.com/vllm-project/vllm/pull/53709), author's own Hopper benchmark |
| Native MXFP4 W4A8 backends gated to Blackwell SM100/SM120 | read from PR + search; **not yet verified in source** |
| Per-backend capability gates live in `is_supported_config` on each kernel class, **not** in the oracle | read from source |

⚠️ **Remaining verification step:** the capability conditionals are in the kernel classes, not the
oracle — `TrtLlmMxfp4ExpertsMonolithic`, `DeepGemmFP4Experts`, `MarlinExperts`.`is_supported_config()`.
Promote "SM90 → Marlin" from *read from PR* to *read from source* by quoting those. Corroborating
logic meanwhile: the PR's entire premise is a **Hopper** speedup obtained by leaving FP4, which only
makes sense if Hopper's FP4 path is the slow one.

⇒ **Runtime precision ≠ storage precision, again.** Experts are stored MXFP4 but computed at
**BF16 tensor-core rate (989.5 TFLOP/s)**, because Marlin W4A16 dequantizes in-kernel to 16-bit
activations. Memory traffic stays at the FP4 rate; only the math runs at BF16. Price the expert
GEMMs against 989.5, **not** 1,979.

### The FP8-expert escape hatch is closed twice over
`VLLM_DSV4_FP4_DEQUANT=1` (PR #53709) re-encodes MXFP4 → block-FP8 (e4m3, 128×128) **at load
time**, bit-exact (MX scales are pure powers of two, so e4m3 absorbs the per-32 group ratio), and
runs the existing block-FP8 MoE kernel on FP8 tensor cores. It is unavailable here for two
independent reasons:

1. **It is not in any release.** PR #53709 is **open, not merged** (dated 2026-08-25). On the
   pinned v0.29.0 the flag does not exist.
2. **It would not fit if it did.** 2× expert bytes = **+92.17 GB/GPU → 220.73 GB/GPU** vs 141 GB.
   This is the Phase-2 number, now with an engine-side reason to care about it.

Published speedups from that PR — note the split:
| | gain |
|---|---|
| prefill 2,048 tok | 1.53× |
| prefill 8,192 tok | 1.38× |
| concurrent prefill, 32 req @ 8k | **2.04×** |
| **decode, 32 concurrent** | **1.13×** |

**Decode barely moves.** Exactly what a memory-bound decode predicts: doubling the bytes you
stream while speeding up the math is close to a wash. Prefill is compute-bound, so it gains.
This is a strong independent signal for the Phase-4 ranking.

### Settled from the recipe rather than assumed
- `cudagraph_mode: FULL_DECODE_ONLY` → decode runs under CUDA graphs, prefill does not.
  Launch overhead ≈ graph-replay (~2 µs), not eager (~5 µs). *(read from docs — vendor recipe)*
- `--kv-cache-dtype fp8` → KV at 1 B/element. *(read from docs — vendor recipe)*
- `--compilation-config mode 3` → torch.compile level, fusion decisions delegated there.
- `--enable-expert-parallel` **absent** → TP-only for the MoE at the vendor's shape.

### MoE sharding is expert-parallel *inside* the TP group
`model.py` `MoE.__init__`: `n_local_experts = n_routed_experts // world_size`,
`experts_start_idx = rank * n_local_experts`. Each rank owns **48 whole experts**, not a width
shard of all 384. So the "TP8" label hides EP8 behaviour for the routed bank — which is why the
collective is an `all_reduce` over a full-width accumulator rather than a reduce-scatter.

### First-cut decode arithmetic (Phase-4 seed, NOT yet checked twice)
At 32 tokens × top-6 over 384 experts, expected **distinct** experts touched per layer
= `384 × (1 − (1−1/384)^192) ≈ 151` globally, **≈18.9 of the 48 on each rank**.
Each expert = 3 × (7168×3072) MXFP4 = 33.03 MB + 2.06 MB scales = **35.09 MB**.
⇒ ≈663 MB per layer per rank × 61 layers ≈ **40.4 GB streamed per decode step per rank**
(≈42% of the 96.7 GB local bank). At 4.8 TB/s ⇒ **≈8.4 ms/step**, ⇒ ~3,800 tok/s aggregate at
batch 32. **Sanity check: PR #53709 measured 1,642–1,856 tok/s decode at 32 concurrent — same
order of magnitude**, which is the right kind of agreement for a roofline floor to have with a
real measurement. Assumes uniform routing; real imbalance only makes it worse.

## 12a. Test suite — `tests/test_derivations.py`, 29 passing

`python3 tests/test_derivations.py` (standalone, no deps; also pytest-collectable).
Two kinds of test, deliberately:
- **golden** — pins every number the design note quotes, so a refactor that moves a figure fails loudly
- **reasoning** — pins the *arguments*. A number can stay right while its justification rots.
  e.g. `test_fp4_expert_storage_is_forced_not_merely_labelled` asserts the FP8 reading *exceeds*
  `total_size`, which is what makes FP4 a derivation instead of a label;
  `test_tid2eid_int32_produces_the_exact_known_residual` pins the 9,308,160 B derivation;
  `test_kv_window_term_is_invariant_in_context` pins the O(1) split at 2k/8k/1M.

**A failing test already caught a real error.** `220.73 GB` for the FP8-expert conversion mixed
derived weights with *assumed* workspace. The clean derived figure is **weights alone = 206.71
GB/GPU** vs 141 GB. Design note §2.6 corrected. Rule going forward: **never compare a total
containing assumed terms against a hard capacity limit** when the derived terms already settle it.

## 12b. D2 draft exists — `docs/DESIGN-NOTE.md`

Written 2026-09-16. Sections 1 (engine assumptions) and 2 (memory fit) complete. Section 3
(bound hypotheses) partial: hypothesis 1 has arithmetic, 2 and 3 have shape but no numbers, and
**none of the three controlled experiments is written**. Section 4 lists next steps ranked by
impact.

House style for this file, keep it: **no em dashes, no en dashes**, plain human prose, short
sentences. Verified zero of both.

Length is ~3,700 words, over the "one to two pages" ask. That is fine for a working draft with
status markers, but **the final pass must cut it down**. The case says match the reference note's
rigor, not its length. Cut candidates: §1.2 detail, §2.4 table duplication, §4 prose.

## 13. Phase 4 — COMPLETE. `derive/bounds.py` + D2 §3 written.

Per rank per decode step, baseline, 8×H200 TP8:

| Rank | Term | Time | Bound |
|---|---|---:|---|
| 1 | expert weight streaming (routed 40.47 GB + shared 4.03 GB) | 9.27 ms | memory |
| 2 | launch/dispatch, ~4,326 kernels under graph capture | 2.16 ms | latency |
| 3 | other HBM (attn weights 4.69, KV 1.17, head 0.23) | 1.41 ms | memory |
| | compute, all precisions (770.67 GFLOP) | 0.77 ms | compute |
| | collectives, bandwidth only (224.93 MB) | 0.25 ms | comm |

- **Total HBM 51.26 GB/step → 10.68 ms. AI = 15.03 FLOP/B vs BF16 ridge 206 → memory-bound by 14×.**
- **Step floor 13.09 ms → 2,444 tok/s** vs PR #53709's measured 1,642–1,856 tok/s.
  **Floor sits 1.40× ABOVE measurement, which is the only correct direction.** Below would be an error.
- FP32 row is 2% of FLOPs but **31% of compute time** (router + mHC on CUDA cores at 67 TF/s).
  Irrelevant at batch 32; first to matter if batch grows.

### Corrections this phase forced
1. **Collectives are 154, not 156.** Speculation disabled ⇒ MTP block never runs ⇒ contributes
   zero collectives. 156 counted a block that does not execute. Test updated; D2 §1.4 updated.
2. **`hc_split_sinkhorn` is ONE fused TileLang kernel**, not 20 launches. `T.serial(iters-1)` runs
   in registers on a 4×4 fragment (`inf_kernel.py:372-438`). The draft's worry about 2,440 small
   mHC ops was wrong. Killed it.
3. `220.73 GB` appeared a second time in §1.2; corrected to weights-alone 206.71 GB.

### The sharpest prediction in the note (D2 §3, Experiment 3)
On the 30 ratio-4 layers the indexer selects `index_topk = 1024` positions. Past ctx 4,096 there
are more than 1,024 compressed slots, so **the attention gather stops growing entirely — it is
capped by `index_topk`, not by context.** Only the 31 ratio-128 layers (ctx/128) and the indexer's
own scan (ctx/4) still scale. ⇒ **quadrupling context 8k→32k adds <3% to the step.** Falsifiable,
specific, and cheap to test.

### Communication defended into 4th/5th, not assumed
Bandwidth: 224.93 MB @ 900 GB/s = **0.25 ms**. But 154 separate collectives each carry a
small-message NVLink latency floor of ~5–10 µs; at 7 µs that is **1.08 ms**, which would make it
rank 3. Honest answer is a **range, 0.25–1.3 ms**: 4th or 5th on bandwidth, 3rd at worst, never
1st (rank 1 is 9.27 ms). Under the 2-node TP16 shape it could become rank 1 or 2; not priced.

## 14. Phase 5 — remaining

- [ ] **D1, the planner-schema YAML.** Not started. Follow `mimo-v2.5.yaml`: `spec:` + `provenance:`
      with verified/estimated/open/unmodelled, layer schedules written out, checkpoint names verbatim.
- [ ] Trim D2 from ~5,100 words toward the one-to-two page ask.
- [ ] Optional: promote the Marlin claim from *read from PR* to *read from source*.

## 13b. Phase 4 — original checklist (all done)

- [ ] Re-derive the expert-streaming number carefully (it is the likely #1 bound) and add the
      attention-weight and KV streaming terms.
- [ ] Roofline ridges: **BF16 989.5 TFLOP/s for the expert GEMMs (Marlin W4A16)**, FP8 1,979 for
      the backbone, FP32 67 for the router and the fp32 all-reduces, HBM 4.8 TB/s, NVLink 900 GB/s.
- [ ] Price the ≈156 collectives; remember `wo_b`'s all-reduce is **fp32 on the wire**.
- [ ] Rank top-3 with compute/memory/comm labels + one falsifiable experiment each
      (hold fixed / vary / observable / rejection threshold / what it cannot establish).
- [ ] Then Phase 5: D1 YAML + D2 design note.

## 9. Conventions for this repo

- Deliverables live at the repo root or a clearly named folder; source material stays in `context/`.
- Every number in the design note traces to a file + field, or is labeled an assumption inline.
- Checkpoint names (tensors, modules, quant labels) copied **verbatim**, never normalized.
- Engine claims carry their tag: *read from engine source* / *read from docs* / *assumed*.
