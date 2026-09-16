# DeepSeek-V4-Pro: Decode-Step Execution Prediction

Response to the GitM Git Machine execution-graph case: predict how **DeepSeek-V4-Pro** executes
one decode step, derived from the checkpoint's own files, before ever seeing it run. No GPU, no
traces, nothing executed.

**Status:** Complete. Both deliverables are in [`DELIVERABLES/`](DELIVERABLES/).

---

## The task

| | |
|---|---|
| **Checkpoint** | `deepseek-ai/DeepSeek-V4-Pro` at revision `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| **Scale** | 1.6 T total, 49 B active, MXFP4 experts with FP8 backbone, 61 layers plus 1 MTP |
| **Engine** | vLLM `v0.29.0` |
| **Baseline** | 32 sequences, 8,192 cached tokens each, 1 new token per sequence per step, text only, speculation disabled |
| **Hardware** | 8 x H200 SXM, NVLink, TP8. See *Hardware divergence* below |
| **Deliverables** | D1 planner-schema YAML, D2 design note |

Full assignment: [`context/case-deepseek-v4-pro.md`](context/case-deepseek-v4-pro.md).

---

## Repo layout

```
DELIVERABLES/            THE SUBMISSION. Both deliverables and nothing else.
  deepseek-v4-pro.yaml   D1. Model spec in the planner schema.
  DESIGN-NOTE.md         D2. Design note, submission length.
  README.md              Index and how to reproduce.
docs/DESIGN-NOTE.md      Long working version of D2, about 3x the length. The appendix.
.agent/context.md        Working context: every finding, decision, and open question,
                         in the order they happened.
derive/
  weights.py             Weight accounting. Validates against the index's total_size.
  fit.py                 Per-GPU memory fit. Sharding aware.
  bounds.py              Per-step cost model: bytes, FLOPs, collectives, launches, ranking.
tests/
  test_derivations.py    39 checks: the numbers, the arguments behind them, and D1
                         against config.json.
context/
  case-deepseek-v4-pro.md              The assignment.
  GitM-Technical-Overview.extracted.md Company overview, PDF text decoded by hand.
  GLM-5.2-DESIGN-NOTE.reference.md     GitM's own worked example at the same baseline.
  checkpoint/                          config.json, index, inference/*.py from the HF repo.
```

## Reproduce

```bash
python3 derive/weights.py          # weight accounting vs published total_size
python3 derive/fit.py              # per-GPU fit at 8xH200 / TP8
python3 derive/bounds.py           # per-step cost model and the ranking
python3 tests/test_derivations.py  # 39 checks, no dependencies
```

None of these needs a GPU, network access, or the weights themselves. They read only the metadata
in `context/checkpoint/`.

---

## Headline results

### The weight derivation is byte exact

```
PREDICTED  864,704,792,696 bytes   145,116 tensors
PUBLISHED  864,704,792,696 bytes   145,116 tensors
DIFFERENCE              +0 bytes         0
```

Not "within 2 percent". Zero, on both bytes and tensor count. Every projection dimension,
per-layer-kind rate, and expert bank size is confirmed. The case names shapes as the most common
failure mode, so this is the gate everything else sits on.

### The predicted decode step

Per rank, per step, at the baseline:

| Rank | Term | Time | Bound by |
|---|---|---:|---|
| 1 | Expert weight streaming, routed plus shared | 9.27 ms | memory |
| 2 | Launch and dispatch, about 4,326 kernels under graph capture | 2.16 ms | latency |
| 3 | Other HBM traffic: attention weights, KV, head | 1.41 ms | memory |
| | Compute, all precisions | 0.77 ms | compute |
| | Collectives, bandwidth only | 0.25 ms | communication |

Total HBM traffic is 51.26 GB per rank per step. Arithmetic intensity is **15.03 FLOP per byte
against a BF16 ridge of 206**, so the step is memory bound by a factor of 14.

**Step floor 13.09 ms, about 2,444 tokens per second aggregate.** PR #53709 measured 1,642 to
1,856 tokens per second for decode at 32 concurrent on real Hopper hardware. The floor sits
**1.40x above** the measurement, which is the only correct relationship for a roofline floor. A
floor landing below a real measurement would mean an arithmetic error.

### Memory, per GPU at TP8

| | GB per GPU |
|---|---:|
| Weights, replicated on every rank | 7.38 |
| Weights, sharded (857.33 divided by 8) | 107.17 |
| **Weights total** | **114.54** |
| KV cache at fp8, replicated | 1.42 |
| Compressor state, fp32 | 0.60 |
| **Derived subtotal** | **116.56** |

Capacity lower bound: **116.56 GB of 141 GB, so it fits at 82.7 percent.** Every number in that
total is derived.

A configuration you would actually deploy, at the recipe's own `--gpu-memory-utilization 0.9`,
comes to 128.56 GB against 126.90 usable, so 1.66 GB over. That second number is **not** presented
as a finding, because it sits underneath 12 GB of assumed workspace and reserve. What is firm is
that 116.56 GB is derived and leaves 24.44 GB for workspace, communication, reserve, and
fragmentation. Whether that suffices is an engine question, not a checkpoint question.

---

## Findings

### Structure

- **61 layers plus 1 MTP block.** `compress_ratios` has **62** entries, which is
  `n_layers + n_mtp_layers`. That resolves what first read as an off-by-one.
- **Three layer schedules that do not tile:** 30 backbone layers at compress ratio 4 carry an
  overlapping compressor plus an indexer; 31 at ratio 128 carry a compressor only; 3 layers route
  by hash table instead of a learned gate, and the other 58 carry `gate.bias`.
- **MLA-style latent KV:** one KV head, head dimension 512, with LoRA on both the query and the
  output projection.
- **mHC residuals** run 20 Sinkhorn iterations at two sites per layer, but as **one fused
  TileLang kernel**, not 20 launches.
- **MoE is expert parallel inside the TP group.** Each rank owns 48 whole experts out of 384,
  not a width shard of all 384.

### Memory

- **Per-GPU weights are not `total / 8`.** `model.py` uses plain `Linear`, which is replicated,
  for `wq_a`, `wkv`, both compressors, and most expensively `shared_experts`. The naive division
  gives 108.09 GB and understates by **6.46 GB**.
- **KV is replicated, not sharded**, because there is one KV head and `wkv` is a plain `Linear`.
- **KV is small:** 44.34 MB per sequence at fp8, of which only 40.34 MB grows with context. An
  uncompressed latent cache would be 256 MB, so the compressor buys about 5.8x.
- **FP4 scale metadata is 48.36 GB.** FP4 scales are `[out, in/32]`, keeping full output
  resolution, so they cost one sixteenth of expert weight bytes rather than one sixteen-thousandth.

### Engine

- **Experts are MXFP4**, the OCP MX format, confirmed independently from the checkpoint's shapes.
- **On H200 they run through Marlin W4A16**, read from source one gate at a time:
  `TrtLlmMxfp4ExpertsBase` requires `is_device_capability_family(100)`, `DeepGemmFP4Experts`
  requires family 100 or 120, and `MarlinExpertsBase` requires only `has_device_capability((7, 5))`.
  Hopper is 9.0, so it fails both Blackwell checks and clears Marlin's floor. `marlin_moe.py`
  asserts `hidden_states.dtype in [torch.float16, torch.bfloat16]`, so experts are **stored at FP4
  but computed at the BF16 rate of 989.5 TFLOP/s**, not FP8's 1,979. Memory traffic stays at the
  FP4 rate.
- **The FP8 expert conversion is closed off twice.** The vLLM flag is in an open, unmerged PR so
  it does not exist in v0.29.0, and it would add 92.17 GB per GPU, taking weights alone to 206.71
  GB against 141 GB.
- **Collectives are 154 per step, not 156.** Speculation is disabled, so the MTP block is resident
  at 1.92 GB per GPU but never runs and contributes no collectives. A plain TP convention would
  predict 124 and miss the indexer's third collective on 30 layers.
- **`wo_b`'s all-reduce carries fp32**, because `RowParallelLinear` casts before the call. The
  most frequent collective in the graph moves 4 bytes per element, not 2.

### Checkpoint versus implementation

1. **`gate.tid2eid` is stored int64 while `model.py` declares int32.** Derived, not assumed: at
   int32 the prediction fell exactly 9,308,160 bytes short, which is precisely
   `3 hash layers x 129,280 x 6 x 4 bytes`.
2. **`wo_a` is FP8 in the checkpoint but loaded BF16 by the reference implementation.** The
   vendor's own comment at `model.py:539` says so.
3. **No `trust_remote_code` path exists.** No `auto_map`, no `modeling_*.py`. `inference/` is a
   standalone implementation needing `convert.py` and `torchrun`. vLLM runs its own in-tree code,
   so bound analysis cites that and never `inference/model.py`.

### Hardware divergence

A vendor recipe exists at `recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro`, and the assignment says a
published recipe supersedes the 8 x H200 default. But that recipe targets **8 x AMD MI355X with
288 GB each**, where the memory question answers itself.

We priced the assigned H200s and treated the divergence as a finding. The brief says the fit
section carries the weight for this model, and that is only true at 141 GB per GPU. The recipe is
still cited for the settings that do not depend on hardware, specifically `--kv-cache-dtype fp8`
and `--gpu-memory-utilization 0.9`.

---

## The three experiments

Each varies one engine-level knob and records one number the engine already reports. None needs a
profiler, a trace, or a code change.

| | Vary | Predicted | Rejected if |
|---|---|---|---|
| **E1** expert streaming dominates | batch 1 to 64 | 2.6x step time for 32x batch | within 1.3x, or more than 8x |
| **E2** launch is rank 2 | graph capture on or off | 2.5x ratio | below 1.3x, or above 4x |
| **E3** non-expert traffic is dense weights, not KV | context 2k to 32k | flat within 5 percent | grows more than 15 percent |

E3 is the sharpest claim. On the 30 ratio-4 layers the indexer selects `index_topk`, which is
1,024 positions. Once context passes 4,096 there are more than 1,024 compressed slots, so **the
attention gather stops growing entirely.** It is capped by `index_topk`, not by context. Only the
ratio-128 layers and the indexer's own scan still scale, so quadrupling context adds under 3
percent to the step.

Each experiment also states what it cannot establish. E1, for instance, cannot separate bandwidth
saturation from expert load imbalance, because both bend the curve the same way.

---

## Deliverables

Both live in [`DELIVERABLES/`](DELIVERABLES/).

**D1**, [`DELIVERABLES/deepseek-v4-pro.yaml`](DELIVERABLES/deepseek-v4-pro.yaml). Follows
`mimo-v2.5.yaml`. Both non-tiling schedules written out in full, per-op precision in
`op_dtype_overrides`, checkpoint module names kept verbatim, and family contradictions captured in
comments rather than smoothed away. The `provenance` block separates nine verified claims from two
estimated fields, four open questions, and four unmodelled areas. Nine tests check it against
`config.json` field by field so it cannot drift.

**D2**, [`DELIVERABLES/DESIGN-NOTE.md`](DELIVERABLES/DESIGN-NOTE.md). Engine assumptions, memory
fit, and three ranked bound hypotheses with falsifiable experiments, at submission length.
[`docs/DESIGN-NOTE.md`](docs/DESIGN-NOTE.md) keeps the full reasoning at about three times the
length, as an appendix rather than a replacement.

## What remains

- **Count the kernels.** The 4,326 figure is the softest number in the note and it carries rank 2.
  One profiler trace replaces the estimate with a count.
- **Settle whether collectives are latency bound**, which decides whether communication is rank 3
  or rank 5. Recording NCCL time during experiment 1's batch sweep does it.
- **Replace the three assumed memory terms** with vLLM's startup memory profile.
- **Price the two fabrics separately** for the two-node TP16 shape.

---

## Method note

Two evidence classes are kept separate throughout, because the case is explicit that conflating
them is the failure: **what the checkpoint establishes** (shapes, counts, storage precision, layer
schedules) versus **what depends on the engine** (fusion, launch behaviour, collective
implementation, runtime precision). Every number carries its provenance, and every assumption is
labelled rather than smuggled.

The test suite enforces both halves. Golden tests pin every number the design note quotes.
Reasoning tests pin the arguments, because a number can stay correct while its justification
quietly rots. A third group checks D1 against `config.json` field by field, because a hand-written
spec that drifts from the checkpoint is worse than no spec, since it still looks authoritative. One example: `test_fp4_expert_storage_is_forced_not_merely_labelled` asserts that
the FP8 reading exceeds `total_size` on its own, which is what makes FP4 a derivation rather than
a label copied from a config field.
