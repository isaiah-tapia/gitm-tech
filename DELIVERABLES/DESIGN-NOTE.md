# DeepSeek-V4-Pro: Predicted Execution of One Decode Step

| | |
|---|---|
| Checkpoint | `deepseek-ai/DeepSeek-V4-Pro` at `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| Engine | vLLM `v0.29.0` |
| Hardware priced | 8 x H200 SXM, NVLink, TP8 |
| Baseline | 32 sequences, 8,192 cached tokens, 1 new token per sequence per step, text only, speculation disabled |

No GPU, no traces, nothing run. Every figure is a roofline floor, a lower bound rather than a
target. Arithmetic reproduces with `python3 derive/{weights,fit,bounds}.py`; 39 checks in
`tests/test_derivations.py` pin both the numbers and the arguments behind them.

**Hardware note.** A vendor recipe exists and supersedes the default, but it targets 8 x AMD
MI355X at 288 GB, where the memory question answers itself. I priced the assigned H200s, because
the brief says the fit section carries the weight for this model and that is only true at 141 GB.
The recipe's hardware-independent settings (`--kv-cache-dtype fp8`,
`--gpu-memory-utilization 0.9`, `cudagraph_mode: FULL_DECODE_ONLY`) are adopted and cited.

**Evidence classes.** The checkpoint establishes shapes, counts, storage precision and layer
schedules. The engine establishes fusion, launch behaviour, kernel selection and runtime
precision. They are never mixed below. Note that this checkpoint has **no `trust_remote_code`
path**: no `auto_map`, no `modeling_*.py`. `inference/` is a standalone implementation needing
`convert.py` and `torchrun`, so every execution claim cites vLLM and not `inference/model.py`.

---

## 1. Engine assumptions

### 1.1 The experts are stored FP4 and computed BF16

Expert weights are MXFP4, derived not labelled: `[out, in//2]` in `float4_e2m1fn_x2` with a scale
`[out, in//32]` in `float8_e8m0fnu` is the OCP MX format by definition. **(checkpoint)**

H200 is SM90 and has no native FP4 tensor path. vLLM's DeepSeek-V4 backend priority is
`FLASHINFER_TRTLLM_MXFP4_MXFP8`, `DEEPGEMM_MXFP4`, `MARLIN`, `BATCHED_MARLIN`
(`fused_moe/oracle/mxfp4.py`). Reading each gate: **(source, v0.29.0)**

| Backend | Device check | SM90 |
|---|---|---|
| `TrtLlmMxfp4ExpertsBase` | `p.is_cuda() and p.is_device_capability_family(100) and has_flashinfer()` | no |
| `DeepGemmFP4Experts` | `is_device_capability_family(100)` or `(120)` | no |
| `MarlinExpertsBase` | `p.is_cuda() and p.has_device_capability((7, 5))` | **yes** |

Hopper is 9.0, so it fails both Blackwell family checks and clears Marlin's floor. That the
activations are 16 bit is source too, not naming convention: `marlin_moe.py` asserts
`hidden_states.dtype in [torch.float16, torch.bfloat16]`.

> **Experts are stored at FP4 but computed at the BF16 rate of 989.5 TFLOP/s, not FP8's 1,979.**
> Memory traffic stays at the FP4 rate. Only the math is slower.

This halves the peak that the largest term in the step is priced against.

### 1.2 The FP8 conversion is closed off twice

`VLLM_DSV4_FP4_DEQUANT=1` re-encodes MXFP4 to block FP8 at load time, bit exact. It is in an
**open, unmerged PR** (#53709), so it does not exist in v0.29.0 **(source)**, and it would add
92.17 GB per GPU, taking weights alone to 206.71 GB against 141 GB.

That PR's benchmarks carry a useful signal: the conversion buys **2.04x on concurrent prefill but
only 1.13x on decode**. Exactly what memory bound decode predicts, since doubling the bytes read
while speeding up the math roughly cancels. Used in section 3.

### 1.3 Launch, collectives, sharding

`cudagraph_mode: FULL_DECODE_ONLY` **(docs)** means decode runs captured and prefill does not, so
per-kernel cost is device-side dispatch rather than CPU launch. `mode: 3` delegates fusion to
torch.compile; which ops actually fuse is **assumed**, I did not read the generated graph. NCCL
algorithm selection is **assumed**.

Collectives counted from `dist.*` call sites, not from convention:

| Collective | Where | Count |
|---|---|---:|
| `wo_b` all-reduce, **fp32** | every running block | 61 |
| MoE all-reduce, fp32 accumulator | every running block | 61 |
| Indexer all-reduce | ratio-4 layers only | 30 |
| Embedding all-reduce, logits all-gather | endpoints | 2 |
| | | **154** |

Three easy misses. A flat two-per-layer gives 124 and drops the indexer's third collective on 30
layers. `RowParallelLinear` calls `y = y.float()` **before** the all-reduce, so the most frequent
collective moves 4 bytes per element, not 2. And it is 61 blocks, not 62: speculation is disabled,
so the MTP block is resident at 1.92 GB per GPU but never executes. Counting it gives 156 and
bills a block that does not run. Both halves need saying, since leaving it out of the memory total
understates the footprint by the same 1.92 GB.

`MoE.__init__` gives each rank `n_routed_experts // world_size` **whole** experts, so what is
labelled TP is expert-parallel for the routed bank. That is why the collective is an all-reduce
over a full-width accumulator rather than a reduce-scatter.

---

## 2. Memory fit

### 2.1 The shapes are verified before anything is built on them

```
PREDICTED  864,704,792,696 bytes   145,116 tensors
PUBLISHED  864,704,792,696 bytes   145,116 tensors
DIFFERENCE              +0 bytes         0
```

Exact on both. A wrong projection dimension would have to be cancelled by another wrong dimension
to the byte. Getting there surfaced a contradiction: `model.py` declares `gate.tid2eid` as
`int32`, but at int32 the total falls exactly 9,308,160 bytes short, which is precisely
`3 hash layers x 129,280 x 6 x 4 bytes`. **The stored tensor is int64**, derived from the residual
rather than assumed.

### 2.2 Per-GPU weights are not the total over 8

`model.py` uses plain `Linear`, which is replicated, for `wq_a`, `wkv`, both compressors, and most
expensively `ffn.shared_experts` (4.10 GB per GPU alone, because the shared expert runs for every
token and is never routed).

| | GB per GPU |
|---|---:|
| Replicated on every rank | 7.38 |
| Sharded (857.33 over 8) | 107.17 |
| **Weights per GPU** | **114.54** |
| What `total / 8` would say | 108.09 |
| **Understated by** | **6.46** |

### 2.3 Quantization metadata is 48 GB

FP8 scales are `[out/128, in/128]` and negligible. **FP4 scales are `[out, in/32]`**, keeping full
output resolution, so they cost **one sixteenth** of expert weight bytes rather than one
sixteen-thousandth. The expert bank is 773.71 GB of weights plus **48.36 GB of scales**.

### 2.4 KV is small and replicated

`num_key_value_heads` is 1 and `wkv` is a plain `Linear`, so every rank holds the whole 512-wide
latent and attends with its local query heads. Nothing about the KV divides by TP.

The cache is `window + context/ratio` entries per layer, and **only the second term grows**. Per
sequence: 3,997,696 elements fixed, plus 4,924 elements per token of context (30 layers contribute
`512/4 + 128/4`, 31 contribute `512/128`).

| At 8,192 context, fp8 | |
|---|---:|
| Fixed, sliding window, O(1) in context | 4.00 MB/seq |
| Growing, compressed | 40.34 MB/seq |
| **32 sequences, per rank** | **1.42 GB** |

An uncompressed latent would be 256 MB per sequence, so the compressor buys about 5.8x. Counting
the window per-token inflates the rate about 33x and caps concurrency far below what the hardware
allows. Compressor `kv_state`/`score_state` are fp32 and sized by ratio, not context: 0.60 GB.

### 2.5 The result, as two numbers

**Capacity lower bound**, weights plus KV plus compressor state, everything derived:

```
116.56 GB of 141 GB raw     fits, 82.7% of capacity
```

**A configuration you would deploy**, at the recipe's own utilization 0.9:

| | GB | |
|---|---:|---|
| Weights | 114.54 | derived |
| KV at fp8 | 1.42 | derived |
| Compressor state | 0.60 | derived |
| Activation workspace | 4.00 | **assumed** |
| Communication buffers | 2.00 | **assumed** |
| Reserve | 6.00 | explicit |
| **Total** | **128.56** | vs 126.90 usable, **-1.66** |

I will not call that a failure to fit. The shortfall is 1.66 GB sitting underneath 12 GB of
numbers I guessed, so the assumptions dominate the result and the result is not a finding about
the model. What **is** a finding: 116.56 GB per GPU is derived and leaves 24.44 GB for workspace,
communication, reserve and fragmentation. Whether that suffices is a question about vLLM's
allocator, and one line of its startup memory profile settles it.

### 2.6 The minimal shape if experts go to FP8

`--expert-dtype fp8` doubles the bank: +92.17 GB per GPU, taking **weights alone** to 206.71 GB
against 141 GB. I compare weights against raw capacity here, with no workspace in the total,
because the weights overflow on their own and adding a guess would make it load bearing in an
argument that does not need one.

The whole checkpoint in that form is about 1,602 GB against 1,128 GB of node capacity, so the
minimal shape is **two nodes at TP16**: replicated stays 7.38, sharded drops to about 99.7, so
roughly 107 GB per GPU. Divisibility holds at 16 for experts, heads, `o_groups` and vocabulary.

That shape moves the two per-layer all-reduces off NVLink and onto the inter-node fabric: 122
collectives per step at a fraction of the bandwidth and several times the latency, with the `wo_b`
one carrying fp32. I have not priced those two fabrics separately, and it is the one place where
communication could plausibly become rank 1.

---

## 3. Top three bound hypotheses

Per rank per step. HBM traffic totals **51.26 GB**, of which routed experts are 79.0 percent,
attention and compressor and indexer weights 9.1, shared expert 7.9, and everything else
(mHC, router, KV gather, indexer scan, `lm_head`) 4.1 combined.

Compute totals **770.67 GFLOP**: 457.41 at FP8, 297.14 at BF16 (Marlin experts and attention
core), 16.12 at FP32. Arithmetic intensity is **15.03 FLOP per byte against a BF16 ridge of 206**,
so the step is memory bound by a factor of 14. Not a close call. The FP32 row is 2 percent of the
FLOPs but 31 percent of the compute time, because the router and mHC run on CUDA cores at 67
TFLOP/s; irrelevant at batch 32, first to matter if batch grows.

| Rank | Term | Time | Bound by |
|---|---|---:|---|
| 1 | Expert weight streaming, routed plus shared | 9.27 ms | **memory** |
| 2 | Launch and dispatch, about 4,326 kernels under capture | 2.16 ms | **latency** |
| 3 | Other HBM: attention weights, KV, head | 1.41 ms | **memory** |
| | Compute, all precisions | 0.77 ms | compute |
| | Collectives, bandwidth only | 0.25 ms | communication |

Serialized, the step floor is **13.09 ms**, about **2,444 tokens per second** at batch 32. PR
#53709 measured **1,642 to 1,856 tok/s** for decode at 32 concurrent on real Hopper. **The floor
sits 1.40x above the measurement**, which is the only correct relationship: a floor assumes perfect
overlap and vendor peak, so one landing below a real measurement would mean an arithmetic error.
The 1.4x gap is where imperfect overlap, real bandwidth efficiency and routing imbalance live.

### Where communication sits, argued rather than assumed

On bandwidth it is trivial: 224.93 MB at 900 GB/s is 0.25 ms, under 2 percent. On latency it may
not be. There are 154 separate collectives, and a small-message all-reduce over NVLink has a floor
around 5 to 10 microseconds regardless of payload; at 7 that is 1.08 ms, which would make it rank
3 and push the other HBM traffic to 4. So the honest answer is a **range of 0.25 to 1.3 ms**:
fourth or fifth on bandwidth, third at worst, and never first, because rank 1 is 9.27 ms and the
entire collective budget cannot approach it. Graph capture amortizes the CPU side of launching
these but not the device-side synchronisation between ranks, which is why this is a latency
question rather than a launch one.

### Experiment 1, for hypothesis 1

**Hold fixed** model, engine, hardware, TP8, context 8,192, graph mode, KV dtype, speculation off.
**Vary** batch across 1, 2, 4, 8, 16, 32, 64. **Record** median per-step latency.

Distinct experts touched per layer is `384 x (1 - (1 - 1/384)^(6B))`, strongly sublinear: about
5.9 at batch 1, 45 at 8, 151 at 32, 243 at 64. Routed traffic follows that curve; every other term
is either flat in batch (dense weights, launch) or linear (KV gather, collective payloads). Three
different worlds, three different shapes.

**Predicted** about 5.0 ms at batch 1 rising to about 13.1 ms at batch 32, so **2.6x for a 32x
batch increase**, with per-token throughput improving roughly 12x before flattening.

**Rejected if** batch 32 lands within **1.3x** of batch 1, which would mean something
batch-independent dominates; or if latency grows more than **8x**, which would mean the step is
closer to linear in tokens than a shared bulk read.

**Cannot establish** that the bytes are expert weights specifically, since any term following the
same routing curve looks identical; confirming volume needs `dram__bytes_read.sum`. It also cannot
separate bandwidth saturation from expert load imbalance, because both bend the curve the same
way. Imbalance needs a trace, since it depends on what the tokens are.

### Experiment 2, for hypothesis 2

**Hold fixed** everything, batch 32, context 8,192. **Vary** `cudagraph_mode` between
`FULL_DECODE_ONLY` and `NONE`. **Record** median per-step latency in both and the ratio.

Capture removes per-kernel CPU launch and leaves the kernels untouched, so the difference is
almost entirely the quantity in question.

**Predicted** about 13.1 ms captured against about 32.6 ms eager, a ratio of **2.5x**, from roughly
4,326 kernels at the assumed 0.5 and 5.0 microsecond costs.

**Rejected if** the ratio is below **1.3x**, meaning the kernel count or per-launch cost is far
lower than assumed and launch is not rank 2; or above **4x**, meaning launch is a larger share of
the captured step than 2.16 ms and moves toward rank 1.

**Cannot establish** a kernel count, only an aggregate cost, and it conflates kernel launch with
collective launch since capture amortizes both. A profiler kernel trace gives the count, and that
is the cheaper way to check the softest number here.

### Experiment 3, for hypothesis 3

**Hold fixed** batch 32, TP8, engine, graph mode, KV dtype. **Vary** context across 2,048, 8,192
and 32,768. **Record** median per-step latency plus reported KV block usage to confirm context
actually changed.

This is the sharpest prediction in the note and it rests on a detail that is easy to miss. On the
30 ratio-4 layers the indexer selects `index_topk`, which is 1,024. Once context passes 4,096
there are more than 1,024 compressed slots, so **the attention gather stops growing entirely**. It
is capped by `index_topk`, not by context. Only the 31 ratio-128 layers, which gather
`context/128`, and the indexer's own scan, which reads `context/4` to choose its top-k, still
scale. Quadrupling context from 8,192 to 32,768 adds roughly 1.5 GB to a 51.26 GB step.

**Predicted nearly flat, within about 5 percent across a 16x range of context.**

**Rejected if** latency grows more than **15 percent** from 2,048 to 32,768, meaning either the
engine reads the full cache rather than a sparse gather or the indexer scan is more expensive than
modelled; either promotes KV in the ranking. Linear growth would mean the sparse path is not being
taken at all, the most interesting of the three failures.

**Cannot establish** a split between attention weights and the mHC, router and norm traffic in the
same bucket, since none of those depend on context; that needs per-kernel timing. It says nothing
about prefill, where the gather is genuinely quadratic in the windowed layers.

### What the three share

Each varies one engine-level knob and records one number the engine already reports. None needs a
profiler, a trace, or a code change. The cheapest experiment that can reject a hypothesis is worth
more than a better-instrumented one that takes a week, and if all three run as predicted the model
above is good enough to plan against. If any one fails its rejection condition, the ranking is
wrong and I would rather find out in an afternoon.

---
