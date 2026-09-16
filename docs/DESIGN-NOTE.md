# DeepSeek-V4-Pro: Predicted Execution of One Decode Step

**Draft, complete through section 3.** Section 4 says what I would do next and why.

| | |
|---|---|
| Checkpoint | `deepseek-ai/DeepSeek-V4-Pro`, revision `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| Engine | vLLM `v0.29.0` |
| Hardware priced | 8 x H200 SXM, NVLink, TP8 |
| Baseline | 32 sequences, 8,192 cached tokens each, 1 new token per sequence per step, text only, speculation disabled |

Every number here comes from the checkpoint's own files or from vLLM source at the pinned tag.
No traces, no GPU, nothing was run. Each figure is a roofline floor, meaning a lower bound on
time rather than a target.

Reproduce the arithmetic:

```bash
python3 derive/weights.py        # weight accounting, validated against the index
python3 derive/fit.py            # per-GPU fit at 8xH200 / TP8
python3 derive/bounds.py         # per-step cost model and the ranking
python3 tests/test_derivations.py  # 30 checks on the numbers and the arguments
```

## A note on the hardware

The assignment specifies 8 x H200 and says a published vendor recipe supersedes that default.
A recipe does exist, at `recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro`, but it targets 8 x AMD
MI355X with 288 GB per GPU. On that hardware the memory question answers itself, since the whole
checkpoint is 864.70 GB against 2,304 GB of node capacity.

I priced the assigned H200s instead, and I want to be straight about why. The brief for this
model says the fit section carries the weight. That is only true at 141 GB per GPU. Taking the
MI355X path would follow the letter of the instruction while removing the section the case says
matters most. The recipe is still cited throughout for the settings that do not depend on the
hardware, specifically `--kv-cache-dtype fp8` and `--gpu-memory-utilization 0.9`.

## How I separate what I know from what I am guessing

The case is explicit that mixing these up is the failure mode, so here is the rule I used
everywhere below. There are three sources, and they are not interchangeable.

The **checkpoint files** (`config.json` and `model.safetensors.index.json`) establish shapes,
counts, storage precision, and layer schedules. These are facts about the model.

The **reference implementation** (`inference/model.py`) establishes semantics, meaning what each
operation actually is and how the modules wire together. It is DeepSeek's own code, but it is not
what runs in production.

The **engine** (vLLM at the pinned tag) establishes execution, meaning which operations fuse,
what runs under CUDA graphs, which kernels get picked, and what precision the math actually uses.

One thing worth stating plainly: **this checkpoint has no `trust_remote_code` path.** There is no
`auto_map` in `config.json` and no `modeling_*.py` anywhere in the repo. The `inference/` folder
is a standalone implementation that needs `convert.py` and `torchrun` to run at all. It is not a
Hugging Face integration. vLLM uses its own in-tree implementation, so every claim about
execution below cites vLLM and not `inference/model.py`.

---

# 1. Engine assumptions

Each item is tagged with where it came from: **source** means I read it in vLLM at the pinned
tag, **docs** means the vendor recipe or official documentation, and **assumed** means I am
guessing and the guess is load bearing.

## 1.1 The expert kernel path, which turned out to be the important one

The routed experts are stored in MXFP4. I did not take this from a label. The shapes in
`model.py` are `[out, in//2]` in `float4_e2m1fn_x2` with a scale of `[out, in//32]` in
`float8_e8m0fnu`, and e2m1 values with one e8m0 scale per 32 elements along K is exactly the OCP
MX format. **(checkpoint)**

H200 is SM90, and SM90 has no native FP4 tensor path. That arrives with Blackwell. So the
question is what vLLM does instead, and the answer changes the arithmetic for the largest term
in the whole step.

vLLM picks a MoE backend from a priority list. For DeepSeek-V4 that list is
`FLASHINFER_TRTLLM_MXFP4_MXFP8`, then `DEEPGEMM_MXFP4`, then `MARLIN`, then `BATCHED_MARLIN`.
**(source: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`)**

The first two are gated to Blackwell, so Hopper falls through to Marlin. That is now read from
source rather than inferred, one class at a time: **(source, all at v0.29.0)**

| Backend | Device check | SM90? |
|---|---|---|
| `TrtLlmMxfp4ExpertsBase` | `p.is_cuda() and p.is_device_capability_family(100) and has_flashinfer()` | no |
| `DeepGemmFP4Experts` | `is_device_capability_family(100)` or `(120)` | no |
| `MarlinExpertsBase` | `p.is_cuda() and p.has_device_capability((7, 5))` | **yes** |

Hopper is capability 9.0, so it fails both Blackwell family checks and passes Marlin's 7.5 floor.
Marlin is third in the priority list and the first one that accepts the device.

That the activations are 16 bit is also in source, not inferred from the backend's name:
`marlin_moe.py` asserts `hidden_states.dtype in [torch.float16, torch.bfloat16]`. The kernel
unpacks the FP4 weights inside itself and does the multiply in BF16. So:

> **The experts are stored at FP4 but computed at the BF16 rate of 989.5 TFLOP/s, not the FP8
> rate of 1,979.** Memory traffic stays at the FP4 rate. Only the math is slower.

This is the third place in this model where storage precision and runtime precision differ, and
it is the one that matters most.

## 1.2 The FP8 conversion that is not available

There is a way to make the experts run on FP8 tensor cores. `convert.py` accepts
`--expert-dtype fp8`, and separately vLLM has an opt-in flag `VLLM_DSV4_FP4_DEQUANT=1` that
re-encodes MXFP4 to block FP8 at load time. The conversion is bit exact, because MX scales are
powers of two and e4m3 has enough range to absorb the per-32 group scale. **(PR #53709)**

Neither is usable here, for two separate reasons.

The flag does not exist in the pinned release. PR #53709 is open, not merged, so on v0.29.0 there
is nothing to turn on. **(source)**

And it would not fit anyway. FP8 experts are twice the bytes of MXFP4, which adds 92.17 GB per
GPU and takes weights alone to 206.71 GB against 141 GB of capacity. Section 2.6 covers this.

There is a useful signal buried in that PR's benchmarks. Converting to FP8 speeds up concurrent
prefill by 2.04x but speeds up decode by only 1.13x. That gap is exactly what you would expect if
decode is memory bound, because doubling the bytes you read while making the math faster roughly
cancels out. Prefill is compute bound, so it gains. I use this in section 3.

## 1.3 Launch behaviour

The vendor recipe sets `--compilation-config '{"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}'`.
**(docs)**

So decode runs inside a captured CUDA graph and prefill does not. Launch overhead for the decode
step becomes device-side dispatch rather than CPU-side launch. With roughly 4,326 kernels per
step, the difference between those two costs is worth about 19 ms, which makes graph capture the
single largest engine-level effect on this workload. Section 3 puts numbers on it.

`mode: 3` is a torch.compile level, which means fusion decisions are delegated to the compiler.
Which specific operations fuse is **assumed** and I have not read the generated graph.

## 1.4 Collectives

I counted these by finding the actual `dist.all_reduce` and `dist.all_gather` call sites in
`model.py` rather than assuming a convention. **(checkpoint semantics, to be confirmed against
vLLM's own implementation)**

| Collective | Where | Count per step |
|---|---|---|
| `wo_b` all-reduce, in fp32 | every block that runs | 61 |
| MoE all-reduce over the fp32 accumulator | every block that runs | 61 |
| Indexer all-reduce on `index_score` | ratio-4 layers only | 30 |
| Embedding all-reduce | input | 1 |
| Logits all-gather | output | 1 |
| | | **154** |

Three things here are easy to miss. First, the usual assumption of two all-reduces per layer gives
124 and misses the indexer's third collective on half the layers. Second, and more expensive,
`RowParallelLinear.forward` calls `y = y.float()` before the all-reduce. **The most frequent
collective in the graph moves 4 bytes per element, not 2.**

Third, the count is 61 blocks and not 62. Speculation is disabled at this baseline, so the MTP
block sits in memory and never runs. It costs 1.92 GB per GPU of residency and contributes
nothing to the decode graph. Both halves of that need saying, because counting it in the
collective total bills a block that does not execute, and leaving it out of the memory total
understates the footprint by 1.92 GB.

Which NCCL algorithm gets used, ring or tree, is **assumed**. I have not read vLLM's collective
setup.

## 1.5 How the experts are actually split

The label on this deployment is TP8, but the routed experts are not width sharded. `model.py`
computes `n_local_experts = n_routed_experts // world_size` and gives each rank a contiguous
block of expert indices. Each rank owns 48 complete experts out of 384. That is expert
parallelism happening inside what is labelled tensor parallelism, and it explains why the MoE
collective is a full width all-reduce rather than a reduce-scatter. **(checkpoint semantics)**

`--enable-expert-parallel` is not in the vendor recipe, so there is no additional expert
parallel layer on top of this. **(docs)**

## 1.6 KV precision

The recipe sets `--kv-cache-dtype fp8`, so the cache is 1 byte per element. **(docs)**

This is worth flagging as an engine decision rather than a model fact. `config.json` declares
`bfloat16` and its `quantization_config` does not cover the cache. The reference implementation
allocates the cache in BF16 and the code comments say so twice: "kv could also use fp8 format,
though current implementation uses bf16". So the checkpoint supports BF16 and the deployment
chooses FP8. Both numbers appear in section 2.

---

# 2. Memory fit

## 2.1 First, a check that the shapes are right

Before any fit arithmetic, I rebuilt every weight tensor in the checkpoint from the shapes in
`model.py` and the counts in `config.json`, then compared the total against the `total_size`
published in `model.safetensors.index.json`.

```
PREDICTED  864,704,792,696 bytes   145,116 tensors
PUBLISHED  864,704,792,696 bytes   145,116 tensors
DIFFERENCE              +0 bytes         0
```

Exact on both. This is the strongest check available without running anything, because a wrong
projection dimension would have to be cancelled by another wrong dimension to the byte. The case
warns that shapes are where most submissions fail, so this is the gate everything else sits on.

Getting there surfaced one contradiction between the checkpoint and its own reference code.
`model.py` declares `gate.tid2eid` as `int32`. At int32 my total came up 9,308,160 bytes short.
That residual is exactly 3 hash layers times 129,280 vocab times 6 experts times 4 bytes, which
is precisely the int32 to int64 difference on that one tensor. **The stored tensor is int64.** I
did not assume that, I derived it from the gap.

## 2.2 Per-GPU weights are not the total divided by 8

This is the part most accountings get wrong, and it is worth being concrete about.

`model.py` uses plain `Linear` for several modules, and a plain `Linear` is replicated on every
rank rather than sharded. That includes `wq_a`, `wkv`, both compressors, and most expensively
`shared_experts`. Only `ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding`,
`ParallelHead`, and the routed expert bank actually split.

| | GB per GPU |
|---|---:|
| Replicated on every rank | 7.38 |
| Sharded (857.33 divided by 8) | 107.17 |
| **Weights per GPU** | **114.54** |
| What `total / 8` would tell you | 108.09 |
| **Understated by** | **6.46** |

The largest single replicated item is `shared_experts` at 4.10 GB per GPU on its own, because the
shared expert runs for every token and is never routed.

## 2.3 The quantization metadata is 48 GB and easy to miss

FP8 weights in this checkpoint carry a scale of shape `[out/128, in/128]`, which is negligible.
FP4 weights carry a scale of shape `[out, in/32]`, which keeps full output resolution. That makes
FP4 scales **one sixteenth** of the expert weight bytes rather than one sixteen-thousandth.

The expert bank is 773.71 GB of weights plus **48.36 GB of scale metadata**. Anyone treating FP4
scales like FP8 scales loses 48 GB from the budget without noticing.

## 2.4 The KV cache is small, and it is replicated rather than sharded

Two facts drive this. `num_key_value_heads` is 1, so there is a single latent KV of 512
dimensions shared across all 128 query heads. And `wkv` is a plain `Linear`, so every rank
computes and holds the whole latent. Attention runs local query heads against the complete cache,
so there is nothing to shard.

The cache itself is `window_size + (context / compress_ratio)` entries per layer. Only the second
term grows with context. The window term is fixed at 128 entries no matter how long the sequence
gets. Splitting those apart matters, because counting the whole cache as growing would inflate
the rate and cap concurrency far below what the hardware allows.

| At 8,192 context | fp8 (recipe) | bf16 (checkpoint) |
|---|---:|---:|
| Fixed, sliding window | 4.00 MB/seq | 8.00 MB/seq |
| Growing, compressed | 40.34 MB/seq | 80.67 MB/seq |
| **Per sequence** | **44.34 MB** | **88.67 MB** |
| **32 sequences, per rank** | **1.42 GB** | **2.84 GB** |

For comparison, an uncompressed latent cache at the same context would be 256 MB per sequence.
The compressor buys about 5.8x. Against 114.54 GB of weights, the KV cache is close to a rounding
error, which already tells you something about where the bottleneck will not be.

The compressors also keep `kv_state` and `score_state` buffers in fp32. These are sized by the
compression ratio rather than by context, and they come to 0.60 GB per GPU. Slightly
counter-intuitively the ratio-128 layers carry the larger buffer, 128 slots against 8.

## 2.5 The fit result, stated as two numbers

The case asks for the capacity lower bound and the configuration you would actually deploy to be
kept separate, so here they are separately.

**Capacity lower bound.** Weights, KV cache, and compressor state only, with no workspace and no
reserve:

```
116.56 GB of 141 GB raw capacity     fits, at 82.7%
```

Every number in that total is derived from the checkpoint. Nothing in it is a guess.

**A configuration you would actually deploy,** at the recipe's own `--gpu-memory-utilization 0.9`:

| | GB per GPU | |
|---|---:|---|
| Weights | 114.54 | derived |
| KV cache at fp8 | 1.42 | derived |
| Compressor state, fp32 | 0.60 | derived |
| Activation workspace | 4.00 | **assumed** |
| Communication buffers | 2.00 | **assumed** |
| Reserve for fragmentation | 6.00 | explicit choice |
| **Total** | **128.56** | |
| Usable at util 0.9 | 126.90 | |
| **Headroom** | **-1.66** | |

I am not going to call that a failure to fit, and I want to explain why rather than hedge. The
shortfall is 1.66 GB and it sits underneath 12 GB of numbers I guessed. The assumptions dominate
the result, so the result is not a finding about the model.

What is a finding: **116.56 GB per GPU is derived and leaves 24.44 GB for workspace,
communication buffers, reserve, and fragmentation.** Whether 24.44 GB is enough is a question
about vLLM's allocator, not a question about this checkpoint. Section 4 says how I would settle
it, and it takes one line of engine output rather than more arithmetic.

Using BF16 KV instead of FP8 adds 1.42 GB, which does not change the shape of the answer.

## 2.6 If the experts were converted to FP8

Running `convert.py --expert-dtype fp8` doubles the expert bank. That adds 92.17 GB per GPU, which
takes **weights alone** to 206.71 GB against 141 GB of capacity. It does not fit on one node, and
it is not close.

Worth stating that comparison carefully. I am putting weights against raw capacity, with no
workspace and no reserve in the total, because the weights overflow on their own. Adding my
assumed numbers would make a guess load bearing in an argument that does not need one.

The whole checkpoint in that form is about 1,602 GB against 1,128 GB of node capacity, so the
minimal shape that holds it is **two nodes at TP16**. Replicated weights stay at 7.38 GB per GPU
and the sharded part drops to about 99.7 GB, for roughly 107 GB per GPU. Every divisibility
requirement still holds at 16: 384 experts, 128 heads, 16 output groups, and 129,280 vocabulary
all divide evenly.

That shape introduces a specific cost into the decode graph. The two per-layer all-reduces,
`wo_b` and the MoE accumulator, stop being NVLink operations and become inter-node fabric
operations. At 62 layers that is 124 collectives per step crossing the slower fabric, and the
`wo_b` one is carrying fp32. Those two fabrics have to be priced separately, which section 3
flags as unfinished.

---

# 3. Top three bound hypotheses

Reproduce every figure in this section with `python3 derive/bounds.py`.

## 3.1 The predicted step, in one table

Per rank, per decode step, at the baseline. Bytes are HBM traffic, not resident footprint.

| Term | GB | share |
|---|---:|---:|
| Routed expert weights, selected only | 40.472 | 79.0% |
| Attention, compressor and indexer weights | 4.686 | 9.1% |
| Shared expert weights, every token | 4.030 | 7.9% |
| mHC, router and norms | 0.673 | 1.3% |
| KV gather for sparse attention | 0.664 | 1.3% |
| KV scan for the indexer top-k | 0.503 | 1.0% |
| `lm_head` | 0.232 | 0.5% |
| **Total** | **51.260** | |

At 4.8 TB/s that is **10.68 ms** of pure memory time.

Compute, split by the precision each operation actually runs at:

| | GFLOP | peak | time |
|---|---:|---:|---:|
| FP8 tensor core, backbone projections | 457.41 | 1,979 TF/s | 0.231 ms |
| BF16 tensor core, Marlin experts and attention core | 297.14 | 989.5 TF/s | 0.300 ms |
| FP32 CUDA core, router and mHC | 16.12 | 67 TF/s | 0.241 ms |
| **Total** | **770.67** | | **0.772 ms** |

Arithmetic intensity is **15.03 FLOP per byte** against a BF16 ridge of 206. The step is memory
bound by a factor of 14, which is not a close call.

That FP32 row deserves a second look. It is 2% of the FLOPs and 31% of the compute time, because
the router and the mHC mixing run on CUDA cores at 67 TF/s rather than on tensor cores. It still
does not matter at this batch size, but it is the row that would matter first if batch grew.

Collectives, at TP8 on NVLink, using ring cost of 2(N-1)/N times payload:

| Collective | count | payload | moved |
|---|---:|---:|---:|
| `wo_b` all-reduce, fp32 | 61 | 0.918 MB | 97.94 MB |
| MoE all-reduce, fp32 | 61 | 0.918 MB | 97.94 MB |
| Indexer all-reduce, fp32 | 30 | 0.262 MB | 13.76 MB |
| Embedding all-reduce | 1 | 0.459 MB | 0.80 MB |
| Logits all-gather | 1 | 16.55 MB | 14.48 MB |
| **Total** | **154** | | **224.93 MB** |

At 900 GB/s that is **0.25 ms** of bandwidth time.

Note the count is 154 rather than the 156 you get from 62 blocks. Speculation is disabled, so the
MTP block is resident in memory but never executes. It contributes zero collectives. Counting it
would bill a block that does not run.

Launches: roughly **4,326 kernels per step**. Under the recipe's `FULL_DECODE_ONLY` graph capture
at an assumed 0.5 microseconds of device-side dispatch each, that is **2.16 ms**. In eager mode at
an assumed 5 microseconds of CPU launch each it would be 21.6 ms.

## 3.2 The ranking

| Rank | Term | Time | Bound by |
|---|---|---:|---|
| 1 | Expert weight streaming, routed plus shared | 9.27 ms | **memory** |
| 2 | Kernel launch and dispatch, under graph capture | 2.16 ms | **latency** |
| 3 | All other HBM traffic: attention weights, KV, head | 1.41 ms | **memory** |
| | Compute, all precisions | 0.77 ms | compute |
| | Collectives, bandwidth only | 0.25 ms | communication |

Serializing memory, communication and launch gives a step floor of **13.09 ms**, which is about
**2,444 tokens per second** aggregate at batch 32.

That number can be checked. PR #53709 measured 1,642 to 1,856 tokens per second for decode at 32
concurrent requests on Hopper. My floor sits **1.40x above** the measurement. A floor above a
measurement is the only correct relationship, since a floor assumes perfect overlap and vendor
peak. A floor that came out below a real measurement would mean I had made an arithmetic error.
The 1.4x gap is where imperfect overlap, real bandwidth efficiency and routing imbalance live.

## 3.3 Where communication sits, and why it is not in the top three

The case asks for this to be argued rather than assumed, and the argument has two halves that
point in different directions.

On **bandwidth**, communication is trivial. 224.93 MB at 900 GB/s is 0.25 ms, under 2% of the
step. Even with the fp32 cast doubling the wire cost of the two most frequent collectives, the
payloads are tiny because decode moves 32 tokens, not 8,192.

On **latency**, it might not be trivial at all. There are 154 separate collectives, and a small
message all-reduce over NVLink has a floor of roughly 5 to 10 microseconds regardless of payload.
At 7 microseconds that is **1.08 ms**, which would put communication at rank 3 and push the other
HBM traffic to rank 4.

So the honest statement is a range. Communication costs somewhere between **0.25 ms and about
1.3 ms** depending on whether per-collective latency dominates payload, and I cannot settle that
from the checkpoint or from engine source. It is fourth or fifth on bandwidth alone and third at
worst. It does not lead under any reading, because rank 1 is 9.27 ms and the entire collective
budget cannot approach that.

One more reason it does not lead: CUDA graph capture amortizes the CPU side of launching these
collectives, but it does not remove the device side synchronisation between ranks. That is why
this shows up as a latency question rather than a launch question.

Under the two node TP16 shape from section 2.6 the answer changes completely. The 122 per layer
all-reduces would cross the inter-node fabric instead of NVLink, at perhaps a quarter of the
bandwidth and several times the latency. Communication would plausibly become rank 1 or 2 there.
I have not priced that shape, and section 4 says so.

---

## Experiment 1, for hypothesis 1: expert weight streaming dominates

**Hold fixed.** Model, revision, engine version, hardware, TP8, context at 8,192 tokens,
`cudagraph_mode: FULL_DECODE_ONLY`, `--kv-cache-dtype fp8`, speculation disabled.

**Vary.** Batch size across 1, 2, 4, 8, 16, 32, 64.

**Why this discriminates.** The expected number of distinct experts touched per layer is
`384 x (1 - (1 - 1/384)^(6B))`, which is strongly sublinear in batch. It is about 5.9 experts at
batch 1, 45 at batch 8, 151 at batch 32, and 243 at batch 64. Routed expert traffic follows that
curve. Every other term in the step is either flat in batch, like the dense weights and the
launch overhead, or linear in batch, like the KV gather and the collective payloads. So the three
candidate worlds produce three different shapes.

**Record.** Median per step latency in milliseconds at each batch size, from the engine's own
step timing.

**Predicted.** Step latency rises from about 5.0 ms at batch 1 to about 13.1 ms at batch 32. That
is **2.6x for a 32x increase in batch**, and per token throughput should improve roughly 12x
across that range before flattening.

**Rejection condition.** If step latency at batch 32 is **within 1.3x** of step latency at batch
1, expert streaming is not the dominant term and something batch independent is. Equally, if step
latency grows **more than 8x** across that range, the step is closer to linear in tokens than the
model predicts and the expert term is not behaving as a shared bulk read.

**What it cannot establish.** It does not prove the bytes are expert weights. Any term that
follows the same sublinear routing curve would produce the same shape, and confirming the volume
needs a profiler counter such as `dram__bytes_read.sum`. It also cannot separate bandwidth
saturation from expert load imbalance, because both make the curve bend the same way. Imbalance
specifically needs a trace, since it depends on what the tokens are.

## Experiment 2, for hypothesis 2: launch and dispatch is rank 2

**Hold fixed.** Everything in experiment 1, with batch pinned at 32 and context at 8,192.

**Vary.** `cudagraph_mode` between `FULL_DECODE_ONLY` and `NONE`.

**Why this discriminates.** Graph capture removes per kernel CPU launch cost and leaves the
kernels themselves untouched. The difference between the two runs is therefore almost entirely
launch overhead, which is the quantity in question.

**Record.** Median per step latency in both modes, and the ratio between them.

**Predicted.** About 13.1 ms captured against about 32.6 ms eager, a ratio of **2.5x**. That
follows directly from roughly 4,326 kernels at the assumed 0.5 and 5.0 microsecond per kernel
costs.

**Rejection condition.** If the eager to captured ratio is **below 1.3x**, the kernel count is far
lower than 4,326 or the per launch cost is far below 5 microseconds, and launch overhead is not
rank 2. If the ratio exceeds **4x**, my per kernel estimate is too low and launch is a larger
share of the captured step than 2.16 ms suggests, which would move it toward rank 1.

**What it cannot establish.** It conflates kernel launch with collective launch, since graph
capture amortizes both and this experiment cannot separate them. It also gives an aggregate cost
rather than a kernel count, so it cannot confirm the 4,326 figure directly. A profiler kernel
trace would give the count, and that is the cheaper way to check the softest number in this note.

## Experiment 3, for hypothesis 3: the non-expert traffic is dense weights, not KV

**Hold fixed.** Batch at 32, TP8, engine, graph mode, KV dtype.

**Vary.** Context length across 2,048, 8,192 and 32,768 cached tokens.

**Why this discriminates.** This is the sharpest prediction in the note, and it comes from a
detail that is easy to miss. On the 30 ratio-4 layers the indexer selects `index_topk` positions,
which is 1,024. Once context passes 4,096 tokens there are more than 1,024 compressed slots, so
**the attention gather stops growing entirely.** It is capped by `index_topk`, not by context.
Only two things still scale with context: the 31 ratio-128 layers, which gather `context/128`
slots, and the indexer's own scan, which must read `context/4` entries to choose its top-k.

So quadrupling context from 8,192 to 32,768 adds roughly 1.5 GB to a 51.26 GB step. That is under
3%. If KV traffic were the dominant non-expert term, or if the gather were not actually sparse,
quadrupling context would move the step substantially.

**Record.** Median per step latency at each context length, plus the reported KV cache block usage
to confirm the context actually changed.

**Predicted.** Step latency **nearly flat**, within about 5% across a 16x range of context.

**Rejection condition.** If step latency grows by **more than 15%** from 2,048 to 32,768 tokens,
then either the engine is reading the full KV cache rather than a sparse gather, or the indexer
scan is more expensive than modeled. Either finding rejects this hypothesis's composition and
promotes KV traffic in the ranking. A **linear** growth in context would mean the sparse path is
not being taken at all, which would be the most interesting failure of the three.

**What it cannot establish.** It does not separate the attention weights from the mHC, router and
norm traffic inside the same bucket, since none of those depend on context. Splitting that bucket
needs per kernel timing. It also says nothing about prefill, where the gather is genuinely
quadratic in the windowed layers and this whole analysis does not apply.

## A note on what these three share

All three experiments vary one engine-level knob and record one number that the engine already
reports. None of them needs a profiler, a trace, or a code change. That is deliberate. The
cheapest experiment that can reject a hypothesis is worth more than a better instrumented one that
takes a week to set up, and if all three run as predicted the model in section 3.1 is good enough
to plan against. If any one of them fails its rejection condition, the ranking in 3.2 is wrong and
I would rather find that out in an afternoon.
# 4. What I would do next, and why

Ranked by how much they would change the conclusions.

**Settle the workspace number.** The 24.44 GB of headroom is derived but what fills it is
guessed. vLLM prints its actual KV cache allocation and memory profile at startup. One line of
that output replaces three assumed numbers and turns section 2.5 from a hedge into a result.
This is far cheaper than more arithmetic.

**Count the kernels.** The 4,326 figure is the softest number in this note and it carries rank 2
in the ranking. A single profiler kernel trace replaces the estimate with a count, and experiment
2 only measures the aggregate cost rather than confirming the count itself.

**Settle whether collectives are latency bound.** Section 3.3 gives communication as a range from
0.25 ms to about 1.3 ms, and the two ends sit in different positions in the ranking. Recording
NCCL time separately during the batch sweep in experiment 1 would settle it, since the payload
changes 32x while the op count stays fixed. Flat means latency bound, scaling means bandwidth
bound.

**Price the two fabrics separately.** The two-node TP16 shape in section 2.6 is derived, but I
have not put numbers on what moving 122 per layer all-reduces from NVLink to the inter-node
fabric actually costs. That is the difference between naming a shape and recommending one, and it
is the one place where communication could plausibly become rank 1.

**Check the expert imbalance assumption.** The 18.9 experts per rank figure assumes uniform
routing. Real routing is not uniform, and imbalance makes the MoE all-reduce wait on the slowest
rank. This is the one number in this note that genuinely cannot be derived from the checkpoint,
because it depends on what the tokens are. It needs a trace, and I should say so rather than
model it.

## What I got wrong along the way, since it is worth recording

I first read `compress_ratios` as having 61 entries and concluded that layer 60 had no
compressor, which contradicted the tensor census showing 61 compressors. The list actually has 62
entries, because it covers `n_layers + n_mtp_layers`, and the single zero is the MTP block. The
census was right and my reading was wrong. I mention it because the same mistake on a list with
no cross-check would have silently produced a layer schedule that was wrong in one position while
still summing correctly.
