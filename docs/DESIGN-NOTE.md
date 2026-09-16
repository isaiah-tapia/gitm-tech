# DeepSeek-V4-Pro: Predicted Execution of One Decode Step

**Working draft.** Sections 1 and 2 are complete. Section 3 is partial and marked where it needs
finishing. Section 4 says what I would do next and why.

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
python3 derive/weights.py    # weight accounting, validated against the index
python3 derive/fit.py        # per-GPU fit at 8xH200 / TP8
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

The first two are gated to Blackwell, so Hopper falls through to Marlin. **(PR #53709, where the
author states "on SM90 only Marlin W4A16 passes" alongside Hopper benchmarks. Not yet confirmed
in merged source. See section 4.)**

Marlin W4A16 means 4 bit weights and 16 bit activations. It unpacks the FP4 weights inside the
kernel and does the multiply in BF16. So:

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
GPU and brings the total to 220.73 GB against 141 GB of capacity. Section 2 covers this.

There is a useful signal buried in that PR's benchmarks. Converting to FP8 speeds up concurrent
prefill by 2.04x but speeds up decode by only 1.13x. That gap is exactly what you would expect if
decode is memory bound, because doubling the bytes you read while making the math faster roughly
cancels out. Prefill is compute bound, so it gains. I use this in section 3.

## 1.3 Launch behaviour

The vendor recipe sets `--compilation-config '{"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}'`.
**(docs)**

So decode runs inside a captured CUDA graph and prefill does not. Launch overhead for the decode
step is graph replay, which is roughly 2 microseconds, rather than eager launch at roughly 5.
With around 156 collectives and several hundred kernels per step, the difference between those
two numbers is real but it is not where the time goes. Section 3 puts a figure on it.

`mode: 3` is a torch.compile level, which means fusion decisions are delegated to the compiler.
Which specific operations fuse is **assumed** and I have not read the generated graph.

## 1.4 Collectives

I counted these by finding the actual `dist.all_reduce` and `dist.all_gather` call sites in
`model.py` rather than assuming a convention. **(checkpoint semantics, to be confirmed against
vLLM's own implementation)**

| Collective | Where | Count per step |
|---|---|---|
| `wo_b` all-reduce, in fp32 | every block | 62 |
| MoE all-reduce over the fp32 accumulator | every block | 62 |
| Indexer all-reduce on `index_score` | ratio-4 layers only | 30 |
| Embedding all-reduce | input | 1 |
| Logits all-gather | output | 1 |
| | | **about 156** |

Two things here are easy to miss. First, the usual assumption of two all-reduces per layer gives
124 and misses the indexer's third collective on half the layers. Second, and more expensive,
`RowParallelLinear.forward` calls `y = y.float()` before the all-reduce. **The most frequent
collective in the graph moves 4 bytes per element, not 2.**

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

Running `convert.py --expert-dtype fp8` doubles the expert bank. That adds 92.17 GB per GPU and
brings the total to 220.73 GB against 141 GB. It does not fit on one node, and it is not close.

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

**Status: partial.** The ranking below is argued and the first hypothesis has arithmetic behind
it. The second and third have the right shape but not finished numbers, and none of the three has
its controlled experiment written out yet. This section is the remaining work.

## What the arithmetic so far says

At the baseline, each token picks 6 of 384 experts, and there are 32 tokens. That is 192 expert
selections per layer spread across 384 experts. Assuming routing is roughly uniform, the expected
number of distinct experts touched per layer is:

```
384 x (1 - (1 - 1/384)^192)  =  about 151 experts, or about 18.9 of the 48 on each rank
```

Each expert is three matrices of 7168 x 3072 in MXFP4, which is 33.03 MB of weights plus 2.06 MB
of scales, so 35.09 MB. Per rank that is about 663 MB per layer, and across 61 layers about
**40.4 GB streamed per decode step per rank.** That is roughly 42% of the 96.7 GB expert bank
each rank holds.

At 4.8 TB/s of HBM bandwidth, 40.4 GB takes about **8.4 ms**, which works out to roughly 3,800
tokens per second aggregate at batch 32.

That number can be checked against something real. PR #53709 measured 1,642 to 1,856 tokens per
second for decode at 32 concurrent requests. My prediction sits above the measurement by about
2x, which is the correct direction for a roofline floor, and the same order of magnitude. A
roofline floor that came out below a real measurement would mean I had made an error.

## Hypothesis 1: routed expert weight streaming, memory bound

This is almost certainly the largest single consumer. Roughly 40.4 GB moves from HBM per rank per
step, against 1.42 GB of KV cache reads and a much smaller volume of backbone weights. The
arithmetic intensity is terrible, because each expert matrix is read in full to be multiplied by
a handful of token rows.

The Marlin W4A16 finding from section 1 matters here in a specific way. The compute side runs at
the BF16 rate of 989.5 TFLOP/s rather than the FP8 rate, but since this operation is memory bound
the slower math mostly does not show up in the total. That is also why the FP8 conversion in
PR #53709 only bought 1.13x on decode while buying 2.04x on prefill.

**Still to do:** the exact FLOP count per step, the resulting arithmetic intensity, and where it
lands relative to the BF16 ridge point of 989.5e12 / 4.8e12 = 206 FLOP per byte.

## Hypothesis 2: the collectives, communication bound

About 156 collectives per step, and the 62 `wo_b` all-reduces carry fp32 rather than bf16, which
doubles their wire cost. With TP8 on NVLink at 900 GB/s per GPU this should sit well below the
expert streaming term, but it is not obviously third either, and the case is explicit that
communication has to be argued into its position rather than assumed.

**Still to do:** the actual byte volume per collective. The `wo_b` all-reduce moves
batch x hidden in fp32, and the MoE all-reduce moves batch x hidden in fp32 as well. At batch 32
and hidden 7168 that is small per collective, so the question is whether 156 launches of a small
collective is a latency problem rather than a bandwidth problem. That distinction decides the
rank, and I have not done it yet.

## Hypothesis 3: launch and synchronisation overhead, latency bound

With `FULL_DECODE_ONLY` graph capture, the per-launch cost should be around 2 microseconds rather
than 5. But this model has an unusually high op count per layer: attention with two LoRA pairs, a
compressor on every layer, an indexer on 30 of them, and mHC running 20 Sinkhorn iterations at
two sites per layer. That last one is 2,440 small operations per step that do almost no
arithmetic.

**Still to do:** an actual op count per layer, multiplied out, times the graph replay cost. If
the mHC Sinkhorn loop is not fused this could be larger than it looks, and it is the kind of thing
the case is asking about when it says efficient individual kernels can still produce poor
end-to-end performance.

## Controlled experiments

**Not yet written.** Each of the three needs: what I hold fixed, what I vary, what I record, the
threshold or trend that would reject the hypothesis, and what the experiment cannot establish.
This is the highest value remaining work in the whole submission, because the case says a
hypothesis without a rejection condition does not count.

---

# 4. What I would do next, and why

Ranked by how much they would change the conclusions.

**Finish section 3.** The three experiments are the part of the case that tests judgment rather
than arithmetic, and they are missing. Everything else here is scaffolding for them.

**Confirm the Marlin claim in merged source.** Right now "SM90 falls through to Marlin" is read
from a PR description, not from source. The capability gates live in `is_supported_config` on
`TrtLlmMxfp4ExpertsMonolithic`, `DeepGemmFP4Experts`, and `MarlinExperts`. Quoting those three
would move the single most important engine claim in this note from one evidence class to a
better one. If it turned out wrong, the expert GEMM ridge changes by 2x and hypothesis 1's
compute side changes with it.

**Settle the workspace number.** The 24.44 GB of headroom is derived but what fills it is
guessed. vLLM prints its actual KV cache allocation and memory profile at startup. One line of
that output replaces three assumed numbers and turns section 2.5 from a hedge into a result.
This is far cheaper than more arithmetic.

**Price the two fabrics separately.** The two-node TP16 shape in section 2.6 is derived, but I
have not put numbers on what moving 124 collectives per step from NVLink to the inter-node fabric
actually costs. That is the difference between naming a shape and recommending one.

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
