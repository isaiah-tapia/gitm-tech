# Git Machine — Execution Graph Case: DeepSeek V4 Pro

This case evaluates how you reason about model execution on GPUs. Cap your time at three hours; the scope below is cut to fit that, and stretch items are marked optional. No GPU, no traces, nothing to run. Everything you need is public.

## The task

Predict how DeepSeek V4 Pro executes one decode step, from the checkpoint's own files, before ever seeing it run.

Primary sources are the model repo on Hugging Face: `config.json`, `model.safetensors.index.json`, and the modeling implementation the checkpoint actually loads (the `trust_remote_code` path where present). Vendor serving recipes, where published, are fair sources for the deployment shape. Serving-engine source and official documentation are also allowed, with the version pinned (e.g. a vLLM tag or commit), because the checkpoint alone does not determine kernel fusion, launch behavior, collective implementation, or runtime precision. Keep the two evidence classes separate throughout: what the checkpoint establishes, and what depends on the engine. Do not use third-party writeups of the architecture. The point is what you can derive, not what you can find.

## Baseline workload

All fit and bound claims are evaluated at one fixed operating point:

- 32 active sequences, decode phase
- 8,192 cached tokens per sequence
- one new token per sequence per step
- text-only, speculation disabled
- hardware and precision per the assignment parameters below

State any additional assumption the config leaves open; every assumption stated is fine, every assumption smuggled is not.

## Deliverables

Two artifacts.

**1. A model spec YAML in the planner schema.** Follow the format of [`gitm/planner/models/mimo-v2.5.yaml`](https://github.com/GitM-Labs/runtime/blob/main/gitm/planner/models/mimo-v2.5.yaml) in the GitM-Labs/runtime repo, including its conventions: layer schedules written out when they do not tile, per-op precision noted, and the checkpoint's own names kept verbatim (quantization labels, tensor and module names as they appear in the files, not renamed into nicer vocabulary). Where the config contradicts what the model family usually does, capture it in a comment; that is signal, not noise.

**2. A short design note** (markdown, one to two pages) with the sections below. [`docs/glm-5.2/DESIGN-NOTE.md`](https://github.com/GitM-Labs/runtime/blob/main/docs/glm-5.2/DESIGN-NOTE.md) in the same repo is a completed example at fuller scope than asked here; match its rigor, not its length.

- **Engine assumptions.** What the serving engine you pinned determines beyond the checkpoint: which ops fuse, what runs under CUDA graphs, which attention backend and collective implementation apply, where runtime precision differs from storage precision. Mark each item as read from engine source, read from docs, or assumed.
- **Memory fit.** Per-GPU accounting at the baseline workload: weights, quantization metadata, KV or recurrent state, activation workspace, communication buffers, and an explicit reserve. Distinguish the capacity lower bound from a configuration you would actually deploy. If the model does not fit the assigned hardware, derive the minimal shape that holds it and name the communication ops that shape introduces into the decode graph.
- **Top-3 bound hypotheses, ranked.** For the three largest time consumers in your predicted decode step: what they are, and whether each is compute-, memory-, or communication-bound at the baseline. If your layout includes tensor, expert, or pipeline parallelism, derive the communication cost from that layout and defend where it sits in the ranking. For each hypothesis, one controlled experiment: what you would hold fixed, what you would vary, the observable you would record, and the threshold or trend that would reject the hypothesis. Say what the experiment cannot establish.

## Scoring

Correctness beats coverage: an incomplete section with documented next steps ("here is what I would do with two more hours and why") scores better than a complete section built on shaky arithmetic.

Wrong-but-specific beats vague: a precise claim that turns out false can be tested and corrected, "it depends" cannot. Explicitly identified uncertainty scores just as well. "The config establishes X; engine source is needed to determine Y" is exactly the judgment the job requires. What fails is arithmetic that does not follow from the files, family assumptions presented as facts, and hypotheses without a rejection condition.

Shapes are where most submissions fail. Projection dims, per-layer-kind KV byte rates, and expert bank sizes follow from the config by arithmetic. Check them twice.

## Assignment parameters

Hardware: one 8×H200 SXM node over NVLink, TP8, FP8 weights and KV where the checkpoint supports it. A published vendor serving recipe for this model supersedes this default; cite it. Find the official checkpoint yourself; state the exact repo and revision you worked from and the engine version you pinned, since your numbers are only checkable against those. Alternatives to any default here are allowed with a stated justification.

## DeepSeek V4 Pro

This is the large sibling, and the fit section carries the weight: show whether the weights hold on one 8×H200 node at the checkpoint's precision, and if not, derive the minimal multi-node shape. Choose your TP/EP/PP layout explicitly, derive the communication ops it introduces into the decode graph, price them against NVLink and the inter-node fabric separately, and defend where communication sits in your ranking rather than assuming it leads or trails.
