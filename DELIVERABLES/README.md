# Deliverables

Submission for the GitM execution-graph case: predict how **DeepSeek-V4-Pro** executes one decode
step, derived from the checkpoint's own files, before ever seeing it run.

| File | What it is |
|---|---|
| [`deepseek-v4-pro.yaml`](deepseek-v4-pro.yaml) | **D1.** Model spec in the planner schema. Intended path in `GitM-Labs/runtime` is `gitm/planner/models/deepseek-v4-pro.yaml`. |
| [`DESIGN-NOTE.md`](DESIGN-NOTE.md) | **D2.** Engine assumptions, memory fit, and three ranked bound hypotheses with falsifiable experiments. |

## What was worked from

- Checkpoint `deepseek-ai/DeepSeek-V4-Pro` at revision `b5968e9190ef611bbf34a7229255be88a0e937c1`
- Engine vLLM `v0.29.0`
- Hardware 8 x H200 SXM, NVLink, TP8
- Baseline: 32 sequences, 8,192 cached tokens each, 1 new token per sequence per step, text only,
  speculation disabled

No GPU, no traces, nothing run.

## Reproducing the arithmetic

Everything in D2 is computed from the metadata in `context/checkpoint/`, which is the checkpoint's
`config.json`, `model.safetensors.index.json`, and `inference/*.py`. No GPU, no network, and none
of the actual weights are needed.

```bash
python3 derive/weights.py          # weight accounting vs the index's published total_size
python3 derive/fit.py              # per-GPU memory fit at 8xH200 / TP8
python3 derive/bounds.py           # per-step cost model and the ranking
python3 tests/test_derivations.py  # 39 checks, no dependencies
```

The test suite covers three things: the numbers D2 quotes, the arguments behind them, and D1
against `config.json` field by field. The second group matters because a number can stay correct
while its justification quietly rots. For example,
`test_fp4_expert_storage_is_forced_not_merely_labelled` asserts that the FP8 reading of the expert
bank exceeds the checkpoint's `total_size` on its own, which is what makes FP4 a derivation rather
than a label copied from a config field.

## The one number worth checking first

```
PREDICTED  864,704,792,696 bytes   145,116 tensors
PUBLISHED  864,704,792,696 bytes   145,116 tensors
DIFFERENCE              +0 bytes         0
```

Byte exact and tensor-count exact against the index. Every projection dimension, per-layer-kind KV
rate and expert bank size in both deliverables sits on top of this, and a wrong shape would have to
be cancelled by another wrong shape to the byte.

## Supporting material, outside this folder

- `docs/DESIGN-NOTE.md` is the long working version of D2, with the full reasoning at roughly three
  times the length. D2 here is the submission; that one is the appendix.
- `.agent/context.md` is the running log: every finding, decision, correction and open question in
  the order they happened.
- `derive/` and `tests/` are the arithmetic and its checks.
