# .agent/context.md — working context

Living file. Claude reads this at the start of a session and updates it as work proceeds.
Everything below is either **sourced** (traceable to a file in `context/` or a public primary
source) or **assumed** (marked as such). Never promote an assumption to a fact without a citation.

---

## 1. What this repo is

A take-home / evaluation case for **GitM (Git Machine)**, a GPU execution runtime company.
Repo root: `/Users/isaiahtapia/gitm-tech`. Currently contains only source material — no code yet.

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

**Time cap:** 3 hours. Scope is cut to fit; stretch items marked optional.

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

## 6. Current state

**Status:** context ingested, nothing built yet.

Not yet started:
- [ ] Locate the official DeepSeek V4 Pro checkpoint; record exact repo + revision.
- [ ] Pull `config.json`, `model.safetensors.index.json`, `trust_remote_code` modeling file.
- [ ] Check whether a vendor serving recipe exists (it supersedes the 8×H200/TP8/FP8 default).
- [ ] Pin an engine version (vLLM tag or commit) and record it.
- [ ] Derive shapes: projection dims, per-layer-kind KV byte rates, expert bank sizes. Twice.
- [ ] Memory fit arithmetic at the baseline workload.
- [ ] Choose TP/EP/PP layout, enumerate comm ops in the decode graph.
- [ ] Rank top-3 bounds + design one falsifiable experiment each.
- [ ] Write D1 (YAML) and D2 (design note).

## 7. Open questions / to resolve

- Does a checkpoint literally named "DeepSeek V4 Pro" exist publicly as of the work date? If the
  name doesn't resolve, that's a finding to state explicitly, not paper over — and the fallback
  (nearest official DeepSeek checkpoint, named and justified) must be declared up front.
- The reference files `gitm/planner/models/mimo-v2.5.yaml` and `docs/glm-5.2/DESIGN-NOTE.md` are
  in `github.com/GitM-Labs/runtime` — need to confirm they're reachable to match the schema.
  If not reachable, the YAML schema has to be inferred and that inference stated.

## 8. Decisions log

_(append: date — decision — why. Nothing yet.)_

## 9. Conventions for this repo

- Deliverables live at the repo root or a clearly named folder; source material stays in `context/`.
- Every number in the design note traces to a file + field, or is labeled an assumption inline.
- Checkpoint names (tensors, modules, quant labels) copied **verbatim**, never normalized.
- Engine claims carry their tag: *read from engine source* / *read from docs* / *assumed*.
