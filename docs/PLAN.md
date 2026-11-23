# Plan: fine-tuning DSpark for coding agents on Inkling 2 Mini

## Goal

Make Inkling 2 Mini's bundled DSpark drafter better at **coding and coding-agent
traffic**, measured locally on Apple Silicon (MLX):

- **Primary metric:** mean accepted tokens per verify round (τ) and per-position
  acceptance, on held-out coding-agent prompts at temperature 0.0 and 0.6.
- **Product metric:** decode tok/s and end-to-end tok/s vs. the stock drafter and
  vs. no speculation. TTFT is reported but shouldn't change.
- **Invariant:** speculation stays lossless. At temperature 0, greedy output with any
  drafter must equal greedy output without one (modulo the verify-shape caveat in
  [dspark.md](dspark.md)).

## Constraints and scaling knobs

Local box: M1 Max (24-core GPU), 64 GB unified memory (MLX recommended working set
55.7 GB), ~514 GB free disk, ~12 MB/s download.

Every stage takes its caps from a config, so the same pipeline runs small on the Mac
and larger on a rented GPU with blob storage later.

| Knob | Local small run | Scale-out (GPU + blob) |
|---|---|---|
| max context per training window | 4K–8K (set by #survey) | 16K–32K |
| training traces | ~1–3K | 10K+ |
| teacher top-k logits | 32 | 32–64 |
| feature storage | local disk, BF16 | object store, BF16/FP8 |
| Stage 1 runtime | mlx-vlm on the Mac | GPU runtime (needs a feature-parity check) |

Memory isn't the binding constraint at long context. Target KV is ~32 KB/token
because only 8 of 48 visits are global. **Disk is**: cached drafter features cost
18 KB/token, so 1K traces × 25K tokens ≈ 460 GB. **Time is too**: Stage 1 is one
target prefill per token. Context-length scaling gets measured in Phase 0, not assumed.

## Phases

Each phase is a GitHub issue. Work lands through PRs that reference the issue.

### Phase 0: Runtime and checkpoint verification
- Load the target + bundled DSpark in mlx-vlm 0.8.1 and record peak memory.
- Greedy parity: drafter vs. no drafter, token-for-token, on a few coding prompts.
- Context-scaling probe: prefill tok/s with feature capture, peak memory, and decode
  tok/s at 2K / 4K / 8K / 16K / 32K tokens of context.

### Phase 1: Benchmark harness and stock baseline
- `specdec bench`: prompts → JSONL of TTFT, prefill tok/s, decode tok/s, E2E tok/s,
  per-round accept lengths, per-position acceptance, peak memory.
- Prompt suites:
  - (a) short single-turn coding prompts
  - (b) agent prompts cut from **held-out** trajectories at an assistant turn, bucketed by context length
- Conditions: no-spec vs. stock DSpark × T ∈ {0.0, 0.6} × fixed seeds, ≥3 repeats,
  report median.

### Phase 2: Dataset
- Source: `nvidia/Open-SWE-Traces` v2.0 (5 agent scaffolds × several teacher models,
  with `thinking-low` / `thinking-medium` variants).
- Render with Inkling's own chat template (`specdec.data.render`). Assistant mask
  = loss mask.
- Select by length (survey results). Use whole traces ≤ cap, plus prefix-truncated
  windows of longer traces.
- Filter degenerate loops: repeated n-grams in reasoning, repeated identical tool
  calls, truncated/empty turns.
- **Split by repository** so held-out prompts come from repos never seen in training.

### Phase 3: Stage 1, teacher cache
- Chunked target prefill (`prefill_step_size` 256) with
  `capture_layer_ids=[5,23,41,48]`.
- Store per window: token ids, loss mask, features `[T, 4, 2304]` BF16, teacher
  top-k (ids int32 + logprobs fp16) on assistant positions. Use framework-agnostic
  safetensors shards + manifest.
- The target is unloaded before Stage 2.

### Phase 4: Stage 2, DSpark trainer (MLX)
- Training forward = inference math, batched over many anchors per window. Context
  K/V is computed once per window. Blocks get a structured mask: context `< a`, the
  window limit on local layers, bidirectional within the block. The Markov head is
  teacher-forced on ground-truth tokens.
- Loss per block slot k = 1..7, weighted βᵏ⁻¹:
  forward KL(teacher top-k ‖ draft) + λ·CE(label).
- Parity test first: training forward == `draft_block` logits for the stock weights.
- Train all 318M drafter params (FP32 master weights, AdamW, ~5 GB of state). The
  target embedding and LM head stay frozen.

### Phase 5: Evaluation
- Fine-tuned vs. stock on the held-out suites from Phase 1, same seeds and settings.
- Ablations, as time allows: off-policy (dataset text) vs. on-policy (Inkling's own
  generations) training data, β, λ, context cap.

### Phase 6: Scale-out
- Port Stage 1 (and optionally Stage 2) to a rented GPU. Keep the shard format.
- Check MLX vs. GPU feature parity before mixing data sources.
- Before renting, estimate GPU-hours from the Phase 0/3 throughput measurements.

## Experiment tracking

- `configs/*.yaml` define every run. `runs/<date>-<name>/` holds `config.yaml`,
  `env.json` (git SHA, mlx/mlx-vlm versions, checkpoint revision, prompt-set hash),
  `metrics.jsonl`, `summary.json`.
- Small run outputs (configs, summaries, metrics) are committed. Large artifacts
  (features, checkpoints) stay out of git (`data/`, `runs/*/cache`,
  `runs/*/checkpoints`).
- `RESULTS.md` holds the headline comparison table, updated via PR.

## Risks and open questions

- **Off-policy data.** The traces were written by Qwen/DeepSeek/MiniMax teachers,
  not Inkling. Teacher-forcing them gives the right *labels* (Inkling's own
  distribution), but on contexts Inkling wouldn't produce. On-policy generation is
  the ablation.
- **Reasoning format.** Inkling's template keeps `reasoning_content` for every turn.
  Check that this matches what Inkling sees at inference in real agent loops.
- **Feature parity across runtimes** (see [dspark.md](dspark.md)).
- **Greedy parity at verify shapes.** Chunk size and verify shape can flip
  near-ties. Parity checks should report mismatches, not assume zero.
