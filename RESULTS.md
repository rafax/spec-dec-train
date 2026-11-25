# Results

Hardware/runtime for all rows: M1 Max (24-core GPU), 64 GB, macOS 26.5, mlx 0.32.4,
mlx-vlm 0.8.1, `thinkingmachines/Inkling-2-Mini` @ `7a67ae2`, native NVFP4/MXFP8 target,
BF16 DSpark. Run artifacts live under [`runs/`](runs/).

**Where things stand (stock DSpark drafter, no fine-tuning yet):**

| Decode speedup vs no-spec | Stock mlx-vlm | With `specdec.fast_verify` (#11) |
|---|---:|---:|
| Short coding prompts, T=0 (block 8 / best block) | 1.04× | 1.54× / 1.68× (block 4) |
| Short coding prompts, T=0.6 | 0.85× | 1.34× / 1.54× (block 4) |
| Real agent turns, 3K–33K context, T=0 (median of 9) | 1.46× | **2.03×** (1.37–2.55×) |

**Clean reruns.** The first runs (no `-rerun` suffix) shared the Mac with an idle Unsloth
Studio `llama-server` that held ~12 GB. All timing runs were repeated with it unloaded
(`-rerun` dirs, `other_model_servers: []`, 2.9 GB swap). Acceptance was bit-identical and
decode speeds were within ~1–2%, so the background server did not affect the numbers.
`env.json` now records any other model servers and warns when one is present.

## Stock DSpark vs no speculation (short coding prompts)

6 single-turn coding prompts, 256 generated tokens, reasoning effort `low`, 2 repeats,
conditions interleaved per prompt. Speedups are paired per prompt/repeat (median).
Run: [`runs/2026-11-24-p1-block-sweep-rerun`](runs/2026-11-24-p1-block-sweep-rerun/summary.json).

| T | Condition | Decode tok/s | τ (tokens / target forward) | Decode speedup |
|---|---|---:|---:|---:|
| 0.0 | no-spec | 18.5 | n/a | 1.00 |
| 0.0 | DSpark, block 4 | 17.1 | 2.96 | 0.93 |
| 0.0 | DSpark, block 6 | 17.2 | 3.30 | 0.93 |
| 0.0 | DSpark, block 8 (default) | 18.7 | 3.88 | **1.01** |
| 0.6 | no-spec | 18.3 | n/a | 1.00 |
| 0.6 | DSpark, block 4 | 15.4 | 2.70 | 0.84 |
| 0.6 | DSpark, block 6 | 16.7 | 3.25 | 0.91 |
| 0.6 | DSpark, block 8 (default) | 15.4 | 3.24 | **0.84** |

Per-position acceptance P(accepted ≥ k), block 8:

- T=0: 0.78 / 0.61 / 0.48 / 0.37 / 0.30 / 0.23 / 0.15
- T=0.6: 0.73 / 0.52 / 0.38 / 0.28 / 0.20 / 0.14 / 0.12

Single-turn prompts are the drafter's worst case. On real agent turns it accepts far more
(see below).

## Why verify is so expensive here: a BF16 small-batch matmul problem

Verify-cost probe at 1K context ([`runs/2026-11-24-p0-verify-cost-rerun`](runs/2026-11-24-p0-verify-cost-rerun/summary.json)):
plain decode 55 ms/token in real generation; target verify **56 ms for 1 token, 122 ms for
2, 137 ms for 4, 174 ms for 8**; one DSpark draft block 11–22 ms. Break-even is τ ≈ 3.1–3.9
at block 8.

The jump from 1 to 2 tokens is not the MoE. Component profile
([`runs/2026-11-24-p0-verify-profile`](runs/2026-11-24-p0-verify-profile/summary.json),
`scripts/profile_verify.py`, which stubs one component at a time):

| Verify tokens | Full | Attention | MoE | Everything else |
|---:|---:|---:|---:|---:|
| 1 | 52.5 ms | 17.7 ms | 22.9 ms | 12.8 ms |
| 2 | 122.0 ms | **60.9 ms** | 36.7 ms | 19.7 ms |
| 8 | 174.0 ms | 57.6 ms | 74.3 ms | 20.1 ms |

- Attention jumps by a fixed ~40 ms at 2 tokens, then stays flat. The fused SDPA kernel
  itself is fine (2.2 → 2.9 ms over all 48 visits).
- The cause is the **BF16 projections**: q/k/v/gate/o are unquantized, ~2.5 GB per token
  over 48 layer visits. MLX runs 1 row as a GEMV at ~320 GB/s. At **2–8 rows it switches to
  a GEMM that reaches only 57–81 GB/s**:

  | Rows | `x @ W.T` (stock) | Rows as separate GEMVs |
  |---:|---:|---:|
  | 1 | 7.8 ms (319 GB/s) | 7.8 ms |
  | 2 | 30.7 ms (81 GB/s) | 15.2 ms |
  | 8 | 44.1 ms (57 GB/s) | 34.0 ms |

  The same applies to the 926 MB BF16 `lm_head` and to the BF16 drafter, whose
  8-token block reads ~1.5 GB: ~5 ms at full bandwidth, 20 ms measured.
- MoE cost grows the way it should (more distinct experts). mlx-vlm's grouped verify kernel
  is already the fastest of the three MoE paths (96 ms vs 128–138 ms at 8 tokens, MoE alone).
- A crude workaround (split 2–8-row BF16 `Linear` calls into per-row GEMVs) cut verify to
  82 ms at 2 tokens, 109 ms at 4, 163 ms at 8. The kernels below replace it.
- This is specific to older GPUs: mlx-vlm ships small-batch dense kernels, but gates them on
  GPU generation ≥ 15 (M3 and later; this M1 Max is `applegpu_g13s`), and Inkling 2's model
  code does not call them.

## Fast verify kernels (#11)

`specdec.fast_verify` computes `x @ W.T` for 2–8 rows with two Metal kernels
(`mx.fast.metal_kernel`) that read each weight element once and accumulate in FP32:

- **2–4 rows:** multi-row GEMV. A simdgroup owns 4 output rows, and each lane streams a
  4-vector of every row per step for all input rows.
- **5–8 rows:** 8×8 simdgroup-matrix MMA. Rows are padded to 8, and K is split over 8 simdgroups.

`--fast-verify` (scripts) or `+fast` (bench conditions) patches `nn.Linear` for eligible
calls: no bias, inference mode, 2–8 rows, BF16/FP16/FP32. That covers the target's
attention projections, the drafter and the drafter's use of the BF16 head. It also
replaces Inkling's verify head: stock multiplies FP32 activations by a cached **FP32 copy**
of the 201K×2304 head (1.85 GB per call). The kernel reads the BF16 head and promotes it
in-register instead, which is exact and reads half the bytes. One-row decode keeps MLX's
GEMV, so no-spec numbers do not change.

Microbenchmark ([`runs/2026-11-25-p1-small-matmul`](runs/2026-11-25-p1-small-matmul/summary.json),
`scripts/bench_small_matmul.py`, real shapes, distinct weights per layer):

| ms (rows = tokens) | Stock, 1 row | Stock, 2 / 4 / 8 rows | Fast, 2 / 4 / 8 rows |
|---|---:|---:|---:|
| Attention projections, 48 visits (2.4 GiB) | 9.3 | 33.5 / 33.8 / 34.0 | 11.3 / 11.3 / 14.8 |
| Verify head (stock: FP32 copy; fast: BF16) | 5.6 | 11.4 / 11.5 / 11.8 | 3.0 / 3.8 / 4.9 |
| One drafter block, incl. its head (1.3 GiB) | 4.5 | 17.4 / 16.0 / 16.6 | 4.8 / 4.7 / 6.7 |

Verify and draft cost at 1K context, the same probe as above
([`runs/2026-11-25-p1-verify-cost-fast`](runs/2026-11-25-p1-verify-cost-fast/summary.json)):

| Verify tokens | Stock verify | Fast verify | Stock draft | Fast draft |
|---:|---:|---:|---:|---:|
| 1 | 56.0 ms | 56.5 ms | n/a | n/a |
| 2 | 121.7 ms | **72.5 ms** | 11.2 ms | 8.3 ms |
| 4 | 137.4 ms | **95.1 ms** | 19.8 ms | 9.8 ms |
| 8 | 174.2 ms | **132.4 ms** | 21.9 ms | 12.9 ms |

End to end on the 6 short coding prompts, interleaved in one process, with paired speedups
([`runs/2026-11-25-p1-bench-fast`](runs/2026-11-25-p1-bench-fast/summary.json)):

| T | Condition | Decode tok/s | τ | Speedup (median, min–max) |
|---|---|---:|---:|---:|
| 0.0 | no-spec | 18.2 | n/a | 1.00 |
| 0.0 | DSpark (stock) | 18.9 | 3.88 | 1.04 (0.90–1.22) |
| 0.0 | DSpark + fast | 28.0 | 3.82 | 1.54 (1.34–1.93) |
| 0.0 | DSpark block 6 + fast | 29.0 | 3.44 | 1.61 (1.43–1.97) |
| 0.0 | DSpark block 4 + fast | 30.2 | 2.86 | **1.68** (1.47–1.95) |
| 0.6 | no-spec | 18.2 | n/a | 1.00 |
| 0.6 | DSpark (stock) | 15.5 | 3.24 | 0.85 (0.74–1.19) |
| 0.6 | DSpark + fast | 24.2 | 3.33 | 1.34 (1.13–1.76) |
| 0.6 | DSpark block 6 + fast | 26.4 | 3.17 | 1.43 (1.26–1.96) |
| 0.6 | DSpark block 4 + fast | 27.9 | 2.66 | **1.54** (1.28–1.91) |

**Parity** ([`runs/2026-11-25-p1-fast-verify-parity`](runs/2026-11-25-p1-fast-verify-parity/summary.json),
`scripts/check_fast_verify.py`). This compares 8-token verify logits against 8 sequential
one-token decode steps (what plain greedy computes), at 48 blocks along the smoke prompts'
greedy outputs, 384 positions in all:

| vs. sequential decode | Stock verify | Fast verify |
|---|---:|---:|
| Argmax agreement | 382/384 | 383/384 |
| Logprob gap at disagreements | 0.040, 0.042 | 0.042 |
| Mean / max KL | 5.4e-4 / 0.027 | 5.6e-4 / 0.030 |
| Mean \|Δlogit\| | 0.181 | 0.178 |

Fast verify is as close to decode as stock verify is. Stock verify already differs from
decode by BF16 rounding, which is why stock DSpark diverges from no-spec greedy at near-ties
(`divergence.json`). On the long agent turns below, greedy output matched no-spec on 9/9
prompts for both. Every verify forward routed 353 projections through the kernel: 48 visits ×
7 attention projections, plus 16 dynamic-conv projections, plus the head. Unit tests:
`tests/test_fast_verify.py`.

**What is left in an 8-token verify (132 ms vs 56 ms for one token):** mostly the NVFP4 MoE.
8 tokens route to many more distinct experts, and the grouped verify kernel runs well below
bandwidth (96 ms for the MoE alone in the microbenchmark). The draft block, now 12.9 ms,
is dominated by the drafter's sequential 7-step Markov head and small ops rather than weight
reads. With verify cheaper, block 4 now beats block 8 on short prompts (low τ). On agent
turns, where τ is 5–8, block 8 is probably still best, but that is not yet measured.

## Long-context agent prompts

Teacher pass (Stage 1 inner loop) over the first L tokens of 2 long traces
([`runs/2026-11-24-p0-context-rerun`](runs/2026-11-24-p0-context-rerun/summary.json)):

| Context | Teacher tok/s | Peak mem | Features/window |
|---|---:|---:|---:|
| 2K | 195 | 23.3 GB | 36 MB |
| 4K | 183 | 25.1 GB | 72 MB |
| 8K | 174 | 25.3 GB | 144 MB |
| 16K | 175 | 25.5 GB | 288 MB |
| 32K | 173 | 26.0 GB | 576 MB |

Decode on real agent turns: prompt cut right before an assistant turn, greedy, ≤128 tokens,
one sample each. The 3 traces are pinned with `--traces`, and no-spec, stock DSpark and
DSpark + fast run back to back per prompt
([`runs/2026-11-25-p1-context-fast`](runs/2026-11-25-p1-context-fast/summary.json)).
Greedy output matched no-spec on 9/9 prompts for both drafter runs.

| Trace | Prompt tokens | TTFT | No-spec tok/s | τ (stock / fast) | Stock DSpark | DSpark + fast |
|---|---:|---:|---:|---:|---:|---:|
| sweagent | 3,168 | 15 s | 18.0 | 5.54 / 5.62 | 1.39× | 2.00× |
| sweagent | 4,580 | 21 s | 18.1 | 5.82 | 1.48× | 2.14× |
| openhands | 5,391 | 25 s | 17.9 | 7.38 | 1.81× | 2.55× |
| opencode | 11,570 | 55 s | 17.6 | 7.78 | 1.84× | 2.53× |
| sweagent | 15,459 | 73 s | 17.4 | 6.33 | 1.46× | 2.03× |
| openhands | 15,602 | 74 s | 17.3 | 5.24 / 5.56 | 1.26× | 1.85× |
| sweagent | 27,422 | 141 s | 16.4 | 8.00 | 1.72× | 2.20× |
| opencode | 31,629 | 160 s | 16.5 | 4.74 | 1.10× | 1.50× |
| openhands | 32,623 | 170 s | 16.4 | 4.50 | 1.01× | 1.37× |

- **Stock: median 1.46× (1.01–1.84×). Fast: median 2.03× (1.37–2.55×)**, at τ 4.5–8.0.
  Long-context agent data is where the drafter pays off most, and where acceptance varies
  most. The two ~32K turns with τ≈4.5 are the weakest, which makes them the obvious target
  for fine-tuning.
- The stock column reproduces the earlier runs
  ([`p0-context-rerun`](runs/2026-11-24-p0-context-rerun/summary.json),
  [`p0-context`](runs/2026-11-24-p0-context/summary.json)) within ±0.06×, except the
  3,168-token case.
- **Warm-up artifact:** the first speculative generation in a process pays ~0.8 s of
  one-time setup. That turned the 3,168-token case into 1.09× in the rerun, and the
  5,391-token openhands case into 1.25× in the first run (1.39× and 1.81× here). Tokens and
  acceptance were identical. `probe_context.py` now warms up every decode condition;
  `bench.py` always did.
- **Memory is not the constraint**: 26 GB peak at 32K. **Time is**: Stage 1 runs at
  ~175–195 tok/s, about 1.5 h per 1M tokens. **Disk** costs 18 KB/token.

## Data: Open-SWE-Traces v2.0 under Inkling's template

15,265 traces from 43 agent/teacher/source groups, rendered with Inkling's chat template
([`runs/2026-11-24-p2-survey`](runs/2026-11-24-p2-survey/summary.txt)).

| Cap | Traces that fit whole (sample) | ≈ pool in v2.0 (size-weighted) | Assistant tokens in a prefix of this length |
|---|---:|---:|---:|
| 4K | 0.2% | n/a | 179 |
| 8K | 1.7% | ~5K traces | 594 |
| 16K | 9.4% | ~33K traces, ~115M asst tokens | 2,420 |
| 32K | 29.8% | ~120K traces | 7,921 |

- Median trace is 50K tokens. System prompt + tool schemas + issue fill most of the first 4K,
  so 2–4K caps carry almost no assistant signal.
- `thinking-low`/`thinking-medium` and Claude Code / mini-SWE-agent groups are shortest:
  12–44% fit in 16K.
- Short traces skew easier: 72% resolved at ≤16K vs 38% at >32K, with a median of 26 vs 109
  messages. They still cover ~700 repos in the sample alone.
