# Results

Hardware/runtime for all rows: M1 Max (24-core GPU), 64 GB, macOS 26.5, mlx 0.32.4,
mlx-vlm 0.8.1, `thinkingmachines/Inkling-2-Mini` @ `7a67ae2`, native NVFP4/MXFP8 target,
BF16 DSpark. Run artifacts live under [`runs/`](runs/).

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
- A crude workaround (split 2–8-row BF16 `Linear` calls into per-row GEMVs) cuts verify to
  **82 ms at 2 tokens (1.48×), 109 ms at 4, 163 ms at 8**. Top-1 tokens are unchanged, and its
  logit drift is smaller than the stock path's own 1-token vs 8-token drift. A proper
  multi-row GEMV kernel, which reads each weight row once for up to 8 tokens, should bring
  8-token verify to roughly 125 ms and the draft block to roughly 6 ms. That is an
  estimate, not yet measured.

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
one sample each. Greedy parity held on every trace.

| Trace | Prompt tokens | TTFT | No-spec → DSpark tok/s | τ | Speedup |
|---|---:|---:|---:|---:|---:|
| sweagent | 3,168 | 17 s | 17.7 → 19.4 | 5.54 | 1.09× (first spec run, includes warm-up) |
| sweagent | 4,580 | 23 s | 17.8 → 26.7 | 5.82 | 1.50× |
| openhands | 5,391 | 28 s | 17.6 → 32.3 | 7.38 | 1.84× |
| opencode* | 11,570 | 55 s | 17.5 → 32.0 | 7.78 | 1.83× |
| sweagent | 15,459 | 78 s | 17.3 → 25.5 | 6.33 | 1.47× |
| openhands | 15,602 | 77 s | 17.3 → 21.8 | 5.24 | 1.26× |
| sweagent | 27,422 | 137 s | 16.6 → 27.5 | 8.00 | 1.66× |
| opencode* | 31,629 | 161 s | 16.5 → 18.1 | 4.74 | 1.10× |
| openhands | 32,623 | 165 s | 16.3 → 16.6 | 4.50 | 1.01× |

\* From the first run ([`runs/2026-11-24-p0-context`](runs/2026-11-24-p0-context/summary.json)).
The survey grew between runs, so the default trace pick changed; `--traces` now pins traces.

- **Median 1.5× on warm agent turns (range 1.0–1.84×)**, at τ 4.5–8.0. Long-context agent
  data is where the stock drafter already pays off, and where acceptance varies most.
- **Warm-up artifact:** the first speculative generation in a process pays ~0.8 s of
  one-time setup. In the first run that turned the 5,391-token openhands case into 1.25×.
  The rerun gives 1.84×, with identical tokens and acceptance. `probe_context.py` now warms up
  speculative decoding too; `bench.py` always did.
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
