# Results

Hardware/runtime for all rows: M1 Max (24-core GPU), 64 GB, macOS 26.5, mlx 0.32.4,
mlx-vlm 0.8.1, `thinkingmachines/Inkling-2-Mini` @ `7a67ae2`, native NVFP4/MXFP8 target,
BF16 DSpark. Run artifacts live under [`runs/`](runs/).

Background load during these runs: Unsloth Studio's `llama-server` (Inkling 2 Mini GGUF
Q4_K_XL) stayed loaded but **idle** the whole time. Its last request finished at 23:15 on
2026-11-23; the first run here started at 23:26. It held ~12 GB plus a 16 GB mmapped GGUF,
and swap stayed at ~9 GB. An idle server takes no GPU time, and mlx-vlm wires model memory
during generation, so timings were stable: no-spec decode was 18.29 ± 0.10 tok/s (CV 0.6%)
over 24 runs, with flat swap across each run. Acceptance, parity and data numbers don't depend on
timing. From now on, `env.json` records any other model servers and warns when they are present.

## Stock DSpark vs no speculation (short coding prompts)

6 single-turn coding prompts, 256 generated tokens, reasoning effort `low`, 2 repeats,
conditions interleaved per prompt. Speedups are paired per prompt/repeat (median).
Run: [`runs/2026-11-24-p1-block-sweep`](runs/2026-11-24-p1-block-sweep/summary.json).

| T | Condition | Decode tok/s | τ (tokens / target forward) | Decode speedup |
|---|---|---:|---:|---:|
| 0.0 | no-spec | 18.3 | n/a | 1.00 |
| 0.0 | DSpark, block 4 | 17.2 | 2.96 | 0.94 |
| 0.0 | DSpark, block 6 | 17.2 | 3.30 | 0.94 |
| 0.0 | DSpark, block 8 (default) | 18.8 | 3.88 | **1.03** |
| 0.6 | no-spec | 18.3 | n/a | 1.00 |
| 0.6 | DSpark, block 4 | 15.7 | 2.70 | 0.86 |
| 0.6 | DSpark, block 6 | 16.9 | 3.25 | 0.92 |
| 0.6 | DSpark, block 8 (default) | 15.5 | 3.24 | **0.85** |

Per-position acceptance P(accepted ≥ k), block 8:

- T=0: 0.78 / 0.61 / 0.48 / 0.37 / 0.30 / 0.23 / 0.15
- T=0.6: 0.73 / 0.52 / 0.38 / 0.28 / 0.20 / 0.14 / 0.12

## Why τ≈3.9 is only break-even here

Verify-cost probe at 1K context ([`runs/2026-11-24-p0-verify-cost`](runs/2026-11-24-p0-verify-cost/summary.json)):

| Step | ms | vs plain decode (55 ms in real generation) |
|---|---:|---:|
| target verify, 2 tokens | 120 | 2.2× |
| target verify, 4 tokens | 139 | 2.5× |
| target verify, 8 tokens | 174 | 3.2× |
| DSpark draft block | 12–22 | ~0.3× |

Each extra verify token routes to different experts (8 of 128 per layer), so verify cost
jumps at 2 tokens and keeps climbing. Including round overhead, **break-even on this Mac is
τ ≈ 3.9 at block 8**. Acceptance gains translate roughly linearly above that: τ 4.9 → ~1.26×.

## Long-context agent prompts

Real agent trajectories cut right before an assistant turn; greedy, 128 tokens; 2 traces per
length ([`runs/2026-11-24-p0-context`](runs/2026-11-24-p0-context/summary.json)).

| Context | Teacher pass (Stage 1) tok/s | Peak mem | Features/window | TTFT | Decode no-spec → DSpark | τ |
|---|---:|---:|---:|---:|---:|---:|
| 2K | 192 | 23.3 GB | 36 MB | n/a | n/a | n/a |
| 4K | 189 | 23.4 GB | 72 MB | n/a | n/a | n/a |
| ~5K | 185 | 25.3 GB | 144 MB | 25 s | 17.9 → 22.4 | 7.4 |
| 12–16K | 180 | 25.6 GB | 288 MB | 55–74 s | 17.4 → 26.9 | 5.2–7.8 |
| 32K | 172 | 26.0 GB | 576 MB | 163 s | 16.4 → 17.4 | 4.5–4.7 |

- **Memory is not the constraint**: 26 GB peak at 32K.
- **Time is**: Stage 1 runs at ~185 tok/s, about 1.5 h per 1M tokens. **Disk** costs 18 KB/token.
- DSpark accepts far more on real agent continuations (τ 4.5–7.8) than on single-turn prompts
  (τ 3.9), and **acceptance falls with depth**. Long-context agent data is where fine-tuning
  has the most room.
- Greedy parity held on all long-context runs. On short prompts, 5/6 diverged, all at
  near-ties of 0.008–0.14 nats, where DSpark usually picked the reference argmax
  ([`divergence.json`](runs/2026-11-23-p0-verify/divergence.json)).

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
