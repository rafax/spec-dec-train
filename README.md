<img src="assets/icon.svg" width="64" height="64" alt="">

# spec-dec-train

Speculative decoding experiments for **Inkling 2 Mini** (`thinkingmachines/Inkling-2-Mini`)
on Apple Silicon with MLX. It benchmarks the bundled **DSpark** drafter on coding and
coding-agent prompts, then distills a coding-specialized drafter offline:

1. cache target features and teacher logits
2. train the drafter on the cache
3. compare against stock DSpark

- [docs/PLAN.md](docs/PLAN.md): phases, scaling knobs, experiment tracking
- [docs/dspark.md](docs/dspark.md): verified checkpoint/runtime/architecture notes
- Work is tracked in [issues](../../issues); changes land via PRs.

## Setup

```bash
uv sync                      # Python 3.13, mlx 0.32.4, mlx-vlm 0.8.1
cp .env.example .env         # add a read-only HF token
hf download thinkingmachines/Inkling-2-Mini   # ~20 GB
```

On M1/M2-class GPUs, pass `--fast-verify` to the scripts, or append `+fast` to a bench
condition (`dspark+fast`). This runs 2–8-token verify and drafting on
`specdec.fast_verify`'s kernels instead of MLX's slow small-batch GEMM, which takes stock
DSpark from ~1.0× to ~1.5× on short prompts and ~2× on agent turns
([RESULTS.md](RESULTS.md#fast-verify-kernels-11)).

## Layout

```
src/specdec/     library code (data rendering, benchmarking, caching, training)
scripts/         entry points
configs/         run configs
docs/            plan and technical notes
data/, runs/     local artifacts (gitignored except small run summaries)
```
