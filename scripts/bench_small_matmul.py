"""Microbenchmark `x @ W.T` for 1-8 rows on Inkling's unquantized projections (#11).

Workloads (random weights with the model's shapes and dtypes, distinct per layer so they
stream from DRAM as in the real forward):

  attn48     q/k/v/r/gate/o/sink projections over the 48 looped layer visits (BF16, 2.6 GB)
  head_f32   target verify head: FP32 activations x FP32 copy of lm_head (1.85 GB)
  head_bf16  same math reading the BF16 lm_head and promoting in-kernel (0.93 GB)
  drafter    one DSpark draft block: 3 layers (attention + 9216-wide MLP) + BF16 lm_head

Methods: MLX `x @ W.T` (stock), one GEMV per row (rowwise), mlx-vlm's exact verify GEMV
and its native block kernel where they accept the shape, and specdec.fast_verify's GEMV
(2-4 rows), MMA (5-8 rows) and automatic choice.

Usage:
    uv run python scripts/bench_small_matmul.py --out runs/2026-11-25-p1-small-matmul
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn

from specdec import fast_verify as fv
from specdec import runtime as rt

H, VOCAB = 2304, 201024
ATTN = {"q": 3072, "k": 1024, "v": 1024, "r": 384, "gate": 3072, "sink": 24}  # (N, K=H)
VISITS = [*range(6), *list(range(6, 18)) * 3, *range(18, 24)]  # core layers 6-17 x 3 passes


def make_workloads():
    def w(n, k, dtype=mx.bfloat16):
        return (mx.random.normal((n, k)) * 0.02).astype(dtype)

    layers = [[w(n, H) for n in ATTN.values()] + [w(H, 3072)] for _ in range(24)]
    head = w(VOCAB, H)
    head32 = head.astype(mx.float32)
    drafter = [
        [w(3072, H), w(1024, H), w(1024, H), w(H, 3072), w(9216, H), w(9216, H), w(H, 9216)]
        for _ in range(3)
    ]
    mx.eval(layers, head, head32, drafter)
    return {
        # (list of weights per call, activation dtype)
        "attn48": ([wt for i in VISITS for wt in layers[i]], mx.bfloat16),
        "head_f32": ([head32], mx.float32),
        "head_bf16": ([head], mx.float32),
        "drafter": ([wt for layer in drafter for wt in layer] + [head], mx.bfloat16),
    }


def stock(x, w):
    return x @ w.T


def rowwise(x, w):
    return mx.concatenate([x[:, t : t + 1] @ w.T for t in range(x.shape[1])], axis=1)


def mlxvlm_exact(x, w):
    from mlx_vlm.models.exact_speculative_verify import exact_speculative_verify_weight

    return exact_speculative_verify_weight(w, x) if x.dtype == w.dtype else None


def mlxvlm_block(x, w):
    from mlx_vlm.models.linear import native_block_linear

    layer = nn.Linear(w.shape[1], w.shape[0], bias=False)
    layer.weight = w
    layer.eval()
    return native_block_linear(layer, x) if x.dtype == w.dtype else None


def ours(kernel=None):
    def fn(x, w):
        return fv.small_matmul(x, w, kernel) if fv.supported(x, w) else None

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    env = rt.env_info()
    mx.random.seed(0)
    workloads = make_workloads()
    methods = {"stock": stock, "rowwise": rowwise, "mlxvlm_exact": mlxvlm_exact,
               "mlxvlm_block": mlxvlm_block, "ours_gemv": ours("gemv"), "ours_mma": ours("mma"),
               "ours": ours()}

    results = {}
    for wname, (weights, xdtype) in workloads.items():
        nbytes = sum(w.nbytes for w in weights)
        for m in args.rows:
            xs = {k: mx.random.normal((1, m, k)).astype(xdtype) for k in {w.shape[1] for w in weights}}
            mx.eval(xs)
            for name, fn in methods.items():
                if m == 1 and name != "stock":
                    continue

                def run(fn=fn, xs=xs, weights=weights):
                    outs = [fn(xs[w.shape[1]], w) for w in weights]
                    if any(o is None for o in outs):
                        return False
                    mx.eval(outs)
                    return True

                try:
                    if not run():
                        continue
                except (ValueError, RuntimeError) as e:  # kernel rejected the shape
                    print(f"{wname} {name} m={m}: {e}")
                    continue
                times = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    run()
                    times.append(time.perf_counter() - t0)
                ms = statistics.median(times) * 1000
                results.setdefault(wname, {}).setdefault(name, {})[m] = {
                    "ms": round(ms, 2), "gb_s": round(nbytes / ms / 1e6, 1)}
        print(f"\n{wname}: {nbytes / 2**30:.2f} GiB of weights, ms (GB/s)")
        print(f"{'method':14s}" + "".join(f"{'rows=' + str(m):>15s}" for m in args.rows))
        for name, r in results[wname].items():
            cells = "".join(f"{r[m]['ms']:7.2f} ({r[m]['gb_s']:3.0f})" if m in r else f"{'-':>15s}"
                            for m in args.rows)
            print(f"{name:14s}{cells}")

    tiles = {"gemv": fv.GEMV_TILE, "mma": fv.MMA_TILE, "gemv_max_rows": fv.GEMV_MAX_ROWS}
    rt.write_json(args.out / "summary.json", {"env": env, "tiles": tiles,
                                              "settings": {"rows": args.rows, "repeats": args.repeats},
                                              "results": results})


if __name__ == "__main__":
    main()
