"""Where does the target's verify time go? (#1 follow-up)

Verify cost jumps from ~53 ms (1 token) to ~120 ms (2 tokens) and then grows slowly. With
8 of 128 NVFP4 experts at ~4.5 MB each, one token reads only ~1.7 GB of expert weights
(~5 ms at M1 Max bandwidth), so the jump is not memory traffic. This probe times the real
verify forward (capture + cache transaction, rolled back) at T = 1..8 tokens with whole
components stubbed out, plus the MoE alone on each of its code paths:

  full         the runtime's verify forward
  no_moe       SparseMoE returns zeros            -> MoE share = full - no_moe
  no_attn      Attention returns zeros            -> attention share = full - no_attn
  no_moe_attn  both stubbed (convs, attention-residual mixing, loop messages, lm_head)
  moe_*        all 48 MoE visits alone: grouped (verify kernel), masked, gather

Each row also reports graph-build time on the CPU (Python op dispatch before the final eval).

Usage:
    uv run python scripts/profile_verify.py --out runs/2026-11-24-p0-verify-profile
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.inkling2 import language as L
from mlx_vlm.speculative.cache_state import abort_speculative_round

from specdec import runtime as rt
from specdec.teacher import CAPTURE_IDS, language_model

SIZES = (1, 2, 3, 4, 8)


def med_ms(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=rt.DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, default=Path("prompts/coding_smoke.jsonl"))
    ap.add_argument("--depth", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--fast-verify", action="store_true", help="2-8-row projections on specdec.fast_verify")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rt.set_fast_verify(args.fast_verify)

    env = rt.env_info(args.model)
    tokenizer = rt.load_tokenizer(args.model)
    model, _, _ = rt.load_target(args.model)
    lm = language_model(model)

    lines = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]
    ids: list[int] = []
    while len(ids) < args.depth + 16:
        for p in lines:
            ids += rt.render_messages(tokenizer, p["messages"])
    ctx, cont = ids[: args.depth], ids[args.depth : args.depth + 16]
    cache = lm.make_cache()
    for s in range(0, len(ctx), 256):
        lm(inputs=mx.array([ctx[s : s + 256]]), cache=cache)
        mx.eval([c.state for c in cache])

    build = {}

    def verify(t):
        t0 = time.perf_counter()
        out = lm(inputs=mx.array([cont[:t]]), cache=cache, capture_layer_ids=list(CAPTURE_IDS),
                 speculative_verify=True)
        build[t] = (time.perf_counter() - t0) * 1000
        mx.eval(out.logits, out.hidden_states)
        abort_speculative_round(out.gdn_states)

    orig_moe, orig_attn = L.SparseMoE.__call__, L.Attention.__call__

    def zero_moe(self, x, speculative_verify=False):
        return mx.zeros_like(x)

    def zero_attn(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        return mx.zeros_like(x)

    variants = {
        "full": (orig_moe, orig_attn),
        "no_moe": (zero_moe, orig_attn),
        "no_attn": (orig_moe, zero_attn),
        "no_moe_attn": (zero_moe, zero_attn),
    }
    rows = {}
    for name, (moe, attn) in variants.items():
        L.SparseMoE.__call__, L.Attention.__call__ = moe, attn
        try:
            for t in SIZES:
                verify(t)  # warm up this shape
                ms = med_ms(lambda t=t: verify(t), args.repeats)
                rows.setdefault(name, {})[t] = {"ms": ms, "build_ms": round(build[t], 2)}
        finally:
            L.SparseMoE.__call__, L.Attention.__call__ = orig_moe, orig_attn

    # MoE alone over all 48 visits, per code path. Random activations route more diversely
    # than real consecutive tokens, so this is a pessimistic bound on distinct experts.
    layers, visits = lm.model.layers, lm.model.layer_indices
    mx.random.seed(0)

    def moe_path(path, t):
        xs = [mx.random.normal((1, t, 2304)).astype(mx.bfloat16) for _ in visits]
        mx.eval(xs)

        def run():
            outs = []
            for i, x in zip(visits, xs):
                mlp = layers[i].mlp
                if path == "grouped":
                    outs.append(mlp(x, speculative_verify=True))
                elif path == "masked":
                    outs.append(mlp(x, speculative_verify=False))
                else:  # plain gather_qmm / gather_qmv
                    idx, w = L._route(x, mlp.gate_weight, mlp.e_score_correction_bias,
                                      mlp.top_k, mlp.route_scale)
                    outs.append(L._combine_routes(mlp.switch_mlp(x, idx), w))
            mx.eval(outs)

        run()
        return med_ms(run, args.repeats)

    for path in ("grouped", "masked", "gather"):
        for t in SIZES:
            rows.setdefault(f"moe_{path}", {})[t] = {"ms": moe_path(path, t)}

    print(f"depth {args.depth}, median ms over {args.repeats} repeats (build = CPU graph construction)")
    print(f"{'variant':14s}" + "".join(f"{'T=' + str(t):>16s}" for t in SIZES))
    for name, r in rows.items():
        cells = "".join(
            f"{r[t]['ms']:8.1f} ({r[t]['build_ms']:4.1f})" if "build_ms" in r[t] else f"{r[t]['ms']:16.1f}"
            for t in SIZES
        )
        print(f"{name:14s}{cells}")
    full, no_moe, no_attn, rest = (rows[k] for k in ("full", "no_moe", "no_attn", "no_moe_attn"))
    print("\nattribution (ms): MoE = full - no_moe, attention = full - no_attn")
    for t in SIZES:
        print(f"  T={t}: full {full[t]['ms']:6.1f} | MoE {full[t]['ms'] - no_moe[t]['ms']:6.1f}"
              f" | attention {full[t]['ms'] - no_attn[t]['ms']:6.1f} | rest {rest[t]['ms']:6.1f}")

    rt.write_json(args.out / "summary.json", {"env": env, "settings": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "ms": rows})


if __name__ == "__main__":
    main()
