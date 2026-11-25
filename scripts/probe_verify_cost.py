"""Cost model for DSpark rounds: why does tau ~3.9 not translate into a speedup?

Measures, at a fixed context depth:
  t_dec      plain one-token decode step (no capture, no transaction)
  t_ver(b)   the runtime's verify forward for b tokens (capture + cache transaction, rolled back)
  t_draft(b) one DSpark draft_block call producing b-1 proposals (steady state: ~4 new context rows)

and combines them with measured per-position acceptance P(accepted >= k) into the
predicted decode speedup per block size b:

  speedup(b) = E_b * t_dec / (t_ver(b) + t_draft(b)),   E_b = 1 + sum_{k<b} P(acc >= k)

Usage:
    uv run python scripts/probe_verify_cost.py --acceptance runs/2026-11-23-p0-verify/summary.json \
        --out runs/2026-11-24-p0-verify-cost
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.speculative.cache_state import abort_speculative_round

from specdec import runtime as rt
from specdec.teacher import CAPTURE_IDS, language_model


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000, min(times) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=rt.DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, default=Path("prompts/coding_smoke.jsonl"))
    ap.add_argument("--acceptance", type=Path, help="summary.json with per_position_t0")
    ap.add_argument("--depth", type=int, default=1024, help="context tokens before the timed step")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--fast-verify", action="store_true", help="2-8-row projections on specdec.fast_verify")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rt.set_fast_verify(args.fast_verify)

    env = rt.env_info(args.model)
    env["swap_used_mb_start"] = rt.swap_used_mb()
    tokenizer = rt.load_tokenizer(args.model)
    model, _, _ = rt.load_target(args.model)
    draft, _, _ = rt.load_dspark(args.model)
    lm = language_model(model)

    # Realistic coding context: prompts + code, repeated up to the requested depth.
    lines = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]
    ids: list[int] = []
    while len(ids) < args.depth + 16:
        for p in lines:
            ids += rt.render_messages(tokenizer, p["messages"])
    ctx, cont = ids[: args.depth], ids[args.depth : args.depth + 16]

    # Prefill with capture (also yields the drafter's context features).
    cache = lm.make_cache()
    hidden = []
    for s in range(0, len(ctx), 256):
        out = lm(inputs=mx.array([ctx[s : s + 256]]), cache=cache, capture_layer_ids=list(CAPTURE_IDS))
        hidden.append(mx.concatenate(out.hidden_states, axis=-1))
        mx.eval(hidden[-1], [c.state for c in cache])
    hidden = mx.concatenate(hidden, axis=1)

    results = {"depth": args.depth}

    def verify(b):
        out = lm(inputs=mx.array([cont[:b]]), cache=cache, capture_layer_ids=list(CAPTURE_IDS),
                 speculative_verify=True)
        mx.eval(out.logits, out.hidden_states)
        abort_speculative_round(out.gdn_states)

    for b in range(1, 9):
        verify(b)  # warm up this shape
        results[f"t_ver_{b}"] = timed(lambda b=b: verify(b), args.repeats)

    # Plain decode: sequential one-token steps on a scratch copy of the cache state.
    def decode_steps(n=args.repeats):
        times = []
        for i in range(n):
            t0 = time.perf_counter()
            o = lm(inputs=mx.array([[cont[i % len(cont)]]]), cache=cache)
            mx.eval(o.logits)
            times.append(time.perf_counter() - t0)
        return statistics.median(times) * 1000, min(times) * 1000

    # Drafter: prime its cache with the full context once, then time steady-state blocks
    # that each add 4 new context rows (≈ accepted + 1 at tau ~ 4).
    draft_cache = draft.reset(model)
    greedy = lambda logits: mx.argmax(logits, axis=-1)
    mx.eval(draft.draft_block(ctx[-1], hidden[:, :-4], draft_cache, 8, greedy))
    new_rows = hidden[:, -4:]
    for b in (2, 4, 8):
        mx.eval(draft.draft_block(ctx[-1], new_rows, draft_cache, b, greedy))
        results[f"t_draft_{b}"] = timed(
            lambda b=b: mx.eval(draft.draft_block(ctx[-1], new_rows, draft_cache, b, greedy)), args.repeats
        )
    results["t_dec"] = decode_steps()  # last: it advances the target cache
    env["swap_used_mb_end"] = rt.swap_used_mb()

    t_dec = results["t_dec"][0]
    print(f"depth {args.depth}: plain decode {t_dec:.1f} ms/token ({1000 / t_dec:.1f} tok/s)")
    for b in range(1, 9):
        print(f"  verify b={b}: {results[f't_ver_{b}'][0]:7.1f} ms  ({results[f't_ver_{b}'][0] / t_dec:4.2f}x decode)")
    for b in (2, 4, 8):
        print(f"  draft  b={b}: {results[f't_draft_{b}'][0]:7.1f} ms")

    model_rows = []
    if args.acceptance:
        per_pos = json.loads(args.acceptance.read_text())["per_position_t0"]
        t_draft = {b: results[f"t_draft_{min((2, 4, 8), key=lambda x: abs(x - b))}"][0] for b in range(2, 9)}
        print("\nPredicted decode speedup with measured stock acceptance:")
        for b in range(2, 9):
            e_b = 1 + sum(per_pos[: b - 1])
            cost = results[f"t_ver_{b}"][0] + t_draft[b]
            speedup = e_b * t_dec / cost
            # tau needed for break-even at this block size
            model_rows.append({"block": b, "E_tokens": round(e_b, 3), "round_ms": round(cost, 1),
                               "pred_speedup": round(speedup, 3),
                               "breakeven_tokens": round(cost / t_dec, 2)})
            print(f"  b={b}: E[tokens/round]={e_b:4.2f}  round={cost:6.1f} ms  -> {speedup:4.2f}x"
                  f"   (break-even needs {cost / t_dec:4.2f} tokens/round)")

    rt.write_json(args.out / "summary.json", {"env": env, "settings": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "timings_ms_median_min": results, "cost_model": model_rows})


if __name__ == "__main__":
    main()
