"""Phase 0 (#9): verify mlx-vlm + native Inkling 2 Mini + bundled DSpark on this Mac.

For each prompt: greedy no-spec vs greedy DSpark (parity + speed + acceptance),
then one T=0.6 DSpark run as a sampling sanity check.

Usage:
    uv run python scripts/verify_runtime.py --out runs/2026-11-23-p0-verify
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import mlx.core as mx

from specdec import runtime as rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=rt.DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, default=Path("prompts/coding_smoke.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--prefill-step-size", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    env = rt.env_info(args.model)
    print(json.dumps(env, indent=2))
    tokenizer = rt.load_tokenizer(args.model)
    model, processor, load_stats = rt.load_target(args.model)
    print("target loaded:", load_stats)
    draft, kind, draft_stats = rt.load_dspark(args.model)
    print("drafter loaded:", draft_stats)

    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    common = dict(max_tokens=args.max_tokens, prefill_step_size=args.prefill_step_size)

    # Warm up kernels/caches so the first measured prompt is not penalized.
    warm = rt.render_messages(tokenizer, prompts[0]["messages"], args.reasoning_effort)
    rt.generate(model, processor, warm, max_tokens=16, prefill_step_size=args.prefill_step_size)
    rt.generate(model, processor, warm, draft=draft, draft_kind=kind, max_tokens=16,
                prefill_step_size=args.prefill_step_size)

    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.jsonl").open("w") as f:
        for p in prompts:
            ids = rt.render_messages(tokenizer, p["messages"], args.reasoning_effort)
            base = rt.generate(model, processor, ids, temperature=0.0, **common)
            spec = rt.generate(model, processor, ids, draft=draft, draft_kind=kind, temperature=0.0, **common)
            samp = rt.generate(model, processor, ids, draft=draft, draft_kind=kind, temperature=0.6, seed=1, **common)
            div = rt.first_divergence(base.tokens, spec.tokens)
            row = {
                "id": p["id"],
                "prompt_tokens": len(ids),
                "greedy_parity": div is None,
                "first_divergence": div,
                "nospec_t0": base.to_json(),
                "dspark_t0": spec.to_json(),
                "dspark_t06": samp.to_json(include_tokens=False),
            }
            f.write(json.dumps(row) + "\n")
            rows.append(row)
            a0, a6 = spec.acceptance(), samp.acceptance()
            print(
                f"{p['id']:12s} prompt={len(ids):4d} gen={base.n_gen:3d} parity={'OK' if div is None else f'DIVERGE@{div}'} | "
                f"no-spec {base.decode_tps:5.1f} tok/s ttft {base.ttft_s:5.2f}s | "
                f"DSpark T0 {spec.decode_tps:5.1f} tok/s tau {a0.get('tau')} | "
                f"T0.6 {samp.decode_tps:5.1f} tok/s tau {a6.get('tau')} | peak {spec.peak_gb} GB"
            )

    def med(key, cond):
        return round(statistics.median(r[cond][key] for r in rows), 2)

    summary = {
        "env": env,
        "load": {"target": load_stats, "drafter": draft_stats},
        "settings": vars(args) | {"prompts": str(args.prompts), "out": str(args.out)},
        "n_prompts": len(rows),
        "greedy_parity": f"{sum(r['greedy_parity'] for r in rows)}/{len(rows)}",
        "median": {
            cond: {
                "decode_tps": med("decode_tps", cond),
                "e2e_tps": med("e2e_tps", cond),
                "ttft_s": med("ttft_s", cond),
                "peak_gb": med("peak_gb", cond),
            }
            for cond in ("nospec_t0", "dspark_t0", "dspark_t06")
        },
        "tau": {
            cond: round(statistics.median(r[cond]["acceptance"]["tau"] for r in rows), 3)
            for cond in ("dspark_t0", "dspark_t06")
        },
        "per_position_t0": [
            round(statistics.mean(r["dspark_t0"]["acceptance"]["per_position"][k] for r in rows
                                  if r["dspark_t0"]["acceptance"]["per_position"][k] is not None), 3)
            for k in range(7)
        ],
        "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2),
    }
    summary["speedup_decode_t0"] = round(
        summary["median"]["dspark_t0"]["decode_tps"] / summary["median"]["nospec_t0"]["decode_tps"], 2
    )
    rt.write_json(args.out / "summary.json", summary)
    rt.write_json(args.out / "env.json", env)
    print(json.dumps({k: summary[k] for k in ("greedy_parity", "median", "tau", "per_position_t0", "speedup_decode_t0")}, indent=2))


if __name__ == "__main__":
    main()
