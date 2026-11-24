"""Benchmark harness (#2): conditions x prompts x temperatures x repeats -> JSONL + summary.

Conditions:
  nospec          plain decoding
  dspark          bundled DSpark, trained block (8)
  dspark@B        bundled DSpark with runtime block size B (1 anchor + B-1 proposals)
  <dir>[@B]       fine-tuned drafter directory (config.json + mtp.safetensors)

Speedups are paired: each speculative run is divided by the no-spec run of the same
prompt / temperature / repeat, then the median is reported.

Usage:
    uv run python scripts/bench.py --conditions nospec dspark@3 dspark@4 dspark@6 dspark \
        --temps 0.0 --repeats 2 --out runs/2026-11-24-p1-block-sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from specdec import runtime as rt


def parse_condition(cond: str, default_model: str):
    if cond == "nospec":
        return None, None
    source, _, block = cond.partition("@")
    source = default_model if source == "dspark" else source
    return source, int(block) if block else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=rt.DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, default=Path("prompts/coding_smoke.jsonl"))
    ap.add_argument("--conditions", nargs="+", default=["nospec", "dspark"])
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--prefill-step-size", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    env = rt.env_info(args.model)
    env["swap_used_mb_start"] = rt.swap_used_mb()
    tokenizer = rt.load_tokenizer(args.model)
    model, processor, load_stats = rt.load_target(args.model)
    drafters = {}
    for cond in args.conditions:
        source, _ = parse_condition(cond, args.model)
        if source and source not in drafters:
            draft, kind, _ = rt.load_dspark(source)
            drafters[source] = (draft, kind)

    prompts = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]
    rendered = {p["id"]: rt.render_messages(tokenizer, p["messages"], args.reasoning_effort) for p in prompts}
    common = {"max_tokens": args.max_tokens, "prefill_step_size": args.prefill_step_size}

    # Warm-up every condition once (kernel compilation for each verify shape).
    warm = rendered[prompts[0]["id"]]
    for cond in args.conditions:
        source, block = parse_condition(cond, args.model)
        draft, kind = drafters.get(source, (None, None))
        rt.generate(model, processor, warm, draft=draft, draft_kind=kind, draft_block_size=block,
                    max_tokens=24, prefill_step_size=args.prefill_step_size)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / "results.jsonl").open("w") as f:
        for rep in range(args.repeats):
            for temp in args.temps:
                for p in prompts:
                    # Interleave conditions per prompt so drift (thermals) hits all equally.
                    for cond in args.conditions:
                        source, block = parse_condition(cond, args.model)
                        draft, kind = drafters.get(source, (None, None))
                        rec = rt.generate(model, processor, rendered[p["id"]], draft=draft, draft_kind=kind,
                                          draft_block_size=block, temperature=temp, seed=rep, **common)
                        row = {"condition": cond, "prompt": p["id"], "temp": temp, "repeat": rep,
                               **rec.to_json(include_tokens=False)}
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        rows.append(row)
                        acc = rec.acceptance()
                        print(f"rep{rep} T={temp} {p['id']:12s} {cond:10s} decode {rec.decode_tps:5.1f} tok/s"
                              f" e2e {rec.e2e_tps:5.1f} tau {acc.get('tau', '-')}", flush=True)

    base = {(r["prompt"], r["temp"], r["repeat"]): r for r in rows if r["condition"] == "nospec"}
    summary = defaultdict(dict)
    for temp in args.temps:
        for cond in args.conditions:
            rs = [r for r in rows if r["condition"] == cond and r["temp"] == temp]
            s = {
                "n": len(rs),
                "decode_tps": round(statistics.median(r["decode_tps"] for r in rs), 2),
                "e2e_tps": round(statistics.median(r["e2e_tps"] for r in rs), 2),
                "ttft_s": round(statistics.median(r["ttft_s"] for r in rs), 3),
                "peak_gb": max(r["peak_gb"] for r in rs),
            }
            if cond != "nospec":
                s["tau"] = round(statistics.median(r["acceptance"]["tau"] for r in rs), 3)
                pairs = [r["decode_tps"] / base[(r["prompt"], r["temp"], r["repeat"])]["decode_tps"]
                         for r in rs if (r["prompt"], r["temp"], r["repeat"]) in base]
                if pairs:
                    s["paired_decode_speedup"] = round(statistics.median(pairs), 3)
                depth = max(len(r["acceptance"]["per_position"]) for r in rs)
                s["per_position"] = [
                    round(statistics.mean(v), 3) if (v := [r["acceptance"]["per_position"][k] for r in rs
                                                          if k < len(r["acceptance"]["per_position"])
                                                          and r["acceptance"]["per_position"][k] is not None]) else None
                    for k in range(depth)
                ]
            summary[f"T={temp}"][cond] = s
    env["swap_used_mb_end"] = rt.swap_used_mb()
    rt.write_json(args.out / "summary.json", {"env": env, "load": load_stats, "settings": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "summary": summary})
    rt.write_json(args.out / "env.json", env)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
