"""Explain greedy divergences between no-spec and speculative runs.

For every prompt whose greedy outputs differ, re-run the target on
prompt + common prefix and report the teacher's top tokens at the divergence
position. A near-tie (tiny top-1/top-2 logit gap) points to verify-shape
numerics; a large gap where speculation picked a non-argmax token points to a bug.

Usage:
    uv run python scripts/check_divergence.py runs/2026-11-23-p0-verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from specdec import runtime as rt
from specdec.teacher import language_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default=rt.DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, default=Path("prompts/coding_smoke.jsonl"))
    ap.add_argument("--reasoning-effort", default="low")
    args = ap.parse_args()

    tokenizer = rt.load_tokenizer(args.model)
    model, _, _ = rt.load_target(args.model)
    lm = language_model(model)
    prompts = {json.loads(l)["id"]: json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()}

    report = []
    for line in (args.run_dir / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        d = r["first_divergence"]
        if d is None:
            continue
        base, spec = r["nospec_t0"]["tokens"], r["dspark_t0"]["tokens"]
        ids = rt.render_messages(tokenizer, prompts[r["id"]]["messages"], args.reasoning_effort)
        prefix = ids + base[:d]
        # Two reference shapes: one-shot prefill (bulk) and token-by-token decode.
        cache = lm.make_cache()
        logits = lm(inputs=mx.array([prefix]), cache=cache).logits[0, -1].astype(mx.float32)
        cache = lm.make_cache()
        lm(inputs=mx.array([prefix[:-1]]), cache=cache)
        step = lm(inputs=mx.array([prefix[-1:]]), cache=cache).logits[0, -1].astype(mx.float32)
        mx.eval(logits, step)
        top = mx.argsort(-logits)[:5].tolist()
        lp = (logits - mx.logsumexp(logits)).tolist()
        entry = {
            "id": r["id"],
            "position": d,
            "nospec_token": base[d],
            "dspark_token": spec[d],
            "top5": [(t, tokenizer.decode([t]), round(lp[t], 4)) for t in top],
            "gap_top1_top2": round(lp[top[0]] - lp[top[1]], 4),
            "nospec_vs_dspark_logit_gap": round(float(logits[base[d]] - logits[spec[d]]), 4),
            "argmax_bulk": top[0],
            "argmax_step": int(mx.argmax(step).item()),
        }
        report.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    rt.write_json(args.run_dir / "divergence.json", report)


if __name__ == "__main__":
    main()
