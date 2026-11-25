"""Logit parity of specdec.fast_verify against stock verify and plain decode (#11).

Plain greedy decoding computes one token per forward. Speculative verify computes a block
of 8 in one forward, so stock verify already drifts from decode by BF16 rounding (that is
why stock DSpark diverges from no-spec greedy at near-ties). The fast kernels change the
reduction order again; the bar is that they stay as close to decode as stock verify does.

For each prompt, walk the stored greedy no-spec continuation. Every --stride tokens, take
the next 8 tokens as a verify block and compute their logits three ways from the same
cache: stock verify, fast verify (both rolled back), and 8 sequential decode steps.

Usage:
    uv run python scripts/check_fast_verify.py --greedy runs/2026-11-23-p0-verify/results.jsonl \
        --out runs/2026-11-25-p1-fast-verify-parity
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_vlm.speculative.cache_state import abort_speculative_round

from specdec import fast_verify as fv
from specdec import runtime as rt
from specdec.teacher import CAPTURE_IDS, language_model

BLOCK = 8


def compare(a: mx.array, ref: mx.array) -> dict:
    """a, ref: (positions, vocab) FP32 logits."""
    lp_a = a - mx.logsumexp(a, axis=-1, keepdims=True)
    lp_r = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    kl = (mx.exp(lp_r) * (lp_r - lp_a)).sum(axis=-1)
    diff = mx.abs(a - ref)
    top2 = mx.sort(lp_r, axis=-1)[:, -2:]
    agree = mx.argmax(a, axis=-1) == mx.argmax(ref, axis=-1)
    mx.eval(kl, diff, top2, agree)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "kl": kl.tolist(),
        "argmax_agree": agree.tolist(),
        "ref_gap": (top2[:, 1] - top2[:, 0]).tolist(),  # decode's top1 - top2 logprob gap
    }


def summarize(rows: list[dict]) -> dict:
    kl = [k for r in rows for k in r["kl"]]
    agree = [a for r in rows for a in r["argmax_agree"]]
    gaps = [g for r in rows for a, g in zip(r["argmax_agree"], r["ref_gap"]) if not a]
    return {
        "positions": len(agree),
        "max_abs": round(max(r["max_abs"] for r in rows), 4),
        "mean_abs": round(sum(r["mean_abs"] for r in rows) / len(rows), 5),
        "kl_max": float(f"{max(kl):.3g}"),
        "kl_mean": float(f"{sum(kl) / len(kl):.3g}"),
        "argmax_agree": f"{sum(agree)}/{len(agree)}",
        "disagree_ref_gaps": [round(g, 4) for g in gaps],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=rt.DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, default=Path("prompts/coding_smoke.jsonl"))
    ap.add_argument("--greedy", type=Path, default=Path("runs/2026-11-23-p0-verify/results.jsonl"),
                    help="verify_runtime results with nospec_t0 tokens")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    env = rt.env_info(args.model)
    tokenizer = rt.load_tokenizer(args.model)
    model, _, _ = rt.load_target(args.model)
    lm = language_model(model)
    print(f"target training={lm.training}")
    prompts = {json.loads(l)["id"]: json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()}
    greedy = [json.loads(l) for l in args.greedy.read_text().splitlines() if l.strip()]

    def verify(block, fast):
        rt.set_fast_verify(fast)
        try:
            out = lm(inputs=mx.array([block]), cache=cache, capture_layer_ids=list(CAPTURE_IDS),
                     speculative_verify=True)
            logits = out.logits[0].astype(mx.float32)
            mx.eval(logits, out.hidden_states)
            abort_speculative_round(out.gdn_states)
        finally:
            rt.set_fast_verify(False)
        return logits

    per = {"stock_vs_decode": [], "fast_vs_decode": [], "fast_vs_stock": []}
    coverage = Counter()
    for g in greedy:
        ids = rt.render_messages(tokenizer, prompts[g["id"]]["messages"], args.reasoning_effort)
        cont = g["nospec_t0"]["tokens"]
        cache = lm.make_cache()
        for s in range(0, len(ids), 256):
            lm(inputs=mx.array([ids[s : s + 256]]), cache=cache)
            mx.eval([c.state for c in cache])
        pos = 0
        while pos + BLOCK <= len(cont):
            block = cont[pos : pos + BLOCK]
            stock = verify(block, False)
            fv.STATS.clear()
            fast = verify(block, True)
            coverage.update(fv.STATS)
            decode = []
            for t in block:  # advances the cache by the block
                decode.append(lm(inputs=mx.array([[t]]), cache=cache).logits[0, -1].astype(mx.float32))
                mx.eval(decode[-1])
            decode = mx.stack(decode)
            for name, (a, ref) in {"stock_vs_decode": (stock, decode), "fast_vs_decode": (fast, decode),
                                   "fast_vs_stock": (fast, stock)}.items():
                per[name].append({"prompt": g["id"], "pos": pos, **compare(a, ref)})
            nxt = cont[pos + BLOCK : pos + args.stride]
            if nxt:
                lm(inputs=mx.array([nxt]), cache=cache)
                mx.eval([c.state for c in cache])
            pos += args.stride
        s, f = per["stock_vs_decode"][-1], per["fast_vs_decode"][-1]
        print(f"{g['id']:12s} last block: stock max|d| {s['max_abs']:.3f} | fast max|d| {f['max_abs']:.3f}")

    blocks = len(per["fast_vs_decode"])
    summary = {name: summarize(rows) for name, rows in per.items()}
    summary["kernel_calls_per_verify"] = {k: round(v / blocks, 1) for k, v in sorted(coverage.items())}
    print(json.dumps(summary, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "blocks.jsonl").open("w") as f:
        for name, rows in per.items():
            for r in rows:
                f.write(json.dumps({"comparison": name, **r}) + "\n")
    rt.write_json(args.out / "summary.json", {"env": env, "settings": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "summary": summary})


if __name__ == "__main__":
    main()
