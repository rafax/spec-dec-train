"""Phase 0 (#1): how do Stage 1 cost and speculative decoding scale with context length?

For each context length L and a few long real agent trajectories:

* teacher pass  - chunked prefill over the first L tokens with DSpark feature capture
                  and top-k teacher logprobs (the Stage 1 inner loop): tok/s, peak memory,
                  feature bytes
* decode        - an agent prompt cut at the last assistant turn starting <= L, greedy,
                  no-spec vs DSpark: TTFT, decode tok/s, acceptance, peak memory

Usage:
    uv run python scripts/probe_context.py --out runs/2026-11-23-p0-context
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np

from specdec import runtime as rt
from specdec.data.open_swe import load_row, messages_and_tools, read_survey
from specdec.data.render import render
from specdec.teacher import CAPTURE_IDS, teacher_forward


def pick_traces(survey, min_tokens, n):
    long = [r for r in survey if r["n_tokens"] >= min_tokens]
    long.sort(key=lambda r: zlib.crc32(f"{r['file']}#{r['row']}".encode()))
    picked, seen = [], set()
    for r in long:  # prefer different agent scaffolds
        agent = r["group"].split("/")[0]
        if agent not in seen:
            picked.append(r)
            seen.add(agent)
        if len(picked) == n:
            break
    return picked


def teacher_pass(model, ids, chunk, topk):
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    host_s, n = 0.0, 0
    for c in teacher_forward(model, ids, chunk=chunk, topk=topk):
        h0 = time.perf_counter()
        # Simulate the host copy Stage 1 does before writing shards.
        np.asarray(c.features.view(mx.uint16))
        np.asarray(c.topk_ids)
        np.asarray(c.topk_logprobs.astype(mx.float16))
        host_s += time.perf_counter() - h0
        n += c.n
    dt = time.perf_counter() - t0
    return {
        "tokens": n,
        "seconds": round(dt, 2),
        "tok_s": round(n / dt, 1),
        "host_copy_s": round(host_s, 2),
        "peak_gb": rt.gb(mx.get_peak_memory()),
        "feature_mb": round(n * len(CAPTURE_IDS) * 2304 * 2 / 2**20, 1),
        "topk_mb": round(n * topk * 6 / 2**20, 1),
    }


def agent_prompt(rendered, limit):
    starts = [s for s, _ in rendered.assistant_spans() if 0 < s <= limit]
    return rendered.input_ids[: starts[-1]] if starts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=rt.DEFAULT_MODEL)
    ap.add_argument("--survey", type=Path, default=Path("data/surveys/open_swe_v2.jsonl"))
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--n-traces", type=int, default=2)
    ap.add_argument("--traces", nargs="+", help="exact traces as <file>#<row> (overrides --n-traces)")
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    env = rt.env_info(args.model)
    tokenizer = rt.load_tokenizer(args.model)
    model, processor, load_stats = rt.load_target(args.model)
    draft, kind, _ = (None, None, None) if args.skip_decode else rt.load_dspark(args.model)
    print("loaded", load_stats)

    survey = read_survey(args.survey)
    if args.traces:  # pin exact traces; the default pick depends on what the survey holds
        by_id = {f"{r['file']}#{r['row']}": r for r in survey}
        traces = [by_id[t] for t in args.traces]
    else:
        traces = pick_traces(survey, max(args.lengths) + 1024, args.n_traces)
    rendered = []
    for r in traces:
        messages, tools = messages_and_tools(load_row(r["file"], r["row"]))
        rendered.append((r, render(tokenizer, messages, tools)))
        print(f"trace {r['group']} #{r['row']}: {rendered[-1][1].n_tokens} tokens")

    # Warm-up (kernel compilation) on a short prefix. Speculative decoding needs its own:
    # the first DSpark generation in a process pays ~0.8 s of one-time setup, which turned a
    # 1.84x run into 1.25x on a 59-token reply in runs/2026-11-24-p0-context.
    warm = rendered[0][1].input_ids[:512]
    list(teacher_forward(model, warm, chunk=args.chunk, topk=args.topk))
    if draft is not None:
        for d, k in ((None, None), (draft, kind)):
            rt.generate(model, processor, warm, draft=d, draft_kind=k, max_tokens=32,
                        prefill_step_size=args.chunk)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / "results.jsonl").open("w") as f:
        for L in args.lengths:
            for r, rend in rendered:
                row = {"length": L, "trace": f"{r['file']}#{r['row']}", "group": r["group"]}
                row["teacher"] = teacher_pass(model, rend.input_ids[:L], args.chunk, args.topk)
                prompt = None if args.skip_decode else agent_prompt(rend, L)
                if prompt is not None:
                    common = {"max_tokens": args.gen_tokens, "prefill_step_size": args.chunk}
                    base = rt.generate(model, processor, prompt, temperature=0.0, **common)
                    spec = rt.generate(model, processor, prompt, draft=draft, draft_kind=kind,
                                       temperature=0.0, **common)
                    row["prompt_tokens"] = len(prompt)
                    row["nospec"] = base.to_json(include_tokens=False)
                    row["dspark"] = spec.to_json(include_tokens=False)
                    row["greedy_parity"] = rt.first_divergence(base.tokens, spec.tokens) is None
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
                t = row["teacher"]
                msg = f"L={L:6d} teacher {t['tok_s']:7.1f} tok/s peak {t['peak_gb']:5.1f} GB feat {t['feature_mb']:7.1f} MB"
                if "dspark" in row:
                    b, s = row["nospec"], row["dspark"]
                    msg += (f" | prompt {row['prompt_tokens']:6d} ttft {b['ttft_s']:6.1f}s"
                            f" decode {b['decode_tps']:5.1f} -> {s['decode_tps']:5.1f} tok/s"
                            f" tau {s['acceptance'].get('tau')} peak {s['peak_gb']} GB parity {row['greedy_parity']}")
                print(msg, flush=True)

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 2) if xs else None

    table = []
    for L in args.lengths:
        rs = [r for r in rows if r["length"] == L]
        table.append({
            "length": L,
            "teacher_tok_s": med([r["teacher"]["tok_s"] for r in rs]),
            "teacher_peak_gb": max(r["teacher"]["peak_gb"] for r in rs),
            "feature_mb": med([r["teacher"]["feature_mb"] for r in rs]),
            "ttft_s": med([r["nospec"]["ttft_s"] for r in rs if "nospec" in r]),
            "decode_tps_nospec": med([r["nospec"]["decode_tps"] for r in rs if "nospec" in r]),
            "decode_tps_dspark": med([r["dspark"]["decode_tps"] for r in rs if "dspark" in r]),
            "tau": med([r["dspark"]["acceptance"].get("tau") for r in rs if "dspark" in r]),
            "decode_peak_gb": max([r["dspark"]["peak_gb"] for r in rs if "dspark" in r], default=None),
        })
    rt.write_json(args.out / "summary.json", {"env": env, "load": load_stats, "settings": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "table": table})
    rt.write_json(args.out / "env.json", env)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
