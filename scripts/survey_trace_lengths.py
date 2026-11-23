"""Survey token lengths of agent trajectories rendered with Inkling 2's chat template.

Reads Open-SWE-Traces parquet files (downloading them first if given as repo
paths), renders every trajectory with Inkling's own template, and reports how
much assistant-token training signal survives under several context caps:

* fits    - whole trajectory <= cap
* prefix  - assistant tokens inside the first `cap` tokens (prefix truncation)

Per-trace records go to --out (jsonl) so later stages can select by length.

Usage:
    uv run python -I scripts/survey_trace_lengths.py --files-json files.json --out data/surveys/open_swe_v2.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from specdec.data.render import RenderError, render

CAPS = (4096, 8192, 16384, 32768)
REPO = "nvidia/Open-SWE-Traces"


def group_of(path: str) -> str:
    # data/<agent>/<teacher>/<source>[/<thinking>]/train-xxxxx.parquet
    return "/".join(path.split("/")[1:-1])


def survey_row(tokenizer, row) -> dict:
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    rendered = render(tokenizer, messages, row.get("tools"))
    mask = rendered.assistant_mask
    prefix = {}
    for cap in CAPS:
        prefix[str(cap)] = sum(mask[:cap])
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    return {
        "instance_id": row.get("instance_id"),
        "trajectory_id": row.get("trajectory_id"),
        "repo": row.get("repo"),
        "language": row.get("language"),
        "resolved": row.get("resolved"),
        "n_tokens": rendered.n_tokens,
        "n_assistant": rendered.n_assistant,
        "n_messages": len(messages),
        "n_assistant_spans": len(rendered.assistant_spans()),
        "prefix_assistant": prefix,
        "teacher": (meta.get("teacher_model") or {}).get("name"),
    }


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def summarize(name, recs):
    lens = [r["n_tokens"] for r in recs]
    if not lens:
        return
    asst = sum(r["n_assistant"] for r in recs) / sum(lens)
    line = (
        f"{name:58s} n={len(recs):5d} p25={pct(lens, .25):6,} med={pct(lens, .5):6,} "
        f"p75={pct(lens, .75):6,} asst={asst:4.0%} |"
    )
    for cap in CAPS:
        fits = sum(r["n_tokens"] <= cap for r in recs) / len(recs)
        line += f" {cap // 1024:>2}k:{fits:4.0%}"
    print(line)


def summarize_caps(recs):
    print("\nTraining signal per cap (mean assistant tokens per trace):")
    for cap in CAPS:
        fit = [r for r in recs if r["n_tokens"] <= cap]
        fit_asst = statistics.mean([r["n_assistant"] for r in fit]) if fit else 0
        prefix = statistics.mean([r["prefix_assistant"][str(cap)] for r in recs])
        print(
            f"  cap {cap:>6,}: {len(fit) / len(recs):5.1%} of traces fit whole "
            f"(mean {fit_asst:5.0f} asst tok) | prefix-truncated: {prefix:5.0f} asst tok/trace"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-json", type=Path, required=True, help="JSON list of repo-relative parquet paths")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/open-swe-traces"))
    ap.add_argument("--max-rows-per-file", type=int, default=400)
    ap.add_argument("--tokenizer", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--out", type=Path, default=Path("data/surveys/open_swe_v2.jsonl"))
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    paths = json.loads(args.files_json.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_recs, by_group, by_agent = [], collections.defaultdict(list), collections.defaultdict(list)
    with args.out.open("w") as out:
        for path in paths:
            local = hf_hub_download(REPO, path, repo_type="dataset", local_dir=args.raw_dir)
            table = pq.read_table(local)
            idx = list(range(table.num_rows))
            random.Random(path).shuffle(idx)
            errors = 0
            for i in idx[: args.max_rows_per_file]:
                row = table.slice(i, 1).to_pylist()[0]
                try:
                    rec = survey_row(tokenizer, row)
                except (RenderError, KeyError, TypeError, ValueError) as exc:
                    errors += 1
                    if errors <= 3:
                        print(f"  skip {path}#{i}: {str(exc)[:120]}", file=sys.stderr)
                    continue
                rec.update(file=path, row=i, group=group_of(path))
                out.write(json.dumps(rec) + "\n")
                all_recs.append(rec)
                by_group[rec["group"]].append(rec)
                by_agent[rec["group"].split("/")[0]].append(rec)
            print(f"  {path}: {table.num_rows} rows, kept {min(len(idx), args.max_rows_per_file) - errors}, errors {errors}", file=sys.stderr)

    print("\nPer group (tokens under Inkling template; % of traces that fit whole at each cap)")
    for g in sorted(by_group):
        summarize(g, by_group[g])
    print("\nPer agent scaffold")
    for a in sorted(by_agent):
        summarize(a, by_agent[a])
    summarize("ALL", all_recs)
    summarize_caps(all_recs)


if __name__ == "__main__":
    main()
