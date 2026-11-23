"""Thin, instrumented wrapper around mlx-vlm for Inkling 2 Mini + DSpark.

One place that knows how to load the target/drafter, run a single timed
generation from exact token ids, and capture per-round acceptance statistics.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

DEFAULT_MODEL = "thinkingmachines/Inkling-2-Mini"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def resolve_snapshot(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(model_id)
    if path.exists():
        return path
    return Path(snapshot_download(model_id, local_files_only=True))


def env_info(model_id: str = DEFAULT_MODEL) -> dict[str, Any]:
    import mlx_vlm

    snapshot = resolve_snapshot(model_id)
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        chip = platform.processor()
    info = mx.device_info()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
        "mlx_vlm": mlx_vlm.__version__,
        "macos": platform.mac_ver()[0],
        "chip": chip,
        "gpu": info.get("device_name"),
        "memory_gb": round(info.get("memory_size", 0) / 2**30, 1),
        "max_working_set_gb": round(info.get("max_recommended_working_set_size", 0) / 2**30, 1),
        "model": model_id,
        "model_revision": snapshot.name,
    }


def gb(nbytes: int) -> float:
    return round(nbytes / 2**30, 2)


def load_target(model_id: str = DEFAULT_MODEL):
    from mlx_vlm import load

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model, processor = load(model_id)
    stats = {
        "load_s": round(time.perf_counter() - t0, 1),
        "active_gb": gb(mx.get_active_memory()),
        "peak_gb": gb(mx.get_peak_memory()),
    }
    return model, processor, stats


def load_dspark(model_id: str = DEFAULT_MODEL):
    from mlx_vlm.speculative.drafters import load_drafter

    before = mx.get_active_memory()
    t0 = time.perf_counter()
    draft, kind = load_drafter(model_id)
    mx.eval(draft.parameters())
    stats = {
        "load_s": round(time.perf_counter() - t0, 1),
        "kind": kind,
        "added_gb": gb(mx.get_active_memory() - before),
    }
    return draft, kind, stats


def load_tokenizer(model_id: str = DEFAULT_MODEL):
    """The HF tokenizer + chat template; the same object the data pipeline renders with."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(resolve_snapshot(model_id))


def render_messages(tokenizer, messages: list[dict], reasoning_effort=None) -> list[int]:
    """Prompt token ids via Inkling's own chat template (no generation prompt needed)."""
    from specdec.data.render import render

    return render(tokenizer, messages, reasoning_effort=reasoning_effort).input_ids


@dataclass
class GenRecord:
    prompt_tokens: int
    tokens: list[int]
    text: str
    finish_reason: str | None
    ttft_s: float
    decode_tps: float
    e2e_tps: float
    total_s: float
    mlx_prompt_tps: float
    mlx_generation_tps: float
    peak_gb: float
    accept_lens: list[float] = field(default_factory=list)
    draft_lens: list[int] = field(default_factory=list)

    @property
    def n_gen(self) -> int:
        return len(self.tokens)

    def acceptance(self) -> dict[str, Any]:
        """Summaries of per-round acceptance (empty when not speculative)."""
        if not self.draft_lens:
            return {}
        rounds = len(self.draft_lens)
        drafted = sum(self.draft_lens)
        accepted = sum(self.accept_lens)
        depth = max(self.draft_lens)
        # P(position k accepted | drafted), k = 1..depth
        per_pos = []
        for k in range(1, depth + 1):
            eligible = [a for a, d in zip(self.accept_lens, self.draft_lens) if d >= k]
            per_pos.append(round(sum(a >= k for a in eligible) / len(eligible), 4) if eligible else None)
        return {
            "rounds": rounds,
            "mean_accepted": round(accepted / rounds, 3),
            # tokens emitted per target forward = accepted + 1 bonus
            "tau": round((accepted + rounds) / rounds, 3),
            "accept_rate": round(accepted / drafted, 4) if drafted else None,
            "per_position": per_pos,
        }

    def to_json(self, include_tokens: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_tokens:
            d.pop("tokens")
        d["n_gen"] = self.n_gen
        d["acceptance"] = self.acceptance()
        return d


def generate(
    model,
    processor,
    input_ids: list[int],
    *,
    draft=None,
    draft_kind: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int = 0,
    prefill_step_size: int = 256,
    draft_block_size: int | None = None,
    **extra,
) -> GenRecord:
    from mlx_vlm import stream_generate

    mx.random.seed(seed)
    mx.reset_peak_memory()
    kwargs: dict[str, Any] = dict(
        input_ids=mx.array([input_ids]),
        max_tokens=max_tokens,
        temperature=temperature,
        prefill_step_size=prefill_step_size,
        seed=seed,
        **extra,
    )
    if draft is not None:
        kwargs.update(draft_model=draft, draft_kind=draft_kind or "dflash")
        if draft_block_size:
            kwargs["draft_block_size"] = draft_block_size

    tokens, pieces, last = [], [], None
    t0 = time.perf_counter()
    t_first = None
    for res in stream_generate(model, processor, "", **kwargs):
        if res.is_draft:
            continue
        if res.token is not None:
            if t_first is None:
                t_first = time.perf_counter()
            tokens.append(int(res.token))
        pieces.append(res.text)
        last = res
    t_end = time.perf_counter()
    t_first = t_first or t_end
    n = len(tokens)
    return GenRecord(
        prompt_tokens=len(input_ids),
        tokens=tokens,
        text="".join(pieces),
        finish_reason=getattr(last, "finish_reason", None),
        ttft_s=round(t_first - t0, 4),
        decode_tps=round((n - 1) / (t_end - t_first), 2) if n > 1 and t_end > t_first else 0.0,
        e2e_tps=round(n / (t_end - t0), 2) if t_end > t0 else 0.0,
        total_s=round(t_end - t0, 3),
        mlx_prompt_tps=round(getattr(last, "prompt_tps", 0.0), 2),
        mlx_generation_tps=round(getattr(last, "generation_tps", 0.0), 2),
        peak_gb=gb(mx.get_peak_memory()),
        accept_lens=list(getattr(draft, "accept_lens", []) or []) if draft is not None else [],
        draft_lens=list(getattr(draft, "draft_lens", []) or []) if draft is not None else [],
    )


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
