"""Frozen-target teacher pass: chunked prefill that captures DSpark context features
and top-k teacher log-probs. This is the Stage 1 inner loop (and the context probe).

Calls the language model exactly like mlx-vlm's own chunked prefill does for text
(``embed`` -> ``language_model`` with a shared per-visit cache), so the captured
features are what the runtime feeds the drafter.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import mlx.core as mx

# Residual entering visits 5, 23, 41 and the final normed hidden (index 48).
CAPTURE_IDS = (5, 23, 41, 48)


@dataclass
class TeacherChunk:
    start: int
    features: mx.array  # [n, len(CAPTURE_IDS), hidden] in the model dtype (BF16)
    topk_ids: mx.array  # [n, k] int32, sorted by descending logprob
    topk_logprobs: mx.array  # [n, k] float32 (log-softmax at T=1, after softcap)

    @property
    def n(self) -> int:
        return self.features.shape[0]


def language_model(model):
    return model.language_model if hasattr(model, "language_model") else model


def teacher_forward(
    model,
    ids: list[int],
    *,
    chunk: int = 256,
    topk: int = 32,
    capture_ids: tuple[int, ...] = CAPTURE_IDS,
) -> Iterator[TeacherChunk]:
    """Teacher-force ``ids`` through the frozen target, yielding evaluated chunks.

    Row ``i`` of a chunk holds the features at position ``start + i`` and the
    teacher's distribution over the token at ``start + i + 1``.
    """
    lm = language_model(model)
    cache = lm.make_cache()
    for start in range(0, len(ids), chunk):
        x = mx.array([ids[start : start + chunk]])
        out = lm(inputs=x, cache=cache, capture_layer_ids=list(capture_ids))
        features = mx.stack([h[0] for h in out.hidden_states], axis=1)
        logits = out.logits[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        part = mx.argpartition(-logprobs, kth=topk - 1, axis=-1)[:, :topk]
        vals = mx.take_along_axis(logprobs, part, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        topk_ids = mx.take_along_axis(part, order, axis=-1).astype(mx.int32)
        topk_lp = mx.take_along_axis(vals, order, axis=-1)
        mx.eval(features, topk_ids, topk_lp, [c.state for c in cache])
        yield TeacherChunk(start, features, topk_ids, topk_lp)
        del out, logits, logprobs
        mx.clear_cache()
