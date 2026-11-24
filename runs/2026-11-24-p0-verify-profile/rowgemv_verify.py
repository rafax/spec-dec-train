"""Verify forward timing with BF16 Linear 2..8-row calls split into per-row GEMVs."""
import json, statistics, time
from pathlib import Path
import mlx.core as mx, mlx.nn as nn
from mlx_vlm.speculative.cache_state import abort_speculative_round
from specdec import runtime as rt
from specdec.teacher import CAPTURE_IDS, language_model

tok = rt.load_tokenizer(); model, _, _ = rt.load_target(); lm = language_model(model)
lines = [json.loads(l) for l in Path("prompts/coding_smoke.jsonl").read_text().splitlines() if l.strip()]
ids = []
while len(ids) < 1040:
    for p in lines: ids += rt.render_messages(tok, p["messages"])
ctx, cont = ids[:1024], ids[1024:1040]
cache = lm.make_cache()
for s in range(0, 1024, 256):
    lm(inputs=mx.array([ctx[s:s+256]]), cache=cache); mx.eval([c.state for c in cache])

def verify(t):
    out = lm(inputs=mx.array([cont[:t]]), cache=cache, capture_layer_ids=list(CAPTURE_IDS), speculative_verify=True)
    mx.eval(out.logits, out.hidden_states); abort_speculative_round(out.gdn_states)
    return out.logits

def med(t, n=15):
    verify(t); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); verify(t); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

orig = nn.Linear.__call__
def rowwise(self, x):
    if x.ndim >= 2 and 2 <= x.shape[-2] <= 8 and self.weight.dtype == mx.bfloat16 and "bias" not in self:
        rows = [x[..., t:t+1, :] @ self.weight.T for t in range(x.shape[-2])]
        return mx.concatenate(rows, axis=-2)
    return orig(self, x)

base = {t: med(t) for t in (1, 2, 3, 4, 8)}
ref = {t: verify(t) for t in (2, 8)}
nn.Linear.__call__ = rowwise
# lm_head may be called via a non-Linear path; report whether patched calls happen
patched = {t: med(t) for t in (1, 2, 3, 4, 8)}
new = {t: verify(t) for t in (2, 8)}
nn.Linear.__call__ = orig
for t in (1, 2, 3, 4, 8):
    print(f"T={t}: stock {base[t]:6.1f} ms -> per-row GEMV {patched[t]:6.1f} ms ({base[t]/patched[t]:4.2f}x)")
for t in (2, 8):
    a, b = ref[t].astype(mx.float32), new[t].astype(mx.float32)
    print(f"T={t}: argmax equal {bool(mx.all(a.argmax(-1) == b.argmax(-1)))}, max |dlogit| {float(mx.abs(a-b).max()):.4f}")
