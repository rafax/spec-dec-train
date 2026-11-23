# Inkling 2 Mini + DSpark: verified technical notes

Everything here was checked against the checkpoint headers or runtime source on
2026-11-23. Sources: `thinkingmachines/Inkling-2-Mini` @ `7a67ae2`, `mlx-vlm` 0.8.1
(`mlx_vlm/speculative/drafters/inkling2_dspark/dspark.py`, `speculative/dflash.py`,
`models/inkling2/language.py`).

## Checkpoint

| File | Size | Contents |
|---|---|---|
| `model-00001-of-00001.safetensors` | 19.53 GB | target: NVFP4 routed experts (layers 0–19), MXFP8 experts (20–23), BF16 attention/embeddings/LM head, FP32 router + loop gates, vision tower |
| `mtp.safetensors` | 0.636 GB | DSpark drafter, **BF16, 317.9M params**, unprefixed tensor names |

Target text config: hidden 2304, vocab 201,024 (200,058 unpadded), 24 physical layers,
loop `core_start=6, core_end=18, num_passes=3` → **48 layer visits** (0–5 prelude,
6–17 / 18–29 / 30–41 core passes, 42–47 coda). 20 of 24 layers are sliding-window
(1024); layers 5, 11, 17, 23 are global, so 8 of the 48 visits keep full-length KV.
Final logit softcap 30. Default sampling temperature 1.0.

## DSpark drafter (`mtp_config`)

```
num_hidden_layers 3   (layers 0, 2 local/window 1024; layer 1 global)
block_size 8          (1 anchor + 7 proposals)
target_layer_ids [5, 23, 41, 48]
markov_rank 256, markov_head_type rnn_hidden
intermediate_size 9216, 24 q heads / 8 kv heads, head_dim 128, rope_theta 1e6
```

**Target features.** `capture_layer_ids=[5,23,41,48]` taps the residual stream
*entering* visits 5 (last prelude layer), 23 (core pass 2), 41 (last layer of core
pass 3), plus index 48 = the final RMS-normed hidden. Concatenated per token:
4 × 2304 = 9216 values (18 KB/token in BF16).

**Drafter forward (one block):**

1. Block input `x = [embed(anchor), mask_embed × 7]`, using the target's own
   `embed_tokens` + `embed_norm` (the drafter has no embedding of its own).
2. For drafter layer *i*, context features are
   `ctx_i = RMSNorm_i(Σ_j softmax(fuse_weight[i])_j · h_j)`, where `fuse_weight` is
   `[4, 4]` (3 layers + 1 row for the Markov head). Context K/V come from
   `k_proj_ctx`/`v_proj_ctx` (+ `k_norm_ctx`) and are the **only** thing written to
   the persistent drafter KV cache.
3. Block queries attend to all context (window 1024 on local layers) **and the whole
   block bidirectionally** (no causal mask inside the block). RoPE: context at its
   absolute positions; the anchor sits at position `a`.
4. `x → norm → target lm_head` gives base logits for the 7 proposal slots.
5. **Markov head** (low-rank GRU over drafted tokens):
   `f_rnn = RMSNorm(mix_row3(h_{a-1}))`,
   `ctx_k = W_ctx·[x_k ; f_rnn] + b`,
   `[g, c, o] = W_state·[s ; E_markov(prev_k)] + ctx_k`,
   `s ← σ(g)·s + (1−σ(g))·tanh(c)`,
   `logits_k = lm_head(x_k + hidden_out(tanh(o)))`, then softcap 30.
   `prev_1` = anchor; afterwards it is the token just drafted (inference) or the
   ground-truth token (teacher forcing in training).

The runtime folds `lm_head @ hidden_out` once at bind time. Training can use the
unfolded form above, so **Stage 2 only needs the frozen target embedding + LM head
(~1.9 GB BF16), not the 20 GB target.**

**Round alignment** (`_dflash_rounds`): verify input is `[bonus, d1..d7]`; target
features at `[bonus, d1..d_accepted]` become the next round's context; the next
anchor is the new bonus token. For training: **context = features at positions
`< a`, anchor = token `a`, labels = tokens `a+1 … a+7`.**

## Runtime (mlx-vlm 0.8.1)

- Loads the native checkpoint unconverted, with no extraction step:
  `load_drafter("thinkingmachines/Inkling-2-Mini")` → `(model, "dflash")`.
- A fine-tuned drafter = a directory with the target's `config.json` (for
  `mtp_config`) plus a new `mtp.safetensors`.
- `prefer_requested_block_size = True`: the block is fixed at 8 unless
  `draft_block_size` is passed, so there's no adaptive shrinking.
- Per-round stats: `draft.accept_lens` / `draft.draft_lens` (reset per request);
  lifetime counters `speculative_total_rounds` / `speculative_total_accepted`.
- **Gotcha:** `MarkovHead.fold()` caches the folded weights at the first `bind()`.
  After swapping drafter weights in place, set `draft._target = None` so the next
  `reset()` re-folds; otherwise the Markov bias stays stale.
- Activations are floating point (the reference runtime quantizes activations), so
  MLX features differ slightly from features produced by vLLM/SGLang on GPU. This
  matters if Stage 1 ever runs on a GPU while deployment stays on MLX.

## Alternatives considered

- **oMLX ≥ 0.8.1 / 0.8.3**: Inkling 2 Mini + DSpark ("Lightning MTP"), including
  copy-from-context drafting. It's a menu-bar server, so it's harder to instrument for
  per-round acceptance. Worth benchmarking end-to-end as a reference point.
- **mlx-lm PR #2086** (open): target only, no DSpark.
- **nativ-community/Inkling-2-Mini-nvfp4**: patched mlx-vlm 0.8.0 wheel, superseded
  by upstream 0.8.1.
