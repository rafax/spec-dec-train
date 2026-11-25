"""Fast 2-8-row matmuls for speculative verify and drafting on M1/M2-class GPUs (#11).

MLX multiplies one activation row by a dense weight with a GEMV that streams the weight at
~290-320 GB/s on an M1 Max. For 2-8 rows it switches to a tiled GEMM that reaches only
45-90 GB/s, so verifying a block of tokens costs far more than decoding one. That hits every
unquantized projection Inkling 2 runs per token: the BF16 attention projections (~2.5 GB
over 48 layer visits), the vocabulary head (an FP32 copy, 1.85 GB) and the whole BF16 DSpark
drafter including its use of the target head. mlx-vlm has small-batch dense kernels, but
gates them on GPU generation >= 15 (M3+) and Inkling 2's model code does not call them.

Two kernels, both reading each weight element once and accumulating in FP32:

* rows 2-4: multi-row GEMV. A simdgroup owns TM output rows; each lane streams a 4-vector
  of every weight row per step and accumulates all rows' dot products; simd_sum finishes.
* rows 5-8: 8x8 simdgroup-matrix MMA. A simdgroup owns NT 8-column output tiles over a
  1/KS slice of K; partial tiles are summed in threadgroup memory. Rows are padded to 8.

The reduction order differs from MLX's GEMV and GEMM, so outputs match stock to rounding,
not bitwise (stock GEMM and GEMV already disagree with each other the same way).

`enable()` routes eligible `nn.Linear` calls (target attention, drafter layers, drafter's
use of the BF16 head) and Inkling 2's verify vocabulary projection through these kernels.
One-row decode keeps MLX's GEMV, so plain decoding is unchanged.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import mlx.core as mx
from mlx import nn

MIN_ROWS, MAX_ROWS = 2, 8
GEMV_MAX_ROWS = 4  # above this the MMA kernel is faster (runs/*-p1-small-matmul)
GEMV_TILE = {"SG": 2, "TM": 4}  # simdgroups per threadgroup, output rows per simdgroup
MMA_TILE = {"NT": 2, "KS": 8}  # 8-column output tiles per simdgroup, K splits per threadgroup

STATS: Counter = Counter()  # calls per path while enabled, to confirm coverage

_GEMV_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint first = (threadgroup_position_in_grid.y * SG + simdgroup_index_in_threadgroup) * TM;
    if (first >= N) return;
    constexpr int KV = K / 4;
    const device vec<TW, 4>* w4 = (const device vec<TW, 4>*)weight;
    const device vec<TX, 4>* x4 = (const device vec<TX, 4>*)x;
    float acc[M][TM];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        #pragma unroll
        for (int r = 0; r < TM; ++r) acc[m][r] = 0.0f;
    }
    for (int kv = lane; kv < KV; kv += 32) {
        float4 xv[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) xv[m] = float4(x4[m * KV + kv]);
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            const float4 w = float4(w4[min(int(first) + r, N - 1) * KV + kv]);
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[m][r] += dot(w, xv[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            const float s = simd_sum(acc[m][r]);
            if (lane == uint(m * TM + r) % 32 && first + r < uint(N)) out[m * N + first + r] = TO(s);
        }
    }
"""

_MMA_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;  // which K slice
    const int n0 = int(threadgroup_position_in_grid.y) * (8 * NT);
    // MLX's 8x8 fragment layout: this lane holds elements (fm, fn) and (fm, fn + 1).
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_matrix<float, 8, 8> acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = simdgroup_matrix<float, 8, 8>(0);
    const int xrow = min(int(fm), M - 1);  // rows past M repeat the last row; never stored
    constexpr int KC = K / KS;
    const int kbeg = int(sg) * KC;
    for (int k = kbeg; k < kbeg + KC; k += 8) {
        simdgroup_matrix<float, 8, 8> a;  // A[m][kk] = x[m][k + kk]
        a.thread_elements()[0] = float(x[xrow * K + k + fn]);
        a.thread_elements()[1] = float(x[xrow * K + k + fn + 1]);
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
            simdgroup_matrix<float, 8, 8> b;  // B[kk][j] = W[n0 + 8t + j][k + kk]
            b.thread_elements()[0] = float(weight[min(n0 + 8 * t + fn, N - 1) * K + k + fm]);
            b.thread_elements()[1] = float(weight[min(n0 + 8 * t + fn + 1, N - 1) * K + k + fm]);
            simdgroup_multiply_accumulate(acc[t], a, b, acc[t]);
        }
    }
    threadgroup float partial[KS][NT][64];
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
        partial[sg][t][fm * 8 + fn] = acc[t].thread_elements()[0];
        partial[sg][t][fm * 8 + fn + 1] = acc[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && fm < M) {
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                float s = 0.0f;
                for (int g = 0; g < KS; ++g) s += partial[g][t][fm * 8 + fn + e];
                const int n = n0 + 8 * t + fn + e;
                if (n < N) out[fm * N + n] = TO(s);
            }
        }
    }
"""


@lru_cache(maxsize=2)
def _kernel(kind: str):
    return mx.fast.metal_kernel(
        name=f"specdec_small_{kind}",
        input_names=["x", "weight"],
        output_names=["out"],
        source=_GEMV_SOURCE if kind == "gemv" else _MMA_SOURCE,
    )


# Output dtype of `x @ w.T` for the dtype pairs the kernels handle.
_OUT_DTYPE = {
    (mx.bfloat16, mx.bfloat16): mx.bfloat16,
    (mx.float16, mx.float16): mx.float16,
    (mx.float32, mx.float32): mx.float32,
    (mx.float32, mx.bfloat16): mx.float32,  # exact BF16 -> FP32 promotion, half the bytes
    (mx.float32, mx.float16): mx.float32,
}


def _rows(x: mx.array) -> int:
    return x.size // x.shape[-1] if x.ndim >= 2 and x.shape[-1] else 0


def supported(x: mx.array, w: mx.array) -> bool:
    """Whether `small_matmul(x, w)` handles this call (otherwise use `x @ w.T`)."""
    return (
        w.ndim == 2
        and x.ndim >= 2
        and x.shape[-1] == w.shape[1]
        and MIN_ROWS <= _rows(x) <= MAX_ROWS
        and (x.dtype, w.dtype) in _OUT_DTYPE
        and w.shape[1] % (8 * MMA_TILE["KS"]) == 0
        and mx.default_device() == mx.gpu
    )


def small_matmul(x: mx.array, w: mx.array, kernel: str | None = None) -> mx.array:
    """`x @ w.T` for 2-8 rows of `x`, reading `w` once. Caller checks `supported`."""
    k, n, rows = w.shape[1], w.shape[0], _rows(x)
    kind = kernel or ("gemv" if rows <= GEMV_MAX_ROWS else "mma")
    out_dtype = _OUT_DTYPE[(x.dtype, w.dtype)]
    common = [("TX", x.dtype), ("TW", w.dtype), ("TO", out_dtype), ("M", rows), ("K", k), ("N", n)]
    if kind == "gemv":
        sg, tm = GEMV_TILE["SG"], GEMV_TILE["TM"]
        template = [*common, ("SG", sg), ("TM", tm)]
        grid, group = (32, sg * -(-n // (sg * tm)), 1), (32, sg, 1)
    else:
        nt, ks = MMA_TILE["NT"], MMA_TILE["KS"]
        template = [*common, ("NT", nt), ("KS", ks)]
        grid, group = (32, ks * -(-n // (8 * nt)), 1), (32, ks, 1)
    STATS[f"{kind}:{rows}"] += 1
    out = _kernel(kind)(
        inputs=[mx.contiguous(x.reshape(rows, k)), w],
        template=template,
        grid=grid,
        threadgroup=group,
        output_shapes=[(rows, n)],
        output_dtypes=[out_dtype],
    )[0]
    return out.reshape(*x.shape[:-1], n)


# ---------------------------------------------------------------------------------------
# Routing

_ORIGINAL: dict[str, object] = {}


def _linear_call(self, x):
    if "bias" not in self and not self.training and supported(x, self["weight"]):
        return small_matmul(x, self["weight"])
    return _ORIGINAL["linear"](self, x)


def _inkling2_logits_from_norm(self, h):
    """Inkling 2's verify head: FP32 activations x BF16 lm_head promoted in-kernel.

    Stock multiplies by a cached FP32 copy of the head (1.85 GB per call); promoting BF16
    to FP32 is exact, so reading the BF16 weight gives the same products at half the bytes.
    """
    head = self.lm_head
    if (
        not self.training
        and type(head) is nn.Linear
        and "bias" not in head
        and getattr(self, "sharding_group", None) is None
    ):
        h32 = h.astype(self.model.embed_norm.weight.dtype).astype(mx.float32)
        if supported(h32, head.weight):
            logits = small_matmul(h32, head.weight)
            cap = self.config.final_logit_softcapping
            if cap:
                logits = mx.tanh(logits / cap) * cap
            return logits[..., : self.config.unpadded_vocab_size]
    return _ORIGINAL["inkling2_logits"](self, h)


def enable() -> None:
    """Route eligible 2-8-row projections through the kernels (idempotent)."""
    from mlx_vlm.models.inkling2.language import LanguageModel

    if _ORIGINAL:
        return
    _ORIGINAL["linear"] = nn.Linear.__call__
    _ORIGINAL["inkling2_logits"] = LanguageModel._logits_from_norm
    nn.Linear.__call__ = _linear_call
    LanguageModel._logits_from_norm = _inkling2_logits_from_norm


def disable() -> None:
    from mlx_vlm.models.inkling2.language import LanguageModel

    if not _ORIGINAL:
        return
    nn.Linear.__call__ = _ORIGINAL.pop("linear")
    LanguageModel._logits_from_norm = _ORIGINAL.pop("inkling2_logits")


def enabled() -> bool:
    return bool(_ORIGINAL)
