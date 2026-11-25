"""Kernel variants for 8-row x @ W.T (scratch)."""
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(0, "/Users/rafal/.t3/projects/spec-dec-train/scripts")
from bench_small_matmul import make_workloads  # noqa: E402

from specdec import fast_verify as fv  # noqa: E402

# --- MMA: each simdgroup owns NT 8-column output tiles; K split over KS simdgroups ---------
MMA_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;   // 0..KS-1, splits K
    const int n0 = int(threadgroup_position_in_grid.y) * (8 * NT);
    // MLX fragment layout: lane holds (fm, fn) and (fm, fn + 1) of an 8x8 tile.
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_matrix<float, 8, 8> acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = simdgroup_matrix<float, 8, 8>(0);
    const int xrow = min(int(fm), M - 1);
    constexpr int KC = K / KS;
    const int kbeg = int(sg) * KC;
    for (int k = kbeg; k < kbeg + KC; k += 8) {
        simdgroup_matrix<float, 8, 8> a;
        // A[m][kk] = x[m][k + kk]: lane holds A[fm][fn], A[fm][fn+1]
        a.thread_elements()[0] = float(x[xrow * K + k + fn]);
        a.thread_elements()[1] = float(x[xrow * K + k + fn + 1]);
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
            // B[kk][j] = W[n0 + 8t + j][k + kk]: lane holds B[fm][fn], B[fm][fn+1]
            const int n = min(n0 + 8 * t + fn, N - 1);
            const int n1 = min(n0 + 8 * t + fn + 1, N - 1);
            simdgroup_matrix<float, 8, 8> b;
            b.thread_elements()[0] = float(weight[n * K + k + fm]);
            b.thread_elements()[1] = float(weight[n1 * K + k + fm]);
            simdgroup_multiply_accumulate(acc[t], a, b, acc[t]);
        }
    }
    // reduce the KS partial tiles through threadgroup memory
    threadgroup float red[KS][NT][64];
    for (int t = 0; t < NT; ++t) {
        red[sg][t][fm * 8 + fn] = acc[t].thread_elements()[0];
        red[sg][t][fm * 8 + fn + 1] = acc[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int t = 0; t < NT; ++t) {
            for (int e = 0; e < 2; ++e) {
                const int col = fn + e;
                float s = 0;
                for (int g = 0; g < KS; ++g) s += red[g][t][fm * 8 + col];
                const int n = n0 + 8 * t + col;
                if (fm < M && n < N) out[fm * N + n] = TO(s);
            }
        }
    }
"""

MMA = mx.fast.metal_kernel(name="mma_small", input_names=["x", "weight"], output_names=["out"],
                           source=MMA_SRC)


def mma(nt, ks):
    def fn(x, w):
        k, n = w.shape[1], w.shape[0]
        if k % (8 * ks):
            return None
        rows = x.size // k
        od = fv._OUT_DTYPE[(x.dtype, w.dtype)]
        return MMA(inputs=[x.reshape(rows, k), w],
                   template=[("TX", x.dtype), ("TW", w.dtype), ("TO", od), ("M", rows), ("K", k),
                             ("N", n), ("NT", nt), ("KS", ks)],
                   grid=(32, ks * ((n + 8 * nt - 1) // (8 * nt)), 1), threadgroup=(32, ks, 1),
                   output_shapes=[(rows, n)], output_dtypes=[od])[0]
    return fn


# --- GEMV variant: 16-byte BF16 weight loads (8 values), FP32 x -------------------------------
G16_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint first = (threadgroup_position_in_grid.y * SG + simdgroup_index_in_threadgroup) * TM;
    if (first >= N) return;
    constexpr int K8 = K / 8;
    const device uint4* w8 = (const device uint4*)weight;
    const device float4* x4 = (const device float4*)x;
    float acc[M][TM];
    for (int m = 0; m < M; ++m) for (int r = 0; r < TM; ++r) acc[m][r] = 0.0f;
    for (int kv = lane; kv < K8; kv += 32) {
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int n = min(int(first) + r, N - 1);
            const uint4 u = w8[n * K8 + kv];
            const float4 lo = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u),
                                     as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
            const float4 hi = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xffff0000u),
                                     as_type<float>(u.w << 16), as_type<float>(u.w & 0xffff0000u));
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m][r] += dot(lo, x4[m * (K / 4) + 2 * kv]) + dot(hi, x4[m * (K / 4) + 2 * kv + 1]);
            }
        }
    }
    for (int m = 0; m < M; ++m) for (int r = 0; r < TM; ++r) {
        const float s = simd_sum(acc[m][r]);
        if (lane == uint(m * TM + r) % 32 && first + r < uint(N)) out[m * N + first + r] = TO(s);
    }
"""
G16 = mx.fast.metal_kernel(name="g16_small", input_names=["x", "weight"], output_names=["out"],
                           source=G16_SRC)


def g16(sg, tm):
    def fn(x, w):
        if w.dtype != mx.bfloat16:
            return None
        k, n = w.shape[1], w.shape[0]
        rows = x.size // k
        od = fv._OUT_DTYPE[(x.dtype, w.dtype)]
        xf = x.reshape(rows, k).astype(mx.float32)
        per = sg * tm
        return G16(inputs=[xf, w], template=[("TO", od), ("M", rows), ("K", k), ("N", n),
                                             ("SG", sg), ("TM", tm)],
                   grid=(32, sg * ((n + per - 1) // per), 1), threadgroup=(32, sg, 1),
                   output_shapes=[(rows, n)], output_dtypes=[od])[0]
    return fn


def base(cfg, xf32=False):
    def fn(x, w):
        if xf32 and x.dtype == mx.bfloat16:
            od = mx.bfloat16
            y = fv.small_matmul(x.astype(mx.float32), w, cfg)
            return y.astype(od)
        return fv.small_matmul(x, w, cfg)
    return fn


def main():
    mx.random.seed(0)
    wl = make_workloads()
    methods = {"stock": lambda x, w: x @ w.T,
               "base_sg4_tm2": base({"SG": 4, "TM": 2, "TV": 1}),
               "base_sg2_tm4": base({"SG": 2, "TM": 4, "TV": 1}),
               "base_xf32_sg4_tm2": base({"SG": 4, "TM": 2, "TV": 1}, True)}
    for sg, tm in ((4, 2), (2, 4), (4, 4), (8, 1), (8, 2)):
        methods[f"g16_sg{sg}_tm{tm}"] = g16(sg, tm)
    for nt in (1, 2, 4):
        for ks in (1, 2, 4, 8):
            methods[f"mma_nt{nt}_ks{ks}"] = mma(nt, ks)
    for m in (int(a) for a in sys.argv[1:] or ["8"]):
        for name_w, (weights, xd) in wl.items():
            nbytes = sum(w.nbytes for w in weights)
            xs = {k: mx.random.normal((1, m, k)).astype(xd) for k in {w.shape[1] for w in weights}}
            ref = [(xs[w.shape[1]].astype(mx.float32) @ w.astype(mx.float32).T) for w in weights[:3]]
            mx.eval(xs, ref)
            print(f"\n{name_w} rows={m} ({nbytes / 2**30:.2f} GiB)")
            rows = []
            for name, fn in methods.items():
                outs = [fn(xs[w.shape[1]], w) for w in weights]
                if any(o is None for o in outs):
                    continue
                mx.eval(outs)
                err = max(mx.abs(o.reshape(r.shape).astype(mx.float32) - r).max().item()
                          for o, r in zip(outs, ref))
                ts = []
                for _ in range(7):
                    t0 = time.perf_counter()
                    mx.eval([fn(xs[w.shape[1]], w) for w in weights])
                    ts.append(time.perf_counter() - t0)
                ms = statistics.median(ts) * 1000
                rows.append((ms, name, err))
            for ms, name, err in sorted(rows):
                print(f"  {name:22s} {ms:7.2f} ms {nbytes / ms / 1e6:6.0f} GB/s  maxerr {err:.3g}")


main()
