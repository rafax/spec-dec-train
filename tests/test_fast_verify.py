import mlx.core as mx
import pytest
from mlx import nn

from specdec import fast_verify as fv


def reference(x, w):
    return x.astype(mx.float32) @ w.astype(mx.float32).T


@pytest.mark.parametrize("kernel", ["gemv", "mma"])
@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize(
    ("n", "k", "xdtype", "wdtype"),
    [
        (3072, 2304, mx.bfloat16, mx.bfloat16),  # attention q/gate
        (24, 2304, mx.bfloat16, mx.bfloat16),  # sink projection: N below one tile
        (2304, 3072, mx.bfloat16, mx.bfloat16),  # o_proj
        (1037, 2304, mx.float32, mx.bfloat16),  # verify head: FP32 x BF16, ragged N
        (515, 256, mx.float32, mx.float32),
        (300, 512, mx.float16, mx.float16),
    ],
)
def test_matches_fp32_reference(kernel, rows, n, k, xdtype, wdtype):
    mx.random.seed(rows)
    w = (mx.random.normal((n, k)) * 0.02).astype(wdtype)
    x = mx.random.normal((1, rows, k)).astype(xdtype)
    assert fv.supported(x, w)
    y = fv.small_matmul(x, w, kernel)
    ref = reference(x, w)
    assert y.shape == (1, rows, n)
    assert y.dtype == (mx.float32 if xdtype == mx.float32 else xdtype)
    # Accumulation is FP32 in both; the output is rounded once to its dtype.
    tol = {mx.bfloat16: 2**-7, mx.float16: 2**-10, mx.float32: 1e-5}[y.dtype]
    err = mx.abs(y.astype(mx.float32) - ref) / mx.maximum(mx.abs(ref), 1.0)
    assert err.max().item() <= tol


def test_supported_bounds():
    w = mx.zeros((64, 256), mx.bfloat16)
    assert not fv.supported(mx.zeros((1, 1, 256), mx.bfloat16), w)  # decode keeps MLX GEMV
    assert not fv.supported(mx.zeros((1, 9, 256), mx.bfloat16), w)  # prefill keeps GEMM
    assert not fv.supported(mx.zeros((1, 4, 256), mx.float16), w)  # mixed dtypes
    assert not fv.supported(mx.zeros((1, 4, 200), mx.bfloat16), mx.zeros((64, 200), mx.bfloat16))
    assert fv.supported(mx.zeros((2, 4, 256), mx.bfloat16), w)  # 8 rows across a batch


def test_enable_routes_linear_and_restores():
    layer = nn.Linear(256, 96, bias=False)
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    biased = nn.Linear(256, 96)
    biased.eval()
    x = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)
    stock = layer(x)
    original = nn.Linear.__call__
    fv.enable()
    try:
        fv.STATS.clear()
        fast = layer(x)
        biased(x.astype(mx.float32))
        layer.train()
        layer(x)
        assert fv.STATS == {"mma:8": 1}  # bias and training mode fall back to stock
    finally:
        fv.disable()
    assert nn.Linear.__call__ is original
    assert mx.allclose(fast.astype(mx.float32), stock.astype(mx.float32), atol=0.02).item()
