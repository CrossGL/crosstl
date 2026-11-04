"""Deterministic public MLX Hadamard workloads with independent references.

The upstream ``test_hadamard`` sweep, its CPU comparison and
``test_hadamard_grad_vmap`` run unchanged in ``verify_hadamard``. Upstream's
``test_hadamard_m_only`` is skipped without MLX's Metal backend, so these
workloads cover its ``m``-only transforms, both scales and the non-donated
output, and the layouts the other tests do not reach: every stage combination,
the strided ``n1`` stage, reversed and transposed inputs and batches that need
more than one launch. References apply the Kronecker product of the pinned
``h<m>`` matrix and a Sylvester matrix in float64.
"""

import math
import zlib

from demos.integrations.mlx.portable_host import hadamard_layout

DTYPES = ("float32", "float16", "bfloat16")
# Unit roundoff of each storage type. Every butterfly level, codelet term and
# storage rounding adds at most this times the row's absolute sum, so
# ``bound`` is a worst-case forward error, not a fitted tolerance.
UNIT_ROUNDOFF = {"float32": 2.0**-24, "float16": 2.0**-11, "bfloat16": 2.0**-8}


def cases():
    for size in (2, 4, 16, 256, 8192, 12, 20, 28, 24, 40, 56, 28 * 256):
        for dtype in DTYPES:
            yield {
                "id": f"rows-{size}-{dtype}",
                "shape": [3, size],
                "dtype": dtype,
                "layout": "contiguous",
                "scale": None,
            }
    # The n1 stage only runs above 8192: n = 2^14 = 128 * 128.
    for size in (16384, 12 * 16384, 28 * 16384):
        yield {
            "id": f"strided-{size}",
            "shape": [2, size],
            "dtype": "float32",
            "layout": "contiguous",
            "scale": 0.5,
        }
    for m in (12, 20, 28):
        for shape in ([m], [4, m], [3, 5, m]):
            for scale in (None, 0.25):
                # A temporary input is donated; a named one keeps its buffer.
                for layout in ("contiguous", "donated"):
                    yield {
                        "id": f"m-only-{m}-{len(shape)}-{scale}-{layout}",
                        "shape": shape,
                        "dtype": "float32",
                        "layout": layout,
                        "scale": scale,
                    }
    for layout in ("reversed", "transposed", "broadcast"):
        yield {
            "id": f"{layout}-64",
            "shape": [5, 64],
            "dtype": "float32",
            "layout": layout,
            "scale": 0.125,
        }
    # 65,536 rows of two exceed one launch's 65,535 workgroups.
    yield {
        "id": "batched-2",
        "shape": [65536, 2],
        "dtype": "float32",
        "layout": "contiguous",
        "scale": None,
    }


def sylvester(np, n):
    matrix = np.ones((1, 1))
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def sylvester_transform(np, values):
    """``H_n @ x`` along the last axis for Sylvester's ``H_n``, in float64.

    ``H_2n = [[H_n, H_n], [H_n, -H_n]]``, so each butterfly pass forms the sum
    and difference of the halves of every ``2h`` block. The explicit matrix of
    a 28 * 2^14 transform would hold 1.7 TB.
    """
    result = values.astype(np.float64)
    n = result.shape[-1]
    if n & (n - 1):
        raise ValueError("Sylvester transforms need a power-of-two length")
    h = 1
    while h < n:
        view = result.reshape(*result.shape[:-1], n // (2 * h), 2, h)
        low = view[..., 0, :].copy()
        view[..., 0, :] += view[..., 1, :]
        view[..., 1, :] = low - view[..., 1, :]
        h *= 2
    return result


def codelet_matrix(np, matrices, m):
    """The pinned ``m`` x ``m`` codelet, checked to be a Hadamard matrix."""
    rows = matrices[m].strip().split("\n")
    codelet = np.array([[1.0 if c == "+" else -1.0 for c in row] for row in rows])
    # The kernel's codelet and this reference parse the same pinned text,
    # so check independently that it is an m x m Hadamard matrix.
    if codelet.shape != (m, m) or not np.array_equal(
        codelet @ codelet.T, m * np.eye(m)
    ):
        raise ValueError(f"Pinned h{m} is not a Hadamard matrix")
    return codelet


def reference(np, matrices, values, scale):
    """``scale * (H_m kron H_n) @ x`` along the last axis, in float64.

    Element ``a * n + c`` of a row is codelet row ``a``, Sylvester column
    ``c``: the product transforms each of the ``m`` length-``n`` slices, then
    mixes the slices by the codelet.
    """
    size = values.shape[-1]
    n, m = hadamard_layout.decompose(size)
    slices = values.reshape(*values.shape[:-1], m, n)
    result = sylvester_transform(np, slices)
    if m > 1:
        result = np.einsum("ab,...bn->...an", codelet_matrix(np, matrices, m), result)
    return result.reshape(values.shape) * scale


def bound_for(np, source, dtype, scale):
    """The worst-case error of ``scale * H @ x`` in ``dtype`` storage."""
    n, m = hadamard_layout.decompose(source.shape[-1])
    depth = math.log2(n) + m + 3
    absolute = float(np.abs(source.astype(np.float64)).sum(axis=-1).max())
    return depth * UNIT_ROUNDOFF[dtype] * abs(scale) * absolute


def operand(mx, np, case):
    shape = case["shape"]
    state = np.random.RandomState(zlib.crc32(case["id"].encode("ascii")))
    values = state.normal(size=shape).astype(np.float32)
    array = mx.array(values).astype(getattr(mx, case["dtype"]))
    layout = case["layout"]
    if layout == "reversed":
        array = array[:, ::-1]
    elif layout == "transposed":
        array = mx.array(values.T.copy()).astype(getattr(mx, case["dtype"])).T
    elif layout == "broadcast":
        array = mx.broadcast_to(array[:1], shape)
    return array


def launches(case):
    """Each Hadamard entry the case dispatches and how many launches it takes."""
    size = case["shape"][-1]
    transforms = math.prod(case["shape"]) // size
    batch = hadamard_layout.batch_limit(size)
    return {
        entry: -(-transforms // batch)
        for entry in hadamard_layout.entries(size, case["dtype"])
    }


def run(mx, np, host, mlx_root):
    """Run every case through ``mx.hadamard_transform``; return the errors."""
    from demos.integrations.mlx.portable_host.hadamard_packages import matrices

    matrices = matrices(mlx_root)
    records = []
    for case in cases():
        array = operand(mx, np, case)
        mx.eval(array)
        source = np.array(array.astype(mx.float32))
        kwargs = {} if case["scale"] is None else {"scale": case["scale"]}
        start = host.dispatch_count if host else 0
        if case["layout"] == "donated":
            # Only the transform references the product, so MLX donates it.
            result = mx.hadamard_transform(array * 1.0, **kwargs)
        else:
            result = mx.hadamard_transform(array, **kwargs)
        mx.eval(result)
        if result.shape != array.shape or result.dtype != array.dtype:
            raise RuntimeError(f"Hadamard {case['id']} changed its shape or type")
        size = case["shape"][-1]
        scale = case["scale"] if case["scale"] is not None else size**-0.5
        expected = reference(np, matrices, source, scale)
        observed = np.array(result.astype(mx.float32)).astype(np.float64)
        bound = bound_for(np, source, case["dtype"], scale)
        error = float(np.abs(observed - expected).max())
        record = {
            "id": case["id"],
            "dtype": case["dtype"],
            "shape": case["shape"],
            "layout": case["layout"],
            "maximumError": error,
            "bound": bound,
            "dispatchCount": host.dispatch_count - start if host else 0,
        }
        records.append(record)
        if not error <= bound:
            raise RuntimeError(f"Hadamard {case['id']} exceeds its reference bound")
    return records


def validate(records, *, native):
    expected = [case["id"] for case in cases()]
    if [record.get("id") for record in records] != expected:
        raise ValueError("Hadamard workload inventory is incomplete")
    for record in records:
        count = record.get("dispatchCount")
        if (
            not record["maximumError"] <= record["bound"]
            or type(count) is not int
            or (count <= 0 if native else count != 0)
        ):
            raise ValueError(
                f"Hadamard workload {record['id']} did not execute natively"
            )
