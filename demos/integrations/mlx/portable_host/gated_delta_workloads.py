"""Public MLX gated-delta workloads for the paths upstream tests only on Metal.

``test_gated_delta_sequential``, ``test_gated_delta_simdgroup``,
``test_gated_delta_nax`` and ``test_gated_delta_vjp`` are skipped without
MLX's Metal backend, so these workloads run their comparisons through
``mx.fast.gated_delta_update`` in both workers, with upstream's criteria:
the forward kernel against the masked fallback (an all-true mask always
falls back) with ``atol = rtol = 1e-4``, and the backward kernel against
``GATED_DELTA_VJP_FALLBACK=1`` with ``atol = rtol = 1e-5``. ``GATED_DELTA_CHUNK``
selects the sequential (0) or eight-step fused chunk (8) forward. Without NAX
a requested 16-step chunk runs the eight-step kernel and the backward is
always sequential, so those cases keep the tighter tolerances of the kernels
they run. The cases use upstream's ``vjp_dims``: one step, 33 steps, and
grouped value heads at two and three per key head. Upstream's 128-sequence
batch and 1,111-step sequence would pass state arrays of up to 67 million
elements through the host's per-dispatch marshalling, so they are left to
native Metal runs of the unchanged tests.

Every environment variable a workload sets is restored afterwards, so the
unchanged tests that follow see upstream's defaults.
"""

import contextlib
import os
import zlib

# (B, Hk, Hv, T, Dk, Dv): upstream's base, unaligned and grouped-head dims.
DIMS = {
    "base": (1, 32, 32, 1, 128, 128),
    "unaligned": (1, 32, 32, 33, 128, 128),
    "diff-heads": (1, 16, 32, 33, 128, 128),
    "diff-heads2": (1, 16, 48, 33, 128, 128),
}
# Forward workloads: GATED_DELTA_CHUNK and the kernel family it selects here.
CHUNKS = {"sequential": ("0", "sequential"), "simdgroup": ("8", "fused")}
# test_gated_delta_nax's request, which resolves to the eight-step kernel.
NAX_DIMS = ("base", "unaligned")
FORWARD_TOLERANCE = 1e-4
VJP_TOLERANCE = 1e-5
VJP_CHUNKS = ("0", "16")


@contextlib.contextmanager
def scoped_env(**values):
    saved = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def cases():
    for name, (chunk, kernel) in CHUNKS.items():
        for dims in DIMS:
            yield {
                "id": f"{name}-{dims}",
                "kind": "forward",
                "chunk": chunk,
                "kernel": kernel,
                "dims": dims,
            }
    for dims in NAX_DIMS:
        yield {
            "id": f"nax-{dims}",
            "kind": "forward",
            "chunk": "16",
            "kernel": "fused",
            "dims": dims,
        }
    for dims in DIMS:
        yield {
            "id": f"vjp-{dims}",
            "kind": "vjp",
            "chunk": None,
            "kernel": "vjp",
            "dims": dims,
        }


def operands(mx, np, case):
    """upstream runner()'s operand distributions, from a per-case seed."""
    B, Hk, Hv, T, Dk, Dv = DIMS[case["dims"]]
    state = np.random.RandomState(zlib.crc32(case["id"].encode("ascii")))
    q = state.normal(size=(B, T, Hk, Dk))
    k = state.normal(size=(B, T, Hk, Dk))
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = state.normal(size=(B, T, Hv, Dv))
    g = state.uniform(size=(B, T, Hv))
    if case["kind"] == "vjp":
        # test_gated_delta_vjp's decays are sigmoids of normals.
        g = 1 / (1 + np.exp(-state.normal(size=(B, T, Hv))))
    b = 1 / (1 + np.exp(-state.normal(size=(B, T, Hv))))
    h0 = state.normal(size=(B, Hv, Dv, Dk))
    arrays = [mx.array(value.astype(np.float32)) for value in (q, k, v, g, b, h0)]
    cotangents = [
        mx.array((state.normal(size=(B, T, Hv, Dv)) * 1e-2).astype(np.float32)),
        mx.array((state.normal(size=(B, Hv, Dv, Dk)) * 1e-2).astype(np.float32)),
    ]
    return arrays, cotangents


def _close(mx, reference, observed, tolerance):
    return bool(mx.allclose(reference, observed, atol=tolerance, rtol=tolerance))


def _error(mx, reference, observed):
    return float(mx.abs(reference - observed).max())


def _forward(mx, case, arrays):
    q, k, v, g, b, h0 = arrays
    B, _, _, T, _, _ = DIMS[case["dims"]]
    # An all-true mask always takes the fallback, as upstream's runner does.
    out_ref, hf_ref = mx.fast.gated_delta_update(
        q, k, v, g, b, initial_state=h0, mask=mx.ones((B, T))
    )
    mx.eval(out_ref, hf_ref)
    with scoped_env(GATED_DELTA_CHUNK=case["chunk"]):
        out, hf = mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0)
        mx.eval(out, hf)
    pairs = [(out_ref, out), (hf_ref, hf)]
    return pairs, FORWARD_TOLERANCE


def _vjp(mx, case, arrays, cotangents):
    def f(q, k, v, g, b, h0):
        return mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0)

    def run(fallback, chunk):
        with scoped_env(
            GATED_DELTA_VJP_FALLBACK="1" if fallback else "0",
            GATED_DELTA_CHUNK_VJP=chunk,
        ):
            outs, vjps = mx.vjp(f, arrays, cotangents)
            mx.eval(outs, vjps)
        return [*outs, *vjps]

    reference = run(True, "0")
    pairs = []
    for chunk in VJP_CHUNKS:
        pairs += list(zip(reference, run(False, chunk)))
    return pairs, VJP_TOLERANCE


def run(mx, np, host, mlx_root):
    """Run every case; return each comparison's error and dispatch count."""
    records = []
    for case in cases():
        arrays, cotangents = operands(mx, np, case)
        mx.eval(*arrays, *cotangents)
        start = host.dispatch_count if host else 0
        if case["kind"] == "forward":
            pairs, tolerance = _forward(mx, case, arrays)
        else:
            pairs, tolerance = _vjp(mx, case, arrays, cotangents)
        record = {
            "id": case["id"],
            "dims": list(DIMS[case["dims"]]),
            "tolerance": tolerance,
            "maximumErrors": [_error(mx, a, b) for a, b in pairs],
            "close": [_close(mx, a, b, tolerance) for a, b in pairs],
            "dispatchCount": host.dispatch_count - start if host else 0,
        }
        records.append(record)
        if not all(record["close"]):
            raise RuntimeError(f"Gated delta {case['id']} differs from its fallback")
    return records


def validate(records, *, native):
    expected = [case["id"] for case in cases()]
    if [record.get("id") for record in records] != expected:
        raise ValueError("Gated-delta workload inventory is incomplete")
    for record in records:
        count = record.get("dispatchCount")
        if (
            not record["close"]
            or not all(record["close"])
            or type(count) is not int
            or (count <= 0 if native else count != 0)
        ):
            raise ValueError(
                f"Gated-delta workload {record['id']} did not execute natively"
            )
