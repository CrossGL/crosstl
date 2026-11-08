"""Metal atomic results received by a variable of another scalar type.

``uint slot = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);`` with an ``atomic_int`` counter converts
the ``int`` result implicitly. CrossGL atomics return the element type of the atomic object and need a receiving
context of that type, so the importer makes the conversion explicit.
"""

import pytest

import crosstl

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

KERNEL = """
struct Args {{ uint indexCount; atomic_int instanceCount; }};
kernel void k(device Args* indirect [[buffer(0)]],
              device atomic_uint* counters [[buffer(1)]],
              device uint* result [[buffer(2)]],
              uint i [[thread_position_in_grid]]) {{
    {body}
}}
"""


def _translate(tmp_path, body, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + KERNEL.format(body=body))
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "uint slot = atomic_fetch_add_explicit(&(indirect[0].instanceCount), 1,"
            " memory_order_relaxed); result[i] = slot;",
            "uint slot = uint(atomicAdd(buffer_load(indirect, 0).instanceCount, 1));",
        ),
        (
            "uint slot; slot = atomic_fetch_sub_explicit(&(indirect[0].instanceCount),"
            " 1, memory_order_relaxed); result[i] = slot;",
            "slot = uint(atomicAdd(buffer_load(indirect, 0).instanceCount, -(1)));",
        ),
        (
            "int seen = atomic_fetch_max_explicit(&counters[1], 3u,"
            " memory_order_relaxed); result[i] = uint(seen);",
            "int seen = int(atomicMax(counters[1], 3u));",
        ),
        (
            "result[i] = atomic_fetch_or_explicit(&(indirect[0].instanceCount), 2,"
            " memory_order_relaxed);",
            "buffer_store(result, i, uint(atomicOr(buffer_load(indirect, 0)"
            ".instanceCount, 2)));",
        ),
    ],
)
def test_atomic_result_converts_to_the_receiving_type(tmp_path, body, expected):
    assert expected in _translate(tmp_path, body, "cgl")


@pytest.mark.parametrize(
    "body",
    [
        "uint same = atomic_fetch_add_explicit(&counters[0], 1u,"
        " memory_order_relaxed); result[i] = same;",
        "auto same = atomic_fetch_add_explicit(&counters[0], 1u,"
        " memory_order_relaxed); result[i] = same;",
    ],
)
def test_atomic_result_of_the_receiving_type_is_not_converted(tmp_path, body):
    cgl = _translate(tmp_path, body, "cgl")
    assert "uint same = atomicAdd(counters[0], 1u);" in cgl


def test_converted_atomic_result_lowers_to_an_interlocked_original(tmp_path):
    body = (
        "uint slot = atomic_fetch_add_explicit(&(indirect[0].instanceCount), 1,"
        " memory_order_relaxed); result[i] = slot;"
    )
    hlsl = _translate(tmp_path, body, "directx")
    assert "int __crossgl_atomic_expr_0;" in hlsl
    assert (
        "InterlockedAdd(indirect[0].instanceCount, 1, __crossgl_atomic_expr_0);" in hlsl
    )
    assert "uint slot = uint(__crossgl_atomic_expr_0);" in hlsl

    glsl = _translate(tmp_path, body, "opengl")
    assert "uint slot = uint(atomicAdd(indirect[0].instanceCount, 1));" in glsl
