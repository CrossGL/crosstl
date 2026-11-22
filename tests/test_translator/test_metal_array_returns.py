"""Metal output of functions that return fixed-size arrays (#3728).

Metal cannot return a C array, so the emitter returns a CrossGL array as ``metal::array<T, N>`` and holds a local that
takes such a result, or that the function returns, as one too. ``array<T, N>`` is an aggregate with no conversion
from a C array or from the pointer a C-array parameter is adjusted to, and none to that pointer either, so
``return x;`` of a parameter, structure member or constant, ``array<T, N> r = x;``, a nested ``array<T, N>``
initialized with nested braces, an ``array<T, N>`` local passed to a C-array parameter and a C array assigned from an
array result used to fail to compile, and a generic function returning its array argument was refused.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

_PROGRAM = """
shader P {{
    struct S {{ float v[2]; float w; }};
    const float K[3] = float[3](1.0, 2.0, 3.0);
    buffer float* outp;
    {functions}
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float v = outp[i];
            float a[2];
            a[0] = v;
            a[1] = v + 1.0;
            {body}
        }}
    }}
}}
"""

_GRID = (
    "float g[2][3];"
    " for (int r = 0; r < 2; r++) { for (int k = 0; k < 3; k++) { g[r][k] = v * float(r * 3 + k + 1); } }"
)

# (functions, entry body, Metal text the output must contain)
_CASES = {
    "parameter": (
        "float[2] same(float x[2]) { return x; }",
        "float c[2] = same(a); outp[i] = c[1];",
        "array<float, 2> same(float x[2]) { return {x[0], x[1]}; }",
    ),
    "nested-parameter": (
        "float[2][3] grid(float x[2][3]) { return x; }",
        f"{_GRID} float h[2][3] = grid(g); outp[i] = h[1][2];",
        "return {{{{x[0][0], x[0][1], x[0][2]}}, {{x[1][0], x[1][1], x[1][2]}}}};",
    ),
    "structure-member": (
        "float[2] member(S s) { return s.v; }",
        "S s; s.v[0] = v; s.v[1] = v * 2.0; s.w = 0.0; float c[2] = member(s); outp[i] = c[1];",
        "array<float, 2> member(S s) { return {s.v[0], s.v[1]}; }",
    ),
    "program-constant": (
        "float[3] table() { return K; }",
        "float c[3] = table(); outp[i] = c[2] + table()[0];",
        "array<float, 3> table() { return {K[0], K[1], K[2]}; }",
    ),
    "generic": (
        "T pick<T>(T x) { return x; }",
        "float c[2] = pick(a); outp[i] = c[1] + pick(a)[0];",
        "array<float, 2> c = pick_float_2(a);",
    ),
    "returned-local-from-parameter": (
        "float[2] scaled(float x[2]) { float r[2] = x; r[1] = r[1] * 2.0; return r; }",
        "float c[2] = scaled(a); outp[i] = c[1];",
        "array<float, 2> r = {x[0], x[1]};",
    ),
    "value-local-argument": (
        "float[2] same(float x[2]) { return x; }"
        " float[2] twice(float x[2]) { float y[2] = same(x); y[0] = y[0] * 2.0; return same(y); }",
        "float c[2] = twice(a); outp[i] = c[0] + c[1];",
        "return same(y.data());",
    ),
    "value-local-inout-argument": (
        "float[2] same(float x[2]) { return x; } void bump(inout float x[2]) { x[1] = x[1] + 10.0; }",
        "float d[2] = same(a); bump(d); outp[i] = d[1];",
        "bump(d.data());",
    ),
    "assignment-from-result": (
        "float[2] same(float x[2]) { return x; }",
        "float c[2]; c = same(a); outp[i] = c[1];",
        "auto __crossgl_array_result_0 = same(a);",
    ),
    "value-assignment": (
        "float[2] same(float x[2]) { return x; }",
        "float d[2] = same(a); d = same(a); outp[i] = d[1];",
        "d = same(a);",
    ),
    "nested-local-argument": (
        "float[2][3] grid(float x[2][3]) { return x; }"
        " float total(float x[2][3]) { return x[1][2] + x[0][0]; } float first(float x[3]) { return x[0]; }",
        f"{_GRID} float h[2][3] = grid(g); outp[i] = total(h) + first(h[1]);",
        "first(h[1].data())",
    ),
    "nested-literal-local": (
        "float[2][2] make(float s) { float r[2][2] = {{s, 1.0}, {2.0, s * 3.0}}; return r; }",
        "float m[2][2] = make(v); outp[i] = m[1][1];",
        "array<array<float, 2>, 2> r = {{{{s, 1.0}}, {{2.0, s * 3.0}}}};",
    ),
}


def _compact(text):
    return "".join(text.split())


def _translate(tmp_path, source, suffix="cgl"):
    path = tmp_path / f"program.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "program.metal").write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "program.metal", "-o", "p.air"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "warning" not in result.stderr, result.stderr[-2000:]


@pytest.mark.parametrize("case", sorted(_CASES))
def test_array_return_compiles(tmp_path, case):
    functions, body, expected = _CASES[case]
    code = _translate(tmp_path, _PROGRAM.format(functions=functions, body=body))
    assert _compact(expected) in _compact(code), code
    _compile(tmp_path, code)


_GLSL = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer B { float data[]; };
float[2] same(float x[2]) { return x; }
vec3[2] rows(vec3 x[2]) { vec3 r[2] = x; r[1] = r[1] * 2.0; return r; }
float[2][2] grid(float s) { float r[2][2] = {{s, 1.0}, {2.0, s * 3.0}}; return r; }
void main() {
    uint i = gl_GlobalInvocationID.x;
    float a[2] = float[2](data[i], data[i] + 1.0);
    float c[2] = same(a);
    vec3 q[2] = vec3[2](vec3(data[i]), vec3(1.0));
    vec3 w[2] = rows(q);
    float m[2][2] = grid(data[i]);
    data[i] = c[1] + w[1].y + w[0].x + m[1][1];
}
"""


def test_glsl_array_returns_compile(tmp_path):
    code = _translate(tmp_path, _GLSL, suffix="comp")
    assert _compact("array<float3, 2> r = {x[0], x[1]};") in _compact(code)
    _compile(tmp_path, code)


@pytest.mark.parametrize(
    "functions, body, construct",
    [
        (
            "float[2][3] grid(float x[2][3]) { return x; } float total(float x[2][3]) { return x[1][2]; }",
            f"{_GRID} outp[i] = total(grid(g));",
            "array argument to 'total'",
        ),
        (
            "float[2] same(float x[2]) { return x; }",
            "S s = S(same(a), 1.0); outp[i] = s.v[1];",
            "array member initializer",
        ),
    ],
    ids=["nested-result-argument", "structure-constructor"],
)
def test_array_results_metal_cannot_copy_once_are_refused(
    tmp_path, functions, body, construct
):
    with pytest.raises(MetalLoweringRefused, match=re.escape(construct)):
        _translate(tmp_path, _PROGRAM.format(functions=functions, body=body))


_PARITY_FUNCTIONS = """
    float[2] same(float x[2]) { return x; }
    float[2][3] grid(float x[2][3]) { return x; }
    float[2] member(S s) { return s.v; }
    float[3] table() { return K; }
    T pick<T>(T x) { return x; }
    float[2] twice(float x[2]) { float y[2] = same(x); y[0] = y[0] * 2.0; return same(y); }
    void bump(inout float x[2]) { x[1] = x[1] + 10.0; }
"""

_PARITY_BODY = f"""
    {_GRID}
    S s;
    s.v[0] = v * 3.0;
    s.v[1] = v * 5.0;
    s.w = 0.0;
    float c[2];
    c = same(a);
    float d[2] = twice(a);
    bump(d);
    float h[2][3] = grid(g);
    float m[2] = member(s);
    float t[3] = table();
    float p[2] = pick(a);
    outp[i] = c[0] + 2.0 * c[1] + 3.0 * d[0] + 5.0 * d[1] + 7.0 * h[1][2] + 11.0 * m[1] + 13.0 * t[2]
        + 17.0 * p[1] + pick(a)[0];
"""


def _expected(v):
    # c = [v, v + 1], d = [2v, v + 11], h[1][2] = 6v, m[1] = 5v, t[2] = 3, p = [v, v + 1].
    return (
        v + 2 * (v + 1) + 3 * (2 * v) + 5 * (v + 11) + 7 * (6 * v) + 11 * (5 * v)
    ) + (13 * 3 + 17 * (v + 1) + v)


def test_array_returns_compute_the_source_results_on_the_apple_gpu(tmp_path):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    code = _translate(
        tmp_path, _PROGRAM.format(functions=_PARITY_FUNCTIONS, body=_PARITY_BODY)
    )
    library, error = device.newLibraryWithSource_options_error_(code, None, None)
    assert library is not None, error
    entry = re.search(r"kernel void\s+(\w+)\(", code).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error

    count = 64
    values = [0.25 * index - 3.0 for index in range(count)]
    data = struct.pack(f"<{count}f", *values)
    buffer = device.newBufferWithBytes_length_options_(
        data, len(data), Metal.MTLResourceStorageModeShared
    )
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(count, 1, 1), Metal.MTLSizeMake(count, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    assert commands.error() is None

    raw = buffer.contents().as_buffer(len(data))
    results = struct.unpack(f"<{count}f", bytes(raw))
    for value, result in zip(values, results):
        assert result == pytest.approx(_expected(value), rel=1e-6, abs=1e-5)
