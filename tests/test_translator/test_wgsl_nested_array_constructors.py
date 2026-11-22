"""WGSL output writes a CrossGL constructor of an array of arrays as nested WGSL array value constructors.

``float[2][3](float[3](...), float[3](...))`` constructs two elements, each a ``float[3]`` (GLSL 4.60 section 4.1.9),
so WGSL output writes ``array<array<f32, 3>, 2>(array<f32, 3>(...), array<f32, 3>(...))``. It used to copy the outer
callee as written, ``float[2][3](...)``, which naga rejects.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import FunctionCallNode, IdentifierNode, LiteralNode
from crosstl.translator.codegen.wgsl_array_constructors import wgsl_array_constructor
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.refusals import WGSLRefusal

PROGRAM = """shader Probe {{
    struct Pair {{ float a; int b; }};
    RWStructuredBuffer<float> fin @set(0) @binding(0);
    RWStructuredBuffer<int> iin @set(0) @binding(1);
    RWStructuredBuffer<float> outp @set(0) @binding(2);
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        const int N = 3;
        float corner(float m[2][3], int r) {{ return m[r][2]; }}
        void main(uvec3 id @ gl_GlobalInvocationID) {{
            uint i = id.x;
{body}
        }}
    }}
}}
"""


def _naga(tmp_path, code, *outputs):
    if not shutil.which("naga"):
        pytest.skip("naga is not installed")
    (tmp_path / "out.wgsl").write_text(code)
    checked = subprocess.run(
        ["naga", "out.wgsl", *map(str, outputs)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def _translate(tmp_path, body):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM.format(body=body))
    return crosstl.translate(str(path), backend="wgsl", format_output=False)


def test_two_dimensional_constructor(tmp_path):
    code = _translate(
        tmp_path,
        "float a[2][3] = float[2][3](float[3](0.0, 1.0, 2.0), float[3](3.0, 4.0, 5.0));\n"
        "outp[i] = a[1][2];",
    )
    assert (
        "var a: array<array<f32, 3>, 2> = array<array<f32, 3>, 2>("
        "array<f32, 3>(0.0, 1.0, 2.0), array<f32, 3>(3.0, 4.0, 5.0));" in code
    )
    assert "float[" not in code
    _naga(tmp_path, code)


def test_three_dimensional_constructor(tmp_path):
    code = _translate(
        tmp_path,
        "float b[2][2][2] = float[2][2][2]("
        "float[2][2](float[2](1.0, 2.0), float[2](3.0, 4.0)), "
        "float[2][2](float[2](5.0, 6.0), float[2](7.0, 8.0)));\n"
        "outp[i] = b[1][1][0];",
    )
    assert (
        "array<array<array<f32, 2>, 2>, 2>(array<array<f32, 2>, 2>(array<f32, 2>(1.0, 2.0), "
        "array<f32, 2>(3.0, 4.0)), array<array<f32, 2>, 2>(array<f32, 2>(5.0, 6.0), "
        "array<f32, 2>(7.0, 8.0)))" in code
    )
    _naga(tmp_path, code)


def test_constant_sized_short_vector_and_struct_constructors(tmp_path):
    code = _translate(
        tmp_path,
        "float d[2][N] = float[2][N](float[N](1.0, 2.0, 3.0), float[N](4.0, 5.0, 6.0));\n"
        "float e[3][2] = float[3][2](float[2](1.0, 2.0));\n"
        "vec2 v[2][1] = vec2[2][1](vec2[1](vec2(1.0, 2.0)), vec2[1](vec2(3.0, 4.0)));\n"
        "Pair s[1][2] = Pair[1][2](Pair[2](Pair(1.0, 2), Pair(3.0, 4)));\n"
        "outp[i] = d[1][1] + e[0][1] + e[2][1] + v[1][0].y + s[0][1].a;",
    )
    assert (
        "array<array<f32, N>, 2>(array<f32, N>(1.0, 2.0, 3.0), array<f32, N>(4.0, 5.0, 6.0))"
        in code
    )
    # The elements a short constructor leaves out are zero values.
    assert (
        "array<array<f32, 2>, 3>(array<f32, 2>(1.0, 2.0), array<f32, 2>(), array<f32, 2>())"
        in code
    )
    assert (
        "array<array<vec2<f32>, 1>, 2>(array<vec2<f32>, 1>(vec2<f32>(1.0, 2.0)), "
        "array<vec2<f32>, 1>(vec2<f32>(3.0, 4.0)))" in code
    )
    assert (
        "array<array<Pair, 2>, 1>(array<Pair, 2>(Pair(1.0, 2), Pair(3.0, 4)))" in code
    )
    _naga(tmp_path, code)


def test_constructor_as_an_argument_and_a_subscripted_value(tmp_path):
    code = _translate(
        tmp_path,
        "outp[i] = corner(float[2][3](float[3](1.0, 2.0, 3.0), float[3](4.0, 5.0, 6.0)), 1)\n"
        "    + float[2][3](float[3](1.0, 2.0, 3.0), float[3](4.0, 5.0, 6.0))[1][0];",
    )
    nested = "array<array<f32, 3>, 2>(array<f32, 3>(1.0, 2.0, 3.0), array<f32, 3>(4.0, 5.0, 6.0))"
    assert f"corner({nested}, 1)" in code
    assert f"{nested}[1][0]" in code
    _naga(tmp_path, code)


def test_constructor_longer_than_its_first_dimension_is_refused(tmp_path):
    with pytest.raises(Exception, match="expects at most 1 elements, got 2"):
        _translate(
            tmp_path,
            "float a[1][2] = float[1][2](float[2](1.0, 2.0), float[2](3.0, 4.0));\n"
            "outp[i] = a[0][0];",
        )


def test_unsized_inner_dimension_is_refused():
    codegen = WGSLCodeGen()
    codegen._structs_by_name = {}
    call = FunctionCallNode(
        IdentifierNode("float[2][]"), [LiteralNode(1.0, None), LiteralNode(2.0, None)]
    )
    with pytest.raises(WGSLRefusal, match="only its first dimension may be unsized"):
        wgsl_array_constructor(codegen, call)


def _vulkan_runtime():
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    for runtime in ("vulkan-lavapipe", "vulkan-moltenvk"):
        try:
            provisioned = provision(runtime)
        except Exception:
            continue
        if provisioned.available and provisioned.mode == "host":
            env = dict(provisioned.env)
            if env.get("CROSSTL_CONFORMANCE_VKRUN"):
                return env
    return None


def test_two_dimensional_constructor_computes_its_elements(tmp_path):
    env = _vulkan_runtime()
    if env is None:
        pytest.skip("no host Vulkan runtime with vkrun")
    code = _translate(
        tmp_path,
        "float a[2][3] = float[2][3](float[3](0.0, 1.0, 2.0), float[3](3.0, 4.0, 5.0));\n"
        "a[(iin[i] & 1)][2] = fin[i];\n"
        "outp[i] = a[1][2] * 100.0 + a[0][2] * 10.0 + a[1][0];",
    )
    binary = tmp_path / "out.spv"
    _naga(tmp_path, code, binary)
    count = 4
    fin = [float(7 + n) for n in range(count)]
    iin = [*range(count)]
    files = [tmp_path / name for name in ("fin.bin", "iin.bin", "outp.bin")]
    files[0].write_bytes(struct.pack(f"<{count}f", *fin))
    files[1].write_bytes(struct.pack(f"<{count}i", *iin))
    files[2].write_bytes(bytes(4 * count))
    run = subprocess.run(
        [
            env["CROSSTL_CONFORMANCE_VKRUN"],
            str(binary),
            "compute_main",
            str(count),
            "1",
            "1",
        ]
        + [f"{binding}:inout:{path}" for binding, path in enumerate(files)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
    )
    assert run.returncode == 0 and run.stdout.startswith("ok"), run.stdout + run.stderr

    def expected(value, row):
        a = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        a[row][2] = value
        return a[1][2] * 100.0 + a[0][2] * 10.0 + a[1][0]

    assert list(struct.unpack(f"<{count}f", files[2].read_bytes())) == [
        expected(value, row & 1) for value, row in zip(fin, iin)
    ]
