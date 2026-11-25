"""HLSL output of writes through a dynamic CrossGL matrix column.

CrossGL ``m[k]`` is column ``k``; HLSL reads it as ``transpose(m)[k]``, which is not an lvalue. DXC accepts an
assignment, an increment and an ``out``/``inout`` argument through ``transpose(m)[k]`` and writes a temporary nothing
reads, so every such write must go through the matrix itself: element stores ``m[r][k]``, the column setter, or a
temporary that is stored back after the call. Writes whose value is used are refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused

KERNEL = """
shader main {{
    struct S {{ mat2 m; float pad; }};
    void bump(inout vec2 v, float r) {{ v.y = v.y + r; }}
    void sety(inout float s) {{ s = 9.0; }}
    void fill(out vec3 v) {{ v = vec3(1.0, 2.0, 3.0); }}
    float bumpr(inout vec2 v) {{ v.x = v.x * 2.0; return v.x; }}
    compute {{
        layout(local_size_x = 2) in;
        buffer float* fin;
        buffer mat2* outp;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            int k = int(fin[i]);
            mat2 m = mat2(1.0, 2.0, 3.0, 4.0);
            mat2x3 q = mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
            mat2 ms[2];
            ms[0] = mat2(1.0);
            ms[1] = mat2(2.0);
            S s;
            s.m = mat2(3.0);
            s.pad = 0.0;
            int j = 0;
            {body}
            outp[i] = m + ms[0] + ms[1] + s.m + mat2(q[0].xy, q[1].yz) + mat2(float(j));
        }}
    }}
}}
"""

# A write through transpose(...)[...] that DXC would apply to a temporary.
LOST_WRITE = re.compile(
    r"transpose\([^;]*\)\[[^\]]*\](?:\.[xyzw]+)?\s*(?:[-+*/]?=(?!=)|\+\+|--)"
)


def _hlsl(tmp_path, body, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(KERNEL.format(body=body), encoding="utf-8")
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _dxc(tmp_path, code, *extra):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "probe.hlsl"
    path.write_text(code, encoding="utf-8")
    for target in ((), ("-spirv", "-fspv-target-env=vulkan1.1")):
        result = subprocess.run(
            [dxc, "-T", "cs_6_6", "-E", "CSMain", "-WX", *extra, *target, str(path)]
            + ["-Fo", str(tmp_path / "probe.out")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "body, expected",
    [
        ("m[k].y = 7.0;", ["m[1][k] = (7.0);"]),
        ("m[k].x += 1.0;", ["m[0][k] += (1.0);"]),
        ("m[k].y++;", ["m[1][k] += (1);"]),
        ("--m[k].x;", ["m[0][k] -= (1);"]),
        (
            "m[k] *= 2.0;",
            ["m[0][k] *= __crossgl_column[0]", "m[1][k] *= __crossgl_column[1]"],
        ),
        (
            "m[k].yx = vec2(1.0, 2.0);",
            ["m[1][k] = __crossgl_column[0]", "m[0][k] = __crossgl_column[1]"],
        ),
        (
            "m[k] = vec2(5.0, 6.0);",
            ["m = __crossgl_set_column_float2x2(m, k, float2(5.0, 6.0));"],
        ),
        (
            "q[k].zx = vec2(10.0, 20.0);",
            ["q[2][k] = __crossgl_column[0]", "q[0][k] = __crossgl_column[1]"],
        ),
        ("ms[1 - k][k].y = 5.0;", ["ms[(1 - k)][1][k] = (5.0);"]),
        ("s.m[k].x -= 1.0;", ["s.m[0][k] -= (1.0);"]),
        ("outp[i][k].y = 9.0;", ["outp[i][1][k] = (9.0);"]),
        (
            "q[j++ + k].y *= 3.0;",
            [
                "__crossgl_column_index = (j++ + k)",
                "q[1][__crossgl_column_index] *= (3.0)",
            ],
        ),
    ],
)
def test_column_assignments_store_into_the_matrix(tmp_path, body, expected):
    code = _hlsl(tmp_path, body)
    flat = _flat(code)
    for fragment in expected:
        assert _flat(fragment) in flat, code
    assert not LOST_WRITE.search(flat), code
    _dxc(tmp_path, code)


@pytest.mark.parametrize(
    "body, stores",
    [
        (
            "bump(m[k], 0.5);",
            [
                "m[0][__crossgl_integer_index] = __crossgl_argument",
                "m[1][__crossgl_integer_index] = __crossgl_argument",
            ],
        ),
        ("sety(m[k].x);", ["m[0][__crossgl_integer_index] = (__crossgl_argument"]),
        ("fill(q[k]);", ["q[2][__crossgl_integer_index] = __crossgl_argument"]),
        (
            "float r = bumpr(m[k]) + 1.0; j = int(r);",
            ["m[1][__crossgl_integer_index] = __crossgl_argument"],
        ),
        (
            "float fr = modf(fin[i] + 0.25, ms[k][k].x); j = int(fr * 4.0);",
            [
                "ms[__crossgl_integer_index][0][__crossgl_integer_index_] = (__crossgl_argument"
            ],
        ),
    ],
)
def test_column_arguments_are_stored_back(tmp_path, body, stores):
    code = _hlsl(tmp_path, body)
    flat = _flat(code)
    for fragment in stores:
        assert _flat(fragment) in flat, code
    # The callee receives a temporary, never transpose(...)[...] itself.
    assert not re.search(r"(?:bump|sety|fill|bumpr|modf\w*)\(\s*transpose", flat), code
    _dxc(tmp_path, code)


def test_value_returning_call_keeps_its_result(tmp_path):
    code = _flat(_hlsl(tmp_path, "float r = bumpr(m[k]) + 1.0; j = int(r);"))
    match = re.search(r"(__crossgl_argument\d+) = bumpr\(", code)
    assert match, code
    # The comma expression ends with the call's result, after the copy-back.
    assert re.search(re.escape(match.group(1)) + r"\)\) \+ 1\.0\)", code), code


def test_modf_keeps_its_fraction(tmp_path):
    code = _flat(
        _hlsl(
            tmp_path, "float fr = modf(fin[i] + 0.25, ms[k][k].x); j = int(fr * 4.0);"
        )
    )
    match = re.search(r"(__crossgl_argument\d+) = crosstl_modf_float\(", code)
    assert match, code
    assert re.search(re.escape(match.group(1)) + r"\)\);", code), code


@pytest.mark.parametrize(
    "body",
    [
        "float z = (m[k].y = 3.0); j = int(z);",
        "float w = m[k].y++; j = int(w);",
        "for (int n = 0; n < 2; m[k].y++) { n++; }",
        "float z = (m[k] = vec2(1.0)).x; j = int(z);",
    ],
)
def test_used_column_writes_are_refused(tmp_path, body):
    with pytest.raises(DirectXLoweringRefused, match="dynamic matrix column"):
        _hlsl(tmp_path, body)


BUFFER_BLOCK = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { mat2 m; mat2x3 q; float f[]; };
void bump(inout vec2 v) { v.y += 0.5; }
void main() {
    int k = int(f[0]);
    m[k].y++;
    bump(m[k]);
    q[k].x--;
}
"""


def test_buffer_block_column_increments_and_arguments_store_bytes(tmp_path):
    path = tmp_path / "block.comp"
    path.write_text(BUFFER_BLOCK, encoding="utf-8")
    code = crosstl.translate(str(path), backend="directx", format_output=False)
    flat = _flat(code)
    assert "b.Store((k * 8 + 4), asuint((asfloat(b.Load((k * 8 + 4))) + (1))));" in flat
    assert (
        "b.Store((16 + k * 16), asuint((asfloat(b.Load((16 + k * 16))) - (1))));"
        in flat
    )
    assert re.search(
        r"bump\((__crossgl_argument\d+)\), \(b\.Store2\(\(__crossgl_integer_index \* 8\), "
        r"asuint\(\1\)\)\)",
        flat,
    ), code
    assert not LOST_WRITE.search(flat), code
    _dxc(tmp_path, code)
