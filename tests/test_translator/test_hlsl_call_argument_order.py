"""HLSL output evaluates call arguments in source order.

DXC's DXIL code generator evaluates the arguments of function and intrinsic calls from right to left, while
CrossGL (GLSL 4.60 section 6.1.1) evaluates them from left to right and copies an ``inout`` argument in at its
turn. The HLSL generator therefore evaluates the arguments up to the last one with a side effect into
temporaries, in order, inside a comma expression at the call's own position.
"""

import re
import shutil
import subprocess

import numpy as np
import pytest

import crosstl

HEADER = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Inp { int a[]; };
layout(std430, binding = 1) buffer Outp { int o[]; };
shared int tile[64];
int counter;
int bump(inout int t) { t = t * 3 + 1; return t - 2; }
int two(int p, int q) { return p * 100 + q; }
int three(int p, int q, int r) { return (p * 100 + q) * 100 + r; }
void scale(inout int p, int q) { p = p * 10 + q; }
void put(out int dst, int v) { dst = v; }
int next() { counter = counter * 2 + 1; return counter; }
int sum2(int v[2], int q) { return v[0] + v[1] * 10 + q * 100; }
"""

# Statements whose stores DXC folds to constants once ``x`` is a constant, with the source-order value of each.
FOLDED = [
    ("int y = x; o[0] = two(y, bump(y));", lambda x: x * 100 + 3 * x - 1),
    ("int y = x; o[1] = max(bump(y), y);", lambda x: max(3 * x - 1, 3 * x + 1)),
    (
        "counter = x; o[2] = three(next(), next(), counter);",
        lambda x: ((2 * x + 1) * 100 + 4 * x + 3) * 100 + 4 * x + 3,
    ),
    ("int m = x; scale(m, bump(m)); o[3] = m;", lambda x: x * 10 + 3 * x - 1),
    ("int y = x; o[4] = two(bump(y), y);", lambda x: (3 * x - 1) * 100 + 3 * x + 1),
    (
        "int y = x; o[5] = clamp(y, bump(y), y + 100);",
        lambda x: min(max(x, 3 * x - 1), 3 * x + 101),
    ),
    (
        "int s[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0); int k = x & 1; put(s[k], bump(k));"
        " o[6] = s[x & 1] * 16 + s[k];",
        lambda x: (3 * (x & 1) - 1) * 16,
    ),
    (
        "int v[2] = int[2](x, x + 1); o[7] = sum2(v, bump(v[0]));",
        lambda x: x + (x + 1) * 10 + (3 * x - 1) * 100,
    ),
]


def _shader(body):
    return (
        HEADER
        + "void main() {\n    int x = a[gl_GlobalInvocationID.x];\n"
        + body
        + "\n}\n"
    )


def _translate(tmp_path, source):
    path = tmp_path / "shader.comp"
    path.write_text(source)
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _entry(code):
    return code[code.index("void CSMain") :]


def test_in_argument_is_read_before_a_later_side_effect(tmp_path):
    code = _entry(_translate(tmp_path, _shader("int y = x; o[0] = two(y, bump(y));")))
    assert (
        "(__crossgl_argument0 = y, __crossgl_argument1 = bump(y), "
        "two(__crossgl_argument0, __crossgl_argument1))"
    ) in code
    assert "    int __crossgl_argument0;\n" in code


def test_built_in_arguments_are_ordered_too(tmp_path):
    code = _entry(_translate(tmp_path, _shader("int y = x; o[0] = max(bump(y), y);")))
    assert "(__crossgl_argument0 = bump(y), max(__crossgl_argument0, y))" in code


def test_inout_argument_is_copied_in_at_its_turn_and_back_after_the_call(tmp_path):
    code = _entry(
        _translate(tmp_path, _shader("int m = x; scale(m, bump(m)); o[0] = m;"))
    )
    assert (
        "(__crossgl_argument0 = m, __crossgl_argument1 = bump(m), "
        "scale(__crossgl_argument0, __crossgl_argument1), (m = __crossgl_argument0));"
    ) in code


def test_out_argument_location_is_fixed_at_its_turn(tmp_path):
    code = _entry(
        _translate(
            tmp_path,
            _shader(
                "int s[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0); int k = x & 1;"
                " put(s[k], bump(k)); o[0] = s[k];"
            ),
        )
    )
    match = re.search(
        r"\((\w+) = k, (__crossgl_argument\d+) = bump\(k\), put\((__crossgl_argument\d+), \2\), "
        r"\(s\[\1\] = \3\)\);",
        code,
    )
    assert match, code


def test_array_argument_is_copied_at_its_turn(tmp_path):
    code = _entry(
        _translate(
            tmp_path,
            _shader("int v[2] = int[2](x, x + 1); o[0] = sum2(v, bump(v[0]));"),
        )
    )
    assert "    int __crossgl_argument0[2];\n" in code
    assert "(__crossgl_argument0 = v, __crossgl_argument1 = " in code


def test_ordered_call_stays_where_the_source_evaluates_it(tmp_path):
    code = _entry(
        _translate(
            tmp_path,
            _shader(
                "int y = x; o[0] = (x > 0) ? two(y, bump(y)) : 0;\n"
                "for (int i = 0; i < two(y, bump(y)); i++) { o[1] += 1; }"
            ),
        )
    )
    assert "? int((__crossgl_argument0 = y, __crossgl_argument1 = bump(y), two(" in code
    assert (
        "(i < int((__crossgl_argument2 = y, __crossgl_argument3 = bump(y), two(" in code
    )


@pytest.mark.parametrize(
    "statement",
    [
        "o[0] = two(x, x + 1);",  # no side effect
        "int y = x; o[0] = two(bump(y), 3);",  # the other argument is a constant
        "int y = x; o[0] = two(3, bump(y));",
        "int y = x; ivec2 v = ivec2(bump(y), y); o[0] = v.x;",  # constructors are left to right in DXC
        "int y = x; o[0] = bump(y);",
        # the side effect writes a variable no other argument reads
        "int y = x; int z = x + 1; o[0] = two(z, bump(y));",
        "int y = x; int z = x + 1; o[0] = max(bump(y), z);",
        "int y = x; int z = x + 1; scale(z, bump(y)); o[0] = z;",
        # constants built by operators and constructors read nothing a side effect can write
        "o[0] = int(dot(vec2(1.0, -2.0), vec2(float(next()), 1.0)));",
        "o[0] = two(int(2.5 * 2.0), next());",
        "o[0] = clamp(ivec2(next(), 1), ivec2(0), ivec2(1 + 2, 4)).x;",
    ],
)
def test_calls_whose_order_cannot_matter_are_unchanged(tmp_path, statement):
    code = _entry(_translate(tmp_path, _shader(statement)))
    assert "__crossgl_argument" not in code


def test_a_side_effect_on_a_module_variable_orders_every_call_argument(tmp_path):
    # next() writes counter; two() reads counter through another call, so both are ordered.
    code = _entry(
        _translate(
            tmp_path,
            _shader(
                "counter = x; o[0] = two(next(), counter);\no[1] = two(next(), three(1, 2, counter));"
            ),
        )
    )
    assert (
        "int((__crossgl_argument0 = next(), two(__crossgl_argument0, counter)))" in code
    )
    assert "int((__crossgl_argument1 = next(), two(__crossgl_argument1, three(" in code


def _stored_constants(dxc, tmp_path, code):
    source = tmp_path / "folded.hlsl"
    source.write_text(code)
    listing = tmp_path / "folded.ll"
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fc", str(listing)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    body = listing.read_text().split("define void @CSMain", 1)[1]
    return [
        int(value)
        for value in re.findall(
            r"@dx\.op\.(?:raw)?[bB]ufferStore\.i32\([^)]*?, i32 undef, i32 (-?\d+)",
            body,
        )
    ]


@pytest.mark.parametrize("x", [436, -437, 7, 0])
def test_dxil_computes_the_source_order_values(tmp_path, x):
    """DXC folds each store to a constant; every one is the value of left-to-right evaluation."""
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    body = "\n".join(f"{{ {statement} }}" for statement, _ in FOLDED)
    code = _translate(tmp_path, _shader(body))
    folded, count = re.subn(
        r"int x = asint\(inp\.Load\([^;]*\)\);", f"int x = {x};", code
    )
    assert count == 1, code
    stored = _stored_constants(dxc, tmp_path, folded)
    expected = [int(np.int32(value(x))) for _, value in FOLDED]
    assert stored == expected


REMAINDER = """shader Probe {
    float bump(inout float t) { t = t * 3.0 + 1.0; return t - 2.0; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* v;
        void main() {
            float y = v[0];
            float r = bump(y) % y;
            float z = v[0];
            v[1] = r + fmod(bump(z), z);
        }
    }
}
"""


def test_floating_remainder_operands_are_ordered(tmp_path):
    """A floating ``%`` becomes a call of the exact ``fmod`` helper, whose arguments DXC evaluates right to left; its
    operands are ordered as an ``fmod`` call's are, and the initializer's comma expression is cast to its type.
    """
    path = tmp_path / "remainder.cgl"
    path.write_text(REMAINDER)
    code = crosstl.translate(str(path), backend="directx", format_output=False)
    assert (
        "float r = float((__crossgl_argument0 = bump(y), "
        "crosstl_fmod_float(__crossgl_argument0, y)));"
    ) in code
    assert (
        "(__crossgl_argument1 = bump(z), crosstl_fmod_float(__crossgl_argument1, z))"
        in code
    )
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    source = tmp_path / "remainder.hlsl"
    source.write_text(code)
    result = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            "-WX",
            str(source),
            "-Fo",
            str(tmp_path / "r.dxil"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
