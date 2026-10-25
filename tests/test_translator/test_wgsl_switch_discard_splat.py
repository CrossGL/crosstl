"""WGSL switch clauses, ``discard``, scalar-to-vector stores and integer operands.

C-style switches (HLSL, GLSL, MSL, CrossGL) share one body between consecutive labels and fall through a body that
does not end in a jump. A WGSL clause lists every selector of its body, never falls through, and a switch needs
exactly one ``default``. WGSL also spells the fragment kill ``discard;`` and has no implicit scalar-to-vector
conversion: a shift amount is ``u32`` (``vecN<u32>`` for a vector) and bitwise operands share one type.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import (
    BlockNode,
    BreakNode,
    CaseNode,
    ExpressionStatementNode,
    IdentifierNode,
    LiteralNode,
    SwitchNode,
)
from crosstl.translator.codegen.wgsl_switch_clauses import (
    ends_in_jump,
    wgsl_switch_clauses,
)

HLSL_SWITCH = """
RWStructuredBuffer<int> results : register(u0);
[numthreads(8, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    int r = 0;
    switch (int(id.x)) {
        case 0:
        case 1:
            r = 10;
            break;
        case 2:
            r = 20;
        case 3:
            r += 1;
            break;
        default:
            r = -1;
            break;
    }
    results[id.x] = r;
}
"""

HLSL_SWITCH_WITHOUT_DEFAULT = """
RWStructuredBuffer<int> results : register(u0);
[numthreads(8, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    int r = 0;
    switch (int(id.x)) {
        case 0: r = 1; break;
        case 1: r = 2; break;
    }
    results[id.x] = r;
}
"""

CROSSGL_DEFAULT_BETWEEN_CASES = """
shader Switches {
    compute {
        buffer int results[];
        layout(local_size_x = 8) in;
        void main() {
            int r = 0;
            switch (int(gl_GlobalInvocationID.x)) {
                case 4:
                default:
                    r = -1;
                case 5:
                    r += 5;
                    break;
                case 6:
                    r = 6;
                    break;
            }
            results[gl_GlobalInvocationID.x] = r;
        }
    }
}
"""

HLSL_DISCARD = """
float4 PSMain(float4 c : SV_Position) : SV_Target {
    if (c.x < 0.5) {
        discard;
    }
    clip(c.y - 0.5);
    return c;
}
"""

CROSSGL_DISCARD = """
shader Discards {
    fragment {
        vec4 main(vec4 c @ gl_FragCoord) @ gl_FragColor {
            if (c.x < 0.5) {
                discard;
            }
            return c;
        }
    }
}
"""

HLSL_SCALAR_STORES = """
RWStructuredBuffer<float4> results : register(u0);
float4 widen(float4 v) { return v * 2.0; }
float3 fill(float s) { return s; }
[numthreads(8, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float4 extra = 0;
    float3 v = 1.5;
    int2 k = 3;
    uint2 bits = 6;
    extra += 1;
    v = id.x;
    v.xy = 2.0;
    v.xz *= 3.0;
    k &= 1;
    bits >>= 1u;
    results[id.x] = widen(0.5) + extra + float4(v + fill(0.25), k.x) + float4(bits, bits);
}
"""


HLSL_INTEGER_OPERANDS = """
RWStructuredBuffer<uint2> results : register(u0);
[numthreads(8, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint2 bits = id.xy;
    int2 k = int2(id.xy);
    int amount = int(id.z);
    uint word = id.x << amount;
    bits = bits >> 1;
    k = k << 2u;
    bits <<= 3;
    k >>= amount;
    k = k & 7;
    bits = 1u | bits;
    bits ^= 5u;
    results[id.x] = bits + uint2(k) + word;
}
"""


def _translate(tmp_path, source, backend="wgsl", suffix="hlsl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def _naga_validates(tmp_path, code):
    validator = shutil.which("naga")
    if validator is None:
        return
    (tmp_path / "out.wgsl").write_text(code)
    result = subprocess.run(
        [validator, "out.wgsl"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr + code


def _switch_text(code):
    flat = _flat(code)
    start = flat.index("switch (")
    return flat[start : flat.index("} }", start) + 3]


def test_switch_labels_sharing_a_body_become_one_clause(tmp_path):
    wgsl = _translate(tmp_path, HLSL_SWITCH)
    switch = _switch_text(wgsl)
    assert switch.count("default") == 1, switch
    assert "case 0, 1: { r = 10; break; }" in switch
    assert "default: { r = -1; break; }" in switch
    _naga_validates(tmp_path, wgsl)


def test_switch_body_that_falls_through_runs_the_next_body(tmp_path):
    switch = _switch_text(_translate(tmp_path, HLSL_SWITCH))
    assert "case 2: { r = 20; r += 1; break; }" in switch
    assert "case 3: { r += 1; break; }" in switch


def test_switch_without_default_gets_one(tmp_path):
    wgsl = _translate(tmp_path, HLSL_SWITCH_WITHOUT_DEFAULT)
    switch = _switch_text(wgsl)
    assert switch.count("default") == 1, switch
    assert switch.endswith("default: { break; } }")
    _naga_validates(tmp_path, wgsl)


def test_default_between_cases_keeps_its_position(tmp_path):
    wgsl = _translate(tmp_path, CROSSGL_DEFAULT_BETWEEN_CASES, suffix="cgl")
    switch = _switch_text(wgsl)
    assert "case 4, default: { r = -1; r += 5; break; }" in switch, switch
    assert "case 5: { r += 5; break; }" in switch
    assert "case 6: { r = 6; break; }" in switch
    _naga_validates(tmp_path, wgsl)


def test_clause_declarations_are_scoped_per_clause(tmp_path):
    source = HLSL_SWITCH.replace(
        "r = 20;\n", "int t = 20;\n            r = t;\n"
    ).replace("r += 1;\n", "int u = 1;\n            r += u;\n")
    wgsl = _translate(tmp_path, source)
    switch = _switch_text(wgsl)
    assert "case 2: { var t: i32 = 20; r = t; var u: i32 = 1; r += u; break; }" in (
        switch
    ), switch
    assert "case 3: { var u: i32 = 1; r += u; break; }" in switch
    _naga_validates(tmp_path, wgsl)


def _case(value, *statements):
    literal = None if value is None else LiteralNode(value, "int")
    return CaseNode(value=literal, statements=list(statements))


def _discard():
    return ExpressionStatementNode(IdentifierNode("discard"))


def test_jumps_end_a_clause_but_other_statements_fall_through():
    assert ends_in_jump([BreakNode()])
    assert ends_in_jump([BlockNode([BreakNode()])])
    assert ends_in_jump([_discard()])
    assert not ends_in_jump([])
    assert not ends_in_jump([ExpressionStatementNode(IdentifierNode("x"))])


def test_clauses_follow_source_order():
    first, second, third = (
        ExpressionStatementNode(IdentifierNode(name)) for name in ("a", "b", "c")
    )
    node = SwitchNode(
        expression=IdentifierNode("s"),
        cases=[
            _case(1),
            _case(None, first),
            _case(2, second, BreakNode()),
            _case(3, third),
            _case(4),
        ],
    )
    clauses = wgsl_switch_clauses(node)
    assert [[getattr(v, "value", v) for v in c.selectors] for c in clauses] == [
        [1, None],
        [2],
        [3],
        [4],
    ]
    assert clauses[0].statements[:2] == [first, second]
    assert clauses[2].statements == [third]
    assert clauses[3].statements == []


@pytest.mark.parametrize(
    "source, suffix, count",
    [(HLSL_DISCARD, "hlsl", 2), (CROSSGL_DISCARD, "cgl", 1)],
)
def test_discard_is_the_wgsl_statement(tmp_path, source, suffix, count):
    wgsl = _translate(tmp_path, source, suffix=suffix)
    assert wgsl.count("discard;") == count, wgsl
    assert "discard_" not in wgsl
    _naga_validates(tmp_path, wgsl)


def test_scalar_stored_to_a_vector_is_splatted(tmp_path):
    wgsl = _translate(tmp_path, HLSL_SCALAR_STORES)
    flat = _flat(wgsl)
    for text in (
        "var extra: vec4<f32> = vec4<f32>(f32(0));",
        "var v: vec3<f32> = vec3<f32>(1.5);",
        "var k: vec2<i32> = vec2<i32>(3);",
        "var bits: vec2<u32> = vec2<u32>(u32(6));",
        "extra += vec4<f32>(f32(1));",
        "v = vec3<f32>(f32(id.x));",
        "k &= vec2<i32>(1);",
        "bits >>= vec2<u32>(1u);",
        "return vec3<f32>(s);",
        "widen(vec4<f32>(0.5))",
        "vec4<f32>(vec2<f32>(bits), vec2<f32>(bits))",
    ):
        assert text in flat, (text, wgsl)
    _naga_validates(tmp_path, wgsl)


def test_swizzle_stores_of_a_scalar(tmp_path):
    flat = _flat(_translate(tmp_path, HLSL_SCALAR_STORES))
    assert "let v_xy: vec2<f32> = vec2<f32>(2.0); v.x = (v_xy).x; v.y = (v_xy).y;" in (
        flat
    ), flat
    assert "v.x *= 3.0; v.z *= 3.0;" in flat


def test_shift_amounts_and_bitwise_scalars(tmp_path):
    wgsl = _translate(tmp_path, HLSL_INTEGER_OPERANDS)
    flat = _flat(wgsl)
    for text in (
        "var word: u32 = (id.x << u32(amount));",
        "bits = (bits >> vec2<u32>(1));",
        "k = (k << vec2<u32>(2u));",
        "bits <<= vec2<u32>(3);",
        "k >>= vec2<u32>(u32(amount));",
        "k = (k & vec2<i32>(7));",
        "bits = (vec2<u32>(1u) | bits);",
        "bits ^= vec2<u32>(5u);",
    ):
        assert text in flat, (text, wgsl)
    _naga_validates(tmp_path, wgsl)


GLSL_VECTOR_CONVERSIONS = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Out { ivec4 values[]; };
void main() {
    uvec2 u = gl_GlobalInvocationID.xy;
    ivec2 whole = ivec2(u);
    ivec4 parts = ivec4(u, u);
    values[0] = ivec4(whole, parts.xy);
}
"""


def test_vector_conversion_wraps_only_component_arguments(tmp_path):
    wgsl = _translate(tmp_path, GLSL_VECTOR_CONVERSIONS, suffix="glsl")
    flat = _flat(wgsl)
    assert "var whole: vec2<i32> = vec2<i32>(u);" in flat, wgsl
    assert "vec4<i32>(vec2<i32>(u), vec2<i32>(u))" in flat, wgsl
    assert "vec2<i32>(vec2<i32>(" not in flat, wgsl
    _naga_validates(tmp_path, wgsl)
