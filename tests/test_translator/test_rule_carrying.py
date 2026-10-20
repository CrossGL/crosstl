"""Rule carrying: source operation rules survive translation (docs/source/adr/0001-rule-carrying.md)."""

import re

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.codegen.registry import backend_names
from crosstl.translator.semantics import (
    FAMILIES,
    RuleInexpressibleError,
    apply_rule_carrying,
)
from crosstl.translator.semantics.hub_types import (
    INT32,
    UINT32,
    HubType,
    TypeScope,
    arithmetic_result,
)
from crosstl.translator.semantics.rule_table import (
    SOURCE_RULES,
    TARGET_ENTRIES,
    target_entry,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

GLSL_KERNEL = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Inp {{ int a[]; }};
layout(std430, binding = 1) buffer Outp {{ int o[]; }};
void main() {{
    uint i = gl_GlobalInvocationID.x;
    int x = a[i];
{body}
}}
"""


def translate(tmp_path, source, backend, suffix=".comp"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def glsl(tmp_path, body, backend):
    return translate(tmp_path, GLSL_KERNEL.format(body=body), backend)


def parse_cgl(text):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(text)


def test_rule_table_is_total_over_registered_backends():
    for family in FAMILIES:
        for backend in backend_names():
            assert (family, backend) in TARGET_ENTRIES, (family, backend)
    assert target_entry("wrap", "plugin-target").kind == "refuse"
    register_default_sources()
    for source in SOURCE_RULES:
        assert SOURCE_REGISTRY.get(source) is not None, source


def test_glsl_signed_add_wraps_on_metal(tmp_path):
    metal = glsl(tmp_path, "    o[i] = (x + 1 > x) ? 1 : 0;", "metal")
    assert "as_type<int>(as_type<uint>(x) + 1u) > x" in metal


def test_glsl_signed_arithmetic_stays_native_on_glsl(tmp_path):
    output = glsl(tmp_path, "    o[i] = x * 3 - x;", "opengl")
    assert "x * 3" in output
    assert "uint(" not in output.split("void main")[1]


def test_hlsl_signed_multiply_wraps_on_directx_and_metal(tmp_path):
    source = """RWStructuredBuffer<int> o : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int x = o[id.x];
    o[id.x] = x * 2;
}
"""
    hlsl = translate(tmp_path, source, "directx", ".hlsl")
    assert "int((uint(x) * 2u))" in hlsl
    metal = translate(tmp_path, source, "metal", ".hlsl")
    assert "as_type<int>(as_type<uint>(x) * 2u)" in metal


def test_spirv_target_computes_wrapped_add_on_unsigned_operands(tmp_path):
    spirv = glsl(tmp_path, "    o[i] = x + 7;", "vulkan")
    lines = spirv.splitlines()
    uint_id = next(
        line.split(" = ")[0]
        for line in lines
        if line.endswith("OpTypeInt 32 0") and " = " in line
    )
    assert any(f"OpIAdd {uint_id} " in line for line in lines)
    assert any("OpBitcast" in line for line in lines)


def test_metal_and_cuda_sources_carry_no_wrap_obligation(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    o[i] = o[i] + 1;
}
"""
    hlsl = translate(tmp_path, source, "directx", ".metal")
    assert "o[i] = (o.Load(i) + 1);" in hlsl


def test_bounded_loop_counter_stays_native_and_its_bound_wraps(tmp_path):
    body = "    int count = 0;\n    for (int k = x; k < x + 4; ++k) { count += 1; }\n    o[i] = count;"
    metal = glsl(tmp_path, body, "metal")
    assert "k < as_type<int>(as_type<uint>(x) + 4u); ++k)" in metal
    assert "count = as_type<int>(as_type<uint>(count) + 1u);" in metal


def test_loop_counter_without_strict_bound_wraps(tmp_path):
    body = "    int s = 0;\n    for (int k = 0; k <= x; k++) { s = k; }\n    o[i] = s;"
    metal = glsl(tmp_path, body, "metal")
    assert "k = as_type<int>(as_type<uint>(k) + 1u)" in metal


def test_loop_counter_written_in_the_body_is_not_proven(tmp_path):
    body = "    int s = 0;\n    for (int k = 0; k < x; ++k) { k = k * 2; s = k; }\n    o[i] = s;"
    metal = glsl(tmp_path, body, "metal")
    assert "k = as_type<int>(as_type<uint>(k) + 1u)" in metal


def test_compound_assignment_and_increment_statements_wrap(tmp_path):
    body = "    o[i] += x;\n    o[i]++;\n    --x;\n    o[i] *= x;"
    metal = glsl(tmp_path, body, "metal")
    assert "as_type<int>(as_type<uint>(" in metal
    for text in ("+= x", "++;", "--x", "*= x"):
        assert text not in metal
    assert "x = as_type<int>(as_type<uint>(x) - 1u);" in metal


def test_increment_used_as_a_value_is_refused_where_it_needs_a_polyfill(tmp_path):
    # Targets whose emitters write back inout parameters call a step helper instead
    # (test_wrapping_step_values.py).
    body = "    int y = x++;\n    o[i] = y;"
    with pytest.raises(RuleInexpressibleError) as refusal:
        glsl(tmp_path, body, "cuda")
    assert (
        refusal.value.project_diagnostic_code == "project.translate.rule-inexpressible"
    )
    assert "x++" in glsl(tmp_path, body, "opengl")


def test_compound_assignment_to_impure_lvalue_is_refused(tmp_path):
    source = GLSL_KERNEL.format(body="    o[next(x)] += 1;").replace(
        "void main()", "int next(int v) { return v; }\nvoid main()"
    )
    with pytest.raises(RuleInexpressibleError):
        translate(tmp_path, source, "metal")


def test_vector_and_negation_wrap(tmp_path):
    body = (
        "    ivec2 v = ivec2(x, 3) * 5;\n    ivec2 w = -v;\n    int n = -x;\n"
        "    o[i] = w.x + v.y + n + -1;"
    )
    metal = glsl(tmp_path, body, "metal")
    assert "as_type<int2>(as_type<uint2>(int2(x, 3)) * 5u)" in metal
    assert "as_type<int2>(0u - as_type<uint2>(v))" in metal
    assert "as_type<int>(0u - as_type<uint>(x))" in metal
    # A negated literal is a constant, so it is not rewritten as a wrapping negation.
    assert "as_type<uint>(-1)" in metal


def test_literal_overflow_folds_to_the_wrapped_value(tmp_path):
    metal = glsl(tmp_path, "    o[i] = 2147483647 + 1;", "metal")
    assert "-2147483647 - 1" in metal
    assert "2147483647 + 1" not in metal


def test_hlsl_bool_operand_is_converted_before_reinterpretation(tmp_path):
    source = """RWStructuredBuffer<int> o : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int x = o[0];
    bool b = x > 2;
    o[1] = x + b;
}
"""
    metal = translate(tmp_path, source, "metal", ".hlsl")
    assert "as_type<uint>(int(b))" in metal


def test_crossgl_intrinsics_lower_per_target(tmp_path):
    source = """shader W {
    compute {
        layout(std430, binding = 0) buffer Data { int values[]; };
        [numthreads(1, 1, 1)]
        void main() {
            int a = values[0];
            values[1] = __cgl_wrap_add(a, 1);
            values[2] = a + 1;
        }
    }
}
"""
    metal = translate(tmp_path, source, "metal", ".cgl")
    assert "as_type<int>(as_type<uint>(a) + 1u)" in metal
    assert metal.count("as_type<uint>") == 1
    glsl_out = translate(tmp_path, source, "opengl", ".cgl")
    assert "__cgl_" not in glsl_out
    assert glsl_out.count("a + 1") == 2


def test_rule_carrying_is_idempotent():
    program = parse_cgl("""shader W {
    int f(int a, int b) { return a * b + a; }
}
""")
    apply_rule_carrying(program, "opengl", "metal")
    once = repr(program.functions[0].body.statements[0].value)
    report = apply_rule_carrying(program, "opengl", "metal")
    assert repr(program.functions[0].body.statements[0].value) == once
    assert report.annotated == {}
    assert program.annotations["rule_carrying"]["target"] == "metal"


def test_unknown_target_refuses_carried_rules():
    program = parse_cgl("""shader W {
    int f(int a, int b) { return a + b; }
}
""")
    with pytest.raises(RuleInexpressibleError):
        apply_rule_carrying(program, "directx", "plugin-target")
    untouched = parse_cgl("""shader W {
    int f(int a, int b) { return a + b; }
}
""")
    apply_rule_carrying(untouched, "metal", "plugin-target")
    assert isinstance(untouched.functions[0].body.statements[0].value, A.BinaryOpNode)


def test_rust_native_entry_records_its_configuration():
    program = parse_cgl("""shader W {
    int f(int a, int b) { return a + b; }
}
""")
    report = apply_rule_carrying(program, "vulkan", "rust")
    assert report.discharged == {"native": 1}
    assert report.configurations == {"overflow-checks = false"}


def test_usual_arithmetic_conversions():
    assert arithmetic_result(INT32, UINT32) == UINT32
    assert arithmetic_result(INT32, HubType("i", 16)) == INT32
    assert arithmetic_result(HubType("b", 1), INT32) == INT32
    assert arithmetic_result(INT32.with_lanes(3), INT32) == INT32.with_lanes(3)
    assert arithmetic_result(INT32, HubType("i", 64)) == HubType("i", 64)
    assert arithmetic_result(UINT32, HubType("i", 64)) == HubType("i", 64)
    assert arithmetic_result(INT32, HubType("f", 32)) == HubType("f", 32)
    assert arithmetic_result(INT32, None) is None


def test_scopes_resolve_shadowed_names():
    program = parse_cgl("""shader W {
    int g;
    float f(float g) { return g + 1.0; }
}
""")
    scope = TypeScope(program)
    scope.declare("g", A.PrimitiveType("int"))
    scope.push()
    scope.declare("g", A.PrimitiveType("float"))
    assert scope.infer(A.IdentifierNode("g")).kind == "f"
    scope.pop()
    assert scope.infer(A.IdentifierNode("g")) == INT32


def test_metal_bit_cast_operand_keeps_its_own_signedness(tmp_path):
    source = """shader W {
    compute {
        layout(std430, binding = 0) buffer Data { int values[]; };
        [numthreads(1, 1, 1)]
        void main() {
            ivec2 v = ivec2(values[0], values[1]);
            uvec2 p = asuint(v);
            ivec2 r = asint(p + p);
            ivec2 q = asint(asuint(v) * asuint(v));
            values[2] = r.x + q.y;
        }
    }
}
"""
    metal = translate(tmp_path, source, "metal", ".cgl")
    assert "int2 r = as_type<int2>(p + p);" in metal
    assert "int2 q = as_type<int2>(as_type<uint2>(v) * as_type<uint2>(v));" in metal


HLSL_UNSIGNED_KERNEL = """RWStructuredBuffer<uint> o : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    uint a = o[0];
    uint b = o[1];
    int s = int(o[2]);
{body}
}}
"""


def hlsl(tmp_path, body, backend):
    return translate(tmp_path, HLSL_UNSIGNED_KERNEL.format(body=body), backend, ".hlsl")


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "wgsl", "cuda"])
def test_hlsl_unsigned_division_by_zero_gives_all_ones(tmp_path, backend):
    out = hlsl(tmp_path, "    o[3] = a / b;\n    o[4] = a % 0u;", backend)
    assert "cgl_rule_udiv_ones(a, b)" in out
    assert "cgl_rule_urem_ones(a, 0" in out
    # Each helper is defined once: one return of the all-ones constant per helper.
    returns = [line for line in out.splitlines() if "return" in line]
    assert sum("4294967295" in line for line in returns) == 2
    assert "__cgl_" not in out


def test_hlsl_division_by_a_nonzero_literal_and_signed_division_stay_native(tmp_path):
    metal = hlsl(
        tmp_path, "    o[3] = a / 4u + a % 3u;\n    o[4] = uint(s / 2);", "metal"
    )
    assert "cgl_rule" not in metal
    assert "a / 4u" in metal and "a % 3u" in metal


def test_hlsl_unsigned_vector_and_compound_division(tmp_path):
    body = (
        "    uint2 v = uint2(a, b);\n    v /= uint2(b, a);\n    a %= b;\n"
        "    uint m = a / s;\n    o[3] = v.x + v.y + a + m;"
    )
    metal = hlsl(tmp_path, body, "metal")
    assert "uint2 cgl_rule_udiv_ones2(uint2 x0, uint2 x1)" in metal
    assert "v = cgl_rule_udiv_ones2(v, uint2(b, a));" in metal
    assert "a = cgl_rule_urem_ones(a, b);" in metal
    # A signed operand is converted to the operation's unsigned type first.
    assert "cgl_rule_udiv_ones(a, uint(s))" in metal
    for text in ("/=", "%="):
        assert text not in metal


def test_hlsl_unsigned_division_helper_reaches_spirv(tmp_path):
    spirv = hlsl(tmp_path, "    o[3] = a / b;", "vulkan")
    assert "OpFunctionCall" in spirv
    assert spirv.count("OpUDiv") == 1
    assert re.search(r"OpConstant %\S+ 4294967295\b", spirv)


def test_glsl_unsigned_division_carries_no_obligation(tmp_path):
    source = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Data { uint d[]; };
void main() {
    d[2] = d[0] / d[1];
}
"""
    metal = translate(tmp_path, source, "metal")
    assert "cgl_rule" not in metal


def test_reserved_helper_name_in_the_program_is_refused():
    program = parse_cgl("""shader W {
    uint cgl_rule_udiv_ones(uint x0, uint x1) { return 0u; }
    uint f(uint a, uint b) { return a / b; }
}
""")
    with pytest.raises(RuleInexpressibleError) as refusal:
        apply_rule_carrying(program, "directx", "metal")
    assert "reserved" in str(refusal.value)


def test_udiv_helpers_are_injected_once_and_carrying_is_idempotent():
    program = parse_cgl("""shader W {
    uint f(uint a, uint b) { return a / b + a % b; }
    uvec3 g(uvec3 a, uvec3 b) { return a / b; }
}
""")
    report = apply_rule_carrying(program, "directx", "metal")
    names = [function.name for function in program.functions]
    assert names == [
        "cgl_rule_udiv_ones",
        "cgl_rule_udiv_ones3",
        "cgl_rule_urem_ones",
        "f",
        "g",
    ]
    assert report.discharged == {"polyfill": 3}
    apply_rule_carrying(program, "directx", "metal")
    assert [function.name for function in program.functions] == names


def test_crossgl_udiv_intrinsic_lowers_to_the_helper(tmp_path):
    source = """shader W {
    compute {
        layout(std430, binding = 0) buffer Data { uint values[]; };
        [numthreads(1, 1, 1)]
        void main() {
            values[2] = __cgl_udiv_ones(values[0], values[1]);
            values[3] = values[0] / values[1];
        }
    }
}
"""
    # The anonymous block's members are read through its instance, which Metal lowers to byte offsets.
    operands = {
        "metal": r"\(\*reinterpret_cast<const device uint ?\*>\(data \+ 0\)\),\s*"
        r"\(\*reinterpret_cast<const device uint ?\*>\(data \+ 4\)\)",
        "opengl": r"data\.values\[0\], data\.values\[1\]",
    }
    for backend, pattern in operands.items():
        out = translate(tmp_path, source, backend, ".cgl")
        assert re.search(rf"= cgl_rule_udiv_ones\({pattern}\);", out), out
        assert "__cgl_" not in out
