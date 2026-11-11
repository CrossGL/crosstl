"""GLSL generation looks names up exactly and refuses a name its tables do not hold.

Each refusal below replaces output that the generator used to write verbatim: a call to an undeclared function,
a call through an array element, a member the structure does not declare, a mesh or ray-tracing operation with the
wrong operand count, and a ``gl_`` or ``SV_`` semantic that names no GLSL built-in variable.
"""

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_exact_lookups import (
    GLSL_BITCAST_CANONICAL_NAMES,
    call_names_resolving_to,
    glsl_undeclared_call_reasons,
)
from tests.lowering_refusal_helpers import refused_glsl


def generate(source):
    return GLSLCodeGen().generate(crosstl.translator.parse(source))


def refusal(source):
    return refused_glsl(GLSLCodeGen().generate, crosstl.translator.parse(source))


def compute_program(helper):
    return f"shader S {{ {helper} compute {{ void main() {{}} }} }}"


def vertex_member_program(semantic):
    return f"""
    shader S {{
        struct VSOut {{ vec4 pos @gl_Position; float v @{semantic}; }};
        vertex {{
            VSOut main(vec3 p @POSITION) {{
                VSOut o;
                o.pos = vec4(p, 1.0);
                o.v = 1.0;
                return o;
            }}
        }}
    }}
    """


def test_call_to_an_undeclared_function_is_refused():
    message = refusal(compute_program("float f(float x) { return mystery(x); }"))
    assert "call to 'mystery'" in message
    assert "neither a GLSL built-in function nor a function the program declares" in (
        message
    )


def test_call_through_an_array_element_is_refused():
    message = refusal(
        compute_program("float f(float x) { float y[2]; return y[0](x); }")
    )
    assert "call through 'y[0]'" in message


@pytest.mark.parametrize(
    "helper, expected",
    [
        pytest.param(
            "float f(float x) { return clamp(sin(x), 0.0, 1.0); }",
            "return clamp(sin(x), 0.0, 1.0);",
            id="builtin",
        ),
        pytest.param(
            "float g(float x) { return x; } float f(float x) { return g(x); }",
            "return g(x);",
            id="declared-function",
        ),
        pytest.param(
            "float f() { float y[2] = float[2](1.0, 2.0); return y[1]; }",
            "float y[2] = float[2](1.0, 2.0);",
            id="array-constructor",
        ),
        pytest.param(
            "float f() { float y[3]; return float(y.length()); }",
            "return float(y.length());",
            id="length-method",
        ),
        pytest.param(
            "struct P { float a; }; P f(float x) { return P(x); }",
            "return P(x);",
            id="structure-constructor",
        ),
        pytest.param(
            "float f(float x) { return lerp(x, 1.0, frac(x)); }",
            "return mix(x, 1.0, fract(x));",
            id="renamed-builtin",
        ),
    ],
)
def test_calls_glsl_can_resolve_are_written(helper, expected):
    assert expected in generate(compute_program(helper))


def test_a_called_helper_counts_as_declared_only_where_the_output_declares_it():
    generator = GLSLCodeGen()
    generator.glsl_unresolved_call_targets = {"helper": None, "unused": None}
    declared = "float helper(float x) {\n    return x;\n}\nvoid main() {\n    helper(1.0);\n}\n"
    undeclared = "void main() {\n    float y = helper(1.0);\n    return helper(y);\n}\n"
    macro = "#define helper(x) (x)\nvoid main() {\n    helper(1.0);\n}\n"

    assert glsl_undeclared_call_reasons(generator, declared) == []
    assert glsl_undeclared_call_reasons(generator, macro) == []
    assert glsl_undeclared_call_reasons(generator, undeclared) == [
        "call to 'helper', which is neither a GLSL built-in function nor a function the program declares"
    ]


def test_member_a_structure_does_not_declare_is_refused():
    message = refusal(
        compute_program("struct P { float a; }; float f(P p) { return p.b; }")
    )
    assert "member 'b' is not declared by structure 'P'" in message


def test_declared_member_is_written():
    code = generate(
        compute_program("struct P { float a; }; float f(P p) { return p.a; }")
    )
    assert "return p.a;" in code


@pytest.mark.parametrize(
    "statement, reason",
    [
        pytest.param(
            "SetMeshOutputCounts(1, 2, 3);",
            "'SetMeshOutputCounts' becomes SetMeshOutputsEXT, which takes 2 arguments; the call passes 3",
            id="mesh-output-counts",
        ),
        pytest.param(
            "TraceRay(1, 2, 3, 4, 5, 6, 7, 8);",
            "'TraceRay' becomes traceRayEXT, which takes 11 operands; the call passes 8",
            id="hlsl-shaped-trace-ray",
        ),
        pytest.param(
            "ReportHit(1.0, 0u, 2.0);",
            "'ReportHit' becomes reportIntersectionEXT, which takes 2 operands; the call passes 3",
            id="report-hit-with-attributes",
        ),
        pytest.param(
            "IgnoreHit(1);",
            "'IgnoreHit' becomes ignoreIntersectionEXT, which takes 0 operands; the call passes 1",
            id="ignore-hit-operand",
        ),
    ],
)
def test_mesh_and_ray_operations_with_the_wrong_operand_count_are_refused(
    statement, reason
):
    message = refusal(compute_program(f"float f() {{ {statement} return 0.0; }}"))
    assert reason in message


def test_ray_tracing_jump_statements_take_no_operands():
    code = generate(compute_program("void f() { IgnoreHit(); }"))
    assert "    ignoreIntersectionEXT;" in code


@pytest.mark.parametrize(
    "semantic, reason",
    [
        pytest.param(
            "gl_FragBogus",
            "semantic 'gl_FragBogus' is in the gl_ namespace GLSL reserves",
            id="unknown-gl-name",
        ),
        pytest.param(
            "SV_Bogus",
            "semantic 'SV_Bogus' is a Direct3D system value with no GLSL built-in variable",
            id="unmapped-system-value",
        ),
    ],
)
def test_semantics_that_name_no_glsl_builtin_are_refused(semantic, reason):
    assert reason in refusal(vertex_member_program(semantic))


def test_user_semantic_stays_a_varying():
    assert "out float v;" in generate(vertex_member_program("MY_VALUE"))


def test_glsl_builtin_semantic_outside_the_table_is_kept():
    # gl_Layer is a GLSL built-in variable (GLSL 4.60 section 7.1) that semantic_map does not list.
    generator = GLSLCodeGen()
    assert "gl_Layer" not in generator.semantic_map
    assert generator.map_semantic("gl_Layer") == "gl_Layer"


def test_call_names_resolving_to_lists_unrenamed_members_and_renaming_keys():
    mapping = {"frac": "fract", "lerp": "mix", "rint": "roundEven", "sin": "sinpi"}
    assert call_names_resolving_to(mapping, {"fract", "roundEven", "sin", "cos"}) == {
        "frac",
        "fract",
        "rint",
        "roundEven",
        "cos",
    }


def test_call_classifications_use_exact_sets():
    generator = GLSLCodeGen()
    assert generator.is_fragment_only_call_name("ddx")
    assert generator.is_fragment_only_call_name("dFdx")
    assert generator.is_fragment_only_call_name("interpolate_at_centroid")
    assert not generator.is_fragment_only_call_name("mystery")
    assert generator.glsl_fixed_array_readonly_intrinsic("lerp")
    assert generator.glsl_fixed_array_readonly_intrinsic("frac")
    assert not generator.glsl_fixed_array_readonly_intrinsic("mystery")


def test_bitcast_table_lists_every_spelling_of_the_three_bitcasts():
    assert set(GLSL_BITCAST_CANONICAL_NAMES.values()) == set(
        GLSLCodeGen.GLSL_BITCAST_FUNCTIONS
    )
    for name in GLSLCodeGen.GLSL_BITCAST_FUNCTIONS:
        assert GLSL_BITCAST_CANONICAL_NAMES[name] == name
