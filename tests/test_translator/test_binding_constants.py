"""Index attributes that name enumerators or integer constants keep their value on every target.

Metal samples index resources with enumerators of a header shared with the host (``[[buffer(BufferIndexColors)]]``),
and GLSL layout qualifiers take constant expressions (``layout(location = BASE + 1)``). The generators read only
integer literals: the HLSL generator assigned the next free register, the Metal generator dropped the attribute and
the GLSL generator refused or printed the AST, so the binding moved without a diagnostic.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.semantics import BindingIndexUnresolvedError
from crosstl.translator.semantics.binding_constants import fold_binding_constants

METAL_ENUM_BINDINGS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    typedef enum BufferIndex { BufferIndexUniforms = 3, BufferIndexColors = 7 } BufferIndex;
    typedef enum TextureIndex { TextureIndexBase = 2 } TextureIndex;
    struct Uniforms { float4 tint; };
    struct VertexOut { float4 position [[position]]; float2 uv; };
    vertex VertexOut vmain(uint vid [[vertex_id]],
                           constant Uniforms &u [[buffer(BufferIndexUniforms)]]) {
        VertexOut out;
        out.position = float4(float(vid), 0.0, 0.0, 1.0) * u.tint;
        out.uv = float2(0.5);
        return out;
    }
    fragment float4 fmain(VertexOut in [[stage_in]],
                          device const float4 *colors [[buffer(BufferIndexColors)]],
                          texture2d<float> base [[texture(TextureIndexBase)]],
                          sampler smp [[sampler(TextureIndexBase)]]) {
        return base.sample(smp, in.uv) * colors[0];
    }
    """)

CGL_CONSTANT_EXPRESSIONS = textwrap.dedent("""
    shader main {
        enum Slots { First, Second = 4, Third, Fourth = Second + Third * 2 };
        enum Scoped { X = 2, Y };
        const int BASE = 3;
        struct S { vec4 p; };
        vertex {
            vec4 main(StructuredBuffer<S> s @buffer(Third) @constant,
                      StructuredBuffer<S> t @buffer(Scoped::Y) @constant,
                      StructuredBuffer<S> u @buffer((BASE + 1) * 2) @constant,
                      StructuredBuffer<S> v @buffer(Fourth) @constant) @gl_Position {
                return s[0].p + t[0].p + u[0].p + v[0].p;
            }
        }
    }
    """)

GLSL_LOCATION_EXPRESSION = textwrap.dedent("""
    #version 450
    const int BASE_LOCATION = 2;
    layout(location = BASE_LOCATION + 1) in vec3 position;
    layout(location = 0) out vec4 color;
    void main() { color = vec4(position, 1.0); }
    """)


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    # Unformatted output: the formatter depends on whether clang-format and SPIRV-Tools are installed.
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _matches(code, pattern):
    """Search a regex, or an ``(OpName, decoration)`` pair resolved through the SPIR-V names."""
    if isinstance(pattern, str):
        return re.search(pattern, code)
    name, decoration = pattern
    ids = re.findall(rf'OpName (%\S+) "{name}"', code)
    return any(
        re.search(rf"OpDecorate {re.escape(i)} {decoration}\b", code) for i in ids
    )


def _crossgl_bindings(source):
    program = parse(source)
    fold_binding_constants(program, "opengl")
    return {
        parameter.name: [
            argument.value
            for attribute in parameter.attributes
            if attribute.name == "buffer"
            for argument in attribute.arguments
        ]
        for parameter in next(iter(program.stages.values())).entry_point.parameters
    }


def test_crossgl_enumerators_and_constant_expressions_fold():
    assert _crossgl_bindings(CGL_CONSTANT_EXPRESSIONS) == {
        "s": [5],
        "t": [3],
        "u": [8],
        "v": [14],
    }


@pytest.mark.parametrize(
    ("backend", "patterns"),
    [
        (
            "directx",
            [
                r"Uniforms> u : register\(b3\)",
                r"colors : register\(t7\)",
                r"base : register\(t2\)",
                r"smp : register\(s2\)",
            ],
        ),
        (
            "metal",
            [
                r"Uniforms\s*&\s*u \[\[buffer\(3\)\]\]",
                r"\*\s*colors \[\[buffer\(7\)\]\]",
                r"base \[\[texture\(2\)\]\]",
                r"smp \[\[sampler\(2\)\]\]",
            ],
        ),
        (
            "opengl",
            [
                r"binding = 3\) uniform Uniforms",
                r"binding = 7\) readonly buffer colorsBuffer",
                r"binding = 2\) uniform sampler2D base",
            ],
        ),
        ("vulkan", [("uUniform", "Binding 3"), ("colors", "Binding 7")]),
    ],
)
def test_metal_enumerator_bindings_survive(tmp_path, backend, patterns):
    code = _translate(tmp_path, METAL_ENUM_BINDINGS, ".metal", backend)
    for pattern in patterns:
        assert _matches(code, pattern), (pattern, code)


def test_metal_enumerator_bindings_compile(tmp_path):
    if not shutil.which("xcrun"):
        pytest.skip("the Metal toolchain is not installed")
    code = _translate(tmp_path, METAL_ENUM_BINDINGS, ".metal", "metal")
    (tmp_path / "a.metal").write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize(
    ("backend", "pattern"),
    [
        ("opengl", r"layout\(location = 3\) in vec3 position;"),
        ("vulkan", ("CrossGL_fragment_input_input_position", "Location 3")),
        ("directx", r"float3 position\s*: TEXCOORD3;"),
        ("metal", r"float3 position \[\[user\(locn3\)\]\];"),
    ],
)
def test_glsl_location_expression_survives(tmp_path, backend, pattern):
    code = _translate(tmp_path, GLSL_LOCATION_EXPRESSION, ".frag", backend)
    assert _matches(code, pattern), code
    assert "BinaryOpNode" not in code


def _shader(declarations, parameter_attribute):
    return textwrap.dedent(f"""
        shader main {{
            {declarations}
            struct S {{ vec4 p; }};
            vertex {{
                vec4 main(StructuredBuffer<S> s {parameter_attribute} @constant) @gl_Position {{
                    return s[0].p;
                }}
            }}
        }}
        """)


@pytest.mark.parametrize(
    ("declarations", "attribute", "message"),
    [
        ("", "@buffer(Missing)", "Missing is not an enumerator or integer constant"),
        (
            "const int SPEC @constant_id(0) = 3;",
            "@buffer(SPEC)",
            "SPEC is a specialization constant",
        ),
        ("const float SCALE = 2.0;", "@buffer(SCALE)", "SCALE is not an integer"),
        ("const int BASE = 1;", "@buffer(BASE - 2)", "evaluates to -1"),
        (
            "enum A { P = 1 }; enum B { P = 2 };",
            "@buffer(P)",
            "P names several constants",
        ),
        ("enum Loop { Q = Q + 1 };", "@buffer(Q)", "defined in terms of itself"),
    ],
)
@pytest.mark.parametrize("backend", ["opengl", "directx", "metal", "vulkan"])
def test_unresolvable_index_is_refused(
    tmp_path, backend, declarations, attribute, message
):
    with pytest.raises(BindingIndexUnresolvedError, match=re.escape(message)):
        _translate(tmp_path, _shader(declarations, attribute), ".cgl", backend)


def test_literal_and_register_spellings_are_left_to_the_generators():
    program = parse(textwrap.dedent("""
            shader main {
                cbuffer Constants @register(b1) @space(space2) { vec4 tint; };
                struct S { vec4 p; };
                vertex {
                    vec4 main(StructuredBuffer<S> s @buffer(4) @constant) @gl_Position {
                        return s[0].p * tint;
                    }
                }
            }
            """))
    assert fold_binding_constants(program, "metal") == 0
