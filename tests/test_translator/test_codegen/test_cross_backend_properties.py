import re
from dataclasses import dataclass

from hypothesis import HealthCheck, given, settings, strategies as st

import crosstl.translator
import crosstl.translator.codegen as codegen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen


IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)


@dataclass(frozen=True)
class NumericCase:
    source_type: str
    local_initializer: str
    update_expression: str


NUMERIC_CASES = (
    NumericCase("float", "a", "tmp + b"),
    NumericCase("int", "a", "tmp + b"),
    NumericCase("uint", "a", "tmp + b"),
    NumericCase("vec2", "a", "tmp + b"),
    NumericCase("vec3", "a", "tmp + b"),
    NumericCase("vec4", "a", "tmp + b"),
    NumericCase("ivec2", "a", "tmp + b"),
    NumericCase("uvec2", "a", "tmp + b"),
)


def _assert_codegen_output_is_usable(backend, generated):
    assert isinstance(generated, str)
    assert generated.strip(), f"{backend} generated empty output"
    assert "Traceback" not in generated
    assert "<crosstl." not in generated
    assert "NotImplemented" not in generated


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    numeric_case=st.sampled_from(NUMERIC_CASES),
)
def test_numeric_helper_codegen_is_total_and_deterministic_for_all_backends(
    suffix, numeric_case
):
    function_name = f"fn_{suffix}"
    shader = f"""
    shader Property_{suffix} {{
        {numeric_case.source_type} {function_name}(
            {numeric_case.source_type} a,
            {numeric_case.source_type} b
        ) {{
            {numeric_case.source_type} tmp = {numeric_case.local_initializer};
            tmp = {numeric_case.update_expression};
            return tmp;
        }}
    }}
    """

    ast = crosstl.translator.parse(shader)

    for backend in codegen.backend_names():
        first = codegen.get_codegen(backend).generate(ast)
        second = codegen.get_codegen(backend).generate(ast)
        assert first == second, f"{backend} code generation is not deterministic"
        _assert_codegen_output_is_usable(backend, first)


@settings(max_examples=20, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    loop_bound=st.integers(min_value=1, max_value=16),
    threshold=st.floats(
        min_value=-8.0,
        max_value=8.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
def test_control_flow_codegen_is_total_for_all_backends(
    suffix, loop_bound, threshold
):
    threshold_literal = f"{threshold:.6f}"
    shader = f"""
    shader ControlFlow_{suffix} {{
        float accumulate_{suffix}(float seed, float step) {{
            float acc = seed;
            for (int i = 0; i < {loop_bound}; i++) {{
                acc = acc + step;
            }}
            if (acc > {threshold_literal}) {{
                return acc;
            }} else {{
                return step;
            }}
        }}
    }}
    """

    ast = crosstl.translator.parse(shader)

    for backend in codegen.backend_names():
        generated = codegen.get_codegen(backend).generate(ast)
        _assert_codegen_output_is_usable(backend, generated)


def _texture_sampling_shader(suffix, arrayed, helper):
    array_suffix = "[4]" if arrayed else ""
    declarations = f"""
        sampler2D tex_{suffix}{array_suffix};
        sampler smp_{suffix}{array_suffix};
    """

    if helper and arrayed:
        helper_source = f"""
        vec4 sample_{suffix}(sampler2D textures[], sampler samplers[], vec2 uv) {{
            return texture(textures[2], samplers[2], uv);
        }}
        """
        sample_expression = f"sample_{suffix}(tex_{suffix}, smp_{suffix}, input.uv)"
    elif helper:
        helper_source = f"""
        vec4 sample_{suffix}(sampler2D tex, sampler smp, vec2 uv) {{
            return texture(tex, smp, uv);
        }}
        """
        sample_expression = f"sample_{suffix}(tex_{suffix}, smp_{suffix}, input.uv)"
    else:
        index = "[1]" if arrayed else ""
        helper_source = ""
        sample_expression = f"texture(tex_{suffix}{index}, smp_{suffix}{index}, input.uv)"

    return f"""
    shader TextureProperty_{suffix} {{
        {declarations}

        struct FSInput {{
            vec2 uv @ TEXCOORD0;
        }};

        {helper_source}

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {sample_expression};
            }}
        }}
    }}
    """


def _assert_no_source_sampler_leaks_to_glsl(glsl, suffix):
    assert f"smp_{suffix}" not in glsl
    assert "SamplerState" not in glsl
    assert "[[sampler" not in glsl


@settings(max_examples=20, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    arrayed=st.booleans(),
    helper=st.booleans(),
)
def test_primary_graphics_texture_sampling_contracts_are_preserved(
    suffix, arrayed, helper
):
    shader = _texture_sampling_shader(suffix, arrayed, helper)
    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate_stage(ast, "fragment")
    glsl = GLSLCodeGen().generate_stage(ast, "fragment")
    metal = MetalCodeGen().generate_stage(ast, "fragment")

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert ".Sample(" in hlsl
    assert "SamplerState" in hlsl
    assert not re.search(r"\btexture\s*\(", hlsl)

    assert re.search(r"\btexture\s*\(", glsl)
    assert f"sampler2D tex_{suffix}" in glsl
    _assert_no_source_sampler_leaks_to_glsl(glsl, suffix)

    assert ".sample(" in metal
    assert re.search(r"\bsampler\b", metal)
    assert not re.search(r"\btexture\s*\(", metal)
