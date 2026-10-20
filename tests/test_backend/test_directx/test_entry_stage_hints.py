"""HLSL entry-point name prefixes decide a stage only for a function that can be that entry point.

Before, any function whose name began with a profile's two letters became a stage entry, so the helper
``float assignValue(float x)`` of a compute shader was imported as a ``task`` (amplification) entry point and
``float msToSeconds(float ms)`` as a mesh entry point.
"""

import textwrap

import pytest

import crosstl
from crosstl.backend.DirectX import HLSLLexer, HLSLParser
from crosstl.backend.DirectX.entry_stage_hints import (
    entry_name_prefix_stage,
    entry_name_stage_hint,
)


def parse_functions(code):
    ast = HLSLParser(HLSLLexer(textwrap.dedent(code)).tokenize()).parse()
    return {function.name: function for function in ast.functions}


@pytest.mark.parametrize(
    "name,stage",
    [
        ("VSMain", "vertex"),
        ("vs_main", "vertex"),
        ("vsMain", "vertex"),
        ("vsmain", "vertex"),
        ("VS", "vertex"),
        ("PS2", "fragment"),
        ("CSMain", "compute"),
        ("ASMain", "task"),
        ("MS_Main", "mesh"),
        ("assignValue", None),
        ("AsFloat", None),
        ("msecs", None),
        ("hsvToRgb", None),
        ("dsample", None),
        ("psi", None),
        ("vsync", None),
        ("gsum", None),
        ("main", None),
    ],
)
def test_a_profile_prefix_must_be_a_whole_word(name, stage):
    assert entry_name_prefix_stage(name) == stage


@pytest.mark.parametrize(
    "name,attributes,stage",
    [
        ("asFloat", set(), None),
        ("msToSeconds", set(), None),
        ("csCount", set(), None),
        ("gsEmit", set(), None),
        ("hsConstants", set(), None),
        ("dsWeights", set(), None),
        ("ASMain", {"numthreads"}, "task"),
        ("MSMain", {"numthreads"}, "mesh"),
        ("GSMain", {"maxvertexcount"}, "geometry"),
        ("HSMain", {"patchconstantfunc"}, "tessellation_control"),
        ("DSMain", {"domain"}, "tessellation_evaluation"),
        ("VSMain", set(), "vertex"),
    ],
)
def test_a_prefix_needs_the_attributes_its_stage_requires(name, attributes, stage):
    assert entry_name_stage_hint(name, attributes) == stage


def test_a_compute_helper_named_assign_is_not_a_task_entry():
    functions = parse_functions("""
        RWStructuredBuffer<float> outBuf : register(u0);

        float assignValue(float x) {
            return x * 2.0;
        }

        [numthreads(8, 1, 1)]
        void CSMain(uint3 id : SV_DispatchThreadID) {
            outBuf[id.x] = assignValue((float)id.x);
        }
    """)
    assert functions["assignValue"].qualifier is None
    assert "task" not in functions["assignValue"].qualifiers
    assert functions["CSMain"].qualifier == "compute"


def test_helpers_with_stage_like_names_stay_helpers():
    functions = parse_functions("""
        RWStructuredBuffer<float> outBuf : register(u0);

        float msToSeconds(float ms) { return ms * 0.001; }
        float asFloat(uint bits) { return asfloat(bits); }
        float csScale(float x) { return x * 3.0; }
        float4 VSHelper(float4 p) { return p * 2.0; }

        [numthreads(8, 1, 1)]
        void CSMain(uint3 id : SV_DispatchThreadID) {
            outBuf[id.x] = msToSeconds(asFloat(id.x)) + csScale(1.0)
                + VSHelper(float4(1.0, 1.0, 1.0, 1.0)).x;
        }
    """)
    for helper in ("msToSeconds", "asFloat", "csScale", "VSHelper"):
        assert functions[helper].qualifier is None, helper
    assert functions["CSMain"].qualifier == "compute"


def test_a_called_vertex_named_function_without_semantics_is_a_helper():
    functions = parse_functions("""
        struct PSInput {
            float4 position : SV_POSITION;
            float2 uv : TEXCOORD0;
        };

        float2 VSFlip(float2 uv) { return float2(uv.x, 1.0 - uv.y); }

        PSInput VSMain(float3 position : POSITION, float2 uv : TEXCOORD0) {
            PSInput result;
            result.position = float4(position, 1.0f);
            result.uv = VSFlip(uv);
            return result;
        }

        float4 PSMain(PSInput input) : SV_TARGET {
            return float4(input.uv, 0.0f, 1.0f);
        }
    """)
    assert functions["VSFlip"].qualifier is None
    assert functions["VSMain"].qualifier == "vertex"
    assert functions["PSMain"].qualifier == "fragment"


def test_prefixed_entry_points_keep_their_stages():
    functions = parse_functions("""
        struct VSInput {
            float3 position : POSITION;
        };
        struct PSInput {
            float4 position : SV_POSITION;
        };

        PSInput VSMain(VSInput input) {
            PSInput result;
            result.position = float4(input.position, 1.0f);
            return result;
        }

        [numthreads(32, 1, 1)]
        void ASMain(uint gid : SV_GroupID) {
        }
    """)
    assert functions["VSMain"].qualifier == "vertex"
    assert functions["ASMain"].qualifier == "task"


def test_the_imported_compute_shader_has_one_entry_point(tmp_path):
    source = tmp_path / "helper.hlsl"
    source.write_text(textwrap.dedent("""
        RWStructuredBuffer<float> outBuf : register(u0);

        float assignValue(float x) {
            return x * 2.0;
        }

        [numthreads(8, 1, 1)]
        void CSMain(uint3 id : SV_DispatchThreadID) {
            outBuf[id.x] = assignValue((float)id.x);
        }
    """))
    crossgl = crosstl.translate(str(source), backend="cgl")
    assert "task" not in crossgl
    assert crossgl.count("@ stage_entry") == 1
    glsl = crosstl.translate(str(source), backend="opengl")
    assert "float assignValue(float x)" in glsl
