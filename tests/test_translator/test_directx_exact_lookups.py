"""DirectX lookups that used to pass an unknown key through as its own HLSL spelling.

Each lookup now either names the HLSL spelling exactly or refuses the key:

- target profiles: every accepted spelling is listed, and any other is refused;
- bitcasts: each CrossGL spelling names its HLSL intrinsic (``asint``, ``asuint``, ``asfloat``);
- component-wise float unary calls: the set of call names is exact, so return-type inference does not treat an
  unmapped name as an HLSL intrinsic;
- semantics: an unmapped semantic is written only when it is an HLSL identifier, and semantic attributes with an
  argument (``@location(N)`` on an entry parameter, ``@user(name)``) keep that argument instead of dropping it.
"""

from __future__ import annotations

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from tests.lowering_refusal_helpers import refused


def _hlsl(source, **options):
    return HLSLCodeGen(**options).generate(parse(source))


def _signature(code, entry):
    return next(line for line in code.splitlines() if f" {entry}(" in line)


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("dx10", "dx10"),
        ("DX11", "dx11"),
        ("dx12", "dx12"),
        ("d3d_12", "dx12"),
        ("Direct3D-11", "dx11"),
        ("directx-10", "dx10"),
    ],
)
def test_target_profile_spellings_name_a_canonical_profile(spelling, canonical):
    assert HLSLCodeGen(target_profile=spelling).target_profile == canonical


@pytest.mark.parametrize("spelling", ["dx9", "dx13", "d3d", "vulkan", ""])
def test_unknown_target_profiles_are_refused(spelling):
    with pytest.raises(ValueError, match="DirectX target profile must be one of"):
        HLSLCodeGen(target_profile=spelling)


def test_every_bitcast_spelling_names_an_hlsl_intrinsic():
    assert set(HLSLCodeGen.HLSL_BITCAST_INTRINSICS) == set(
        HLSLCodeGen.HLSL_BITCAST_FUNCTION_TARGETS
    )
    assert set(HLSLCodeGen.HLSL_BITCAST_INTRINSICS.values()) == {
        "asint",
        "asuint",
        "asfloat",
    }


def test_glsl_bitcast_spellings_become_hlsl_intrinsics():
    code = _hlsl("""
        shader S {
            fragment {
                vec4 main(vec2 uv @location(0)) @gl_FragColor {
                    uint bits = floatBitsToUint(uv.x);
                    return vec4(
                        intBitsToFloat(int(bits)),
                        uintBitsToFloat(bits),
                        float(floatBitsToInt(uv.y)),
                        1.0
                    );
                }
            }
        }
        """)
    assert "asuint(" in code
    assert "asfloat(" in code
    assert "asint(" in code
    for spelling in HLSLCodeGen.HLSL_BITCAST_FUNCTION_TARGETS:
        if not spelling.startswith("as"):
            assert f"{spelling}(" not in code


def test_componentwise_float_unary_calls_match_their_hlsl_intrinsics():
    codegen = HLSLCodeGen()
    names = set(codegen.function_map) | set(
        codegen.HLSL_COMPONENTWISE_FLOAT_UNARY_FUNCTIONS
    )
    expected = {
        name
        for name in names
        if codegen.function_map.get(name, name)
        in codegen.HLSL_COMPONENTWISE_FLOAT_UNARY_FUNCTIONS
    }
    assert codegen.hlsl_componentwise_float_unary_calls == expected
    assert "unknownCall" not in codegen.hlsl_componentwise_float_unary_calls


@pytest.mark.parametrize("text", ["COLOR 0", "TEXCOORD[1]", "1st", "SV_Target(0)"])
def test_semantic_text_that_is_not_an_identifier_is_refused(text):
    codegen = HLSLCodeGen()
    assert codegen.hlsl_semantic_name(text) == text
    message = refused(DirectXLoweringRefused, codegen.refusals.raise_pending)
    assert f"'{text}' is not an HLSL semantic" in message


def test_user_and_system_semantics_are_written_as_given():
    codegen = HLSLCodeGen()
    assert codegen.hlsl_semantic_name("NORMAL1") == "NORMAL1"
    assert codegen.hlsl_semantic_name("SV_InstanceID") == "SV_InstanceID"
    codegen.refusals.raise_pending()


def test_entry_parameter_locations_name_varying_slots():
    code = _hlsl("""
        shader S {
            fragment {
                void main(vec2 uv @location(1), vec4 tint @location(3), out vec4 color @location(2)) {
                    color = vec4(uv, 0.0, 1.0) * tint;
                }
            }
        }
        """)
    signature = _signature(code, "PSMain")
    assert "float2 uv : TEXCOORD1" in signature
    assert "float4 tint : TEXCOORD3" in signature
    assert "out float4 color : SV_Target2" in signature
    assert ": location" not in code


def test_vertex_parameter_locations_match_stage_structure_members():
    code = _hlsl("""
        shader S {
            struct VSOut { vec4 pos @gl_Position; vec2 uv @location(1); }
            vertex {
                VSOut main(vec3 position @location(0), vec2 uv @location(1)) {
                    VSOut o;
                    o.pos = vec4(position, 1.0);
                    o.uv = uv;
                    return o;
                }
            }
        }
        """)
    signature = _signature(code, "VSMain")
    assert "float3 position : TEXCOORD0" in signature
    assert "float2 uv : TEXCOORD1" in signature
    assert "float2 uv: TEXCOORD1;" in code


def test_defaulted_parameters_skip_explicit_locations():
    code = _hlsl("""
        shader S {
            vertex {
                vec4 main(vec2 a @location(0), vec2 b) @gl_Position {
                    return vec4(a, b);
                }
            }
        }
        """)
    signature = _signature(code, "VSMain")
    assert "float2 a : TEXCOORD0" in signature
    assert "float2 b : TEXCOORD1" in signature


def test_inout_fragment_location_parameter_is_refused():
    source = """
    shader S {
        fragment {
            void main(inout vec4 color @location(0)) {
                color = color * 0.5;
            }
        }
    }
    """
    message = refused(DirectXLoweringRefused, _hlsl, source)
    assert "inout at a location" in message


def test_user_semantics_keep_their_names():
    code = _hlsl("""
        shader S {
            struct VSOut { vec4 pos @gl_Position; vec4 color @user(color0); }
            vertex {
                VSOut main(vec3 p @location(0), vec4 tint @user(tint1)) {
                    VSOut o;
                    o.pos = vec4(p, 1.0);
                    o.color = tint;
                    return o;
                }
            }
        }
        """)
    assert "float4 color: color0;" in code
    assert "float4 tint : tint1" in _signature(code, "VSMain")
    assert ": user" not in code


def test_user_semantic_with_system_value_prefix_is_refused():
    source = """
    shader S {
        struct VSOut { vec4 pos @gl_Position; vec4 color @user(sv_custom); }
        vertex {
            VSOut main(vec3 p @location(0)) {
                VSOut o;
                o.pos = vec4(p, 1.0);
                o.color = o.pos;
                return o;
            }
        }
    }
    """
    message = refused(DirectXLoweringRefused, _hlsl, source)
    assert "SV_ prefix is reserved for system values" in message


def test_semantic_attribute_with_an_unknown_argument_is_refused():
    source = """
    shader S {
        struct VSOut { vec4 pos @gl_Position; vec4 color @COLOR(1); }
        vertex {
            VSOut main(vec3 p @location(0)) {
                VSOut o;
                o.pos = vec4(p, 1.0);
                o.color = o.pos;
                return o;
            }
        }
    }
    """
    message = refused(DirectXLoweringRefused, _hlsl, source)
    assert "semantic '@COLOR' takes no argument in HLSL" in message


def test_specialization_constant_on_a_resource_parameter_is_not_a_semantic():
    code = _hlsl("""
        shader S {
            constant bool use_left @function_constant(2);
            compute {
                void main(StructuredBuffer<uint> left @buffer(3) @function_constant(use_left),
                          RWStructuredBuffer<uint> results @buffer(7),
                          uint i @gl_GlobalInvocationID) {
                    uint value = 0u;
                    if (use_left) { value = left[i]; }
                    results[i] = value;
                }
            }
        }
        """)
    assert "function_constant" not in code
    assert "StructuredBuffer<uint> left : register(t3)" in code


def test_helpers_receive_the_fragment_built_ins_they_read():
    source = """
    shader S {
        float shade() { return gl_FragCoord.x * 0.5; }
        float twice() { return shade() * 2.0; }
        fragment {
            vec4 main() @gl_FragColor { return vec4(twice(), 0.0, 0.0, 1.0); }
        }
    }
    """
    code = _hlsl(source)
    assert "return (shade(_crossglFragCoord) * 2.0);" in code
    assert "float4 PSMain(float4 _crossglFragCoord : SV_Position)" in code
    assert "twice(_crossglFragCoord)" in code


@pytest.mark.parametrize(
    ("stage", "entry"),
    [
        ("compute", "void main() { float v = shade(); }"),
        ("vertex", "vec4 main() @gl_Position { return vec4(shade()); }"),
    ],
)
def test_fragment_built_in_without_a_value_in_the_caller_is_refused(stage, entry):
    # The CrossGL name of the built-in used to be passed as the argument: `shade(gl_FragCoord)`.
    source = f"""
    shader S {{
        float shade() {{ return gl_FragCoord.x * 0.5; }}
        {stage} {{
            {entry}
        }}
    }}
    """
    message = refused(DirectXLoweringRefused, _hlsl, source)
    assert "'shade' reads fragment built-in 'gl_FragCoord'" in message


def test_renamed_user_functions_are_declared_and_called_by_one_name():
    source = """
    shader S {
        float line(float x) { return x * 2.0; }
        float half3(float x) { return x + 1.0; }
        fragment {
            vec4 main() @gl_FragColor { return vec4(line(1.0), half3(2.0), 0.0, 1.0); }
        }
    }
    """
    code = _hlsl(source)
    assert "float line_(float x) {" in code and "float half3_(float x) {" in code
    assert "return float4(line_(1.0), half3_(2.0), 0.0, 1.0);" in code


_HOISTED_TILE = """
#include <metal_stdlib>
using namespace metal;
constant int N = 2;
kernel void k(device float* out [[buffer(0)]],
              uint tid [[thread_position_in_threadgroup]]) {{
    {local}
    threadgroup float tile[N];
    tile[tid] = 1.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[tid] = tile[0];
}}
"""


def _hoisted_tile(tmp_path, local):
    import crosstl

    path = tmp_path / "tile.metal"
    path.write_text(_HOISTED_TILE.format(local=local))
    return crosstl.translate(str(path), backend="directx")


def test_hoisted_groupshared_extent_keeps_the_global_it_names(tmp_path):
    assert "groupshared float k_tile[N];" in _hoisted_tile(tmp_path, "")


def test_hoisted_groupshared_extent_takes_a_local_constant_value(tmp_path):
    assert "groupshared float k_tile[4];" in _hoisted_tile(tmp_path, "const int N = 4;")


def test_hoisted_groupshared_extent_naming_an_unknown_local_is_refused(tmp_path):
    # Valid Metal the generator cannot fold. At global scope `N` names the global 2, not the local 4; the
    # extent used to be written unchanged, so threads 2 and 3 wrote past a two-element array.
    with pytest.raises(DirectXLoweringRefused, match="its extent names 'N'"):
        _hoisted_tile(tmp_path, "constexpr int N = int(4.0f);")
