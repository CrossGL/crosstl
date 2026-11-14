"""GLSL bit-field built-ins and ``mix`` with a boolean selector on Metal, HLSL, SPIR-V, GLSL and WGSL.

Execution is covered by the ``g_bitfield`` and ``g_mix_select`` conformance cases; these tests check the spelling
each target receives, the typed refusals and the native checkers when they are installed.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen

BITFIELDS = """shader Bitfields {
    RWStructuredBuffer<uint> words @register(u0);
    RWStructuredBuffer<int> ints @register(u1);
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            uint e = bitfieldExtract(words[i], 3, 5);
            int s = bitfieldExtract(ints[i], int(i % 8u), 4);
            uint n = bitfieldInsert(words[i], i, 4, 8);
            ivec2 m = bitfieldInsert(ivec2(ints[i]), ivec2(3), 2, 3);
            words[i] = e + uint(s) + n + uint(m.y);
        }
    }
}"""

SELECTS = """shader Selects {
    RWStructuredBuffer<float> values @register(u0);
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            bool c = values[i] > 0.5;
            float f = mix(values[i], 2.0, c);
            vec3 g = mix(vec3(values[i]), vec3(1.0), bvec3(c, false, true));
            vec2 h = mix(vec2(values[i]), vec2(0.0), c);
            float w = mix(values[i], 2.0, 0.25);
            values[i] = f + g.z + h.y + w;
        }
    }
}"""

BITFIELD_CALL = re.compile(r"\bbitfield(?:Extract|Insert)\(")


def _check(command, tmp_path):
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    if "unable to find utility" in result.stderr:
        pytest.skip("xcrun has no Metal compiler")
    assert result.returncode == 0, result.stdout + result.stderr


def _metal_compiles(code, tmp_path):
    if shutil.which("xcrun"):
        (tmp_path / "out.metal").write_text(code, encoding="utf-8")
        _check(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )


def _hlsl_compiles(code, tmp_path):
    dxc = shutil.which("dxc")
    if dxc:
        (tmp_path / "out.hlsl").write_text(code, encoding="utf-8")
        _check(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", "-HV", "2021", "out.hlsl"], tmp_path
        )


def _spirv_validates(code, tmp_path):
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(code, encoding="utf-8")
        _check(
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
            tmp_path,
        )
        _check(["spirv-val", "--target-env", "vulkan1.1", "out.spv"], tmp_path)


def test_metal_spells_bit_fields_with_extract_and_insert_bits(tmp_path):
    metal = MetalCodeGen().generate(parse(BITFIELDS))

    assert BITFIELD_CALL.findall(metal) == []
    assert "extract_bits(words[i], uint(3), uint(5))" in metal
    assert "insert_bits(int2(ints[i]), int2(3), uint(2), uint(3))" in metal
    _metal_compiles(metal, tmp_path)


def test_hlsl_bit_field_helpers_never_shift_by_the_full_width(tmp_path):
    hlsl = HLSLCodeGen().generate(parse(BITFIELDS))

    assert BITFIELD_CALL.findall(hlsl) == []
    assert (
        "return (value >> offset) & (bits == 0u ? 0u : (0xFFFFFFFFu >> (32u - bits)));"
        in hlsl
    )
    assert (
        "asint(asuint(value) << ((32u - offset - bits) & 31u)) >> ((32u - bits) & 31u)"
        in hlsl
    )
    assert "return bits == 0u ? (int)0 : field;" in hlsl
    assert (
        "int2 __crossgl_bitfieldInsert_int2(int2 base, int2 insert, uint offset, uint bits)"
        in hlsl
    )
    _hlsl_compiles(hlsl, tmp_path)


def test_spirv_uses_the_bit_field_instructions(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(BITFIELDS))

    opcodes = re.findall(r"= (OpBitField\w+) ", assembly)
    assert sorted(opcodes) == [
        "OpBitFieldInsert",
        "OpBitFieldInsert",
        "OpBitFieldSExtract",
        "OpBitFieldUExtract",
    ]
    _spirv_validates(assembly, tmp_path)


def test_bit_fields_of_floating_point_values_are_refused():
    source = BITFIELDS.replace(
        "bitfieldExtract(words[i], 3, 5)", "uint(bitfieldExtract(1.5, 3, 5))"
    )
    with pytest.raises(
        DirectXLoweringRefused, match="bitfieldExtract requires a 32-bit integer"
    ):
        HLSLCodeGen().generate(parse(source))
    with pytest.raises(
        SpirvLoweringRefused, match="bitfieldExtract requires a 32-bit integer"
    ):
        VulkanSPIRVCodeGen().generate(parse(source))


def test_metal_bit_field_functions_import_as_glsl_built_ins(tmp_path):
    source = tmp_path / "bits.metal"
    source.write_text(
        """#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* words [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    uint e = extract_bits(words[i], 3u, 5u);
    words[i] = insert_bits(e, i, 4u, 8u);
}
""",
        encoding="utf-8",
    )
    crossgl = crosstl.translate(str(source), backend="cgl")
    hlsl = crosstl.translate(str(source), backend="directx")

    assert "bitfieldExtract(" in crossgl and "bitfieldInsert(" in crossgl
    assert "extract_bits" not in crossgl and "insert_bits" not in crossgl
    assert "__crossgl_bitfieldExtract_uint(" in hlsl


def test_boolean_mix_selects_on_every_target(tmp_path):
    glsl = GLSLCodeGen().generate(parse(SELECTS))
    metal = MetalCodeGen().generate(parse(SELECTS))
    hlsl = HLSLCodeGen().generate(parse(SELECTS))
    assembly = VulkanSPIRVCodeGen().generate(parse(SELECTS))

    # The selector is never converted to a weight, which would turn an unselected infinity into NaN.
    assert "float(c)" not in glsl and "vec3(bvec3" not in glsl
    assert "mix(values[i], 2.0, c)" in glsl
    assert "mix(vec3(values[i]), vec3(1.0), bvec3(c, false, true))" in glsl
    assert "mix(vec2(values[i]), vec2(0.0), bvec2(c))" in glsl
    assert "select(values[i], 2.0, c)" in metal
    assert "select(float2(values[i]), float2(0.0), bool2(c))" in metal
    assert "select(c, 2.0, values[i])" in hlsl
    assert "select((bool2)(c), float2(0.0, 0.0), float2(values[i], values[i]))" in hlsl
    # A numeric weight keeps the interpolating form.
    assert "mix(values[i], 2.0, 0.25)" in metal and "lerp(values[i], 2.0, 0.25)" in hlsl
    assert assembly.count("OpSelect") >= 3
    _metal_compiles(metal, tmp_path)
    _hlsl_compiles(hlsl, tmp_path)
    _spirv_validates(assembly, tmp_path)
    if shutil.which("glslangValidator"):
        (tmp_path / "out.comp").write_text(glsl, encoding="utf-8")
        _check(["glslangValidator", "out.comp"], tmp_path)


def test_wgsl_boolean_mix_is_select(tmp_path):
    # WGSL's mix takes only a floating-point weight; select(f, t, cond) keeps the argument order.
    source = SELECTS.replace(
        "void main(uint i @gl_GlobalInvocationID) {",
        "void main(uvec3 id @gl_GlobalInvocationID) {\n            uint i = id.x;",
    )
    wgsl = WGSLCodeGen().generate(parse(source))

    assert "select(values[i], 2.0, c)" in wgsl
    assert (
        "select(vec3<f32>(values[i]), vec3<f32>(1.0), vec3<bool>(c, false, true))"
        in wgsl
    )
    # WGSL's select accepts a scalar condition with vector values, so it is not splatted.
    assert "select(vec2<f32>(values[i]), vec2<f32>(0.0), c)" in wgsl
    assert "mix(values[i], 2.0, 0.25)" in wgsl
    _naga_validates(wgsl, tmp_path)


def test_wgsl_boolean_mix_with_an_inline_comparison_selector(tmp_path):
    source = """shader Inline {
    RWStructuredBuffer<vec3> values @register(u0);
    compute {
        @numthreads(64, 1, 1)
        void main(uvec3 id @gl_GlobalInvocationID) {
            uint i = id.x;
            vec3 a = values[i];
            vec3 b = vec3(1.0);
            vec3 r = mix(a, b, a.x > 0.5);
            vec3 s = mix(a, b, lessThan(a, b));
            vec3 t = mix(normalize(a), normalize(b), lessThan(a, b));
            values[i] = r + s + t;
        }
    }
}"""
    wgsl = WGSLCodeGen().generate(parse(source))

    assert "mix(" not in wgsl
    assert wgsl.count("select(a, b, ") == 2
    # Values with no recorded type are not taken to be scalars and refused.
    assert "select(normalize(a), normalize(b), (a < b))" in wgsl
    _naga_validates(wgsl, tmp_path)


def test_wgsl_boolean_mix_with_a_selector_of_another_width_is_refused():
    source = SELECTS.replace(
        "void main(uint i @gl_GlobalInvocationID) {",
        "void main(uvec3 id @gl_GlobalInvocationID) {\n            uint i = id.x;",
    ).replace("bvec3(c, false, true)", "bvec2(c, false)")

    with pytest.raises(ValueError, match="2-component selector and 3-component values"):
        WGSLCodeGen().generate(parse(source))


def _naga_validates(code, tmp_path):
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(code, encoding="utf-8")
        _check(["naga", "out.wgsl"], tmp_path)
