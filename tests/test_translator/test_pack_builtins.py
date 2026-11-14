"""GLSL packing built-ins (GLSL 4.60 section 8.4) on Metal, HLSL and SPIR-V.

Execution against the GLSL formulas is covered by the ``g_pack`` and ``g_unpack`` conformance cases; these tests
check the spelling each target receives, the typed refusals and the native checkers when they are installed.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.codegen.pack_builtins import PACK_FUNCTIONS
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

PACKING = """shader Packing {
    RWStructuredBuffer<uint> words @register(u0);
    RWStructuredBuffer<float> values @register(u1);
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            vec4 v = vec4(values[i], values[i + 1u], values[i + 2u], values[i + 3u]);
            uint a = packUnorm4x8(v) ^ packSnorm4x8(v);
            uint b = packUnorm2x16(v.xy) ^ packSnorm2x16(v.zw) ^ packHalf2x16(v.xy);
            vec4 c = unpackUnorm4x8(a) + unpackSnorm4x8(b);
            vec2 d = unpackUnorm2x16(a) + unpackSnorm2x16(b) + unpackHalf2x16(a);
            words[i] = a + b + uint(c.x + d.y);
        }
    }
}"""

DOUBLE_PACKING = """shader DoublePacking {
    RWStructuredBuffer<uint> words @register(u0);
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            double d = packDouble2x32(uvec2(i, words[i]));
            uvec2 halves = unpackDouble2x32(d);
            words[i] = halves.x ^ halves.y;
        }
    }
}"""

UNPACKED_CALL = re.compile(r"\b(?:un)?pack(?:Snorm|Unorm|Half|Double)\w*\(")


def _check(command, tmp_path):
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    if "unable to find utility" in result.stderr:
        pytest.skip("xcrun has no Metal compiler")
    assert result.returncode == 0, result.stdout + result.stderr


def test_metal_spells_packing_with_its_standard_library(tmp_path):
    metal = MetalCodeGen().generate(parse(PACKING))

    assert UNPACKED_CALL.findall(metal) == []
    for name in ("snorm4x8", "unorm4x8", "snorm2x16", "unorm2x16"):
        assert f"pack_float_to_{name}(" in metal
        assert f"unpack_{name}_to_float(uint(" in metal
    assert "as_type<uint>(half2(v.xy))" in metal
    assert "float2(as_type<half2>(uint(a)))" in metal
    if shutil.which("xcrun"):
        (tmp_path / "out.metal").write_text(metal, encoding="utf-8")
        _check(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )


def test_metal_refuses_double_packing():
    with pytest.raises(MetalLoweringRefused) as refused:
        MetalCodeGen().generate(parse(DOUBLE_PACKING))
    assert refused.value.code == "project.translate.metal-lowering-unsupported"
    assert any("packDouble2x32" in reason for reason in refused.value.reasons)
    assert any("unpackDouble2x32" in reason for reason in refused.value.reasons)


def test_hlsl_calls_helpers_with_the_glsl_formulas(tmp_path):
    hlsl = HLSLCodeGen().generate(parse(PACKING))

    assert UNPACKED_CALL.findall(hlsl) == []
    assert "uint4 q = uint4(round(saturate(v) * 255.0));" in hlsl
    assert "int4 s = int4(p << uint4(24u, 16u, 8u, 0u)) >> 24;" in hlsl
    assert "uint2 h = f32tof16(v);" in hlsl
    assert "__crossgl_packUnorm4x8((float4)(v))" in hlsl
    assert "__crossgl_unpackHalf2x16((uint)(a))" in hlsl
    # Each helper is emitted once, however many calls use it.
    assert hlsl.count("uint __crossgl_packUnorm4x8(float4 v)") == 1
    double = HLSLCodeGen().generate(parse(DOUBLE_PACKING))
    assert "asdouble(v.x, v.y)" in double and "asuint(d, lo, hi);" in double
    dxc = shutil.which("dxc")
    if dxc:
        for name, code in (("pack.hlsl", hlsl), ("double.hlsl", double)):
            (tmp_path / name).write_text(code, encoding="utf-8")
            _check(
                [dxc, "-T", "cs_6_0", "-E", "CSMain", name, "-Fo", name + ".o"],
                tmp_path,
            )


def test_spirv_uses_glsl_std_450_packing_instructions(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(PACKING))
    double = VulkanSPIRVCodeGen().generate(parse(DOUBLE_PACKING))

    instructions = set(re.findall(r"OpExtInst %\w+ %\w+ (\w+)", assembly + double))
    assert instructions == {entry[2] for entry in PACK_FUNCTIONS.values()}
    assert "OpCapability Float64" in double
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        for name, code in (("pack", assembly), ("double", double)):
            (tmp_path / f"{name}.spvasm").write_text(code, encoding="utf-8")
            _check(
                [
                    "spirv-as",
                    "--target-env",
                    "vulkan1.1",
                    f"{name}.spvasm",
                    "-o",
                    f"{name}.spv",
                ],
                tmp_path,
            )
            _check(["spirv-val", "--target-env", "vulkan1.1", f"{name}.spv"], tmp_path)


def test_spirv_refuses_a_packing_operand_of_another_shape():
    source = PACKING.replace("packUnorm4x8(v)", "packUnorm4x8(v.xy)")
    with pytest.raises(
        SpirvLoweringRefused, match="packUnorm4x8 requires a vec4 operand"
    ):
        VulkanSPIRVCodeGen().generate(parse(source))


def test_a_user_function_named_like_a_packing_built_in_is_called(tmp_path):
    source = PACKING.replace(
        "    compute {",
        "    uint packUnorm4x8(vec4 v) { return uint(v.x); }\n    compute {",
        1,
    )
    for codegen in (HLSLCodeGen, MetalCodeGen):
        code = codegen().generate(parse(source))
        assert "packUnorm4x8(v)" in code
        assert "__crossgl_packUnorm4x8" not in code
        assert "pack_float_to_unorm4x8" not in code


def test_metal_packing_functions_import_as_glsl_built_ins(tmp_path):
    source = tmp_path / "pack.metal"
    source.write_text(
        """#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* words [[buffer(0)]], device const float4* values [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    float4 v = values[i];
    uint a = pack_float_to_unorm4x8(v) ^ pack_float_to_snorm2x16(v.xy);
    float4 c = unpack_snorm4x8_to_float(a);
    float2 d = unpack_unorm2x16_to_float(a);
    words[i] = a + uint(c.x + d.y);
}
""",
        encoding="utf-8",
    )
    crossgl = crosstl.translate(str(source), backend="cgl")
    hlsl = crosstl.translate(str(source), backend="directx")

    for name in ("packUnorm4x8", "packSnorm2x16", "unpackSnorm4x8", "unpackUnorm2x16"):
        assert f"{name}(" in crossgl
        assert f"__crossgl_{name}(" in hlsl
    assert "pack_float_to" not in crossgl and "_to_float" not in crossgl
