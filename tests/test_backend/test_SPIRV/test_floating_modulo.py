"""SPIR-V floating modulo keeps its rounding direction through CrossGL."""

from crosstl.backend.SPIRV import VulkanCrossGLCodeGen
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser

FLOATING_MODULO_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_value %out_color
OpExecutionMode %main OriginUpperLeft
OpName %in_value "inValue"
OpName %out_color "outColor"
OpDecorate %in_value Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_v4float = OpTypePointer Output %v4float
%one = OpConstant %float 1.0
%in_value = OpVariable %ptr_input_v2float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%entry = OpLabel
%loaded = OpLoad %v2float %in_value
%x = OpCompositeExtract %float %loaded 0
%y = OpCompositeExtract %float %loaded 1
%floored = OpFMod %float %x %y
%truncated = OpFRem %float %x %y
%outv = OpCompositeConstruct %v4float %floored %truncated %one %one
OpStore %out_color %outv
OpReturn
OpFunctionEnd
"""


def _crossgl(assembly):
    ast = VulkanParser(VulkanLexer(assembly).tokenize()).parse()
    return VulkanCrossGLCodeGen.VulkanToCrossGLConverter().generate(ast)


def test_fmod_imports_as_floored_mod_and_frem_as_remainder():
    crossgl = _crossgl(FLOATING_MODULO_ASSEMBLY)

    # OpFMod takes the sign of the divisor, which is CrossGL mod (x - y * floor(x / y)).
    assert "mod(inValue[0], inValue[1])" in crossgl
    # OpFRem takes the sign of the dividend, which is the truncated remainder %.
    assert "(inValue[0] % inValue[1])" in crossgl
    assert crossgl.count("%") == 1
