"""The SPIR-V importer names built-ins through exact tables and refuses a BuiltIn CrossGL does not name."""

import pytest

import crosstl
from crosstl.backend.SPIRV.import_exact_lookups import (
    SpirvBuiltinUnsupportedError,
    builtin_variable_name,
    constant_or_literal,
    keyword_safe_identifier,
)
from crosstl.backend.SPIRV.VulkanCrossGLCodeGen import VulkanToCrossGLConverter
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser

FRAGMENT_TEMPLATE = """OpCapability Shader
{capabilities}OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %flag %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %flag BuiltIn {builtin}
OpDecorate %color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_input_bool = OpTypePointer Input %bool
%ptr_output_v4float = OpTypePointer Output %v4float
%flag = OpVariable %ptr_input_bool Input
%color = OpVariable %ptr_output_v4float Output
%one = OpConstant %float 1
%zero = OpConstant %float 0
%white = OpConstantComposite %v4float %one %one %one %one
%black = OpConstantComposite %v4float %zero %zero %zero %zero
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %bool %flag
%out = OpSelect %v4float %loaded %white %black
OpStore %color %out
OpReturn
OpFunctionEnd
"""


def _parser():
    return VulkanParser(VulkanLexer("OpCapability Shader\n").tokenize())


def _write(tmp_path, builtin, capabilities=""):
    path = tmp_path / "shader.spvasm"
    path.write_text(
        FRAGMENT_TEMPLATE.format(builtin=builtin, capabilities=capabilities)
    )
    return str(path)


@pytest.mark.parametrize("backend", ["cgl", "metal", "directx", "opengl"])
def test_builtin_without_crossgl_variable_is_refused(tmp_path, backend):
    # SPV_EXT_fragment_fully_covered: the input used to become an ordinary bool named FullyCoveredEXT.
    path = _write(
        tmp_path,
        "FullyCoveredEXT",
        'OpCapability FragmentFullyCoveredEXT\nOpExtension "SPV_EXT_fragment_fully_covered"\n',
    )
    with pytest.raises(SpirvBuiltinUnsupportedError) as raised:
        crosstl.translate(path, backend=backend)
    assert raised.value.builtin == "FullyCoveredEXT"
    assert raised.value.code == "project.translate.spirv-builtin-unsupported"
    assert raised.value.missing_capabilities == ("spirv.builtin.FullyCoveredEXT",)


def test_known_builtin_keeps_its_crossgl_variable(tmp_path):
    output = crosstl.translate(_write(tmp_path, "FrontFacing"), backend="cgl")
    assert "@gl_FrontFacing" in output
    assert "@builtin(" not in output


def test_builtin_attribute_refuses_an_unnamed_builtin():
    converter = VulkanToCrossGLConverter()
    assert converter.crossgl_builtin_attribute("FragCoord", "Input") == "@gl_FragCoord"
    with pytest.raises(SpirvBuiltinUnsupportedError) as raised:
        converter.crossgl_builtin_attribute("FullyCoveredEXT", "Input")
    assert raised.value.storage_class == "Input"


def test_every_parser_builtin_has_a_crossgl_variable():
    for builtin, name in VulkanParser.SPIRV_BUILTIN_VARIABLE_NAMES.items():
        assert (
            builtin_variable_name(VulkanParser.SPIRV_BUILTIN_VARIABLE_NAMES, builtin)
            == name
        )
        assert name.startswith("gl_")


def test_type_qualifiers_keep_the_mapped_base_type():
    converter = VulkanToCrossGLConverter()
    assert converter.map_type("flat vec2") == "flat float2"
    assert converter.map_type("vec4[3]") == "float4[3]"
    assert converter.map_type("Light") == "Light"


def test_unnamed_values_avoid_crossgl_keywords():
    parser = _parser()
    assert parser.spirv_fallback_identifier("%in", "value") == "in_"
    assert parser.spirv_fallback_identifier("%struct", "value") == "struct_"
    assert parser.spirv_fallback_identifier("%light", "value") == "light"
    assert parser.spirv_fallback_identifier("%12", "value") == "value_12"


def test_constant_operands_resolve_through_the_constant_table():
    parser = _parser()
    constants = {"%four": "4"}
    assert parser.spirv_integer_constant_operand("%four", constants) == 4
    assert parser.spirv_integer_constant_operand("3", constants) == 3
    assert parser.spirv_integer_constant_operand("%spec", constants) is None


def test_helpers_are_exact():
    assert keyword_safe_identifier({"in": "in_"}, "in") == "in_"
    assert keyword_safe_identifier({"in": "in_"}, "light") == "light"
    assert constant_or_literal({"%a": "1"}, "%a") == "1"
    assert constant_or_literal({}, "2") == "2"
    with pytest.raises(SpirvBuiltinUnsupportedError):
        builtin_variable_name({}, "WarpIDNV")
