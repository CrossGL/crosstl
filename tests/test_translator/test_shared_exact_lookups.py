"""Shared translator helpers look names up exactly instead of passing an unknown name through.

Before, ``normalize_stage_name`` returned any name that was not a stage alias as if it were a stage name, so a
misspelt requested stage (``generate_program(ast, target_stage="pixle")``) selected no stage and GLSL, Metal, HLSL and
WebGL returned a program without an entry point, and a Slang ``[shader("bogus")]`` entry was imported as a plain
function. The CrossGL parser also replaced a function body it could not parse with an empty one unless the caller
asked for strict bodies. Each test below fails on that code.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.arithmetic_conversions import (
    ArithmeticScalarKind,
    source_integer_shape,
)
from crosstl.translator.ast import ShaderStage
from crosstl.translator.codegen import (
    GLSLCodeGen,
    HLSLCodeGen,
    MetalCodeGen,
    WebGLCodeGen,
    WGSLCodeGen,
)
from crosstl.translator.codegen.enum_utils import enum_value_expression
from crosstl.translator.codegen.glsl_buffer_layout import std430_layout_type_name
from crosstl.translator.codegen.loop_induction import counter_integer_shape
from crosstl.translator.codegen.stage_utils import (
    should_emit_qualified_function,
    stage_matches,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import (
    COOPERATIVE_MATRIX_FIELD_LABELS,
    Parser,
)
from crosstl.translator.stage_utils import (
    UnknownShaderStageError,
    normalize_stage_name,
    requested_stage_name,
)
from crosstl.translator.validation import (
    _canonical_stage_interface_type,
    _single_value_metadata_name,
)

FRAGMENT_PROGRAM = textwrap.dedent("""
    shader S {
        struct FragmentOutput { vec4 color; }
        fragment {
            FragmentOutput main() {
                FragmentOutput output;
                output.color = vec4(1.0);
                return output;
            }
        }
    }
    """)

STAGE_EMITTERS = [GLSLCodeGen, MetalCodeGen, HLSLCodeGen, WGSLCodeGen, WebGLCodeGen]


def parse(source):
    return Parser(Lexer(source).tokens).parse()


@pytest.mark.parametrize(
    "name", ["gl_FragColor", "numthreads", "inline", "precise", "location", "kernel_"]
)
def test_a_name_that_spells_no_stage_normalizes_to_none(name):
    assert normalize_stage_name(name) is None


@pytest.mark.parametrize(
    "name,stage",
    [
        ("pixel", "fragment"),
        ("PS", "fragment"),
        ("closesthit", "ray_closest_hit"),
        (ShaderStage.COMPUTE, "compute"),
        ("ShaderStage.VERTEX", "vertex"),
    ],
)
def test_stage_names_and_aliases_still_normalize(name, stage):
    assert normalize_stage_name(name) == stage
    assert requested_stage_name(name) == stage


def test_no_requested_stage_means_every_stage():
    assert requested_stage_name(None) is None
    assert stage_matches(None, "vertex")


@pytest.mark.parametrize(
    "check",
    [
        lambda: stage_matches("pixle", "fragment"),
        lambda: should_emit_qualified_function("pixle", "fragment"),
        lambda: requested_stage_name("pixle"),
    ],
)
def test_a_misspelt_requested_stage_is_refused_by_the_stage_filters(check):
    with pytest.raises(UnknownShaderStageError) as refused:
        check()
    assert refused.value.code == "project.translate.unknown-shader-stage"


@pytest.mark.parametrize("emitter", STAGE_EMITTERS, ids=lambda cls: cls.__name__)
def test_a_misspelt_requested_stage_is_refused_by_every_stage_emitter(emitter):
    with pytest.raises(UnknownShaderStageError):
        emitter().generate_program(parse(FRAGMENT_PROGRAM), target_stage="pixle")


@pytest.mark.parametrize("emitter", STAGE_EMITTERS, ids=lambda cls: cls.__name__)
def test_a_stage_alias_still_selects_its_stage(emitter):
    output = emitter().generate_program(parse(FRAGMENT_PROGRAM), target_stage="pixel")
    assert "1.0" in output


def _slang_entry(tmp_path, stage):
    path = tmp_path / "entry.slang"
    path.write_text(textwrap.dedent(f"""
        [shader("{stage}")]
        float4 main(float4 p : SV_Position) : SV_Target
        {{
            return p;
        }}
        """))
    return path


def test_a_slang_entry_naming_no_stage_is_refused(tmp_path):
    with pytest.raises(UnknownShaderStageError):
        crosstl.translate(str(_slang_entry(tmp_path, "bogus")), backend="cgl")


def test_a_slang_pixel_entry_imports_as_a_fragment_stage(tmp_path):
    output = crosstl.translate(str(_slang_entry(tmp_path, "pixel")), backend="cgl")
    assert "fragment {" in output


def test_vector_generic_arguments_keep_their_crossgl_scalar():
    ast = parse(textwrap.dedent("""
        shader S {
            compute {
                void main() {
                    vec4<f32> a = vec4<f32>(1.0);
                    vec4<u8> b = vec4<u8>(1, 2, 3, 4);
                }
            }
        }
        """))
    body = ast.stages[ShaderStage.COMPUTE].entry_point.body.statements
    assert [statement.var_type.element_type.name for statement in body] == [
        "float",
        "u8",
    ]


def test_every_cooperative_matrix_contract_field_has_a_label():
    parser = Parser(Lexer("shader S {}").tokens)
    contract_fields = {
        "element_type",
        "rows",
        "cols",
        "scope",
        "use",
        "layout",
        "fragment_layout",
        "subgroup_size",
        "elements_per_lane",
        "fragment_provenance",
        "fragment_mapping",
        "fragment_mapping_provenance",
    }
    assert set(COOPERATIVE_MATRIX_FIELD_LABELS) == contract_fields
    assert hasattr(parser, "merge_cooperative_matrix_field")


@pytest.mark.parametrize(
    "type_name,shape",
    [
        ("int3", (ArithmeticScalarKind.SIGNED_INTEGER, 32, 3)),
        ("i16vec2", (ArithmeticScalarKind.SIGNED_INTEGER, 16, 2)),
        ("u64vec4", (ArithmeticScalarKind.UNSIGNED_INTEGER, 64, 4)),
        ("vec3", None),
        ("float2", None),
    ],
)
def test_source_integer_vector_components_are_read_exactly(type_name, shape):
    assert source_integer_shape(type_name) == shape


def test_minimum_precision_counters_are_bounded_by_sixteen_bits():
    assert counter_integer_shape("min16int") == source_integer_shape("short")
    assert counter_integer_shape("min16uint") == source_integer_shape("ushort")
    assert counter_integer_shape("uint") == source_integer_shape("uint")
    assert counter_integer_shape("Particle") is None


@pytest.mark.parametrize(
    "type_name,layout_name",
    [
        ("short", "int16_t"),
        ("half", "float16_t"),
        ("vec3", "vec3"),
        ("mat3", "mat3"),
        ("float4", "float4"),
        ("Particle", None),
        ("double", None),
    ],
)
def test_std430_layout_names_are_read_exactly(type_name, layout_name):
    assert std430_layout_type_name(type_name) == layout_name


class _EnumGenerator:
    enum_variant_constants = {"Kind::Y": "Kind_Y"}
    enum_variant_constructors = {}
    enum_variant_constructor_fields = {}
    current_expression_expected_type = None


def test_an_identifier_that_is_no_enum_value_has_no_enum_expression():
    assert enum_value_expression(_EnumGenerator(), "Kind::Y") == "Kind_Y"
    assert enum_value_expression(_EnumGenerator(), "output") is None


def test_enum_values_and_variables_still_translate_to_metal_and_hlsl():
    source = textwrap.dedent("""
        shader main {
            enum Kind { X, Y = 4, }
            compute {
                RWStructuredBuffer<int> data @buffer(0);
                void main() {
                    int value = 2;
                    data[0] = value + Kind::Y;
                }
            }
        }
        """)
    metal = MetalCodeGen().generate(parse(source))
    hlsl = HLSLCodeGen().generate(parse(source))
    for output in (metal, hlsl):
        assert "Kind_Y" in output
        assert "value + Kind_Y" in output


def test_validation_metadata_names_are_read_exactly():
    assert _single_value_metadata_name("group") == "set"
    assert _single_value_metadata_name("set") == "set"
    assert _single_value_metadata_name("readonly") is None
    assert _canonical_stage_interface_type("float3") == "vec3"
    assert _canonical_stage_interface_type("VertexData") == "VertexData"
