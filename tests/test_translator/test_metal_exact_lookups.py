"""Metal semantics, SPIR-V built-ins and identifiers are looked up exactly, and unknown ones are refused.

The Metal compiler ignores an attribute it does not know with only a warning (``-Wunknown-attributes``), so a
semantic written as ``[[bogus]]`` compiled and silently lost its meaning.
"""

import pytest

import crosstl
from crosstl.backend.SPIRV.VulkanAst import LayoutNode
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_exact_lookups import (
    SPIRV_BUILTIN_SEMANTICS,
    is_metal_attribute_spelling,
)
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused


def _metal(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


COMPUTE_PARAMETER = """
shader S {
    compute {
        layout(local_size_x = 1) in;
        void main(uvec3 id @ SEMANTIC) { uint x = id.x; }
    }
}
"""


def test_an_unknown_semantic_on_a_compute_parameter_is_refused(tmp_path):
    source = COMPUTE_PARAMETER.replace("SEMANTIC", "bogus_attr")
    with pytest.raises(
        MetalLoweringRefused, match="semantic 'bogus_attr' names no Metal attribute"
    ):
        _metal(tmp_path, source)


def test_a_metal_attribute_spelling_is_kept_on_a_compute_parameter(tmp_path):
    source = COMPUTE_PARAMETER.replace("SEMANTIC", "threads_per_grid")
    assert "uint3 id [[threads_per_grid]]" in _metal(tmp_path, source)


def test_an_unknown_semantic_on_a_helper_function_is_refused(tmp_path):
    source = """
shader S {
    float h(float a) @ bogus_attr { return a; }
    compute {
        layout(local_size_x = 1) in;
        void main() { float x = h(1.0); }
    }
}
"""
    with pytest.raises(MetalLoweringRefused, match="'bogus_attr'"):
        _metal(tmp_path, source)


def test_a_user_semantic_vertex_input_takes_the_next_attribute(tmp_path):
    source = """
shader S {
    vertex {
        vec4 main(uint v @ bogus_attr) @ gl_Position { return vec4(float(v)); }
    }
}
"""
    output = _metal(tmp_path, source)
    assert "uint v [[attribute(0)]];" in output
    assert "bogus_attr" not in output


@pytest.mark.parametrize(
    "text",
    [
        "position",
        "[[position]]",
        "threads_per_grid",
        "visible",
        "accept_intersection",
        "attribute(0)",
        "attribute(12)",
        "color(1)",
        "color(0), index(1)",
        "user(locn0)",
        "depth(any)",
        "depth(less)",
        "buffer(3)",
        "raster_order_group(0)",
        "function_constant(use_fog)",
        "sample_mask, post_depth_coverage",
    ],
)
def test_metal_attribute_spellings(text):
    assert is_metal_attribute_spelling(text)


@pytest.mark.parametrize(
    "text",
    [
        "bogus_attr",
        "row_major",
        "gl_DrawID",
        "in_Position",
        "TEXCOORD0",
        "attribute(x)",
        "attribute(-1)",
        "depth(sometimes)",
        "user(1a)",
        "position]]",
    ],
)
def test_names_that_are_not_metal_attributes(text):
    assert not is_metal_attribute_spelling(text)


def _spirv_input(builtin):
    return LayoutNode(
        [("builtin", builtin)],
        layout_type="IN",
        data_type="int",
        variable_name="v",
    )


@pytest.mark.parametrize(
    "builtin, attribute",
    [
        ("VertexIndex", " [[vertex_id]]"),
        ("InstanceIndex", " [[instance_id]]"),
        ("FrontFacing", " [[front_facing]]"),
        ("SampleId", " [[sample_id]]"),
        ("Position", " [[position]]"),
    ],
)
def test_spirv_builtin_inputs_take_their_metal_attribute(builtin, attribute):
    generator = MetalCodeGen()
    assert generator.spirv_stage_input_attribute(_spirv_input(builtin)) == attribute
    assert generator.refusals.reasons == []


@pytest.mark.parametrize("builtin", ["DrawIndex", "CullDistance", "NotABuiltIn"])
def test_spirv_builtins_without_a_metal_attribute_are_refused(builtin):
    generator = MetalCodeGen()
    generator.spirv_stage_input_attribute(_spirv_input(builtin))
    assert generator.refusals.reasons


def test_every_spirv_builtin_maps_to_a_gl_name():
    assert all(name.startswith("gl_") for name in SPIRV_BUILTIN_SEMANTICS.values())


@pytest.mark.parametrize(
    "name", ["x", "filter::linear", "metal::fast", "nullptr", "true", "half"]
)
def test_identifiers_metal_spells_unchanged(name):
    generator = MetalCodeGen()
    assert generator.metal_local_identifier_name(name) == name
    assert generator.refusals.reasons == []


@pytest.mark.parametrize("name", ["device", "constant", "kernel", "a b", "x[0]", "1x"])
def test_identifiers_metal_cannot_spell_are_refused(name):
    generator = MetalCodeGen()
    generator.metal_local_identifier_name(name)
    assert generator.refusals.reasons
