"""A ``layout(...)`` qualifier may follow another declaration qualifier.

GLSL 4.20 (section 4.11 of the specification, "Order and Repetition of Qualification") accepts the qualifiers of
a declaration in any order. Before, the CrossGL parser skipped the ``uniform`` of ``uniform layout(rgba16f)
image2D img;`` as an unknown token, so the declaration silently lost its storage qualifier.
"""

import pytest

from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def _shader(stage_declaration, shader_declaration=""):
    return f"""
shader S {{
    {shader_declaration}
    compute {{
        layout(local_size_x = 8) in;
        {stage_declaration}
        void main() {{
            imageStore(img, ivec2(0, 0), vec4(1.0));
        }}
    }}
}}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _stage_variable(ast, name):
    for stage in ast.stages.values():
        for variable in stage.local_variables:
            if variable.name == name:
                return variable
    raise AssertionError(f"no stage variable {name}")


def _attributes(variable):
    return [
        (attribute.name, [getattr(arg, "value", arg) for arg in attribute.arguments])
        for attribute in variable.attributes
    ]


def test_qualifier_before_layout_keeps_the_storage_qualifier():
    ast = _parse(_shader("uniform layout(rgba16f) image2D img;"))
    variable = _stage_variable(ast, "img")
    assert variable.qualifiers == ["uniform"]
    assert _attributes(variable) == [("rgba16f", [])]


def test_layout_after_qualifier_matches_layout_before_qualifier():
    interleaved = _stage_variable(
        _parse(_shader("uniform layout(rgba16f) image2D img;")), "img"
    )
    leading = _stage_variable(
        _parse(_shader("layout(rgba16f) uniform image2D img;")), "img"
    )
    assert interleaved.qualifiers == leading.qualifiers
    assert _attributes(interleaved) == _attributes(leading)
    assert str(interleaved.var_type) == str(leading.var_type)


def test_shader_scope_layouts_on_both_sides_of_a_qualifier_are_merged_in_order():
    ast = _parse(
        _shader(
            "uniform layout(rgba16f) image2D img;",
            "layout(binding = 2) uniform layout(rgba8) image2D outer_img;",
        )
    )
    (variable,) = (v for v in ast.global_variables if v.name == "outer_img")
    assert variable.qualifiers == ["uniform"]
    assert _attributes(variable) == [("binding", [2]), ("rgba8", [])]


def test_layout_between_two_qualifiers_keeps_both_qualifiers():
    ast = _parse(_shader("uniform layout(r32f) readonly image2D img;"))
    variable = _stage_variable(ast, "img")
    assert "uniform" in variable.qualifiers
    assert _attributes(variable) == [("r32f", [])]
    leading = _stage_variable(
        _parse(_shader("layout(r32f) uniform readonly image2D img;")), "img"
    )
    assert variable.qualifiers == leading.qualifiers


@pytest.mark.parametrize("codegen", [GLSLCodeGen, MetalCodeGen, SlangCodeGen])
def test_both_qualifier_orders_generate_the_same_code(codegen):
    interleaved = codegen().generate(
        _parse(_shader("uniform layout(rgba16f) image2D img;"))
    )
    leading = codegen().generate(
        _parse(_shader("layout(rgba16f) uniform image2D img;"))
    )
    assert interleaved == leading


def test_layout_without_a_preceding_qualifier_still_needs_the_layout_branch():
    # The interleaved form only extends a qualifier sequence; a declaration that starts with ``layout`` keeps the
    # existing leading-layout path, so a stage layout such as ``layout(local_size_x = 8) in;`` is unaffected.
    ast = _parse(_shader("layout(rgba16f) uniform image2D img;"))
    (stage,) = ast.stages.values()
    assert stage.execution_config
