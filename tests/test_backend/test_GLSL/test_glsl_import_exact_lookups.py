"""The GLSL importer spells operators and names through exact tables and refuses an operator CrossGL lacks."""

import pytest

import crosstl
from crosstl.backend.GLSL.import_exact_lookups import (
    GLSL_ONLY_OPERATORS,
    GLSLImportOperatorError,
    crossgl_operator,
    declared_name,
    folded_operator,
)
from crosstl.backend.GLSL.OpenglAst import BinaryOpNode, UnaryOpNode, VariableNode
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter


def _import(tmp_path, source, suffix="comp"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="cgl")


def test_logical_xor_becomes_inequality_of_booleans(tmp_path):
    output = _import(
        tmp_path,
        """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { int v[]; };
bool differ(bool a, bool b) { return a ^^ b; }
void main() { v[0] = differ(v[1] > 0, v[2] > 0) ? 1 : 0; }
""",
    )
    assert "return (a != b);" in output
    assert "^^" not in output


@pytest.mark.parametrize(
    "glsl, crossgl",
    [
        ("texture2D(tex, uv)", "texture(tex, input.uv)"),
        ("texture2DLod(tex, uv, 0.0)", "textureLod(tex, input.uv, 0.0)"),
        ("texture2DProj(tex, vec3(uv, 1.0))", "textureProj(tex, vec3(input.uv, 1.0))"),
        (
            "textureGrad(tex, uv, uv, uv)",
            "textureGrad(tex, input.uv, input.uv, input.uv)",
        ),
        ("textureLod(tex, uv, 1.0)", "textureLod(tex, input.uv, 1.0)"),
    ],
)
def test_texture_functions_name_their_core_function(tmp_path, glsl, crossgl):
    output = _import(
        tmp_path,
        f"""#version 450
uniform sampler2D tex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {{ color = {glsl}; }}
""",
        suffix="frag",
    )
    assert crossgl in output


def test_every_texture_function_has_a_crossgl_function():
    converter = GLSLToCrossGLConverter()
    assert set(converter.texture_function_names) == set(
        converter.texture_function_operations
    )
    for name in converter.texture_function_operations:
        descriptor = converter.resource_function_descriptor(name)
        assert descriptor["function"] in converter.texture_function_operations


def test_renamed_function_and_clashing_block_members_keep_their_new_names(tmp_path):
    output = _import(
        tmp_path,
        """#version 450
layout(std140, binding = 0) uniform A { float scale; } a;
layout(std140, binding = 1) uniform B { float scale; } b;
layout(location = 0) out vec4 color;
float compute(float x) { return x * 2.0; }
void main() { color = vec4(compute(a.scale) + b.scale); }
""",
        suffix="frag",
    )
    # ``compute`` is a CrossGL keyword, and both blocks flatten a member named ``scale``.
    assert "float compute_(float x)" in output
    assert "float B_scale;" in output
    assert "(compute_(scale) + B_scale)" in output


def test_layout_constant_folds_through_the_operator_table(tmp_path):
    output = _import(
        tmp_path,
        """#version 450
layout(local_size_x = 2 + 2, local_size_y = 3 * 2) in;
layout(std430, binding = 0) buffer B { int v[]; };
void main() { v[0] = 1; }
""",
    )
    assert "local_size_x = 4" in output
    assert "local_size_y = 6" in output


@pytest.mark.parametrize("node_type", [BinaryOpNode, UnaryOpNode])
def test_operator_without_crossgl_spelling_is_refused(node_type):
    converter = GLSLToCrossGLConverter()
    a, b = VariableNode("", "a"), VariableNode("", "b")
    node = (
        BinaryOpNode(a, "<=>", b)
        if node_type is BinaryOpNode
        else UnaryOpNode("<=>", a)
    )
    with pytest.raises(GLSLImportOperatorError) as raised:
        converter.generate_expression(node)
    assert raised.value.operator == "<=>"
    assert raised.value.code == "project.translate.opengl-import-operator-unsupported"


def test_operator_lookups_are_exact():
    assert crossgl_operator("+") == "+"
    assert crossgl_operator("PLUS") == "+"
    assert crossgl_operator(",") == ","
    for operator, spelling in GLSL_ONLY_OPERATORS.items():
        assert crossgl_operator(operator) == spelling
    assert folded_operator("<=>") is None
    assert folded_operator(None) is None
    with pytest.raises(GLSLImportOperatorError):
        crossgl_operator("**")


def test_declared_name_reads_only_renamed_declarations():
    assert declared_name({"compute": "compute_"}, "compute") == "compute_"
    assert declared_name({"compute": "compute_"}, "shade") == "shade"
