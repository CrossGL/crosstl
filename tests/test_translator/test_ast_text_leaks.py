"""Generated code that prints an AST node is refused with a typed diagnostic instead of being returned."""

import pytest

import crosstl
from crosstl.translator.codegen.ast_text_leaks import (
    CHECKED_TARGETS,
    GeneratedAstTextError,
    refuse_ast_text,
)
from crosstl.translator.codegen.metal_codegen import MetalCodeGen

SHADER = """
shader S {
    fragment {
        vec4 color @output @location(0);
        void main() {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    }
}
"""


@pytest.mark.parametrize(
    "code",
    [
        "float2 loc = ConstructorNode(constructor_type = NamedType(name = vec<T, 2>, generic_args = []), arguments = 2);",
        "return texture(FunctionCallNode(function=IdentifierNode(name=sampler2D), arguments=2), uv);",
        "x = <crosstl.translator.ast.SwitchNode object at 0x10abc>;",
        "float v = PrimitiveType(name=float, size_bits=None);",
    ],
)
@pytest.mark.parametrize("target", sorted(CHECKED_TARGETS))
def test_printed_ast_nodes_are_refused(code, target):
    with pytest.raises(GeneratedAstTextError) as error:
        refuse_ast_text(f"void main() {{\n    {code}\n}}\n", target)
    assert error.value.code.endswith(f"{target}-ast-node-unlowered")
    assert error.value.node_text.split("(")[0].lstrip("<") in code


@pytest.mark.parametrize(
    "code",
    [
        "float4 c = LightNode(1.0);",
        "f(x = 3);",
        "SceneNode node = SceneNode(transform);",
        "// ConstructorNode is mentioned in a comment without arguments",
    ],
)
def test_target_code_that_only_looks_like_a_node_passes(code):
    refuse_ast_text(code, "metal")


@pytest.mark.parametrize("target", ["vulkan", "mojo", "cgl"])
def test_targets_where_the_shape_is_legal_are_not_checked(target):
    refuse_ast_text("IdentifierNode(name=x)", target)


def test_translate_refuses_an_emitter_that_prints_a_node(tmp_path, monkeypatch):
    path = tmp_path / "shader.cgl"
    path.write_text(SHADER)
    generate = MetalCodeGen.generate

    def leaky_generate(self, ast, *args, **kwargs):
        code = generate(self, ast, *args, **kwargs)
        return code.replace(
            "float4(1.0, 0.0, 0.0, 1.0)", "IdentifierNode(name=unlowered)"
        )

    monkeypatch.setattr(MetalCodeGen, "generate", leaky_generate)
    with pytest.raises(GeneratedAstTextError) as error:
        crosstl.translate(str(path), backend="metal", format_output=False)
    assert error.value.node_text.startswith("IdentifierNode(name=unlowered)")


def test_translate_returns_clean_code(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(SHADER)
    code = crosstl.translate(str(path), backend="metal", format_output=False)
    assert "float4(1.0, 0.0, 0.0, 1.0)" in code
