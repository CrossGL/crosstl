import pytest

from crosstl.translator.ast import (
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.generic_function_utils import (
    GenericMemberCallSpecializationError,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen


def mlx_shaped_generic_member_call():
    receiver = MemberAccessNode(IdentifierNode("self"), "Atile")
    return FunctionCallNode(
        MemberAccessNode(receiver, "load"),
        [IdentifierNode("As")],
        generic_args=[
            PrimitiveType("float"),
            LiteralNode(1, PrimitiveType("int")),
            LiteralNode(1, PrimitiveType("int")),
            LiteralNode(36, PrimitiveType("int")),
            LiteralNode(1, PrimitiveType("int")),
        ],
        source_location={"line": 7, "column": 9},
    )


@pytest.mark.parametrize(
    ("target_name", "generate"),
    [
        ("DirectX", lambda call: HLSLCodeGen().generate_expression(call)),
        ("OpenGL", lambda call: GLSLCodeGen().generate_expression(call)),
        (
            "Vulkan SPIR-V",
            lambda call: VulkanSPIRVCodeGen().process_expression(call),
        ),
    ],
)
def test_unmaterialized_generic_member_call_fails_with_structured_diagnostic(
    target_name, generate
):
    call = mlx_shaped_generic_member_call()

    with pytest.raises(GenericMemberCallSpecializationError) as exc_info:
        generate(call)

    error = exc_info.value
    assert error.project_diagnostic_code == (
        "project.translate.generic-member-call-unresolved"
    )
    assert error.missing_capabilities == ("generic.member-call-specialization",)
    assert error.target_name == target_name
    assert error.receiver == "self.Atile"
    assert error.method_name == "load"
    assert error.generic_arguments == ("float", "1", "1", "36", "1")
    assert error.source_location == {"line": 7, "column": 9}
    assert "self.Atile.load<float, 1, 1, 36, 1>" in str(error)


def test_nongeneric_member_calls_remain_ordinary_target_calls():
    call = FunctionCallNode(
        MemberAccessNode(IdentifierNode("tile"), "load"),
        [IdentifierNode("source")],
    )

    assert HLSLCodeGen().generate_expression(call) == "tile.load(source)"
    assert GLSLCodeGen().generate_expression(call) == "tile.load(source)"
