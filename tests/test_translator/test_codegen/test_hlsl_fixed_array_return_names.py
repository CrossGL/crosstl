"""Name disambiguation for HLSL fixed-array return structs."""

import hashlib
import re

from crosstl.translator.ast import (
    ArrayLiteralNode,
    ArrayType,
    BlockNode,
    ExecutionModel,
    FunctionNode,
    LiteralNode,
    NamedType,
    PrimitiveType,
    ReturnNode,
    ShaderNode,
    StructMemberNode,
    StructNode,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen


def _colliding_fixed_array_return_shader():
    members = [
        StructMemberNode("x", PrimitiveType("float")),
        StructMemberNode("y", PrimitiveType("float")),
    ]
    return ShaderNode(
        "FixedArrayReturnNameCollision",
        ExecutionModel.COMPUTE_KERNEL,
        structs=[
            StructNode("Pair", list(members)),
            # A user struct already owns the default lowered struct name.
            StructNode("CrossGLFixedArray_Pair_2", list(members)),
        ],
        functions=[
            FunctionNode(
                "make_pairs",
                ArrayType(NamedType("Pair"), 2),
                [],
                BlockNode(
                    [
                        ReturnNode(
                            ArrayLiteralNode(
                                [
                                    ArrayLiteralNode(
                                        [LiteralNode(3.0, PrimitiveType("float"))]
                                    )
                                ]
                            )
                        )
                    ]
                ),
            )
        ],
    )


def test_colliding_fixed_array_return_struct_uses_sha256_content_suffix():
    codegen = HLSLCodeGen()
    generated = codegen.generate(_colliding_fixed_array_return_shader())

    match = re.search(
        r"\b(CrossGLFixedArray_Pair_2_([0-9a-f]{8})) make_pairs\(\)", generated
    )
    assert match is not None, generated
    lowered_name, suffix = match.groups()
    type_text = codegen.type_name_string(ArrayType(NamedType("Pair"), 2))
    assert suffix == hashlib.sha256(type_text.encode("utf-8")).hexdigest()[:8]
    assert f"struct {lowered_name} {{" in generated
    assert "struct CrossGLFixedArray_Pair_2 {" in generated


def test_colliding_fixed_array_return_struct_name_is_deterministic():
    first = HLSLCodeGen().generate(_colliding_fixed_array_return_shader())
    second = HLSLCodeGen().generate(_colliding_fixed_array_return_shader())
    assert first == second
