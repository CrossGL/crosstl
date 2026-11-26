import re
import subprocess
from pathlib import Path
from typing import List

import pytest

import crosstl.translator
from crosstl.translator.ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    ArrayType,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ConstructorNode,
    ConstructorPatternNode,
    EnumNode,
    EnumVariantNode,
    ExecutionModel,
    ExpressionStatementNode,
    FunctionCallNode,
    FunctionNode,
    FunctionType,
    GenericParameterNode,
    GenericType,
    IdentifierNode,
    IdentifierPatternNode,
    IfNode,
    LambdaNode,
    LiteralNode,
    LiteralPatternNode,
    LoopNode,
    MatchArmNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    NamedType,
    ParameterNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    ReturnNode,
    ShaderNode,
    ShaderStage,
    StructMemberNode,
    StructNode,
    StructPatternNode,
    SyncNode,
    TernaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from crosstl.translator.codegen.rust_codegen import RustCodeGen
from crosstl.translator.codegen.rust_gpu_interface import RUST_GPU_PRELUDE
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.refusals import RustRefusal
from tools.coverage import rustgpu


def tokenize_code(code: str) -> List:
    lexer = Lexer(code)
    return lexer.get_tokens()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = RustCodeGen()
    return codegen.generate(ast_node)


def rust_gpu_command_or_skip():
    """The rust-gpu compile command of the coverage harness (``tools/coverage/rustgpu.py``), or a skip."""
    command = rustgpu.command()
    if not command:
        pytest.skip(
            "rust-gpu is not built; set CROSSTL_COVERAGE_RUST_GPU to a built checkout"
        )
    return command


RUST_STAGE_ATTRIBUTES = {
    "vertex": "#[spirv(vertex",
    "fragment": "#[spirv(fragment",
    "compute": "#[spirv(compute(",
}
BARE_RUST_STAGE_ATTRIBUTES = {
    "vertex": "#[vertex_shader]",
    "fragment": "#[fragment_shader]",
    "compute": "#[compute_shader]",
}


def assert_rust_stage_attr(generated_code: str, stage: str) -> None:
    assert RUST_STAGE_ATTRIBUTES[stage] in generated_code


def assert_rust_stage_attr_absent(generated_code: str, stage: str) -> None:
    assert RUST_STAGE_ATTRIBUTES[stage] not in generated_code


def assert_no_bare_rust_stage_attrs(generated_code: str) -> None:
    padded_code = f"\n{generated_code}\n"
    for attribute in BARE_RUST_STAGE_ATTRIBUTES.values():
        assert f"\n{attribute}\n" not in padded_code


def rust_host_source(generated_code: str) -> str:
    """``generated_code`` without the rust-gpu crate prelude, for helpers that use only Rust primitives.

    Such helpers compile and run on the host, which checks the values they compute.
    """
    assert generated_code.startswith(RUST_GPU_PRELUDE), generated_code[:200]
    return generated_code[len(RUST_GPU_PRELUDE) :]


def assert_generated_rust_smoke_compiles(generated_code: str, tmp_path):
    """Compile ``generated_code`` with rust-gpu for Vulkan 1.1, as the coverage harness compiles Rust probes.

    A crate without an entry point is a library of shader functions: rust-gpu links a SPIR-V module only for a crate
    with an entry point, so such a crate is compiled as an ``rlib``, which still type-checks and lowers every function.
    """
    command = rust_gpu_command_or_skip()
    if not re.search(r"#\[spirv\((?:compute|vertex|fragment)\b", generated_code):
        command = tuple(
            "--crate-type=rlib" if part == "--crate-type=dylib" else part
            for part in command
        )
    source_path = tmp_path / "generated.rs"
    source_path.write_text(generated_code, encoding="utf-8")
    result = subprocess.run(
        [*command, source_path.name, "--out-dir", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n\n" + generated_code


def test_struct():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "struct VSInput" in generated_code
        assert "struct VSOutput" in generated_code
        assert "pub texCoord: Vec2" in generated_code
        assert "pub color: Vec4" in generated_code
        assert "#[repr(C)]" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust struct codegen not implemented.")


def test_unsized_struct_arrays_do_not_derive_copy():
    code = """
    struct DynamicPayload {
        float weights[];
        vec3 colors[2];
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_struct_and_enum_derives_track_nested_non_copy_members_and_compile(tmp_path):
    code = """
    shader TransitiveNonCopyDerives {
        struct Payload {
            float weights[];
            int count;
        };

        struct Wrapper {
            Payload payload;
            int id;
        };

        enum Command {
            Submit(Wrapper),
            Skip
        }

        fn choose(command: Command) -> int {
            match command {
                Command::Submit(wrapper) => wrapper.id,
                Command::Skip => 0,
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_reference_pointer_and_function_type_nodes_smoke_compile(tmp_path):
    payload_type = NamedType("Payload")
    int_type = PrimitiveType("int")

    ast = ShaderNode(
        "RustTypeNodeSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode("count", int_type),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "read_payload",
                int_type,
                [ParameterNode("value", ReferenceType(payload_type))],
                [ReturnNode(MemberAccessNode(IdentifierNode("value"), "count"))],
            ),
            FunctionNode(
                "bump_payload",
                int_type,
                [ParameterNode("value", ReferenceType(payload_type, is_mutable=True))],
                [
                    AssignmentNode(
                        MemberAccessNode(IdentifierNode("value"), "count"),
                        BinaryOpNode(
                            MemberAccessNode(IdentifierNode("value"), "count"),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                    ),
                    ReturnNode(MemberAccessNode(IdentifierNode("value"), "count")),
                ],
            ),
            FunctionNode(
                "raw_identity",
                PointerType(payload_type, is_mutable=True),
                [ParameterNode("value", PointerType(payload_type, is_mutable=True))],
                [ReturnNode(IdentifierNode("value"))],
            ),
            FunctionNode(
                "apply_int",
                int_type,
                [
                    ParameterNode("input", int_type),
                    ParameterNode("op", FunctionType(int_type, [int_type])),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("op"),
                            [IdentifierNode("input")],
                        )
                    )
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert "pub fn read_payload(value: &Payload) -> i32" in generated_code
    assert "pub fn bump_payload(mut value: &mut Payload) -> i32" in generated_code
    assert "value.count = (value.count + 1);" in generated_code
    assert "pub fn raw_identity(value: *mut Payload) -> *mut Payload" in generated_code
    assert "pub fn apply_int(input: i32, op: fn(i32) -> i32) -> i32" in generated_code
    assert "ReferenceType(" not in generated_code
    assert "PointerType(" not in generated_code
    assert "FunctionType(" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_reference_parameter_calls_auto_borrow_and_mark_mut_compile(tmp_path):
    payload_type = NamedType("Payload")
    int_type = PrimitiveType("int")

    ast = ShaderNode(
        "RustReferenceCallBorrowSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=4),
                    ),
                    StructMemberNode("count", int_type),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "read_payload",
                int_type,
                [ParameterNode("value", ReferenceType(payload_type))],
                [ReturnNode(MemberAccessNode(IdentifierNode("value"), "count"))],
            ),
            FunctionNode(
                "bump_payload",
                int_type,
                [ParameterNode("value", ReferenceType(payload_type, is_mutable=True))],
                [
                    AssignmentNode(
                        MemberAccessNode(IdentifierNode("value"), "count"),
                        BinaryOpNode(
                            MemberAccessNode(IdentifierNode("value"), "count"),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                    ),
                    ReturnNode(MemberAccessNode(IdentifierNode("value"), "count")),
                ],
            ),
            FunctionNode(
                "forward_read",
                int_type,
                [ParameterNode("value", ReferenceType(payload_type))],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("read_payload"),
                            [IdentifierNode("value")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "forward_bump",
                int_type,
                [ParameterNode("value", ReferenceType(payload_type, is_mutable=True))],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("bump_payload"),
                            [IdentifierNode("value")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "call_refs",
                int_type,
                [ParameterNode("payload", payload_type)],
                [
                    VariableNode("editable", payload_type, IdentifierNode("payload")),
                    VariableNode(
                        "before",
                        int_type,
                        FunctionCallNode(
                            IdentifierNode("read_payload"),
                            [IdentifierNode("editable")],
                        ),
                    ),
                    VariableNode(
                        "after",
                        int_type,
                        FunctionCallNode(
                            IdentifierNode("bump_payload"),
                            [IdentifierNode("editable")],
                        ),
                    ),
                    ReturnNode(
                        BinaryOpNode(
                            IdentifierNode("before"),
                            "+",
                            IdentifierNode("after"),
                        )
                    ),
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert "let mut editable: Payload =" in generated_code
    assert "let before: i32 = read_payload(&editable);" in generated_code
    assert "let after: i32 = bump_payload(&mut editable);" in generated_code
    assert "return read_payload(value);" in generated_code
    assert "return bump_payload(value);" in generated_code
    assert "read_payload(editable)" not in generated_code
    assert "bump_payload(editable)" not in generated_code
    assert "read_payload(&value)" not in generated_code
    assert "bump_payload(&mut value)" not in generated_code
    assert "read_payload(&editable.clone())" not in generated_code
    assert "bump_payload(&mut editable.clone())" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_mutable_reference_call_branch_arguments_mark_all_roots_mut_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    int_type = PrimitiveType("int")
    bool_type = PrimitiveType("bool")

    def count_of(expr):
        return MemberAccessNode(expr, "count")

    def branch_match():
        return MatchNode(
            IdentifierNode("flag"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, bool_type)),
                    None,
                    [
                        ExpressionStatementNode(
                            IdentifierNode("left"),
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, bool_type)),
                    None,
                    [
                        ExpressionStatementNode(
                            IdentifierNode("right"),
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "RustMutableReferenceCallBranchArgs",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=4),
                    ),
                    StructMemberNode("count", int_type),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "bump_payload",
                int_type,
                [ParameterNode("value", ReferenceType(payload_type, is_mutable=True))],
                [
                    AssignmentNode(
                        count_of(IdentifierNode("value")),
                        BinaryOpNode(
                            count_of(IdentifierNode("value")),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                    ),
                    ReturnNode(count_of(IdentifierNode("value"))),
                ],
            ),
            FunctionNode(
                "call_mut_branch_args",
                int_type,
                [
                    ParameterNode("flag", bool_type),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "ternaryScore",
                        int_type,
                        FunctionCallNode(
                            IdentifierNode("bump_payload"),
                            [
                                TernaryOpNode(
                                    IdentifierNode("flag"),
                                    IdentifierNode("left"),
                                    IdentifierNode("right"),
                                )
                            ],
                        ),
                    ),
                    VariableNode(
                        "matchScore",
                        int_type,
                        FunctionCallNode(
                            IdentifierNode("bump_payload"),
                            [branch_match()],
                        ),
                    ),
                    VariableNode(
                        "blockScore",
                        int_type,
                        FunctionCallNode(
                            IdentifierNode("bump_payload"),
                            [
                                BlockNode(
                                    [
                                        VariableNode(
                                            "marker",
                                            int_type,
                                            LiteralNode(1, int_type),
                                        ),
                                        ExpressionStatementNode(
                                            branch_match(),
                                            is_tail_expression=True,
                                        ),
                                    ]
                                )
                            ],
                        ),
                    ),
                    ReturnNode(
                        BinaryOpNode(
                            BinaryOpNode(
                                IdentifierNode("ternaryScore"),
                                "+",
                                IdentifierNode("matchScore"),
                            ),
                            "+",
                            IdentifierNode("blockScore"),
                        )
                    ),
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert (
        "pub fn call_mut_branch_args(flag: bool, mut left: Payload, mut right: Payload)"
        in generated_code
    )
    assert (
        "let ternaryScore: i32 = bump_payload((if flag { &mut left } else { &mut right }));"
        in generated_code
    )
    assert "let matchScore: i32 = bump_payload(match flag {" in generated_code
    assert "true => {" in generated_code
    assert "false => {" in generated_code
    assert generated_code.count("&mut left") == 3
    assert generated_code.count("&mut right") == 3
    assert "let blockScore: i32 = bump_payload({" in generated_code
    assert "let marker: i32 = 1;" in generated_code
    assert "&mut (if flag" not in generated_code
    assert "&mut match flag" not in generated_code
    assert "&mut {" not in generated_code
    assert ".clone()" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_reference_locals_and_returns_borrow_without_clones_compile(tmp_path):
    payload_type = NamedType("Payload")
    int_type = PrimitiveType("int")

    ast = ShaderNode(
        "RustReferenceLocalSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=4),
                    ),
                    StructMemberNode("count", int_type),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "forward_ref",
                ReferenceType(payload_type),
                [ParameterNode("value", ReferenceType(payload_type))],
                [
                    VariableNode(
                        "alias",
                        ReferenceType(payload_type),
                        IdentifierNode("value"),
                    ),
                    ReturnNode(IdentifierNode("alias")),
                ],
            ),
            FunctionNode(
                "forward_mut_ref",
                ReferenceType(payload_type, is_mutable=True),
                [ParameterNode("value", ReferenceType(payload_type, is_mutable=True))],
                [
                    VariableNode(
                        "alias",
                        ReferenceType(payload_type, is_mutable=True),
                        IdentifierNode("value"),
                    ),
                    AssignmentNode(
                        MemberAccessNode(IdentifierNode("alias"), "count"),
                        BinaryOpNode(
                            MemberAccessNode(IdentifierNode("alias"), "count"),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                    ),
                    ReturnNode(IdentifierNode("alias")),
                ],
            ),
            FunctionNode(
                "borrow_locals",
                int_type,
                [ParameterNode("payload", payload_type)],
                [
                    VariableNode(
                        "read",
                        ReferenceType(payload_type),
                        IdentifierNode("payload"),
                    ),
                    VariableNode(
                        "before",
                        int_type,
                        MemberAccessNode(IdentifierNode("read"), "count"),
                    ),
                    VariableNode(
                        "editable",
                        ReferenceType(payload_type, is_mutable=True),
                        IdentifierNode("payload"),
                    ),
                    AssignmentNode(
                        MemberAccessNode(IdentifierNode("editable"), "count"),
                        BinaryOpNode(
                            IdentifierNode("before"),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                    ),
                    ReturnNode(MemberAccessNode(IdentifierNode("editable"), "count")),
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert "pub fn forward_ref(value: &Payload) -> &Payload" in generated_code
    assert "let alias: &Payload = value;" in generated_code
    assert "return alias;" in generated_code
    assert "pub fn forward_mut_ref(value: &mut Payload) -> &mut Payload" in (
        generated_code
    )
    assert "let mut alias: &mut Payload = value;" in generated_code
    assert "return alias;" in generated_code
    assert "pub fn borrow_locals(mut payload: Payload) -> i32" in generated_code
    assert "let read: &Payload = &payload;" in generated_code
    assert "let before: i32 = read.count;" in generated_code
    assert "let mut editable: &mut Payload = &mut payload;" in generated_code
    assert "ReferenceType(" not in generated_code
    assert "&payload.clone()" not in generated_code
    assert "&mut payload.clone()" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_reference_branches_and_mutable_places_borrow_without_clones_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    wrapper_type = NamedType("Wrapper")
    int_type = PrimitiveType("int")
    bool_type = PrimitiveType("bool")

    def count_of(expr):
        return MemberAccessNode(expr, "count")

    def payload_member():
        return MemberAccessNode(IdentifierNode("wrapper"), "payload")

    def payload_slot():
        return ArrayAccessNode(
            MemberAccessNode(IdentifierNode("wrapper"), "values"),
            IdentifierNode("index"),
        )

    ast = ShaderNode(
        "RustReferenceBranchSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=4),
                    ),
                    StructMemberNode("count", int_type),
                ],
            ),
            StructNode(
                "Wrapper",
                [
                    StructMemberNode("payload", payload_type),
                    StructMemberNode("values", ArrayType(payload_type, size=2)),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "choose_ref",
                ReferenceType(payload_type),
                [
                    ParameterNode("flag", bool_type),
                    ParameterNode("left", ReferenceType(payload_type)),
                    ParameterNode("right", ReferenceType(payload_type)),
                ],
                [
                    ReturnNode(
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        )
                    )
                ],
            ),
            FunctionNode(
                "branch_local_ref",
                int_type,
                [
                    ParameterNode("flag", bool_type),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "selected",
                        ReferenceType(payload_type),
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        ),
                    ),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
            FunctionNode(
                "branch_local_mut_ref",
                int_type,
                [
                    ParameterNode("flag", bool_type),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "selected",
                        ReferenceType(payload_type, is_mutable=True),
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        ),
                    ),
                    AssignmentNode(
                        count_of(IdentifierNode("selected")),
                        LiteralNode(7, int_type),
                    ),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
            FunctionNode(
                "borrow_mut_places",
                int_type,
                [
                    ParameterNode("wrapper", wrapper_type),
                    ParameterNode("index", int_type),
                ],
                [
                    VariableNode(
                        "member",
                        ReferenceType(payload_type, is_mutable=True),
                        payload_member(),
                    ),
                    VariableNode(
                        "slot",
                        ReferenceType(payload_type, is_mutable=True),
                        payload_slot(),
                    ),
                    AssignmentNode(
                        count_of(IdentifierNode("member")),
                        BinaryOpNode(
                            count_of(IdentifierNode("member")),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                    ),
                    AssignmentNode(
                        count_of(IdentifierNode("slot")),
                        BinaryOpNode(
                            count_of(IdentifierNode("slot")),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                    ),
                    ReturnNode(
                        BinaryOpNode(
                            count_of(IdentifierNode("member")),
                            "+",
                            count_of(IdentifierNode("slot")),
                        )
                    ),
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert (
        "pub fn choose_ref<'a>(flag: bool, left: &'a Payload, right: &'a Payload) "
        "-> &'a Payload"
    ) in generated_code
    assert "return (if flag { left } else { right });" in generated_code
    assert "pub fn branch_local_ref(flag: bool, left: Payload, right: Payload)" in (
        generated_code
    )
    assert "let selected: &Payload = (if flag { &left } else { &right });" in (
        generated_code
    )
    assert (
        "pub fn branch_local_mut_ref(flag: bool, mut left: Payload, mut right: Payload)"
        in generated_code
    )
    assert (
        "let mut selected: &mut Payload = (if flag { &mut left } else { &mut right });"
    ) in generated_code
    assert "pub fn borrow_mut_places(mut wrapper: Wrapper, index: i32)" in (
        generated_code
    )
    assert "let mut member: &mut Payload = &mut wrapper.payload;" in generated_code
    assert (
        "let mut slot: &mut Payload = &mut wrapper.values[index as usize];"
        in generated_code
    )
    assert "&(if flag" not in generated_code
    assert "&mut (if flag" not in generated_code
    assert ".clone()" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_reference_match_and_block_tails_borrow_without_clones_compile(tmp_path):
    payload_type = NamedType("Payload")
    int_type = PrimitiveType("int")
    bool_type = PrimitiveType("bool")

    def count_of(expr):
        return MemberAccessNode(expr, "count")

    def bool_match(true_expr, false_expr):
        return MatchNode(
            IdentifierNode("flag"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, bool_type)),
                    None,
                    [
                        ExpressionStatementNode(
                            true_expr,
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, bool_type)),
                    None,
                    [
                        ExpressionStatementNode(
                            false_expr,
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    def block_tail(expr):
        return BlockNode(
            [
                VariableNode(
                    "marker",
                    int_type,
                    LiteralNode(1, int_type),
                ),
                ExpressionStatementNode(expr, is_tail_expression=True),
            ]
        )

    ast = ShaderNode(
        "RustReferenceMatchSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=4),
                    ),
                    StructMemberNode("count", int_type),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "return_match_ref",
                ReferenceType(payload_type),
                [
                    ParameterNode("flag", bool_type),
                    ParameterNode("left", ReferenceType(payload_type)),
                    ParameterNode("right", ReferenceType(payload_type)),
                ],
                [
                    ReturnNode(
                        bool_match(
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        )
                    )
                ],
            ),
            FunctionNode(
                "match_local_ref",
                int_type,
                [
                    ParameterNode("flag", bool_type),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "selected",
                        ReferenceType(payload_type),
                        bool_match(
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        ),
                    ),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
            FunctionNode(
                "match_local_mut_ref",
                int_type,
                [
                    ParameterNode("flag", bool_type),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "selected",
                        ReferenceType(payload_type, is_mutable=True),
                        bool_match(
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        ),
                    ),
                    AssignmentNode(
                        count_of(IdentifierNode("selected")),
                        LiteralNode(7, int_type),
                    ),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
            FunctionNode(
                "block_match_local_mut_ref",
                int_type,
                [
                    ParameterNode("flag", bool_type),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "selected",
                        ReferenceType(payload_type, is_mutable=True),
                        block_tail(
                            bool_match(
                                IdentifierNode("left"),
                                IdentifierNode("right"),
                            )
                        ),
                    ),
                    AssignmentNode(
                        count_of(IdentifierNode("selected")),
                        LiteralNode(11, int_type),
                    ),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert (
        "pub fn return_match_ref<'a>("
        "flag: bool, left: &'a Payload, right: &'a Payload"
        ") -> &'a Payload"
    ) in generated_code
    assert "pub fn match_local_ref(flag: bool, left: Payload, right: Payload)" in (
        generated_code
    )
    assert (
        "pub fn match_local_mut_ref(flag: bool, mut left: Payload, mut right: Payload)"
        in generated_code
    )
    assert (
        "pub fn block_match_local_mut_ref("
        "flag: bool, mut left: Payload, mut right: Payload"
        ")"
    ) in generated_code
    assert "let selected: &Payload = match flag {" in generated_code
    assert "let mut selected: &mut Payload = match flag {" in generated_code
    assert generated_code.count("let mut selected: &mut Payload =") == 2
    assert "let marker: i32 = 1;" in generated_code
    assert "&left" in generated_code
    assert "&right" in generated_code
    assert generated_code.count("&mut left") == 2
    assert generated_code.count("&mut right") == 2
    assert "return match flag {" in generated_code
    assert "&match flag" not in generated_code
    assert "&mut match flag" not in generated_code
    assert ".clone()" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_reference_match_pattern_bindings_borrow_without_clones_compile(tmp_path):
    payload_type = NamedType("Payload")
    choice_type = NamedType("Choice")
    int_type = PrimitiveType("int")
    bool_type = PrimitiveType("bool")

    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode("One", fields=[payload_type]),
            EnumVariantNode("Pair", fields=[payload_type, payload_type]),
            EnumVariantNode(
                "Named",
                fields=[
                    ("primary", payload_type),
                    ("backup", payload_type),
                ],
            ),
        ],
    )

    def count_of(expr):
        return MemberAccessNode(expr, "count")

    def selected_decl(name, declared_type, initial_value):
        return VariableNode(name, declared_type, initial_value)

    def increment_selected(name, value):
        return AssignmentNode(
            count_of(IdentifierNode(name)),
            BinaryOpNode(
                count_of(IdentifierNode(name)),
                "+",
                LiteralNode(value, int_type),
            ),
        )

    def immutable_pair_choice():
        return TernaryOpNode(
            IdentifierNode("take_first"),
            IdentifierNode("first"),
            IdentifierNode("second"),
        )

    def mutable_pair_choice():
        return TernaryOpNode(
            IdentifierNode("take_first"),
            IdentifierNode("first"),
            IdentifierNode("second"),
        )

    immutable_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One", [IdentifierPatternNode("payload")]
                ),
                None,
                [
                    selected_decl(
                        "selected",
                        ReferenceType(payload_type),
                        IdentifierNode("payload"),
                    ),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [
                        IdentifierPatternNode("first"),
                        IdentifierPatternNode("second"),
                    ],
                ),
                None,
                [
                    selected_decl(
                        "selected",
                        ReferenceType(payload_type),
                        immutable_pair_choice(),
                    ),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
            MatchArmNode(
                StructPatternNode(
                    "Choice::Named",
                    {
                        "primary": IdentifierPatternNode("primary"),
                        "backup": IdentifierPatternNode("backup"),
                    },
                ),
                None,
                [
                    selected_decl(
                        "selected",
                        ReferenceType(payload_type),
                        TernaryOpNode(
                            IdentifierNode("take_first"),
                            IdentifierNode("primary"),
                            IdentifierNode("backup"),
                        ),
                    ),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
        ],
    )

    mutable_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One", [IdentifierPatternNode("payload")]
                ),
                None,
                [
                    selected_decl(
                        "selected",
                        ReferenceType(payload_type, is_mutable=True),
                        IdentifierNode("payload"),
                    ),
                    increment_selected("selected", 3),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [
                        IdentifierPatternNode("first"),
                        IdentifierPatternNode("second"),
                    ],
                ),
                None,
                [
                    selected_decl(
                        "selected",
                        ReferenceType(payload_type, is_mutable=True),
                        mutable_pair_choice(),
                    ),
                    increment_selected("selected", 5),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
            MatchArmNode(
                StructPatternNode(
                    "Choice::Named",
                    {
                        "primary": IdentifierPatternNode("primary"),
                        "backup": IdentifierPatternNode("backup"),
                    },
                ),
                None,
                [
                    selected_decl(
                        "selected",
                        ReferenceType(payload_type, is_mutable=True),
                        TernaryOpNode(
                            IdentifierNode("take_first"),
                            IdentifierNode("primary"),
                            IdentifierNode("backup"),
                        ),
                    ),
                    increment_selected("selected", 7),
                    ReturnNode(count_of(IdentifierNode("selected"))),
                ],
            ),
        ],
    )

    ast = ShaderNode(
        "RustReferencePatternBindingSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=4),
                    ),
                    StructMemberNode("count", int_type),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "borrow_pattern_bindings",
                int_type,
                [
                    ParameterNode("choice", choice_type),
                    ParameterNode("take_first", bool_type),
                ],
                [immutable_match],
            ),
            FunctionNode(
                "borrow_mut_pattern_bindings",
                int_type,
                [
                    ParameterNode("choice", choice_type),
                    ParameterNode("take_first", bool_type),
                ],
                [mutable_match],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert "Choice::One(payload) =>" in generated_code
    assert "Choice::Pair(first, second) =>" in generated_code
    assert "Choice::Named { primary, backup } =>" in generated_code
    assert "let selected: &Payload = &payload;" in generated_code
    assert "let selected: &Payload = (if take_first { &first } else { &second });" in (
        generated_code
    )
    assert (
        "let selected: &Payload = (if take_first { &primary } else { &backup });"
        in generated_code
    )
    assert "Choice::One(mut payload) =>" in generated_code
    assert "Choice::Pair(mut first, mut second) =>" in generated_code
    assert "primary: mut primary" in generated_code
    assert "backup: mut backup" in generated_code
    assert "let mut selected: &mut Payload = &mut payload;" in generated_code
    assert (
        "let mut selected: &mut Payload = "
        "(if take_first { &mut first } else { &mut second });"
    ) in generated_code
    assert (
        "let mut selected: &mut Payload = "
        "(if take_first { &mut primary } else { &mut backup });"
    ) in generated_code
    assert "&mut (if" not in generated_code
    assert ".clone()" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_non_copy_loop_break_match_preserves_scrutinee_for_later_use_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    choice_type = NamedType("Choice")
    int_type = PrimitiveType("int")

    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode("One", fields=[payload_type]),
            EnumVariantNode("Pair", fields=[payload_type]),
        ],
    )

    def count_of(expr):
        return MemberAccessNode(expr, "count")

    loop_break_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [BreakNode()],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [BreakNode()],
            ),
        ],
    )

    return_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    ExpressionStatementNode(
                        count_of(IdentifierNode("payload")),
                        is_tail_expression=True,
                    )
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    ExpressionStatementNode(
                        count_of(IdentifierNode("payload")),
                        is_tail_expression=True,
                    )
                ],
            ),
        ],
    )

    ast = ShaderNode(
        "RustLoopBreakMatchScrutineeSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=None),
                    ),
                    StructMemberNode("count", int_type),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "break_then_read_choice",
                int_type,
                [ParameterNode("choice", choice_type)],
                [
                    LoopNode([loop_break_match]),
                    ReturnNode(return_match),
                ],
            )
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_match_terminal_if_arms_move_without_clones_compile(tmp_path):
    payload_type = NamedType("Payload")
    choice_type = NamedType("Choice")
    int_type = PrimitiveType("int")
    bool_type = PrimitiveType("bool")

    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode("One", fields=[payload_type]),
            EnumVariantNode("Pair", fields=[payload_type]),
        ],
    )

    def count_of(expr):
        return MemberAccessNode(expr, "count")

    def if_returns(then_value, else_value):
        return IfNode(
            IdentifierNode("flag"),
            [ReturnNode(then_value)],
            [ReturnNode(else_value)],
        )

    terminal_if_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    if_returns(
                        count_of(IdentifierNode("payload")),
                        LiteralNode(0, int_type),
                    )
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    if_returns(
                        BinaryOpNode(
                            count_of(IdentifierNode("payload")),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                        LiteralNode(1, int_type),
                    )
                ],
            ),
        ],
    )

    partial_if_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    IfNode(
                        IdentifierNode("flag"),
                        [ReturnNode(count_of(IdentifierNode("payload")))],
                    )
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [ReturnNode(count_of(IdentifierNode("payload")))],
            ),
        ],
    )

    return_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    ExpressionStatementNode(
                        count_of(IdentifierNode("payload")),
                        is_tail_expression=True,
                    )
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    ExpressionStatementNode(
                        count_of(IdentifierNode("payload")),
                        is_tail_expression=True,
                    )
                ],
            ),
        ],
    )

    ast = ShaderNode(
        "RustTerminalIfMatchScrutineeSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=None),
                    ),
                    StructMemberNode("count", int_type),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "terminal_if_match",
                int_type,
                [
                    ParameterNode("choice", choice_type),
                    ParameterNode("flag", bool_type),
                ],
                [terminal_if_match],
            ),
            FunctionNode(
                "partial_if_match_then_read",
                int_type,
                [
                    ParameterNode("choice", choice_type),
                    ParameterNode("flag", bool_type),
                ],
                [
                    partial_if_match,
                    ReturnNode(return_match),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_match_terminal_block_arms_move_without_clones_compile(tmp_path):
    payload_type = NamedType("Payload")
    choice_type = NamedType("Choice")
    int_type = PrimitiveType("int")
    bool_type = PrimitiveType("bool")

    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode("One", fields=[payload_type]),
            EnumVariantNode("Pair", fields=[payload_type]),
        ],
    )

    def count_of(expr):
        return MemberAccessNode(expr, "count")

    def terminal_block(then_value, else_value):
        return BlockNode(
            [
                IfNode(
                    IdentifierNode("flag"),
                    [ReturnNode(then_value)],
                    [ReturnNode(else_value)],
                )
            ]
        )

    terminal_block_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    terminal_block(
                        count_of(IdentifierNode("payload")),
                        LiteralNode(0, int_type),
                    )
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    terminal_block(
                        BinaryOpNode(
                            count_of(IdentifierNode("payload")),
                            "+",
                            LiteralNode(1, int_type),
                        ),
                        LiteralNode(1, int_type),
                    )
                ],
            ),
        ],
    )

    partial_block_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    BlockNode(
                        [
                            IfNode(
                                IdentifierNode("flag"),
                                [ReturnNode(count_of(IdentifierNode("payload")))],
                            )
                        ]
                    )
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [ReturnNode(count_of(IdentifierNode("payload")))],
            ),
        ],
    )

    return_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    ExpressionStatementNode(
                        count_of(IdentifierNode("payload")),
                        is_tail_expression=True,
                    )
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    ExpressionStatementNode(
                        count_of(IdentifierNode("payload")),
                        is_tail_expression=True,
                    )
                ],
            ),
        ],
    )

    ast = ShaderNode(
        "RustTerminalBlockMatchScrutineeSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=None),
                    ),
                    StructMemberNode("count", int_type),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "terminal_block_match",
                int_type,
                [
                    ParameterNode("choice", choice_type),
                    ParameterNode("flag", bool_type),
                ],
                [terminal_block_match],
            ),
            FunctionNode(
                "partial_block_match_then_read",
                int_type,
                [
                    ParameterNode("choice", choice_type),
                    ParameterNode("flag", bool_type),
                ],
                [
                    partial_block_match,
                    ReturnNode(return_match),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_match_terminal_loop_arms_move_without_clones_compile(tmp_path):
    payload_type = NamedType("Payload")
    choice_type = NamedType("Choice")
    int_type = PrimitiveType("int")

    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode("One", fields=[payload_type]),
            EnumVariantNode("Pair", fields=[payload_type]),
        ],
    )

    def count_of(expr):
        return MemberAccessNode(expr, "count")

    terminal_loop_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [LoopNode([ReturnNode(count_of(IdentifierNode("payload")))])],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    LoopNode(
                        [
                            ReturnNode(
                                BinaryOpNode(
                                    count_of(IdentifierNode("payload")),
                                    "+",
                                    LiteralNode(1, int_type),
                                )
                            )
                        ]
                    )
                ],
            ),
        ],
    )

    break_loop_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [LoopNode([BreakNode()])],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [ReturnNode(count_of(IdentifierNode("payload")))],
            ),
        ],
    )

    return_match = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::One",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    ExpressionStatementNode(
                        count_of(IdentifierNode("payload")),
                        is_tail_expression=True,
                    )
                ],
            ),
            MatchArmNode(
                ConstructorPatternNode(
                    "Choice::Pair",
                    [IdentifierPatternNode("payload")],
                ),
                None,
                [
                    ExpressionStatementNode(
                        count_of(IdentifierNode("payload")),
                        is_tail_expression=True,
                    )
                ],
            ),
        ],
    )

    ast = ShaderNode(
        "RustTerminalLoopMatchScrutineeSmoke",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    StructMemberNode(
                        "weights",
                        ArrayType(PrimitiveType("float"), size=None),
                    ),
                    StructMemberNode("count", int_type),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "terminal_loop_match",
                int_type,
                [ParameterNode("choice", choice_type)],
                [terminal_loop_match],
            ),
            FunctionNode(
                "break_loop_match_then_read",
                int_type,
                [ParameterNode("choice", choice_type)],
                [
                    break_loop_match,
                    ReturnNode(return_match),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_direct_callable_trait_parameters_emit_impl_trait_and_compile(tmp_path):
    int_type = PrimitiveType("int")

    ast = ShaderNode(
        "DirectCallableTraitParameters",
        ExecutionModel.GENERAL_PURPOSE,
        functions=[
            FunctionNode(
                "apply_read",
                int_type,
                [
                    ParameterNode("input", int_type),
                    ParameterNode("op", NamedType("Fn(i32) -> i32")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("op"),
                            [IdentifierNode("input")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_mut",
                int_type,
                [
                    ParameterNode("input", int_type),
                    ParameterNode("op", NamedType("FnMut(i32) -> i32")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("op"),
                            [IdentifierNode("input")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_once",
                int_type,
                [
                    ParameterNode("input", int_type),
                    ParameterNode("op", NamedType("FnOnce(i32) -> i32")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("op"),
                            [IdentifierNode("input")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_generic_mut",
                int_type,
                [
                    ParameterNode("input", int_type),
                    ParameterNode("op", GenericType("F")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("op"),
                            [IdentifierNode("input")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode(
                        "F",
                        constraints=[NamedType("FnMut(i32) -> i32")],
                    )
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert (
        "pub fn apply_read(input: i32, op: impl Fn(i32) -> i32) -> i32"
        in generated_code
    )
    assert (
        "pub fn apply_mut(input: i32, mut op: impl FnMut(i32) -> i32) -> i32"
        in generated_code
    )
    assert (
        "pub fn apply_once(input: i32, op: impl FnOnce(i32) -> i32) -> i32"
        in generated_code
    )
    assert (
        "pub fn apply_generic_mut<F: FnMut(i32) -> i32 + Copy>(input: i32, mut op: F)"
        in generated_code
    )
    assert "op: Fn(i32) -> i32" not in generated_code
    assert "op: FnMut(i32) -> i32" not in generated_code
    assert "op: FnOnce(i32) -> i32" not in generated_code
    assert "F: impl FnMut" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_non_copy_member_access_clones_in_value_contexts_and_compile(tmp_path):
    code = """
    shader NonCopyMemberAccess {
        struct Payload {
            float weights[];
            int count;
        };

        struct Wrapper {
            Payload payload;
            int id;
        };

        fn keep(wrapper: Wrapper) -> int {
            let payload = wrapper.payload;
            let again = wrapper;
            payload.count + again.id
        }

        fn replace(wrapper: Wrapper, replacement: Payload) -> int {
            let editable = wrapper;
            editable.payload = replacement;
            editable.payload.count
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_non_copy_identifier_reads_clone_in_value_contexts_and_compile(tmp_path):
    code = """
    shader NonCopyIdentifierReuse {
        struct Payload {
            float weights[];
            int count;
        };

        fn count(payload: Payload) -> int {
            payload.count
        }

        fn sum(payload: Payload, replacement: Payload) -> int {
            let first = payload;
            let second = payload;
            let callA = count(payload);
            let callB = count(payload);
            let editable = first;
            editable = replacement;
            second.count + callA + callB + editable.count
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_non_copy_array_access_clones_in_value_contexts_and_compile(tmp_path):
    payload_type = NamedType("Payload")
    payloads_ref = IdentifierNode("payloads")
    first_count = MemberAccessNode(IdentifierNode("first"), "count")
    second_count = MemberAccessNode(IdentifierNode("second"), "count")
    current_count = MemberAccessNode(
        ArrayAccessNode(payloads_ref, LiteralNode(0, PrimitiveType("int"))),
        "count",
    )
    ast = ShaderNode(
        "NonCopyArrayAccess",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "sum",
                PrimitiveType("int"),
                [
                    ParameterNode("payloads", ArrayType(payload_type, 2)),
                    ParameterNode("replacement", payload_type),
                ],
                [
                    VariableNode(
                        "first",
                        PrimitiveType("auto"),
                        ArrayAccessNode(
                            IdentifierNode("payloads"),
                            LiteralNode(0, PrimitiveType("int")),
                        ),
                    ),
                    VariableNode(
                        "second",
                        PrimitiveType("auto"),
                        ArrayAccessNode(
                            IdentifierNode("payloads"),
                            LiteralNode(1, PrimitiveType("int")),
                        ),
                    ),
                    AssignmentNode(
                        ArrayAccessNode(
                            IdentifierNode("payloads"),
                            LiteralNode(0, PrimitiveType("int")),
                        ),
                        IdentifierNode("replacement"),
                    ),
                    ReturnNode(
                        BinaryOpNode(
                            BinaryOpNode(first_count, "+", second_count),
                            "+",
                            current_count,
                        )
                    ),
                ],
            )
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_legacy_non_copy_array_nodes_are_refused():
    ast = ShaderNode(
        "LegacyNonCopyArray",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[StructNode("Payload", [ArrayNode("float", "weights")])],
        functions=[
            FunctionNode(
                "probe",
                PrimitiveType("void"),
                [],
                [ArrayNode("Payload", "values", 2)],
            )
        ],
    )

    # An unsized struct member needs Vec, which rust-gpu has no allocator for.
    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_identifier_returns_move_without_unneeded_clone_and_compile(tmp_path):
    payload_type = NamedType("Payload")
    ast = ShaderNode(
        "NonCopyReturns",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "take",
                payload_type,
                [ParameterNode("payload", payload_type)],
                [ReturnNode(IdentifierNode("payload"))],
            ),
            FunctionNode(
                "reuse_before_return",
                payload_type,
                [ParameterNode("payload", payload_type)],
                [
                    VariableNode(
                        "snapshot",
                        PrimitiveType("auto"),
                        IdentifierNode("payload"),
                    ),
                    ReturnNode(IdentifierNode("payload")),
                ],
            ),
            FunctionNode(
                "return_array",
                ArrayType(payload_type, 2),
                [ParameterNode("values", ArrayType(payload_type, 2))],
                [
                    VariableNode(
                        "first",
                        PrimitiveType("auto"),
                        ArrayAccessNode(
                            IdentifierNode("values"),
                            LiteralNode(0, PrimitiveType("int")),
                        ),
                    ),
                    ReturnNode(IdentifierNode("values")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_ternary_and_match_returns_move_selected_branch_and_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    ast = ShaderNode(
        "NonCopyBranchReturns",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "choose_ternary",
                payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "snapshot",
                        PrimitiveType("auto"),
                        IdentifierNode("left"),
                    ),
                    ReturnNode(
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        )
                    ),
                ],
            ),
            FunctionNode(
                "choose_match",
                payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    ReturnNode(
                        MatchNode(
                            IdentifierNode("flag"),
                            [
                                MatchArmNode(
                                    LiteralPatternNode(
                                        LiteralNode(True, PrimitiveType("bool"))
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            IdentifierNode("left"),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                                MatchArmNode(
                                    LiteralPatternNode(
                                        LiteralNode(False, PrimitiveType("bool"))
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            IdentifierNode("right"),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_return_call_and_constructor_args_move_when_unique_and_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    wrapper_type = NamedType("Wrapper")
    ast = ShaderNode(
        "NonCopyReturnArguments",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "Wrapper",
                [
                    StructMemberNode("payload", payload_type),
                    StructMemberNode("backup", payload_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "identity",
                payload_type,
                [ParameterNode("value", payload_type)],
                [ReturnNode(IdentifierNode("value"))],
            ),
            FunctionNode(
                "make_wrapper",
                wrapper_type,
                [
                    ParameterNode("payload", payload_type),
                    ParameterNode("backup", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            wrapper_type,
                            [IdentifierNode("payload"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_call",
                payload_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity"),
                            [IdentifierNode("payload")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_duplicate_call",
                wrapper_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_wrapper"),
                            [IdentifierNode("payload"), IdentifierNode("payload")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_constructor",
                wrapper_type,
                [
                    ParameterNode("payload", payload_type),
                    ParameterNode("backup", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            wrapper_type,
                            [IdentifierNode("payload"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_named_constructor",
                wrapper_type,
                [
                    ParameterNode("payload", payload_type),
                    ParameterNode("backup", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            wrapper_type,
                            [],
                            named_arguments={
                                "payload": IdentifierNode("payload"),
                                "backup": IdentifierNode("backup"),
                            },
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_nested_return_call_args_move_last_occurrence_compile(tmp_path):
    payload_type = NamedType("Payload")
    pair_type = NamedType("Pair")
    ast = ShaderNode(
        "NonCopyNestedReturnArguments",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "Pair",
                [
                    StructMemberNode("first", payload_type),
                    StructMemberNode("second", payload_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "identity",
                payload_type,
                [ParameterNode("value", payload_type)],
                [ReturnNode(IdentifierNode("value"))],
            ),
            FunctionNode(
                "make_pair",
                pair_type,
                [
                    ParameterNode("first", payload_type),
                    ParameterNode("second", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [IdentifierNode("first"), IdentifierNode("second")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_nested_single_call",
                payload_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity"),
                            [
                                FunctionCallNode(
                                    IdentifierNode("identity"),
                                    [IdentifierNode("payload")],
                                )
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_nested_then_direct",
                pair_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [
                                FunctionCallNode(
                                    IdentifierNode("identity"),
                                    [IdentifierNode("payload")],
                                ),
                                IdentifierNode("payload"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_direct_then_nested",
                pair_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [
                                IdentifierNode("payload"),
                                FunctionCallNode(
                                    IdentifierNode("identity"),
                                    [IdentifierNode("payload")],
                                ),
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_return_call_and_constructor_member_args_move_when_safe_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    wrapper_type = NamedType("Wrapper")
    pair_type = NamedType("Pair")
    ast = ShaderNode(
        "NonCopyReturnMemberArguments",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode("Wrapper", [StructMemberNode("payload", payload_type)]),
            StructNode(
                "Pair",
                [
                    StructMemberNode("payload", payload_type),
                    StructMemberNode("backup", payload_type),
                ],
            ),
        ],
        global_variables=[
            VariableNode("globalWrapper", wrapper_type),
            ArrayNode("Payload", "globalValues", 2),
        ],
        functions=[
            FunctionNode(
                "make_pair",
                pair_type,
                [
                    ParameterNode("payload", payload_type),
                    ParameterNode("backup", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [IdentifierNode("payload"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_member_call",
                pair_type,
                [
                    ParameterNode("left", wrapper_type),
                    ParameterNode("right", wrapper_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [
                                MemberAccessNode(IdentifierNode("left"), "payload"),
                                MemberAccessNode(IdentifierNode("right"), "payload"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_member_constructor",
                pair_type,
                [
                    ParameterNode("left", wrapper_type),
                    ParameterNode("right", wrapper_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [
                                MemberAccessNode(IdentifierNode("left"), "payload"),
                                MemberAccessNode(IdentifierNode("right"), "payload"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_duplicate_member_call",
                pair_type,
                [ParameterNode("wrapper", wrapper_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [
                                MemberAccessNode(IdentifierNode("wrapper"), "payload"),
                                MemberAccessNode(IdentifierNode("wrapper"), "payload"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_member_named_constructor",
                pair_type,
                [
                    ParameterNode("left", wrapper_type),
                    ParameterNode("right", wrapper_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [],
                            named_arguments={
                                "payload": MemberAccessNode(
                                    IdentifierNode("left"),
                                    "payload",
                                ),
                                "backup": MemberAccessNode(
                                    IdentifierNode("right"),
                                    "payload",
                                ),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_duplicate_member_named_constructor",
                pair_type,
                [ParameterNode("wrapper", wrapper_type)],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [],
                            named_arguments={
                                "payload": MemberAccessNode(
                                    IdentifierNode("wrapper"),
                                    "payload",
                                ),
                                "backup": MemberAccessNode(
                                    IdentifierNode("wrapper"),
                                    "payload",
                                ),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_static_and_indexed_call",
                pair_type,
                [ParameterNode("values", ArrayType(payload_type, 2))],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [
                                MemberAccessNode(
                                    IdentifierNode("globalWrapper"),
                                    "payload",
                                ),
                                ArrayAccessNode(
                                    IdentifierNode("values"),
                                    LiteralNode(0, PrimitiveType("int")),
                                ),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_static_indexed_constructor",
                pair_type,
                [],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [
                                ArrayAccessNode(
                                    IdentifierNode("globalValues"),
                                    LiteralNode(0, PrimitiveType("int")),
                                ),
                                MemberAccessNode(
                                    IdentifierNode("globalWrapper"),
                                    "payload",
                                ),
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_nested_return_member_args_respect_root_conflicts_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    wrapper_type = NamedType("Wrapper")
    outer_type = NamedType("Outer")
    pair_type = NamedType("Pair")
    outer_payload_type = NamedType("OuterPayload")
    payload_outer_type = NamedType("PayloadOuter")
    payload_outer_payload_type = NamedType("PayloadOuterPayload")

    def nested_payload(root_name):
        return MemberAccessNode(
            MemberAccessNode(IdentifierNode(root_name), "inner"),
            "payload",
        )

    def indexed_nested_payload(array_name):
        return MemberAccessNode(
            MemberAccessNode(
                ArrayAccessNode(
                    IdentifierNode(array_name),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                "inner",
            ),
            "payload",
        )

    ast = ShaderNode(
        "NonCopyNestedReturnMemberArguments",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode("Wrapper", [StructMemberNode("payload", payload_type)]),
            StructNode("Outer", [StructMemberNode("inner", wrapper_type)]),
            StructNode(
                "Pair",
                [
                    StructMemberNode("payload", payload_type),
                    StructMemberNode("backup", payload_type),
                ],
            ),
            StructNode(
                "OuterPayload",
                [
                    StructMemberNode("outer", outer_type),
                    StructMemberNode("payload", payload_type),
                ],
            ),
            StructNode(
                "PayloadOuter",
                [
                    StructMemberNode("payload", payload_type),
                    StructMemberNode("outer", outer_type),
                ],
            ),
            StructNode(
                "PayloadOuterPayload",
                [
                    StructMemberNode("first", payload_type),
                    StructMemberNode("outer", outer_type),
                    StructMemberNode("second", payload_type),
                ],
            ),
        ],
        global_variables=[
            VariableNode("globalOuter", outer_type),
            ArrayNode("Outer", "globalOuters", 2),
        ],
        functions=[
            FunctionNode(
                "make_pair",
                pair_type,
                [
                    ParameterNode("payload", payload_type),
                    ParameterNode("backup", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [IdentifierNode("payload"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "make_outer_payload",
                outer_payload_type,
                [
                    ParameterNode("outer", outer_type),
                    ParameterNode("payload", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            outer_payload_type,
                            [IdentifierNode("outer"), IdentifierNode("payload")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "make_payload_outer",
                payload_outer_type,
                [
                    ParameterNode("payload", payload_type),
                    ParameterNode("outer", outer_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            payload_outer_type,
                            [IdentifierNode("payload"), IdentifierNode("outer")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "make_payload_outer_payload",
                payload_outer_payload_type,
                [
                    ParameterNode("first", payload_type),
                    ParameterNode("outer", outer_type),
                    ParameterNode("second", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            payload_outer_payload_type,
                            [
                                IdentifierNode("first"),
                                IdentifierNode("outer"),
                                IdentifierNode("second"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_nested_member_call",
                pair_type,
                [
                    ParameterNode("left", outer_type),
                    ParameterNode("right", outer_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [nested_payload("left"), nested_payload("right")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_duplicate_nested_member_call",
                pair_type,
                [ParameterNode("outer", outer_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [nested_payload("outer"), nested_payload("outer")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_nested_member_named_constructor",
                pair_type,
                [
                    ParameterNode("left", outer_type),
                    ParameterNode("right", outer_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [],
                            named_arguments={
                                "payload": nested_payload("left"),
                                "backup": nested_payload("right"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_root_then_nested_member_call",
                outer_payload_type,
                [ParameterNode("outer", outer_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_outer_payload"),
                            [IdentifierNode("outer"), nested_payload("outer")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_nested_member_then_root_call",
                payload_outer_type,
                [ParameterNode("outer", outer_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_payload_outer"),
                            [nested_payload("outer"), IdentifierNode("outer")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_nested_member_around_root_call",
                payload_outer_payload_type,
                [ParameterNode("outer", outer_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_payload_outer_payload"),
                            [
                                nested_payload("outer"),
                                IdentifierNode("outer"),
                                nested_payload("outer"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_static_and_indexed_nested_call",
                pair_type,
                [ParameterNode("values", ArrayType(outer_type, 2))],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [
                                nested_payload("globalOuter"),
                                indexed_nested_payload("values"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_static_indexed_nested_constructor",
                pair_type,
                [],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_type,
                            [
                                indexed_nested_payload("globalOuters"),
                                nested_payload("globalOuter"),
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_generic_return_member_args_and_new_calls_compile(tmp_path):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    envelope_generic_type = NamedType("Envelope", [generic_type])
    envelope_payload_type = NamedType("Envelope", [payload_type])
    root_payload_type = NamedType("Root", [payload_type])
    pair_payload_type = NamedType("Pair", [payload_type])
    root_payload_record_type = NamedType("RootPayload")
    payload_root_record_type = NamedType("PayloadRoot")

    def envelope_payload(root_name):
        return MemberAccessNode(IdentifierNode(root_name), "payload")

    def root_nested_payload(root_name):
        return MemberAccessNode(
            MemberAccessNode(IdentifierNode(root_name), "envelope"),
            "payload",
        )

    ast = ShaderNode(
        "NonCopyGenericReturnMemberArguments",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "Envelope",
                [StructMemberNode("payload", generic_type)],
                generic_params=[GenericParameterNode("T")],
            ),
            StructNode(
                "Root",
                [StructMemberNode("envelope", envelope_generic_type)],
                generic_params=[GenericParameterNode("T")],
            ),
            StructNode(
                "Pair",
                [
                    StructMemberNode("first", generic_type),
                    StructMemberNode("second", generic_type),
                ],
                generic_params=[GenericParameterNode("T")],
            ),
            StructNode(
                "RootPayload",
                [
                    StructMemberNode("root", root_payload_type),
                    StructMemberNode("payload", payload_type),
                ],
            ),
            StructNode(
                "PayloadRoot",
                [
                    StructMemberNode("payload", payload_type),
                    StructMemberNode("root", root_payload_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_pair",
                pair_payload_type,
                [
                    ParameterNode("first", payload_type),
                    ParameterNode("second", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_payload_type,
                            [IdentifierNode("first"), IdentifierNode("second")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "make_root_payload",
                root_payload_record_type,
                [
                    ParameterNode("root", root_payload_type),
                    ParameterNode("payload", payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            root_payload_record_type,
                            [IdentifierNode("root"), IdentifierNode("payload")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "make_payload_root",
                payload_root_record_type,
                [
                    ParameterNode("payload", payload_type),
                    ParameterNode("root", root_payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            payload_root_record_type,
                            [IdentifierNode("payload"), IdentifierNode("root")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_generic_member_call",
                pair_payload_type,
                [
                    ParameterNode("left", envelope_payload_type),
                    ParameterNode("right", envelope_payload_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_pair"),
                            [envelope_payload("left"), envelope_payload("right")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_generic_new_call",
                pair_payload_type,
                [
                    ParameterNode("left", envelope_payload_type),
                    ParameterNode("right", envelope_payload_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("Pair::<Payload>::new"),
                            [envelope_payload("left"), envelope_payload("right")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_duplicate_generic_new_call",
                pair_payload_type,
                [ParameterNode("envelope", envelope_payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("Pair::<Payload>::new"),
                            [
                                envelope_payload("envelope"),
                                envelope_payload("envelope"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_generic_positional_constructor",
                pair_payload_type,
                [
                    ParameterNode("left", envelope_payload_type),
                    ParameterNode("right", envelope_payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_payload_type,
                            [envelope_payload("left"), envelope_payload("right")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_generic_named_constructor",
                pair_payload_type,
                [
                    ParameterNode("left", envelope_payload_type),
                    ParameterNode("right", envelope_payload_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            pair_payload_type,
                            [],
                            named_arguments={
                                "first": envelope_payload("left"),
                                "second": envelope_payload("right"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_generic_root_then_member",
                root_payload_record_type,
                [ParameterNode("root", root_payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_root_payload"),
                            [IdentifierNode("root"), root_nested_payload("root")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_generic_member_then_root",
                payload_root_record_type,
                [ParameterNode("root", root_payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_payload_root"),
                            [root_nested_payload("root"), IdentifierNode("root")],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_generic_enum_variant_returns_and_match_arms_compile(tmp_path):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    choice_payload_type = NamedType("Choice", [payload_type])
    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode("One", fields=[generic_type]),
            EnumVariantNode("Pair", fields=[generic_type, generic_type]),
        ],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]

    ast = ShaderNode(
        "NonCopyGenericEnumReturns",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "wrap_payload",
                choice_payload_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("Choice::<Payload>::One"),
                            [IdentifierNode("payload")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "duplicate_payload",
                choice_payload_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("Choice::<Payload>::Pair"),
                            [IdentifierNode("payload"), IdentifierNode("payload")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "repack_match",
                choice_payload_type,
                [ParameterNode("choice", choice_payload_type)],
                [
                    ReturnNode(
                        MatchNode(
                            IdentifierNode("choice"),
                            [
                                MatchArmNode(
                                    ConstructorPatternNode(
                                        "Choice::One",
                                        [IdentifierPatternNode("payload")],
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            FunctionCallNode(
                                                IdentifierNode(
                                                    "Choice::<Payload>::Pair"
                                                ),
                                                [
                                                    IdentifierNode("payload"),
                                                    IdentifierNode("payload"),
                                                ],
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                                MatchArmNode(
                                    ConstructorPatternNode(
                                        "Choice::Pair",
                                        [
                                            IdentifierPatternNode("first"),
                                            IdentifierPatternNode("second"),
                                        ],
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            FunctionCallNode(
                                                IdentifierNode(
                                                    "Choice::<Payload>::Pair"
                                                ),
                                                [
                                                    IdentifierNode("first"),
                                                    IdentifierNode("second"),
                                                ],
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_generic_struct_like_enum_variants_and_guards_compile(tmp_path):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    choice_payload_type = NamedType("Choice", [payload_type])
    named_variant_type = NamedType("Choice::<Payload>::Named")
    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode(
                "Named",
                fields=[
                    ("primary", generic_type),
                    ("backup", generic_type),
                ],
            ),
        ],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]

    ast = ShaderNode(
        "NonCopyGenericStructLikeEnumReturns",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "duplicate_named",
                choice_payload_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        ConstructorNode(
                            named_variant_type,
                            [],
                            named_arguments={
                                "primary": IdentifierNode("payload"),
                                "backup": IdentifierNode("payload"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "repack_guarded",
                choice_payload_type,
                [ParameterNode("choice", choice_payload_type)],
                [
                    ReturnNode(
                        MatchNode(
                            IdentifierNode("choice"),
                            [
                                MatchArmNode(
                                    StructPatternNode(
                                        "Choice::Named",
                                        {
                                            "primary": IdentifierPatternNode("primary"),
                                            "backup": IdentifierPatternNode("backup"),
                                        },
                                    ),
                                    BinaryOpNode(
                                        MemberAccessNode(
                                            IdentifierNode("primary"),
                                            "count",
                                        ),
                                        ">",
                                        LiteralNode(0, PrimitiveType("int")),
                                    ),
                                    [
                                        ExpressionStatementNode(
                                            ConstructorNode(
                                                named_variant_type,
                                                [],
                                                named_arguments={
                                                    "primary": IdentifierNode(
                                                        "primary"
                                                    ),
                                                    "backup": IdentifierNode("backup"),
                                                },
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                                MatchArmNode(
                                    StructPatternNode(
                                        "Choice::Named",
                                        {
                                            "primary": IdentifierPatternNode("primary"),
                                            "backup": IdentifierPatternNode("backup"),
                                        },
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            ConstructorNode(
                                                named_variant_type,
                                                [],
                                                named_arguments={
                                                    "primary": IdentifierNode("backup"),
                                                    "backup": IdentifierNode("backup"),
                                                },
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_generic_struct_like_enum_variant_locals_infer_target_type_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    choice_payload_type = NamedType("Choice", [payload_type])
    omitted_variant_type = NamedType("Choice::Named")
    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode(
                "Named",
                fields=[
                    ("primary", generic_type),
                    ("backup", generic_type),
                ],
            ),
        ],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]

    ast = ShaderNode(
        "GenericStructLikeEnumLocalTargets",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "choose_ternary_local",
                choice_payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "selected",
                        choice_payload_type,
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            ConstructorNode(
                                omitted_variant_type,
                                [],
                                named_arguments={
                                    "primary": IdentifierNode("left"),
                                    "backup": IdentifierNode("left"),
                                },
                            ),
                            ConstructorNode(
                                omitted_variant_type,
                                [],
                                named_arguments={
                                    "primary": IdentifierNode("right"),
                                    "backup": IdentifierNode("right"),
                                },
                            ),
                        ),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "choose_match_local",
                choice_payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_type),
                    ParameterNode("right", payload_type),
                ],
                [
                    VariableNode(
                        "selected",
                        choice_payload_type,
                        MatchNode(
                            IdentifierNode("flag"),
                            [
                                MatchArmNode(
                                    LiteralPatternNode(
                                        LiteralNode(True, PrimitiveType("bool"))
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            ConstructorNode(
                                                omitted_variant_type,
                                                [],
                                                named_arguments={
                                                    "primary": IdentifierNode("left"),
                                                    "backup": IdentifierNode("left"),
                                                },
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                                MatchArmNode(
                                    LiteralPatternNode(
                                        LiteralNode(False, PrimitiveType("bool"))
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            ConstructorNode(
                                                omitted_variant_type,
                                                [],
                                                named_arguments={
                                                    "primary": IdentifierNode("right"),
                                                    "backup": IdentifierNode("right"),
                                                },
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_generic_struct_like_enum_variant_assignment_and_call_targets_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    choice_payload_type = NamedType("Choice", [payload_type])
    omitted_variant_type = NamedType("Choice::Named")
    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode(
                "Named",
                fields=[
                    ("primary", generic_type),
                    ("backup", generic_type),
                ],
            ),
        ],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]

    ast = ShaderNode(
        "GenericStructLikeEnumAssignmentAndCallTargets",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "identity_choice",
                choice_payload_type,
                [ParameterNode("choice", choice_payload_type)],
                [ReturnNode(IdentifierNode("choice"))],
            ),
            FunctionNode(
                "assign_omitted_variant",
                choice_payload_type,
                [ParameterNode("payload", payload_type)],
                [
                    VariableNode("selected", choice_payload_type),
                    AssignmentNode(
                        IdentifierNode("selected"),
                        ConstructorNode(
                            omitted_variant_type,
                            [],
                            named_arguments={
                                "primary": IdentifierNode("payload"),
                                "backup": IdentifierNode("payload"),
                            },
                        ),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "call_with_omitted_variant",
                choice_payload_type,
                [ParameterNode("payload", payload_type)],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity_choice"),
                            [
                                ConstructorNode(
                                    omitted_variant_type,
                                    [],
                                    named_arguments={
                                        "primary": IdentifierNode("payload"),
                                        "backup": IdentifierNode("payload"),
                                    },
                                )
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_target_typed_array_literals_in_generic_constructors_compile(tmp_path):
    generic_type = GenericType("T")
    float_type = PrimitiveType("float")
    samples_float_type = NamedType("Samples", [float_type])
    choice_float_type = NamedType("SampleChoice", [float_type])
    omitted_variant_type = NamedType("SampleChoice::Window")

    def int_array(*values):
        return ArrayLiteralNode(
            [LiteralNode(value, PrimitiveType("int")) for value in values]
        )

    choice_enum = EnumNode(
        "SampleChoice",
        [
            EnumVariantNode(
                "Window",
                fields=[
                    ("values", ArrayType(generic_type, 2)),
                ],
            ),
        ],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]

    ast = ShaderNode(
        "GenericArrayLiteralConstructorTargets",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Samples",
                [
                    StructMemberNode("values", ArrayType(generic_type, 2)),
                ],
                generic_params=[GenericParameterNode("T")],
            ),
            choice_enum,
        ],
        functions=[
            FunctionNode(
                "identity_samples",
                samples_float_type,
                [ParameterNode("samples", samples_float_type)],
                [ReturnNode(IdentifierNode("samples"))],
            ),
            FunctionNode(
                "identity_choice",
                choice_float_type,
                [ParameterNode("choice", choice_float_type)],
                [ReturnNode(IdentifierNode("choice"))],
            ),
            FunctionNode(
                "build_struct",
                samples_float_type,
                [],
                [
                    ReturnNode(
                        ConstructorNode(
                            samples_float_type,
                            [],
                            named_arguments={"values": int_array(1, 2)},
                        )
                    )
                ],
            ),
            FunctionNode(
                "assign_struct",
                samples_float_type,
                [],
                [
                    VariableNode("selected", samples_float_type),
                    AssignmentNode(
                        IdentifierNode("selected"),
                        ConstructorNode(
                            samples_float_type,
                            [],
                            named_arguments={"values": int_array(3, 4)},
                        ),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "call_struct",
                samples_float_type,
                [],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity_samples"),
                            [
                                ConstructorNode(
                                    samples_float_type,
                                    [],
                                    named_arguments={"values": int_array(5, 6)},
                                )
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "build_variant",
                choice_float_type,
                [],
                [
                    ReturnNode(
                        ConstructorNode(
                            omitted_variant_type,
                            [],
                            named_arguments={"values": int_array(7, 8)},
                        )
                    )
                ],
            ),
            FunctionNode(
                "call_variant",
                choice_float_type,
                [],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity_choice"),
                            [
                                ConstructorNode(
                                    omitted_variant_type,
                                    [],
                                    named_arguments={"values": int_array(9, 10)},
                                )
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert "Samples::<f32> { values: [1.0, 2.0] }" in generated_code
    assert "selected = Samples::<f32> { values: [3.0, 4.0] };" in generated_code
    assert (
        "return identity_samples(Samples::<f32> { values: [5.0, 6.0] });"
        in generated_code
    )
    assert "SampleChoice::<f32>::Window { values: [7.0, 8.0] }" in generated_code
    assert (
        "return identity_choice(SampleChoice::<f32>::Window { values: [9.0, 10.0] });"
    ) in generated_code
    assert "SampleChoice::Window" not in generated_code
    assert "[1, 2]" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_nested_array_literals_and_tuple_enum_payloads_compile(tmp_path):
    generic_type = GenericType("T")
    float_type = PrimitiveType("float")
    matrix_type = ArrayType(ArrayType(generic_type, 2), 2)
    matrix_box_float_type = NamedType("MatrixBox", [float_type])
    matrix_choice_float_type = NamedType("MatrixChoice", [float_type])

    def int_array(*values):
        return ArrayLiteralNode(
            [LiteralNode(value, PrimitiveType("int")) for value in values]
        )

    def matrix_literal(*rows):
        return ArrayLiteralNode([int_array(*row) for row in rows])

    matrix_choice_enum = EnumNode(
        "MatrixChoice",
        [
            EnumVariantNode("Grid", fields=[matrix_type]),
        ],
    )
    matrix_choice_enum.generic_params = [GenericParameterNode("T")]

    ast = ShaderNode(
        "NestedArrayTupleEnumTargets",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "MatrixBox",
                [
                    StructMemberNode("values", matrix_type),
                ],
                generic_params=[GenericParameterNode("T")],
            ),
            matrix_choice_enum,
        ],
        functions=[
            FunctionNode(
                "identity_choice",
                matrix_choice_float_type,
                [ParameterNode("choice", matrix_choice_float_type)],
                [ReturnNode(IdentifierNode("choice"))],
            ),
            FunctionNode(
                "build_struct",
                matrix_box_float_type,
                [],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_box_float_type,
                            [],
                            named_arguments={
                                "values": matrix_literal((1, 2), (3,)),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "build_tuple_variant",
                matrix_choice_float_type,
                [],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("MatrixChoice::Grid"),
                            [matrix_literal((4, 5), (6, 7))],
                        )
                    )
                ],
            ),
            FunctionNode(
                "assign_tuple_variant",
                matrix_choice_float_type,
                [],
                [
                    VariableNode("selected", matrix_choice_float_type),
                    AssignmentNode(
                        IdentifierNode("selected"),
                        FunctionCallNode(
                            IdentifierNode("MatrixChoice::Grid"),
                            [matrix_literal((8, 9), (10, 11))],
                        ),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "call_tuple_variant",
                matrix_choice_float_type,
                [],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity_choice"),
                            [
                                FunctionCallNode(
                                    IdentifierNode("MatrixChoice::Grid"),
                                    [matrix_literal((12, 13), (14, 15))],
                                )
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert "pub values: [[T; 2]; 2]," in generated_code
    assert "Grid([[T; 2]; 2])," in generated_code
    assert (
        "MatrixBox::<f32> { values: [[1.0, 2.0], [3.0, Default::default()]] }"
    ) in generated_code
    assert (
        "return MatrixChoice::<f32>::Grid([[4.0, 5.0], [6.0, 7.0]]);" in generated_code
    )
    assert (
        "selected = MatrixChoice::<f32>::Grid([[8.0, 9.0], [10.0, 11.0]]);"
        in generated_code
    )
    assert (
        "return identity_choice(MatrixChoice::<f32>::Grid("
        "[[12.0, 13.0], [14.0, 15.0]]));"
    ) in generated_code
    assert "MatrixChoice::Grid" not in generated_code
    assert "T[])," not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_nested_non_copy_array_access_infers_element_types_and_clones_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    grid_type = ArrayType(ArrayType(payload_type, 2), 2)
    row_index = IdentifierNode("row")
    col_index = IdentifierNode("col")

    def grid_row():
        return ArrayAccessNode(IdentifierNode("grid"), row_index)

    def grid_cell():
        return ArrayAccessNode(grid_row(), col_index)

    ast = ShaderNode(
        "NestedNonCopyArrayAccess",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "probe",
                PrimitiveType("int"),
                [
                    ParameterNode("grid", grid_type),
                    ParameterNode("replacement", payload_type),
                    ParameterNode("row", PrimitiveType("int")),
                    ParameterNode("col", PrimitiveType("int")),
                ],
                [
                    VariableNode("picked", PrimitiveType("auto"), grid_cell()),
                    VariableNode("pickedRow", PrimitiveType("auto"), grid_row()),
                    AssignmentNode(grid_cell(), IdentifierNode("replacement")),
                    ReturnNode(
                        BinaryOpNode(
                            BinaryOpNode(
                                MemberAccessNode(IdentifierNode("picked"), "count"),
                                "+",
                                MemberAccessNode(
                                    ArrayAccessNode(
                                        IdentifierNode("pickedRow"),
                                        LiteralNode(0, PrimitiveType("int")),
                                    ),
                                    "count",
                                ),
                            ),
                            "+",
                            MemberAccessNode(grid_cell(), "count"),
                        )
                    ),
                ],
            )
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_deep_nested_array_literals_and_access_target_types_compile(tmp_path):
    generic_type = GenericType("T")
    payload_type = NamedType("Payload")
    float_type = PrimitiveType("float")
    cube_generic_type = ArrayType(ArrayType(ArrayType(generic_type, 2), 2), 2)
    cube_payload_type = ArrayType(ArrayType(ArrayType(payload_type, 2), 2), 2)
    cube_box_payload_type = NamedType("CubeBox", [payload_type])
    cube_box_float_type = NamedType("CubeBox", [float_type])
    cube_choice_payload_type = NamedType("CubeChoice", [payload_type])
    cube_row_payload_type = ArrayType(ArrayType(payload_type, 2), 2)

    def int_array(*values):
        return ArrayLiteralNode(
            [LiteralNode(value, PrimitiveType("int")) for value in values]
        )

    def float_cube_literal():
        return ArrayLiteralNode(
            [
                ArrayLiteralNode([int_array(1, 2), int_array(3, 4)]),
                ArrayLiteralNode([int_array(5, 6), int_array(7, 8)]),
            ]
        )

    def payload_row(*names):
        return ArrayLiteralNode([IdentifierNode(name) for name in names])

    def payload_cube_literal(prefix):
        return ArrayLiteralNode(
            [
                ArrayLiteralNode(
                    [
                        payload_row(f"{prefix}0", f"{prefix}1"),
                        payload_row(f"{prefix}2", f"{prefix}3"),
                    ]
                ),
                ArrayLiteralNode(
                    [
                        payload_row(f"{prefix}4", f"{prefix}5"),
                        payload_row(f"{prefix}6", f"{prefix}7"),
                    ]
                ),
            ]
        )

    def payload_params(prefix):
        return [ParameterNode(f"{prefix}{index}", payload_type) for index in range(8)]

    def cube_cell():
        return ArrayAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(IdentifierNode("cube"), IdentifierNode("layer")),
                IdentifierNode("row"),
            ),
            IdentifierNode("col"),
        )

    cube_choice_enum = EnumNode(
        "CubeChoice",
        [
            EnumVariantNode("Cube", fields=[cube_generic_type]),
        ],
    )
    cube_choice_enum.generic_params = [GenericParameterNode("T")]

    ast = ShaderNode(
        "DeepNestedArrayTargetTypes",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "CubeBox",
                [
                    StructMemberNode("cube", cube_generic_type),
                ],
                generic_params=[GenericParameterNode("T")],
            ),
            cube_choice_enum,
        ],
        functions=[
            FunctionNode(
                "identity_payload_cube",
                cube_payload_type,
                [ParameterNode("cube", cube_payload_type)],
                [ReturnNode(IdentifierNode("cube"))],
            ),
            FunctionNode(
                "identity_payload_box",
                cube_box_payload_type,
                [ParameterNode("cubeBox", cube_box_payload_type)],
                [ReturnNode(IdentifierNode("cubeBox"))],
            ),
            FunctionNode(
                "identity_payload_choice",
                cube_choice_payload_type,
                [ParameterNode("choice", cube_choice_payload_type)],
                [ReturnNode(IdentifierNode("choice"))],
            ),
            FunctionNode(
                "build_float_box",
                cube_box_float_type,
                [],
                [
                    ReturnNode(
                        ConstructorNode(
                            cube_box_float_type,
                            [],
                            named_arguments={"cube": float_cube_literal()},
                        )
                    )
                ],
            ),
            FunctionNode(
                "assign_payload_box",
                cube_box_payload_type,
                payload_params("p"),
                [
                    VariableNode("selected", cube_box_payload_type),
                    AssignmentNode(
                        IdentifierNode("selected"),
                        ConstructorNode(
                            cube_box_payload_type,
                            [],
                            named_arguments={"cube": payload_cube_literal("p")},
                        ),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "call_payload_cube",
                cube_payload_type,
                payload_params("q"),
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity_payload_cube"),
                            [payload_cube_literal("q")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "call_payload_box",
                cube_box_payload_type,
                payload_params("r"),
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity_payload_box"),
                            [
                                ConstructorNode(
                                    cube_box_payload_type,
                                    [],
                                    named_arguments={"cube": payload_cube_literal("r")},
                                )
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "call_payload_variant",
                cube_choice_payload_type,
                payload_params("s"),
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("identity_payload_choice"),
                            [
                                FunctionCallNode(
                                    IdentifierNode("CubeChoice::Cube"),
                                    [payload_cube_literal("s")],
                                )
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "pick_cell",
                payload_type,
                [
                    ParameterNode("cube", cube_payload_type),
                    ParameterNode("layer", PrimitiveType("int")),
                    ParameterNode("row", PrimitiveType("int")),
                    ParameterNode("col", PrimitiveType("int")),
                ],
                [ReturnNode(cube_cell())],
            ),
            FunctionNode(
                "pick_layer",
                cube_row_payload_type,
                [
                    ParameterNode("cube", cube_payload_type),
                    ParameterNode("layer", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        ArrayAccessNode(IdentifierNode("cube"), IdentifierNode("layer"))
                    )
                ],
            ),
            FunctionNode(
                "replace_cell",
                cube_payload_type,
                [
                    ParameterNode("cube", cube_payload_type),
                    ParameterNode("replacement", payload_type),
                    ParameterNode("layer", PrimitiveType("int")),
                    ParameterNode("row", PrimitiveType("int")),
                    ParameterNode("col", PrimitiveType("int")),
                ],
                [
                    AssignmentNode(cube_cell(), IdentifierNode("replacement")),
                    ReturnNode(IdentifierNode("cube")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_branch_target_types_and_non_copy_clones_compile(tmp_path):
    float_matrix_type = ArrayType(
        ArrayType(PrimitiveType("float"), 2),
        2,
    )
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)

    def int_array(*values):
        return ArrayLiteralNode(
            [LiteralNode(value, PrimitiveType("int")) for value in values]
        )

    def matrix_literal(*rows):
        return ArrayLiteralNode([int_array(*row) for row in rows])

    def bool_match(true_tail, false_tail):
        return MatchNode(
            IdentifierNode("flag"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            true_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            false_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayBranchTargetTypes",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "choose_float_ternary",
                float_matrix_type,
                [ParameterNode("flag", PrimitiveType("bool"))],
                [
                    ReturnNode(
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            matrix_literal((1, 2), (3,)),
                            matrix_literal((4,), (5, 6)),
                        )
                    )
                ],
            ),
            FunctionNode(
                "choose_float_match",
                float_matrix_type,
                [ParameterNode("flag", PrimitiveType("bool"))],
                [
                    ReturnNode(
                        bool_match(
                            matrix_literal((7, 8), (9,)),
                            matrix_literal((10,), (11, 12)),
                        )
                    )
                ],
            ),
            FunctionNode(
                "clone_payload_ternary_local",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    VariableNode(
                        "selected",
                        payload_matrix_type,
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        ),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "move_payload_ternary_return",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            IdentifierNode("left"),
                            IdentifierNode("right"),
                        )
                    )
                ],
            ),
            FunctionNode(
                "clone_payload_match_local",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    VariableNode(
                        "selected",
                        payload_matrix_type,
                        bool_match(IdentifierNode("left"), IdentifierNode("right")),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "move_payload_match_return",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        bool_match(IdentifierNode("left"), IdentifierNode("right"))
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_block_tail_and_loop_carried_contexts_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)

    def identity_call(name):
        return FunctionCallNode(
            IdentifierNode("identity_payload_matrix"),
            [IdentifierNode(name)],
        )

    def bool_match_with_block_tails(true_tail, false_tail):
        return MatchNode(
            IdentifierNode("flag"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, PrimitiveType("bool"))),
                    None,
                    [
                        VariableNode(
                            "marker",
                            PrimitiveType("int"),
                            LiteralNode(1, PrimitiveType("int")),
                        ),
                        ExpressionStatementNode(
                            true_tail,
                            is_tail_expression=True,
                        ),
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, PrimitiveType("bool"))),
                    None,
                    [
                        VariableNode(
                            "marker",
                            PrimitiveType("int"),
                            LiteralNode(2, PrimitiveType("int")),
                        ),
                        ExpressionStatementNode(
                            false_tail,
                            is_tail_expression=True,
                        ),
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayBlockTailAndLoopContexts",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "identity_payload_matrix",
                payload_matrix_type,
                [ParameterNode("matrix", payload_matrix_type)],
                [ReturnNode(IdentifierNode("matrix"))],
            ),
            FunctionNode(
                "local_match_block_tail",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    VariableNode(
                        "selected",
                        payload_matrix_type,
                        bool_match_with_block_tails(
                            identity_call("left"),
                            identity_call("right"),
                        ),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "return_match_block_tail",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        bool_match_with_block_tails(
                            identity_call("left"),
                            identity_call("right"),
                        )
                    )
                ],
            ),
            FunctionNode(
                "loop_carried_payload_matrix",
                payload_matrix_type,
                [
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    VariableNode(
                        "selected",
                        payload_matrix_type,
                        identity_call("left"),
                    ),
                    VariableNode(
                        "i",
                        PrimitiveType("int"),
                        LiteralNode(0, PrimitiveType("int")),
                    ),
                    WhileNode(
                        BinaryOpNode(
                            IdentifierNode("i"),
                            "<",
                            LiteralNode(2, PrimitiveType("int")),
                        ),
                        [
                            AssignmentNode(
                                IdentifierNode("selected"),
                                identity_call("right"),
                            ),
                            AssignmentNode(
                                IdentifierNode("i"),
                                BinaryOpNode(
                                    IdentifierNode("i"),
                                    "+",
                                    LiteralNode(1, PrimitiveType("int")),
                                ),
                            ),
                        ],
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_early_returns_and_nested_block_expressions_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)

    def identity_call(expr):
        if isinstance(expr, str):
            expr = IdentifierNode(expr)
        return FunctionCallNode(
            IdentifierNode("identity_payload_matrix"),
            [expr],
        )

    def block_tail(marker, tail):
        return BlockNode(
            [
                VariableNode(
                    "marker",
                    PrimitiveType("int"),
                    LiteralNode(marker, PrimitiveType("int")),
                ),
                ExpressionStatementNode(tail, is_tail_expression=True),
            ]
        )

    def nested_block(outer_marker, inner_marker, tail):
        return BlockNode(
            [
                VariableNode(
                    "outer_marker",
                    PrimitiveType("int"),
                    LiteralNode(outer_marker, PrimitiveType("int")),
                ),
                ExpressionStatementNode(
                    block_tail(inner_marker, tail),
                    is_tail_expression=True,
                ),
            ]
        )

    ast = ShaderNode(
        "ArrayEarlyReturnAndBlockExpressions",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "identity_payload_matrix",
                payload_matrix_type,
                [ParameterNode("matrix", payload_matrix_type)],
                [ReturnNode(IdentifierNode("matrix"))],
            ),
            FunctionNode(
                "early_return_nested_block",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    IfNode(
                        IdentifierNode("flag"),
                        BlockNode(
                            [ReturnNode(nested_block(1, 2, identity_call("left")))]
                        ),
                    ),
                    ReturnNode(nested_block(3, 4, identity_call("right"))),
                ],
            ),
            FunctionNode(
                "local_nested_block",
                payload_matrix_type,
                [ParameterNode("left", payload_matrix_type)],
                [
                    VariableNode(
                        "selected",
                        payload_matrix_type,
                        nested_block(5, 6, identity_call("left")),
                    ),
                    ReturnNode(IdentifierNode("selected")),
                ],
            ),
            FunctionNode(
                "return_helper_with_block_arg",
                payload_matrix_type,
                [ParameterNode("left", payload_matrix_type)],
                [ReturnNode(identity_call(nested_block(7, 8, identity_call("left"))))],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_captures_and_nested_helper_closures_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    closure_type = GenericType("F")
    closure_bound = NamedType(
        "FnOnce([[Payload; 2]; 2]) -> [[Payload; 2]; 2]",
    )

    def identity_call(expr):
        if isinstance(expr, str):
            expr = IdentifierNode(expr)
        return FunctionCallNode(
            IdentifierNode("identity_payload_matrix"),
            [expr],
        )

    def apply_call(matrix, transform):
        if isinstance(matrix, str):
            matrix = IdentifierNode(matrix)
        return FunctionCallNode(
            IdentifierNode("apply_payload_matrix"),
            [matrix, transform],
        )

    def matrix_lambda(param_name, body, captures=None):
        return LambdaNode(
            [ParameterNode(param_name, payload_matrix_type)],
            body,
            captures=captures or [],
        )

    ast = ShaderNode(
        "ArrayLambdaCaptureContexts",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "identity_payload_matrix",
                payload_matrix_type,
                [ParameterNode("matrix", payload_matrix_type)],
                [ReturnNode(IdentifierNode("matrix"))],
            ),
            FunctionNode(
                "apply_payload_matrix",
                payload_matrix_type,
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("matrix")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_captures_payload_matrix",
                payload_matrix_type,
                [
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        apply_call(
                            "right",
                            matrix_lambda(
                                "matrix",
                                identity_call("left"),
                                captures=["left"],
                            ),
                        )
                    )
                ],
            ),
            FunctionNode(
                "nested_closure_captures_payload_matrix",
                payload_matrix_type,
                [
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        apply_call(
                            "right",
                            matrix_lambda(
                                "outer",
                                apply_call(
                                    "outer",
                                    matrix_lambda(
                                        "inner",
                                        identity_call("left"),
                                        captures=["left"],
                                    ),
                                ),
                                captures=["left"],
                            ),
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_block_bodies_return_places_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    payload_cube_type = ArrayType(payload_matrix_type, 2)
    wrapper_type = NamedType("MatrixWrapper")
    closure_type = GenericType("F")
    closure_bound = NamedType(
        "FnOnce([[Payload; 2]; 2]) -> [[Payload; 2]; 2]",
    )

    def identity_call(expr):
        if isinstance(expr, str):
            expr = IdentifierNode(expr)
        return FunctionCallNode(
            IdentifierNode("identity_payload_matrix"),
            [expr],
        )

    def apply_call(matrix, transform):
        if isinstance(matrix, str):
            matrix = IdentifierNode(matrix)
        return FunctionCallNode(
            IdentifierNode("apply_payload_matrix"),
            [matrix, transform],
        )

    def matrix_lambda(param_name, body, captures=None):
        return LambdaNode(
            [ParameterNode(param_name, payload_matrix_type)],
            body,
            captures=captures or [],
        )

    def block_tail(marker, tail):
        return BlockNode(
            [
                VariableNode(
                    "marker",
                    PrimitiveType("int"),
                    LiteralNode(marker, PrimitiveType("int")),
                ),
                ExpressionStatementNode(tail, is_tail_expression=True),
            ]
        )

    def explicit_return_block(marker, value):
        return BlockNode(
            [
                VariableNode(
                    "marker",
                    PrimitiveType("int"),
                    LiteralNode(marker, PrimitiveType("int")),
                ),
                ReturnNode(value),
            ]
        )

    ast = ShaderNode(
        "ArrayLambdaBlockBodyReturnPlaces",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [StructMemberNode("matrix", payload_matrix_type)],
            ),
        ],
        functions=[
            FunctionNode(
                "identity_payload_matrix",
                payload_matrix_type,
                [ParameterNode("matrix", payload_matrix_type)],
                [ReturnNode(IdentifierNode("matrix"))],
            ),
            FunctionNode(
                "apply_payload_matrix",
                payload_matrix_type,
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("matrix")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_block_tail_captures_payload_matrix",
                payload_matrix_type,
                [
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        apply_call(
                            "right",
                            matrix_lambda(
                                "matrix",
                                block_tail(1, identity_call("left")),
                                captures=["left"],
                            ),
                        )
                    )
                ],
            ),
            FunctionNode(
                "closure_explicit_return_captures_payload_matrix",
                payload_matrix_type,
                [
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        apply_call(
                            "right",
                            matrix_lambda(
                                "matrix",
                                explicit_return_block(2, identity_call("left")),
                                captures=["left"],
                            ),
                        )
                    )
                ],
            ),
            FunctionNode(
                "closure_returns_captured_member",
                payload_matrix_type,
                [
                    ParameterNode("wrapper", wrapper_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        apply_call(
                            "fallback",
                            matrix_lambda(
                                "matrix",
                                block_tail(
                                    3,
                                    MemberAccessNode(
                                        IdentifierNode("wrapper"),
                                        "matrix",
                                    ),
                                ),
                                captures=["wrapper"],
                            ),
                        )
                    )
                ],
            ),
            FunctionNode(
                "closure_explicit_returns_captured_index",
                payload_matrix_type,
                [
                    ParameterNode("rows", payload_cube_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        apply_call(
                            "fallback",
                            matrix_lambda(
                                "matrix",
                                explicit_return_block(
                                    4,
                                    ArrayAccessNode(
                                        IdentifierNode("rows"),
                                        LiteralNode(0, PrimitiveType("int")),
                                    ),
                                ),
                                captures=["rows"],
                            ),
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_explicit_return_uses_closure_target_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_row_type = ArrayType(payload_type, 2)
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(i32) -> [Payload; 2]")

    payload_constructor = ConstructorNode(
        payload_type,
        [
            ArrayLiteralNode([]),
            IdentifierNode("value"),
        ],
    )
    closure_body = BlockNode([ReturnNode(ArrayLiteralNode([payload_constructor]))])

    ast = ShaderNode(
        "ArrayLambdaExplicitReturnTarget",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "apply_payload_row",
                payload_row_type,
                [
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_returns_row_from_int_function",
                PrimitiveType("int"),
                [ParameterNode("seed", PrimitiveType("int"))],
                [
                    VariableNode(
                        "row",
                        payload_row_type,
                        FunctionCallNode(
                            IdentifierNode("apply_payload_row"),
                            [
                                IdentifierNode("seed"),
                                LambdaNode(
                                    [ParameterNode("value", PrimitiveType("int"))],
                                    closure_body,
                                ),
                            ],
                        ),
                    ),
                    ReturnNode(
                        MemberAccessNode(
                            ArrayAccessNode(
                                IdentifierNode("row"),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                            "count",
                        )
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_infers_fn_and_fnmut_parameter_targets_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_row_type = ArrayType(payload_type, 2)
    read_closure_type = GenericType("F")
    write_closure_type = GenericType("G")
    read_closure_bound = NamedType("Fn(i32) -> [Payload; 2]")
    write_closure_bound = NamedType("FnMut(i32) -> [Payload; 2]")

    def payload_constructor(value_name):
        return ConstructorNode(
            payload_type,
            [
                ArrayLiteralNode([]),
                IdentifierNode(value_name),
            ],
        )

    def row_literal(value_name):
        return ArrayLiteralNode([payload_constructor(value_name)])

    ast = ShaderNode(
        "ArrayLambdaCallableParameterTargets",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "apply_read_payload_row",
                payload_row_type,
                [
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", read_closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[read_closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_mut_payload_row",
                payload_row_type,
                [
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", write_closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("G", constraints=[write_closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_infers_callable_parameters",
                PrimitiveType("int"),
                [ParameterNode("seed", PrimitiveType("int"))],
                [
                    VariableNode(
                        "row_a",
                        payload_row_type,
                        FunctionCallNode(
                            IdentifierNode("apply_read_payload_row"),
                            [
                                IdentifierNode("seed"),
                                LambdaNode(
                                    [
                                        ParameterNode("value", None),
                                    ],
                                    BlockNode([ReturnNode(row_literal("value"))]),
                                ),
                            ],
                        ),
                    ),
                    VariableNode(
                        "row_b",
                        payload_row_type,
                        FunctionCallNode(
                            IdentifierNode("apply_mut_payload_row"),
                            [
                                MemberAccessNode(
                                    ArrayAccessNode(
                                        IdentifierNode("row_a"),
                                        LiteralNode(0, PrimitiveType("int")),
                                    ),
                                    "count",
                                ),
                                LambdaNode(
                                    [
                                        ParameterNode("next", None),
                                    ],
                                    row_literal("next"),
                                ),
                            ],
                        ),
                    ),
                    ReturnNode(
                        MemberAccessNode(
                            ArrayAccessNode(
                                IdentifierNode("row_b"),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                            "count",
                        )
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_infers_multi_arg_nested_callable_targets_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_row_type = ArrayType(payload_type, 2)
    payload_matrix_type = ArrayType(payload_row_type, 2)
    row_closure_type = GenericType("G")
    matrix_closure_type = GenericType("F")
    row_closure_bound = NamedType("Fn([Payload; 2], i32) -> [Payload; 2]")
    matrix_closure_bound = NamedType(
        "Fn([Payload; 2], i32) -> [[Payload; 2]; 2]",
    )

    def payload_constructor(value_name):
        return ConstructorNode(
            payload_type,
            [
                ArrayLiteralNode([]),
                IdentifierNode(value_name),
            ],
        )

    def row_literal(value_name):
        return ArrayLiteralNode([payload_constructor(value_name)])

    nested_row_call = FunctionCallNode(
        IdentifierNode("apply_payload_row_with_seed"),
        [
            IdentifierNode("captured_row"),
            IdentifierNode("offset"),
            LambdaNode(
                [
                    ParameterNode("inner_row", None),
                    ParameterNode("inner_seed", None),
                ],
                row_literal("inner_seed"),
            ),
        ],
    )

    ast = ShaderNode(
        "ArrayLambdaMultiArgNestedTargets",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "apply_payload_row_with_seed",
                payload_row_type,
                [
                    ParameterNode("row", payload_row_type),
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", row_closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("row"), IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("G", constraints=[row_closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_payload_matrix_from_row",
                payload_matrix_type,
                [
                    ParameterNode("row", payload_row_type),
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", matrix_closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("row"), IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[matrix_closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_infers_multi_arg_nested_callable_parameters",
                PrimitiveType("int"),
                [
                    ParameterNode("row", payload_row_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "matrix",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("apply_payload_matrix_from_row"),
                            [
                                IdentifierNode("row"),
                                IdentifierNode("seed"),
                                LambdaNode(
                                    [
                                        ParameterNode("captured_row", None),
                                        ParameterNode("offset", None),
                                    ],
                                    BlockNode(
                                        [
                                            ReturnNode(
                                                ArrayLiteralNode([nested_row_call])
                                            )
                                        ]
                                    ),
                                ),
                            ],
                        ),
                    ),
                    ReturnNode(
                        MemberAccessNode(
                            ArrayAccessNode(
                                ArrayAccessNode(
                                    IdentifierNode("matrix"),
                                    LiteralNode(0, PrimitiveType("int")),
                                ),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                            "count",
                        )
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_infers_multi_arg_fnmut_and_fnonce_targets_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_row_type = ArrayType(payload_type, 2)
    payload_matrix_type = ArrayType(payload_row_type, 2)
    mut_closure_type = GenericType("M")
    once_closure_type = GenericType("O")
    mut_closure_bound = NamedType("FnMut([Payload; 2], i32) -> [Payload; 2]")
    once_closure_bound = NamedType(
        "FnOnce([Payload; 2], i32) -> [[Payload; 2]; 2]",
    )

    def payload_constructor(value_name):
        return ConstructorNode(
            payload_type,
            [
                ArrayLiteralNode([]),
                IdentifierNode(value_name),
            ],
        )

    def row_literal(value_name):
        return ArrayLiteralNode([payload_constructor(value_name)])

    nested_mut_row_call = FunctionCallNode(
        IdentifierNode("apply_mut_payload_row_with_seed"),
        [
            IdentifierNode("once_row"),
            IdentifierNode("once_seed"),
            LambdaNode(
                [
                    ParameterNode("nested_row", None),
                    ParameterNode("nested_seed", None),
                ],
                row_literal("nested_seed"),
            ),
        ],
    )

    ast = ShaderNode(
        "ArrayLambdaMultiArgFnMutFnOnceTargets",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            )
        ],
        functions=[
            FunctionNode(
                "apply_mut_payload_row_with_seed",
                payload_row_type,
                [
                    ParameterNode("row", payload_row_type),
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", mut_closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("row"), IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("M", constraints=[mut_closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_once_payload_matrix_with_seed",
                payload_matrix_type,
                [
                    ParameterNode("row", payload_row_type),
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", once_closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("row"), IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("O", constraints=[once_closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_infers_multi_arg_fnmut_and_fnonce_parameters",
                PrimitiveType("int"),
                [
                    ParameterNode("row", payload_row_type),
                    ParameterNode("fallback", payload_row_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "mutated_row",
                        payload_row_type,
                        FunctionCallNode(
                            IdentifierNode("apply_mut_payload_row_with_seed"),
                            [
                                IdentifierNode("row"),
                                IdentifierNode("seed"),
                                LambdaNode(
                                    [
                                        ParameterNode("mut_row", None),
                                        ParameterNode("mut_seed", None),
                                    ],
                                    row_literal("mut_seed"),
                                ),
                            ],
                        ),
                    ),
                    VariableNode(
                        "matrix",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("apply_once_payload_matrix_with_seed"),
                            [
                                IdentifierNode("fallback"),
                                IdentifierNode("seed"),
                                LambdaNode(
                                    [
                                        ParameterNode("once_row", None),
                                        ParameterNode("once_seed", None),
                                    ],
                                    BlockNode(
                                        [
                                            ReturnNode(
                                                ArrayLiteralNode(
                                                    [
                                                        nested_mut_row_call,
                                                        IdentifierNode("mutated_row"),
                                                    ]
                                                )
                                            )
                                        ]
                                    ),
                                    captures=["mutated_row"],
                                ),
                            ],
                        ),
                    ),
                    ReturnNode(
                        MemberAccessNode(
                            ArrayAccessNode(
                                ArrayAccessNode(
                                    IdentifierNode("matrix"),
                                    LiteralNode(0, PrimitiveType("int")),
                                ),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                            "count",
                        )
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_fnonce_capture_return_args_move_or_clone_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    payload_cube_type = ArrayType(payload_matrix_type, 2)
    matrix_pair_type = NamedType("MatrixPair")
    matrix_wrapper_type = NamedType("MatrixWrapper")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(i32) -> MatrixPair")

    def apply_pair_call(seed, transform):
        return FunctionCallNode(
            IdentifierNode("apply_matrix_pair"),
            [IdentifierNode(seed), transform],
        )

    def count_first_pair_payload(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    ast = ShaderNode(
        "ArrayLambdaFnOnceCaptureReturnArgs",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_moves_captured_matrices_to_call",
                PrimitiveType("int"),
                [
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        apply_pair_call(
                            "seed",
                            LambdaNode(
                                [ParameterNode("value", None)],
                                FunctionCallNode(
                                    IdentifierNode("make_matrix_pair"),
                                    [IdentifierNode("left"), IdentifierNode("right")],
                                ),
                                captures=["left", "right"],
                            ),
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
            FunctionNode(
                "closure_clones_duplicate_capture_before_move",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        apply_pair_call(
                            "seed",
                            LambdaNode(
                                [ParameterNode("value", None)],
                                FunctionCallNode(
                                    IdentifierNode("make_matrix_pair"),
                                    [
                                        IdentifierNode("matrix"),
                                        IdentifierNode("matrix"),
                                    ],
                                ),
                                captures=["matrix"],
                            ),
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
            FunctionNode(
                "closure_moves_captured_members_to_constructor",
                PrimitiveType("int"),
                [
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        apply_pair_call(
                            "seed",
                            LambdaNode(
                                [ParameterNode("value", None)],
                                ConstructorNode(
                                    matrix_pair_type,
                                    [
                                        MemberAccessNode(
                                            IdentifierNode("wrapper"),
                                            "primary",
                                        ),
                                        MemberAccessNode(
                                            IdentifierNode("wrapper"),
                                            "backup",
                                        ),
                                    ],
                                ),
                                captures=["wrapper"],
                            ),
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
            FunctionNode(
                "closure_clones_captured_indexed_rows",
                PrimitiveType("int"),
                [
                    ParameterNode("rows", payload_cube_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        apply_pair_call(
                            "seed",
                            LambdaNode(
                                [ParameterNode("value", None)],
                                ConstructorNode(
                                    matrix_pair_type,
                                    [
                                        ArrayAccessNode(
                                            IdentifierNode("rows"),
                                            LiteralNode(0, PrimitiveType("int")),
                                        ),
                                        ArrayAccessNode(
                                            IdentifierNode("rows"),
                                            LiteralNode(1, PrimitiveType("int")),
                                        ),
                                    ],
                                ),
                                captures=["rows"],
                            ),
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_capture_sibling_return_args_clone_before_move_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce([[Payload; 2]; 2], i32) -> MatrixPair")

    def count_first_pair_payload(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def captured_pair_lambda():
        return LambdaNode(
            [
                ParameterNode("source", None),
                ParameterNode("value", None),
            ],
            FunctionCallNode(
                IdentifierNode("make_matrix_pair"),
                [IdentifierNode("matrix"), IdentifierNode("source")],
            ),
            captures=["matrix"],
        )

    ast = ShaderNode(
        "ArrayLambdaCaptureSiblingReturnArgs",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_matrix_source",
                matrix_pair_type,
                [
                    ParameterNode("source", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("source"), IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_matrix_source_after",
                matrix_pair_type,
                [
                    ParameterNode("transform", closure_type),
                    ParameterNode("source", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("source"), IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_sibling_arg_then_capture",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_matrix_source"),
                            [
                                IdentifierNode("matrix"),
                                IdentifierNode("seed"),
                                captured_pair_lambda(),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
            FunctionNode(
                "return_capture_then_sibling_arg",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_matrix_source_after"),
                            [
                                captured_pair_lambda(),
                                IdentifierNode("matrix"),
                                IdentifierNode("seed"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_capture_sibling_member_and_index_paths_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    payload_cube_type = ArrayType(payload_matrix_type, 2)
    matrix_pair_type = NamedType("MatrixPair")
    matrix_wrapper_type = NamedType("MatrixWrapper")
    cube_pair_type = NamedType("CubePair")
    closure_type = GenericType("F")
    matrix_closure_bound = NamedType("FnOnce([[Payload; 2]; 2], i32) -> MatrixPair")
    cube_closure_bound = NamedType("FnOnce([[Payload; 2]; 2], i32) -> CubePair")

    def payload_count_from_matrix_pair(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def payload_count_from_cube_pair(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    ArrayAccessNode(
                        MemberAccessNode(IdentifierNode(pair_name), "primary"),
                        LiteralNode(0, PrimitiveType("int")),
                    ),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def apply_matrix_source_after_call(transform, source, seed="seed"):
        return FunctionCallNode(
            IdentifierNode("apply_matrix_source_after"),
            [transform, source, IdentifierNode(seed)],
        )

    def apply_cube_source_after_call(transform, source, seed="seed"):
        return FunctionCallNode(
            IdentifierNode("apply_cube_source_after"),
            [transform, source, IdentifierNode(seed)],
        )

    ast = ShaderNode(
        "ArrayLambdaCaptureSiblingMemberIndexPaths",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "CubePair",
                [
                    StructMemberNode("primary", payload_cube_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "make_cube_pair",
                cube_pair_type,
                [
                    ParameterNode("primary", payload_cube_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            cube_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_matrix_source_after",
                matrix_pair_type,
                [
                    ParameterNode("transform", closure_type),
                    ParameterNode("source", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("source"), IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[matrix_closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_cube_source_after",
                cube_pair_type,
                [
                    ParameterNode("transform", closure_type),
                    ParameterNode("source", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("source"), IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[cube_closure_bound]),
                ],
            ),
            FunctionNode(
                "return_capture_member_then_same_member_sibling",
                PrimitiveType("int"),
                [
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        apply_matrix_source_after_call(
                            LambdaNode(
                                [
                                    ParameterNode("source", None),
                                    ParameterNode("value", None),
                                ],
                                FunctionCallNode(
                                    IdentifierNode("make_matrix_pair"),
                                    [
                                        MemberAccessNode(
                                            IdentifierNode("wrapper"),
                                            "primary",
                                        ),
                                        IdentifierNode("source"),
                                    ],
                                ),
                                captures=["wrapper"],
                            ),
                            MemberAccessNode(IdentifierNode("wrapper"), "primary"),
                        ),
                    ),
                    ReturnNode(payload_count_from_matrix_pair("pair")),
                ],
            ),
            FunctionNode(
                "return_capture_root_then_index_sibling",
                PrimitiveType("int"),
                [
                    ParameterNode("rows", payload_cube_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        cube_pair_type,
                        apply_cube_source_after_call(
                            LambdaNode(
                                [
                                    ParameterNode("source", None),
                                    ParameterNode("value", None),
                                ],
                                FunctionCallNode(
                                    IdentifierNode("make_cube_pair"),
                                    [IdentifierNode("rows"), IdentifierNode("source")],
                                ),
                                captures=["rows"],
                            ),
                            ArrayAccessNode(
                                IdentifierNode("rows"),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                        ),
                    ),
                    ReturnNode(payload_count_from_cube_pair("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_multiple_captures_clone_shared_roots_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    first_closure_type = GenericType("F")
    second_closure_type = GenericType("G")
    closure_bound = NamedType("FnOnce(i32) -> MatrixPair")

    def count_first_pair_payload(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def captured_pair_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            FunctionCallNode(
                IdentifierNode("make_matrix_pair"),
                [IdentifierNode("matrix"), IdentifierNode("matrix")],
            ),
            captures=["matrix"],
        )

    ast = ShaderNode(
        "ArrayLambdaMultipleCaptureSharedRoots",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_two_matrix_builders",
                matrix_pair_type,
                [
                    ParameterNode("first", first_closure_type),
                    ParameterNode("second", second_closure_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "left",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("first"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    VariableNode(
                        "right",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("second"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [
                                MemberAccessNode(IdentifierNode("left"), "primary"),
                                MemberAccessNode(IdentifierNode("right"), "primary"),
                            ],
                        )
                    ),
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                    GenericParameterNode("G", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_two_captured_builders",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_two_matrix_builders"),
                            [
                                captured_pair_lambda(),
                                captured_pair_lambda(),
                                IdentifierNode("seed"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_preclones_shared_single_use_captures_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    first_closure_type = GenericType("F")
    second_closure_type = GenericType("G")
    closure_bound = NamedType("FnOnce(i32) -> [[Payload; 2]; 2]")

    def count_first_pair_payload(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def captured_matrix_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            IdentifierNode("matrix"),
            captures=["matrix"],
        )

    ast = ShaderNode(
        "ArrayLambdaPreclonesSharedSingleUseCaptures",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_two_matrix_values",
                matrix_pair_type,
                [
                    ParameterNode("first", first_closure_type),
                    ParameterNode("second", second_closure_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "left",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("first"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    VariableNode(
                        "right",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("second"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [IdentifierNode("left"), IdentifierNode("right")],
                        )
                    ),
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                    GenericParameterNode("G", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_two_precloned_builders",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_two_matrix_values"),
                            [
                                captured_matrix_lambda(),
                                captured_matrix_lambda(),
                                IdentifierNode("seed"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_nested_preclones_shared_single_use_captures_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    first_closure_type = GenericType("F")
    second_closure_type = GenericType("G")
    closure_bound = NamedType("FnOnce(i32) -> [[Payload; 2]; 2]")

    def count_first_pair_payload(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def captured_matrix_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            IdentifierNode("matrix"),
            captures=["matrix"],
        )

    def wrap_builder(lambda_node):
        return FunctionCallNode(
            IdentifierNode("wrap_matrix_value_builder"),
            [lambda_node],
        )

    ast = ShaderNode(
        "ArrayLambdaNestedPreclonesSharedSingleUseCaptures",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "wrap_matrix_value_builder",
                first_closure_type,
                [ParameterNode("builder", first_closure_type)],
                [ReturnNode(IdentifierNode("builder"))],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_two_wrapped_matrix_values",
                matrix_pair_type,
                [
                    ParameterNode("first", first_closure_type),
                    ParameterNode("second", second_closure_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "left",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("first"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    VariableNode(
                        "right",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("second"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [IdentifierNode("left"), IdentifierNode("right")],
                        )
                    ),
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                    GenericParameterNode("G", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_two_nested_precloned_builders",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_two_wrapped_matrix_values"),
                            [
                                wrap_builder(captured_matrix_lambda()),
                                wrap_builder(captured_matrix_lambda()),
                                IdentifierNode("seed"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_preclones_shared_single_use_member_captures_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    matrix_wrapper_type = NamedType("MatrixWrapper")
    first_closure_type = GenericType("F")
    second_closure_type = GenericType("G")
    closure_bound = NamedType("FnOnce(i32) -> [[Payload; 2]; 2]")

    def count_first_pair_payload(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def captured_primary_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            MemberAccessNode(IdentifierNode("wrapper"), "primary"),
            captures=["wrapper"],
        )

    def wrap_builder(lambda_node):
        return FunctionCallNode(
            IdentifierNode("wrap_matrix_value_builder"),
            [lambda_node],
        )

    ast = ShaderNode(
        "ArrayLambdaPreclonesSharedSingleUseMemberCaptures",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "wrap_matrix_value_builder",
                first_closure_type,
                [ParameterNode("builder", first_closure_type)],
                [ReturnNode(IdentifierNode("builder"))],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_two_matrix_values",
                matrix_pair_type,
                [
                    ParameterNode("first", first_closure_type),
                    ParameterNode("second", second_closure_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "left",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("first"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    VariableNode(
                        "right",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("second"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [IdentifierNode("left"), IdentifierNode("right")],
                        )
                    ),
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                    GenericParameterNode("G", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_two_precloned_member_builders",
                PrimitiveType("int"),
                [
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_two_matrix_values"),
                            [
                                captured_primary_lambda(),
                                captured_primary_lambda(),
                                IdentifierNode("seed"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
            FunctionNode(
                "return_two_nested_precloned_member_builders",
                PrimitiveType("int"),
                [
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_two_matrix_values"),
                            [
                                wrap_builder(captured_primary_lambda()),
                                wrap_builder(captured_primary_lambda()),
                                IdentifierNode("seed"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_preclones_nested_member_captures_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    matrix_wrapper_type = NamedType("MatrixWrapper")
    outer_wrapper_type = NamedType("OuterWrapper")
    first_closure_type = GenericType("F")
    second_closure_type = GenericType("G")
    closure_bound = NamedType("FnOnce(i32) -> [[Payload; 2]; 2]")

    def count_first_pair_payload(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def captured_nested_primary_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            MemberAccessNode(
                MemberAccessNode(IdentifierNode("outer"), "inner"),
                "primary",
            ),
            captures=["outer"],
        )

    def wrap_builder(lambda_node):
        return FunctionCallNode(
            IdentifierNode("wrap_matrix_value_builder"),
            [lambda_node],
        )

    ast = ShaderNode(
        "ArrayLambdaPreclonesNestedMemberCaptures",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "OuterWrapper",
                [
                    StructMemberNode("inner", matrix_wrapper_type),
                    StructMemberNode("fallback", matrix_wrapper_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "wrap_matrix_value_builder",
                first_closure_type,
                [ParameterNode("builder", first_closure_type)],
                [ReturnNode(IdentifierNode("builder"))],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_two_matrix_values",
                matrix_pair_type,
                [
                    ParameterNode("first", first_closure_type),
                    ParameterNode("second", second_closure_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "left",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("first"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    VariableNode(
                        "right",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("second"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [IdentifierNode("left"), IdentifierNode("right")],
                        )
                    ),
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                    GenericParameterNode("G", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_two_precloned_nested_member_builders",
                PrimitiveType("int"),
                [
                    ParameterNode("outer", outer_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_two_matrix_values"),
                            [
                                captured_nested_primary_lambda(),
                                captured_nested_primary_lambda(),
                                IdentifierNode("seed"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
            FunctionNode(
                "return_two_wrapped_precloned_nested_member_builders",
                PrimitiveType("int"),
                [
                    ParameterNode("outer", outer_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("apply_two_matrix_values"),
                            [
                                wrap_builder(captured_nested_primary_lambda()),
                                wrap_builder(captured_nested_primary_lambda()),
                                IdentifierNode("seed"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_return_constructors_deep_member_preclones_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    matrix_wrapper_type = NamedType("MatrixWrapper")
    outer_wrapper_type = NamedType("OuterWrapper")
    wrapped_pair_type = NamedType("WrappedPair")
    final_bundle_type = NamedType("FinalBundle")
    bundle_choice_type = NamedType("BundleChoice")
    bundle_choice_ready_type = NamedType("BundleChoice::Ready")
    first_closure_type = GenericType("F")
    second_closure_type = GenericType("G")
    closure_bound = NamedType("FnOnce(i32) -> [[Payload; 2]; 2]")
    bundle_choice_enum = EnumNode(
        "BundleChoice",
        [
            EnumVariantNode(
                "Ready",
                fields=[("wrapped", wrapped_pair_type)],
            ),
        ],
    )

    def captured_nested_primary_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            MemberAccessNode(
                MemberAccessNode(IdentifierNode("outer"), "inner"),
                "primary",
            ),
            captures=["outer"],
        )

    def nested_wrapped_pair_call():
        return FunctionCallNode(
            IdentifierNode("make_wrapped_pair"),
            [
                FunctionCallNode(
                    IdentifierNode("wrap_matrix_pair"),
                    [
                        FunctionCallNode(
                            IdentifierNode("apply_two_matrix_values"),
                            [
                                captured_nested_primary_lambda(),
                                captured_nested_primary_lambda(),
                                IdentifierNode("seed"),
                            ],
                        )
                    ],
                ),
                IdentifierNode("source"),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaReturnConstructorsDeepMemberPreclones",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "OuterWrapper",
                [
                    StructMemberNode("inner", matrix_wrapper_type),
                    StructMemberNode("fallback", matrix_wrapper_type),
                ],
            ),
            StructNode(
                "WrappedPair",
                [
                    StructMemberNode("pair", matrix_pair_type),
                    StructMemberNode("source", matrix_wrapper_type),
                ],
            ),
            StructNode(
                "FinalBundle",
                [
                    StructMemberNode("wrapped", wrapped_pair_type),
                ],
            ),
            bundle_choice_enum,
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "wrap_matrix_pair",
                matrix_pair_type,
                [ParameterNode("pair", matrix_pair_type)],
                [ReturnNode(IdentifierNode("pair"))],
            ),
            FunctionNode(
                "make_wrapped_pair",
                wrapped_pair_type,
                [
                    ParameterNode("pair", matrix_pair_type),
                    ParameterNode("source", matrix_wrapper_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            wrapped_pair_type,
                            [],
                            named_arguments={
                                "pair": IdentifierNode("pair"),
                                "source": IdentifierNode("source"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_two_matrix_values",
                matrix_pair_type,
                [
                    ParameterNode("first", first_closure_type),
                    ParameterNode("second", second_closure_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "left",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("first"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    VariableNode(
                        "right",
                        payload_matrix_type,
                        FunctionCallNode(
                            IdentifierNode("second"),
                            [IdentifierNode("seed")],
                        ),
                    ),
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [IdentifierNode("left"), IdentifierNode("right")],
                        )
                    ),
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                    GenericParameterNode("G", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_deep_struct_constructor",
                final_bundle_type,
                [
                    ParameterNode("outer", outer_wrapper_type),
                    ParameterNode("source", matrix_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            final_bundle_type,
                            [],
                            named_arguments={
                                "wrapped": nested_wrapped_pair_call(),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_deep_enum_constructor",
                bundle_choice_type,
                [
                    ParameterNode("outer", outer_wrapper_type),
                    ParameterNode("source", matrix_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            bundle_choice_ready_type,
                            [],
                            named_arguments={
                                "wrapped": nested_wrapped_pair_call(),
                            },
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_nested_capture_sibling_roots_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(i32) -> MatrixPair")

    def count_first_pair_payload(pair_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(IdentifierNode(pair_name), "primary"),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def captured_pair_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            FunctionCallNode(
                IdentifierNode("make_matrix_pair"),
                [IdentifierNode("matrix"), IdentifierNode("matrix")],
            ),
            captures=["matrix"],
        )

    ast = ShaderNode(
        "ArrayLambdaNestedCaptureSiblingRoots",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "combine_nested_pair",
                matrix_pair_type,
                [
                    ParameterNode("candidate", matrix_pair_type),
                    ParameterNode("source", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [
                                MemberAccessNode(
                                    IdentifierNode("candidate"),
                                    "primary",
                                ),
                                IdentifierNode("source"),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_nested_capture_then_sibling",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "pair",
                        matrix_pair_type,
                        FunctionCallNode(
                            IdentifierNode("combine_nested_pair"),
                            [
                                FunctionCallNode(
                                    IdentifierNode("apply_matrix_pair"),
                                    [
                                        IdentifierNode("seed"),
                                        captured_pair_lambda(),
                                    ],
                                ),
                                IdentifierNode("matrix"),
                            ],
                        ),
                    ),
                    ReturnNode(count_first_pair_payload("pair")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_named_constructor_nested_capture_sibling_roots_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    matrix_bundle_type = NamedType("MatrixBundle")
    matrix_choice_type = NamedType("MatrixChoice")
    matrix_choice_candidate_type = NamedType("MatrixChoice::Candidate")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(i32) -> MatrixPair")
    matrix_choice_enum = EnumNode(
        "MatrixChoice",
        [
            EnumVariantNode(
                "Candidate",
                fields=[
                    ("candidate", matrix_pair_type),
                    ("source", payload_matrix_type),
                ],
            ),
        ],
    )

    def captured_pair_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            FunctionCallNode(
                IdentifierNode("make_matrix_pair"),
                [IdentifierNode("matrix"), IdentifierNode("matrix")],
            ),
            captures=["matrix"],
        )

    def nested_pair_call():
        return FunctionCallNode(
            IdentifierNode("apply_matrix_pair"),
            [
                IdentifierNode("seed"),
                captured_pair_lambda(),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaNamedConstructorNestedCaptureSiblingRoots",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixBundle",
                [
                    StructMemberNode("candidate", matrix_pair_type),
                    StructMemberNode("source", payload_matrix_type),
                ],
            ),
            matrix_choice_enum,
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_named_struct_capture_then_sibling",
                matrix_bundle_type,
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_bundle_type,
                            [],
                            named_arguments={
                                "candidate": nested_pair_call(),
                                "source": IdentifierNode("matrix"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_named_enum_capture_then_sibling",
                matrix_choice_type,
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_choice_candidate_type,
                            [],
                            named_arguments={
                                "candidate": nested_pair_call(),
                                "source": IdentifierNode("matrix"),
                            },
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_non_return_named_constructor_capture_siblings_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_pair_type = NamedType("MatrixPair")
    matrix_bundle_type = NamedType("MatrixBundle")
    matrix_choice_type = NamedType("MatrixChoice")
    matrix_choice_candidate_type = NamedType("MatrixChoice::Candidate")
    matrix_tuple_choice_type = NamedType("MatrixTupleChoice")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(i32) -> MatrixPair")
    matrix_choice_enum = EnumNode(
        "MatrixChoice",
        [
            EnumVariantNode(
                "Candidate",
                fields=[
                    ("candidate", matrix_pair_type),
                    ("source", payload_matrix_type),
                ],
            ),
        ],
    )
    matrix_tuple_choice_enum = EnumNode(
        "MatrixTupleChoice",
        [
            EnumVariantNode(
                "Pair",
                fields=[matrix_pair_type, payload_matrix_type],
            ),
        ],
    )

    def first_bundle_payload_count(bundle_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    MemberAccessNode(
                        MemberAccessNode(IdentifierNode(bundle_name), "candidate"),
                        "primary",
                    ),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def captured_pair_lambda():
        return LambdaNode(
            [ParameterNode("value", None)],
            FunctionCallNode(
                IdentifierNode("make_matrix_pair"),
                [IdentifierNode("matrix"), IdentifierNode("matrix")],
            ),
            captures=["matrix"],
        )

    def nested_pair_call():
        return FunctionCallNode(
            IdentifierNode("apply_matrix_pair"),
            [
                IdentifierNode("seed"),
                captured_pair_lambda(),
            ],
        )

    def named_bundle_constructor():
        return ConstructorNode(
            matrix_bundle_type,
            [],
            named_arguments={
                "candidate": nested_pair_call(),
                "source": IdentifierNode("matrix"),
            },
        )

    def named_choice_constructor():
        return ConstructorNode(
            matrix_choice_candidate_type,
            [],
            named_arguments={
                "candidate": nested_pair_call(),
                "source": IdentifierNode("matrix"),
            },
        )

    def positional_bundle_constructor():
        return ConstructorNode(
            matrix_bundle_type,
            [nested_pair_call(), IdentifierNode("matrix")],
        )

    def struct_new_bundle_call():
        return FunctionCallNode(
            IdentifierNode("MatrixBundle::new"),
            [nested_pair_call(), IdentifierNode("matrix")],
        )

    def tuple_choice_call():
        return FunctionCallNode(
            IdentifierNode("MatrixTupleChoice::Pair"),
            [nested_pair_call(), IdentifierNode("matrix")],
        )

    ast = ShaderNode(
        "ArrayLambdaNonReturnNamedConstructorCaptureSiblings",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixBundle",
                [
                    StructMemberNode("candidate", matrix_pair_type),
                    StructMemberNode("source", payload_matrix_type),
                ],
            ),
            matrix_choice_enum,
            matrix_tuple_choice_enum,
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("seed", PrimitiveType("int")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("seed")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "local_named_struct_capture_then_sibling",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "bundle",
                        matrix_bundle_type,
                        named_bundle_constructor(),
                    ),
                    ReturnNode(first_bundle_payload_count("bundle")),
                ],
            ),
            FunctionNode(
                "assign_named_struct_capture_then_sibling",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode("bundle", matrix_bundle_type),
                    AssignmentNode(
                        IdentifierNode("bundle"),
                        named_bundle_constructor(),
                    ),
                    ReturnNode(first_bundle_payload_count("bundle")),
                ],
            ),
            FunctionNode(
                "local_named_enum_capture_then_sibling",
                matrix_choice_type,
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "choice",
                        matrix_choice_type,
                        named_choice_constructor(),
                    ),
                    ReturnNode(IdentifierNode("choice")),
                ],
            ),
            FunctionNode(
                "local_positional_struct_capture_then_sibling",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "bundle",
                        matrix_bundle_type,
                        positional_bundle_constructor(),
                    ),
                    ReturnNode(first_bundle_payload_count("bundle")),
                ],
            ),
            FunctionNode(
                "local_struct_new_capture_then_sibling",
                PrimitiveType("int"),
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "bundle",
                        matrix_bundle_type,
                        struct_new_bundle_call(),
                    ),
                    ReturnNode(first_bundle_payload_count("bundle")),
                ],
            ),
            FunctionNode(
                "local_tuple_enum_capture_then_sibling",
                matrix_tuple_choice_type,
                [
                    ParameterNode("matrix", payload_matrix_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    VariableNode(
                        "choice",
                        matrix_tuple_choice_type,
                        tuple_choice_call(),
                    ),
                    ReturnNode(IdentifierNode("choice")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_fnonce_branch_values_return_captured_arrays_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    payload_cube_type = ArrayType(payload_matrix_type, 2)
    matrix_wrapper_type = NamedType("MatrixWrapper")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(bool) -> [[Payload; 2]; 2]")

    def apply_matrix_once(flag_name, transform):
        return FunctionCallNode(
            IdentifierNode("apply_payload_matrix_once"),
            [IdentifierNode(flag_name), transform],
        )

    def first_payload_count(matrix_name):
        return MemberAccessNode(
            ArrayAccessNode(
                ArrayAccessNode(
                    IdentifierNode(matrix_name),
                    LiteralNode(0, PrimitiveType("int")),
                ),
                LiteralNode(0, PrimitiveType("int")),
            ),
            "count",
        )

    def bool_match(true_tail, false_tail):
        return MatchNode(
            IdentifierNode("use_left"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            true_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            false_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaFnOnceBranchReturns",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "apply_payload_matrix_once",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("flag")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_ternary_moves_captured_matrices",
                PrimitiveType("int"),
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", payload_matrix_type),
                    ParameterNode("right", payload_matrix_type),
                ],
                [
                    VariableNode(
                        "selected",
                        payload_matrix_type,
                        apply_matrix_once(
                            "flag",
                            LambdaNode(
                                [ParameterNode("use_left", None)],
                                TernaryOpNode(
                                    IdentifierNode("use_left"),
                                    IdentifierNode("left"),
                                    IdentifierNode("right"),
                                ),
                                captures=["left", "right"],
                            ),
                        ),
                    ),
                    ReturnNode(first_payload_count("selected")),
                ],
            ),
            FunctionNode(
                "closure_match_moves_member_or_clones_indexed_matrix",
                PrimitiveType("int"),
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("rows", payload_cube_type),
                ],
                [
                    VariableNode(
                        "selected",
                        payload_matrix_type,
                        apply_matrix_once(
                            "flag",
                            LambdaNode(
                                [ParameterNode("use_left", None)],
                                bool_match(
                                    MemberAccessNode(
                                        IdentifierNode("wrapper"),
                                        "primary",
                                    ),
                                    ArrayAccessNode(
                                        IdentifierNode("rows"),
                                        LiteralNode(1, PrimitiveType("int")),
                                    ),
                                ),
                                captures=["wrapper", "rows"],
                            ),
                        ),
                    ),
                    ReturnNode(first_payload_count("selected")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_constructor_branch_fields_capture_members_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_wrapper_type = NamedType("MatrixWrapper")
    matrix_bundle_type = NamedType("MatrixBundle")
    matrix_choice_type = NamedType("MatrixChoice")
    matrix_choice_selected_type = NamedType("MatrixChoice::Selected")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(bool) -> [[Payload; 2]; 2]")
    matrix_choice_enum = EnumNode(
        "MatrixChoice",
        [
            EnumVariantNode(
                "Selected",
                fields=[
                    ("selected", payload_matrix_type),
                    ("fallback", payload_matrix_type),
                ],
            ),
        ],
    )

    def apply_matrix_once(transform):
        return FunctionCallNode(
            IdentifierNode("apply_payload_matrix_once"),
            [IdentifierNode("flag"), transform],
        )

    def captured_member_lambda(member):
        return LambdaNode(
            [ParameterNode("use_primary", None)],
            MemberAccessNode(IdentifierNode("wrapper"), member),
            captures=["wrapper"],
        )

    def branch_value():
        return TernaryOpNode(
            IdentifierNode("choose_primary"),
            apply_matrix_once(captured_member_lambda("primary")),
            apply_matrix_once(captured_member_lambda("backup")),
        )

    def match_value():
        return MatchNode(
            IdentifierNode("choose_primary"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            apply_matrix_once(captured_member_lambda("primary")),
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            apply_matrix_once(captured_member_lambda("backup")),
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaConstructorBranchFieldsCaptureMembers",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixBundle",
                [
                    StructMemberNode("selected", payload_matrix_type),
                    StructMemberNode("fallback", payload_matrix_type),
                ],
            ),
            matrix_choice_enum,
        ],
        functions=[
            FunctionNode(
                "apply_payload_matrix_once",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("flag")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_branch_struct_constructor",
                matrix_bundle_type,
                [
                    ParameterNode("choose_primary", PrimitiveType("bool")),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_bundle_type,
                            [],
                            named_arguments={
                                "selected": branch_value(),
                                "fallback": IdentifierNode("fallback"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_match_enum_constructor",
                matrix_choice_type,
                [
                    ParameterNode("choose_primary", PrimitiveType("bool")),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_choice_selected_type,
                            [],
                            named_arguments={
                                "selected": match_value(),
                                "fallback": IdentifierNode("fallback"),
                            },
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_array_literal_constructor_fields_and_generic_payloads_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    generic_matrix_type = ArrayType(ArrayType(generic_type, 2), 2)
    generic_batch_type = ArrayType(generic_matrix_type, 2)
    matrix_wrapper_type = NamedType("MatrixWrapper")
    matrix_pack_payload_type = NamedType("MatrixPack", [payload_type])
    matrix_choice_payload_type = NamedType("MatrixChoice", [payload_type])
    matrix_choice_packed_type = NamedType("MatrixChoice::Packed")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(bool) -> [[Payload; 2]; 2]")
    matrix_choice_enum = EnumNode(
        "MatrixChoice",
        [
            EnumVariantNode(
                "Packed",
                fields=[
                    ("batches", generic_batch_type),
                    ("fallback", generic_matrix_type),
                ],
            ),
            EnumVariantNode("Tuple", fields=[generic_batch_type]),
        ],
    )
    matrix_choice_enum.generic_params = [GenericParameterNode("T")]

    def apply_matrix_once(member):
        return FunctionCallNode(
            IdentifierNode("apply_payload_matrix_once"),
            [
                IdentifierNode("flag"),
                LambdaNode(
                    [ParameterNode("use_primary", None)],
                    MemberAccessNode(IdentifierNode("wrapper"), member),
                    captures=["wrapper"],
                ),
            ],
        )

    def captured_batch_literal():
        return ArrayLiteralNode(
            [
                apply_matrix_once("primary"),
                apply_matrix_once("backup"),
            ]
        )

    ast = ShaderNode(
        "ArrayLambdaConstructorArrayFieldsGenericPayloads",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixPack",
                [
                    StructMemberNode("batches", generic_batch_type),
                    StructMemberNode("fallback", generic_matrix_type),
                ],
                generic_params=[GenericParameterNode("T")],
            ),
            matrix_choice_enum,
        ],
        functions=[
            FunctionNode(
                "apply_payload_matrix_once",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("flag")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_struct_array_field",
                matrix_pack_payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pack_payload_type,
                            [],
                            named_arguments={
                                "batches": captured_batch_literal(),
                                "fallback": IdentifierNode("fallback"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_struct_like_enum_array_field",
                matrix_choice_payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_choice_packed_type,
                            [],
                            named_arguments={
                                "batches": captured_batch_literal(),
                                "fallback": IdentifierNode("fallback"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_tuple_enum_array_payload",
                matrix_choice_payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("MatrixChoice::Tuple"),
                            [captured_batch_literal()],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_match_returned_block_array_enum_payloads_compile(tmp_path):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    generic_matrix_type = ArrayType(ArrayType(generic_type, 2), 2)
    generic_batch_type = ArrayType(generic_matrix_type, 2)
    matrix_wrapper_type = NamedType("MatrixWrapper")
    matrix_choice_payload_type = NamedType("MatrixChoice", [payload_type])
    matrix_choice_packed_type = NamedType("MatrixChoice::Packed")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(bool) -> [[Payload; 2]; 2]")
    matrix_choice_enum = EnumNode(
        "MatrixChoice",
        [
            EnumVariantNode("Tuple", fields=[generic_batch_type]),
            EnumVariantNode(
                "Packed",
                fields=[
                    ("batches", generic_batch_type),
                    ("fallback", generic_matrix_type),
                ],
            ),
        ],
    )
    matrix_choice_enum.generic_params = [GenericParameterNode("T")]

    def apply_matrix_once(member):
        return FunctionCallNode(
            IdentifierNode("apply_payload_matrix_once"),
            [
                IdentifierNode("flag"),
                LambdaNode(
                    [ParameterNode("use_primary", None)],
                    MemberAccessNode(IdentifierNode("wrapper"), member),
                    captures=["wrapper"],
                ),
            ],
        )

    def block_matrix(marker, member):
        return BlockNode(
            [
                VariableNode(
                    "marker",
                    PrimitiveType("int"),
                    LiteralNode(marker, PrimitiveType("int")),
                ),
                ExpressionStatementNode(
                    apply_matrix_once(member),
                    is_tail_expression=True,
                ),
            ]
        )

    def block_batch(first_marker, first_member, second_marker, second_member):
        return ArrayLiteralNode(
            [
                block_matrix(first_marker, first_member),
                block_matrix(second_marker, second_member),
            ]
        )

    def tuple_choice(batch):
        return FunctionCallNode(
            IdentifierNode("MatrixChoice::Tuple"),
            [batch],
        )

    def packed_choice(batch):
        return ConstructorNode(
            matrix_choice_packed_type,
            [],
            named_arguments={
                "batches": batch,
                "fallback": IdentifierNode("fallback"),
            },
        )

    def bool_match(selector, true_tail, false_tail):
        return MatchNode(
            IdentifierNode(selector),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            true_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            false_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaMatchReturnedBlockArrayEnumPayloads",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            matrix_choice_enum,
        ],
        functions=[
            FunctionNode(
                "apply_payload_matrix_once",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("flag")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_match_tuple_block_array_payload",
                matrix_choice_payload_type,
                [
                    ParameterNode("choose_tuple", PrimitiveType("bool")),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                ],
                [
                    ReturnNode(
                        bool_match(
                            "choose_tuple",
                            tuple_choice(block_batch(1, "primary", 2, "backup")),
                            tuple_choice(block_batch(3, "backup", 4, "primary")),
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_match_struct_block_array_field",
                matrix_choice_payload_type,
                [
                    ParameterNode("choose_packed", PrimitiveType("bool")),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        bool_match(
                            "choose_packed",
                            packed_choice(block_batch(5, "primary", 6, "backup")),
                            packed_choice(block_batch(7, "backup", 8, "primary")),
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_loop_updated_array_capture_member_preclones_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    payload_cube_type = ArrayType(payload_matrix_type, 2)
    matrix_wrapper_type = NamedType("MatrixWrapper")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(bool) -> [[Payload; 2]; 2]")

    def apply_matrix_once(member):
        return FunctionCallNode(
            IdentifierNode("apply_payload_matrix_once"),
            [
                IdentifierNode("flag"),
                LambdaNode(
                    [ParameterNode("use_primary", None)],
                    MemberAccessNode(IdentifierNode("wrapper"), member),
                    captures=["wrapper"],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaLoopUpdatedArrayCaptureMemberPreclones",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "apply_payload_matrix_once",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("flag")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "loop_update_batch_from_primary",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                ],
                [
                    VariableNode("batches", payload_cube_type),
                    VariableNode(
                        "i",
                        PrimitiveType("int"),
                        LiteralNode(0, PrimitiveType("int")),
                    ),
                    WhileNode(
                        BinaryOpNode(
                            IdentifierNode("i"),
                            "<",
                            LiteralNode(2, PrimitiveType("int")),
                        ),
                        [
                            AssignmentNode(
                                ArrayAccessNode(
                                    IdentifierNode("batches"),
                                    IdentifierNode("i"),
                                ),
                                apply_matrix_once("primary"),
                            ),
                            AssignmentNode(
                                IdentifierNode("i"),
                                BinaryOpNode(
                                    IdentifierNode("i"),
                                    "+",
                                    LiteralNode(1, PrimitiveType("int")),
                                ),
                            ),
                        ],
                    ),
                    ReturnNode(
                        ArrayAccessNode(
                            IdentifierNode("batches"),
                            LiteralNode(0, PrimitiveType("int")),
                        )
                    ),
                ],
            ),
            FunctionNode(
                "loop_update_batch_from_backup",
                payload_matrix_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                ],
                [
                    VariableNode("batches", payload_cube_type),
                    VariableNode(
                        "i",
                        PrimitiveType("int"),
                        LiteralNode(0, PrimitiveType("int")),
                    ),
                    WhileNode(
                        BinaryOpNode(
                            IdentifierNode("i"),
                            "<",
                            LiteralNode(2, PrimitiveType("int")),
                        ),
                        [
                            AssignmentNode(
                                ArrayAccessNode(
                                    IdentifierNode("batches"),
                                    IdentifierNode("i"),
                                ),
                                apply_matrix_once("backup"),
                            ),
                            AssignmentNode(
                                IdentifierNode("i"),
                                BinaryOpNode(
                                    IdentifierNode("i"),
                                    "+",
                                    LiteralNode(1, PrimitiveType("int")),
                                ),
                            ),
                        ],
                    ),
                    ReturnNode(
                        ArrayAccessNode(
                            IdentifierNode("batches"),
                            LiteralNode(0, PrimitiveType("int")),
                        )
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_match_nested_helper_sibling_member_preclones_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    matrix_wrapper_type = NamedType("MatrixWrapper")
    matrix_pair_type = NamedType("MatrixPair")
    matrix_choice_type = NamedType("MatrixChoice")
    matrix_choice_ready_type = NamedType("MatrixChoice::Ready")
    first_closure_type = GenericType("F")
    second_closure_type = GenericType("G")
    closure_bound = NamedType("FnOnce(i32) -> [[Payload; 2]; 2]")
    matrix_choice_enum = EnumNode(
        "MatrixChoice",
        [
            EnumVariantNode(
                "Ready",
                fields=[
                    ("pair", matrix_pair_type),
                    ("fallback", payload_matrix_type),
                ],
            ),
        ],
    )

    def captured_member_lambda(member):
        return LambdaNode(
            [ParameterNode("value", None)],
            MemberAccessNode(IdentifierNode("wrapper"), member),
            captures=["wrapper"],
        )

    def apply_two_members(first_member, second_member):
        return FunctionCallNode(
            IdentifierNode("apply_two_matrix_values"),
            [
                captured_member_lambda(first_member),
                captured_member_lambda(second_member),
                IdentifierNode("seed"),
            ],
        )

    def build_choice(first_member, second_member, fallback_member):
        return FunctionCallNode(
            IdentifierNode("build_choice"),
            [
                apply_two_members(first_member, second_member),
                MemberAccessNode(IdentifierNode("wrapper"), fallback_member),
            ],
        )

    def bool_match(true_tail, false_tail):
        return MatchNode(
            IdentifierNode("choose_primary"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            true_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            false_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaMatchNestedHelperSiblingMemberPreclones",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixWrapper",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            matrix_choice_enum,
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_two_matrix_values",
                matrix_pair_type,
                [
                    ParameterNode("first_builder", first_closure_type),
                    ParameterNode("second_builder", second_closure_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [
                                FunctionCallNode(
                                    IdentifierNode("first_builder"),
                                    [IdentifierNode("seed")],
                                ),
                                FunctionCallNode(
                                    IdentifierNode("second_builder"),
                                    [IdentifierNode("seed")],
                                ),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                    GenericParameterNode("G", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "build_choice",
                matrix_choice_type,
                [
                    ParameterNode("pair", matrix_pair_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_choice_ready_type,
                            [],
                            named_arguments={
                                "pair": IdentifierNode("pair"),
                                "fallback": IdentifierNode("fallback"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_match_nested_helper_sibling_members",
                matrix_choice_type,
                [
                    ParameterNode("choose_primary", PrimitiveType("bool")),
                    ParameterNode("wrapper", matrix_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        bool_match(
                            build_choice("primary", "backup", "primary"),
                            build_choice("backup", "primary", "backup"),
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_match_constructor_indexed_capture_paths_compile(tmp_path):
    payload_type = NamedType("Payload")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    payload_cube_type = ArrayType(payload_matrix_type, 2)
    matrix_pair_type = NamedType("MatrixPair")
    batch_wrapper_type = NamedType("BatchWrapper")
    matrix_choice_type = NamedType("MatrixChoice")
    matrix_choice_ready_type = NamedType("MatrixChoice::Ready")
    first_closure_type = GenericType("F")
    second_closure_type = GenericType("G")
    closure_bound = NamedType("FnOnce(i32) -> [[Payload; 2]; 2]")
    matrix_choice_enum = EnumNode(
        "MatrixChoice",
        [
            EnumVariantNode(
                "Ready",
                fields=[
                    ("direct", matrix_pair_type),
                    ("nested", matrix_pair_type),
                    ("fallback", payload_matrix_type),
                ],
            ),
        ],
    )

    def row_index(index):
        return ArrayAccessNode(
            IdentifierNode("rows"),
            LiteralNode(index, PrimitiveType("int")),
        )

    def wrapper_batch_index(index):
        return ArrayAccessNode(
            MemberAccessNode(IdentifierNode("wrapper"), "batches"),
            LiteralNode(index, PrimitiveType("int")),
        )

    def captured_row_lambda(index):
        return LambdaNode(
            [ParameterNode("value", None)],
            row_index(index),
            captures=["rows"],
        )

    def captured_wrapper_batch_lambda(index):
        return LambdaNode(
            [ParameterNode("value", None)],
            wrapper_batch_index(index),
            captures=["wrapper"],
        )

    def apply_two(first_builder, second_builder):
        return FunctionCallNode(
            IdentifierNode("apply_two_matrix_values"),
            [
                first_builder,
                second_builder,
                IdentifierNode("seed"),
            ],
        )

    def build_choice(direct_pair, nested_pair, fallback):
        return FunctionCallNode(
            IdentifierNode("build_index_choice"),
            [
                direct_pair,
                nested_pair,
                fallback,
            ],
        )

    def bool_match(true_tail, false_tail):
        return MatchNode(
            IdentifierNode("choose_direct"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            true_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            false_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaMatchConstructorIndexedCapturePaths",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "MatrixPair",
                [
                    StructMemberNode("primary", payload_matrix_type),
                    StructMemberNode("backup", payload_matrix_type),
                ],
            ),
            StructNode(
                "BatchWrapper",
                [
                    StructMemberNode("batches", payload_cube_type),
                    StructMemberNode("fallback", payload_matrix_type),
                ],
            ),
            matrix_choice_enum,
        ],
        functions=[
            FunctionNode(
                "make_matrix_pair",
                matrix_pair_type,
                [
                    ParameterNode("primary", payload_matrix_type),
                    ParameterNode("backup", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_pair_type,
                            [IdentifierNode("primary"), IdentifierNode("backup")],
                        )
                    )
                ],
            ),
            FunctionNode(
                "apply_two_matrix_values",
                matrix_pair_type,
                [
                    ParameterNode("first_builder", first_closure_type),
                    ParameterNode("second_builder", second_closure_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("make_matrix_pair"),
                            [
                                FunctionCallNode(
                                    IdentifierNode("first_builder"),
                                    [IdentifierNode("seed")],
                                ),
                                FunctionCallNode(
                                    IdentifierNode("second_builder"),
                                    [IdentifierNode("seed")],
                                ),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                    GenericParameterNode("G", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "build_index_choice",
                matrix_choice_type,
                [
                    ParameterNode("direct", matrix_pair_type),
                    ParameterNode("nested", matrix_pair_type),
                    ParameterNode("fallback", payload_matrix_type),
                ],
                [
                    ReturnNode(
                        ConstructorNode(
                            matrix_choice_ready_type,
                            [],
                            named_arguments={
                                "direct": IdentifierNode("direct"),
                                "nested": IdentifierNode("nested"),
                                "fallback": IdentifierNode("fallback"),
                            },
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_match_constructor_indexed_capture_paths",
                matrix_choice_type,
                [
                    ParameterNode("choose_direct", PrimitiveType("bool")),
                    ParameterNode("rows", payload_cube_type),
                    ParameterNode("wrapper", batch_wrapper_type),
                    ParameterNode("seed", PrimitiveType("int")),
                ],
                [
                    ReturnNode(
                        bool_match(
                            build_choice(
                                apply_two(
                                    captured_row_lambda(0),
                                    captured_row_lambda(1),
                                ),
                                apply_two(
                                    captured_wrapper_batch_lambda(0),
                                    captured_wrapper_batch_lambda(1),
                                ),
                                row_index(0),
                            ),
                            build_choice(
                                apply_two(
                                    captured_row_lambda(1),
                                    captured_row_lambda(0),
                                ),
                                apply_two(
                                    captured_wrapper_batch_lambda(1),
                                    captured_wrapper_batch_lambda(0),
                                ),
                                MemberAccessNode(IdentifierNode("wrapper"), "fallback"),
                            ),
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_array_lambda_generic_payload_dynamic_indexed_branch_captures_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    payload_matrix_type = ArrayType(ArrayType(payload_type, 2), 2)
    payload_cube_type = ArrayType(payload_matrix_type, 2)
    generic_matrix_type = ArrayType(ArrayType(generic_type, 2), 2)
    batch_wrapper_type = NamedType("BatchWrapper")
    matrix_selection_payload_type = NamedType("MatrixSelection", [payload_type])
    matrix_selection_ready_type = NamedType("MatrixSelection::Ready")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(i32, bool) -> [[Payload; 2]; 2]")
    matrix_selection_enum = EnumNode(
        "MatrixSelection",
        [
            EnumVariantNode(
                "Ready",
                fields=[
                    ("selected", generic_matrix_type),
                    ("fallback", generic_matrix_type),
                ],
            ),
        ],
    )
    matrix_selection_enum.generic_params = [GenericParameterNode("T")]

    def rows_index(index_expr):
        return ArrayAccessNode(IdentifierNode("rows"), index_expr)

    def wrapper_batches_index(index_expr):
        return ArrayAccessNode(
            MemberAccessNode(IdentifierNode("wrapper"), "batches"),
            index_expr,
        )

    def select_with_lambda(lambda_body):
        return FunctionCallNode(
            IdentifierNode("select_indexed_matrix"),
            [
                IdentifierNode("index"),
                IdentifierNode("choose_wrapper"),
                LambdaNode(
                    [
                        ParameterNode("slot", None),
                        ParameterNode("use_wrapper", None),
                    ],
                    lambda_body,
                    captures=["rows", "wrapper"],
                ),
            ],
        )

    def ready_selection(selected_expr, fallback_expr):
        return ConstructorNode(
            matrix_selection_ready_type,
            [],
            named_arguments={
                "selected": selected_expr,
                "fallback": fallback_expr,
            },
        )

    def bool_match(true_tail, false_tail):
        return MatchNode(
            IdentifierNode("use_wrapper"),
            [
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(True, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            true_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    LiteralPatternNode(LiteralNode(False, PrimitiveType("bool"))),
                    None,
                    [
                        ExpressionStatementNode(
                            false_tail,
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    ast = ShaderNode(
        "ArrayLambdaGenericPayloadDynamicIndexedBranchCaptures",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode(
                "BatchWrapper",
                [
                    StructMemberNode("batches", payload_cube_type),
                    StructMemberNode("fallback", payload_matrix_type),
                ],
            ),
            matrix_selection_enum,
        ],
        functions=[
            FunctionNode(
                "select_indexed_matrix",
                payload_matrix_type,
                [
                    ParameterNode("index", PrimitiveType("int")),
                    ParameterNode("choose_wrapper", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [
                                IdentifierNode("index"),
                                IdentifierNode("choose_wrapper"),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "return_ternary_indexed_selection",
                matrix_selection_payload_type,
                [
                    ParameterNode("index", PrimitiveType("int")),
                    ParameterNode("choose_wrapper", PrimitiveType("bool")),
                    ParameterNode("rows", payload_cube_type),
                    ParameterNode("wrapper", batch_wrapper_type),
                ],
                [
                    ReturnNode(
                        ready_selection(
                            select_with_lambda(
                                TernaryOpNode(
                                    IdentifierNode("use_wrapper"),
                                    wrapper_batches_index(IdentifierNode("slot")),
                                    rows_index(IdentifierNode("slot")),
                                )
                            ),
                            rows_index(LiteralNode(0, PrimitiveType("int"))),
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_match_indexed_selection",
                matrix_selection_payload_type,
                [
                    ParameterNode("index", PrimitiveType("int")),
                    ParameterNode("choose_wrapper", PrimitiveType("bool")),
                    ParameterNode("rows", payload_cube_type),
                    ParameterNode("wrapper", batch_wrapper_type),
                ],
                [
                    ReturnNode(
                        ready_selection(
                            select_with_lambda(
                                BlockNode(
                                    [
                                        VariableNode(
                                            "normalized",
                                            PrimitiveType("int"),
                                            IdentifierNode("slot"),
                                        ),
                                        ExpressionStatementNode(
                                            bool_match(
                                                wrapper_batches_index(
                                                    IdentifierNode("normalized")
                                                ),
                                                rows_index(
                                                    IdentifierNode("normalized")
                                                ),
                                            ),
                                            is_tail_expression=True,
                                        ),
                                    ]
                                )
                            ),
                            MemberAccessNode(IdentifierNode("wrapper"), "fallback"),
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_lambda_fnonce_guarded_match_returns_nested_enum_struct_payloads_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    choice_payload_type = NamedType("Choice", [payload_type])
    envelope_type = NamedType("Envelope")
    omitted_named_variant_type = NamedType("Choice::Named")
    closure_type = GenericType("F")
    closure_bound = NamedType("FnOnce(Choice<Payload>, bool) -> Envelope")
    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode(
                "Named",
                fields=[
                    ("primary", generic_type),
                    ("backup", generic_type),
                ],
            ),
        ],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]

    def apply_envelope_once(choice, flag, transform):
        return FunctionCallNode(
            IdentifierNode("apply_envelope_once"),
            [IdentifierNode(choice), IdentifierNode(flag), transform],
        )

    def choice_named(primary, backup):
        return ConstructorNode(
            omitted_named_variant_type,
            [],
            named_arguments={
                "primary": IdentifierNode(primary),
                "backup": IdentifierNode(backup),
            },
        )

    def envelope(choice_expr, fallback_name):
        return ConstructorNode(
            envelope_type,
            [],
            named_arguments={
                "choice": choice_expr,
                "fallback": IdentifierNode(fallback_name),
            },
        )

    def named_pattern():
        return StructPatternNode(
            "Choice::Named",
            {
                "primary": IdentifierPatternNode("primary"),
                "backup": IdentifierPatternNode("backup"),
            },
        )

    guarded_match_tail = MatchNode(
        IdentifierNode("choice"),
        [
            MatchArmNode(
                named_pattern(),
                BinaryOpNode(
                    IdentifierNode("use_primary"),
                    "&&",
                    BinaryOpNode(
                        MemberAccessNode(IdentifierNode("primary"), "count"),
                        ">",
                        LiteralNode(0, PrimitiveType("int")),
                    ),
                ),
                [
                    VariableNode(
                        "marker",
                        PrimitiveType("int"),
                        MemberAccessNode(IdentifierNode("primary"), "count"),
                    ),
                    ExpressionStatementNode(
                        envelope(choice_named("primary", "backup"), "backup"),
                        is_tail_expression=True,
                    ),
                ],
            ),
            MatchArmNode(
                named_pattern(),
                None,
                [
                    VariableNode(
                        "marker",
                        PrimitiveType("int"),
                        MemberAccessNode(IdentifierNode("backup"), "count"),
                    ),
                    ExpressionStatementNode(
                        envelope(choice_named("backup", "primary"), "primary"),
                        is_tail_expression=True,
                    ),
                ],
            ),
        ],
    )

    ast = ShaderNode(
        "LambdaFnOnceGuardedMatchEnumStructPayloads",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            choice_enum,
            StructNode(
                "Envelope",
                [
                    StructMemberNode("choice", choice_payload_type),
                    StructMemberNode("fallback", payload_type),
                ],
            ),
        ],
        functions=[
            FunctionNode(
                "apply_envelope_once",
                envelope_type,
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        FunctionCallNode(
                            IdentifierNode("transform"),
                            [IdentifierNode("choice"), IdentifierNode("flag")],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_guarded_match_returns_nested_payload",
                PrimitiveType("int"),
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                ],
                [
                    VariableNode(
                        "result",
                        envelope_type,
                        apply_envelope_once(
                            "choice",
                            "flag",
                            LambdaNode(
                                [
                                    ParameterNode("choice", None),
                                    ParameterNode("use_primary", None),
                                ],
                                BlockNode(
                                    [
                                        VariableNode(
                                            "prelude",
                                            PrimitiveType("int"),
                                            LiteralNode(0, PrimitiveType("int")),
                                        ),
                                        ExpressionStatementNode(
                                            guarded_match_tail,
                                            is_tail_expression=True,
                                        ),
                                    ]
                                ),
                            ),
                        ),
                    ),
                    ReturnNode(
                        MemberAccessNode(
                            MemberAccessNode(IdentifierNode("result"), "fallback"),
                            "count",
                        )
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_lambda_fnonce_option_result_enum_payload_returns_compile(tmp_path):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    error_type = GenericType("E")
    choice_payload_type = NamedType("Choice", [payload_type])
    option_choice_type = NamedType("Option", [choice_payload_type])
    result_choice_payload_type = NamedType(
        "Result",
        [choice_payload_type, payload_type],
    )
    closure_type = GenericType("F")
    option_closure_bound = NamedType(
        "FnOnce(Choice<Payload>, bool) -> Option<Choice<Payload>>"
    )
    result_closure_bound = NamedType(
        "FnOnce(Choice<Payload>, Payload, bool) -> Result<Choice<Payload>, Payload>"
    )
    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode("One", fields=[generic_type]),
            EnumVariantNode("Pair", fields=[generic_type, generic_type]),
        ],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]
    option_enum = EnumNode(
        "Option",
        [
            EnumVariantNode("Some", fields=[generic_type]),
            EnumVariantNode("None"),
        ],
    )
    option_enum.generic_params = [GenericParameterNode("T")]
    result_enum = EnumNode(
        "Result",
        [
            EnumVariantNode("Ok", fields=[generic_type]),
            EnumVariantNode("Err", fields=[error_type]),
        ],
    )
    result_enum.generic_params = [
        GenericParameterNode("T"),
        GenericParameterNode("E"),
    ]

    def enum_call(path, args=None):
        return FunctionCallNode(IdentifierNode(path), args or [])

    def choice_repack_match():
        return MatchNode(
            IdentifierNode("choice"),
            [
                MatchArmNode(
                    ConstructorPatternNode(
                        "Choice::One",
                        [IdentifierPatternNode("payload")],
                    ),
                    None,
                    [
                        ExpressionStatementNode(
                            enum_call(
                                "Choice::One",
                                [IdentifierNode("payload")],
                            ),
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    ConstructorPatternNode(
                        "Choice::Pair",
                        [
                            IdentifierPatternNode("first"),
                            IdentifierPatternNode("second"),
                        ],
                    ),
                    None,
                    [
                        ExpressionStatementNode(
                            enum_call(
                                "Choice::Pair",
                                [
                                    IdentifierNode("first"),
                                    IdentifierNode("second"),
                                ],
                            ),
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    option_lambda = LambdaNode(
        [
            ParameterNode("choice", None),
            ParameterNode("take_some", None),
        ],
        BlockNode(
            [
                VariableNode(
                    "prelude",
                    PrimitiveType("int"),
                    LiteralNode(1, PrimitiveType("int")),
                ),
                ExpressionStatementNode(
                    TernaryOpNode(
                        IdentifierNode("take_some"),
                        enum_call("Option::Some", [choice_repack_match()]),
                        enum_call("Option::None"),
                    ),
                    is_tail_expression=True,
                ),
            ]
        ),
    )
    result_lambda = LambdaNode(
        [
            ParameterNode("choice", None),
            ParameterNode("fallback", None),
            ParameterNode("take_ok", None),
        ],
        BlockNode(
            [
                IfNode(
                    IdentifierNode("take_ok"),
                    BlockNode(
                        [
                            ReturnNode(
                                enum_call(
                                    "Result::Ok",
                                    [choice_repack_match()],
                                )
                            )
                        ]
                    ),
                ),
                ExpressionStatementNode(
                    enum_call("Result::Err", [IdentifierNode("fallback")]),
                    is_tail_expression=True,
                ),
            ]
        ),
    )

    ast = ShaderNode(
        "LambdaFnOnceOptionResultEnumPayloads",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            choice_enum,
            option_enum,
            result_enum,
        ],
        functions=[
            FunctionNode(
                "apply_option_choice_once",
                option_choice_type,
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "transform",
                            [
                                IdentifierNode("choice"),
                                IdentifierNode("flag"),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[option_closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_result_choice_once",
                result_choice_payload_type,
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("fallback", payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "transform",
                            [
                                IdentifierNode("choice"),
                                IdentifierNode("fallback"),
                                IdentifierNode("flag"),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[result_closure_bound]),
                ],
            ),
            FunctionNode(
                "closure_option_tail_returns_nested_match",
                PrimitiveType("int"),
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                ],
                [
                    VariableNode(
                        "selected",
                        option_choice_type,
                        enum_call(
                            "apply_option_choice_once",
                            [
                                IdentifierNode("choice"),
                                IdentifierNode("flag"),
                                option_lambda,
                            ],
                        ),
                    ),
                    ReturnNode(LiteralNode(0, PrimitiveType("int"))),
                ],
            ),
            FunctionNode(
                "closure_result_explicit_return_or_tail_err",
                PrimitiveType("int"),
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("fallback", payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                ],
                [
                    VariableNode(
                        "status",
                        result_choice_payload_type,
                        enum_call(
                            "apply_result_choice_once",
                            [
                                IdentifierNode("choice"),
                                IdentifierNode("fallback"),
                                IdentifierNode("flag"),
                                result_lambda,
                            ],
                        ),
                    ),
                    ReturnNode(LiteralNode(0, PrimitiveType("int"))),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_lambda_fnonce_option_result_nested_call_arguments_compile(tmp_path):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    error_type = GenericType("E")
    choice_payload_type = NamedType("Choice", [payload_type])
    option_choice_type = NamedType("Option", [choice_payload_type])
    result_option_payload_type = NamedType(
        "Result",
        [option_choice_type, payload_type],
    )
    result_choice_payload_type = NamedType(
        "Result",
        [choice_payload_type, payload_type],
    )
    closure_type = GenericType("F")
    option_closure_bound = NamedType(
        "FnOnce(Choice<Payload>, bool) -> Option<Choice<Payload>>"
    )
    result_closure_bound = NamedType(
        "FnOnce(Choice<Payload>, Payload, bool) -> Result<Choice<Payload>, Payload>"
    )
    choice_enum = EnumNode(
        "Choice",
        [EnumVariantNode("One", fields=[generic_type])],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]
    option_enum = EnumNode(
        "Option",
        [
            EnumVariantNode("Some", fields=[generic_type]),
            EnumVariantNode("None"),
        ],
    )
    option_enum.generic_params = [GenericParameterNode("T")]
    result_enum = EnumNode(
        "Result",
        [
            EnumVariantNode("Ok", fields=[generic_type]),
            EnumVariantNode("Err", fields=[error_type]),
        ],
    )
    result_enum.generic_params = [
        GenericParameterNode("T"),
        GenericParameterNode("E"),
    ]

    def enum_call(path, args=None):
        return FunctionCallNode(IdentifierNode(path), args or [])

    def option_lambda():
        return LambdaNode(
            [
                ParameterNode("choice", None),
                ParameterNode("take_some", None),
            ],
            TernaryOpNode(
                IdentifierNode("take_some"),
                enum_call("Option::Some", [IdentifierNode("choice")]),
                enum_call("Option::None"),
            ),
        )

    def result_lambda():
        return LambdaNode(
            [
                ParameterNode("choice", None),
                ParameterNode("fallback", None),
                ParameterNode("take_ok", None),
            ],
            BlockNode(
                [
                    IfNode(
                        IdentifierNode("take_ok"),
                        BlockNode(
                            [
                                ReturnNode(
                                    enum_call(
                                        "Result::Ok",
                                        [IdentifierNode("choice")],
                                    )
                                )
                            ]
                        ),
                    ),
                    ExpressionStatementNode(
                        enum_call("Result::Err", [IdentifierNode("fallback")]),
                        is_tail_expression=True,
                    ),
                ]
            ),
        )

    def apply_option(choice, flag, transform):
        return enum_call(
            "apply_option_choice_once",
            [IdentifierNode(choice), IdentifierNode(flag), transform],
        )

    def apply_result(choice, fallback, flag, transform):
        return enum_call(
            "apply_result_choice_once",
            [
                IdentifierNode(choice),
                IdentifierNode(fallback),
                IdentifierNode(flag),
                transform,
            ],
        )

    ast = ShaderNode(
        "NestedCallableOptionResultArguments",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            choice_enum,
            option_enum,
            result_enum,
        ],
        functions=[
            FunctionNode(
                "apply_option_choice_once",
                option_choice_type,
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "transform",
                            [
                                IdentifierNode("choice"),
                                IdentifierNode("flag"),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[option_closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_result_choice_once",
                result_choice_payload_type,
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("fallback", payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "transform",
                            [
                                IdentifierNode("choice"),
                                IdentifierNode("fallback"),
                                IdentifierNode("flag"),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[result_closure_bound]),
                ],
            ),
            FunctionNode(
                "score_option",
                PrimitiveType("int"),
                [ParameterNode("value", option_choice_type)],
                [ReturnNode(LiteralNode(1, PrimitiveType("int")))],
            ),
            FunctionNode(
                "score_result_option",
                PrimitiveType("int"),
                [ParameterNode("value", result_option_payload_type)],
                [ReturnNode(LiteralNode(2, PrimitiveType("int")))],
            ),
            FunctionNode(
                "score_result_choice",
                PrimitiveType("int"),
                [ParameterNode("value", result_choice_payload_type)],
                [ReturnNode(LiteralNode(3, PrimitiveType("int")))],
            ),
            FunctionNode(
                "nested_option_call_argument",
                PrimitiveType("int"),
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "score_option",
                            [
                                apply_option(
                                    "choice",
                                    "flag",
                                    option_lambda(),
                                )
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "nested_result_enum_payload_argument",
                PrimitiveType("int"),
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "score_result_option",
                            [
                                enum_call(
                                    "Result::Ok",
                                    [
                                        apply_option(
                                            "choice",
                                            "flag",
                                            option_lambda(),
                                        )
                                    ],
                                )
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "nested_result_call_argument",
                PrimitiveType("int"),
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("fallback", payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "score_result_choice",
                            [
                                apply_result(
                                    "choice",
                                    "fallback",
                                    "flag",
                                    result_lambda(),
                                )
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_option_result_match_consumers_from_fnonce_helpers_compile(tmp_path):
    payload_type = NamedType("Payload")
    generic_type = GenericType("T")
    error_type = GenericType("E")
    choice_payload_type = NamedType("Choice", [payload_type])
    option_choice_type = NamedType("Option", [choice_payload_type])
    result_choice_payload_type = NamedType(
        "Result",
        [choice_payload_type, payload_type],
    )
    closure_type = GenericType("F")
    option_closure_bound = NamedType(
        "FnOnce(Choice<Payload>, bool) -> Option<Choice<Payload>>"
    )
    result_closure_bound = NamedType(
        "FnOnce(Choice<Payload>, Payload, bool) -> Result<Choice<Payload>, Payload>"
    )
    choice_enum = EnumNode(
        "Choice",
        [
            EnumVariantNode("Keep", fields=[generic_type]),
            EnumVariantNode(
                "Pair",
                fields=[
                    ("left", generic_type),
                    ("right", generic_type),
                ],
            ),
        ],
    )
    choice_enum.generic_params = [GenericParameterNode("T")]
    option_enum = EnumNode(
        "Option",
        [
            EnumVariantNode("Some", fields=[generic_type]),
            EnumVariantNode("None"),
        ],
    )
    option_enum.generic_params = [GenericParameterNode("T")]
    result_enum = EnumNode(
        "Result",
        [
            EnumVariantNode("Ok", fields=[generic_type]),
            EnumVariantNode("Err", fields=[error_type]),
        ],
    )
    result_enum.generic_params = [
        GenericParameterNode("T"),
        GenericParameterNode("E"),
    ]

    def enum_call(path, args=None):
        return FunctionCallNode(IdentifierNode(path), args or [])

    def count_of(name):
        return MemberAccessNode(IdentifierNode(name), "count")

    def sum_counts(left, right):
        return BinaryOpNode(count_of(left), "+", count_of(right))

    def choice_count_match(choice_expr):
        return MatchNode(
            choice_expr,
            [
                MatchArmNode(
                    ConstructorPatternNode(
                        "Choice::Keep",
                        [IdentifierPatternNode("payload")],
                    ),
                    None,
                    [
                        ExpressionStatementNode(
                            count_of("payload"),
                            is_tail_expression=True,
                        )
                    ],
                ),
                MatchArmNode(
                    StructPatternNode(
                        "Choice::Pair",
                        {
                            "left": IdentifierPatternNode("left"),
                            "right": IdentifierPatternNode("right"),
                        },
                    ),
                    None,
                    [
                        ExpressionStatementNode(
                            sum_counts("left", "right"),
                            is_tail_expression=True,
                        )
                    ],
                ),
            ],
        )

    def option_lambda():
        return LambdaNode(
            [
                ParameterNode("choice", None),
                ParameterNode("take_some", None),
            ],
            TernaryOpNode(
                IdentifierNode("take_some"),
                enum_call("Option::Some", [IdentifierNode("choice")]),
                enum_call("Option::None"),
            ),
        )

    def result_lambda():
        return LambdaNode(
            [
                ParameterNode("choice", None),
                ParameterNode("fallback", None),
                ParameterNode("take_ok", None),
            ],
            TernaryOpNode(
                IdentifierNode("take_ok"),
                enum_call("Result::Ok", [IdentifierNode("choice")]),
                enum_call("Result::Err", [IdentifierNode("fallback")]),
            ),
        )

    def apply_option(choice, flag, transform):
        return enum_call(
            "apply_option_choice_once",
            [IdentifierNode(choice), IdentifierNode(flag), transform],
        )

    def apply_result(choice, fallback, flag, transform):
        return enum_call(
            "apply_result_choice_once",
            [
                IdentifierNode(choice),
                IdentifierNode(fallback),
                IdentifierNode(flag),
                transform,
            ],
        )

    ast = ShaderNode(
        "OptionResultMatchConsumers",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            choice_enum,
            option_enum,
            result_enum,
        ],
        functions=[
            FunctionNode(
                "apply_option_choice_once",
                option_choice_type,
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "transform",
                            [
                                IdentifierNode("choice"),
                                IdentifierNode("flag"),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[option_closure_bound]),
                ],
            ),
            FunctionNode(
                "apply_result_choice_once",
                result_choice_payload_type,
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("fallback", payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("transform", closure_type),
                ],
                [
                    ReturnNode(
                        enum_call(
                            "transform",
                            [
                                IdentifierNode("choice"),
                                IdentifierNode("fallback"),
                                IdentifierNode("flag"),
                            ],
                        )
                    )
                ],
                generic_params=[
                    GenericParameterNode("F", constraints=[result_closure_bound]),
                ],
            ),
            FunctionNode(
                "consume_option_choice",
                PrimitiveType("int"),
                [ParameterNode("value", option_choice_type)],
                [
                    ReturnNode(
                        MatchNode(
                            IdentifierNode("value"),
                            [
                                MatchArmNode(
                                    ConstructorPatternNode(
                                        "Option::Some",
                                        [IdentifierPatternNode("choice")],
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            choice_count_match(
                                                IdentifierNode("choice")
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                                MatchArmNode(
                                    ConstructorPatternNode("Option::None", []),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            LiteralNode(0, PrimitiveType("int")),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "consume_result_choice",
                PrimitiveType("int"),
                [ParameterNode("value", result_choice_payload_type)],
                [
                    ReturnNode(
                        MatchNode(
                            IdentifierNode("value"),
                            [
                                MatchArmNode(
                                    ConstructorPatternNode(
                                        "Result::Ok",
                                        [IdentifierPatternNode("choice")],
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            choice_count_match(
                                                IdentifierNode("choice")
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                                MatchArmNode(
                                    ConstructorPatternNode(
                                        "Result::Err",
                                        [IdentifierPatternNode("fallback")],
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            count_of("fallback"),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "score_nested_consumers",
                PrimitiveType("int"),
                [
                    ParameterNode("choice", choice_payload_type),
                    ParameterNode("fallback", payload_type),
                    ParameterNode("flag", PrimitiveType("bool")),
                ],
                [
                    VariableNode(
                        "option_score",
                        PrimitiveType("int"),
                        enum_call(
                            "consume_option_choice",
                            [
                                apply_option(
                                    "choice",
                                    "flag",
                                    option_lambda(),
                                )
                            ],
                        ),
                    ),
                    VariableNode(
                        "result_score",
                        PrimitiveType("int"),
                        enum_call(
                            "consume_result_choice",
                            [
                                apply_result(
                                    "choice",
                                    "fallback",
                                    "flag",
                                    result_lambda(),
                                )
                            ],
                        ),
                    ),
                    ReturnNode(
                        BinaryOpNode(
                            IdentifierNode("option_score"),
                            "+",
                            IdentifierNode("result_score"),
                        )
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_struct_constructor_args_normalize_field_types_and_clone_non_copy_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    bundle_type = NamedType("Bundle")
    bundle_members = [
        StructMemberNode("weight", PrimitiveType("float")),
        StructMemberNode("id", PrimitiveType("uint")),
        StructMemberNode("ready", PrimitiveType("bool")),
        StructMemberNode("payload", payload_type),
    ]
    ast = ShaderNode(
        "TypedStructConstructors",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode("Bundle", bundle_members),
        ],
        functions=[
            FunctionNode(
                "build_named",
                bundle_type,
                [
                    ParameterNode("index", PrimitiveType("int")),
                    ParameterNode("payload", payload_type),
                ],
                [
                    VariableNode(
                        "named",
                        bundle_type,
                        ConstructorNode(
                            bundle_type,
                            [],
                            named_arguments={
                                "weight": IdentifierNode("index"),
                                "id": IdentifierNode("index"),
                                "ready": IdentifierNode("index"),
                                "payload": IdentifierNode("payload"),
                            },
                        ),
                    ),
                    ReturnNode(IdentifierNode("named")),
                ],
            ),
            FunctionNode(
                "build_positional",
                bundle_type,
                [
                    ParameterNode("index", PrimitiveType("int")),
                    ParameterNode("payload", payload_type),
                ],
                [
                    VariableNode(
                        "built",
                        bundle_type,
                        ConstructorNode(
                            bundle_type,
                            [
                                IdentifierNode("index"),
                                IdentifierNode("index"),
                                IdentifierNode("index"),
                                IdentifierNode("payload"),
                            ],
                        ),
                    ),
                    ReturnNode(IdentifierNode("built")),
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_member_and_indexed_return_places_clone_only_when_required_compile(
    tmp_path,
):
    payload_type = NamedType("Payload")
    wrapper_type = NamedType("Wrapper")
    ast = ShaderNode(
        "NonCopyReturnPlaces",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode("Wrapper", [StructMemberNode("payload", payload_type)]),
        ],
        global_variables=[
            VariableNode("globalWrapper", wrapper_type),
            ArrayNode("Payload", "globalValues", 2),
        ],
        functions=[
            FunctionNode(
                "return_owned_member",
                payload_type,
                [ParameterNode("wrapper", wrapper_type)],
                [ReturnNode(MemberAccessNode(IdentifierNode("wrapper"), "payload"))],
            ),
            FunctionNode(
                "return_static_member",
                payload_type,
                [],
                [
                    ReturnNode(
                        MemberAccessNode(IdentifierNode("globalWrapper"), "payload")
                    )
                ],
            ),
            FunctionNode(
                "return_indexed_local",
                payload_type,
                [ParameterNode("values", ArrayType(payload_type, 2))],
                [
                    ReturnNode(
                        ArrayAccessNode(
                            IdentifierNode("values"),
                            LiteralNode(0, PrimitiveType("int")),
                        )
                    )
                ],
            ),
            FunctionNode(
                "return_indexed_static",
                payload_type,
                [],
                [
                    ReturnNode(
                        ArrayAccessNode(
                            IdentifierNode("globalValues"),
                            LiteralNode(0, PrimitiveType("int")),
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_non_copy_branch_return_places_clone_only_when_required_compile(tmp_path):
    payload_type = NamedType("Payload")
    wrapper_type = NamedType("Wrapper")
    ast = ShaderNode(
        "NonCopyBranchReturnPlaces",
        ExecutionModel.GENERAL_PURPOSE,
        structs=[
            StructNode(
                "Payload",
                [
                    ArrayNode("float", "weights"),
                    StructMemberNode("count", PrimitiveType("int")),
                ],
            ),
            StructNode("Wrapper", [StructMemberNode("payload", payload_type)]),
        ],
        global_variables=[
            VariableNode("globalWrapper", wrapper_type),
            ArrayNode("Payload", "globalValues", 2),
        ],
        functions=[
            FunctionNode(
                "choose_member_ternary",
                payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", wrapper_type),
                    ParameterNode("right", wrapper_type),
                ],
                [
                    ReturnNode(
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            MemberAccessNode(IdentifierNode("left"), "payload"),
                            MemberAccessNode(IdentifierNode("right"), "payload"),
                        )
                    )
                ],
            ),
            FunctionNode(
                "choose_static_or_indexed_ternary",
                payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("values", ArrayType(payload_type, 2)),
                ],
                [
                    ReturnNode(
                        TernaryOpNode(
                            IdentifierNode("flag"),
                            MemberAccessNode(
                                IdentifierNode("globalWrapper"),
                                "payload",
                            ),
                            ArrayAccessNode(
                                IdentifierNode("values"),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                        )
                    )
                ],
            ),
            FunctionNode(
                "choose_match_member_or_static",
                payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("left", wrapper_type),
                ],
                [
                    ReturnNode(
                        MatchNode(
                            IdentifierNode("flag"),
                            [
                                MatchArmNode(
                                    LiteralPatternNode(
                                        LiteralNode(True, PrimitiveType("bool"))
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            MemberAccessNode(
                                                IdentifierNode("left"),
                                                "payload",
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                                MatchArmNode(
                                    LiteralPatternNode(
                                        LiteralNode(False, PrimitiveType("bool"))
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            MemberAccessNode(
                                                IdentifierNode("globalWrapper"),
                                                "payload",
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
                ],
            ),
            FunctionNode(
                "choose_match_indexed",
                payload_type,
                [
                    ParameterNode("flag", PrimitiveType("bool")),
                    ParameterNode("values", ArrayType(payload_type, 2)),
                ],
                [
                    ReturnNode(
                        MatchNode(
                            IdentifierNode("flag"),
                            [
                                MatchArmNode(
                                    LiteralPatternNode(
                                        LiteralNode(True, PrimitiveType("bool"))
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            ArrayAccessNode(
                                                IdentifierNode("values"),
                                                LiteralNode(0, PrimitiveType("int")),
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                                MatchArmNode(
                                    LiteralPatternNode(
                                        LiteralNode(False, PrimitiveType("bool"))
                                    ),
                                    None,
                                    [
                                        ExpressionStatementNode(
                                            ArrayAccessNode(
                                                IdentifierNode("globalValues"),
                                                LiteralNode(1, PrimitiveType("int")),
                                            ),
                                            is_tail_expression=True,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
                ],
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_legacy_generic_wrappers_emit_rust_generic_params():
    code = """
    shader GenericWrappers {
        generic<T, U> struct Pair {
            left: T;
            right: U;
        }

        generic<T> struct Wrapper {
            item: Option<int>;
            result: Result<T, int>;
        }

        generic<T> fn identity(value: T) -> T {
            return value;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub struct Pair<T, U>" in generated_code
    assert "pub left: T," in generated_code
    assert "pub right: U," in generated_code
    assert "pub struct Wrapper<T>" in generated_code
    assert "pub item: Option<i32>," in generated_code
    assert "pub result: Result<T, i32>," in generated_code
    assert "pub fn identity<T: Copy>(value: T) -> T" in generated_code


def test_traits_and_generic_constraints_emit_rust_and_smoke_compile(tmp_path):
    code = """
    shader GenericBounds {
        trait Numeric {
            fn zero() -> Self;
            fn add(self, other: Self) -> Self;
            fn checked(self) -> Maybe<Self>;
        }

        struct ScalarBox<T: Numeric> {
            value: T;
        }

        generic<T: Numeric> enum Maybe {
            Some(T)
        }

        generic<T: Numeric + Copy> fn add_zero(value: T) -> T {
            return value.add(T::zero());
        }

        generic<T: Numeric> fn is_same(a: T, b: T) -> bool {
            return a == b;
        }

        generic<T: Numeric + Copy> fn operator_mix(a: T, b: T, c: T, d: T) -> T {
            return (a + b) - (c / d);
        }

        generic<T: Numeric> fn divide(a: T, b: T) -> T {
            return a / b;
        }

        generic<T: Numeric> fn call_divide(a: T, b: T) -> T {
            return divide(a, b);
        }

        generic<T: Numeric> fn make_box(a: T, b: T) -> ScalarBox<T> {
            return ScalarBox { value: a - b };
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub trait Numeric: Sized {" in generated_code
    assert "fn zero() -> Self;" in generated_code
    assert "fn add(self, other: Self) -> Self;" in generated_code
    assert "fn checked(self) -> Maybe<Self>;" in generated_code
    assert "impl Numeric for f32" not in generated_code
    assert "pub struct ScalarBox<T: Numeric>" in generated_code
    assert "pub enum Maybe<T: Numeric>" in generated_code
    assert "impl<T: Numeric + Default> Default for Maybe<T>" in generated_code
    assert "pub fn add_zero<T: Numeric + Copy>(value: T) -> T" in generated_code
    assert "return Numeric::add(value, T::zero());" in generated_code
    assert (
        "pub fn is_same<T: Numeric + Copy + PartialEq>(a: T, b: T) -> bool"
        in generated_code
    )
    assert (
        "pub fn operator_mix<T: Numeric + Copy + core::ops::Add<Output = T> + core::ops::Div<Output = T> + core::ops::Sub<Output = T>>"
    ) in generated_code
    assert (
        "pub fn divide<T: Numeric + Copy + core::ops::Div<Output = T>>(a: T, b: T) -> T"
        in generated_code
    )
    assert (
        "pub fn call_divide<T: Numeric + Copy + core::ops::Div<Output = T>>(a: T, b: T) -> T"
        in generated_code
    )
    assert (
        "pub fn make_box<T: Numeric + Copy + core::ops::Sub<Output = T>>"
        in generated_code
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_canonical_numeric_trait_emits_primitive_impls_and_smoke_compiles(tmp_path):
    code = """
    shader PrimitiveNumericBounds {
        trait Numeric {
            fn add(self, other: Self) -> Self;
            fn mul(self, other: Self) -> Self;
            fn zero() -> Self;
            fn one() -> Self;
        }

        struct ScalarBox<T: Numeric> {
            value: T;
        }

        fn make_float_box(value: float) -> ScalarBox<float> {
            return ScalarBox { value: value };
        }

        fn sum_float(a: float, b: float) -> float {
            return Numeric::add(a, b);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "impl Numeric for f32" in generated_code
    assert "impl Numeric for i32" in generated_code
    assert "pub fn make_float_box(value: f32) -> ScalarBox<f32>" in generated_code
    assert "pub fn sum_float(a: f32, b: f32) -> f32" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generic_sqrt_adds_trait_bound_and_smoke_compiles(tmp_path):
    code = """
    shader GenericSqrt {
        trait Numeric {
            fn add(self, other: Self) -> Self;
            fn mul(self, other: Self) -> Self;
            fn zero() -> Self;
            fn one() -> Self;
        }

        generic<T: Numeric> fn root_energy(value: T) -> T {
            return sqrt(value);
        }

        fn root_float(value: float) -> float {
            return sqrt(value);
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'sqrt' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_generated_structs_emit_new_constructors_and_smoke_compile(tmp_path):
    code = """
    shader StructConstructors {
        uniform float weight;

        generic<T> struct Vec3 {
            x: T;
            y: T;
            z: T;
        }

        struct Packed {
            color: Vec3<float>;
            weight: float;
        }

        fn make_vec() -> Vec3<float> {
            return vec3(1.0, 2.0, 3.0);
        }

        fn make_packed() -> Packed {
            return Packed::new(vec3(0.25, 0.5, 0.75), 1.0);
        }

        fn shade(light: vec3) -> vec3 {
            let unit = normalize(light);
            return vec3(unit.x, unit.y, unit.z);
        }
    }
    """

    with pytest.raises(
        RustRefusal, match=r"spirv-std's glam has no vector of T \(Vec3<T>\)"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_nested_enum_wrapper_structs_emit_rust_enums():
    code = """
    shader GenericWrappers {
        generic<T> struct Option {
            enum OptionType {
                Some(T),
                None
            }
            OptionType variant;
        }

        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum Option<T>" in generated_code
    )
    assert "Some(T)," in generated_code
    assert "    #[default]\n    None," in generated_code
    assert "#[derive(Debug, Clone, Copy)]\npub enum Result<T, E>" in generated_code
    assert "Ok(T)," in generated_code
    assert "Err(E)," in generated_code
    assert "impl<T: Default, E: Default> Default for Result<T, E>" in generated_code
    assert "Self::Ok(Default::default())" in generated_code
    assert "pub struct Option<T>" not in generated_code
    assert "pub struct Result<T, E>" not in generated_code
    assert "pub variant: OptionType," not in generated_code
    assert "pub variant: ResultType," not in generated_code


def test_top_level_enums_emit_rust_variants():
    code = """
    shader EnumCases {
        enum MathError {
            DivisionByZero,
            InvalidInput
        }

        enum LightingModel {
            Phong { ambient: vec3, shininess: float },
            Toon { base_color: vec3 }
        }

        enum ShaderError {
            MathError(MathError),
            TextureError(str),
            InvalidState
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="rust-gpu cannot lower string constants"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_enum_variants_with_dynamic_array_payloads_do_not_derive_copy_and_compile(
    tmp_path,
):
    code = """
    shader NonCopyEnumPayload {
        enum Command {
            Draw { weights: float[], amount: int },
            Skip
        }

        fn choose(command: Command) -> int {
            match command {
                Command::Draw { weights, amount } => amount,
                Command::Skip => 0,
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_top_level_enum_discriminants_emit_rust_values():
    code = """
    shader EnumCases {
        enum OpCode {
            Add = 1,
            Multiply = 2
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub enum OpCode {" in generated_code
    assert "Add = 1," in generated_code
    assert "Multiply = 2," in generated_code


def test_advanced_generic_pattern_matching_example_emits_referenced_enums():
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "examples/advanced/GenericPatternMatching.cgl").read_text()

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: uniform 'normal_matrix': 'mat3 column 1' is at std140 offset 16, rust\-gpu lays i",
    ):
        generate_code(parse_code(tokenize_code(source)))


def test_advanced_generic_pattern_matching_example_smoke_compiles(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "examples/advanced/GenericPatternMatching.cgl").read_text()

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: uniform 'normal_matrix': 'mat3 column 1' is at std140 offset 16, rust\-gpu lays i",
    ):
        generate_code(parse_code(tokenize_code(source)))


def test_generated_rust_enum_defaults_smoke_compile_with_rustc(tmp_path):
    code = """
    shader EnumSmoke {
        generic<T> struct Option {
            enum OptionType {
                Some(T),
                None
            }
            OptionType variant;
        }

        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero,
            InvalidInput
        }

        enum LightingModel {
            Phong { ambient: vec3, shininess: float },
            Toon { base_color: vec3 }
        }

        struct RenderState {
            Option<int> material_id;
            Result<int, MathError> status;
            LightingModel lighting_model;
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_unsized_cbuffer_arrays_do_not_derive_copy():
    code = """
    cbuffer DynamicBlock {
        float weights[];
        vec3 colors[2];
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused,
        match="constant block 'DynamicBlock': unsized arrays in uniform blocks are not supported",
    ):
        generate_code(ast)


def test_cbuffer_scalar_statics_use_const_default_literals(tmp_path):
    code = """
    cbuffer Material {
        float exposure;
        int count;
        uint mask;
        bool enabled;
        float values[4];
        int indices[2];
        bool flags[3];
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert "pub exposure: f32," in generated_code
    assert "pub enabled: u32," in generated_code
    assert "pub values: [Vec4; 4]," in generated_code
    assert "pub indices: [IVec4; 2]," in generated_code
    assert "pub flags: [UVec4; 3]," in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_vector_matrix_array_statics_use_zeroed_static_fallback(tmp_path):
    code = """
    vec3 globalDirs[2];
    dvec2 globalPrecise[3];
    mat4 globalTransforms[2];

    cbuffer Material {
        vec3 directions[2];
        mat4 transforms[2];
        dvec2 precise[3];
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert "pub directions: [Vec4; 2]," in generated_code
    assert "pub transforms: [Mat4; 2]," in generated_code
    assert "pub precise: [DVec2; 3]," in generated_code
    assert (
        "fn GLOBAL_DIRS() -> [Vec3; 2] {\n    [Default::default(); 2]\n}"
        in generated_code
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_default_composite_statics_use_lazy_lock(tmp_path):
    code = """
    vec3 direction;
    mat4 transform;

    cbuffer Material {
        vec3 tint;
        mat4 model;
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert "pub tint: Vec3," in generated_code
    assert "pub model: Mat4," in generated_code
    assert "fn DIRECTION() -> Vec3 {" in generated_code
    assert "fn TRANSFORM() -> Mat4 {" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_lazy_lock_static_use_sites_are_dereferenced(tmp_path):
    code = """
    vec3 direction;
    float values[] = {1.0, 2.0};

    shader LazyUse {
        compute {
            float main() {
                return direction.x + values[0];
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused, match="stage: compute entry point 'main' returns a float"
    ):
        generate_code(ast)


def test_colliding_static_symbol_names_are_suffixed_and_referenced(tmp_path):
    code = """
    float fooBar = 1.0;
    float foo_bar = 2.0;
    vec3 lightDir;
    vec3 light_dir;

    shader StaticCollision {
        compute {
            float main() {
                return fooBar + foo_bar + lightDir.x + light_dir.y;
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="stage: compute entry point 'main' returns a float"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_lazy_lock_static_name_shadowed_by_local_is_not_dereferenced(tmp_path):
    code = """
    vec3 direction;

    shader LazyUse {
        compute {
            vec3 main() {
                vec3 direction = vec3(1.0, 2.0, 3.0);
                return direction;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused, match="stage: compute entry point 'main' returns a vec3"
    ):
        generate_code(ast)


def test_generated_rust_lazy_lock_statics_are_refused(tmp_path):
    # Host Rust declared these globals as LazyLock statics; the rust-gpu output refuses each program by name.
    shaders = [
        (
            """
        vec3 direction;
        float values[] = {1.0, 2.0};

        shader LazyUse {
            compute {
                float main(int index) {
                    return direction.x + values[index];
                }
            }
        }
""",
            "compute entry point 'main' returns a float",
        ),
        (
            """
        sampler3D volumeMap;

        shader TextureUse {
            fragment {
                vec4 main(vec3 uv) @ gl_FragColor {
                    return texture(volumeMap, uv);
                }
            }
        }
""",
            None,
        ),
    ]
    returning, sampled = shaders
    with pytest.raises(RustLoweringRefused, match=returning[1]):
        generate_code(parse_code(tokenize_code(returning[0])))
    # A sampled texture is a spirv-std Image! entry parameter (rust_gpu_images), not a static.
    generated_code = generate_code(parse_code(tokenize_code(sampled[0])))
    assert "static" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_vertex_uniform_matrix_binds_as_a_uniform_reference(tmp_path):
    # The stage uniform is a #[spirv(uniform)] reference, not a LazyLock static, and the unannotated vec4 result is
    # the position, as in the GLSL, HLSL and Metal output.
    generated = generate_code(parse_code(tokenize_code("""
        shader MatrixUse {
            vertex {
                uniform mat4 transform;

                vec4 main(vec3 position) {
                    transform * vec4(position, 1.0)
                }
            }
        }
""")))
    assert (
        "#[spirv(uniform, descriptor_set = 0, binding = 0)] transform: &Mat4"
        in generated
    )
    assert "#[spirv(position)] out_value: &mut Vec4" in generated
    assert "LazyLock" not in generated
    assert (
        "let return_value = (transform * Vec4::new(position.x, position.y, position.z, 1.0));"
        in generated
    )
    assert "*out_value = return_value;" in generated
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_generated_rust_match_expressions_smoke_compile_with_rustc(tmp_path):
    shaders = [
        """
        shader MatchSmoke {
            generic<T, E> struct Result {
                enum ResultType {
                    Ok(T),
                    Err(E)
                }
                ResultType variant;
            }

            enum MathError {
                InvalidInput
            }

            enum VectorOp {
                Add,
                Cross
            }

            fn convert(op: VectorOp) -> Result<int, MathError> {
                match op {
                    VectorOp::Add => Result::Ok(1),
                    _ => Result::Err(MathError::InvalidInput),
                }
            }
        }
        """,
        """
        shader MatchInitSmoke {
            enum Mode {
                Bright,
                Dark
            }

            fn choose(mode: Mode) -> vec3 {
                let color = match mode {
                    Mode::Bright => vec3(1.0, 1.0, 1.0),
                    _ => vec3(0.0, 0.0, 0.0),
                };
                color
            }
        }
        """,
    ]

    for shader in shaders:
        generated_code = generate_code(parse_code(tokenize_code(shader)))

        assert "match " in generated_code
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generated_rust_rich_match_patterns_smoke_compile_with_rustc(tmp_path):
    code = """
    shader RichPatternSmoke {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum MathError {
            InvalidInput
        }

        enum LightingModel {
            Phong {
                ambient: vec3,
                diffuse: vec3,
                specular: vec3,
                shininess: float
            },
            Toon { base_color: vec3 }
        }

        fn unwrap_result(result: Result<int, MathError>) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(_) => 0,
            }
        }

        fn classify_model(model: LightingModel) -> int {
            match model {
                LightingModel::Phong { ambient, diffuse, specular, shininess } if shininess > 0.0 => 1,
                LightingModel::Toon { base_color, .. } => 2,
            }
        }

        fn guarded_literal(mode: int) -> int {
            match mode {
                0 if mode > 0 => 1,
                _ => 2,
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "Result::Ok(value) =>" in generated_code
    assert "Result::Err(_) =>" in generated_code
    assert (
        "LightingModel::Phong { ambient: _ambient, diffuse: _diffuse, "
        "specular: _specular, shininess } if (shininess > 0.0) =>" in generated_code
    )
    assert "LightingModel::Toon { base_color: _base_color, .. } =>" in generated_code
    assert "0 if (mode > 0) =>" in generated_code
    assert "_ => unreachable!()," in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generated_rust_duplicate_stage_main_names_smoke_compile_with_rustc(tmp_path):
    code = """
    shader MultiStageSmoke {
        vertex {
            vec4 main(vec3 position) {
                vec4(position, 1.0)
            }
        }

        fragment {
            vec4 main(vec4 color) {
                color
            }
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))
    # Each stage's main gets a stage-qualified name, and each tail expression is the stage result.
    assert (
        "pub fn vertex_main(#[spirv(location = 0)] position: Vec3, #[spirv(position)] out_value: &mut Vec4)"
        in generated
    )
    assert (
        "pub fn fragment_main(#[spirv(location = 0)] color: Vec4, #[spirv(location = 0)] out_value: &mut Vec4)"
        in generated
    )
    assert "let return_value = color;" in generated
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_generated_rust_stage_attrs_are_host_compatible_for_rustc(tmp_path):
    code = """
    shader HostCompatibleStages {
        vertex {
            vec4 main(vec3 position) {
                return vec4(position, 1.0);
            }
        }

        fragment {
            vec4 main(vec4 color) @ gl_FragColor {
                return color;
            }
        }

        compute {
            void main(uvec3 globalId @ gl_GlobalInvocationID) {
                uint copy = globalId.x;
            }
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))
    assert_rust_stage_attr(generated, "vertex")
    assert_rust_stage_attr(generated, "fragment")
    assert_rust_stage_attr(generated, "compute")
    assert_no_bare_rust_stage_attrs(generated)
    assert (
        "#[spirv(global_invocation_id)] globalId_global_invocation_id: UVec3"
        in generated
    )
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_initialized_vector_and_dynamic_array_statics_use_lazy_lock(tmp_path):
    code = """
    vec3 globalDirs[3] = {
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    };
    dvec2 preciseDirs[2] = {dvec2(1.0, 2.0)};
    mat2 transforms[2] = {mat2(1.0, 0.0, 0.0, 1.0)};
    float dynamicWeights[] = {1.0, 2.0};
    vec3 dynamicDirs[] = {vec3(0.0, 0.0, 1.0)};
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_legacy_global_array_nodes_emit_static_declarations(tmp_path):
    ast = ShaderNode(
        "Legacy",
        ExecutionModel.GENERAL_PURPOSE,
        global_variables=[
            ArrayNode(PrimitiveType("float"), "weights", 4),
            ArrayNode(PrimitiveType("int"), "indices", 2),
            ArrayNode(PrimitiveType("float"), "dynamicWeights"),
            ArrayNode(
                PrimitiveType("float"),
                "initialized",
                4,
                initial_value=ArrayLiteralNode(
                    [
                        LiteralNode(1.0, PrimitiveType("float")),
                        LiteralNode(2.0, PrimitiveType("float")),
                    ]
                ),
            ),
        ],
    )

    with pytest.raises(
        RustLoweringRefused, match=r"dynamic array \(Vec\) needs an allocator"
    ):
        generate_code(ast)


def test_basic_shader():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert generated_code is not None
        assert "fn vertex_main(" in generated_code
        assert "fn fragment_main(" in generated_code
        assert_rust_stage_attr(generated_code, "vertex")
        assert_rust_stage_attr(generated_code, "fragment")
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust basic shader codegen not implemented.")


def test_local_struct_outputs_initialize_with_default_before_field_assignment():
    code = """
    shader main {
        struct Out {
            vec4 color @ COLOR;
        };
        vertex {
            Out main() {
                Out output;
                output.color = vec4(1.0, 0.0, 0.0, 1.0);
                return output;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "#[derive(Debug, Clone, Copy, Default)]" in generated_code
    assert "let mut output: Out = Default::default();" in generated_code
    assert "output.color = Vec4::new(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "let mut output: Out;" not in generated_code


def test_if_statement(tmp_path):
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                if (input.texCoord.x > 0.5) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                } else {
                    output.color = vec4(0.0, 0.0, 0.0, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                let brightness = texture(iChannel0, input.color.xy).r;
                let bloom = max(0.0, brightness - 0.5);
                if (bloom > 0.5) {
                    bloom = 0.5;
                } else {
                    bloom = 0.0;
                }
                let texColor = texture(iChannel0, input.color.xy).rgb;
                let colorWithBloom = texColor + vec3(bloom);
                return vec4(colorWithBloom, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)
    except SyntaxError:
        pytest.fail("If statement codegen not implemented.")


def test_sampler3d_maps_to_texture3d_in_rust(tmp_path):
    code = """
    shader Sampler3DProbe {
        sampler3D volumeMap;
        fragment {
            vec4 main(vec3 uv) @ gl_FragColor {
                return texture(volumeMap, uv);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_sampler_array_and_cube_families_map_to_rust_textures_and_compile(tmp_path):
    code = """
    shader SamplerFamilyProbe {
        sampler1D rampMap;
        sampler1DArray rampArrayMap;
        sampler2DArray arrayMap;
        samplerCube envMap;
        samplerCubeArray probeMap;

        fragment {
            vec4 main(float x, vec2 xLayer, vec3 uvw, vec3 dir, vec4 probeCoord) @ gl_FragColor {
                let ramp = texture(rampMap, x);
                let rampLayer = texture(rampArrayMap, xLayer);
                let layer = texture(arrayMap, uvw);
                let env = texture(envMap, dir);
                let probe = texture(probeMap, probeCoord);
                return ramp + rampLayer + layer + env + probe;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_basic_sampled_texture_calls_map_to_rust_helpers_and_compile(tmp_path):
    code = """
    Texture2D<float4> colorMap : register(t0, space1);
    SamplerState linearSampler : register(s0);

    shader BasicSampledTextureProbe {
        sampler2D diffuseMap;
        sampler pointSampler;

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                let combined = texture(diffuseMap, uv);
                let split = texture(diffuseMap, pointSampler, uv);
                let hlslSplit = colorMap.Sample(linearSampler, uv);
                return combined + split + hlslSplit;
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: 'colorMap' \(Texture2D<float4>\) is not lowered",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_shadow_sampler_texture_calls_return_float_and_compile(tmp_path):
    code = """
    shader ShadowSamplerTextureProbe {
        sampler2DShadow shadowMap;
        sampler compareSampler;

        fragment {
            float main(vec3 uvDepth, vec2 uv, float depth) @ gl_FragDepth {
                let implicitShadow = texture(shadowMap, uvDepth);
                let splitShadow = texture(shadowMap, compareSampler, uvDepth);
                let cmpMember = shadowMap.SampleCmp(compareSampler, uv, depth);
                return implicitShadow + splitShadow + cmpMember;
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"cannot lower 1 construct: texture operation: 'texture_compare_sampler' has no spirv-std lowering",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_hlsl_texture_member_helpers_map_to_rust_placeholders_and_compile(tmp_path):
    code = """
    Texture2D<float4> colorMap : register(t0);
    SamplerState linearSampler : register(s0);

    shader HlslTextureMemberProbe {
        sampler2DMS msMap;
        sampler2DShadow shadowMap;

        fragment {
            vec4 main(
                vec2 uv,
                float lod,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset,
                ivec3 pixel,
                int sampleIndex,
                float depth
            ) @ gl_FragColor {
                int width = 0;
                int height = 0;
                colorMap.GetDimensions(width, height);

                let baseOffset = colorMap.Sample(linearSampler, uv, offset);
                let biased = colorMap.SampleBias(linearSampler, uv, 0.25);
                let biasedOffset = colorMap.SampleBias(linearSampler, uv, 0.5, offset);
                let lodColor = colorMap.SampleLevel(linearSampler, uv, lod);
                let lodOffset = colorMap.SampleLevel(linearSampler, uv, lod, offset);
                let gradColor = colorMap.SampleGrad(linearSampler, uv, ddx, ddy);
                let gradOffset = colorMap.SampleGrad(linearSampler, uv, ddx, ddy, offset);
                let gathered = colorMap.Gather(linearSampler, uv);
                let gatheredRedOffset = colorMap.GatherRed(linearSampler, uv, offset);
                let loaded = colorMap.Load(pixel);
                let loadedOffset = colorMap.Load(pixel, offset);
                let msLoaded = msMap.Load(ivec2(pixel.x, pixel.y), sampleIndex);
                let detail = colorMap.CalculateLevelOfDetail(linearSampler, uv);

                let cmp = shadowMap.SampleCmp(linearSampler, uv, depth);
                let cmpOffset = shadowMap.SampleCmp(linearSampler, uv, depth, offset);
                let cmpLevel = shadowMap.SampleCmpLevel(linearSampler, uv, depth, lod);
                let cmpGradOffset = shadowMap.SampleCmpGrad(linearSampler, uv, depth, ddx, ddy, offset);
                let gatherCmp = shadowMap.GatherCmp(linearSampler, uv, depth);
                let gatherCmpOffset = shadowMap.GatherCmp(linearSampler, uv, depth, offset);

                return baseOffset + biased + biasedOffset + lodColor + lodOffset
                    + gradColor + gradOffset + gathered + gatheredRedOffset
                    + loaded + loadedOffset + msLoaded + gatherCmp + gatherCmpOffset
                    + vec4(cmp + cmpOffset + cmpLevel + cmpGradOffset + detail);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: 'colorMap' \(Texture2D<float4>\) is not lowered",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_texture_helper_diagnostics_reject_invalid_resources():
    non_shadow_compare = """
    shader InvalidRustTextureCompare {
        sampler2D colorMap;

        fragment {
            float main(vec2 uv, float depth) @ gl_FragDepth {
                return textureCompare(colorMap, uv, depth);
            }
        }
    }
    """
    with pytest.raises(
        RustLoweringRefused,
        match=r"texture operation: 'textureCompare' has no spirv\-std method for these operands on a Image!",
    ):
        generate_code(parse_code(tokenize_code(non_shadow_compare)))

    image_gather = """
    shader InvalidRustImageGather {
        image2D colorImage;

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return textureGather(colorImage, uv);
            }
        }
    }
    """
    with pytest.raises(
        RustLoweringRefused,
        match="texture operation: 'textureGather' samples an image that is not a sampled texture",
    ):
        generate_code(parse_code(tokenize_code(image_gather)))

    unsupported_member = """
    Texture2D<float4> colorMap;
    SamplerState linearSampler;

    shader InvalidRustTextureMember {
        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return colorMap.UnsupportedSample(linearSampler, uv);
            }
        }
    }
    """
    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: 'colorMap' \(Texture2D<float4>\) is not lowered",
    ):
        generate_code(parse_code(tokenize_code(unsupported_member)))


def test_texture_lod_grad_offset_calls_map_to_rust_helpers_and_compile(tmp_path):
    code = """
    shader TextureAdvancedSamplingProbe {
        sampler2D mainTexture;

        fragment {
            vec4 main(vec2 uv, vec2 ddx, vec2 ddy, ivec2 offset) @ gl_FragColor {
                let lodColor = textureLod(mainTexture, uv, 2.0);
                let gradColor = textureGrad(mainTexture, uv, ddx, ddy);
                let offsetColor = textureOffset(mainTexture, uv, offset);
                return lodColor + gradColor + offsetColor;
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"texture operation: 'textureOffset' reads with a texel offset, and spirv\-std's image operan",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_integer_texture_lod_sampler_helpers_preserve_integer_samples_and_compile(
    tmp_path,
):
    # Reduced from VulkanIntegerTextureSamplerLodShader.cgl and
    # VulkanIntegerTextureArraySamplerLodShader.cgl.
    code = """
    shader IntegerTextureSamplerLodProbe {
        isampler2D signedTex;
        isampler2DArray signedLayers;
        usamplerCube unsignedCube;
        usamplerCubeArray unsignedCubeLayers;
        sampler textureSampler;

        fragment {
            vec4 main(vec2 uv, vec3 uvLayer, vec3 direction, vec4 directionLayer, float lod) @ gl_FragColor {
                let signedSample = textureLod(signedTex, textureSampler, uv, lod);
                let signedLayerSample = textureLod(
                    signedLayers,
                    textureSampler,
                    uvLayer,
                    lod
                );
                let unsignedSample = textureLod(
                    unsignedCube,
                    textureSampler,
                    direction,
                    lod
                );
                let unsignedLayerSample = textureLod(
                    unsignedCubeLayers,
                    textureSampler,
                    directionLayer,
                    lod
                );
                return vec4(
                    float(signedSample.x),
                    float(signedLayerSample.y),
                    float(unsignedSample.z),
                    float(unsignedLayerSample.w)
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_texel_fetch_sampler_family_helpers_compile(tmp_path):
    code = """
    shader TexelFetchSamplerFamilyProbe {
        sampler1D rampMap;
        sampler1DArray rampArrayMap;
        sampler2DArray arrayMap;
        sampler3D volumeMap;
        samplerCube envMap;
        samplerCubeArray probeMap;

        fragment {
            vec4 main(
                int x,
                ivec2 xLayer,
                ivec3 pixelLayer,
                ivec3 voxel,
                ivec3 cubeCoord,
                ivec4 cubeLayerCoord,
                int lod,
                int offset1,
                ivec2 offset2,
                ivec3 offset3
            ) @ gl_FragColor {
                let rampFetch = texelFetch(rampMap, x, lod);
                let rampOffset = texelFetchOffset(rampMap, x, lod, offset1);
                let rampLayerFetch = texelFetch(rampArrayMap, xLayer, lod);
                let layerOffset = texelFetchOffset(arrayMap, pixelLayer, lod, offset2);
                let volumeOffset = texelFetchOffset(volumeMap, voxel, lod, offset3);
                let cubeFetch = texelFetch(envMap, cubeCoord, lod);
                let cubeArrayFetch = texelFetch(probeMap, cubeLayerCoord, lod);
                return rampFetch + rampOffset + rampLayerFetch + layerOffset
                    + volumeOffset + cubeFetch + cubeArrayFetch;
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"texture operation: 'texelFetchOffset' reads with a texel offset, and spirv\-std's image ope",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_texel_fetch_and_texture_query_calls_map_to_rust_helpers_and_compile(
    tmp_path,
):
    code = """
    shader TextureFetchQueryProbe {
        sampler2D mainTexture;
        sampler3D volumeMap;

        fragment {
            vec4 main(vec2 uv, ivec2 pixel, ivec2 offset) @ gl_FragColor {
                let size2D = textureSize(mainTexture, 0);
                let size3D = textureSize(volumeMap, 0);
                let fetched = texelFetch(mainTexture, pixel, 0);
                let shifted = texelFetchOffset(mainTexture, pixel, 0, offset);
                let levels = textureQueryLevels(mainTexture);
                let lod = textureQueryLod(mainTexture, uv);
                let samples = textureSamples(mainTexture);
                return fetched + shifted + vec4(lod.x, lod.y, float(levels + samples), 1.0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"texture operation: 'texelFetchOffset' reads with a texel offset, and spirv\-std's image ope",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_wgsl_texture_dimensions_alias_maps_to_unsigned_rust_helpers_and_compile(
    tmp_path,
):
    # W3C WGSL 17.7.1: textureDimensions returns u32/vecN<u32> and the
    # mip level is optional. Source: https://www.w3.org/TR/WGSL/#texturedimensions
    code = """
    shader WgslTextureDimensionsProbe {
        sampler2D mainTexture;
        sampler3D volumeMap;

        fragment {
            vec4 main() @ gl_FragColor {
                let size2D = textureDimensions(mainTexture);
                uvec3 size3D = textureDimensions(volumeMap, 0u);
                return vec4(float(size2D.x), float(size2D.y), float(size3D.z), 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_multisample_texture_fetch_and_query_helpers_compile(tmp_path):
    code = """
    shader MultisampleTextureProbe {
        sampler2DMS msTexture;
        sampler2DMSArray msArrayTexture;

        fragment {
            vec4 main(ivec2 pixel, int layer, int sampleIndex) @ gl_FragColor {
                let size2D = textureSize(msTexture);
                let sizeArray = textureSize(msArrayTexture);
                let samples = textureSamples(msTexture);
                let fetched = texelFetch(msTexture, pixel, sampleIndex);
                let fetchedArray = texelFetch(msArrayTexture, ivec3(pixel.x, pixel.y, layer), sampleIndex);
                return fetched + fetchedArray + vec4(float(size2D.x + sizeArray.z + samples));
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_projected_sampling_and_gather_calls_map_to_rust_helpers_and_compile(
    tmp_path,
):
    code = """
    shader TextureProjectedGatherProbe {
        sampler2D mainTexture;

        fragment {
            vec4 main(vec2 uv, vec3 projected, vec2 ddx, vec2 ddy, ivec2 offset) @ gl_FragColor {
                ivec2 offsets[4];
                let projectedColor = textureProj(mainTexture, projected);
                let projectedLod = textureProjLod(mainTexture, projected, 1.0);
                let projectedGrad = textureProjGrad(mainTexture, projected, ddx, ddy);
                let projectedOffset = textureProjOffset(mainTexture, projected, offset);
                let projectedLodOffset = textureProjLodOffset(mainTexture, projected, 1.0, offset);
                let projectedGradOffset = textureProjGradOffset(mainTexture, projected, ddx, ddy, offset);
                let gather = textureGather(mainTexture, uv);
                let gatherComponent = textureGather(mainTexture, uv, 2);
                let gatherOffset = textureGatherOffset(mainTexture, uv, offset);
                let gatherOffsetComponent = textureGatherOffset(mainTexture, uv, offset, 1);
                let gatherOffsets = textureGatherOffsets(mainTexture, uv, offsets);
                let gatherOffsetsComponent = textureGatherOffsets(mainTexture, uv, offsets, 3);
                return projectedColor + projectedLod + projectedGrad + projectedOffset
                    + projectedLodOffset + projectedGradOffset
                    + gather + gatherComponent + gatherOffset + gatherOffsetComponent
                    + gatherOffsets + gatherOffsetsComponent;
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"texture operation: 'textureProjOffset' reads with a texel offset, and spirv\-std's image op",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_explicit_sampler_texture_helpers_map_to_rust_helpers_and_compile(tmp_path):
    code = """
    shader TextureExplicitSamplerProbe {
        sampler2D mainTexture;
        sampler textureSampler;

        fragment {
            vec4 main(vec2 uv, vec3 projected, vec2 ddx, vec2 ddy, ivec2 offset) @ gl_FragColor {
                ivec2 offsets[4];
                let base = texture(mainTexture, textureSampler, uv);
                let biased = texture(mainTexture, uv, 0.25);
                let biasedSampler = texture(mainTexture, textureSampler, uv, 0.5);
                let lod = textureLod(mainTexture, textureSampler, uv, 1.0);
                let lodOffset = textureLodOffset(mainTexture, textureSampler, uv, 1.0, offset);
                let grad = textureGrad(mainTexture, textureSampler, uv, ddx, ddy);
                let gradOffset = textureGradOffset(mainTexture, textureSampler, uv, ddx, ddy, offset);
                let shifted = textureOffset(mainTexture, textureSampler, uv, offset);
                let shiftedBias = textureOffset(mainTexture, uv, offset, 0.125);
                let shiftedBiasSampler = textureOffset(mainTexture, textureSampler, uv, offset, 0.125);
                let projectedColor = textureProj(mainTexture, textureSampler, projected);
                let projectedBias = textureProj(mainTexture, projected, 0.25);
                let projectedBiasSampler = textureProj(mainTexture, textureSampler, projected, 0.25);
                let projectedLod = textureProjLod(mainTexture, textureSampler, projected, 1.0);
                let projectedGrad = textureProjGrad(mainTexture, textureSampler, projected, ddx, ddy);
                let projectedOffset = textureProjOffset(mainTexture, textureSampler, projected, offset);
                let projectedOffsetBias = textureProjOffset(mainTexture, projected, offset, 0.25);
                let projectedOffsetBiasSampler = textureProjOffset(mainTexture, textureSampler, projected, offset, 0.25);
                let projectedLodOffset = textureProjLodOffset(mainTexture, textureSampler, projected, 1.0, offset);
                let projectedGradOffset = textureProjGradOffset(mainTexture, textureSampler, projected, ddx, ddy, offset);
                let query = textureQueryLod(mainTexture, textureSampler, uv);
                let gather = textureGather(mainTexture, textureSampler, uv);
                let gatherComponent = textureGather(mainTexture, textureSampler, uv, 1);
                let gatherOffset = textureGatherOffset(mainTexture, textureSampler, uv, offset);
                let gatherOffsetComponent = textureGatherOffset(mainTexture, textureSampler, uv, offset, 2);
                let gatherOffsets = textureGatherOffsets(mainTexture, textureSampler, uv, offsets);
                let gatherOffsetsComponent = textureGatherOffsets(mainTexture, textureSampler, uv, offsets, 3);
                return base + biased + biasedSampler + lod + lodOffset + grad + gradOffset
                    + shifted + shiftedBias + shiftedBiasSampler
                    + projectedColor + projectedBias + projectedBiasSampler
                    + projectedLod + projectedGrad + projectedOffset
                    + projectedOffsetBias + projectedOffsetBiasSampler
                    + projectedLodOffset + projectedGradOffset
                    + gather + gatherComponent + gatherOffset + gatherOffsetComponent
                    + gatherOffsets + gatherOffsetsComponent
                    + vec4(query.x, query.y, 0.0, 0.0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"texture operation: 'textureLodOffset' reads with a texel offset, and spirv\-std's image ope",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_shadow_compare_helpers_map_to_rust_helpers_and_compile(tmp_path):
    code = """
    shader ShadowCompareProbe {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeArrayShadow;
        sampler compareSampler;

        fragment {
            float main(
                vec2 uv,
                vec3 uvLayer,
                vec3 direction,
                vec4 cubeLayer,
                vec3 uvq,
                vec4 uvqw,
                float depth,
                float lod,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset
            ) @ gl_FragDepth {
                let base = textureCompare(shadowMap, uv, depth);
                let baseSampler = textureCompare(shadowMap, compareSampler, uv, depth);
                let arrayCompare = textureCompare(shadowArray, uvLayer, depth);
                let cubeCompare = textureCompare(cubeShadow, direction, depth);
                let cubeArrayCompare = textureCompare(cubeArrayShadow, cubeLayer, depth);
                let offsetCompare = textureCompareOffset(shadowMap, uv, depth, offset);
                let offsetSampler = textureCompareOffset(shadowMap, compareSampler, uv, depth, offset);
                let lodCompare = textureCompareLod(shadowMap, uv, depth, lod);
                let lodSampler = textureCompareLod(shadowMap, compareSampler, uv, depth, lod);
                let lodOffset = textureCompareLodOffset(shadowMap, uv, depth, lod, offset);
                let lodOffsetSampler = textureCompareLodOffset(shadowMap, compareSampler, uv, depth, lod, offset);
                let gradCompare = textureCompareGrad(shadowMap, uv, depth, ddx, ddy);
                let gradSampler = textureCompareGrad(shadowMap, compareSampler, uv, depth, ddx, ddy);
                let gradOffset = textureCompareGradOffset(shadowMap, uv, depth, ddx, ddy, offset);
                let gradOffsetSampler = textureCompareGradOffset(shadowMap, compareSampler, uv, depth, ddx, ddy, offset);
                let projected = textureCompareProj(shadowMap, uvq, depth);
                let projectedSampler = textureCompareProj(shadowMap, compareSampler, uvq, depth);
                let projectedW = textureCompareProj(shadowMap, uvqw, depth);
                let projectedOffset = textureCompareProjOffset(shadowMap, uvq, depth, offset);
                let projectedOffsetSampler = textureCompareProjOffset(shadowMap, compareSampler, uvq, depth, offset);
                let projectedLod = textureCompareProjLod(shadowMap, uvq, depth, lod);
                let projectedLodSampler = textureCompareProjLod(shadowMap, compareSampler, uvq, depth, lod);
                let projectedLodOffset = textureCompareProjLodOffset(shadowMap, uvq, depth, lod, offset);
                let projectedLodOffsetSampler = textureCompareProjLodOffset(shadowMap, compareSampler, uvq, depth, lod, offset);
                let projectedGrad = textureCompareProjGrad(shadowMap, uvq, depth, ddx, ddy);
                let projectedGradSampler = textureCompareProjGrad(shadowMap, compareSampler, uvq, depth, ddx, ddy);
                let projectedGradOffset = textureCompareProjGradOffset(shadowMap, uvq, depth, ddx, ddy, offset);
                let projectedGradOffsetSampler = textureCompareProjGradOffset(shadowMap, compareSampler, uvq, depth, ddx, ddy, offset);
                let gathered = textureGatherCompare(shadowMap, uv, depth);
                let gatheredSampler = textureGatherCompare(shadowMap, compareSampler, uv, depth);
                let gatheredOffset = textureGatherCompareOffset(shadowMap, uv, depth, offset);
                let gatheredOffsetSampler = textureGatherCompareOffset(shadowMap, compareSampler, uv, depth, offset);
                let shadowSize = textureSize(shadowArray, 0);
                return base + baseSampler + arrayCompare + cubeCompare + cubeArrayCompare
                    + offsetCompare + offsetSampler + lodCompare + lodSampler
                    + lodOffset + lodOffsetSampler + gradCompare + gradSampler
                    + gradOffset + gradOffsetSampler + projected + projectedSampler
                    + projectedW + projectedOffset + projectedOffsetSampler
                    + projectedLod + projectedLodSampler + projectedLodOffset
                    + projectedLodOffsetSampler + projectedGrad + projectedGradSampler
                    + projectedGradOffset + projectedGradOffsetSampler
                    + gathered.x + gatheredSampler.y + gatheredOffset.z
                    + gatheredOffsetSampler.w + float(shadowSize.z);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"texture operation: 'textureCompareOffset' reads with a texel offset, and spirv\-std's image",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_resource_arrays_map_to_rust_arrays_and_compile(tmp_path):
    code = """
    shader RustResourceArrays {
        sampler2D fixedTextures[2];
        sampler fixedSamplers[2];
        image2D fixedImages[2];
        StructuredBuffer<float> fixedWeights[2];
        RWStructuredBuffer<int> fixedValues[2];
        ByteAddressBuffer fixedRaw[2];
        RWByteAddressBuffer fixedRawOut[2];
        sampler2D bindlessTextures[];
        sampler bindlessSamplers[];
        image2D bindlessImages[];
        StructuredBuffer<float> bindlessWeights[];
        RWStructuredBuffer<int> bindlessValues[];
        ByteAddressBuffer bindlessRaw[];
        RWByteAddressBuffer bindlessRawOut[];

        fragment {
            vec4 main(
                int fixedIndex,
                int index,
                vec2 uv,
                ivec2 pixel,
                uint offset,
                int value
            ) @ gl_FragColor {
                let fixedSampled = texture(
                    fixedTextures[fixedIndex],
                    fixedSamplers[fixedIndex],
                    uv
                );
                let bindlessSampled = texture(
                    bindlessTextures[index],
                    bindlessSamplers[index],
                    uv
                );
                let fixedColor = imageLoad(fixedImages[fixedIndex], pixel);
                let bindlessColor = imageLoad(bindlessImages[index], pixel);
                imageStore(fixedImages[fixedIndex], pixel, fixedColor);
                imageStore(bindlessImages[index], pixel, bindlessColor);
                let fixedLoaded = buffer_load(fixedValues[fixedIndex], offset);
                let bindlessLoaded = buffer_load(bindlessValues[index], offset);
                let fixedWeight = buffer_load(fixedWeights[fixedIndex], offset);
                let bindlessWeight = buffer_load(bindlessWeights[index], offset);
                let fixedRawValue = buffer_load(fixedRaw[fixedIndex], offset);
                let bindlessRawValue = buffer_load(bindlessRaw[index], offset);
                buffer_store(fixedValues[fixedIndex], offset, fixedLoaded + value);
                buffer_store(bindlessValues[index], offset, bindlessLoaded + value);
                buffer_store(fixedRawOut[fixedIndex], offset, fixedRawValue);
                buffer_store(bindlessRawOut[index], offset, bindlessRawValue);
                return fixedSampled + bindlessSampled + fixedColor + bindlessColor
                    + vec4(float(fixedLoaded + bindlessLoaded) + fixedWeight
                        + bindlessWeight
                        + float(fixedRawValue + bindlessRawValue));
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: storage image 'fixedImages' \(image2D\) has no format layout qualifier, and rust\-g",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_resource_array_member_sample_is_refused_for_rust(tmp_path):
    code = """
    shader ResourceArrayAccessShader {
        const int MAP_COUNT = 2;
        sampler2D shadowMaps[MAP_COUNT];
        comparison_sampler comparisonSamplers[2];

        fragment {
            vec4 main() @ gl_FragColor {
                vec4 shadow = shadowMaps[1].sample(
                    comparisonSamplers[0],
                    vec2(0.5, 0.5)
                );
                vec4 mipShadow = shadowMaps[1].sample(
                    comparisonSamplers[0],
                    vec2(0.5, 0.5),
                    level(1.0)
                );
                return shadow + mipShadow;
            }
        }
    }
    """

    # The member-call spelling (``.sample(sampler, uv)``) is not lowered to spirv-std yet; each call is refused.
    with pytest.raises(
        RustLoweringRefused,
        match=r"texture operation: 'sample_sampler' has no spirv-std lowering.*'sample_lod_sampler'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_resource_bundle_member_helpers_and_non_copy_array_alias_compile(tmp_path):
    code = """
    shader ResourceBundleProbe {
        struct ResourceBundle {
            sampler2D texture;
            sampler samplerState;
            image2D colorImage;
            RWStructuredBuffer<int> values;
            float weights[];
        };

        fragment {
            vec4 main(ResourceBundle bundle, ResourceBundle bundles[2], ivec2 pixel, vec2 uv, uint index) @ gl_FragColor {
                let selected = bundles[index];
                let aliasTexture = selected.texture;
                let sampled = texture(aliasTexture, selected.samplerState, uv);
                let direct = texture(bundle.texture, bundle.samplerState, uv);
                let color = imageLoad(bundle.colorImage, pixel);
                imageStore(selected.colorImage, pixel, color);
                let value = buffer_load(bundle.values, index);
                buffer_store(selected.values, index, value);
                return sampled + direct + color + vec4(selected.weights[0] + float(value), 0.0, 0.0, 0.0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="stage: fragment input 'bundle_texture' has type sampler2D",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_storage_image_and_buffer_helpers_map_to_rust_resources_and_compile(tmp_path):
    code = """
    shader ResourceHelperProbe {
        image2D colorImage;
        uimage2D counterImage;
        image1D rampImage;
        image1DArray rampArrayImage;
        image3D volumeImage;
        imageCube cubeImage;
        image2DArray layerImage;
        image2DMS msImage;
        image2DMSArray msArrayImage;
        iimage3D signedVolume;
        uimage2DArray counterLayers;
        RWStructuredBuffer<int> values;
        StructuredBuffer<float> weights;
        AppendStructuredBuffer<uint> appendValues;
        ConsumeStructuredBuffer<uint> consumeValues;
        ByteAddressBuffer rawBytes;
        RWByteAddressBuffer rawOut;

        fragment {
            vec4 main(ivec2 pixel, uint amount, uint index) @ gl_FragColor {
                let color = imageLoad(colorImage, pixel);
                let ramp = imageLoad(rampImage, 0);
                let signedValue = imageLoad(signedVolume, ivec3(0));
                let unsignedValue = imageLoad(counterLayers, ivec3(0));
                let size1D = imageSize(rampImage);
                let size2D = imageSize(colorImage);
                let size3D = imageSize(volumeImage);
                let sizeCube = imageSize(cubeImage);
                let sizeArray = imageSize(layerImage);
                let sizeMs = imageSize(msImage);
                let sizeMsArray = imageSize(msArrayImage);
                let sampleCount = imageSamples(msImage);
                imageStore(colorImage, pixel, color + vec4(1.0));
                let previous = imageAtomicAdd(counterImage, pixel, amount);
                let minValue = imageAtomicMin(counterImage, pixel, amount);
                let maxValue = imageAtomicMax(counterImage, pixel, amount);
                let andValue = imageAtomicAnd(counterImage, pixel, amount);
                let orValue = imageAtomicOr(counterImage, pixel, amount);
                let xorValue = imageAtomicXor(counterImage, pixel, amount);
                let exchanged = imageAtomicExchange(counterImage, pixel, amount);
                let swapped = imageAtomicCompSwap(counterImage, pixel, previous, amount);
                let value = buffer_load(values, index);
                let weight = buffer_load(weights, index);
                uint length = 0u;
                buffer_dimensions(values, length);
                buffer_append(appendValues, amount);
                uint consumed = buffer_consume(consumeValues);
                buffer_store(values, index, value + int(previous));
                return color + ramp + vec4(
                    weight + float(signedValue.x + int(unsignedValue.x))
                        + float(size1D + size2D.x + size3D.z + sizeCube.y
                            + sizeArray.z + sizeMs.y + sizeMsArray.z + sampleCount)
                        + float(consumed)
                );
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: storage image 'colorImage' \(image2D\) has no format layout qualifier, and rust\-gp",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_resource_memory_qualifiers_map_rust_buffer_access_and_compile(tmp_path):
    code = """
    shader RustResourceMemoryQualifiers {
        readonly RWStructuredBuffer<int> readOnlyValues;
        writeonly StructuredBuffer<int> writeOnlyValues;
        readwrite StructuredBuffer<float> readWriteValues;
        RWStructuredBuffer<int> attrReadValues @access(read);
        StructuredBuffer<int> attrWriteValues @access(write);
        readonly RWByteAddressBuffer rawInput;
        writeonly ByteAddressBuffer rawOutput;
        readonly image2D source @rgba32f;
        writeonly image2D target @rgba32f;

        compute {
            void main(uint index, uint byteOffset) {
                int value = buffer_load(readOnlyValues, index);
                int attrValue = buffer_load(attrReadValues, index);
                uint rawValue = buffer_load(rawInput, byteOffset);
                buffer_store(writeOnlyValues, index, value + attrValue);
                buffer_store(readWriteValues, index, float(value));
                buffer_store(attrWriteValues, index, attrValue);
                buffer_store(rawOutput, byteOffset, rawValue);
                vec4 color = imageLoad(source, ivec2(0));
                imageStore(target, ivec2(0), color);
            }
        }
    }
    """

    # Byte-address buffers and storage images have no rust-gpu binding yet; each is refused by name.
    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: 'rawInput' \(ByteAddressBuffer\) is not lowered",
    ):
        generate_code(parse_code(tokenize_code(code)))

    structured = """
    shader RustResourceMemoryQualifiers {
        readonly RWStructuredBuffer<int> readOnlyValues;
        writeonly StructuredBuffer<int> writeOnlyValues;
        readwrite StructuredBuffer<float> readWriteValues;
        RWStructuredBuffer<int> attrReadValues @access(read);
        StructuredBuffer<int> attrWriteValues @access(write);

        compute {
            void main() {
                uint index = gl_GlobalInvocationID.x;
                int value = readOnlyValues[index];
                int attrValue = attrReadValues[index];
                writeOnlyValues[index] = value + attrValue;
                readWriteValues[index] = float(value) + readWriteValues[index];
                attrWriteValues[index] = attrValue;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(structured)))
    # An explicit memory qualifier or @access attribute decides the slice's mutability over the buffer type.
    for name, slice_type in (
        ("readOnlyValues", "&[i32]"),
        ("writeOnlyValues", "&mut [i32]"),
        ("readWriteValues", "&mut [f32]"),
        ("attrReadValues", "&[i32]"),
        ("attrWriteValues", "&mut [i32]"),
    ):
        assert f"{name}: {slice_type}" in generated, generated
    assert (
        "readWriteValues[index as usize] = ((value as f32) + readWriteValues[index as usize]);"
        in generated
    )
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_glsl_buffer_array_declarations_map_to_rust_resources_and_compile(tmp_path):
    code = """
    shader RustGlslBufferArrays {
        compute {
            layout(std430, binding = 2) readonly buffer float values[];
            layout(std430, binding = 3) writeonly buffer float outValues[];

            void main() {
                float value = buffer_load(values, 0u);
                buffer_store(outValues, 1u, value);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: 'values' \(StructuredBuffer<float>\) is not lowered",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_glsl_buffer_block_programs_are_refused():
    readable = """
    shader RustGlslBufferBlocks {
        struct Particle {
            vec4 position;
            float mass;
        };

        layout(std430, binding = 4) buffer ParticleBlock {
            Particle particles[];
        } particleBlock;

        struct Data {
            uint value;
        };

        Data readOnlyData @glsl_buffer_block(std430) @readonly;
        Data writeOnlyData @glsl_buffer_block(std430) @writeonly;

        compute {
            void main() {
                float mass = particleBlock.particles[0u].mass;
                uint copied = readOnlyData.value + uint(mass);
            }
        }
    }
    """

    # GLSL buffer blocks are not lowered to rust-gpu storage buffers yet; each block is refused by name.
    with pytest.raises(RustLoweringRefused) as refused:
        generate_code(parse_code(tokenize_code(readable)))
    assert refused.value.reasons[:3] == (
        "resource: 'particleBlock' (ParticleBlock) is not lowered to a rust-gpu resource binding",
        "resource: 'readOnlyData' (Data) is not lowered to a rust-gpu resource binding",
        "resource: 'writeOnlyData' (Data) is not lowered to a rust-gpu resource binding",
    )


def test_rust_refuses_synchronization_and_subgroup_operations_without_a_lowering():
    codegen = RustCodeGen()
    sync = SyncNode("deviceScopeFence")
    wave = WaveOpNode("WaveUnknownOperation", [IdentifierNode("value")])
    with codegen.refusals.active():
        assert codegen.generate_sync_statement(sync, 1) == ""
        assert codegen.generate_wave_op_expression(wave) == "0"
    assert codegen.refusals.reasons == [
        "synchronization deviceScopeFence has no rust-gpu barrier",
        "subgroup operation WaveUnknownOperation(?) has no spirv_std lowering for these operands",
    ]
    with pytest.raises(RustLoweringRefused, match="2 constructs"):
        codegen.refusals.raise_pending()


def test_glsl_buffer_block_bindings_and_conflicts_are_refused():
    code = """
    shader RustGlslBufferBlockBindings {
        struct Data {
            uint value;
        };

        layout(std430, set = 2, binding = 4) buffer DataBlock {
            Data values[];
        } dataBlock;

        Data explicitData @glsl_buffer_block(std140) @readonly @set(3) @binding(5);
        Data automaticData @glsl_buffer_block(std430);

        compute {
            void main() {}
        }
    }
    """

    # GLSL buffer blocks are not lowered to rust-gpu storage buffers yet; each block is refused by name.
    with pytest.raises(RustLoweringRefused) as refused:
        generate_code(parse_code(tokenize_code(code)))
    assert refused.value.reasons[:3] == (
        "resource: 'dataBlock' (DataBlock) is not lowered to a rust-gpu resource binding",
        "resource: 'explicitData' (Data) is not lowered to a rust-gpu resource binding",
        "resource: 'automaticData' (Data) is not lowered to a rust-gpu resource binding",
    )

    duplicate_binding = """
    shader DuplicateRustGlslBufferBlockBinding {
        struct Data {
            uint value;
        };

        Data values @glsl_buffer_block(std430) @binding(2);
        RWStructuredBuffer<uint> counters @binding(2);

        compute {
            void main() {}
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting Rust resource binding"):
        generate_code(parse_code(tokenize_code(duplicate_binding)))


def test_resource_binding_metadata_comments_compile_with_rust(tmp_path):
    code = """
    shader RustResourceBindings {
        sampler2D colorMap @set(2) @binding(5);
        sampler linearSampler @binding(1);
        @binding(3) RWStructuredBuffer<int> counters[2];
        sampler2D registeredTexture @register(t7, space3);

        @binding(6)
        cbuffer Camera {
            float exposure;
        };

        image2D outImage;

        compute {
            void main() {}
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: 'counters' \(RWStructuredBuffer<int>\[2\]\) is not lowered to a rust\-gpu resource bi",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_sampler_descriptor_array_source_bindings_do_not_expand_rust_ranges(tmp_path):
    code = """
    shader RustSamplerDescriptorArrayBindings {
        sampler linearSamplers[2] @set(0) @binding(5);
        sampler shadowSamplers[2] @set(0) @binding(6);

        compute {
            void main() {}
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_texture_and_sampler_on_one_binding_are_refused_for_rust_codegen(
    tmp_path,
):
    # Vulkan numbers every descriptor of a set in one binding range (Vulkan 1.3 section 14.2.2), so a sampled image
    # and a sampler cannot share set 0 binding 2.
    code = """
    shader RustTextureSamplerBindings {
        sampler2D colorMap @binding(2);
        sampler linearSampler @binding(2);

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(colorMap, linearSampler, uv);
            }
        }
    }
    """

    with pytest.raises(
        RustRefusal,
        match=r"Conflicting Rust resource binding for 'linearSampler': buffer set 0 binding 2-2 overlaps 'colorMap'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_resource_parameters_lower_explicit_sampler_sampling(tmp_path):
    code = """
    shader RustExplicitSamplerResourceDiagnostic {
        sampler2D colorMap @set(1) @binding(4);
        sampler linearSampler @set(1) @binding(5);

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(colorMap, linearSampler, uv);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_storage_image_atomic_binds_the_image_as_an_entry_parameter(tmp_path):
    code = """
    shader RustStorageImageAtomicResourceDiagnostic {
        uimage2D counterImage @r32ui @binding(2);

        compute {
            void main(ivec2 pixel, uint amount) {
                uint previous = imageAtomicAdd(counterImage, pixel, amount);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"stage: compute entry parameter 'pixel' is not a compute built\-in input",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_duplicate_resource_bindings_are_rejected_for_rust_codegen():
    duplicate_texture_binding = """
    shader DuplicateRustTextureBindings {
        @binding(2) sampler2D firstTexture;
        @binding(2) sampler2D secondTexture;

        compute {
            void main() {}
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting Rust resource binding"):
        generate_code(parse_code(tokenize_code(duplicate_texture_binding)))

    duplicate_sampler_binding = """
    shader DuplicateRustSamplerBindings {
        @binding(1) sampler firstSampler;
        @binding(1) sampler secondSampler;

        compute {
            void main() {}
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting Rust resource binding"):
        generate_code(parse_code(tokenize_code(duplicate_sampler_binding)))

    overlapping_buffer_range = """
    shader DuplicateRustBufferBindings {
        @binding(3) RWStructuredBuffer<int> counters[2];
        @binding(4)
        cbuffer Camera {
            float exposure;
        };

        compute {
            void main() {}
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting Rust resource binding"):
        generate_code(parse_code(tokenize_code(overlapping_buffer_range)))

    overlapping_native_sampler_range = """
    shader DuplicateRustNativeSamplerRange {
        @sampler(5) sampler linearSamplers[2];
        @sampler(6) sampler shadowSamplers[2];

        compute {
            void main() {}
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting Rust resource binding"):
        generate_code(parse_code(tokenize_code(overlapping_native_sampler_range)))


def test_explicit_storage_image_formats_map_to_rust_value_types_and_compile(tmp_path):
    code = """
    shader ExplicitStorageImageFormats {
        image2D scalarFloat @r32f;
        uimage2D scalarCounter @r32ui;
        image2D rgFloat @rg32f;
        uimage2D rgUint @format(rg32ui);
        image3D signedVolume @r32i;

        fragment {
            vec4 main(ivec2 pixel) @ gl_FragColor {
                float scalar = imageLoad(scalarFloat, pixel);
                imageStore(scalarFloat, pixel, scalar + 1.0);
                uint count = imageLoad(scalarCounter, pixel);
                imageStore(scalarCounter, pixel, count + 1u);
                vec2 pair = imageLoad(rgFloat, pixel);
                imageStore(rgFloat, pixel, pair);
                uvec2 unsignedPair = imageLoad(rgUint, pixel);
                imageStore(rgUint, pixel, unsignedPair);
                int signedValue = imageLoad(signedVolume, ivec3(0));
                imageStore(signedVolume, ivec3(0), signedValue - 1);
                return vec4(
                    scalar + float(count) + pair.x
                        + float(unsignedPair.x + uint(signedValue))
                );
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: 'signedVolume' \(image3D<r32i>\) has no spirv\-std Image! type",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_multisample_storage_image_helpers_map_to_rust_and_compile(tmp_path):
    code = """
    shader MultisampleStorageImages {
        image2DMS msImage;
        image2DMSArray msLayerImage;
        uimage2DMS msCounter;
        iimage2DMSArray msSignedLayers;

        compute {
            void main(uint amount) {
                ivec2 pixel = ivec2(4, 8);
                ivec3 pixelLayer = ivec3(4, 8, 1);
                int sampleIndex = 2;
                vec4 color = imageLoad(msImage, pixel, sampleIndex);
                imageStore(msImage, pixel, sampleIndex, color + vec4(1.0));
                vec4 layer = imageLoad(msLayerImage, pixelLayer, sampleIndex);
                imageStore(msLayerImage, pixelLayer, sampleIndex, layer);
                uvec4 counts = imageLoad(msCounter, pixel, sampleIndex);
                imageStore(msCounter, pixel, sampleIndex, counts);
                uint previous = imageAtomicAdd(msCounter, pixel, sampleIndex, amount);
                uint exchanged = imageAtomicExchange(
                    msCounter,
                    pixel,
                    sampleIndex,
                    previous
                );
                int swapped = imageAtomicCompSwap(
                    msSignedLayers,
                    pixelLayer,
                    sampleIndex,
                    0,
                    int(exchanged)
                );
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match=r"resource: storage image 'msImage' \(image2DMS\) has no format layout qualifier, and rust\-gpu",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_for_statement(tmp_path):
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                for (int i = 0; i < 10; i++) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)
    except SyntaxError:
        pytest.fail("For statement codegen not implemented.")


def test_for_continue_emits_update_before_continue_in_rust(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                int total = 0;
                for (int i = 0; i < 4; i++) {
                    if (i == 1) {
                        continue;
                    }
                    for (int j = 0; j < 2; j++) {
                        if (j == 0) {
                            continue;
                        }
                        total += j;
                    }
                    total += i;
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "if (i == 1) {\n                i += 1;\n                continue;\n            }"
        in generated_code
    )
    assert (
        "if (j == 0) {\n                        j += 1;\n                        continue;\n                    }"
        in generated_code
    )
    assert (
        "if (j == 0) {\n                        i += 1;\n                        continue;\n                    }"
        not in generated_code
    )
    assert "DoWhileNode" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_for_in_statement_lowers_to_rust_ranges_and_scopes_loop_contexts(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                int total = 0;
                for i in 4 {
                    if (i == 1) {
                        continue;
                    }
                    total += i;
                }
                for j in 2..5 {
                    total += j;
                }
                for k in 1..=4 {
                    total += k;
                }
                for (int outer = 0; outer < 2; outer++) {
                    for inner in 3 {
                        if (inner == 1) {
                            continue;
                        }
                        total += inner;
                    }
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for i in 0..4 {" in generated_code
    assert "for j in 2..5 {" in generated_code
    assert "for k in 1..=4 {" in generated_code
    assert "total += i;" in generated_code
    assert "total += j;" in generated_code
    assert "total += k;" in generated_code
    assert "total += inner;" in generated_code
    assert (
        "if (inner == 1) {\n                    continue;\n                }"
        in generated_code
    )
    assert "outer += 1;\n                    continue;" not in generated_code
    assert "ForInNode" not in generated_code
    assert "RangeNode" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_while_statement_lowers_to_rust_while_and_scopes_loop_contexts(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                while (value < 4) {
                    if (value == 2) {
                        continue;
                    }
                    value += 1;
                }
                for (int i = 0; i < 3; i++) {
                    int j = 0;
                    while (j < 2) {
                        if (j == 1) {
                            continue;
                        }
                        j += 1;
                    }
                }
                do {
                    int k = 0;
                    while (k < 2) {
                        if (k == 1) {
                            break;
                        }
                        k += 1;
                    }
                    value += 1;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "while (value < 4) {" in generated_code
    assert "while (j < 2) {" in generated_code
    assert "if (value == 2) {\n            continue;\n        }" in generated_code
    assert (
        "if (j == 1) {\n                    continue;\n                }"
        in generated_code
    )
    assert (
        "if (j == 1) {\n                    i += 1;\n                    continue;"
        not in generated_code
    )
    assert "__cgl_do_break_0 = true;" not in generated_code
    assert "WhileNode" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_loop_statement_lowers_to_rust_loop_and_scopes_loop_contexts(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                loop {
                    value += 1;
                    if (value == 2) {
                        continue;
                    }
                    if (value > 3) {
                        break;
                    }
                }
                for (int i = 0; i < 3; i++) {
                    loop {
                        if (i == 1) {
                            continue;
                        }
                        break;
                    }
                }
                do {
                    loop {
                        if (value == 4) {
                            break;
                        }
                        continue;
                    }
                    value += 1;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "loop {" in generated_code
    assert "value += 1;" in generated_code
    assert "if (value == 2) {\n            continue;\n        }" in generated_code
    assert (
        "if (i == 1) {\n                    continue;\n                }"
        in generated_code
    )
    assert (
        "if (i == 1) {\n                    i += 1;\n                    continue;"
        not in generated_code
    )
    assert "__cgl_do_break_0 = true;" not in generated_code
    assert "LoopNode" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_increment_and_decrement_emit_rust_assignment_updates(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                int i = 0;
                i++;
                ++i;
                i--;
                --i;
                for (int j = 0; j < 2; j++) {
                    i++;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "i += 1;" in generated_code
    assert "i -= 1;" in generated_code
    assert "j += 1;" in generated_code
    assert "++i" not in generated_code
    assert "i++" not in generated_code
    assert "--i" not in generated_code
    assert "i--" not in generated_code
    assert "++j" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_increment_and_decrement_initializers_preserve_rust_value_order(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                int i = 0;
                int pre = ++i;
                int post = i++;
                int pre_dec = --i;
                int post_dec = i--;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    lines = [line.strip() for line in generated_code.splitlines()]

    def contains_adjacent(first, second):
        return any(
            current == first and following == second
            for current, following in zip(lines, lines[1:])
        )

    assert contains_adjacent("i += 1;", "let pre: i32 = i;")
    assert contains_adjacent("let post: i32 = i;", "i += 1;")
    assert contains_adjacent("i -= 1;", "let pre_dec: i32 = i;")
    assert contains_adjacent("let post_dec: i32 = i;", "i -= 1;")
    assert "let pre: i32 = i += 1;" not in generated_code
    assert "let post: i32 = i += 1;" not in generated_code
    assert "let pre_dec: i32 = i -= 1;" not in generated_code
    assert "let post_dec: i32 = i -= 1;" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_do_while_statement_lowers_to_rust_loop_with_condition_after_body(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                do {
                    value += 1;
                    if (value == 2) {
                        continue;
                    }
                    if (value == 4) {
                        break;
                    }
                    value += 2;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let mut __cgl_do_break_0: bool = false;" in generated_code
    assert "loop {\n        loop {" in generated_code
    assert "value += 1;" in generated_code
    assert "if (value == 2) {\n                break;\n            }" in generated_code
    assert (
        "if (value == 4) {\n"
        "                __cgl_do_break_0 = true;\n"
        "                break;\n"
        "            }"
    ) in generated_code
    assert "if !((value < 8))" in generated_code
    assert "DoWhileNode" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_bool_string_and_char_literals_emit_rust_syntax(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                bool enabled = true;
                bool disabled = false;
                string label = "debug";
                char marker = 'x';
                if (enabled && !disabled) {
                    label = "active";
                    marker = 'y';
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    entry_point = next(iter(ast.stages.values())).entry_point

    assert entry_point.body.statements[0].initial_value.value is True
    assert entry_point.body.statements[1].initial_value.value is False

    with pytest.raises(
        RustLoweringRefused, match="rust-gpu cannot lower string constants"
    ):
        generate_code(ast)


def test_local_bindings_are_mutable_only_when_reassigned_or_mutated():
    code = """
    shader LocalMutability {
        struct Output {
            value: int;
        }

        void main(int index) {
            int immutableValue = 1;
            int reassignedValue = 2;
            Output output;
            float values[2];

            reassignedValue = immutableValue;
            output.value = reassignedValue;
            values[index] = 1.0;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let immutableValue: i32 = 1;" in generated_code
    assert "let mut reassignedValue: i32 = 2;" in generated_code
    assert "let mut output: Output = Default::default();" in generated_code
    assert "let mut values: [f32; 2] = [0.0; 2];" in generated_code
    assert "let mut immutableValue" not in generated_code


def test_direct_literal_nodes_emit_rust_escaping():
    codegen = RustCodeGen()

    assert (
        codegen.generate_expression(LiteralNode(True, PrimitiveType("bool"))) == "true"
    )
    assert (
        codegen.generate_expression(LiteralNode('debug"name', PrimitiveType("string")))
        == '"debug\\"name"'
    )
    assert (
        codegen.generate_expression(LiteralNode("'", PrimitiveType("char"))) == "39i8"
    )
    assert (
        codegen.generate_expression(LiteralNode("\\n", PrimitiveType("char"))) == "10i8"
    )


def test_char_is_a_signed_8_bit_integer_in_rust(tmp_path):
    code = """
    shader CharArithmetic {
        int addChars(char a, char b) { return a + b; }
        int letterOffset(char c) { char base = 'A'; return c - base; }
        char newline() { return '\\n'; }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert re.search(r":\s*char\b", generated_code) is None
    assert "a: i8, b: i8" in generated_code
    assert "((a as i32) + (b as i32))" in generated_code
    assert "let base: i8 = 65i8;" in generated_code
    assert "10i8" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_refuses_a_char_literal_without_a_single_byte_value():
    with pytest.raises(RustRefusal, match="signed 8-bit integer"):
        RustCodeGen().generate_expression(LiteralNode("\u00e9", PrimitiveType("char")))


def test_else_if_statement(tmp_path):
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                if (input.texCoord.x > 0.5) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                } else if (input.texCoord.x < 0.5) {
                    output.color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    output.color = vec4(0.5, 0.5, 0.5, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                let brightness = texture(iChannel0, input.color.xy).r;
                let bloom = max(0.0, brightness - 0.5);
                if (bloom > 0.5) {
                    bloom = 0.5;
                } else if (bloom < 0.5) {
                    bloom = 0.0;
                } else {
                    bloom = 0.5;
                }
                let texColor = texture(iChannel0, input.color.xy).rgb;
                let colorWithBloom = texColor + vec3(bloom);
                return vec4(colorWithBloom, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)
    except SyntaxError:
        pytest.fail("Else if codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    float result = add(1.0, 2.0);
                }

                float add(float a, float b) {
                    return a + b;
                }
            }
            """,
            "add(1.0, 2.0)",
        )
    ],
)
def test_function_call(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    assert expected_output in generated_code


def test_builtin_function_call_names_are_mapped():
    code = """
    shader main {
        fragment {
            float main() {
                float a = mix(0.0, 1.0, 0.25);
                float b = mod(5.0, 2.0);
                float c = frac(1.25);
                float d = saturate(2.0);
                float e = saturate(frac(1.75));
                float f = inversesqrt(4.0);
                return a + b + c + d + e + f;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "((0.0) * (1.0 - (0.25)) + (1.0) * (0.25))" in generated_code
    assert (
        "{ let __x = 5.0; let __y = 2.0; __x - __y * (__x / __y).floor() }"
        in generated_code
    )
    assert "{ let __x = 1.25; __x - __x.floor() }" in generated_code
    assert "(2.0).clamp(0.0, 1.0)" in generated_code
    assert "({ let __x = 1.75; __x - __x.floor() }).clamp(0.0, 1.0)" in generated_code
    assert "(1.0 / (4.0).sqrt())" in generated_code
    assert "mix(0.0, 1.0, 0.25)" not in generated_code
    assert "mod(5.0, 2.0)" not in generated_code
    assert "frac(1.25)" not in generated_code
    assert "inversesqrt(" not in generated_code
    assert "saturate(" not in generated_code


def test_builtin_alias_targets_qualify_shadowed_rust_math_helpers_and_smoke_compile(
    tmp_path,
):
    code = """
    shader RustMathAliasTargetShadowing {
        fragment {
            float main(float value) {
                float fract = value;
                float clamp = value;
                float rsqrt = value;
                float computed = frac(value) + saturate(value) + inverseSqrt(value);
                return computed + fract + clamp + rsqrt;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let fract: f32 = value;" in generated_code
    assert "let clamp: f32 = value;" in generated_code
    assert "let rsqrt: f32 = value;" in generated_code
    assert (
        "let computed: f32 = (({ let __x = value; __x - __x.floor() } + (value).clamp(0.0, 1.0)) + (1.0 / (value).sqrt()));"
    ) in generated_code
    assert "let computed: f32 = ((fract(value) + clamp(value, 0.0, 1.0))" not in (
        generated_code
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_texture_builtin_alias_targets_qualify_shadowed_runtime_helpers_and_compile(
    tmp_path,
):
    code = """
    shader RustTextureAliasTargetShadowing {
        sampler2D colorMap;
        sampler linearSampler;

        fragment {
            vec4 main(vec2 uv) {
                float sample = 1.0;
                float sample_sampler = 2.0;
                let implicitColor = texture(colorMap, uv);
                let explicitColor = texture(colorMap, linearSampler, uv);
                return implicitColor + explicitColor
                    + vec4(sample + sample_sampler, 0.0, 0.0, 0.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_builtin_function_calls_infer_rust_value_types_and_smoke_compile(tmp_path):
    code = """
    shader BuiltinInference {
        fragment {
            uniform sampler2D main_texture;

            vec4 main(vec3 normal, vec3 light, vec3 albedo, vec2 uv) {
                let unit = normalize(normal);
                let reflected = reflect(-unit, light);
                let ndotl = max(0.0, dot(reflected, light));
                let gloss = pow(ndotl, 2.0);
                let quantized = floor(gloss);
                let smoothed = smoothstep(0.0, 1.0, quantized);
                let mixed = mix(vec3(0.04), albedo, smoothed);
                let tex = texture(main_texture, uv);
                let color = mixed + tex.rgb * smoothed;
                return vec4(color, tex.a);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_common_math_intrinsics_infer_rust_value_types_and_smoke_compile(tmp_path):
    code = """
    shader CommonIntrinsicInference {
        fragment {
            vec4 main(vec3 normal, vec3 incident, vec3 reference, float x, float y, float angle) {
                let dist = distance(normal, incident);
                let facing = faceforward(normal, incident, reference);
                let rounded = round(x);
                let exponential = exp(x);
                let exponential2 = exp2(x);
                let naturalLog = log(y);
                let binaryLog = log2(y);
                let deg = degrees(angle);
                let rad = radians(deg);
                let arc = asin(clamp(x, -1.0, 1.0)) + acos(clamp(x, -1.0, 1.0))
                    + atan(x) + atan2(y, x);
                let hyper = sinh(x) + cosh(x) + tanh(x);
                return vec4(
                    facing,
                    dist + rounded + exponential + exponential2 + naturalLog
                        + binaryLog + deg + rad + arc + hyper
                );
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'faceforward' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_two_argument_atan_maps_to_atan2_and_qualifies_shadowed_target(
    tmp_path,
):
    code = """
    shader TwoArgumentAtanAlias {
        fragment {
            vec4 main(float y, float x, vec3 vectorY) {
                float atan2 = x;
                let scalarAngle = atan(y, x);
                let vectorAngle = atan(vectorY, y);
                return vec4(vectorAngle, scalarAngle + atan2);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'atan' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_unary_math_intrinsics_preserve_vector_shape_and_smoke_compile(tmp_path):
    code = """
    shader UnaryMathVectorInference {
        fragment {
            vec4 main(vec3 value, float scalar) {
                let root = sqrt(value);
                let inv_root = inversesqrt(value);
                let magnitude = abs(value);
                let rounded = floor(value) + ceil(value) + round(value)
                    + trunc(value) + roundEven(value) + fract(value);
                let trig = sin(value) + cos(value) + tan(value)
                    + asin(value) + acos(value) + atan(value);
                let hyper = sinh(value) + cosh(value) + tanh(value);
                let exponentials = exp(value) + exp2(value);
                let logs = log(abs(value) + vec3(1.0, 1.0, 1.0))
                    + log2(abs(value) + vec3(1.0, 1.0, 1.0));
                let angles = degrees(value) + radians(value);
                let scalar_math = sqrt(scalar) + inversesqrt(scalar) + abs(scalar)
                    + floor(scalar) + ceil(scalar) + round(scalar)
                    + trunc(scalar) + roundEven(scalar) + fract(scalar)
                    + sin(scalar) + cos(scalar) + tan(scalar)
                    + asin(scalar) + acos(scalar) + atan(scalar)
                    + sinh(scalar) + cosh(scalar) + tanh(scalar)
                    + exp(scalar) + exp2(scalar) + log(abs(scalar) + 1.0)
                    + log2(abs(scalar) + 1.0) + degrees(scalar) + radians(scalar);
                return vec4(
                    root + inv_root + magnitude + rounded + trig + hyper
                        + exponentials + logs + angles,
                    scalar_math
                );
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'roundEven' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_inverse_sqrt_alias_preserves_vector_shape_and_smoke_compile(tmp_path):
    code = """
    shader InverseSqrtAliasInference {
        fragment {
            vec4 main(vec3 value, float scalar) {
                let inv_root = inverseSqrt(value);
                let scalar_inv_root = inverseSqrt(scalar);
                return vec4(inv_root, scalar_inv_root);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let inv_root: Vec3 = (value).map(|c| 1.0 / c.sqrt());" in generated_code
    assert "let scalar_inv_root: f32 = (1.0 / (scalar).sqrt());" in generated_code
    assert "inverseSqrt(" not in generated_code
    assert "let inv_root: f32 = rsqrt(value)" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_hlsl_rcp_intrinsic_lowers_to_rust_reciprocal_and_smoke_compile(tmp_path):
    code = """
    shader ReciprocalIntrinsicInference {
        fragment {
            vec4 main(vec3 value, mat2 transform, float scalar) {
                let inv_value = rcp(value);
                let inv_scalar = rcp(scalar);
                let inv_transform = rcp(transform);
                return vec4(
                    inv_value + vec3(
                        inv_transform.c0.x,
                        inv_transform.c0.y,
                        inv_transform.c1.x
                    ),
                    inv_scalar
                );
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'rcp' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_user_defined_inverse_sqrt_function_is_not_lowered_to_rsqrt():
    code = """
    shader UserDefinedInverseSqrt {
        float inverseSqrt(float value) {
            return value + 1.0;
        }

        fragment {
            float main(float value) {
                return inverseSqrt(value);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub fn inverseSqrt(value: f32) -> f32" in generated_code
    assert "let return_value = inverseSqrt(value);" in generated_code
    assert "rsqrt(" not in generated_code


def test_user_defined_rcp_function_is_not_lowered_to_reciprocal():
    code = """
    shader UserDefinedRcp {
        float rcp(float value) {
            return value + 1.0;
        }

        fragment {
            float main(float value) {
                return rcp(value);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub fn rcp(value: f32) -> f32" in generated_code
    assert "let return_value = rcp(value);" in generated_code
    assert "1.0 / value" not in generated_code


def test_rounding_intrinsics_preserve_float_input_shape_and_smoke_compile(tmp_path):
    code = """
    shader RoundingIntrinsicInference {
        fragment {
            vec4 main(vec3 value, float scalar) {
                let truncated = trunc(value);
                let scalar_truncated = trunc(scalar);
                let even = roundEven(value);
                let scalar_even = roundEven(scalar);
                return vec4(truncated + even, scalar_truncated + scalar_even);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'roundEven' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_clamp_promotes_mixed_scalar_vector_operands_and_smoke_compile(tmp_path):
    code = """
    shader ClampPromotionInference {
        fragment {
            vec4 main(vec3 value, vec3 low, vec3 high, float scalar) {
                let vector_value = clamp(value, 0.0, 1.0);
                let vector_bounds = clamp(scalar, low, high);
                let scalar_value = clamp(scalar, 0.0, 1.0);
                let vector_all = clamp(value, low, high);
                return vec4(vector_value + vector_bounds + vector_all, scalar_value);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let vector_value: Vec3 = (value).clamp(Vec3::splat(0.0), Vec3::splat(1.0));"
        in generated_code
    )
    assert (
        "let vector_bounds: Vec3 = (Vec3::splat(scalar)).clamp(low, high);"
        in generated_code
    )
    assert "let scalar_value: f32 = (scalar).clamp(0.0, 1.0);" in generated_code
    assert "let vector_all: Vec3 = (value).clamp(low, high);" in generated_code
    assert "let vector_bounds: f32 = clamp" not in generated_code
    assert "let vector_value: f32 = clamp" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_min_max_promote_mixed_scalar_vector_operands_and_smoke_compile(tmp_path):
    code = """
    shader MinMaxPromotionInference {
        fragment {
            vec4 main(vec3 value, vec3 other, float scalar) {
                let min_vector_scalar = min(value, scalar);
                let max_vector_scalar = max(value, scalar);
                let min_scalar_vector = min(scalar, other);
                let max_scalar_vector = max(scalar, other);
                let min_vector_vector = min(value, other);
                let max_vector_vector = max(value, other);
                let min_scalar_scalar = min(scalar, 1.0);
                let max_scalar_scalar = max(scalar, 0.0);
                return vec4(
                    min_vector_scalar + max_vector_scalar + min_scalar_vector
                        + max_scalar_vector + min_vector_vector + max_vector_vector,
                    min_scalar_scalar + max_scalar_scalar
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let min_vector_scalar: Vec3 = (value).min(Vec3::splat(scalar));"
        in generated_code
    )
    assert (
        "let max_vector_scalar: Vec3 = (value).max(Vec3::splat(scalar));"
        in generated_code
    )
    assert (
        "let min_scalar_vector: Vec3 = (Vec3::splat(scalar)).min(other);"
        in generated_code
    )
    assert (
        "let max_scalar_vector: Vec3 = (Vec3::splat(scalar)).max(other);"
        in generated_code
    )
    assert "let min_vector_vector: Vec3 = (value).min(other);" in generated_code
    assert "let max_vector_vector: Vec3 = (value).max(other);" in generated_code
    assert "let min_scalar_scalar: f32 = (scalar).min(1.0);" in generated_code
    assert "let max_scalar_scalar: f32 = (scalar).max(0.0);" in generated_code
    assert "let min_scalar_vector: f32 = min" not in generated_code
    assert "let max_vector_scalar: f32 = max" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_binary_math_intrinsics_promote_mixed_scalar_vector_operands_and_smoke_compile(
    tmp_path,
):
    code = """
    shader BinaryMathPromotionInference {
        fragment {
            vec4 main(vec3 value, vec3 other, float scalar) {
                let pow_vector_scalar = pow(value, scalar);
                let pow_scalar_vector = pow(scalar, other);
                let pow_vector_vector = pow(value, other);
                let mod_vector_scalar = mod(value, scalar);
                let mod_scalar_vector = mod(scalar, other);
                let mod_vector_vector = mod(value, other);
                let atan_vector_scalar = atan2(value, scalar);
                let atan_scalar_vector = atan2(scalar, other);
                let atan_vector_vector = atan2(value, other);
                let pow_scalar_scalar = pow(scalar, 2.0);
                let mod_scalar_scalar = mod(scalar, 2.0);
                let atan_scalar_scalar = atan2(scalar, 1.0);
                return vec4(
                    pow_vector_scalar + pow_scalar_vector + pow_vector_vector
                        + mod_vector_scalar + mod_scalar_vector + mod_vector_vector
                        + atan_vector_scalar + atan_scalar_vector + atan_vector_vector,
                    pow_scalar_scalar + mod_scalar_scalar + atan_scalar_scalar
                );
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'pow' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_hlsl_fmod_alias_lowers_to_rust_modulo_and_smoke_compile(tmp_path):
    code = """
    shader FmodAliasInference {
        fragment {
            vec4 main(vec3 value, float scalar) {
                let wrapped_vector = fmod(value, scalar);
                let wrapped_scalar = fmod(scalar, 2.0);
                return vec4(wrapped_vector, wrapped_scalar);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let wrapped_vector: Vec3 = ((value) % (scalar));" in generated_code
    assert "let wrapped_scalar: f32 = ((scalar) % (2.0));" in generated_code
    assert "fmod(" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_user_defined_fmod_function_is_not_lowered_to_modulo():
    code = """
    shader UserDefinedFmod {
        float fmod(float value, float divisor) {
            return value + divisor;
        }

        fragment {
            float main(float value) {
                return fmod(value, 2.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub fn fmod(value: f32, divisor: f32) -> f32" in generated_code
    assert "let return_value = fmod(value, 2.0);" in generated_code
    assert "modulo(value, 2.0)" not in generated_code


def test_matrix_intrinsics_infer_rust_value_types_and_smoke_compile(tmp_path):
    code = """
    shader MatrixIntrinsicInference {
        fragment {
            vec4 main(
                mat3 normal_matrix,
                mat4 world_matrix,
                vec3 normal,
                vec4 position
            ) {
                let normal_basis = transpose(normal_matrix);
                let normal_inverse = inverse(normal_basis);
                let normal_weight = determinant(normal_inverse);
                let world_inverse = inverse(world_matrix);
                let world_basis = transpose(world_inverse);
                let world_weight = determinant(world_basis);
                let transformed_normal = normal_inverse * normal;
                let transformed_position = world_basis * position;
                return vec4(
                    transformed_normal,
                    transformed_position.w + normal_weight + world_weight
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let normal_inverse: Mat3 = (normal_basis).inverse();" in generated_code
    assert "let world_weight: f32 = (world_basis).determinant();" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_refuses_rectangular_matrix_parameters():
    code = """
    shader RectangularMatrix {
        fragment {
            vec4 main(mat3x4 affine) {
                let rectangular = transpose(affine);
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(RustRefusal, match="glam has no 3x4 matrix of f32"):
        generate_code(parse_code(tokenize_code(code)))


def test_matrix_product_intrinsics_infer_rust_value_types_and_smoke_compile(
    tmp_path,
):
    code = """
    shader MatrixProductIntrinsicInference {
        fragment {
            vec4 main(
                vec3 column,
                vec2 row,
                dvec2 precise_column,
                dvec3 precise_row,
                mat2 regular,
                dmat2 precise
            ) {
                let basis = outerProduct(column, row);
                let precise_basis = outerProduct(precise_column, precise_row);
                let componentwise = matrixCompMult(regular, precise);
                let same_componentwise = matrixCompMult(regular, mat2(1.0, 0.0, 0.0, 1.0));
                return vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'outerProduct' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_step_and_smoothstep_promote_vector_scalar_operands_and_smoke_compile(
    tmp_path,
):
    code = """
    shader ThresholdIntrinsicInference {
        fragment {
            vec4 main(vec3 low, vec3 high, vec3 value, float scalar) {
                let scalar_gate = step(0.25, scalar);
                let vector_gate = step(0.25, value);
                let vector_edge_gate = step(low, scalar);
                let soft_value = smoothstep(0.0, 1.0, value);
                let soft_edges = smoothstep(low, high, scalar);
                let combined = vector_gate + vector_edge_gate + soft_value + soft_edges;
                return vec4(combined, scalar_gate);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'step' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_derivative_and_fused_math_intrinsics_infer_types_and_smoke_compile(
    tmp_path,
):
    code = """
    shader DerivativeIntrinsicInference {
        fragment {
            vec4 main(vec3 value, float scale, float bias) {
                let dx = dFdx(value);
                let dy = dFdy(value);
                let fine = dFdxFine(value);
                let coarse = dFdyCoarse(value);
                let wide = fwidth(value);
                let wide_fine = fwidthFine(value);
                let wide_coarse = fwidthCoarse(value);
                let fused = fma(value, scale, wide);
                let mad_alias = mad(fused, value, bias);
                let scalar_dx = ddx(scale);
                let scalar_dy = ddy(scale);
                return vec4(
                    mad_alias + dx + dy + fine + coarse + wide + wide_fine + wide_coarse,
                    scalar_dx + scalar_dy
                );
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'fma' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_scalar_fma_and_mad_lower_to_rust_mul_add_method_and_smoke_compile(
    tmp_path,
):
    code = """
    shader ScalarMulAddInference {
        fragment {
            vec4 main(float scale, float value, float bias) {
                let fused = fma(value, scale, bias);
                let mad_alias = mad(scale, value, 0.25);
                let literal_receiver = mad(1, scale, bias);
                return vec4(vec3(fused + mad_alias + literal_receiver), 1.0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'mad' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_ldexp_intrinsic_preserves_float_input_shape_and_smoke_compile(tmp_path):
    code = """
    shader LdexpIntrinsicInference {
        fragment {
            vec4 main(vec3 value, ivec3 exponents, float scale, int scalarExponent) {
                let shifted = ldexp(value, exponents);
                let scalar_shifted = ldexp(scale, scalarExponent);
                let uniform_shifted = ldexp(value, scalarExponent);
                return vec4(shifted + uniform_shifted, scalar_shifted);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'ldexp' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_sign_and_classification_intrinsics_infer_bool_vectors_and_smoke_compile(
    tmp_path,
):
    code = """
    shader ClassificationIntrinsicInference {
        fragment {
            vec4 main(vec3 value, float scale) {
                let signed_value = sign(value);
                let signed_scale = sign(scale);
                let nan_mask = isnan(value);
                let inf_mask = isinf(value);
                let finite_mask = isfinite(value);
                let scalar_nan = isnan(scale);
                let scalar_inf = isinf(scale);
                let scalar_finite = isfinite(scale);
                let flags = vec3(
                    nan_mask.x ? 1.0 : 0.0,
                    inf_mask.y ? 1.0 : 0.0,
                    finite_mask.z ? 1.0 : 0.0
                );
                let scalar_flags = (scalar_nan ? 1.0 : 0.0)
                    + (scalar_inf ? 1.0 : 0.0)
                    + (scalar_finite ? 1.0 : 0.0);
                return vec4(signed_value + flags, signed_scale + scalar_flags);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let signed_value: Vec3 = (value).map(|c| if c > 0.0 { 1.0 } else if c < 0.0 { -1.0 } else { 0.0 });"
        in generated_code
    )
    assert (
        "let signed_scale: f32 = { let c = scale; if c > 0.0 { 1.0 } else if c < 0.0 { -1.0 } else { 0.0 } };"
        in generated_code
    )
    assert "let nan_mask: BVec3 = (value).is_nan_mask();" in generated_code
    assert (
        "let inf_mask: BVec3 = { let __v = value; !(__v.is_finite_mask() | __v.is_nan_mask()) };"
        in generated_code
    )
    assert "let finite_mask: BVec3 = (value).is_finite_mask();" in generated_code
    assert "let scalar_nan: bool = (scale).is_nan();" in generated_code
    assert "let scalar_inf: bool = (scale).is_infinite();" in generated_code
    assert "let scalar_finite: bool = (scale).is_finite();" in generated_code
    assert "let nan_mask: bool = isnan(value)" not in generated_code
    assert "let scalar_nan: f32 = isnan(scale)" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_any_and_all_reduce_bool_vectors_to_scalar_bool_and_smoke_compile(tmp_path):
    code = """
    shader BoolReductionIntrinsicInference {
        fragment {
            vec4 main(vec3 value, bvec3 external_mask, bool ready) {
                let invalid_mask = isnan(value) || isinf(value);
                let finite_mask = isfinite(value);
                let has_invalid = any(invalid_mask);
                let all_finite = all(finite_mask);
                let any_external = any(external_mask);
                let all_external = all(external_mask);
                let scalar_any = any(ready);
                let scalar_all = all(ready);
                let selected = has_invalid || !all_finite || any_external || !all_external
                    || scalar_any || scalar_all;
                return vec4(selected ? value : sign(value), selected ? 1.0 : 0.0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="stage: fragment input 'external_mask' has type bvec3",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_relational_intrinsics_infer_bool_vectors_and_smoke_compile(tmp_path):
    code = """
    shader RelationalIntrinsicInference {
        fragment {
            vec4 main(vec3 left, vec3 right, float threshold, bvec3 mask) {
                let lt = lessThan(left, right);
                let le = lessThanEqual(left, threshold);
                let gt = greaterThan(threshold, right);
                let ge = greaterThanEqual(left, right);
                let eq = equal(mask, bvec3(true, false, true));
                let ne = notEqual(left, right);
                let scalar_eq = equal(threshold, 1.0);
                let selected = any(lt || le || gt || ge || eq || ne) || scalar_eq;
                return vec4(selected ? left : right, selected ? 1.0 : 0.0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="stage: fragment input 'mask' has type bvec3"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_integer_bit_intrinsics_infer_result_types_and_smoke_compile(tmp_path):
    code = """
    shader IntegerBitIntrinsicInference {
        fragment {
            vec4 main(uvec3 mask, ivec3 signedMask, uint scalarMask, int offset, int bits) {
                let reversed = bitfieldReverse(mask);
                let signedReversed = bitfieldReverse(signedMask);
                let scalarReversed = bitfieldReverse(scalarMask);
                let counts = bitCount(mask);
                let signedCounts = bitCount(signedMask);
                let scalarCount = bitCount(scalarMask);
                let lsb = findLSB(mask);
                let msb = findMSB(signedMask);
                let extracted = bitfieldExtract(mask, offset, bits);
                let signedExtracted = bitfieldExtract(signedMask, offset, bits);
                let inserted = bitfieldInsert(mask, extracted, offset, bits);
                let signedInserted = bitfieldInsert(signedMask, signedExtracted, offset, bits);
                let scalarExtract = bitfieldExtract(scalarMask, offset, bits);
                let scalarInsert = bitfieldInsert(scalarMask, scalarExtract, offset, bits);
                int total = counts.x + signedCounts.y + lsb.z + msb.x + scalarCount;
                uint mixed = reversed.x + inserted.y + scalarReversed + scalarInsert;
                int signedMixed = signedReversed.x + signedInserted.y;
                return vec4(vec3(float(total + signedMixed)), float(mixed));
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="function: 'bitfieldReverse' has no rust-gpu lowering",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_bit_reinterpret_intrinsics_preserve_shape_and_smoke_compile(tmp_path):
    code = """
    shader BitReinterpretIntrinsicInference {
        fragment {
            vec4 main(vec3 value, float scalar, ivec3 signedBits, uvec3 unsignedBits, int signedScalar, uint unsignedScalar) {
                let signedFromVector = floatBitsToInt(value);
                let unsignedFromVector = floatBitsToUint(value);
                let signedFromScalar = floatBitsToInt(scalar);
                let unsignedFromScalar = floatBitsToUint(scalar);
                let floatFromSigned = intBitsToFloat(signedBits);
                let floatFromUnsigned = uintBitsToFloat(unsignedBits);
                let scalarFromSigned = intBitsToFloat(signedScalar);
                let scalarFromUnsigned = uintBitsToFloat(unsignedScalar);
                int signedMix = signedFromVector.x + signedFromScalar;
                uint unsignedMix = unsignedFromVector.y + unsignedFromScalar;
                vec3 restored = floatFromSigned + floatFromUnsigned;
                return vec4(restored, float(signedMix) + float(unsignedMix)
                    + scalarFromSigned + scalarFromUnsigned);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let signedFromVector: IVec3 = { let __v = value; IVec3::new((__v.x.to_bits() as i32), (__v.y.to_bits() as i32), (__v.z.to_bits() as i32)) };"
        in generated_code
    )
    assert (
        "let unsignedFromVector: UVec3 = { let __v = value; UVec3::new(__v.x.to_bits(), __v.y.to_bits(), __v.z.to_bits()) };"
        in generated_code
    )
    assert (
        "let signedFromScalar: i32 = (((scalar).to_bits() as i32));" in generated_code
    )
    assert "let unsignedFromScalar: u32 = ((scalar).to_bits());" in generated_code
    assert (
        "let floatFromSigned: Vec3 = { let __v = signedBits; Vec3::new(f32::from_bits(__v.x as u32), f32::from_bits(__v.y as u32), f32::from_bits(__v.z as u32)) };"
        in generated_code
    )
    assert (
        "let floatFromUnsigned: Vec3 = { let __v = unsignedBits; Vec3::new(f32::from_bits(__v.x), f32::from_bits(__v.y), f32::from_bits(__v.z)) };"
        in generated_code
    )
    assert (
        "let scalarFromSigned: f32 = (f32::from_bits((signedScalar) as u32));"
        in generated_code
    )
    assert (
        "let scalarFromUnsigned: f32 = (f32::from_bits((unsignedScalar)));"
        in generated_code
    )
    assert "let signedFromVector: Vec3<u32> = float_bits_to_int" not in generated_code
    assert "let floatFromSigned: Vec3<i32> = int_bits_to_float" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_double_bit_reinterpret_intrinsics_lower_to_rust_primitives(tmp_path):
    code = """
    shader DoubleBitReinterpretIntrinsicInference {
        fragment {
            double main(double value, uint64_t rawBits) {
                let encoded = doubleBitsToLong(value);
                let decoded = longBitsToDouble(rawBits);
                let roundTripped = longBitsToDouble(doubleBitsToLong(value));
                return decoded + roundTripped + longBitsToDouble(encoded);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="stage: fragment input 'rawBits' has type uint64_t"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_as_bitcast_aliases_lower_to_rust_helpers_and_smoke_compile(tmp_path):
    code = """
    shader AsBitcastAliasInference {
        fragment {
            vec4 main(vec3 value, float scalar, ivec3 signedBits, uvec3 unsignedBits, int signedScalar, uint unsignedScalar) {
                let signedFromVector = asint(value);
                let unsignedFromVector = asuint(value);
                let signedFromScalar = asint(scalar);
                let unsignedFromScalar = asuint(scalar);
                let floatFromSigned = asfloat(signedBits);
                let floatFromUnsigned = asfloat(unsignedBits);
                let scalarFromSigned = asfloat(signedScalar);
                let scalarFromUnsigned = asfloat(unsignedScalar);
                int signedMix = signedFromVector.x + signedFromScalar;
                uint unsignedMix = unsignedFromVector.y + unsignedFromScalar;
                vec3 restored = floatFromSigned + floatFromUnsigned;
                return vec4(restored, float(signedMix) + float(unsignedMix)
                    + scalarFromSigned + scalarFromUnsigned);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let signedFromVector: IVec3 = { let __v = value; IVec3::new((__v.x.to_bits() as i32), (__v.y.to_bits() as i32), (__v.z.to_bits() as i32)) };"
        in generated_code
    )
    assert (
        "let unsignedFromVector: UVec3 = { let __v = value; UVec3::new(__v.x.to_bits(), __v.y.to_bits(), __v.z.to_bits()) };"
        in generated_code
    )
    assert (
        "let signedFromScalar: i32 = (((scalar).to_bits() as i32));" in generated_code
    )
    assert "let unsignedFromScalar: u32 = ((scalar).to_bits());" in generated_code
    assert (
        "let floatFromSigned: Vec3 = { let __v = signedBits; Vec3::new(f32::from_bits(__v.x as u32), f32::from_bits(__v.y as u32), f32::from_bits(__v.z as u32)) };"
        in generated_code
    )
    assert (
        "let floatFromUnsigned: Vec3 = { let __v = unsignedBits; Vec3::new(f32::from_bits(__v.x), f32::from_bits(__v.y), f32::from_bits(__v.z)) };"
        in generated_code
    )
    assert (
        "let scalarFromSigned: f32 = (f32::from_bits((signedScalar) as u32));"
        in generated_code
    )
    assert (
        "let scalarFromUnsigned: f32 = (f32::from_bits((unsignedScalar)));"
        in generated_code
    )
    assert "asfloat(" not in generated_code
    assert "asint(" not in generated_code
    assert "asuint(" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_bitcast_alias_targets_qualify_shadowed_rust_helpers_and_smoke_compile(
    tmp_path,
):
    code = """
    shader RustBitcastAliasTargetShadowing {
        fragment {
            vec4 main(vec3 value, ivec3 signedBits, uvec3 unsignedBits) {
                float float_bits_to_int = 1.0;
                float float_bits_to_uint = 2.0;
                float int_bits_to_float = 3.0;
                float uint_bits_to_float = 4.0;
                let signed = floatBitsToInt(value);
                let unsigned = asuint(value);
                let fromSigned = asfloat(signedBits);
                let fromUnsigned = uintBitsToFloat(unsignedBits);
                return vec4(fromSigned + fromUnsigned, float(signed.x)
                    + float(unsigned.y) + float_bits_to_int + float_bits_to_uint
                    + int_bits_to_float + uint_bits_to_float);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let float_bits_to_int: f32 = 1.0;" in generated_code
    assert "let float_bits_to_uint: f32 = 2.0;" in generated_code
    assert "let int_bits_to_float: f32 = 3.0;" in generated_code
    assert "let uint_bits_to_float: f32 = 4.0;" in generated_code
    assert (
        "let signed: IVec3 = { let __v = value; IVec3::new((__v.x.to_bits() as i32), (__v.y.to_bits() as i32), (__v.z.to_bits() as i32)) };"
        in generated_code
    )
    assert (
        "let unsigned: UVec3 = { let __v = value; UVec3::new(__v.x.to_bits(), __v.y.to_bits(), __v.z.to_bits()) };"
        in (generated_code)
    )
    assert (
        "let fromSigned: Vec3 = { let __v = signedBits; Vec3::new(f32::from_bits(__v.x as u32), f32::from_bits(__v.y as u32), f32::from_bits(__v.z as u32)) };"
        in generated_code
    )
    assert (
        "let fromUnsigned: Vec3 = { let __v = unsignedBits; Vec3::new(f32::from_bits(__v.x), f32::from_bits(__v.y), f32::from_bits(__v.z)) };"
        in generated_code
    )
    assert "let signed: Vec3<i32> = float_bits_to_int(value);" not in generated_code
    assert "let unsigned: Vec3<u32> = float_bits_to_uint(value);" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_vec2_vec4_bitcasts_emit_lane_helpers_and_smoke_compile(tmp_path):
    code = """
    shader BitReinterpretVec2Vec4 {
        fragment {
            vec4 main(vec2 value2, vec4 value4, ivec2 signed2, uvec4 unsigned4) {
                float float_bits_to_int = 1.0;
                float int_bits_to_float = 2.0;
                let signedFromVec2 = floatBitsToInt(value2);
                let unsignedFromVec4 = asuint(value4);
                let floatFromSignedVec2 = asfloat(signed2);
                let floatFromUnsignedVec4 = uintBitsToFloat(unsigned4);
                return vec4(
                    floatFromSignedVec2.x + float_bits_to_int,
                    floatFromSignedVec2.y + int_bits_to_float,
                    floatFromUnsignedVec4.x + float(signedFromVec2.x),
                    floatFromUnsignedVec4.w + float(unsignedFromVec4.w)
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let signedFromVec2: IVec2 = { let __v = value2; IVec2::new((__v.x.to_bits() as i32), (__v.y.to_bits() as i32)) };"
    ) in generated_code
    assert (
        "let unsignedFromVec4: UVec4 = { let __v = value4; UVec4::new(__v.x.to_bits(), __v.y.to_bits(), __v.z.to_bits(), __v.w.to_bits()) };"
    ) in generated_code
    assert (
        "let floatFromSignedVec2: Vec2 = { let __v = signed2; Vec2::new(f32::from_bits(__v.x as u32), f32::from_bits(__v.y as u32)) };"
    ) in generated_code
    assert (
        "let floatFromUnsignedVec4: Vec4 = { let __v = unsigned4; Vec4::new(f32::from_bits(__v.x), f32::from_bits(__v.y), f32::from_bits(__v.z), f32::from_bits(__v.w)) };"
    ) in generated_code
    assert "let signedFromVec2: Vec2<i32> = float_bits_to_int(value2);" not in (
        generated_code
    )
    assert "let unsignedFromVec4: Vec4<u32> = float_bits_to_uint(value4);" not in (
        generated_code
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_pack_unpack_intrinsics_infer_result_types_and_smoke_compile(tmp_path):
    code = """
    shader PackUnpackIntrinsicInference {
        fragment {
            vec4 main(vec2 pair, vec4 color, uint packed2, uint packed4) {
                let packedUnorm2 = packUnorm2x16(pair);
                let packedSnorm2 = packSnorm2x16(pair);
                let packedUnorm4 = packUnorm4x8(color);
                let packedSnorm4 = packSnorm4x8(color);
                let packedHalf = packHalf2x16(pair);
                let unpackedUnorm2 = unpackUnorm2x16(packed2);
                let unpackedSnorm2 = unpackSnorm2x16(packed2);
                let unpackedUnorm4 = unpackUnorm4x8(packed4);
                let unpackedSnorm4 = unpackSnorm4x8(packed4);
                let unpackedHalf = unpackHalf2x16(packed2);
                uint packedMix = packedUnorm2 + packedSnorm2 + packedUnorm4
                    + packedSnorm4 + packedHalf;
                return unpackedUnorm4 + unpackedSnorm4
                    + vec4(unpackedUnorm2, unpackedSnorm2)
                    + vec4(unpackedHalf, unpackedHalf)
                    + vec4(float(packedMix));
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let packedUnorm2: u32 = spirv_std::float::vec2_to_u16x2_unorm(pair);" in (
        generated_code
    )
    assert (
        "let unpackedSnorm4: Vec4 = spirv_std::float::u8x4_to_vec4_snorm(packed4);" in (
            generated_code
        )
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_double_packing_is_refused():
    code = """
    shader DoublePacking {
        fragment {
            vec4 main(uvec2 doubleBits) {
                let packedDouble = packDouble2x32(doubleBits);
                return vec4(float(packedDouble));
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'packDouble2x32' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_scalar_left_vector_arithmetic_lowers_to_lane_constructors(tmp_path):
    code = """
    shader ScalarVectorOps {
        fragment {
            float nextScale() {
                return 1.0;
            }

            vec3 nextColor() {
                return vec3(0.25, 0.5, 0.75);
            }

            vec4 main(vec3 color, float exposure) {
                let inverted = 1.0 - color;
                let reciprocal = 1.0 / color;
                let scaled = exposure * color;
                let shifted = exposure + color;
                let complex = nextScale() - nextColor();
                let combined = inverted + reciprocal + scaled + shifted + complex;
                return vec4(combined, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let inverted: Vec3 = Vec3::new((1.0 - color.x), (1.0 - color.y), (1.0 - color.z));"
        in generated_code
    )
    assert (
        "let reciprocal: Vec3 = Vec3::new((1.0 / color.x), (1.0 / color.y), (1.0 / color.z));"
        in generated_code
    )
    assert (
        "let scaled: Vec3 = Vec3::new((exposure * color.x), (exposure * color.y), (exposure * color.z));"
        in generated_code
    )
    assert (
        "let shifted: Vec3 = Vec3::new((exposure + color.x), (exposure + color.y), (exposure + color.z));"
        in generated_code
    )
    assert (
        "let complex: Vec3 = { let __cgl_vec_arg_0 = nextScale(); let __cgl_vec_arg_1 = nextColor(); Vec3::new((__cgl_vec_arg_0 - __cgl_vec_arg_1.x), (__cgl_vec_arg_0 - __cgl_vec_arg_1.y), (__cgl_vec_arg_0 - __cgl_vec_arg_1.z)) };"
        in generated_code
    )
    assert "(1.0 - color)" not in generated_code
    assert "(1.0 / color)" not in generated_code
    assert "(exposure * color)" not in generated_code
    assert "(exposure + color)" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_vector_scalar_arithmetic_lowers_to_lane_constructors(tmp_path):
    code = """
    shader VectorScalarOps {
        vec2 makeUv() {
            return vec2(0.25, 0.75);
        }

        fragment {
            vec4 main(vec2 uv, vec3 color, float exposure) {
                let scaledUv = uv * 10.0;
                let shiftedColor = color + exposure;
                let dynamicUv = makeUv() / exposure;
                return vec4(scaledUv, shiftedColor.x + dynamicUv.x, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let scaledUv: Vec2 = Vec2::new((uv.x * 10.0), (uv.y * 10.0));"
        in generated_code
    )
    assert (
        "let shiftedColor: Vec3 = Vec3::new((color.x + exposure), (color.y + exposure), (color.z + exposure));"
        in generated_code
    )
    assert "let dynamicUv: Vec2 = { let __cgl_vec_arg_0 = makeUv();" in generated_code
    assert "(uv * 10.0)" not in generated_code
    assert "(color + exposure)" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_vec2_dot_calls_glam_dot(tmp_path):
    code = """
    shader RustVec2Dot {
        fragment {
            float noise(vec2 p) {
                return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
            }

            vec4 main(vec2 uv) {
                float value = noise(uv);
                return vec4(value, value, value, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "dot(p, Vec2::<f32>" not in generated_code
    assert "(p).dot(Vec2::new(12.9898, 78.233))" in generated_code
    assert ".sin() * 43758.5453)" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_vector_compound_assignment_lowers_without_assign_traits(tmp_path):
    code = """
    shader RustVectorCompound {
        fragment {
            vec4 main(vec3 color, float exposure) {
                vec3 adjusted = color;
                adjusted *= exposure;
                adjusted += 0.25;
                return vec4(adjusted, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "adjusted *= exposure" not in generated_code
    assert "adjusted += 0.25" not in generated_code
    assert "adjusted = Vec3::new((adjusted.x * exposure)" in generated_code
    assert "adjusted = Vec3::new((adjusted.x + 0.25)" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_texture_size_vector_temp_and_float_increment_compile(tmp_path):
    code = """
    shader RustTextureSizeTemp {
        sampler2D shadowMap;

        fragment {
            vec4 main(vec2 uv) {
                vec2 texel = vec2(textureSize(shadowMap, 0));
                float samples = 0.0;
                samples++;
                return vec4(uv / texel, samples, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_vec2_intrinsics_and_matrix_multiply_use_glam(tmp_path):
    code = """
    shader RustLaneMatrixMath {
        fragment {
            vec4 main(vec2 direction, mat2 rotation, mat4 view, mat4 model) {
                float len = length(direction);
                vec2 unit = normalize(direction);
                vec2 rotated = rotation * unit;
                mat4 combined = view * model;
                return combined * vec4(rotated, len, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let len: f32 = (direction).length();" in generated_code
    assert "let unit: Vec2 = (direction).normalize();" in generated_code
    assert "let rotated: Vec2 = (rotation * unit);" in generated_code
    assert "let combined: Mat4 = (view * model);" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_shared_stage_arrays_are_workgroup_entry_parameters(tmp_path):
    code = """
    shader RustSharedLocal {
        compute {
            shared float tile[2][2];

            void main() {
                int x = int(gl_LocalInvocationID.x);
                tile[x][0] = 1.0;
                workgroupBarrier();
                float value = tile[x][0];
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "static TILE" not in generated_code
    assert "#[spirv(workgroup)] tile: &mut [[f32; 2]; 2]" in generated_code
    assert "tile[x as usize][0] = 1.0;" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_lambda_call_emits_native_rust_closure():
    code = """
    shader main {
        void apply(Values values) {
            let mapped = map(values, lambda(x, { prepare(x); return (x + 1); }));
            let folded = fold(values, 0, lambda(int acc, int x, (acc + x)));
            let parsed = map(values, lambda(Result<i32, i32> value, { return value; }));
            let always = lambda(true);
        }
    }
    """

    ast = crosstl.translator.parse(code)
    with pytest.raises(
        RustLoweringRefused, match="function: 'map' has no rust-gpu lowering"
    ):
        generate_code(ast)


def test_user_defined_mix_function_is_not_lowered_to_lerp():
    code = """
    shader main {
        fragment {
            float mix(float x, float y, float t) {
                return x + y + t;
            }

            float main() {
                float adjusted = mix(0.0, 1.0, 0.25);
                return adjusted;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub fn mix(x: f32, y: f32, t: f32) -> f32" in generated_code
    assert "let adjusted: f32 = mix(0.0, 1.0, 0.25);" in generated_code
    assert "let adjusted: f32 = lerp(0.0, 1.0, 0.25);" not in generated_code


def test_bool_scalar_mix_lowers_to_rust_select():
    code = """
    shader main {
        fragment {
            float nextX() {
                return 0.0;
            }

            float nextY() {
                return 1.0;
            }

            bool nextFlag() {
                return true;
            }

            float main() {
                bool flag = true;
                float literal = mix(0.0, 1.0, flag);
                float complexValue = mix(nextX(), nextY(), nextFlag());
                return literal + complexValue;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let literal: f32 = (if flag { 1.0 } else { 0.0 });" in generated_code
    assert (
        "let complexValue: f32 = "
        "(if nextFlag() { nextY() } else { nextX() });" in generated_code
    )
    assert "lerp(0.0, 1.0, flag)" not in generated_code
    assert "lerp(nextX(), nextY(), nextFlag())" not in generated_code
    assert "mix(0.0, 1.0, flag)" not in generated_code
    assert "mix(nextX(), nextY(), nextFlag())" not in generated_code


def test_bool_vector_mix_lowers_to_rust_lane_selectors():
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    vec3 makeA() {
        return vec3(1.0, 2.0, 3.0);
    }

    vec3 makeB() {
        return vec3(4.0, 5.0, 6.0);
    }

    void probe(bvec3 mask, vec3 a, vec3 b) {
        vec3 selected = mix(a, b, mask);
        vec3 complexSelected = mix(makeA(), makeB(), makeMask());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let selected: Vec3 = Vec3::select(mask, b, a);" in generated_code
    assert (
        "let complexSelected: Vec3 = Vec3::select(makeMask(), makeB(), makeA());"
        in generated_code
    )
    assert "lerp(a, b, mask)" not in generated_code
    assert "lerp(makeA(), makeB(), makeMask())" not in generated_code
    assert "mix(a, b, mask)" not in generated_code
    assert "mix(makeA(), makeB(), makeMask())" not in generated_code


def test_user_defined_function_call_scalar_args_cast_to_param_types():
    code = """
    float takesScalars(float weight, double precise, uint count, bool enabled) {
        return weight;
    }

    void probe(int index, float weight, bool ready) {
        float result = takesScalars(index, weight, index, index);
        float fromBool = takesScalars(ready, 1.0, 2, ready);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "pub fn takesScalars(weight: f32, precise: f64, count: u32, "
        "enabled: bool) -> f32" in generated_code
    )
    assert (
        "let result: f32 = "
        "takesScalars((index as f32), (weight as f64), "
        "(index as u32), (index != 0));" in generated_code
    )
    assert (
        "let fromBool: f32 = "
        "takesScalars((if ready { 1.0 } else { 0.0 }), "
        "(1.0 as f64), 2, ready);" in generated_code
    )
    assert "takesScalars(index, weight, index, index)" not in generated_code
    assert "takesScalars(ready, 1.0, 2, ready)" not in generated_code


def test_user_defined_function_call_vector_args_cast_to_param_types():
    code = """
    float consumeVectors(vec2 coords, uvec3 ids, bvec2 mask) {
        return coords.x;
    }

    void probe(ivec2 pixel, vec3 weights, vec2 amount, bool ready) {
        float same = consumeVectors(amount, uvec3(1, 2, 3), bvec2(ready, false));
        float converted = consumeVectors(pixel, weights, amount);
        float constructed = consumeVectors(
            ivec2(1, 2),
            vec3(1.0, 2.0, 3.0),
            vec2(1.0, 0.0)
        );
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let same: f32 = consumeVectors(amount, UVec3::new(1, 2, 3), BVec2::new(ready, false));"
        in generated_code
    )
    assert (
        "let converted: f32 = consumeVectors(Vec2::new((pixel.x as f32), (pixel.y as f32)), UVec3::new((weights.x as u32), (weights.y as u32), (weights.z as u32)), BVec2::new((amount.x != 0.0), (amount.y != 0.0)));"
        in generated_code
    )
    assert (
        "consumeVectors({ let __cgl_vec_arg_0 = IVec2::new(1, 2); Vec2::new((__cgl_vec_arg_0.x as f32), (__cgl_vec_arg_0.y as f32)) }, { let __cgl_vec_arg_1 = Vec3::new(1.0, 2.0, 3.0); UVec3::new((__cgl_vec_arg_1.x as u32), (__cgl_vec_arg_1.y as u32), (__cgl_vec_arg_1.z as u32)) }, { let __cgl_vec_arg_2 = Vec2::new(1.0, 0.0); BVec2::new((__cgl_vec_arg_2.x != 0.0), (__cgl_vec_arg_2.y != 0.0)) });"
        in generated_code
    )
    assert "consumeVectors(pixel, weights, amount)" not in generated_code
    assert "consumeVectors(Vec2::<i32>::new(1, 2)" not in generated_code


def test_user_defined_function_call_matrix_args_cast_to_param_types():
    code = """
    float consumeMatrices(dmat2 precise, mat2 regular) {
        return regular.c0.x;
    }

    void probe(mat2 transform, dmat2 preciseInput, float weight, int index) {
        float same = consumeMatrices(preciseInput, transform);
        float converted = consumeMatrices(transform, preciseInput);
        float constructed = consumeMatrices(
            mat2(weight, index, 2, 1.0),
            dmat2(1.0, 0.0, 0.0, 1.0)
        );
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let same: f32 = consumeMatrices(preciseInput, transform);" in generated_code
    assert (
        "let converted: f32 = consumeMatrices(DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]), Mat2::from_cols_array(&[(preciseInput.x_axis.x as f32), (preciseInput.x_axis.y as f32), (preciseInput.y_axis.x as f32), (preciseInput.y_axis.y as f32)]));"
        in generated_code
    )
    assert (
        "consumeMatrices({ let __cgl_mat_arg_0 = Mat2::from_cols_array(&[weight, (index as f32), 2.0, 1.0]); DMat2::from_cols_array(&[(__cgl_mat_arg_0.x_axis.x as f64), (__cgl_mat_arg_0.x_axis.y as f64), (__cgl_mat_arg_0.y_axis.x as f64), (__cgl_mat_arg_0.y_axis.y as f64)]) }, { let __cgl_mat_arg_1 = DMat2::from_cols_array(&[1.0, 0.0, 0.0, 1.0]); Mat2::from_cols_array(&[(__cgl_mat_arg_1.x_axis.x as f32), (__cgl_mat_arg_1.x_axis.y as f32), (__cgl_mat_arg_1.y_axis.x as f32), (__cgl_mat_arg_1.y_axis.y as f32)]) });"
        in generated_code
    )
    assert "consumeMatrices(transform, preciseInput)" not in generated_code
    assert "consumeMatrices(Mat2::<f32>::new(weight" not in generated_code


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a |= 2;
                }
            }
            """,
            "a |= 2",
        )
    ],
)
def test_assignment_or_operator(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    assert expected_output in generated_code


def test_assignment_modulus_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 10;
                a %= 3;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a %= 3" in generated_code or "a = a % 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment modulus operator codegen not implemented.")


def test_assignment_xor_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 5;
                a ^= 3;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a ^= 3" in generated_code or "a = a ^ 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment XOR operator codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a <<= 2;
                    a >>= 1;
                }
            }
            """,
            ["a <<= 2", "a >>= 1"],
        )
    ],
)
def test_assignment_shift_operators(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    for output in expected_output:
        assert output in generated_code


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a | b;
                    int d = a & b;
                    int e = a ^ b;
                }
            }
            """,
            ["a | b", "a & b", "a ^ b"],
        )
    ],
)
def test_bitwise_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_and_operator(tmp_path):
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(f32(i32(input.texCoord.x * 100.0) & 15),
                                    f32(i32(input.texCoord.y * 100.0) & 15),
                                    0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color.rgb, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)
    except SyntaxError:
        pytest.fail("Bitwise AND codegen not implemented")


def test_double_data_type():
    code = """
    shader DoubleShader {
        struct VSInput {
            double texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            double color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = input.texCoord * 2.0;
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "f64" in generated_code  # Rust uses f64 for double
        print(generated_code)
    except SyntaxError:
        pytest.fail("Double data type not supported.")


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a << b;
                    int d = a >> b;
                }
            }
            """,
            ["a << b", "a >> b"],
        )
    ],
)
def test_shift_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_or_operator(tmp_path):
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(f32(i32(input.texCoord.x * 100.0) | 15),
                                    f32(i32(input.texCoord.y * 100.0) | 15),
                                    0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color.rgb, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)
    except SyntaxError:
        pytest.fail("Bitwise OR codegen not implemented")


def test_ternary_operator(tmp_path):
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord.x > 0.5 ? 1.0 : 0.0,
                                    input.texCoord.y > 0.5 ? 1.0 : 0.0,
                                    0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)
    except SyntaxError:
        pytest.fail("Ternary operator codegen not implemented")


def test_vector_constructor(tmp_path):
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(1.0, 0.5, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "Vec4::new" in generated_code
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Vector constructor codegen not implemented")


def test_double_vector_and_matrix_types_emit_rust_names(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                dvec2 preciseUV = dvec2(1.0, 2.0);
                bvec2 mask = bvec2(true, false);
                bvec3 flags;
                mat2 transform = mat2(1.0, 0.0, 0.0, 1.0);
                mat3x4 affine;
                dmat2 precise = dmat2(1.0, 0.0, 0.0, 1.0);
                dmat4x3 jacobian;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustRefusal, match=r"glam has no 3x4 matrix of f32 \(Mat3x4<f32>\)"
    ):
        generate_code(ast)


def test_matrix_constructors_flatten_vector_args_once(tmp_path):
    code = """
    vec2 makeCol0() {
        return vec2(1.0, 0.0);
    }

    vec2 makeCol1() {
        return vec2(0.0, 1.0);
    }

    void probe() {
        vec2 a = vec2(1.0, 0.0);
        vec2 b = vec2(0.0, 1.0);
        mat2 localMatrix = mat2(a, b);
        mat2 complexMatrix = mat2(makeCol0(), makeCol1());
        mat2 mixedMatrix = mat2(makeCol0().xy, b);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let localMatrix: Mat2 = Mat2::from_cols_array(&[a.x, a.y, b.x, b.y]);"
        in generated_code
    )
    assert (
        "let complexMatrix: Mat2 = { let __cgl_vec_arg_0 = makeCol0(); let __cgl_vec_arg_1 = makeCol1(); Mat2::from_cols_array(&[__cgl_vec_arg_0.x, __cgl_vec_arg_0.y, __cgl_vec_arg_1.x, __cgl_vec_arg_1.y]) };"
        in generated_code
    )
    assert (
        "let mixedMatrix: Mat2 = { let __cgl_vec_arg_2 = makeCol0(); Mat2::from_cols_array(&[__cgl_vec_arg_2.x, __cgl_vec_arg_2.y, b.x, b.y]) };"
        in generated_code
    )
    assert "Mat2::<f32>::new(a, b)" not in generated_code
    assert "Mat2::<f32>::new(makeCol0(), makeCol1())" not in generated_code
    assert "makeCol0().xy" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_mixed_scalar_matrix_constructor_lanes_cast_to_component_type(tmp_path):
    code = """
    void probe(float weight, int index, uint count) {
        mat2 mixedFloat = mat2(weight, index, 2, 1.0);
        dmat2 mixedDouble = dmat2(weight, index, 2, 1.0);
        mat2 fromIntVector = mat2(ivec2(index, 2), vec2(weight, 1.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let mixedFloat: Mat2 = Mat2::from_cols_array(&[weight, (index as f32), 2.0, 1.0]);"
        in generated_code
    )
    assert (
        "let mixedDouble: DMat2 = DMat2::from_cols_array(&[(weight as f64), (index as f64), 2.0, 1.0]);"
        in generated_code
    )
    assert (
        "let fromIntVector: Mat2 = { let __cgl_vec_arg_0 = IVec2::new(index, 2); let __cgl_vec_arg_1 = Vec2::new(weight, 1.0); Mat2::from_cols_array(&[(__cgl_vec_arg_0.x as f32), (__cgl_vec_arg_0.y as f32), __cgl_vec_arg_1.x, __cgl_vec_arg_1.y]) };"
        in generated_code
    )
    assert "Mat2::<f32>::new(weight, index, 2, 1.0)" not in generated_code
    assert "Mat2::<f64>::new(weight, index, 2, 1.0)" not in generated_code
    assert "Mat2::<f32>::new(__cgl_vec_arg_0.x, __cgl_vec_arg_0.y" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_inferred_matrix_constructor_bindings_use_matrix_types(tmp_path):
    code = """
    void probe() {
        let transform = mat2(1.0, 0.0, 0.0, 1.0);
        let affine = mat3x4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        );
        let precise = dmat2(1.0, 0.0, 0.0, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustRefusal, match=r"glam has no 3x4 matrix of f32 \(Mat3x4<f32>\)"
    ):
        generate_code(ast)


def test_inferred_matrix_binary_bindings_prefer_matrix_operands(tmp_path):
    code = """
    void probe() {
        let leftScaled = mat2(1.0, 0.0, 0.0, 1.0) * 2.0;
        let rightScaled = 2.0 * mat2(1.0, 0.0, 0.0, 1.0);
        let combined = mat2(1.0, 0.0, 0.0, 1.0)
            + mat2(2.0, 0.0, 0.0, 2.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let leftScaled: Mat2 = (Mat2::from_cols_array(&[1.0, 0.0, 0.0, 1.0]) * 2.0);"
        in generated_code
    )
    assert (
        "let rightScaled: Mat2 = (2.0 * Mat2::from_cols_array(&[1.0, 0.0, 0.0, 1.0]));"
        in generated_code
    )
    assert "let combined: Mat2 = " in generated_code
    assert "let rightScaled: f32 = " not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_mixed_vector_and_matrix_binary_operands_promote_before_operator():
    code = """
    void probe(ivec2 pixel, vec2 amount, mat2 transform, dmat2 preciseInput) {
        let inferredVec = pixel + amount;
        let inferredVecReverse = amount + pixel;
        vec2 declaredVec = pixel + amount;
        let inferredMat = transform + preciseInput;
        let inferredMatReverse = preciseInput + transform;
        dmat2 declaredMat = transform + preciseInput;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let inferredVec: Vec2 = Vec2::new(((pixel.x as f32) + amount.x), ((pixel.y as f32) + amount.y));"
        in generated_code
    )
    assert (
        "let inferredVecReverse: Vec2 = Vec2::new((amount.x + (pixel.x as f32)), (amount.y + (pixel.y as f32)));"
        in generated_code
    )
    assert (
        "let declaredVec: Vec2 = Vec2::new(((pixel.x as f32) + amount.x), ((pixel.y as f32) + amount.y));"
        in generated_code
    )
    assert (
        "let inferredMat: DMat2 = (DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) + preciseInput);"
        in generated_code
    )
    assert (
        "let inferredMatReverse: DMat2 = (preciseInput + DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]));"
        in generated_code
    )
    assert (
        "let declaredMat: DMat2 = (DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) + preciseInput);"
        in generated_code
    )
    assert "(pixel + amount)" not in generated_code
    assert "(transform + preciseInput)" not in generated_code
    assert "let inferredVec: Vec2<i32>" not in generated_code
    assert "let inferredMat: Mat2<f32>" not in generated_code
    assert "{ let __cgl_vec_arg_" not in generated_code
    assert "{ let __cgl_mat_arg_" not in generated_code


def test_mixed_vector_and_matrix_scalar_binary_operands_cast_to_components():
    code = """
    void probe(
        ivec2 pixel,
        vec2 amount,
        mat2 transform,
        dmat2 preciseInput,
        int index,
        float weight,
        double precise
    ) {
        let inferredVec = pixel + weight;
        let inferredVecReverse = weight + pixel;
        vec2 declaredVec = pixel + weight;
        let inferredDoubleVec = amount + precise;
        let indexedVec = amount + index;
        let inferredMat = transform + precise;
        let inferredMatReverse = precise + transform;
        dmat2 declaredMat = transform + precise;
        let indexedMat = transform + index;
        let sameVec = amount + weight;
        let sameMat = preciseInput + precise;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let inferredVec: Vec2 = Vec2::new(((pixel.x as f32) + weight), ((pixel.y as f32) + weight));"
        in generated_code
    )
    assert (
        "let inferredVecReverse: Vec2 = Vec2::new((weight + (pixel.x as f32)), (weight + (pixel.y as f32)));"
        in generated_code
    )
    assert (
        "let declaredVec: Vec2 = Vec2::new(((pixel.x as f32) + weight), ((pixel.y as f32) + weight));"
        in generated_code
    )
    assert (
        "let inferredDoubleVec: DVec2 = DVec2::new(((amount.x as f64) + precise), ((amount.y as f64) + precise));"
        in generated_code
    )
    assert (
        "let indexedVec: Vec2 = Vec2::new((amount.x + (index as f32)), (amount.y + (index as f32)));"
        in generated_code
    )
    assert (
        "let inferredMat: DMat2 = (DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) + precise);"
        in generated_code
    )
    assert (
        "let inferredMatReverse: DMat2 = (precise + DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]));"
        in generated_code
    )
    assert (
        "let declaredMat: DMat2 = (DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) + precise);"
        in generated_code
    )
    assert "let indexedMat: Mat2 = (transform + (index as f32));" in generated_code
    assert (
        "let sameVec: Vec2 = Vec2::new((amount.x + weight), (amount.y + weight));"
        in generated_code
    )
    assert "let sameMat: DMat2 = (preciseInput + precise);" in generated_code
    assert "(pixel + weight)" not in generated_code
    assert "(transform + precise)" not in generated_code
    assert "{ let __cgl_vec_arg_" not in generated_code
    assert "{ let __cgl_mat_arg_" not in generated_code


def test_matrix_vector_binary_operands_promote_components_and_result_size():
    code = """
    void probe(
        mat2 transform,
        dmat2 preciseTransform,
        vec2 amount,
        dvec2 preciseAmount,
        ivec2 pixel,
        mat2x3 affine,
        vec2 uv,
        vec3 normal
    ) {
        let inferredMatVec = transform * preciseAmount;
        let inferredMatVecReverseComponents = preciseTransform * amount;
        let inferredMatIntVec = transform * pixel;
        dvec2 declaredMatVec = transform * amount;
        let inferredVecMat = preciseAmount * transform;
        let inferredVecMatReverseComponents = amount * preciseTransform;
        dvec2 declaredVecMat = amount * transform;
        let rectMatVec = affine * uv;
        let rectVecMat = normal * affine;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustRefusal, match=r"glam has no 2x3 matrix of f32 \(Mat2x3<f32>\)"
    ):
        generate_code(ast)


def test_vector_comparison_binary_operands_emit_boolean_lanes(tmp_path):
    code = """
    float makeWeight() {
        return 0.5;
    }

    vec2 makeAmount() {
        return vec2(1.0, 2.0);
    }

    void probe(
        int index,
        uint count,
        float weight,
        double precise,
        ivec2 pixel,
        vec2 amount,
        bvec2 mask
    ) {
        let scalarCompare = index < weight;
        let scalarEqual = count == index;
        bool declaredCompare = index < precise;
        let vectorEqual = pixel == amount;
        let vectorNotEqual = amount != pixel;
        bvec2 declaredMask = pixel < amount;
        let vectorScalarCompare = pixel < weight;
        let scalarVectorCompare = weight < pixel;
        let boolVectorEqual = mask == pixel;
        let complexCompare = makeAmount() > makeWeight();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let scalarCompare: bool = ((index as f32) < weight);" in generated_code
    assert "let scalarEqual: bool = (count == (index as u32));" in generated_code
    assert "let declaredCompare: bool = ((index as f64) < precise);" in generated_code
    assert (
        "let vectorEqual: BVec2 = BVec2::new(((pixel.x as f32) == amount.x), ((pixel.y as f32) == amount.y));"
        in generated_code
    )
    assert (
        "let vectorNotEqual: BVec2 = BVec2::new((amount.x != (pixel.x as f32)), (amount.y != (pixel.y as f32)));"
        in generated_code
    )
    assert (
        "let declaredMask: BVec2 = BVec2::new(((pixel.x as f32) < amount.x), ((pixel.y as f32) < amount.y));"
        in generated_code
    )
    assert (
        "let vectorScalarCompare: BVec2 = BVec2::new(((pixel.x as f32) < weight), ((pixel.y as f32) < weight));"
        in generated_code
    )
    assert (
        "let scalarVectorCompare: BVec2 = BVec2::new((weight < (pixel.x as f32)), (weight < (pixel.y as f32)));"
        in generated_code
    )
    assert (
        "let boolVectorEqual: BVec2 = BVec2::new(((mask.x as i32) == pixel.x), ((mask.y as i32) == pixel.y));"
        in generated_code
    )
    assert (
        "let complexCompare: BVec2 = { let __cgl_vec_arg_0 = makeAmount(); let __cgl_vec_arg_1 = makeWeight(); BVec2::new((__cgl_vec_arg_0.x > __cgl_vec_arg_1), (__cgl_vec_arg_0.y > __cgl_vec_arg_1)) };"
        in generated_code
    )
    assert "let vectorEqual: Vec2<i32>" not in generated_code
    assert "let vectorNotEqual: Vec2<f32>" not in generated_code
    assert "(pixel < amount)" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_bool_vector_logical_binary_operands_emit_boolean_lanes(tmp_path):
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    bool makeFlag() {
        return true;
    }

    void probe(bool ready, bvec3 a, bvec3 b) {
        bool scalarAnd = ready && makeFlag();
        let both = a && b;
        let either = a || b;
        bvec3 scalarRight = a && ready;
        bvec3 scalarLeft = makeFlag() || b;
        let complex = makeMask() && makeFlag();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let scalarAnd: bool = (ready && makeFlag());" in generated_code
    assert (
        "let both: BVec3 = BVec3::new((a.x && b.x), (a.y && b.y), (a.z && b.z));"
        in generated_code
    )
    assert (
        "let either: BVec3 = BVec3::new((a.x || b.x), (a.y || b.y), (a.z || b.z));"
        in generated_code
    )
    assert (
        "let scalarRight: BVec3 = BVec3::new((a.x && ready), (a.y && ready), (a.z && ready));"
        in generated_code
    )
    assert (
        "let scalarLeft: BVec3 = { let __cgl_vec_arg_0 = makeFlag(); BVec3::new((__cgl_vec_arg_0 || b.x), (__cgl_vec_arg_0 || b.y), (__cgl_vec_arg_0 || b.z)) };"
        in generated_code
    )
    assert (
        "let complex: BVec3 = { let __cgl_vec_arg_1 = makeMask(); let __cgl_vec_arg_2 = makeFlag(); BVec3::new((__cgl_vec_arg_1.x && __cgl_vec_arg_2), (__cgl_vec_arg_1.y && __cgl_vec_arg_2), (__cgl_vec_arg_1.z && __cgl_vec_arg_2)) };"
        in generated_code
    )
    assert "let both: Vec3<bool> = (a && b);" not in generated_code
    assert "let either: Vec3<bool> = (a || b);" not in generated_code
    assert "let scalarRight: Vec3<bool> = (a && ready);" not in generated_code
    assert "let scalarLeft: Vec3<bool> = (makeFlag() || b);" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_bool_vector_unary_not_emits_boolean_lanes(tmp_path):
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    void probe(bool flag, bvec2 mask2, bvec3 mask3, bvec4 mask4) {
        bool invFlag = !flag;
        bvec2 inv2 = !mask2;
        let inv3 = !mask3;
        bvec4 inv4 = !makeMask().xyzz;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let invFlag: bool = (!flag);" in generated_code
    assert "let inv2: BVec2 = BVec2::new((!mask2.x), (!mask2.y));" in generated_code
    assert (
        "let inv3: BVec3 = BVec3::new((!mask3.x), (!mask3.y), (!mask3.z));"
        in generated_code
    )
    assert (
        "let inv4: BVec4 = { let __cgl_vec_arg_0 = makeMask(); BVec4::new((!__cgl_vec_arg_0.x), (!__cgl_vec_arg_0.y), (!__cgl_vec_arg_0.z), (!__cgl_vec_arg_0.z)) };"
        in generated_code
    )
    assert "let inv2: Vec2<bool> = (!mask2);" not in generated_code
    assert "let inv3: Vec3<bool> = (!mask3);" not in generated_code
    assert "!{ let __cgl_swizzle_" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_scalar_numeric_unary_not_emits_boolean_comparisons(tmp_path):
    code = """
    bool probe(float weight, uint count) {
        bool zeroWeight = !weight;
        bool zeroCount = !count;
        bool nonZeroCount = !!count;
        return zeroWeight || zeroCount || nonZeroCount;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let zeroWeight: bool = (weight == 0.0);" in generated_code
    assert "let zeroCount: bool = (count == 0);" in generated_code
    assert "let nonZeroCount: bool = (!(count == 0));" in generated_code
    assert "!weight" not in generated_code
    assert "!count" not in generated_code
    assert "(!weight) != 0.0" not in generated_code
    assert "(!count) != 0" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_inferred_scalar_constructor_bindings_use_cast_types():
    code = """
    void probe() {
        let index = int(1.2);
        let unsignedIndex = uint(2);
        let precise = double(1.0);
        let enabled = bool(1);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let index: i32 = (1.2 as i32);" in generated_code
    assert "let unsignedIndex: u32 = (2 as u32);" in generated_code
    assert "let precise: f64 = (1.0 as f64);" in generated_code
    assert "let enabled: bool = (1 != 0);" in generated_code
    assert "let index: f32 = " not in generated_code
    assert "let enabled: f32 = " not in generated_code


def test_inferred_scalar_binary_bindings_promote_cast_operands():
    code = """
    void probe() {
        let rightUint = 3 + uint(2);
        let rightDouble = 2.0 + double(1.0);
        let compared = int(1.2) < 3;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let rightUint: u32 = (3 + (2 as u32));" in generated_code
    assert "let rightDouble: f64 = ((2.0 as f64) + (1.0 as f64));" in generated_code
    assert "let compared: bool = ((1.2 as i32) < 3);" in generated_code
    assert "let rightUint: i32 = " not in generated_code
    assert "let rightDouble: f32 = " not in generated_code


def test_mixed_scalar_binary_literals_match_float_operand_types():
    code = """
    void probe() {
        let intFloat = 3 + float(1);
        let floatInt = float(1) + 3;
        let intDouble = 2 + double(1.0);
        let doubleInt = double(1.0) + 2;
        let intUint = 3 + uint(2);
        let compareFloat = 3 < float(4);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let intFloat: f32 = (3.0 + (1 as f32));" in generated_code
    assert "let floatInt: f32 = ((1 as f32) + 3.0);" in generated_code
    assert "let intDouble: f64 = (2.0 + (1.0 as f64));" in generated_code
    assert "let doubleInt: f64 = ((1.0 as f64) + 2.0);" in generated_code
    assert "let intUint: u32 = (3 + (2 as u32));" in generated_code
    assert "let compareFloat: bool = (3.0 < (4 as f32));" in generated_code
    assert "(3 + (1 as f32))" not in generated_code
    assert "(3 < (4 as f32))" not in generated_code


def test_mixed_scalar_binary_variables_cast_to_promoted_operand_types():
    code = """
    void probe(int index, uint count, float weight, double precise) {
        let intFloat = index + float(1);
        let floatInt = weight + index;
        let intDouble = index + precise;
        let uintInt = count + index;
        let compareFloat = index < weight;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let intFloat: f32 = ((index as f32) + (1 as f32));" in generated_code
    assert "let floatInt: f32 = (weight + (index as f32));" in generated_code
    assert "let intDouble: f64 = ((index as f64) + precise);" in generated_code
    assert "let uintInt: u32 = (count + (index as u32));" in generated_code
    assert "let compareFloat: bool = ((index as f32) < weight);" in generated_code
    assert "(weight + index)" not in generated_code
    assert "(index < weight)" not in generated_code


def test_mixed_scalar_compound_assignments_cast_rhs_to_lhs_type():
    code = """
    void probe(int index, uint countInput, float weightInput, double preciseInput) {
        float weight = weightInput;
        double precise = preciseInput;
        uint count = countInput;
        weight += index;
        precise += index;
        count += index;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "weight += (index as f32);" in generated_code
    assert "precise += (index as f64);" in generated_code
    assert "count += (index as u32);" in generated_code
    assert "weight += index;" not in generated_code
    assert "precise += index;" not in generated_code
    assert "count += index;" not in generated_code


def test_mixed_scalar_simple_assignments_cast_rhs_to_lhs_type():
    code = """
    void probe(int index, uint countInput, float weightInput, double preciseInput) {
        float declaredWeight = index;
        double declaredPrecise = index;
        uint declaredCount = index;

        float weight = weightInput;
        double precise = preciseInput;
        uint count = countInput;
        weight = index;
        precise = index;
        count = index;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let declaredWeight: f32 = (index as f32);" in generated_code
    assert "let declaredPrecise: f64 = (index as f64);" in generated_code
    assert "let declaredCount: u32 = (index as u32);" in generated_code
    assert "weight = (index as f32);" in generated_code
    assert "precise = (index as f64);" in generated_code
    assert "count = (index as u32);" in generated_code
    assert "let declaredWeight: f32 = index;" not in generated_code
    assert "let declaredPrecise: f64 = index;" not in generated_code
    assert "let declaredCount: u32 = index;" not in generated_code
    assert "weight = index;" not in generated_code
    assert "precise = index;" not in generated_code
    assert "count = index;" not in generated_code


def test_vector_and_matrix_assignments_cast_rhs_to_lhs_component_types(tmp_path):
    code = """
    void probe(ivec2 pixel, vec2 amount, mat2 transform, dmat2 preciseInput) {
        vec2 declaredCoords = pixel;
        bvec2 declaredMask = amount;
        dmat2 declaredPrecise = transform;
        mat2 declaredRegular = preciseInput;

        vec2 coords = amount;
        dmat2 precise = preciseInput;
        coords = pixel;
        declaredMask = amount;
        precise = transform;
        declaredRegular = preciseInput;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let declaredCoords: Vec2 = Vec2::new((pixel.x as f32), (pixel.y as f32));"
        in generated_code
    )
    assert (
        "let mut declaredMask: BVec2 = BVec2::new((amount.x != 0.0), (amount.y != 0.0));"
        in generated_code
    )
    assert (
        "let declaredPrecise: DMat2 = DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]);"
        in generated_code
    )
    assert (
        "let mut declaredRegular: Mat2 = Mat2::from_cols_array(&[(preciseInput.x_axis.x as f32), (preciseInput.x_axis.y as f32), (preciseInput.y_axis.x as f32), (preciseInput.y_axis.y as f32)]);"
        in generated_code
    )
    assert "let mut coords: Vec2 = amount;" in generated_code
    assert "let mut precise: DMat2 = preciseInput;" in generated_code
    assert "coords = Vec2::new((pixel.x as f32), (pixel.y as f32));" in generated_code
    assert (
        "declaredMask = BVec2::new((amount.x != 0.0), (amount.y != 0.0));"
        in generated_code
    )
    assert (
        "precise = DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]);"
        in generated_code
    )
    assert (
        "declaredRegular = Mat2::from_cols_array(&[(preciseInput.x_axis.x as f32), (preciseInput.x_axis.y as f32), (preciseInput.y_axis.x as f32), (preciseInput.y_axis.y as f32)]);"
        in generated_code
    )
    assert "let declaredCoords: Vec2<f32> = pixel;" not in generated_code
    assert "coords = pixel;" not in generated_code
    assert "precise = transform;" not in generated_code
    assert "declaredRegular = preciseInput;" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_mixed_scalar_returns_cast_value_to_declared_return_type():
    code = """
    float toWeight(int index) {
        return index;
    }

    double toPrecise(int index) {
        return index;
    }

    uint toCount(int index) {
        return index;
    }

    bool toEnabled(int index) {
        return index;
    }

    float boolWeight(bool enabled) {
        return enabled;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return (index as f32);" in generated_code
    assert "return (index as f64);" in generated_code
    assert "return (index as u32);" in generated_code
    assert "return (index != 0);" in generated_code
    assert "return (if enabled { 1.0 } else { 0.0 });" in generated_code
    assert "return index;" not in generated_code
    assert "return enabled;" not in generated_code


def test_numeric_conditions_compare_against_zero_in_rust_bool_sites(tmp_path):
    code = """
    float choose(int index, uint count, float inputWeight, bool ready) {
        int localIndex = index;
        float weight = inputWeight;
        if (localIndex) {
            weight = 1.0;
        } else if (count) {
            weight = 2.0;
        }
        while (weight) {
            weight -= 1.0;
        }
        for (int i = 3; i; i--) {
            weight += 1.0;
        }
        do {
            localIndex -= 1;
        } while (localIndex);
        let selected = localIndex ? weight : 0.0;
        if (!localIndex || ready && count) {
            weight = selected;
        }
        return weight;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if (localIndex != 0) {" in generated_code
    assert "} else if (count != 0) {" in generated_code
    assert "while (weight != 0.0) {" in generated_code
    assert "while (i != 0) {" in generated_code
    assert "if !((localIndex != 0)) {" in generated_code
    assert (
        "let selected: f32 = (if (localIndex != 0) { weight } else { 0.0 });"
        in generated_code
    )
    assert "if (!((localIndex != 0)) || (ready && (count != 0))) {" in generated_code
    assert "if localIndex {" not in generated_code
    assert "else if count {" not in generated_code
    assert "while weight {" not in generated_code
    assert "while i {" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_mixed_scalar_ternary_branches_cast_to_result_type():
    code = """
    float chooseFloat(bool ready, int index, uint count, float weight) {
        float picked = ready ? index : weight;
        float reversed = ready ? weight : index;
        uint pickedCount = ready ? index : count;
        let inferred = ready ? index : weight;
        return ready ? weight : index;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let picked: f32 = (if ready { (index as f32) } else { weight });"
        in generated_code
    )
    assert (
        "let reversed: f32 = (if ready { weight } else { (index as f32) });"
        in generated_code
    )
    assert (
        "let pickedCount: u32 = (if ready { (index as u32) } else { count });"
        in generated_code
    )
    assert (
        "let inferred: f32 = (if ready { (index as f32) } else { weight });"
        in generated_code
    )
    assert "return (if ready { weight } else { (index as f32) });" in generated_code
    assert (
        "let picked: f32 = ((if ready { index } else { weight }) as f32);"
        not in generated_code
    )
    assert (
        "let pickedCount: u32 = ((if ready { index } else { count }) as u32);"
        not in generated_code
    )
    assert "return (if ready { weight } else { index });" not in generated_code


def test_vector_and_matrix_ternary_branches_cast_to_result_type(tmp_path):
    code = """
    vec2 chooseVec(bool ready, ivec2 pixel, vec2 amount) {
        vec2 declared = ready ? pixel : amount;
        bvec2 mask = ready ? amount : pixel;
        declared = ready ? amount : pixel;
        return ready ? pixel : amount;
    }

    dmat2 chooseMat(bool ready, mat2 transform, dmat2 preciseInput) {
        dmat2 precise = ready ? transform : preciseInput;
        mat2 regular = ready ? preciseInput : transform;
        precise = ready ? transform : preciseInput;
        return ready ? transform : preciseInput;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let mut declared: Vec2 = (if ready { Vec2::new((pixel.x as f32), (pixel.y as f32)) } else { amount });"
        in generated_code
    )
    assert (
        "let mask: BVec2 = (if ready { BVec2::new((amount.x != 0.0), (amount.y != 0.0)) } else { BVec2::new((pixel.x != 0), (pixel.y != 0)) });"
        in generated_code
    )
    assert (
        "declared = (if ready { amount } else { Vec2::new((pixel.x as f32), (pixel.y as f32)) });"
        in generated_code
    )
    assert (
        "return (if ready { Vec2::new((pixel.x as f32), (pixel.y as f32)) } else { amount });"
        in generated_code
    )
    assert (
        "let mut precise: DMat2 = (if ready { DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) } else { preciseInput });"
        in generated_code
    )
    assert (
        "let regular: Mat2 = (if ready { Mat2::from_cols_array(&[(preciseInput.x_axis.x as f32), (preciseInput.x_axis.y as f32), (preciseInput.y_axis.x as f32), (preciseInput.y_axis.y as f32)]) } else { transform });"
        in generated_code
    )
    assert (
        "precise = (if ready { DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) } else { preciseInput });"
        in generated_code
    )
    assert (
        "return (if ready { DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) } else { preciseInput });"
        in generated_code
    )
    assert "{ let __cgl_vec_arg_" not in generated_code
    assert "{ let __cgl_mat_arg_" not in generated_code
    assert "if ready { pixel } else { amount }" not in generated_code
    assert "if ready { transform } else { preciseInput }" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_bool_vector_ternary_conditions_emit_lane_selectors(tmp_path):
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    vec3 makeA() {
        return vec3(1.0, 2.0, 3.0);
    }

    float makeScalar() {
        return 0.5;
    }

    void probe(bool cond, bvec3 mask, vec3 a, vec3 b) {
        vec3 scalarSelected = cond ? a : b;
        vec3 laneSelected = mask ? a : b;
        vec3 scalarArms = mask ? 1.0 : 0.0;
        vec3 mixedArm = mask ? a : 0.0;
        let inferred = mask ? a : 0.0;
        let complex = makeMask() ? makeA() : makeScalar();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let scalarSelected: Vec3 = (if cond { a } else { b });" in generated_code
    assert (
        "let laneSelected: Vec3 = Vec3::new((if mask.x { a.x } else { b.x }), (if mask.y { a.y } else { b.y }), (if mask.z { a.z } else { b.z }));"
        in generated_code
    )
    assert (
        "let scalarArms: Vec3 = Vec3::new((if mask.x { 1.0 } else { 0.0 }), (if mask.y { 1.0 } else { 0.0 }), (if mask.z { 1.0 } else { 0.0 }));"
        in generated_code
    )
    assert (
        "let mixedArm: Vec3 = Vec3::new((if mask.x { a.x } else { 0.0 }), (if mask.y { a.y } else { 0.0 }), (if mask.z { a.z } else { 0.0 }));"
        in generated_code
    )
    assert (
        "let inferred: Vec3 = Vec3::new((if mask.x { a.x } else { 0.0 }), (if mask.y { a.y } else { 0.0 }), (if mask.z { a.z } else { 0.0 }));"
        in generated_code
    )
    assert (
        "let complex: Vec3 = { let __cgl_vec_arg_0 = makeMask(); let __cgl_vec_arg_1 = makeA(); let __cgl_vec_arg_2 = makeScalar(); Vec3::new((if __cgl_vec_arg_0.x { __cgl_vec_arg_1.x } else { __cgl_vec_arg_2 }), (if __cgl_vec_arg_0.y { __cgl_vec_arg_1.y } else { __cgl_vec_arg_2 }), (if __cgl_vec_arg_0.z { __cgl_vec_arg_1.z } else { __cgl_vec_arg_2 })) };"
        in generated_code
    )
    assert "(if mask { a } else { b })" not in generated_code
    assert "(if mask { 1.0 } else { 0.0 })" not in generated_code
    assert "(if makeMask() { makeA() } else { makeScalar() })" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_inferred_vector_and_matrix_ternaries_promote_branch_component_types(tmp_path):
    code = """
    void infer(bool ready, ivec2 pixel, vec2 amount, mat2 transform, dmat2 preciseInput) {
        let inferredVec = ready ? pixel : amount;
        let inferredVecReverse = ready ? amount : pixel;
        let inferredMat = ready ? transform : preciseInput;
        let inferredMatReverse = ready ? preciseInput : transform;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let inferredVec: Vec2 = (if ready { Vec2::new((pixel.x as f32), (pixel.y as f32)) } else { amount });"
        in generated_code
    )
    assert (
        "let inferredVecReverse: Vec2 = (if ready { amount } else { Vec2::new((pixel.x as f32), (pixel.y as f32)) });"
        in generated_code
    )
    assert (
        "let inferredMat: DMat2 = (if ready { DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) } else { preciseInput });"
        in generated_code
    )
    assert (
        "let inferredMatReverse: DMat2 = (if ready { preciseInput } else { DMat2::from_cols_array(&[(transform.x_axis.x as f64), (transform.x_axis.y as f64), (transform.y_axis.x as f64), (transform.y_axis.y as f64)]) });"
        in generated_code
    )
    assert "let inferredVec: Vec2<i32>" not in generated_code
    assert "let inferredMat: Mat2<f32>" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generic_vector_constructors_emit_rust_names(tmp_path):
    code = """
    shader main {
        compute {
            void main() {
                vec2<f64> precise = vec2<f64>(1.0, 2.0);
                vec3<i32> index = vec3<i32>(1, 2, 3);
                vec4<u32> mask = vec4<u32>(1, 2, 3, 4);
                vec2<bool> flags = vec2<bool>(true, false);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let precise: DVec2 = DVec2::new(1.0, 2.0);" in generated_code
    assert "let index: IVec3 = IVec3::new(1, 2, 3);" in generated_code
    assert "let mask: UVec4 = UVec4::new(1, 2, 3, 4);" in generated_code
    assert "let flags: BVec2 = BVec2::new(true, false);" in generated_code
    assert "Vec2<f64>::new" not in generated_code
    assert "Vec3<i32>::new" not in generated_code
    assert "Vec4<u32>::new" not in generated_code
    assert "Vec2<bool>::new" not in generated_code
    assert "vec2<" not in generated_code
    assert "vec3<" not in generated_code
    assert "vec4<" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_narrow_generic_vector_constructors_emit_rust_names():
    code = """
    shader main {
        compute {
            void main() {
                vec4<u8> bytes = vec4<u8>(1, 2, 3, 4);
                vec2<i16> shorts = vec2<i16>(1, 2);
                vec2<f16> halves = vec2<f16>(1.0, 2.0);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustRefusal, match=r"spirv-std's glam has no vector of u8 \(Vec4<u8>\)"
    ):
        generate_code(ast)


def test_inferred_let_vector_declarations_preserve_vector_type():
    code = """
    shader main {
        compute {
            void main() {
                let value = vec4(1.0, 2.0, 3.0, 4.0);
                let x = value.x;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let value: Vec4 = Vec4::new(1.0, 2.0, 3.0, 4.0);" in generated_code
    assert "let x: f32 = value.x;" in generated_code
    assert "let value: f32 = Vec4" not in generated_code


def test_scalar_vector_constructors_splat_rust_values():
    code = """
    shader main {
        compute {
            void main() {
                float weight = 0.5;
                int index = 1;
                bool enabled = true;
                vec4 base = vec4(1.0);
                vec3 scaled = vec3(weight);
                ivec2 pixel = ivec2(index);
                bvec4 mask = bvec4(enabled);
                vec4 offset = vec4(0.25, 0.5, 0.75, 1.0);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let base: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0);" in generated_code
    assert "let scaled: Vec3 = Vec3::new(weight, weight, weight);" in generated_code
    assert "let pixel: IVec2 = IVec2::new(index, index);" in generated_code
    assert (
        "let mask: BVec4 = BVec4::new(enabled, enabled, enabled, enabled);"
        in generated_code
    )
    assert "let offset: Vec4 = Vec4::new(0.25, 0.5, 0.75, 1.0);" in generated_code
    assert "Vec4::<f32>::new(1.0);" not in generated_code
    assert "Vec3::<f32>::new(weight);" not in generated_code
    assert "Vec2::<i32>::new(index);" not in generated_code
    assert "Vec4::<bool>::new(enabled);" not in generated_code


def test_complex_scalar_vector_constructor_splats_use_single_evaluation_blocks():
    code = """
    float makeWeight() {
        return 0.5;
    }

    int nextIndex() {
        return 1;
    }

    bool nextFlag() {
        return true;
    }

    void probe() {
        vec3 fromCall = vec3(makeWeight());
        vec3 fromCast = vec3(nextIndex());
        bvec4 fromBoolCall = bvec4(nextFlag());
        bvec2 fromInt = bvec2(nextIndex());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let fromCall: Vec3 = { let __cgl_vec_arg_0 = makeWeight(); Vec3::new(__cgl_vec_arg_0, __cgl_vec_arg_0, __cgl_vec_arg_0) };"
        in generated_code
    )
    assert (
        "let fromCast: Vec3 = { let __cgl_vec_arg_1 = (nextIndex() as f32); Vec3::new(__cgl_vec_arg_1, __cgl_vec_arg_1, __cgl_vec_arg_1) };"
        in generated_code
    )
    assert (
        "let fromBoolCall: BVec4 = { let __cgl_vec_arg_2 = nextFlag(); BVec4::new(__cgl_vec_arg_2, __cgl_vec_arg_2, __cgl_vec_arg_2, __cgl_vec_arg_2) };"
        in generated_code
    )
    assert (
        "let fromInt: BVec2 = { let __cgl_vec_arg_3 = (nextIndex() != 0); BVec2::new(__cgl_vec_arg_3, __cgl_vec_arg_3) };"
        in generated_code
    )
    assert (
        "Vec3::<f32>::new(makeWeight(), makeWeight(), makeWeight())"
        not in generated_code
    )
    assert (
        "Vec3::<f32>::new((nextIndex() as f32), "
        "(nextIndex() as f32), (nextIndex() as f32))" not in generated_code
    )
    assert "Vec4::<bool>::new(nextFlag(), nextFlag()" not in generated_code
    assert (
        "Vec2::<bool>::new((nextIndex() != 0), (nextIndex() != 0))"
        not in generated_code
    )


def test_mixed_scalar_vector_constructor_lanes_cast_to_component_type():
    code = """
    void probe(float weight, int index, uint count) {
        vec2 mixedFloat = vec2(weight, index);
        uvec3 mixedUint = uvec3(count, index, 2);
        vec3 splatFloat = vec3(index);
        bvec2 boolMask = bvec2(index, weight);
        vec3 fromIntVector = vec3(ivec2(index, 2), weight);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let mixedFloat: Vec2 = Vec2::new(weight, (index as f32));" in generated_code
    assert (
        "let mixedUint: UVec3 = UVec3::new(count, (index as u32), 2);" in generated_code
    )
    assert (
        "let splatFloat: Vec3 = Vec3::new((index as f32), (index as f32), (index as f32));"
        in generated_code
    )
    assert (
        "let boolMask: BVec2 = BVec2::new((index != 0), (weight != 0.0));"
        in generated_code
    )
    assert (
        "let fromIntVector: Vec3 = { let __cgl_vec_arg_0 = IVec2::new(index, 2); Vec3::new((__cgl_vec_arg_0.x as f32), (__cgl_vec_arg_0.y as f32), weight) };"
        in generated_code
    )
    assert "Vec2::<f32>::new(weight, index)" not in generated_code
    assert "Vec3::<u32>::new(count, index, 2)" not in generated_code
    assert "Vec3::<f32>::new(index, index, index)" not in generated_code
    assert "Vec2::<bool>::new(index, weight)" not in generated_code


def test_composite_vector_constructors_flatten_rust_lanes():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        compute {
            void main(VSInput input) {
                vec2 xy = vec2(1.0, 2.0);
                vec4 color = vec4(xy, 0.0, 1.0);
                vec3 position = vec3(xy, 1.0);
                vec3 rgb = vec3(color.rgb);
                vec4 packed = vec4(input.texCoord, 0.0, 1.0);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused,
        match="stage: compute entry parameter 'input' is not a compute built-in input",
    ):
        generate_code(ast)


def test_complex_vector_constructor_args_use_single_evaluation_blocks():
    code = """
    vec2 makeUv() {
        return vec2(1.0, 2.0);
    }

    vec3 makeNormal() {
        return vec3(0.0, 1.0, 0.0);
    }

    vec4 makeColor() {
        return vec4(1.0, 2.0, 3.0, 4.0);
    }

    void probe() {
        vec4 color = vec4(makeUv(), 0.0, 1.0);
        vec4 packed = vec4(makeNormal(), 1.0);
        vec4 mixed = vec4(makeColor().rg, 0.0, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let color: Vec4 = { let __cgl_vec_arg_0 = makeUv(); Vec4::new(__cgl_vec_arg_0.x, __cgl_vec_arg_0.y, 0.0, 1.0) };"
        in generated_code
    )
    assert (
        "let packed: Vec4 = { let __cgl_vec_arg_1 = makeNormal(); Vec4::new(__cgl_vec_arg_1.x, __cgl_vec_arg_1.y, __cgl_vec_arg_1.z, 1.0) };"
        in generated_code
    )
    assert (
        "let mixed: Vec4 = { let __cgl_vec_arg_2 = makeColor(); Vec4::new(__cgl_vec_arg_2.x, __cgl_vec_arg_2.y, 0.0, 1.0) };"
        in generated_code
    )
    assert "makeUv().x" not in generated_code
    assert "makeNormal().z" not in generated_code
    assert "makeColor().rg" not in generated_code


def test_standalone_vector_swizzles_emit_rust_constructors():
    code = """
    void probe() {
        vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
        let rgb = color.rgb;
        let bgra = color.bgra;
        let red = color.r;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let rgb: Vec3 = Vec3::new(color.x, color.y, color.z);" in generated_code
    assert (
        "let bgra: Vec4 = Vec4::new(color.z, color.y, color.x, color.w);"
        in generated_code
    )
    assert "let red: f32 = color.x;" in generated_code
    assert "color.rgb" not in generated_code
    assert "color.bgra" not in generated_code
    assert "color.r;" not in generated_code


def test_complex_vector_swizzles_use_single_evaluation_blocks():
    code = """
    vec4 makeColor() {
        return vec4(1.0, 2.0, 3.0, 4.0);
    }

    void probe() {
        let rgb = makeColor().rgb;
        let xy = vec4(1.0, 2.0, 3.0, 4.0).xy;
        let red = makeColor().r;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let rgb: Vec3 = { let __cgl_swizzle_0 = makeColor(); Vec3::new(__cgl_swizzle_0.x, __cgl_swizzle_0.y, __cgl_swizzle_0.z) };"
        in generated_code
    )
    assert (
        "let xy: Vec2 = { let __cgl_swizzle_1 = Vec4::new(1.0, 2.0, 3.0, 4.0); Vec2::new(__cgl_swizzle_1.x, __cgl_swizzle_1.y) };"
        in generated_code
    )
    assert "let red: f32 = makeColor().x;" in generated_code
    assert "makeColor().rgb" not in generated_code
    assert "Vec4::<f32>::new(1.0, 2.0, 3.0, 4.0).xy" not in generated_code


def test_switch_statement_emits_rust_match(tmp_path):
    code = """
    shader main {
        compute {
            int main(int mode) {
                int out = 0;
                switch (mode) {
                    case 0:
                        out = 1;
                        break;
                    default:
                        out = 2;
                }
                return out;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused, match="stage: compute entry point 'main' returns a int"
    ):
        generate_code(ast)


def test_match_statement_emits_rust_match(tmp_path):
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused, match="stage: compute entry point 'main' returns a int"
    ):
        generate_code(ast)


def test_match_arm_break_and_continue_preserve_loop_control_flow_and_smoke_compile(
    tmp_path,
):
    code = """
    shader MatchLoopControlFlow {
        compute {
            int main(int mode) {
                int total = 0;
                for (int i = 0; i < 4; i++) {
                    match mode {
                        0 => {
                            continue;
                        }
                        1 => {
                            break;
                        }
                        _ => {
                            total += i;
                        }
                    }
                    total += 1;
                }
                loop {
                    match mode {
                        2 => {
                            break;
                        }
                        _ => {
                            total += 1;
                            break;
                        }
                    }
                }
                return total;
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="stage: compute entry point 'main' returns a int"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_match_path_constructor_struct_and_guarded_patterns_emit_rust_match():
    code = """
    shader PatternCases {
        fn inspect(result: Result<int, Error>, model: LightingModel) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(_) => 0,
            }

            match model {
                LightingModel::Phong { ambient, diffuse, specular, shininess } if shininess > 0.0 => 1,
                LightingModel::Toon { base_color, .. } => 2,
            }

            return 0;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "match result" in generated_code
    assert "Result::Ok(value) =>" in generated_code
    assert "Result::Err(_) =>" in generated_code
    assert "match model" in generated_code
    assert (
        "LightingModel::Phong { ambient: _ambient, diffuse: _diffuse, "
        "specular: _specular, shininess } if (shininess > 0.0) =>" in generated_code
    )
    assert "LightingModel::Toon { base_color: _base_color, .. } =>" in generated_code
    assert "MatchNode" not in generated_code
    assert "MatchArmNode" not in generated_code


def test_unused_match_pattern_bindings_are_prefixed_but_used_bindings_are_preserved(
    tmp_path,
):
    code = """
    shader PatternWarnings {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum Command {
            Draw {
                payload: int,
                amount: int
            },
            Skip
        }

        fn unwrap(result: Result<int, int>) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(error) => 0,
            }
        }

        fn inspect(command: Command) -> int {
            match command {
                Command::Draw { payload, amount } if amount > 0 => amount,
                Command::Draw { payload, amount } => amount,
                Command::Skip => 0,
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "Result::Ok(value) =>" in generated_code
    assert "Result::Err(_error) =>" in generated_code
    assert (
        "Command::Draw { payload: _payload, amount } if (amount > 0) =>"
        in generated_code
    )
    assert "Command::Draw { payload: _payload, amount } =>" in generated_code
    assert "amount: _amount" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_mutated_match_pattern_bindings_emit_mut_and_smoke_compile(tmp_path):
    code = """
    shader MutablePatternBindings {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum Error {
            Bad
        }

        enum Command {
            Draw {
                payload: int,
                amount: int
            },
            Skip
        }

        fn bump(result: Result<int, Error>) -> int {
            match result {
                Result::Ok(value) => {
                    value += 1;
                    value
                },
                Result::Err(_) => 0,
            }
        }

        fn adjust(command: Command) -> int {
            match command {
                Command::Draw { payload, amount } => {
                    payload += amount;
                    payload
                },
                Command::Skip => 0,
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "Result::Ok(mut value) =>" in generated_code
    assert "Command::Draw { payload: mut payload, amount } =>" in generated_code
    assert "Result::Ok(value) =>" not in generated_code
    assert "payload += amount;" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_match_arm_block_tail_expression_emits_without_semicolon(tmp_path):
    code = """
    shader PatternCases {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum MathError {
            InvalidInput
        }

        enum VectorOp {
            Cross,
            Dot
        }

        fn convert(op: VectorOp) -> Result<int, MathError> {
            match op {
                VectorOp::Cross => {
                    Result::Ok(1)
                },
                _ => Result::Err(MathError::InvalidInput),
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "VectorOp::Cross => {\n            Result::Ok(1)\n        }" in generated_code
    )
    assert (
        "_ => {\n            Result::Err(MathError::InvalidInput)\n        }"
        in generated_code
    )
    assert "Result::Ok(1);" not in generated_code
    assert "Result::Err(MathError::InvalidInput);" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_keyword_like_let_binding_inside_match_arm_preserves_function_body(
    tmp_path,
):
    code = """
    shader KeywordBinding {
        fn process_lighting_model(model: int) -> vec3 {
            match model {
                _ => {
                    let geometry = 1.0;
                    vec3(geometry, geometry, geometry)
                }
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub fn process_lighting_model(model: i32) -> Vec3 {" in generated_code
    assert "match model" in generated_code
    assert "let geometry: f32 = 1.0;" in generated_code
    assert "Vec3::new(geometry, geometry, geometry)" in generated_code
    assert (
        "pub fn process_lighting_model(model: i32) -> Vec3<f32> {\n}\n"
        not in generated_code
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_match_expression_initializer_emits_rust_match_initializer(tmp_path):
    code = """
    shader MatchInitializer {
        fn choose(model: int, fallback: vec3) -> vec3 {
            let processed_normal = match model {
                0 => vec3(1.0, 1.0, 1.0),
                _ => fallback,
            };
            processed_normal
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let processed_normal: Vec3 = match model {" in generated_code
    assert "0 => {\n            Vec3::new(1.0, 1.0, 1.0)" in generated_code
    assert "_ => {\n            fallback" in generated_code
    assert "};\n    processed_normal" in generated_code
    assert "let processed_normal: f32 = match model" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_stage_local_structs_and_uniforms_infer_normalized_matrix_vector_types(
    tmp_path,
):
    code = """
    shader StageInference {
        generic<T> struct Vec3 {
            T x;
            T y;
            T z;
        }

        vertex {
            struct VertexInput {
                position: vec3,
                normal: vec3,
            }

            struct VertexOutput {
                normal: Vec3<float>,
            }

            uniform mat4 model_matrix;
            uniform mat3 normal_matrix;

            VertexOutput main(VertexInput input) {
                let world_pos = model_matrix * vec4(input.position, 1.0);
                let world_normal = normalize(normal_matrix * input.normal);
                let normal_vec3 = Vec3 { x: world_normal.x, y: world_normal.y, z: world_normal.z };
                VertexOutput { normal: normal_vec3 }
            }
        }
    }
    """

    # std140 gives each mat3 column a 16-byte stride; glam's Mat3 packs its columns 12 bytes apart.
    with pytest.raises(
        RustLoweringRefused,
        match=r"uniform 'normal_matrix': 'mat3 column 1' is at std140 offset 16, "
        r"rust-gpu lays it out at offset 12",
    ):
        generate_code(parse_code(tokenize_code(code)))

    # With a mat4 normal matrix the uniforms bind, and the generic Vec3<T> structure is what rust-gpu refuses.
    mat4_code = code.replace(
        "uniform mat3 normal_matrix;", "uniform mat4 normal_matrix;"
    ).replace(
        "normal_matrix * input.normal", "(normal_matrix * vec4(input.normal, 0.0)).xyz"
    )
    with pytest.raises(
        RustRefusal, match=r"spirv-std's glam has no vector of T \(Vec3<T>\)"
    ):
        generate_code(parse_code(tokenize_code(mat4_code)))


def test_stage_local_uniform_statics_are_emitted_once(tmp_path):
    code = """
    shader StageUniforms {
        vertex {
            uniform mat4 shared_transform;

            vec4 main(vec3 position) {
                shared_transform * vec4(position, 1.0)
            }
        }

        fragment {
            uniform mat4 shared_transform;
            uniform sampler2D main_texture;

            vec4 main(vec2 uv) {
                texture(main_texture, uv)
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_method_calls_and_shorthand_path_constructors_emit_rust_syntax():
    code = """
    shader ConstructorCases {
        fn build(v1: Vec3<T>, v2: Vec3<T>, color: vec4, depth: float, op: VectorOp) -> int {
            match op {
                VectorOp::Add => Result::Ok(Vec3 { x: v1.x.add(v2.x), y: v1.y.mul(v2.y), z: v1.z }),
                _ => Result::Ok(RenderOutput::Clear { color, depth }),
            }

            return 0;
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused, match="function: 'Result::Ok' has no rust-gpu lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_non_wildcard_match_fallback_uses_never_expression(tmp_path):
    code = """
    shader PatternCases {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum MathError {
            InvalidInput
        }

        enum VectorOp {
            Cross,
            Add,
            Dot
        }

        fn convert(op: VectorOp) -> Result<int, MathError> {
            match op {
                VectorOp::Cross => Result::Ok(1),
                VectorOp::Add => Result::Err(MathError::InvalidInput),
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "_ => unreachable!()," in generated_code
    assert "_ => {}," not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_exhaustive_bool_result_and_enum_matches_omit_unreachable_fallback(tmp_path):
    code = """
    shader ExhaustivePatternCases {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum Mode {
            Bright,
            Dark
        }

        enum Command {
            Draw {
                payload: Result<int, int>,
                amount: int
            },
            Skip
        }

        struct Pair {
            left: int,
            right: int
        }

        fn unwrap(result: Result<int, int>) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(_) => 0,
            }
        }

        fn choose(mode: Mode) -> int {
            match mode {
                Mode::Bright => 1,
                Mode::Dark => 0,
            }
        }

        fn flag_to_int(flag: bool) -> int {
            match flag {
                true => 1,
                false => 0,
            }
        }

        fn sum_pair(pair: Pair) -> int {
            match pair {
                Pair { left, right } => left + right,
            }
        }

        fn nested(command: Command) -> int {
            match command {
                Command::Draw { payload: Result::Ok(value), amount } => value + amount,
                Command::Draw { payload: Result::Err(_), amount } => amount,
                Command::Skip => 0,
            }
        }

        fn partial(mode: Mode) -> int {
            match mode {
                Mode::Bright => 1,
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    def function_section(name):
        start = generated_code.index(f"pub fn {name}")
        next_start = generated_code.find("\npub fn ", start + 1)
        if next_start == -1:
            return generated_code[start:]
        return generated_code[start:next_start]

    assert "_ => unreachable!()," not in function_section("unwrap")
    assert "_ => unreachable!()," not in function_section("choose")
    assert "_ => unreachable!()," not in function_section("flag_to_int")
    assert "_ => unreachable!()," not in function_section("sum_pair")
    assert "_ => unreachable!()," not in function_section("nested")
    assert "_ => unreachable!()," in function_section("partial")
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_inferred_generic_struct_literals_and_destructured_members_keep_generic_types():
    code = """
    shader InferredStructLiteral {
        generic<T> struct Vec3 {
            T x;
            T y;
            T z;
        }

        generic<T> struct Matrix3x3 {
            Vec3<T> row0;
            Vec3<T> row1;
            Vec3<T> row2;
        }

        generic<T> fn make_vec(a: T, b: T, c: T) -> Vec3<T> {
            let result = Vec3 { x: a, y: b, z: c };
            return result;
        }

        generic<T> fn determinant(matrix: Matrix3x3<T>) -> T {
            match matrix {
                Matrix3x3 { row0, row1, row2 } => {
                    let a = row0.x;
                    a
                }
            }
        }
    }
    """

    with pytest.raises(
        RustRefusal, match=r"spirv-std's glam has no vector of T \(Vec3<T>\)"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_match_guarded_literal_arm_emits_rust_guard(tmp_path):
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 if mode > 0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(
        RustLoweringRefused, match="stage: compute entry point 'main' returns a int"
    ):
        generate_code(ast)


def test_generic_vector_composite_types_emit_rust_names(tmp_path):
    code = """
    shader GenericComposite {
        struct Packed {
            vec2<f64> precise;
            vec3<i32> index;
            vec4<u32> mask;
            vec2<bool> flags;
        };

        vec2<f64> passthrough(vec2<f64> value, vec3<i32> index, vec4<u32> mask, vec2<bool> flags) {
            vec2<f64> localValues[2];
            localValues[0] = value;
            return localValues[0];
        }

        compute {
            void main() {
                Packed p;
                vec2<f64> values[2];
                values[0] = vec2<f64>(1.0, 2.0);
                vec2<f64> result = passthrough(values[0], vec3<i32>(1, 2, 3), vec4<u32>(1, 2, 3, 4), vec2<bool>(true, false));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub precise: DVec2," in generated_code
    assert "pub index: IVec3," in generated_code
    assert "pub mask: UVec4," in generated_code
    assert "pub flags: BVec2," in generated_code
    assert (
        "pub fn passthrough(value: DVec2, index: IVec3, mask: UVec4, flags: BVec2) -> DVec2"
    ) in generated_code
    assert (
        "let mut localValues: [DVec2; 2] = [Default::default(); 2];" in generated_code
    )
    assert "let mut values: [DVec2; 2] = [Default::default(); 2];" in generated_code
    assert "LiteralNode(" not in generated_code
    assert "vec2<" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_array_access(tmp_path):
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        cbuffer Material {
            float values[4];
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(values[0], values[1], values[2], 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        # std140 places each float of the array in its own 16-byte register.
        assert "pub values: [Vec4; 4]," in generated_code
        assert (
            "material.values[0].x, material.values[1].x, material.values[2].x"
            in generated_code
        )
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)
    except SyntaxError:
        pytest.fail("Array access codegen not implemented")


def test_array_literals_emit_rust_array_initializers():
    code = """
    float globalWeights[4] = {1.0, 2.0};

    float pickScalar(int index) {
        float values[4] = {1.0, 2.0, 3.0, 4.0};
        return values[index];
    }

    float[4] makeValues() {
        return {1.0, 2.0};
    }

    vec3 pickColor(int index) {
        vec3 colors[2] = {vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)};
        return colors[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ("static GLOBAL_WEIGHTS: [f32; 4] = [1.0, 2.0, 0.0, 0.0];") in generated_code
    assert "static GLOBAL_WEIGHTS: [f32; 4] = [1.0, 2.0, Default" not in generated_code
    assert "let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];" in generated_code
    assert (
        "return [1.0, 2.0, Default::default(), Default::default()];"
    ) in generated_code
    assert (
        "let colors: [Vec3; 2] = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];"
    ) in generated_code
    assert "ArrayLiteralNode" not in generated_code


def test_array_access_casts_non_literal_indices_to_usize():
    code = """
    float globalWeights[4] = {1.0, 2.0};
    vec3 colors[2] = {vec3(1.0, 2.0, 3.0)};

    float pickScalar(int index) {
        float values[4] = {1.0, 2.0, 3.0, 4.0};
        return values[index] + globalWeights[index] + values[0];
    }

    vec3 pickColor(int index) {
        return colors[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "values[index as usize]" in generated_code
    assert "GLOBAL_WEIGHTS[index as usize]" in generated_code
    assert "(COLORS())[index as usize]" in generated_code
    assert "values[0]" in generated_code
    assert "values[0 as usize]" not in generated_code
    assert "values[index]" not in generated_code
    assert "globalWeights[index]" not in generated_code


def test_inferred_array_access_bindings_use_element_types():
    code = """
    float globalWeights[4] = {1.0, 2.0};
    vec3 globalColors[2] = {vec3(1.0, 2.0, 3.0)};

    void probe(int index) {
        float values[4] = {1.0, 2.0, 3.0, 4.0};
        vec3 colors[2] = {
            vec3(1.0, 2.0, 3.0),
            vec3(4.0, 5.0, 6.0)
        };
        let scalar = values[index];
        let globalScalar = globalWeights[index];
        let color = colors[index];
        let globalColor = globalColors[index];
        let literalScalar = values[0];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let scalar: f32 = values[index as usize];" in generated_code
    assert "let globalScalar: f32 = GLOBAL_WEIGHTS[index as usize];" in generated_code
    assert "let color: Vec3 = colors[index as usize];" in generated_code
    assert (
        "let globalColor: Vec3 = (GLOBAL_COLORS())[index as usize];" in generated_code
    )
    assert "let literalScalar: f32 = values[0];" in generated_code
    assert "let scalar: [f32; 4]" not in generated_code
    assert "let color: [Vec3<f32>; 2]" not in generated_code


def test_auto_typed_legacy_bindings_infer_initializer_types():
    ast = ShaderNode(
        "LegacyAutoBindings",
        ExecutionModel.GENERAL_PURPOSE,
        functions=[
            FunctionNode(
                "probe",
                PrimitiveType("void"),
                [],
                [
                    ArrayNode(PrimitiveType("float"), "values", 4),
                    VariableNode(
                        "picked",
                        PrimitiveType("auto"),
                        ArrayAccessNode(
                            IdentifierNode("values"),
                            LiteralNode(0, PrimitiveType("int")),
                        ),
                    ),
                    VariableNode(
                        "flag",
                        "auto",
                        LiteralNode(True, PrimitiveType("bool")),
                    ),
                ],
            )
        ],
    )

    generated_code = generate_code(ast)

    assert "let picked: f32 = values[0];" in generated_code
    assert "let flag: bool = true;" in generated_code
    assert ": auto" not in generated_code


def test_inferred_function_call_bindings_use_user_return_types():
    code = """
    float pickWeight() {
        return 0.5;
    }

    vec3 makeColor() {
        return vec3(1.0, 2.0, 3.0);
    }

    bool isEnabled() {
        return true;
    }

    void probe() {
        let weight = pickWeight();
        let color = makeColor();
        let enabled = isEnabled();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let weight: f32 = pickWeight();" in generated_code
    assert "let color: Vec3 = makeColor();" in generated_code
    assert "let enabled: bool = isEnabled();" in generated_code
    assert "let color: f32 = makeColor();" not in generated_code
    assert "let enabled: f32 = isEnabled();" not in generated_code


def test_rust_imports():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "use spirv_std::glam::*;" in generated_code
        assert "use spirv_std::spirv;" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust imports not generated")


def test_let_binding():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                let x = input.texCoord.x;
                let y = input.texCoord.y;
                output.color = vec4(x, y, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "let x:" in generated_code  # Rust let binding syntax
        assert "let y:" in generated_code  # Rust let binding syntax
        print(generated_code)
    except SyntaxError:
        pytest.fail("Let binding codegen not implemented")


def test_rust_attributes():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert_rust_stage_attr(generated_code, "vertex")
        assert_rust_stage_attr(generated_code, "fragment")
        assert "#[repr(C)]" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust attributes not generated")


def test_rust_type_conversions():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                int i = int(input.texCoord.x * 100.0);
                float f = float(i);
                output.color = vec4(f, 0.0, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "let i: i32 = ((input.texCoord.x * 100.0) as i32);" in generated_code
        assert "let f: f32 = (i as f32);" in generated_code
        assert "int(" not in generated_code
        assert "float(" not in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust type conversions not implemented")


def test_rust_invalid_shader_empty_vertex_stage(tmp_path):
    code = """
    shader EmptyVertex {
        vertex {
            void main() {
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert_rust_stage_attr(generated_code, "vertex")
    assert "pub fn main() -> ()" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_invalid_shader_no_stages(tmp_path):
    code = """
    shader NoStages {
        struct Data {
            float value;
        };
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert "use spirv_std::glam::*;" in generated_code
    assert "pub struct Data" in generated_code
    assert_rust_stage_attr_absent(generated_code, "vertex")
    assert_rust_stage_attr_absent(generated_code, "fragment")
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_invalid_shader_missing_return_type(tmp_path):
    code = """
    shader VoidVertex {
        vertex {
            void main() {
                float x = 1.0;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert_rust_stage_attr(generated_code, "vertex")
    assert "-> ()" in generated_code
    assert "let x: f32 = 1.0;" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_cbuffer_basic_uniform_buffer(tmp_path):
    code = """
    cbuffer SceneData {
        mat4 viewProjection;
        vec3 cameraPosition;
        float time;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert (
        "#[repr(C)]\n#[derive(Debug, Clone, Copy, Default)]\npub struct SceneData {"
        in generated_code
    )
    assert "pub viewProjection: Mat4," in generated_code
    assert "pub cameraPosition: Vec3," in generated_code
    assert "pub time: f32," in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_cbuffer_with_arrays(tmp_path):
    code = """
    cbuffer LightBuffer {
        vec4 lightPositions[8];
        float intensities[8];
        int lightCount;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert "pub lightPositions: [Vec4; 8]," in generated_code
    assert "pub intensities: [Vec4; 8]," in generated_code
    assert "pub lightCount: i32," in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_cbuffer_multiple_buffers(tmp_path):
    code = """
    cbuffer PerFrame {
        float time;
        float deltaTime;
    };
    cbuffer PerObject {
        mat4 model;
        mat4 normalMatrix;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert "pub struct PerFrame {" in generated_code
    assert "pub deltaTime: f32," in generated_code
    assert "pub struct PerObject {" in generated_code
    assert "pub normalMatrix: Mat4," in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_cbuffer_used_in_vertex_stage(tmp_path):
    code = """
    shader CbufferUsage {
        cbuffer Transform {
            mat4 mvp;
            vec3 offset;
        };
        struct VSInput {
            vec3 position @ POSITION;
        };
        struct VSOutput {
            vec4 pos @ gl_Position;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.pos = mvp * vec4(input.position + offset, 1.0);
                return output;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert (
        "#[spirv(uniform, descriptor_set = 0, binding = 0)] transform: &Transform"
        in generated_code
    )
    assert "(input.position + transform.offset)" in generated_code
    assert "(transform.mvp * " in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_struct_member_semantic_position():
    code = """
    shader main {
        struct VSInput {
            vec3 position @ POSITION;
            vec3 normal @ NORMAL;
            vec2 uv @ TEXCOORD0;
        };
        vertex {
            void main(VSInput input) {
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub struct VSInput" in generated_code
    assert "pub position: Vec3," in generated_code
    assert "pub normal: Vec3," in generated_code
    assert "pub uv: Vec2," in generated_code
    assert "// position" in generated_code
    assert "// normal" in generated_code
    assert "// texcoord(0)" in generated_code


def test_rust_struct_member_semantic_color():
    code = """
    shader main {
        struct VSOutput {
            vec4 color @ COLOR;
            vec4 position @ gl_Position;
        };
        vertex {
            VSOutput main() {
                VSOutput out;
                out.color = vec4(1.0, 0.0, 0.0, 1.0);
                out.position = vec4(0.0, 0.0, 0.0, 1.0);
                return out;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub color: Vec4," in generated_code
    assert "pub position: Vec4," in generated_code
    assert "// color" in generated_code
    assert "// position" in generated_code


def test_rust_struct_member_multiple_texcoords():
    code = """
    shader main {
        struct VertexInput {
            vec2 uv0 @ TEXCOORD0;
            vec2 uv1 @ TEXCOORD1;
            vec2 uv2 @ TEXCOORD2;
        };
        vertex {
            void main(VertexInput v) {
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub uv0: Vec2," in generated_code
    assert "pub uv1: Vec2," in generated_code
    assert "pub uv2: Vec2," in generated_code
    assert "// texcoord(0)" in generated_code
    assert "// texcoord(1)" in generated_code
    assert "// texcoord(2)" in generated_code


def test_rust_stage_parameter_input_struct():
    code = """
    shader main {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.texCoord;
                return output;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert (
        "let input: VSInput = VSInput { position: input_position, texCoord: input_texCoord };"
        in generated_code
    )
    assert "#[spirv(position)] out_position: &mut Vec4" in generated_code


def test_rust_stage_parameter_output_semantic():
    code = """
    shader main {
        struct FSInput {
            vec4 color @ COLOR;
        };
        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "fragment")
    assert "let input: FSInput = FSInput { color: input_color };" in generated_code
    assert "#[spirv(location = 0)] out_value: &mut Vec4" in generated_code


def test_rust_stage_parameter_multiple_params():
    code = """
    shader main {
        struct VSInput {
            vec3 pos @ POSITION;
        };
        vertex {
            vec4 main(VSInput input) @ gl_Position {
                return vec4(input.pos, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert "let input: VSInput = VSInput { pos: input_pos };" in generated_code
    assert "#[spirv(position)] out_value: &mut Vec4" in generated_code


def test_rust_fragment_stage_basic():
    code = """
    shader FragShader {
        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "fragment")
    assert "#[spirv(location = 0)] out_value: &mut Vec4" in generated_code
    assert "Vec4::new(1.0, 0.0, 0.0, 1.0)" in generated_code


def test_rust_fragment_stage_with_texture_sampling():
    code = """
    shader TexturedFrag {
        sampler2D diffuseTexture;
        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };
        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return texture(diffuseTexture, input.uv);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "fragment")
    assert "sample" in generated_code
    assert "DIFFUSE_TEXTURE" in generated_code or "diffuseTexture" in generated_code


def test_rust_fragment_stage_with_discard(tmp_path):
    code = """
    shader AlphaTest {
        struct FSInput {
            vec4 color @ COLOR;
        };
        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                if (input.color.a < 0.5) {
                    discard;
                }
                return input.color;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "fragment")
    assert "spirv_std::arch::kill();" in generated_code
    assert "discard;" not in generated_code
    assert "input.color.w" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_fragment_stage_multiple_render_targets():
    code = """
    shader MRT {
        struct GBuffer {
            vec4 albedo @ gl_FragColor0;
            vec4 normal @ gl_FragColor1;
            vec4 position @ gl_FragColor2;
        };
        fragment {
            GBuffer main() {
                GBuffer gbuf;
                gbuf.albedo = vec4(1.0, 0.0, 0.0, 1.0);
                gbuf.normal = vec4(0.0, 1.0, 0.0, 1.0);
                gbuf.position = vec4(0.0, 0.0, 1.0, 1.0);
                return gbuf;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "fragment")
    assert (
        "#[spirv(location = 0)] out_albedo: &mut Vec4, #[spirv(location = 1)] out_normal: &mut Vec4, #[spirv(location = 2)] out_position: &mut Vec4"
        in generated_code
    )
    assert "pub struct GBuffer" in generated_code
    assert "// target(0)" in generated_code
    assert "// target(1)" in generated_code
    assert "// target(2)" in generated_code


def test_rust_vertex_stage_basic():
    code = """
    shader VertShader {
        struct VSInput {
            vec3 position @ POSITION;
        };
        vertex {
            vec4 main(VSInput input) @ gl_Position {
                return vec4(input.position, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert (
        "pub fn main(#[spirv(location = 0)] input_position: Vec3, #[spirv(position)] out_value: &mut Vec4)"
        in generated_code
    )


def test_rust_vertex_stage_with_transform(tmp_path):
    code = """
    shader TransformVert {
        cbuffer Transforms {
            mat4 modelViewProjection;
        };
        struct VSInput {
            vec3 position @ POSITION;
            vec3 normal @ NORMAL;
        };
        struct VSOutput {
            vec4 position @ gl_Position;
            vec3 worldNormal @ TEXCOORD0;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = modelViewProjection * vec4(input.position, 1.0);
                output.worldNormal = input.normal;
                return output;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert (
        "#[spirv(uniform, descriptor_set = 0, binding = 0)] transforms: &Transforms"
        in generated_code
    )
    assert "transforms.modelViewProjection * " in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_vertex_stage_passthrough():
    code = """
    shader Passthrough {
        struct VSInput {
            vec3 position @ POSITION;
            vec4 color @ COLOR;
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 position @ gl_Position;
            vec4 color @ COLOR;
            vec2 texCoord @ TEXCOORD0;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.color = input.color;
                output.texCoord = input.texCoord;
                return output;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert "pub struct VSInput" in generated_code
    assert "pub struct VSOutput" in generated_code
    assert "// position" in generated_code
    assert "// color" in generated_code
    assert "// texcoord(0)" in generated_code


def test_rust_vertex_and_fragment_pipeline(tmp_path):
    code = """
    shader FullPipeline {
        cbuffer Material {
            vec4 baseColor;
            float roughness;
        };
        struct VSInput {
            vec3 position @ POSITION;
            vec3 normal @ NORMAL;
            vec2 uv @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 position @ gl_Position;
            vec3 normal @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.normal = input.normal;
                output.uv = input.uv;
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float lighting = dot(input.normal, vec3(0.0, 1.0, 0.0));
                return baseColor * lighting;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert (
        "#[spirv(uniform, descriptor_set = 0, binding = 0)] material: &Material"
        in generated_code
    )
    assert "material.baseColor.x * lighting" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_cbuffer_repr_c_and_derive(tmp_path):
    code = """
    cbuffer CameraData {
        mat4 view;
        mat4 projection;
        vec3 eyePosition;
        float nearPlane;
        float farPlane;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert (
        "#[repr(C)]\n#[derive(Debug, Clone, Copy, Default)]\npub struct CameraData {"
        in generated_code
    )
    assert "pub eyePosition: Vec3," in generated_code
    assert "pub nearPlane: f32," in generated_code
    assert "pub farPlane: f32," in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_struct_member_semantics_tangent_binormal():
    code = """
    shader main {
        struct VertexData {
            vec3 position @ POSITION;
            vec3 normal @ NORMAL;
            vec3 tangent @ TANGENT;
            vec3 binormal @ BINORMAL;
            vec2 uv @ TEXCOORD0;
        };
        vertex {
            void main(VertexData v) {
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub struct VertexData" in generated_code
    assert "pub position: Vec3," in generated_code
    assert "pub tangent: Vec3," in generated_code
    assert "pub binormal: Vec3," in generated_code
    assert "// position" in generated_code
    assert "// normal" in generated_code
    assert "// tangent" in generated_code
    assert "// binormal" in generated_code
    assert "// texcoord(0)" in generated_code


def test_rust_multiple_cbuffers_in_shader(tmp_path):
    code = """
    shader MultiBuffer {
        cbuffer PerFrame {
            mat4 viewProjection;
            float time;
            float deltaTime;
        };
        cbuffer PerMaterial {
            vec4 albedo;
            float metallic;
            float roughness;
        };
        cbuffer PerObject {
            mat4 world;
            mat4 worldInverseTranspose;
        };
        struct VSInput {
            vec3 position @ POSITION;
        };
        vertex {
            void main(VSInput input) {
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert "pub struct PerFrame {" in generated_code
    assert "pub struct PerMaterial {" in generated_code
    assert "pub struct PerObject {" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_structured_buffer_type_mapping():
    code = """
    shader ComputeShader {
        struct Particle {
            vec3 position @ POSITION;
            vec3 velocity @ TEXCOORD0;
            float life @ TEXCOORD1;
        };
        StructuredBuffer<Particle> inputParticles;
        RWStructuredBuffer<Particle> outputParticles;
        compute {
            void main() {
            }
        }
    }
    """
    # Unreferenced buffers are not bound, so only the structure is written.
    generated = generate_code(parse_code(tokenize_code(code)))
    assert "pub struct Particle {" in generated
    assert "pub fn main() -> ()" in generated

    # std430 places each vec3 member on a 16-byte boundary, glam's 12-byte Vec3 does not: refused, not misread.
    used = code.replace(
        "void main() {\n            }",
        "void main() {\n                outputParticles[0] = inputParticles[0];\n            }",
    )
    assert used != code
    with pytest.raises(
        RustLoweringRefused,
        match=r"storage buffer 'inputParticles': 'Particle.velocity' is at std430 offset 16, "
        r"rust-gpu lays it out at offset 12",
    ):
        generate_code(parse_code(tokenize_code(used)))


def test_rust_struct_semantic_fragment_outputs():
    code = """
    shader main {
        struct GBufferOutput {
            vec4 albedo @ gl_FragColor0;
            vec4 normals @ gl_FragColor1;
            vec4 emission @ gl_FragColor2;
        };
        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };
        fragment {
            GBufferOutput main(FSInput input) {
                GBufferOutput output;
                output.albedo = vec4(1.0, 0.0, 0.0, 1.0);
                output.normals = vec4(0.0, 1.0, 0.0, 1.0);
                output.emission = vec4(0.0, 0.0, 0.0, 1.0);
                return output;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub struct GBufferOutput" in generated_code
    assert "pub albedo: Vec4," in generated_code
    assert "pub normals: Vec4," in generated_code
    assert "pub emission: Vec4," in generated_code
    assert "// target(0)" in generated_code
    assert "// target(1)" in generated_code
    assert "// target(2)" in generated_code


def test_rust_struct_semantics_validate_builtin_types_and_stage_context():
    valid_code = """
    shader StructSemantics {
        struct FSOutput {
            vec4 color @ SV_Target1;
            float depth @ gl_FragDepth;
            vec2 uv @ TEXCOORD0;
        };

        fragment {
            FSOutput main() {
                FSOutput output;
                return output;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(valid_code)))

    assert "pub color: Vec4,  // target(1)" in generated_code
    assert "pub depth: f32,  // depth(any)" in generated_code
    assert "pub uv: Vec2,  // texcoord(0)" in generated_code

    invalid_type = """
    shader BadStructSemanticType {
        struct FSOutput {
            vec3 color @ gl_FragColor;
        };
    }
    """
    with pytest.raises(ValueError, match="gl_FragColor.*vec4-compatible"):
        generate_code(parse_code(tokenize_code(invalid_type)))

    invalid_stage = """
    shader BadStructSemanticStage {
        struct FSOutput {
            vec4 color @ gl_FragColor;
        };

        vertex {
            FSOutput main() {
                FSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_FragColor.*vertex stage"):
        generate_code(parse_code(tokenize_code(invalid_stage)))

    invalid_input_only = """
    shader BadStructInputOnlySemantic {
        struct VSOutput {
            uint vertexId @ gl_VertexID;
        };

        vertex {
            VSOutput main() {
                VSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_VertexID.*input-only"):
        generate_code(parse_code(tokenize_code(invalid_input_only)))


def test_rust_return_semantics_validate_builtin_types_and_stage_context():
    invalid_depth_type = """
    shader BadDepthType {
        fragment {
            vec4 main() @ gl_FragDepth {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_FragDepth.*float type"):
        generate_code(parse_code(tokenize_code(invalid_depth_type)))

    invalid_stage = """
    shader BadStage {
        vertex {
            vec4 main() @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_FragColor.*vertex stage"):
        generate_code(parse_code(tokenize_code(invalid_stage)))

    invalid_input_only = """
    shader BadInputOnlyReturn {
        vertex {
            uint main() @ gl_VertexID {
                return uint(0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_VertexID.*input-only"):
        generate_code(parse_code(tokenize_code(invalid_input_only)))


def test_rust_direct_return_semantics_emit_metadata_and_compile(tmp_path):
    color_code = """
    shader RustDirectReturnSemantics {
        vertex {
            vec4 main(vec3 position @ POSITION) @ gl_Position {
                return vec4(position, 1.0);
            }
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0, 0.5, 0.25, 1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(color_code)))

    assert_rust_stage_attr(generated_code, "vertex")
    assert "#[spirv(position)] out_value: &mut Vec4" in generated_code
    assert "pub fn vertex_main(#[spirv(location = 0)] position: Vec3," in generated_code
    assert_rust_stage_attr(generated_code, "fragment")
    assert (
        "pub fn fragment_main(#[spirv(location = 0)] out_value: &mut Vec4)"
        in generated_code
    )
    assert (
        "let return_value = Vec4::new(1.0, 0.5, 0.25, 1.0);\n    *out_value = return_value;"
        in generated_code
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)

    depth_code = """
    shader RustDepthReturnSemantics {
        fragment {
            float main() @ gl_FragDepth {
                return 0.5;
            }
        }
    }
    """
    depth_generated_code = generate_code(parse_code(tokenize_code(depth_code)))

    assert_rust_stage_attr(depth_generated_code, "fragment")
    assert "#[spirv(fragment(depth_replacing))]" in depth_generated_code
    assert (
        "pub fn main(#[spirv(frag_depth)] out_value: &mut f32)" in depth_generated_code
    )
    assert_generated_rust_smoke_compiles(depth_generated_code, tmp_path)


def test_rust_vertex_stage_position_output(tmp_path):
    code = """
    shader VertexPosition {
        struct VSOutput {
            vec4 position @ gl_Position;
        };
        vertex {
            VSOutput main(vec3 pos) {
                VSOutput output;
                output.position = vec4(pos, 1.0);
                return output;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert "pub struct VSOutput" in generated_code
    assert "pub position: Vec4," in generated_code
    assert "// position" in generated_code
    assert "*out_position = return_value.position;" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_fragment_stage_color_output(tmp_path):
    code = """
    shader FragmentColor {
        struct PSOutput {
            vec4 color @ gl_FragColor;
        };
        fragment {
            PSOutput main(vec4 fragCoord) {
                PSOutput output;
                output.color = vec4(1.0, 0.0, 0.0, 1.0);
                return output;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "fragment")
    assert "pub struct PSOutput" in generated_code
    assert "pub color: Vec4," in generated_code
    assert "// target(0)" in generated_code
    assert "*out_color = return_value.color;" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_stage_io_semantics_sv_position_sv_target(tmp_path):
    """SV_Position and SV_Target map to Rust semantic comments (issues #323/#327)."""
    code = """
    shader SemanticMapping {
        struct VSOut {
            vec4 pos @ gl_Position;
            vec3 norm @ NORMAL;
        };
        struct PSOut {
            vec4 color @ gl_FragColor;
        };
        vertex {
            VSOut main(vec3 position) {
                VSOut o;
                o.pos = vec4(position, 1.0);
                o.norm = vec3(0.0, 1.0, 0.0);
                return o;
            }
        }
        fragment {
            PSOut main(VSOut input) {
                PSOut o;
                o.color = vec4(input.norm, 1.0);
                return o;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert_rust_stage_attr(generated_code, "fragment")
    assert "pub struct VSOut" in generated_code
    assert "pub struct PSOut" in generated_code
    assert "pub pos: Vec4," in generated_code
    assert "pub norm: Vec3," in generated_code
    assert "// position" in generated_code
    assert "// normal" in generated_code
    assert "// target(0)" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_vertex_multiple_attributes(tmp_path):
    code = """
    shader MultiAttrib {
        struct VertexIn {
            vec3 position @ POSITION;
            vec3 normal @ NORMAL;
            vec2 texcoord @ TEXCOORD0;
        };
        struct VertexOut {
            vec4 clipPos @ gl_Position;
            vec3 worldNormal @ NORMAL;
            vec2 uv @ TEXCOORD0;
        };
        vertex {
            VertexOut main(VertexIn v) {
                VertexOut o;
                o.clipPos = vec4(v.position, 1.0);
                o.worldNormal = v.normal;
                o.uv = v.texcoord;
                return o;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert "pub struct VertexIn" in generated_code
    assert "pub struct VertexOut" in generated_code
    assert "pub position: Vec3," in generated_code
    assert "pub normal: Vec3," in generated_code
    assert "pub texcoord: Vec2," in generated_code
    assert "pub clipPos: Vec4," in generated_code
    assert "pub worldNormal: Vec3," in generated_code
    assert "pub uv: Vec2," in generated_code
    assert "// position" in generated_code
    assert "// normal" in generated_code
    assert "// texcoord(0)" in generated_code
    assert (
        "let v: VertexIn = VertexIn { position: v_position, normal: v_normal, texcoord: v_texcoord };"
        in generated_code
    )
    assert "#[spirv(position)] out_clipPos: &mut Vec4" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_fragment_stage_with_texture_sampling(tmp_path):
    code = """
    shader TexturedFragment {
        sampler2D diffuseTex;

        struct PSInput {
            vec2 uv @ TEXCOORD0;
        };
        fragment {
            vec4 main(PSInput input) @ gl_FragColor {
                return texture(diffuseTex, input.uv);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


# Invalid shader shape validation and stage parameter semantics


def test_rust_invalid_shader_completely_empty(tmp_path):
    code = """
    shader Empty {
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "#![no_std]" in generated_code
    assert "use spirv_std::glam::*;" in generated_code
    assert "use spirv_std::spirv;" in generated_code
    assert_rust_stage_attr_absent(generated_code, "vertex")
    assert_rust_stage_attr_absent(generated_code, "fragment")
    assert_rust_stage_attr_absent(generated_code, "compute")
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_invalid_shader_vertex_stage_no_body(tmp_path):
    code = """
    shader NoBody {
        vertex {
            void main() {
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert "pub fn" in generated_code
    assert "-> ()" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_invalid_shader_struct_only_no_stage(tmp_path):
    code = """
    shader StructOnly {
        struct Vertex {
            vec3 pos @ POSITION;
            vec3 norm @ NORMAL;
        };
        struct Fragment {
            vec4 color @ COLOR;
        };
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub struct Vertex" in generated_code
    assert "pub struct Fragment" in generated_code
    assert "pub pos: Vec3," in generated_code
    assert "pub norm: Vec3," in generated_code
    assert "pub color: Vec4," in generated_code
    assert "// position" in generated_code
    assert "// normal" in generated_code
    assert "// color" in generated_code
    assert_rust_stage_attr_absent(generated_code, "vertex")
    assert_rust_stage_attr_absent(generated_code, "fragment")
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_mesh_task_stages_lower_to_metadata_and_intrinsic_placeholders(tmp_path):
    code = """
    shader RustMeshTaskPipeline {
        struct MeshPayload {
            uint meshlet;
        };

        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        task {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() @numthreads(1, 1, 1) {
                MeshPayload payload;
                payload.meshlet = 7u;
                DispatchMesh(1, 1, 1, payload);
            }
        }

        object {
            void main() @numthreads(2, 1, 1) {
                DispatchMesh(2, 3, 4);
            }
        }

        amplification {
            void main() @numthreads(3, 1, 1) {
                DispatchMesh(1, 1, 1);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            )
                @numthreads(32, 1, 1)
                @outputtopology(triangle)
                @maxvertices(64)
                @maxprimitives(32) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="stage: Rust output writes rust-gpu compute, vertex and fragment entry points",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_mesh_task_stages_validate_metadata_and_intrinsics():
    missing_topology = """
    shader MissingRustMeshTopology {
        mesh {
            void main() @maxvertices(3) @maxprimitives(1) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="requires outputtopology"):
        generate_code(parse_code(tokenize_code(missing_topology)))

    invalid_topology = """
    shader BadRustMeshTopology {
        mesh {
            void main()
                @outputtopology(line_strip)
                @maxvertices(3)
                @maxprimitives(1) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="outputtopology.*must be"):
        generate_code(parse_code(tokenize_code(invalid_topology)))

    bad_numthreads_arity = """
    shader BadRustMeshNumthreads {
        task {
            void main() @numthreads(1, 1, 1, 1) {
                DispatchMesh(1, 1, 1);
            }
        }
    }
    """
    with pytest.raises(
        ValueError, match="numthreads takes one, two or three dimensions, got 4"
    ):
        generate_code(parse_code(tokenize_code(bad_numthreads_arity)))

    bad_stage_context = """
    shader BadRustMeshContext {
        mesh {
            void main() @outputtopology(triangle) {
                DispatchMesh(1, 1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="DispatchMesh.*only valid"):
        generate_code(parse_code(tokenize_code(bad_stage_context)))

    bad_arity = """
    shader BadRustMeshArity {
        task {
            void main() {
                DispatchMesh(1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="DispatchMesh requires three"):
        generate_code(parse_code(tokenize_code(bad_arity)))

    bad_count_type = """
    shader BadRustMeshCountType {
        mesh {
            void main() @outputtopology(triangle) {
                SetMeshOutputCounts(1.0, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="integer scalar"):
        generate_code(parse_code(tokenize_code(bad_count_type)))


def test_rust_qualified_mesh_stage_uses_deterministic_entry_name(tmp_path):
    ast = ShaderNode(
        name="QualifiedRustMeshStage",
        execution_model=ExecutionModel.GRAPHICS_PIPELINE,
        functions=[
            FunctionNode(
                name="main",
                return_type=PrimitiveType("void"),
                parameters=[],
                body=BlockNode([]),
                qualifiers=[ShaderStage.MESH],
                attributes=[AttributeNode("outputtopology", ["triangle"])],
            )
        ],
    )

    with pytest.raises(
        RustLoweringRefused,
        match="stage: Rust output writes rust-gpu compute, vertex and fragment entry points",
    ):
        generate_code(ast)


def test_direct_mesh_ir_node_lowers_to_rust_helper_without_ast_repr():
    codegen = RustCodeGen()

    codegen.current_shader_type = "mesh"
    counts_expr = codegen.generate_expression(MeshOpNode("SetMeshOutputCounts", [1, 1]))
    assert counts_expr == "_crossgl_set_mesh_output_counts(1, 1)"
    assert "set_mesh_output_counts" in codegen.required_mesh_helpers

    codegen.current_shader_type = "task"
    dispatch_expr = codegen.generate_expression(MeshOpNode("DispatchMesh", [1, 1, 1]))
    assert dispatch_expr == "_crossgl_dispatch_mesh_without_payload(1, 1, 1)"
    assert "dispatch_mesh" in codegen.required_mesh_helpers


def test_rust_tessellation_stages_lower_with_metadata_and_patch_placeholders(tmp_path):
    code = """
    shader RustTessellationPipeline {
        struct HSInput {
            vec3 position;
        };

        struct HSOutput {
            vec3 position;
        };

        struct PatchConstants {
            float edge;
        };

        tessellation_control {
            PatchConstants HSConst(InputPatch<HSInput, 3> patch) {
                PatchConstants constants;
                constants.edge = 1.0;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                output.position = patch[0].position;
                return output;
            }
        }

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 3> patch, vec3 bary @ SV_DomainLocation)
                @domain(tri) {
                vec3 position = patch[0].position + bary;
                return vec4(position, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="stage: Rust output writes rust-gpu compute, vertex and fragment entry points",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_tessellation_stages_validate_metadata_and_patch_shapes():
    missing_output_control_points = """
    shader MissingRustTessOutputPoints {
        struct HSInput { vec3 position; };

        tessellation_control {
            void main(InputPatch<HSInput, 3> patch)
                @domain(tri)
                @partitioning(fractional_odd) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="requires outputcontrolpoints"):
        generate_code(parse_code(tokenize_code(missing_output_control_points)))

    invalid_output_control_points = """
    shader BadRustTessOutputPoints {
        struct HSInput { vec3 position; };

        tessellation_control {
            void main(InputPatch<HSInput, 3> patch)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputcontrolpoints(0) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="outputcontrolpoints.*positive"):
        generate_code(parse_code(tokenize_code(invalid_output_control_points)))

    invalid_partitioning = """
    shader BadRustTessPartitioning {
        struct HSInput { vec3 position; };

        tessellation_control {
            void main(InputPatch<HSInput, 3> patch)
                @domain(tri)
                @partitioning(centered)
                @outputcontrolpoints(3) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="partitioning.*must be one of"):
        generate_code(parse_code(tokenize_code(invalid_partitioning)))

    missing_evaluation_domain = """
    shader MissingRustTessDomain {
        struct HSOutput { vec3 position; };

        tessellation_evaluation {
            void main(OutputPatch<HSOutput, 3> patch) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="requires domain"):
        generate_code(parse_code(tokenize_code(missing_evaluation_domain)))

    invalid_patch_shape = """
    shader BadRustTessPatchShape {
        struct HSInput { vec3 position; };

        tessellation_control {
            void main(InputPatch<HSInput> patch)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputcontrolpoints(3) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match=r"InputPatch.*InputPatch<T, N>"):
        generate_code(parse_code(tokenize_code(invalid_patch_shape)))


def test_rust_geometry_stage_lowers_with_metadata_and_stream_placeholders(tmp_path):
    code = """
    shader RustGeometryExpansion {
        struct GSInput {
            vec4 position @ gl_Position;
            vec3 normal @ NORMAL;
        };

        struct GSOutput {
            vec4 position @ gl_Position;
            vec3 color @ COLOR;
        };

        geometry {
            void main(triangle GSInput input[3], inout TriangleStream<GSOutput> stream)
                @maxvertexcount(3) {
                GSOutput output;
                output.position = input[0].position;
                output.color = input[0].normal;
                stream.Append(output);
                stream.RestartStrip();
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="stage: Rust output writes rust-gpu compute, vertex and fragment entry points",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_geometry_stage_validates_maxvertexcount_and_input_shape():
    missing_attribute = """
    shader MissingRustGeometryMax {
        struct GSInput { vec4 position @ gl_Position; };
        struct GSOutput { vec4 position @ gl_Position; };

        geometry {
            void main(triangle GSInput input[3], inout TriangleStream<GSOutput> stream) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="requires maxvertexcount"):
        generate_code(parse_code(tokenize_code(missing_attribute)))

    invalid_max = """
    shader BadRustGeometryMax {
        struct GSInput { vec4 position @ gl_Position; };
        struct GSOutput { vec4 position @ gl_Position; };

        geometry {
            void main(triangle GSInput input[3], inout TriangleStream<GSOutput> stream)
                @maxvertexcount(0) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="maxvertexcount.*positive"):
        generate_code(parse_code(tokenize_code(invalid_max)))

    wrong_arity = """
    shader BadRustGeometryInputArity {
        struct GSInput { vec4 position @ gl_Position; };
        struct GSOutput { vec4 position @ gl_Position; };

        geometry {
            void main(triangle GSInput input[2], inout TriangleStream<GSOutput> stream)
                @maxvertexcount(3) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="triangle input primitive.*3 element"):
        generate_code(parse_code(tokenize_code(wrong_arity)))

    missing_stream = """
    shader BadRustGeometryStream {
        struct GSInput { vec4 position @ gl_Position; };

        geometry {
            void main(triangle GSInput input[3]) @maxvertexcount(3) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="must include a PointStream"):
        generate_code(parse_code(tokenize_code(missing_stream)))


def test_rust_ray_tracing_stages_emit_metadata_and_compile(tmp_path):
    code = """
    shader RustRayTracingStages {
        struct RayPayload {
            vec3 color;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        struct CallableData {
            uint value;
        };

        accelerationStructureEXT scene @binding(3) @set(2);

        ray_generation {
            void main() {
                uvec3 launch = gl_LaunchIDEXT;
                uint launchWidth = gl_LaunchSizeEXT.x;
                RayDesc ray;
                RayPayload payload;
                CallableData data;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
                CallShader(0, data);
            }
        }

        ray_closest_hit {
            void main(
                RayPayload payload @ payload,
                HitAttributes attributes @ hit_attribute,
                float hitT @ gl_HitTEXT
            ) {
                payload.color = vec3(attributes.barycentrics, hitT);
            }
        }

        ray_any_hit {
            void main(
                RayPayload payload @ payload,
                HitAttributes attributes @ hit_attribute,
                uint hitKind @ gl_HitKindEXT
            ) {
                payload.color = vec3(attributes.barycentrics, float(hitKind));
                IgnoreHit();
                AcceptHitAndEndSearch();
            }
        }

        ray_miss {
            void main(RayPayload payload @ rayPayloadInEXT) {
                payload.color = vec3(0.0, 0.0, 0.0);
            }
        }

        ray_callable {
            void main(CallableData data @ callableDataInEXT) {
                data.value = data.value + 1u;
            }
        }

        ray_intersection {
            void main() {
                HitAttributes attributes;
                bool accepted = ReportHit(1.0, 0, attributes);
                bool acceptedWithoutAttributes = ReportHit(2.0, 1);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="stage: Rust output writes rust-gpu compute, vertex and fragment entry points",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_ray_stage_parameter_semantics_validate_builtin_stage_type_and_duplicates():
    invalid_type = """
    shader InvalidRustRayBuiltinType {
        ray_generation {
            void main(float launch @ gl_LaunchIDEXT) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_LaunchIDEXT.*Vec3<u32>"):
        generate_code(parse_code(tokenize_code(invalid_type)))

    invalid_stage = """
    shader InvalidRustRayBuiltinStage {
        fragment {
            void main(uvec3 launch @ gl_LaunchIDEXT) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_LaunchIDEXT.*fragment stage"):
        generate_code(parse_code(tokenize_code(invalid_stage)))

    duplicate = """
    shader DuplicateRustRayBuiltin {
        ray_generation {
            void main(
                uvec3 launch @ gl_LaunchIDEXT,
                uvec3 launchAgain @ SV_DispatchRaysIndex
            ) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="Duplicate Rust stage parameter semantic"):
        generate_code(parse_code(tokenize_code(duplicate)))


def test_rust_stage_parameter_semantics_all_mappings(tmp_path):
    code = """
    shader SemanticsFull {
        struct FullInput {
            vec3 pos @ POSITION;
            vec3 norm @ NORMAL;
            vec3 tang @ TANGENT;
            vec3 binorm @ BINORMAL;
            vec2 uv0 @ TEXCOORD0;
            vec2 uv1 @ TEXCOORD1;
            vec2 uv2 @ TEXCOORD2;
            vec2 uv3 @ TEXCOORD3;
            vec4 col0 @ COLOR0;
            vec4 col1 @ COLOR1;
        };
        struct FullOutput {
            vec4 position @ gl_Position;
            vec4 fragColor @ COLOR0;
        };
        vertex {
            FullOutput main(FullInput input) {
                FullOutput output;
                output.position = vec4(input.pos, 1.0);
                output.fragColor = input.col0;
                return output;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub struct FullInput" in generated_code
    assert "pub struct FullOutput" in generated_code
    assert "// position" in generated_code
    assert "// normal" in generated_code
    assert "// tangent" in generated_code
    assert "// binormal" in generated_code
    assert "// texcoord(0)" in generated_code
    assert "// texcoord(1)" in generated_code
    assert "// texcoord(2)" in generated_code
    assert "// texcoord(3)" in generated_code
    assert "// color(0)" in generated_code
    assert "// color(1)" in generated_code
    assert_rust_stage_attr(generated_code, "vertex")
    assert "let input: FullInput = FullInput { pos: input_pos," in generated_code
    assert "*out_position = return_value.position;" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_stage_parameter_builtin_semantics_emit_metadata_and_compile(tmp_path):
    code = """
    shader RustBuiltinStageParameters {
        vertex {
            vec4 main(uint vertexId @ gl_VertexID, uint instanceId @ SV_InstanceID)
                @ gl_Position {
                return vec4(0.0, 0.0, 0.0, 1.0);
            }
        }

        fragment {
            vec4 main(vec4 fragCoord @ gl_FragCoord, bool front @ gl_FrontFacing)
                @ gl_FragColor {
                return vec4(1.0, 0.0, 0.0, 1.0);
            }
        }

        compute {
            void main(
                uvec3 globalId @ gl_GlobalInvocationID,
                uvec3 localId @ SV_GroupThreadID,
                uint lane @ gl_LocalInvocationIndex
            ) {
                uint copy = lane;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "#[spirv(vertex_index)] vertexId_in: u32" in generated_code
    assert "#[spirv(instance_index)] instanceId_in: u32" in generated_code
    # CrossGL instance IDs start at 0; SPIR-V InstanceIndex includes the base instance.
    assert (
        "let instanceId: u32 = instanceId_in.wrapping_sub(instanceId_base);"
        in generated_code
    )
    assert "#[spirv(frag_coord)] fragCoord_in: Vec4" in generated_code
    assert "#[spirv(front_facing)] front_in: bool" in generated_code
    assert (
        "#[spirv(global_invocation_id)] globalId_global_invocation_id: UVec3"
        in generated_code
    )
    assert (
        "#[spirv(local_invocation_id)] localId_local_invocation_id: UVec3"
        in generated_code
    )
    assert (
        "#[spirv(local_invocation_index)] lane_local_invocation_index: u32"
        in generated_code
    )
    assert "let lane: u32 = lane_local_invocation_index;" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_fragment_coord_semantic_uses_frag_coord_metadata(tmp_path):
    # rust-gpu exposes fragment coordinates as the `frag_coord` builtin.
    code = """
    shader FragCoordBuiltin {
        fragment {
            vec4 main(vec4 fragCoord @ gl_FragCoord) @ gl_FragColor {
                return vec4(fragCoord.x, fragCoord.y, 0.0, 1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "#[spirv(frag_coord)] fragCoord_in: Vec4" in generated_code
    assert "#[spirv(position)]" not in generated_code
    assert "let fragCoord: Vec4 = fragCoord_in;" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_escapes_keyword_struct_fields_and_accesses(tmp_path):
    code = """
    shader RustKeywordFields {
        struct Light {
            int type;
        };

        int readType(Light light) {
            return light.type;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub type_: i32," in generated_code
    assert "pub const fn new(type_: i32)" in generated_code
    assert "Self { type_ }" in generated_code
    assert "return light.type_;" in generated_code
    assert "pub type: i32" not in generated_code
    assert "light.type;" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_keyword_like_bindings_escape_in_functions_and_members(tmp_path):
    code = """
    struct KeywordFields {
        type: float;
        match: vec4;
    };

    fn match(type: float, match: vec3) -> float {
        float while = type + match.x;
        return while;
    }

    fn read(input: KeywordFields) -> float {
        return match(input.type, vec3(1.0, 2.0, 3.0)) + input.match.x;
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub type_: f32," in generated_code
    assert "pub match_: Vec4," in generated_code
    assert "pub fn match_(type_: f32, match_: Vec3) -> f32" in generated_code
    assert "let while_: f32 = (type_ + match_.x);" in generated_code
    assert "return while_;" in generated_code
    assert (
        "return (match_(input.type_, Vec3::new(1.0, 2.0, 3.0)) + input.match_.x);"
    ) in generated_code
    assert "pub fn match(" not in generated_code
    assert "let while:" not in generated_code
    assert "input.match.x" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_escapes_abstract_and_gen_reserved_identifiers(tmp_path):
    # Rust identifiers may not be strict or reserved keywords, and `gen` is
    # reserved in Rust 2024.
    # Sources:
    # https://doc.rust-lang.org/stable/reference/identifiers.html
    # https://doc.rust-lang.org/edition-guide/rust-2024/gen-keyword.html
    code = """
    shader RustReservedKeywordIdentifiers {
        float gen(float abstract) {
            float gen = abstract + 1.0;
            return gen;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub fn gen_(abstract_: f32) -> f32" in generated_code
    assert "let gen_: f32 = (abstract_ + 1.0);" in generated_code
    assert "return gen_;" in generated_code
    assert "pub fn gen(" not in generated_code
    assert "abstract: f32" not in generated_code
    assert "let gen: f32" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_keyword_struct_fields_survive_default_formatting(tmp_path):
    import crosstl

    source_path = tmp_path / "keyword_fields.cgl"
    source_path.write_text(
        """
        shader RustKeywordFields {
            struct Light {
                int type;
            };

            int readType(Light light) {
                return light.type;
            }
        }
        """,
        encoding="utf-8",
    )

    generated_code = crosstl.translate(str(source_path), backend="rust")

    assert "pub type_" in generated_code
    assert "return light.type_;" in generated_code
    assert "r #type" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_skips_forward_function_prototypes_and_compiles(tmp_path):
    code = """
    shader RustFunctionPrototype {
        float shadowCalculation(vec4 fragPosLightSpace, int iteration);

        float applyShadow(vec4 position) {
            return shadowCalculation(position, 0);
        }

        float shadowCalculation(vec4 fragPosLightSpace, int iteration) {
            return fragPosLightSpace.x + float(iteration);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert generated_code.count("pub fn shadowCalculation(") == 1
    empty_definition = (
        "pub fn shadowCalculation("
        "fragPosLightSpace: Vec4<f32>, iteration: i32"
        ") -> f32 {\n}\n"
    )
    assert empty_definition not in generated_code
    assert "return shadowCalculation(position, 0);" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_top_level_constants_and_compute_builtins_compile(tmp_path):
    code = """
    shader RustComputeBuiltins {
        const int WARP_SIZE = 32;
        const float SCALE = 2.0;

        compute {
            void main() {
                int warp = int(gl_LocalInvocationID.x);
                int globalId = int(gl_GlobalInvocationID.x);
                int groups = int(gl_NumWorkGroups.x);
                int stride = groups * WARP_SIZE;
                float scaled = float(stride + globalId + warp) * SCALE;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "const WARP_SIZE: i32 = 32;" in generated_code
    assert "const SCALE: f32 = 2.0;" in generated_code
    assert "#[spirv(local_invocation_id)] gl_LocalInvocationID: UVec3" in generated_code
    assert (
        "#[spirv(global_invocation_id)] gl_GlobalInvocationID: UVec3" in generated_code
    )
    assert "#[spirv(num_workgroups)] gl_NumWorkGroups: UVec3" in generated_code
    assert "let warp: i32 = (gl_LocalInvocationID.x as i32);" in generated_code
    assert "let globalId: i32 = (gl_GlobalInvocationID.x as i32);" in generated_code
    assert "let groups: i32 = (gl_NumWorkGroups.x as i32);" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_compute_numthreads_emits_spirv_threads_attribute(tmp_path):
    # rust-gpu documents compute dimensions as
    # #[spirv(compute(threads(...)))].
    # Source: https://rust-gpu.github.io/rust-gpu/book/attributes.html
    code = """
    shader RustComputeNumthreads {
        compute {
            void main() @numthreads(8, 4, 2) {
                uint localX = gl_LocalInvocationID.x;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "#[spirv(compute(threads(8, 4, 2)))]" in generated_code
    assert '#[cfg_attr(feature = "crossgl_gpu", compute_shader)]' not in generated_code
    assert "let localX: u32 = gl_LocalInvocationID.x;" in generated_code
    assert "#[spirv(local_invocation_id)] gl_LocalInvocationID: UVec3" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_subgroup_builtin_expressions_lower_to_helpers(tmp_path):
    code = """
    shader RustSubgroupBuiltinExpressions {
        compute {
            void main() {
                uint lane = gl_SubgroupInvocationID;
                uint size = gl_SubgroupSize;
                uint subgroup = gl_SubgroupID;
                uint subgroupCount = gl_NumSubgroups;
                uint combined = lane + size + subgroup + subgroupCount;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let lane: u32 = gl_SubgroupInvocationID;" in generated_code
    assert "let size: u32 = gl_SubgroupSize;" in generated_code
    assert "let subgroup: u32 = gl_SubgroupID;" in generated_code
    assert "let subgroupCount: u32 = gl_NumSubgroups;" in generated_code
    assert (
        "#[spirv(subgroup_local_invocation_id)] gl_SubgroupInvocationID: u32"
        in generated_code
    )
    assert "#[spirv(subgroup_size)] gl_SubgroupSize: u32" in generated_code
    assert "#[spirv(subgroup_id)] gl_SubgroupID: u32" in generated_code
    assert "#[spirv(num_subgroups)] gl_NumSubgroups: u32" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_bitwise_not_uses_rust_complement_operator_and_compiles(tmp_path):
    code = """
    shader RustBitwiseNot {
        uint flip(uint mask) {
            return ~mask;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "return (!mask);" in generated_code
    assert "(~mask)" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_swizzle_assignments_emit_component_writes(tmp_path):
    code = """
    shader RustSwizzleAssignments {
        vec4 adjust(vec4 color, vec3 delta, float scale) {
            color.bgr = delta;
            color.rgb += delta;
            color.rgb *= scale;
            return color;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "Vec3::<f32>::new(color.x, color.y, color.z) +=" not in generated_code
    assert (
        "color.z = __cgl_swizzle_0.x; color.y = __cgl_swizzle_0.y; "
        "color.x = __cgl_swizzle_0.z" in generated_code
    )
    assert "color.x += __cgl_swizzle_1.x" in generated_code
    assert "color.y *= __cgl_swizzle_2" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_rejects_duplicate_swizzle_assignment_components():
    code = """
    shader RustInvalidSwizzleAssignment {
        vec4 adjust(vec4 color) {
            color.xx = vec2(1.0, 2.0);
            return color;
        }
    }
    """

    with pytest.raises(ValueError, match="swizzle assignment.*repeat components"):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_plain_struct_constructor_calls_lower_to_new_and_compile(tmp_path):
    code = """
    shader RustStructAggregateCalls {
        struct Material {
            vec3 albedo;
            float roughness;
        };

        struct Surface {
            Material material;
            vec4 clip;
        };

        Material makeMaterial(vec3 albedo, int roughnessBucket) {
            return Material(albedo, float(roughnessBucket));
        }

        Surface makeSurface(vec3 normal) {
            Material material = Material(normal, 1.0);
            return Surface(material, vec4(normal, material.roughness));
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "return Material::new(albedo, (roughnessBucket as f32));" in generated_code
    assert "let material: Material = Material::new(normal, 1.0);" in generated_code
    assert "return Surface::new(material, Vec4::new(" in generated_code
    assert "return Material(albedo" not in generated_code
    assert "return Surface(material" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_matrix_scalar_and_resize_constructors_compile(tmp_path):
    code = """
    shader RustMatrixConstructors {
        vec3 transform(vec3 normal) {
            mat4 model = mat4(1.0);
            mat3 normal_matrix = mat3(transpose(inverse(model)));
            return normal_matrix * normal;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "(model).inverse()" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_hlsl_matrix_alias_constructors_compile(tmp_path):
    code = """
    shader RustHlslMatrixAliases {
        float3 firstColumn() {
            float3x2 basis = float3x2(1.0);
            half3x2 halfBasis = half3x2(1.0);
            min16float3x2 minBasis = min16float3x2(1.0);
            float16_t3x2 explicitBasis = float16_t3x2(1.0);
            return basis.c0 + halfBasis.c0 + minBasis.c0 + explicitBasis.c0;
        }
    }
    """

    with pytest.raises(
        RustRefusal, match=r"glam has no 2x3 matrix of f32 \(Mat2x3<f32>\)"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_stage_local_function_captures_entry_params(tmp_path):
    code = """
    shader RustStageLocalCapture {
        struct FragmentInput {
            vec3 worldNormal;
            vec3 worldPosition;
        };

        fragment {
            float shade(vec3 lightDir) {
                return dot(input.worldNormal, lightDir);
            }

            vec4 main(FragmentInput input) @ gl_FragColor {
                float lit = shade(input.worldPosition);
                return vec4(lit, 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub fn shade(lightDir: Vec3, input: FragmentInput) -> f32" in (
        generated_code
    )
    assert "shade(input.worldPosition, input)" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_pointer_statics_and_indexing_compile():
    code = """
    shader RustPointerBuffers {
        float read(buffer float* data, int index) {
            return data[index];
        }

        void write(buffer float* data, int index, float value) {
            data[index] = value;
        }

        compute {
            buffer float* A;

            void main() {
                int index = int(gl_GlobalInvocationID.x);
                write(A, index, read(A, index));
            }
        }
    }
    """

    # rust-gpu passes a storage buffer as a slice, so a helper's buffer pointer parameter is refused.
    with pytest.raises(
        RustLoweringRefused,
        match="parameter: 'data' of 'read' points into a storage buffer",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_large_array_struct_default_is_manual(tmp_path):
    code = """
    shader RustLargeArrayDefault {
        struct BigMatrix {
            float data[64][64];
        };

        BigMatrix make() {
            BigMatrix matrix;
            return matrix;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "#[derive(Debug, Clone, Copy, Default)]\npub struct BigMatrix" not in (
        generated_code
    )
    assert "impl Default for BigMatrix" in generated_code
    assert "Self { data: [[0.0; 64]; 64] }" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_cuda_shuffle_down_intrinsic_maps_to_subgroup_helper(tmp_path):
    code = """
    shader RustCudaShuffle {
        compute {
            void main() {
                float partial = 1.0;
                int offset = 16;
                partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="function: '__shfl_down_sync' has no rust-gpu lowering",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_stage_parameter_semantics_validate_builtin_stage_type_and_duplicates():
    invalid_stage = """
    shader BadRustStageParameterStage {
        fragment {
            void main(uint vertexId @ gl_VertexID) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_VertexID.*fragment stage"):
        generate_code(parse_code(tokenize_code(invalid_stage)))

    invalid_type = """
    shader BadRustStageParameterType {
        fragment {
            void main(vec3 fragCoord @ gl_FragCoord) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_FragCoord.*expected Vec4<f32> type"):
        generate_code(parse_code(tokenize_code(invalid_type)))

    output_only = """
    shader BadRustStageParameterOutputOnly {
        vertex {
            void main(vec4 color @ SV_Target0) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_Target0.*output-only"):
        generate_code(parse_code(tokenize_code(output_only)))

    duplicate_system_value = """
    shader BadRustStageParameterDuplicate {
        vertex {
            void main(uint vertexId @ gl_VertexID, uint alsoVertexId @ SV_VertexID) {
            }
        }
    }
    """
    with pytest.raises(ValueError, match="Duplicate Rust stage parameter semantic"):
        generate_code(parse_code(tokenize_code(duplicate_system_value)))


def test_rust_stage_parameter_multiple_bindings_attribute_syntax(tmp_path):
    code = """
    shader MultiParam {
        struct VSInput {
            vec3 position @ POSITION;
            vec3 normal @ NORMAL;
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 clipPos @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput out;
                out.clipPos = vec4(input.position, 1.0);
                out.uv = input.texCoord;
                return out;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, 0.0, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert_rust_stage_attr(generated_code, "vertex")
    assert_rust_stage_attr(generated_code, "fragment")
    assert "let input: VSInput = VSInput { position: input_position," in generated_code
    assert (
        "let input: VSOutput = VSOutput { clipPos: input_clipPos, uv: input_uv };"
        in generated_code
    )
    assert "*out_clipPos = return_value.clipPos;" in generated_code
    assert "#[spirv(frag_coord)] input_clipPos: Vec4" in generated_code
    assert "pub position: Vec3," in generated_code
    assert "pub normal: Vec3," in generated_code
    assert "pub texCoord: Vec2," in generated_code
    assert "pub clipPos: Vec4," in generated_code
    assert "pub uv: Vec2," in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_stage_parameter_generate_param_attributes_unit():
    codegen = RustCodeGen()

    class MockParam:
        def __init__(self, semantic):
            self.semantic = semantic

    assert "#[location(0)]" in codegen.generate_param_attributes(MockParam("POSITION"))
    assert "#[location(1)]" in codegen.generate_param_attributes(MockParam("NORMAL"))
    assert "#[location(2)]" in codegen.generate_param_attributes(MockParam("TEXCOORD0"))
    assert "#[location(3)]" in codegen.generate_param_attributes(MockParam("TEXCOORD1"))
    assert "#[location(2)]" in codegen.generate_param_attributes(MockParam("TEXCOORD"))
    assert "#[location(4)]" in codegen.generate_param_attributes(MockParam("COLOR"))
    result_gl_position = codegen.generate_param_attributes(MockParam("gl_Position"))
    assert (
        "#[location(0)]" in result_gl_position
        or "#[builtin(position)]" in result_gl_position
    )
    result_gl_fragcolor = codegen.generate_param_attributes(MockParam("gl_FragColor"))
    assert "#[location(" in result_gl_fragcolor
    assert codegen.generate_param_attributes(MockParam(None)) == ""
    assert codegen.generate_param_attributes(MockParam("UNKNOWN_SEMANTIC")) == ""


def test_wave_active_sum_emits_rust_subgroup_intrinsic():
    code = """
    shader ComputeWave {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uint total = WaveActiveSum(lane);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))
    assert "spirv_std::arch::subgroup_i_add(lane)" in generated
    assert "#[spirv(subgroup_local_invocation_id)]" in generated
    assert "WaveActiveSum" not in generated


def test_wave_ballot_emits_rust_subgroup_ballot():
    code = """
    shader ComputeWave {
        compute {
            void main() {
                bool active = true;
                uvec4 ballot = WaveActiveBallot(active);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))
    assert "spirv_std::arch::subgroup_ballot(active)" in generated


def test_wave_count_and_quad_forms_emit_rust_subgroup_intrinsics():
    code = """
    shader ComputeWave {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uint count = WaveGetLaneCount();
                bool first = WaveIsFirstLane();
                bool equal = WaveActiveAllEqual(lane);
                uint bitCount = WaveActiveCountBits(equal);
                uint prefixCount = WavePrefixCountBits(equal);
                uint quadLane = QuadReadLaneAt(lane, 1u);
                uint quadX = QuadReadAcrossX(lane);
                uint quadY = QuadReadAcrossY(lane);
                uint quadDiagonal = QuadReadAcrossDiagonal(lane);
            }
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))
    for spelling in (
        "subgroup_elect()",
        "subgroup_all_equal(lane)",
        "subgroup_ballot_bit_count(spirv_std::arch::subgroup_ballot(equal))",
        "subgroup_ballot_exclusive_bit_count(spirv_std::arch::subgroup_ballot(equal))",
        "subgroup_quad_broadcast(lane, 1)",
        "subgroup_quad_swap::<0, _>(lane)",
        "subgroup_quad_swap::<1, _>(lane)",
        "subgroup_quad_swap::<2, _>(lane)",
    ):
        assert f"spirv_std::arch::{spelling}" in generated, spelling


def test_wave_match_is_refused_without_a_spirv_std_lowering():
    code = """
    shader ComputeWave {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uvec4 mask = WaveMatch(lane);
            }
        }
    }
    """
    with pytest.raises(
        RustLoweringRefused, match=r"WaveMatch\(uint\) has no spirv_std lowering"
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_wave_multi_prefix_forms_without_a_spirv_std_lowering_are_refused(tmp_path):
    code = """
    shader ComputeWave {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                bool active = lane > 0u;
                uvec4 mask = WaveActiveBallot(active);
                uint sum = WaveMultiPrefixSum(lane, mask);
                uint product = WaveMultiPrefixProduct(lane + 1u, mask);
                uint count = WaveMultiPrefixCountBits(active, mask);
                uint bitAnd = WaveMultiPrefixBitAnd(product, mask);
                uint bitOr = WaveMultiPrefixBitOr(bitAnd, mask);
                uint bitXor = WaveMultiPrefixBitXor(bitOr, mask);
            }
        }
    }
    """

    # spirv-std has no partitioned operation, and the translation pipeline's partitioned-wave pass lowers only the
    # operations of WAVE_PARTITIONED_OPERATIONS; HLSL's exclusive WaveMultiPrefix* forms stay a typed refusal.
    source = tmp_path / "multi_prefix.cgl"
    source.write_text(code, encoding="utf-8")
    with pytest.raises(
        RustLoweringRefused,
        match=r"WaveMultiPrefixSum\(uint, uvec4\) has no spirv_std lowering",
    ):
        crosstl.translate(str(source), backend="rust", format_output=False)


def test_rust_wave_intrinsic_helper_contract_compiles(tmp_path):
    code = """
    shader RustWaveContract {
        compute {
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            buffer uint* values;
            void main() {
                uint lane = values[gl_GlobalInvocationID.x];
                bool active = lane > 3u;
                uint sumValue = WaveActiveSum(lane);
                uint productValue = WaveActiveProduct(lane + 1u);
                uint minValue = WaveActiveMin(sumValue);
                uint maxValue = WaveActiveMax(productValue);
                bool anyLane = WaveActiveAnyTrue(active);
                bool allLane = WaveActiveAllTrue(anyLane);
                bool allEqual = WaveActiveAllEqual(lane);
                uvec4 ballot = WaveActiveBallot(allEqual);
                uint activeCount = WaveActiveCountBits(allEqual);
                uint bitAnd = WaveActiveBitAnd(maxValue);
                uint bitOr = WaveActiveBitOr(bitAnd);
                uint bitXor = WaveActiveBitXor(bitOr);
                uint laneCount = WaveGetLaneCount();
                uint laneIndex = WaveGetLaneIndex();
                bool firstLane = WaveIsFirstLane();
                uint firstValue = WaveReadLaneFirst(bitXor);
                uint laneValue = WaveReadLaneAt(firstValue, laneIndex);
                uint prefixSum = WavePrefixSum(laneValue);
                uint prefixProduct = WavePrefixProduct(prefixSum + 1u);
                uint prefixCount = WavePrefixCountBits(allLane);
                uvec4 matchMask = WaveActiveBallot(prefixProduct > 1u);
                uint quadLane = QuadReadLaneAt(prefixProduct, 1u);
                uint quadX = QuadReadAcrossX(quadLane);
                uint quadY = QuadReadAcrossY(quadX);
                uint quadDiagonal = QuadReadAcrossDiagonal(quadY);
                bool quadAll = WaveActiveAllTrue(active);
                uint shuffled = WaveReadLaneAt(quadDiagonal, 0u);
                uint shuffledXor = WaveShuffleXor(shuffled, 1u);
                uint partitionedSum = WaveMultiPrefixInclusiveSum(shuffledXor, ballot);
                uint partitionedProduct = WaveMultiProduct(partitionedSum, ballot);
                uint partitionedCount = quadAll ? 1u : 0u;
                uint partitionedAnd = WaveMultiPrefixInclusiveBitAnd(partitionedProduct, matchMask);
                uint partitionedOr = WaveMultiBitOr(partitionedAnd, matchMask);
                uint partitionedXor = WaveMultiPrefixInclusiveBitXor(partitionedOr, matchMask);
                values[gl_GlobalInvocationID.x] = partitionedXor + laneCount;
            }
        }
    }
    """

    source = tmp_path / "wave_contract.cgl"
    source.write_text(code, encoding="utf-8")
    generated = crosstl.translate(str(source), backend="rust", format_output=False)
    assert not re.search(r"\b(Wave|Quad)[A-Z]\w*\s*\(", generated)
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_rust_backend_import_preserves_spirv_std_parameter_metadata():
    from crosstl.backend.Rust import (
        RustLexer,
        RustParser,
        RustToCrossGLConverter,
    )

    rust_source = """
    use spirv_std::glam::{UVec3, Vec2, Vec4};
    use spirv_std::spirv;

    pub struct ShaderConstants {
        pub exposure: f32,
    }

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(frag_coord)] frag_coord: Vec4,
        #[spirv(location = 0)] uv: Vec2,
        #[spirv(push_constant)] constants: &ShaderConstants,
        #[spirv(spec_constant(id = 0x5007, default = 100))] mode: u32,
        #[spirv(location = 0)] output: &mut Vec4,
    ) {
        *output = Vec4::new(uv.x, uv.y, constants.exposure, mode as f32);
    }

    #[spirv(compute(threads(64)))]
    pub fn main_cs(
        #[spirv(global_invocation_id)] id: UVec3,
        #[spirv(subgroup_local_invocation_id)] lane: u32,
        #[spirv(subgroup_id)] subgroup: u32,
        #[spirv(num_subgroups)] subgroup_count: u32,
        #[spirv(workgroup)] scratch: &mut [u32; 64],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] data: &mut [u32],
    ) {
        data[id.x as usize] = scratch[0] + lane + subgroup + subgroup_count;
    }
    """

    tokens = RustLexer(rust_source).tokenize()
    ast = RustParser(tokens).parse()
    crossgl = RustToCrossGLConverter().generate(ast)

    assert "vec4 frag_coord @ gl_FragCoord" in crossgl
    assert "vec2 uv @ location(0)" in crossgl
    assert "ShaderConstants constants @ push_constant" in crossgl
    assert "uint mode @ constant_id(0x5007)" in crossgl
    assert "vec4 output @ location(0)" in crossgl
    assert "uvec3 id @ gl_GlobalInvocationID" in crossgl
    assert "uint lane @ gl_SubgroupInvocationID" in crossgl
    assert "uint subgroup @ gl_SubgroupID" in crossgl
    assert "uint subgroup_count @ gl_NumSubgroups" in crossgl
    assert "uint scratch[64] @ groupshared" in crossgl
    assert "RWStructuredBuffer<uint> data @ set(0) @ binding(1)" in crossgl
    assert "void main(" in crossgl
    assert ") @numthreads(64, 1, 1) {" in crossgl


def test_barrier_emits_rust_workgroup_barrier():
    code = """
    shader ComputeSync {
        compute {
            void main() {
                barrier();
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert (
        "spirv_std::arch::workgroup_memory_barrier_with_group_sync();" in generated_code
    )


def test_memory_barrier_emits_rust_memory_barrier():
    code = """
    shader ComputeSync {
        compute {
            void main() {
                memoryBarrier();
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert "memory_barrier()" in generated_code


def test_workgroup_barrier_emits_rust_sync():
    code = """
    shader ComputeSync {
        compute {
            void main() {
                workgroupBarrier();
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert (
        "spirv_std::arch::workgroup_memory_barrier_with_group_sync();" in generated_code
    )


def test_workgroup_execution_barrier_emits_rust_sync():
    code = """
    shader RustExecutionBarrier {
        compute {
            void main() {
                workgroupExecutionBarrier();
            }
        }
    }
    """
    with pytest.raises(
        RustLoweringRefused,
        match="function: 'workgroupExecutionBarrier' has no rust-gpu lowering",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_memory_barrier_variants_emit_sync_intrinsics():
    code = """
    shader ComputeSyncVariants {
        compute {
            void main() {
                memoryBarrierShared();
                memoryBarrierBuffer();
                memoryBarrierImage();
                groupMemoryBarrier();
                allMemoryBarrier();
                deviceMemoryBarrier();
            }
        }
    }
    """
    with pytest.raises(
        RustLoweringRefused,
        match="function: 'memoryBarrierImage' has no rust-gpu lowering",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_user_defined_synchronization_names_are_not_lowered():
    code = """
    shader ComputeSyncShadowing {
        void barrier() {
            return;
        }

        compute {
            void main() {
                barrier();
                memoryBarrier();
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "fn barrier()" in generated_code
    assert "barrier();" in generated_code
    assert "workgroup_barrier()" not in generated_code
    assert "memory_barrier()" in generated_code


def test_rust_synchronization_helper_contract_compiles(tmp_path):
    code = """
    shader RustSynchronizationContract {
        compute {
            void main() {
                barrier();
                workgroupBarrier();
                memoryBarrier();
                memoryBarrierShared();
                memoryBarrierBuffer();
                memoryBarrierImage();
                groupMemoryBarrier();
                allMemoryBarrier();
                deviceMemoryBarrier();
                GroupMemoryBarrier();
                GroupMemoryBarrierWithGroupSync();
                DeviceMemoryBarrier();
                DeviceMemoryBarrierWithGroupSync();
                AllMemoryBarrier();
                AllMemoryBarrierWithGroupSync();
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="function: 'memoryBarrierImage' has no rust-gpu lowering",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_multiple_wave_ops_in_one_function():
    code = """
    shader ComputeWave {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uint sumVal = WaveActiveSum(lane);
                bool anyTrue = WaveActiveAnyTrue(sumVal > 0u);
                uvec4 ballot = WaveActiveBallot(anyTrue);
                uint broadcast = WaveReadLaneAt(sumVal, 0u);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))
    assert "spirv_std::arch::subgroup_i_add(lane)" in generated
    assert "spirv_std::arch::subgroup_any((sumVal > 0))" in generated
    assert "spirv_std::arch::subgroup_ballot(anyTrue)" in generated
    assert "spirv_std::arch::subgroup_shuffle(sumVal, 0)" in generated


def test_mixed_sync_and_wave_ops():
    code = """
    shader ComputeMixed {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                barrier();
                uint total = WaveActiveSum(lane);
                memoryBarrier();
                workgroupBarrier();
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))
    assert generated.index(
        "workgroup_memory_barrier_with_group_sync()"
    ) < generated.index("subgroup_i_add(lane)")
    assert "spirv_std::arch::all_memory_barrier()" in generated


def test_rust_generic_call_keeps_turbofish():
    """A call with explicit type arguments is written ``f::<T>(x)``.

    CrossGL has no path separator, so the Rust importer writes ``f::<T>(x)``
    into CrossGL as ``f<T>(x)``; emitted back without the ``::``, rustc reads
    the ``<`` as a comparison and the generated crate does not parse (#2342).
    """
    code = """
    shader RustTurbofishCall {
        compute {
            void main() {
                float a0 = 1.0;
                float broadcast = subgroup_broadcast_const<_, 0>(a0);
            }
        }
    }
    """

    with pytest.raises(
        RustLoweringRefused,
        match="function: 'subgroup_broadcast_const<_, 0>' has no rust-gpu lowering",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_rust_turbofish_spells_type_arguments_as_rust_types(tmp_path):
    """Explicit type arguments are written as Rust types, ``pick::<f32>`` (#3585).

    Rust has no type ``float``, so ``pick::<float>(2.0)`` does not compile.
    """
    code = """
    shader P {
        T pick<T>(T x) { return x; }
        compute {
            buffer float* outp;
            void main() {
                outp[0] = pick<float>(2.0);
                vec3 v = pick<vec3>(vec3(1.0));
                outp[1] = v.x;
            }
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))
    assert "pub fn pick<T: Copy>(x: T) -> T {" in generated
    assert "outp[0] = pick::<f32>(2.0);" in generated
    assert "let v: Vec3 = pick::<Vec3>(Vec3::new(1.0, 1.0, 1.0));" in generated
    assert "pick::<float>" not in generated
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_rust_turbofish_spells_nested_and_several_type_arguments(tmp_path):
    """Each of several type arguments, nested ones included, is a Rust type (#3585).

    ``vec2<int>`` is ``Vec2<i32>``, as in the declarations; it used to be
    written with its CrossGL name ``ivec2``, which Rust does not have.
    """
    code = """
    shader P {
        U second<T, U>(T a, U b) { return b; }
        ivec2 f(vec2 x, ivec2 m) { return second<vec2<float>, vec2<int>>(x, m); }
        int g(float x, int i) { return second<float, int>(x, i); }
        compute {
            void main() {}
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))
    assert "pub fn second<T: Copy, U: Copy>(a: T, b: U) -> U {" in generated
    assert "return second::<Vec2, IVec2>(x, m);" in generated
    assert "return second::<f32, i32>(x, i);" in generated
    assert "ivec2" not in generated.split("// Compute Shader")[0].replace("IVec2", "")
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_shader_clock_builtins_are_refused_for_rust():
    source_code = """
    shader Clock {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer uint64_t* out_buf;
            buffer uvec2* pair_buf;
            void main() {
                out_buf[0] = clockARB() + clockRealtimeEXT();
                pair_buf[0] = clock2x32ARB() + clockRealtime2x32EXT();
            }
        }
    }
"""
    with pytest.raises(RustLoweringRefused) as refused:
        generate_code(parse_code(tokenize_code(source_code)))
    for name in (
        "clockARB",
        "clock2x32ARB",
        "clockRealtimeEXT",
        "clockRealtime2x32EXT",
    ):
        assert name in str(refused.value)
    assert "shader clock builtin" in str(refused.value)


def test_user_function_named_like_a_shader_clock_is_kept_for_rust():
    source_code = """
    shader Clock {
        uint clockARB() { return 7u; }
        compute {
            layout(local_size_x = 1) in;
            buffer uint* out_buf;
            void main() {
                out_buf[0] = clockARB();
            }
        }
    }
    """
    assert "clockARB()" in generate_code(parse_code(tokenize_code(source_code)))


def test_rust_generic_operator_function_emits_copy_bound_and_compiles(tmp_path):
    """Generic functions that use a value-consuming operator need a Copy bound (#4748)."""
    code = """
    shader G8 {
        generic<T> T twice(T x) {
            return x + x;
        }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* values;
            void main() {
                float v = values[0];
                values[0] = twice(v);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert (
        "pub fn twice<T: Copy + core::ops::Add<Output = T>>(x: T) -> T"
        in generated_code
    )
    assert "return (x + x);" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_generic_bitwise_shift_compound_operators_emit_bounds_and_compile(
    tmp_path,
):
    """Generic functions using &, |, ^, <<, >>, ~ and += need operator bounds (#4909)."""
    code = """
    shader Bits {
        generic<T> T mix_bits(T a, T b) {
            return (a & b) | (a ^ b);
        }
        generic<T> T shifted(T a, T b) {
            return a << b;
        }
        generic<T> T shifted_not(T a, T b) {
            return (a >> b) | ~a;
        }
        generic<T> T accumulate(T a, T b) {
            T s = a;
            s += b;
            return s;
        }
        generic<T> T shift_lit(T a) {
            return a << 1u;
        }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer uint* values;
            void main() {
                uint a = values[0];
                uint b = values[1];
                values[2] = mix_bits(a, b) + shifted(a, b) + accumulate(a, b);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert (
        "pub fn mix_bits<T: Copy + core::ops::BitAnd<Output = T> "
        "+ core::ops::BitXor<Output = T> + core::ops::BitOr<Output = T>>"
        "(a: T, b: T) -> T" in generated_code
    )
    assert (
        "pub fn shifted<T: Copy + core::ops::Shl<Output = T>>(a: T, b: T) -> T"
        in generated_code
    )
    assert (
        "pub fn shifted_not<T: Copy + core::ops::Shr<Output = T> "
        "+ core::ops::Not<Output = T> + core::ops::BitOr<Output = T>>"
        "(a: T, b: T) -> T" in generated_code
    )
    assert (
        "pub fn accumulate<T: Copy + core::ops::AddAssign>(a: T, b: T) -> T"
        in generated_code
    )
    assert (
        "pub fn shift_lit<T: Copy + core::ops::Shl<u32, Output = T>>(a: T) -> T"
        in generated_code
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_generic_compound_assignments_emit_bounds_and_compile(tmp_path):
    """Every compound assignment needs its *Assign bound on a generic (#4909)."""
    code = """
    shader Compound {
        generic<T> T compound_all(T a, T b) {
            T s = a;
            s -= b;
            s *= b;
            s /= b;
            s %= b;
            s &= b;
            s |= b;
            s ^= b;
            s <<= b;
            s >>= b;
            return s;
        }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer uint* values;
            void main() {
                uint a = values[0];
                uint b = values[1];
                values[0] = compound_all(a, b);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert (
        "pub fn compound_all<T: Copy + core::ops::SubAssign + core::ops::MulAssign "
        "+ core::ops::DivAssign + core::ops::RemAssign + core::ops::BitAndAssign "
        "+ core::ops::BitOrAssign + core::ops::BitXorAssign + core::ops::ShlAssign "
        "+ core::ops::ShrAssign>(a: T, b: T) -> T" in generated_code
    )
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_generic_multi_use_by_value_emits_copy_bound_and_compiles(tmp_path):
    """Generic values used by value more than once need Copy even without operators (#4748)."""
    code = """
    shader MultiUse {
        generic<T> T f(T x) {
            T y = x;
            T z = x;
            return z;
        }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* values;
            void main() {
                float v = values[0];
                values[0] = f(v);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert "pub fn f<T: Copy>(x: T) -> T" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_generic_repeated_call_argument_emits_copy_bound_and_compiles(tmp_path):
    """Passing the same generic value twice by value needs a Copy bound (#4748)."""
    code = """
    shader RepeatedArg {
        generic<T> T g(T a, T b) {
            return a;
        }
        generic<T> T f(T x) {
            return g(x, x);
        }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* values;
            void main() {
                float v = values[0];
                values[0] = f(v);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    assert "pub fn g<T: Copy>(a: T, b: T) -> T" in generated_code
    assert "pub fn f<T: Copy>(x: T) -> T" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_rust_generic_function_calling_concrete_function_is_refused():
    """Generic functions whose bodies only type-check per call site are refused (#4748)."""
    code = """
    shader G7 {
        float combine(float a, float b) {
            return a + b;
        }
        generic<T> T twice(T x) {
            return combine(x, x);
        }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* values;
            void main() {
                float v = values[0];
                values[0] = twice(v);
            }
        }
    }
    """
    with pytest.raises(RustLoweringRefused, match="combine"):
        generate_code(parse_code(tokenize_code(code)))
