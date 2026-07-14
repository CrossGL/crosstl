"""Metal AST Node definitions"""

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    AttributeNode,
    BinaryOpNode,
    BreakNode,
    CallNode,
    CaseNode,
    CastNode,
    ConstantBufferNode,
    ContinueNode,
    DeleteNode,
    DesignatedInitializerNode,
    DiscardNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    MethodCallNode,
    NewNode,
    PostfixOpNode,
    PreprocessorNode,
    RangeForNode,
    ReturnNode,
    ShaderNode,
    StaticAssertNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TextureSampleNode,
    ThreadgroupSyncNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)


class BlockNode(ASTNode):
    """Metal standalone scoped block."""

    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"BlockNode(statements={len(self.statements)})"


class LambdaNode(ASTNode):
    """Metal C++ lambda expression."""

    def __init__(self, capture, params, body, return_type=None, specifiers=None):
        self.capture = capture
        self.params = params
        self.body = body
        self.return_type = return_type
        self.specifiers = specifiers or []

    def __repr__(self):
        return (
            f"LambdaNode(capture={self.capture!r}, "
            f"params={len(self.params)}, statements={len(self.body)})"
        )


class CallableTypeAliasNode(TypeAliasNode):
    """Metal function or function-pointer type alias."""

    def __init__(
        self,
        return_type,
        name,
        parameters,
        *,
        indirection="",
        qualifiers=None,
        array_sizes=None,
        declarator_type_suffix_grouped=False,
        source_location=None,
    ):
        super().__init__(
            return_type,
            name,
            qualifiers=qualifiers,
            array_sizes=array_sizes,
            declarator_type_suffix=indirection,
            declarator_type_suffix_grouped=declarator_type_suffix_grouped,
            source_location=source_location,
        )
        self.return_type = return_type
        self.parameters = list(parameters or [])
        self.indirection = indirection
        self.is_function_type = True
        self.is_function_pointer = "*" in indirection

    def __repr__(self):
        return (
            "CallableTypeAliasNode("
            f"return_type={self.return_type}, name={self.name}, "
            f"parameters={self.parameters}, indirection={self.indirection!r})"
        )


_COMMON_NODES = (
    ASTNode,
    ArrayAccessNode,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CallNode,
    CaseNode,
    CastNode,
    CallableTypeAliasNode,
    ConstantBufferNode,
    ContinueNode,
    DeleteNode,
    DesignatedInitializerNode,
    DiscardNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    LambdaNode,
    MemberAccessNode,
    MethodCallNode,
    NewNode,
    PostfixOpNode,
    PreprocessorNode,
    RangeForNode,
    ReturnNode,
    ShaderNode,
    StaticAssertNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TextureSampleNode,
    ThreadgroupSyncNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)
