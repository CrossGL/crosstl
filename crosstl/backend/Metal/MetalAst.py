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
