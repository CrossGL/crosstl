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


class ConstructorInitializerNode(ASTNode):
    """One Metal constructor member/base initializer."""

    def __init__(
        self,
        target,
        arguments=None,
        *,
        style="paren",
        source_location=None,
    ):
        self.target = target
        self.arguments = list(arguments or [])
        self.style = style
        self.source_location = source_location

    def __repr__(self):
        return (
            "ConstructorInitializerNode("
            f"target={self.target!r}, arguments={self.arguments}, "
            f"style={self.style!r})"
        )


class ConstructorNode(ASTNode):
    """An explicit Metal struct/class constructor contract."""

    def __init__(
        self,
        owner_name,
        params,
        body,
        *,
        initializers=None,
        qualifiers=None,
        template_parameters=None,
        template_parameter_defaults=None,
        declaration_kind="definition",
        source_location=None,
    ):
        self.owner_name = owner_name
        self.name = owner_name
        self.params = list(params or [])
        self.body = body
        self.initializers = list(initializers or [])
        self.qualifiers = list(qualifiers or [])
        self.template_parameters = list(template_parameters or [])
        self.template_parameter_defaults = dict(template_parameter_defaults or {})
        self.generics = [name for _kind, name in self.template_parameters if name]
        self.declaration_kind = declaration_kind
        self.source_location = source_location

    def __repr__(self):
        return (
            "ConstructorNode("
            f"owner_name={self.owner_name!r}, params={self.params}, "
            f"initializers={self.initializers}, body={self.body})"
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
    ConstructorInitializerNode,
    ConstructorNode,
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
