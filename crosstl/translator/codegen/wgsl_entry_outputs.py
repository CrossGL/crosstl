"""Entry-point ``out`` and ``inout`` parameters lowered to a WGSL output structure.

HLSL vertex and pixel shaders can write stage outputs through ``out`` parameters (``out float4 pos : SV_Position``),
alone or beside a return value, and an ``inout`` parameter is both a stage input and a stage output. WGSL entry-point
parameters are pipeline inputs only, and every stage output is the return value or a member of the returned structure
(WGSL section 13.3.1). The lowered entry point therefore:

* returns a new structure whose members are the return value (first) and then each output parameter, carrying the
  parameter's IO attribute; a structure-typed output parameter contributes one member per field;
* declares each output parameter as a function-scope ``var`` of the same name (an ``inout`` one starts from the
  renamed input parameter), so the body is unchanged;
* turns every ``return`` into ``return Output(value, out1, out2, ...)``. WGSL evaluates call arguments left to right
  (WGSL section 14.1), so the return value is computed before the outputs are read, as in HLSL.
"""

from __future__ import annotations

import copy

from ..ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    ASTNode,
    AttributeNode,
    BlockNode,
    CaseNode,
    ExpressionNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PrimitiveType,
    ReturnNode,
    StatementNode,
    StructMemberNode,
    StructNode,
    VariableNode,
)

__all__ = [
    "EntryOutputLoweringError",
    "entry_output_parameters",
    "lower_entry_output_parameters",
]

_OUTPUT_QUALIFIERS = ("out", "inout")
_LOWERED_STAGES = frozenset({"vertex", "fragment"})


class EntryOutputLoweringError(ValueError):
    """An entry point whose output parameters cannot become a WGSL output structure."""


def _qualifiers(parameter):
    return [
        str(qualifier).lower()
        for qualifier in getattr(parameter, "qualifiers", None) or []
        if qualifier is not None
    ]


def _output_qualifier(parameter):
    qualifiers = _qualifiers(parameter)
    for qualifier in _OUTPUT_QUALIFIERS:
        if qualifier in qualifiers:
            return qualifier
    return None


def entry_output_parameters(entry_point, stage_name):
    """The ``out``/``inout`` parameters of a vertex or fragment ``entry_point``, in order.

    A WGSL compute entry point has no stage outputs, so an output parameter there is refused.
    """
    if entry_point is None:
        return []
    parameters = [
        parameter
        for parameter in getattr(entry_point, "parameters", None) or []
        if _output_qualifier(parameter) is not None
    ]
    if parameters and stage_name == "compute":
        raise EntryOutputLoweringError(
            f"WGSL target cannot express the {_output_qualifier(parameters[0])} "
            f"parameter {parameters[0].name} of compute entry point "
            f"{entry_point.name}: a compute shader has no stage outputs"
        )
    if stage_name not in _LOWERED_STAGES:
        return []
    return parameters


def _unique(preferred, used):
    name = preferred
    suffix = 2
    while name in used:
        name = f"{preferred}{suffix}"
        suffix += 1
    used.add(name)
    return name


def _body_names(body):
    if body is None or not hasattr(body, "walk"):
        return set()
    names = set()
    for node in body.walk():
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


class _Context:
    def __init__(self, codegen, entry_point, stage_name, structs_by_name):
        self.codegen = codegen
        self.entry_point = entry_point
        self.stage_name = stage_name
        self.structs_by_name = structs_by_name
        self.function_name = getattr(entry_point, "name", "<anonymous>")

    def fail(self, reason):
        raise EntryOutputLoweringError(
            f"WGSL target cannot lower the output parameters of {self.stage_name} entry "
            f"point {self.function_name}: {reason}"
        )

    def io_attributes(self, attributes):
        """The attributes that bind a stage input or output slot."""
        bound = []
        for attribute in attributes or []:
            probe = StructMemberNode("_probe", None, attributes=[attribute])
            if self.codegen.member_stage_binding(probe)[0] is not None:
                bound.append(attribute)
        return bound

    def struct_of(self, type_node):
        return self.structs_by_name.get(self.codegen.struct_type_name(type_node))


def _return_value_attributes(context):
    entry_point = context.entry_point
    attributes = context.io_attributes(getattr(entry_point, "attributes", None))
    if attributes:
        return attributes
    if context.stage_name == "fragment":
        return [AttributeNode("gl_FragData", [LiteralNode(0, PrimitiveType("int"))])]
    return [AttributeNode("gl_Position")]


def _output_members(context, parameters, used_member_names):
    """Structure members and the expressions that fill them, for each output parameter."""
    members = []
    values = []
    for parameter in parameters:
        parameter_type = getattr(parameter, "param_type", None)
        struct = context.struct_of(parameter_type)
        if struct is None:
            attributes = context.io_attributes(getattr(parameter, "attributes", None))
            if not attributes:
                context.fail(
                    f"output parameter {parameter.name} has no stage output semantic"
                )
            members.append(
                StructMemberNode(
                    parameter.name, parameter_type, attributes=list(attributes)
                )
            )
            values.append(IdentifierNode(parameter.name))
            continue
        for field in getattr(struct, "members", None) or []:
            if context.struct_of(getattr(field, "member_type", None)) is not None:
                context.fail(
                    f"member {field.name} of output parameter {parameter.name} is a "
                    "structure; WGSL stage outputs cannot nest structures"
                )
            name = _unique(f"{parameter.name}_{field.name}", used_member_names)
            members.append(
                StructMemberNode(
                    name,
                    field.member_type,
                    attributes=list(getattr(field, "attributes", None) or []),
                )
            )
            values.append(MemberAccessNode(IdentifierNode(parameter.name), field.name))
    return members, values


class _ReturnRewriter:
    """Path-copies a body so each ``return`` builds the output structure."""

    def __init__(self, context, struct_name, output_values):
        self.context = context
        self.struct_name = struct_name
        self.output_values = output_values
        self.memo = {}

    def build(self, value):
        arguments = [] if value is None else [value]
        arguments.extend(copy.copy(expression) for expression in self.output_values)
        return FunctionCallNode(IdentifierNode(self.struct_name), arguments)

    def rewrite(self, node):
        key = id(node)
        if key in self.memo:
            return self.memo[key]
        result = self._rewrite(node)
        self.memo[key] = result
        return result

    def _rewrite(self, node):
        if isinstance(node, ReturnNode):
            value = getattr(node, "value", None)
            if isinstance(value, list):
                self.context.fail("a return statement returns several values")
            rewritten = copy.copy(node)
            rewritten.value = self.build(value)
            return rewritten
        if isinstance(node, ExpressionNode) or not isinstance(
            node, (StatementNode, CaseNode)
        ):
            return node
        changes = {}
        for field_name, value in vars(node).items():
            if field_name in AST_CHILD_FIELD_EXCLUSIONS:
                continue
            rewritten = self._rewrite_value(value)
            if rewritten is not value:
                changes[field_name] = rewritten
        if not changes:
            return node
        clone = copy.copy(node)
        vars(clone).update(changes)
        return clone

    def _rewrite_value(self, value):
        if isinstance(value, ASTNode):
            return self.rewrite(value)
        if isinstance(value, list):
            items = [self._rewrite_value(item) for item in value]
            if any(new is not old for new, old in zip(items, value)):
                return items
        return value


def lower_entry_output_parameters(
    codegen, entry_point, stage_name, structs_by_name, used_type_names
):
    """The output structure and the lowered entry point for ``entry_point``'s output parameters.

    ``used_type_names`` holds the module's type and declaration names; the new structure's name is added to it.
    """
    context = _Context(codegen, entry_point, stage_name, structs_by_name)
    outputs = entry_output_parameters(entry_point, stage_name)
    return_type = getattr(entry_point, "return_type", None)
    returns_value = not codegen.is_void_type(return_type)
    if returns_value and context.struct_of(return_type) is not None:
        context.fail(
            "it also returns a structure; return every stage output from one place"
        )

    # Plain output parameters keep their names; generated member names avoid them.
    used_member_names = {
        parameter.name
        for parameter in outputs
        if context.struct_of(getattr(parameter, "param_type", None)) is None
    }
    members = []
    if returns_value:
        members.append(
            StructMemberNode(
                _unique("_crossglReturn", used_member_names),
                return_type,
                attributes=_return_value_attributes(context),
            )
        )
    output_members, output_values = _output_members(context, outputs, used_member_names)
    members.extend(output_members)
    struct_name = _unique(f"{context.function_name}Output", used_type_names)
    struct_node = StructNode(struct_name, members)

    body = getattr(entry_point, "body", None)
    reserved = _body_names(body) | {
        getattr(parameter, "name", "")
        for parameter in getattr(entry_point, "parameters", None) or []
    }
    parameters = []
    locals_ = []
    for parameter in getattr(entry_point, "parameters", None) or []:
        qualifier = _output_qualifier(parameter)
        if qualifier is None:
            parameters.append(parameter)
            continue
        initial_value = None
        if qualifier == "inout":
            incoming = copy.copy(parameter)
            incoming.name = _unique(f"{parameter.name}_in", reserved)
            incoming.qualifiers = [
                qualifier
                for qualifier in getattr(parameter, "qualifiers", None) or []
                if str(qualifier).lower() not in _OUTPUT_QUALIFIERS
            ]
            parameters.append(incoming)
            initial_value = IdentifierNode(incoming.name)
        locals_.append(
            VariableNode(parameter.name, parameter.param_type, initial_value)
        )

    rewriter = _ReturnRewriter(context, struct_name, output_values)
    statements = [rewriter.rewrite(statement) for statement in _statements(body)]
    # A value-returning body already returns on every path; a void one may fall off its end.
    if not returns_value and not (
        statements and isinstance(statements[-1], ReturnNode)
    ):
        statements.append(ReturnNode(rewriter.build(None)))

    lowered = copy.copy(entry_point)
    lowered.parameters = parameters
    lowered.return_type = NamedType(struct_name)
    io_attributes = {
        id(attribute)
        for attribute in context.io_attributes(getattr(entry_point, "attributes", None))
    }
    lowered.attributes = [
        attribute
        for attribute in getattr(entry_point, "attributes", None) or []
        if id(attribute) not in io_attributes
    ]
    lowered.body = BlockNode([*locals_, *statements])
    return struct_node, lowered


def _statements(body):
    if body is None:
        return []
    if hasattr(body, "statements"):
        return list(getattr(body, "statements", None) or [])
    if isinstance(body, (list, tuple)):
        return list(body)
    return [body]
