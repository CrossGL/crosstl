"""Conservative return dependencies for resolved, read-only value helpers."""

from dataclasses import dataclass

from ..arithmetic_conversions import source_integer_shape, target_arithmetic_type
from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MatrixType,
    MemberAccessNode,
    NamedType,
    PointerType,
    ReferenceType,
    ReturnNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
)


@dataclass(frozen=True)
class _StorageDependency:
    parameter: int


class UniformReturnAnalysis:
    """Prove body purity separately from uniformity at an individual call.

    Resolved calls and builtin identities come from the target's source-overload
    analysis. Summaries contain formal parameter positions, never caller facts.
    Storage dependencies must resolve to immutable bindings at each caller.
    A read-only storage view alone is not immutable storage. Unknown statements,
    global reads, escapes and recursion fail closed.
    """

    def __init__(self, resolved_calls, pure_builtin_calls, structs=None):
        self.resolved_calls = resolved_calls
        self.pure_builtin_calls = pure_builtin_calls
        self.structs = structs or {}
        self.summaries = {}
        self.visiting = set()

    @staticmethod
    def _qualified(node, excluded):
        qualifiers = [
            *getattr(node, "qualifiers", []),
            *getattr(node, "resource_qualifiers", []),
            *getattr(node, "attributes", []),
        ]
        return any(str(getattr(q, "name", q)).lower() in excluded for q in qualifiers)

    def dependencies(self, call):
        """Return the required argument positions, or None without a proof."""
        if id(call) in self.pure_builtin_calls:
            return frozenset(range(len(call.arguments)))
        function = self.resolved_calls.get(id(call))
        if function is None or len(function.parameters) != len(call.arguments):
            return None
        key = id(function)
        if key in self.visiting:
            return None
        if key not in self.summaries:
            self.visiting.add(key)
            try:
                self.summaries[key] = self._function(function)
            finally:
                self.visiting.remove(key)
        return self.summaries[key]

    def uniform_arguments(self, call, resources=frozenset()):
        """Select dependencies only when every argument is itself effect-free."""
        dependencies = self.dependencies(call)
        if dependencies is None:
            return None
        for argument in call.arguments:
            names = {
                node.name: frozenset()
                for node in argument.walk()
                if isinstance(node, IdentifierNode)
            }
            if self._expression(argument, names, resources) is None:
                return None
        required = set()
        for dependency in dependencies:
            if isinstance(dependency, _StorageDependency):
                argument = call.arguments[dependency.parameter]
                if (
                    not isinstance(argument, IdentifierNode)
                    or argument.name not in resources
                ):
                    return None
                required.add(dependency.parameter)
            else:
                required.add(dependency)
        return [call.arguments[index] for index in sorted(required)]

    def resource_type(self, type_, owner=None, *, read_only=True):
        return (
            isinstance(type_, NamedType)
            and type_.name in (
                {"StructuredBuffer"}
                if read_only
                else {"StructuredBuffer", "RWStructuredBuffer"}
            )
            and len(type_.generic_args) == 1
            and self.value_type(type_.generic_args[0])
            and not self._qualified(type_, {"volatile", "coherent"})
            and not self._qualified(owner, {"volatile", "coherent", "writeonly"})
            and not getattr(owner, "resource_qualifiers", None)
        )

    def value_type(self, type_, visiting=frozenset()):
        if isinstance(type_, (PointerType, ReferenceType)) or self._qualified(
            type_, {"volatile"}
        ):
            return False
        if isinstance(type_, (ArrayType, VectorType, MatrixType)):
            return self.value_type(type_.element_type, visiting)
        name = type_ if isinstance(type_, str) else getattr(type_, "name", None)
        if (
            target_arithmetic_type(name) is not None
            or name in {"min16int", "min16uint"}
            or (
                isinstance(name, str)
                and name.isidentifier()
                and source_integer_shape(name) is not None
            )
        ):
            return True
        if name in visiting or name not in self.structs:
            return False
        return all(
            not self._qualified(member, {"volatile", "static"})
            and self.value_type(member.member_type, visiting | {name})
            for member in self.structs[name].members
        )

    def _function(self, function):
        if (
            function.body is None
            or not self.value_type(function.return_type)
            or self._qualified(function, {"volatile"})
            or any(
                self._qualified(parameter, {"volatile", "out", "inout"})
                or self._qualified(parameter.param_type, {"volatile"})
                for parameter in function.parameters
            )
        ):
            return None
        names = {
            parameter.name: frozenset({index})
            for index, parameter in enumerate(function.parameters)
            if self.value_type(
                parameter.param_type.referenced_type
                if isinstance(parameter.param_type, ReferenceType)
                else parameter.param_type
            )
        }
        resources = set()
        for index, parameter in enumerate(function.parameters):
            if self.resource_type(parameter.param_type, parameter, read_only=False):
                if self.resource_type(parameter.param_type, parameter):
                    resources.add(parameter.name)
                names[parameter.name] = frozenset({_StorageDependency(index)})
        result = self._block(function.body, names, resources)
        return result[0] if result is not None and result[1] else None

    def _block(self, body, names, resources):
        statements = body.statements if isinstance(body, BlockNode) else body
        if not isinstance(statements, list):
            statements = [] if statements is None else [statements]
        names = dict(names)
        resources = set(resources)
        pending = {}
        dependencies = frozenset()
        returned = False
        for statement in statements:
            if returned:
                return None
            if isinstance(statement, ExpressionStatementNode):
                statement = statement.expression
            if isinstance(statement, VariableNode):
                if (
                    statement.name in names
                    or statement.name in pending
                    or self._qualified(
                        statement, {"volatile", "static", "shared", "groupshared"}
                    )
                ):
                    return None
                resource = self.resource_type(statement.var_type, statement)
                if not resource and not self.value_type(statement.var_type):
                    return None
                if resource:
                    if (
                        not isinstance(statement.initial_value, IdentifierNode)
                        or statement.initial_value.name not in resources
                    ):
                        return None
                    resources.add(statement.name)
                if statement.initial_value is None:
                    struct = self.structs.get(getattr(statement.var_type, "name", None))
                    if struct is None or not struct.members:
                        return None
                    pending[statement.name] = (
                        {member.name for member in struct.members},
                        set(),
                        frozenset(),
                    )
                    continue
                used = self._expression(statement.initial_value, names, resources)
                if used is None:
                    return None
                names[statement.name] = used
            elif isinstance(statement, AssignmentNode):
                target = statement.target
                if (
                    statement.operator != "="
                    or not isinstance(target, MemberAccessNode)
                    or not isinstance(target.object_expr, IdentifierNode)
                    or target.object_expr.name not in pending
                ):
                    return None
                name = target.object_expr.name
                fields, initialized, prior = pending[name]
                if target.member not in fields or target.member in initialized:
                    return None
                used = self._expression(statement.value, names, resources)
                if used is None:
                    return None
                initialized = initialized | {target.member}
                prior |= used
                pending[name] = fields, initialized, prior
                if initialized == fields:
                    names[name] = prior
                    del pending[name]
            elif isinstance(statement, ReturnNode):
                used = self._expression(statement.value, names, resources)
                returned = True
            elif isinstance(statement, BlockNode):
                if pending:
                    return None
                result = self._block(statement, names, resources)
                if result is None:
                    return None
                used, returned = result
            elif isinstance(statement, IfNode):
                if pending:
                    return None
                conditions = [statement.condition, *statement.else_if_conditions]
                branches = [
                    statement.then_branch,
                    *statement.else_if_bodies,
                    statement.else_branch,
                ]
                condition_deps = [
                    self._expression(value, names, resources) for value in conditions
                ]
                results = [self._block(branch, names, resources) for branch in branches]
                if any(value is None for value in [*condition_deps, *results]):
                    return None
                used = frozenset().union(
                    *condition_deps, *(result[0] for result in results)
                )
                returned = all(result[1] for result in results)
            else:
                return None
            if used is None:
                return None
            dependencies |= used
        return None if pending else (dependencies, returned)

    def _expression(self, node, names, resources):
        if isinstance(node, LiteralNode):
            return frozenset()
        if isinstance(node, IdentifierNode):
            return names.get(node.name)
        if isinstance(node, ArrayAccessNode):
            if (
                not isinstance(node.array_expr, IdentifierNode)
                or node.array_expr.name not in resources
            ):
                return None
            children = [node.array_expr, node.index_expr]
        elif isinstance(node, MemberAccessNode):
            children = [node.object_expr]
        elif isinstance(node, SwizzleNode):
            children = [node.vector_expr]
        elif isinstance(node, ConstructorNode):
            children = [*node.arguments, *node.named_arguments.values()]
        elif isinstance(node, CastNode):
            if isinstance(node.target_type, (PointerType, ReferenceType)):
                return None
            children = [node.expression]
        elif isinstance(node, BinaryOpNode):
            children = [node.left, node.right]
        elif isinstance(node, UnaryOpNode) and node.operator in {"+", "-", "!", "~"}:
            children = [node.operand]
        elif isinstance(node, TernaryOpNode):
            children = [node.condition, node.true_expr, node.false_expr]
        elif isinstance(node, FunctionCallNode):
            dependencies = self.dependencies(node)
            if dependencies is None:
                return None
            arguments = [
                self._expression(argument, names, resources)
                for argument in node.arguments
            ]
            if any(argument is None for argument in arguments):
                return None
            required = []
            for dependency in dependencies:
                if isinstance(dependency, _StorageDependency):
                    argument = node.arguments[dependency.parameter]
                    if (
                        not isinstance(argument, IdentifierNode)
                        or argument.name not in resources
                    ):
                        return None
                    required.append(arguments[dependency.parameter])
                else:
                    required.append(arguments[dependency])
            return frozenset().union(*required)
        else:
            return None
        dependencies = [self._expression(child, names, resources) for child in children]
        if any(value is None for value in dependencies):
            return None
        return frozenset().union(*dependencies)
