"""Conservative return dependencies for resolved, read-only value helpers."""

from ..arithmetic_conversions import target_arithmetic_type
from ..ast import (
    ArrayType,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MatrixType,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    ReturnNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
)


class UniformReturnAnalysis:
    """Prove body purity separately from uniformity at an individual call.

    Resolved calls and builtin identities come from the target's source-overload
    analysis. Summaries contain formal parameter positions, never caller facts.
    Unknown statements, global reads, memory loads, escapes and recursion fail
    closed. In particular, a read-only storage view is not immutable storage.
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

    def uniform_arguments(self, call):
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
            if self._expression(argument, names) is None:
                return None
        return [call.arguments[index] for index in sorted(dependencies)]

    def _value_type(self, type_, visiting=frozenset()):
        if isinstance(type_, (PointerType, ReferenceType)) or self._qualified(
            type_, {"volatile"}
        ):
            return False
        if isinstance(type_, (ArrayType, VectorType, MatrixType)):
            return self._value_type(type_.element_type, visiting)
        name = type_ if isinstance(type_, str) else getattr(type_, "name", None)
        if target_arithmetic_type(name) is not None:
            return True
        if name in visiting or name not in self.structs:
            return False
        return all(
            not self._qualified(member, {"volatile", "static"})
            and self._value_type(member.member_type, visiting | {name})
            for member in self.structs[name].members
        )

    def _function(self, function):
        if (
            function.body is None
            or not self._value_type(function.return_type)
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
            if self._value_type(
                parameter.param_type.referenced_type
                if isinstance(parameter.param_type, ReferenceType)
                else parameter.param_type
            )
        }
        result = self._block(function.body, names)
        return result[0] if result is not None and result[1] else None

    def _block(self, body, names):
        statements = body.statements if isinstance(body, BlockNode) else body
        if not isinstance(statements, list):
            statements = [] if statements is None else [statements]
        names = dict(names)
        dependencies = frozenset()
        returned = False
        for statement in statements:
            if returned:
                return None
            if isinstance(statement, VariableNode):
                if (
                    statement.name in names
                    or not self._value_type(statement.var_type)
                    or self._qualified(
                        statement, {"volatile", "static", "shared", "groupshared"}
                    )
                ):
                    return None
                used = self._expression(statement.initial_value, names)
                if used is None:
                    return None
                names[statement.name] = used
            elif isinstance(statement, ReturnNode):
                used = self._expression(statement.value, names)
                returned = True
            elif isinstance(statement, BlockNode):
                result = self._block(statement, names)
                if result is None:
                    return None
                used, returned = result
            elif isinstance(statement, IfNode):
                conditions = [statement.condition, *statement.else_if_conditions]
                branches = [
                    statement.then_branch,
                    *statement.else_if_bodies,
                    statement.else_branch,
                ]
                condition_deps = [
                    self._expression(value, names) for value in conditions
                ]
                results = [self._block(branch, names) for branch in branches]
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
        return dependencies, returned

    def _expression(self, node, names):
        if isinstance(node, LiteralNode):
            return frozenset()
        if isinstance(node, IdentifierNode):
            return names.get(node.name)
        if isinstance(node, MemberAccessNode):
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
                self._expression(argument, names) for argument in node.arguments
            ]
            if any(argument is None for argument in arguments):
                return None
            return frozenset().union(*(arguments[index] for index in dependencies))
        else:
            return None
        dependencies = [self._expression(child, names) for child in children]
        if any(value is None for value in dependencies):
            return None
        return frozenset().union(*dependencies)
