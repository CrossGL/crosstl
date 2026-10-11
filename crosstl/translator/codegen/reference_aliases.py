"""Preserve identical source references across value-result call boundaries."""

from copy import copy, deepcopy

from ..ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    ArrayAccessNode,
    ASTNode,
    BinaryOpNode,
    BlockNode,
    ConstantNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IdentifierPatternNode,
    LambdaNode,
    LiteralNode,
    MatchArmNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    SwizzleNode,
    TypeNode,
    UnaryOpNode,
    VariableNode,
)


class ReferenceAliasError(ValueError):
    """A source-reference call cannot retain shared storage identity."""

    project_diagnostic_code = "project.translate.reference-alias-unsupported"
    missing_capabilities = ("reference-alias-lowering",)

    def __init__(self, reason, node):
        super().__init__(f"Cannot preserve shared source references: {reason}")
        self.reason = reason
        self.source_location = getattr(node, "source_location", None)


def _reference(parameter):
    qualifiers = set(parameter.qualifiers or ())
    value = parameter.param_type
    if isinstance(value, ReferenceType):
        value = value.referenced_type
        reference = True
    else:
        reference = "thread" in qualifiers and bool(qualifiers & {"in", "out", "inout"})
    return reference and not isinstance(value, PointerType)


def _writable(parameter):
    qualifiers = set(parameter.qualifiers or ())
    if qualifiers & {"const", "constant", "readonly", "in"}:
        return False
    return bool(qualifiers & {"out", "inout"}) or (
        isinstance(parameter.param_type, ReferenceType)
        and parameter.param_type.is_mutable
    )


def _key(value):
    """Compare structure without locations, annotations or metadata backlinks."""
    if isinstance(value, ASTNode):
        return (
            type(value),
            tuple(
                (name, _key(child))
                for name, child in vars(value).items()
                if name not in AST_CHILD_FIELD_EXCLUSIONS
            ),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_key(child) for child in value)
    if isinstance(value, dict):
        return tuple((name, _key(child)) for name, child in value.items())
    return value


def _pure(node):
    if isinstance(node, (IdentifierNode, LiteralNode)):
        return True
    if isinstance(node, MemberAccessNode):
        return _pure(node.object_expr)
    if isinstance(node, SwizzleNode):
        return _pure(node.vector_expr)
    if isinstance(node, ArrayAccessNode):
        return _pure(node.array_expr) and _pure(node.index_expr)
    if isinstance(node, BinaryOpNode):
        return node.operator in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "&",
            "|",
            "^",
            "<<",
            ">>",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "&&",
            "||",
        } and (_pure(node.left) and _pure(node.right))
    if isinstance(node, UnaryOpNode):
        return node.operator in {"+", "-", "~", "!"} and _pure(node.operand)
    return False


def _location(node):
    return isinstance(
        node, (IdentifierNode, MemberAccessNode, ArrayAccessNode, SwizzleNode)
    ) and _pure(node)


class _Lowering:
    def __init__(self, ast):
        self.ast = ast
        self.functions = {}
        self.names = set()
        for node in ast.walk():
            name = getattr(node, "name", None)
            if isinstance(name, str):
                self.names.add(name)
            if isinstance(node, FunctionNode):
                self.functions.setdefault(node.name, []).append(node)
        self.specializations = {}
        self.created = []

    def fresh(self, stem):
        name = stem
        index = 0
        while name in self.names:
            index += 1
            name = f"{stem}_{index}"
        self.names.add(name)
        return name

    def children(self, node, rewrite):
        result = copy(node)
        memo = {}

        def transform(value):
            if isinstance(value, ASTNode):
                if id(value) not in memo:
                    memo[id(value)] = rewrite(value)
                return memo[id(value)]
            if isinstance(value, list):
                return [transform(child) for child in value]
            if isinstance(value, tuple):
                return tuple(transform(child) for child in value)
            if isinstance(value, dict):
                return {name: transform(child) for name, child in value.items()}
            return value

        for name, value in vars(node).items():
            if name not in AST_CHILD_FIELD_EXCLUSIONS:
                setattr(result, name, transform(value))
        return result

    def rewrite(self, node, bindings):
        if not isinstance(node, ASTNode) or isinstance(node, TypeNode):
            return node
        if isinstance(node, IdentifierNode) and node.name in bindings:
            result = copy(node)
            result.name = bindings[node.name]
            result.identifier = result.name
            return result
        if isinstance(node, BlockNode):
            scope = dict(bindings)
            result = copy(node)
            result.statements = [
                self.rewrite(child, scope) for child in node.statements
            ]
            return result
        if isinstance(node, ForNode):
            scope = dict(bindings)
            result = copy(node)
            result.init = self.rewrite(node.init, scope)
            result.condition = self.rewrite(node.condition, scope)
            result.update = self.rewrite(node.update, scope)
            result.body = self.rewrite(node.body, scope)
            return result
        if isinstance(node, (ForInNode, MatchArmNode)):
            scope = dict(bindings)
            pattern = node.pattern
            if isinstance(pattern, str):
                scope.pop(pattern, None)
            elif isinstance(pattern, ASTNode):
                for binding in pattern.walk():
                    if isinstance(binding, IdentifierPatternNode):
                        scope.pop(binding.name, None)
            elif bindings:
                raise ReferenceAliasError("binding-pattern-unsupported", node)
            result = copy(node)
            if isinstance(node, ForInNode):
                result.iterable = self.rewrite(node.iterable, bindings)
            else:
                result.guard = self.rewrite(node.guard, scope)
            result.body = self.rewrite(node.body, scope)
            return result
        if bindings and isinstance(node, (LambdaNode, FunctionNode)):
            raise ReferenceAliasError("nested-callable-capture-unsupported", node)
        if isinstance(node, (VariableNode, ConstantNode)):
            bindings.pop(node.name, None)
        result = self.children(node, lambda child: self.rewrite(child, bindings))
        return self.call(result) if isinstance(result, FunctionCallNode) else result

    def call(self, node):
        name = getattr(node.function, "name", node.function)
        if not isinstance(name, str):
            return node
        candidates = self.functions.get(name, ())
        if len(candidates) != 1:
            return node
        function = candidates[0]
        parameters = function.parameters
        if len(parameters) != len(node.arguments) or function.generic_params:
            return node
        groups = {}
        for index, (parameter, argument) in enumerate(zip(parameters, node.arguments)):
            if _reference(parameter) and _location(argument):
                groups.setdefault(_key(argument), []).append(index)
        aliases = [
            indices
            for indices in groups.values()
            if len(indices) > 1 and any(_writable(parameters[i]) for i in indices)
        ]
        if not aliases:
            return node
        if not all(_pure(argument) for argument in node.arguments):
            raise ReferenceAliasError("argument-evaluation-unsupported", node)
        if function.body is None:
            raise ReferenceAliasError("callee-body-unavailable", node)
        representatives = list(range(len(parameters)))
        for indices in aliases:
            types = [
                (
                    parameter.param_type.referenced_type
                    if isinstance(parameter.param_type, ReferenceType)
                    else parameter.param_type
                )
                for parameter in (parameters[i] for i in indices)
            ]
            if any(_key(value) != _key(types[0]) for value in types[1:]):
                raise ReferenceAliasError("reference-view-types-differ", node)
            representative = next(i for i in indices if _writable(parameters[i]))
            for index in indices:
                representatives[index] = representative
        identity = (id(function), tuple(representatives))
        selected = self.specializations.get(identity)
        if selected is None:
            selected = deepcopy(function)
            selected.name = self.fresh(f"{name}_shared_references")
            kept = [
                i
                for i, representative in enumerate(representatives)
                if i == representative
            ]
            names = {i: self.fresh(f"crosstl_reference_{i}") for i in kept}
            bindings = {
                parameter.name: names[representatives[i]]
                for i, parameter in enumerate(parameters)
            }
            selected.parameters = [selected.parameters[i] for i in kept]
            for index, parameter in zip(kept, selected.parameters):
                parameter.name = names[index]
            # Cache before walking nested calls so recursive source graphs terminate.
            self.specializations[identity] = selected
            self.created.append(selected)
            selected.body = self.rewrite(selected.body, bindings)
        result = copy(node)
        result.function = IdentifierNode(selected.name)
        result.name = result.function
        result.arguments = [
            argument
            for i, argument in enumerate(node.arguments)
            if representatives[i] == i
        ]
        result.args = result.arguments
        return result

    def run(self):
        for functions in self.functions.values():
            for function in functions:
                function.body = self.rewrite(function.body, {})
        self.ast.functions.extend(self.created)
        return self.ast


def lower_reference_aliases(ast):
    """Specialize identical references without mutating the caller's source AST."""
    if not isinstance(ast, ASTNode):
        return ast
    return _Lowering(deepcopy(ast)).run()
