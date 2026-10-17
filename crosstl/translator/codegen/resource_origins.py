"""Conservative allocation origins for private resource-reference values.

Fields are joined by source aggregate type, not target handle type. This keeps
distinct loader types separate without assuming that instances of the same type
cannot exchange pointers. Assignments and call edges are inclusion constraints;
control-flow alternatives and mutable reference writebacks are always joined.
"""

from collections import defaultdict, deque

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BlockNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    ReferenceType,
    ReturnNode,
    SwitchNode,
    TernaryOpNode,
    TypeNode,
    UnaryOpNode,
    VariableNode,
)


class ResourceOriginAnalysis:
    """Solve possible physical allocations before pointer types are erased."""

    UNKNOWN = ("unknown",)

    def __init__(self, lowering, pointer_type):
        self.lowering = lowering
        self.pointer_type = pointer_type
        self.edges = defaultdict(set)
        self.values = defaultdict(set)
        self.values[self.UNKNOWN].add(None)
        self.expressions = {}
        self.current = None

    def edge(self, destination, source):
        self.edges[source].add(destination)

    def pointer(self, type_):
        return isinstance(type_, self.pointer_type)

    def field(self, owner_type, name):
        return ("field", getattr(owner_type, "name", None), name)

    def origin(self, node, types, slots):
        if node is None:
            return self.UNKNOWN
        if id(node) in self.expressions:
            return self.expressions[id(node)]
        cell = self.UNKNOWN
        if isinstance(node, IdentifierNode):
            cell = slots.get(node.name, self.UNKNOWN)
        elif isinstance(node, MemberAccessNode):
            owner = self.lowering.infer(node.object_expr, types)
            if node.member in self.lowering.fields.get(
                getattr(owner, "name", None), {}
            ):
                cell = self.field(owner, node.member)
        elif isinstance(node, ArrayAccessNode):
            cell = self.origin(node.array_expr, types, slots)
        elif isinstance(node, UnaryOpNode):
            if node.operator == "&" and isinstance(node.operand, ArrayAccessNode):
                cell = self.origin(node.operand.array_expr, types, slots)
            elif node.operator in {"++", "--"}:
                cell = self.origin(node.operand, types, slots)
        elif isinstance(node, BinaryOpNode) and node.operator in {"+", "-"}:
            left = self.lowering.infer(node.left, types)
            right = self.lowering.infer(node.right, types)
            if self.pointer(left) and not self.pointer(right):
                cell = self.origin(node.left, types, slots)
            elif (
                node.operator == "+" and self.pointer(right) and not self.pointer(left)
            ):
                cell = self.origin(node.right, types, slots)
        elif isinstance(node, TernaryOpNode):
            cell = ("expression", id(node))
            self.edge(cell, self.origin(node.true_expr, types, slots))
            self.edge(cell, self.origin(node.false_expr, types, slots))
        elif isinstance(node, FunctionCallNode):
            callee = self.lowering.function_for_call(node, types)
            if callee is not None:
                cell = ("return", id(callee))
        self.expressions[id(node)] = cell
        return cell

    def bind_value(self, type_, value, destination, types, slots):
        if self.pointer(type_):
            self.edge(destination, self.origin(value, types, slots))
            return
        if isinstance(type_, tuple):
            if isinstance(value, ArrayLiteralNode):
                for element in value.elements:
                    self.bind_value(type_[0], element, destination, types, slots)
            elif self.lowering.contains_reference(type_):
                self.edge(destination, self.origin(value, types, slots))
            return
        fields = self.lowering.fields.get(getattr(type_, "name", None))
        if not fields:
            return
        if isinstance(value, ArrayLiteralNode):
            values = value.elements
        elif isinstance(value, FunctionCallNode) and getattr(
            value.function, "name", None
        ) == getattr(type_, "name", None):
            values = value.arguments
        else:
            # Copies and returned aggregates already share their source-type
            # field cells. Constructors contribute their own field assignments.
            return
        for index, (name, field_type) in enumerate(fields.items()):
            self.bind_value(
                field_type,
                values[index] if index < len(values) else None,
                self.field(type_, name),
                types,
                slots,
            )

    def visit(self, node, types, slots):
        if not isinstance(node, ASTNode) or isinstance(node, TypeNode):
            return
        if isinstance(node, BlockNode):
            types, slots = dict(types), dict(slots)
            for statement in node.statements:
                self.visit(statement, types, slots)
            return
        if isinstance(node, VariableNode):
            self.visit(node.initial_value, types, slots)
            type_ = self.lowering.source_type(node.var_type, node)
            if getattr(type_, "name", None) == "auto":
                type_ = self.lowering.infer(node.initial_value, types) or type_
            cell = ("value", id(node))
            if id(node) in self.lowering.workgroup_roots:
                self.values[cell].add(self.lowering.workgroup_roots[id(node)].identity)
            elif node.initial_value is not None:
                self.bind_value(type_, node.initial_value, cell, types, slots)
            types[node.name], slots[node.name] = type_, cell
            return
        if isinstance(node, ForNode):
            types, slots = dict(types), dict(slots)
            for child in (node.init, node.condition, node.body, node.update):
                self.visit(child, types, slots)
            return
        if isinstance(node, SwitchNode):
            self.visit(node.expression, types, slots)
            types, slots = dict(types), dict(slots)
            for case in node.cases:
                self.visit(case.value, types, slots)
                for statement in case.statements:
                    self.visit(statement, types, slots)
            self.visit(node.default_case, types, slots)
            return
        if isinstance(node, AssignmentNode):
            self.visit(node.value, types, slots)
            self.visit(node.target, types, slots)
            if node.operator == "=":
                self.bind_value(
                    self.lowering.infer(node.target, types),
                    node.value,
                    self.origin(node.target, types, slots),
                    types,
                    slots,
                )
            return
        if isinstance(node, FunctionCallNode):
            for argument in node.arguments:
                self.visit(argument, types, slots)
            callee = self.lowering.function_for_call(node, types)
            if callee is not None:
                for parameter, type_, argument in zip(
                    callee.parameters,
                    self.lowering.parameter_types[id(callee)],
                    node.arguments,
                ):
                    cell = ("value", id(parameter))
                    self.bind_value(type_, argument, cell, types, slots)
                    writeback = (
                        isinstance(parameter.param_type, ReferenceType)
                        and parameter.param_type.is_mutable
                    ) or bool(set(parameter.qualifiers) & {"out", "inout"})
                    if writeback and self.pointer(type_):
                        self.edge(self.origin(argument, types, slots), cell)
            elif getattr(node.function, "name", None) in self.lowering.structs:
                self.bind_value(
                    self.lowering.infer(node, types), node, self.UNKNOWN, types, slots
                )
        elif isinstance(node, ReturnNode):
            self.visit(node.value, types, slots)
            self.bind_value(
                self.lowering.returns[id(self.current)],
                node.value,
                ("return", id(self.current)),
                types,
                slots,
            )
        else:
            for child in node.child_nodes():
                self.visit(child, dict(types), dict(slots))
        if self.pointer(self.lowering.infer(node, types)):
            self.origin(node, types, slots)

    def analyze(self):
        for identity, (parameter, pointer, _root) in enumerate(self.lowering.resources):
            if pointer.space != "threadgroup":
                self.values[("value", id(parameter))].add(identity)
        for overloads in self.lowering.functions.values():
            for function in overloads:
                if id(function) not in self.lowering.reachable:
                    continue
                self.current = function
                types = dict(self.lowering.global_types)
                types.update(
                    zip(
                        (p.name for p in function.parameters),
                        self.lowering.parameter_types[id(function)],
                    )
                )
                slots = {p.name: ("value", id(p)) for p in function.parameters}
                self.visit(function.body, types, slots)

        def propagate(seeds):
            pending = deque(seeds)
            queued = set(pending)
            while pending:
                source = pending.popleft()
                queued.remove(source)
                for destination in self.edges[source]:
                    previous = len(self.values[destination])
                    self.values[destination].update(self.values[source])
                    if (
                        len(self.values[destination]) != previous
                        and destination not in queued
                    ):
                        pending.append(destination)
                        queued.add(destination)

        propagate(list(self.values))
        # An unresolved producer is not an impossible value. Propagate unknown
        # after reaching a fixed point so even a join with known roots stays open.
        unresolved = [cell for cell in self.edges if not self.values[cell]]
        for cell in unresolved:
            self.values[cell].add(None)
        propagate(unresolved)
        return {
            node: frozenset(self.values[cell])
            for node, cell in self.expressions.items()
            if self.values[cell] and None not in self.values[cell]
        }
