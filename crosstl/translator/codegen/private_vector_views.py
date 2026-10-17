"""Lower non-escaping scalar views of fixed private vector arrays.

A view remains a location in its original aggregate. No backing array is copied,
and opaque or escaping pointers are left to the existing target diagnostics.
"""

from copy import copy, deepcopy
from dataclasses import dataclass, replace

from ..ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BlockNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MemberAccessNode,
    ParameterNode,
    PointerReinterpretNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    ReturnNode,
    StructNode,
    SwitchNode,
    TypeNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
    WhileNode,
)
from .array_utils import evaluate_literal_int_expression
from .pointer_reinterpret import PointerReinterpretationError, scalar_storage_layout


def _name(node):
    return node if isinstance(node, str) else getattr(node, "name", None)


def _integer(value):
    return LiteralNode(value, PrimitiveType("int"))


def _private(pointer):
    return isinstance(pointer, PointerType) and pointer.address_space in {
        "thread",
        "private",
        "function",
    }


def _pure(node):
    if isinstance(node, (IdentifierNode, LiteralNode, TypeNode)):
        return True
    if isinstance(node, MemberAccessNode):
        return _pure(node.object_expr)
    if isinstance(node, ArrayAccessNode):
        return _pure(node.array_expr) and _pure(node.index)
    if isinstance(node, UnaryOpNode):
        return node.operator in {"+", "-", "~", "!"} and _pure(node.operand)
    if isinstance(node, BinaryOpNode):
        return (
            node.operator in {
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
            }
            and _pure(node.left)
            and _pure(node.right)
        )
    return False


def _qualifiers(node):
    return {
        str(_name(item)).lower()
        for field in ("qualifiers", "attributes", "resource_qualifiers")
        for item in getattr(node, field, ()) or ()
    }


def _unreference(value):
    return value.referenced_type if isinstance(value, ReferenceType) else value


@dataclass(frozen=True)
class _View:
    backing: object
    root: object
    element: object
    lanes: int
    extent: int
    offset: object
    writable: bool


class _Lowering:
    def __init__(self, ast, target):
        self.ast = ast
        self.target = target
        self.structs = {
            node.name: node for node in ast.walk() if isinstance(node, StructNode)
        }
        self.functions = {}
        for node in ast.walk():
            if isinstance(node, FunctionNode):
                self.functions.setdefault(node.name, []).append(node)
        self.names = {
            node.name
            for node in ast.walk()
            if isinstance(getattr(node, "name", None), str)
        }
        self.accessors = {}
        self.ranges = {}
        for name, functions in self.functions.items():
            if len(functions) != 1:
                continue
            function = functions[0]
            body = getattr(function.body, "statements", [])
            if not (
                _private(function.return_type)
                and len(body) == 1
                and isinstance(body[0], ReturnNode)
            ):
                continue
            value = body[0].value
            if not isinstance(value, PointerReinterpretNode):
                continue
            env = {parameter.name: parameter for parameter in function.parameters}
            view = self.view(value, env, {}, {})
            if view is None or not isinstance(view.root, ParameterNode):
                continue
            qualifiers = set(view.root.qualifiers or ())
            reference = isinstance(view.root.param_type, ReferenceType)
            if not reference and not (
                {"in", "inout"} & qualifiers and "thread" in qualifiers
            ):
                continue
            if function.return_type.is_mutable and not (
                reference or "inout" in qualifiers
            ):
                continue
            if scalar_storage_layout(
                _name(function.return_type.pointee_type)
            ) != scalar_storage_layout(_name(view.element)):
                continue
            self.accessors[name] = (function, value)

    def error(self, reason, node):
        raise PointerReinterpretationError(
            f"Cannot preserve private vector-array view: {reason}",
            reason=reason,
            address_space="thread",
            target_backend=self.target,
            source_location=getattr(node, "source_location", None),
        )

    def fresh(self):
        index = 0
        while f"crosstl_view_index_{index}" in self.names:
            index += 1
        name = f"crosstl_view_index_{index}"
        self.names.add(name)
        return name

    def type_of(self, node, env):
        if isinstance(node, IdentifierNode):
            binding = env.get(node.name)
            result = getattr(binding, "var_type", getattr(binding, "param_type", None))
            return (
                result.referenced_type if isinstance(result, ReferenceType) else result
            )
        if isinstance(node, MemberAccessNode):
            struct = self.structs.get(_name(self.type_of(node.object_expr, env)))
            if (
                struct is not None
                and not struct.inheritance
                and not getattr(struct, "is_union", False)
            ):
                member = next(
                    (item for item in struct.members if item.name == node.member), None
                )
                if member is not None:
                    if _qualifiers(member) & {
                        "static",
                        "metal_static",
                        "volatile",
                        "metal_volatile",
                    }:
                        self.error("backing-member-unsupported", node)
                    return member.member_type
        if isinstance(node, ArrayAccessNode):
            parent = self.type_of(node.array_expr, env)
            return getattr(parent, "element_type", None)
        if isinstance(node, LiteralNode):
            return node.literal_type
        return None

    def readonly(self, node, env):
        if isinstance(node, IdentifierNode):
            binding = env.get(node.name)
            qualifiers = _qualifiers(binding)
            value = getattr(binding, "param_type", getattr(binding, "var_type", None))
            return bool(qualifiers & {"const", "constant", "constexpr", "in"}) or (
                isinstance(value, ReferenceType) and not value.is_mutable
            )
        if isinstance(node, ArrayAccessNode):
            return self.readonly(node.array_expr, env)
        if isinstance(node, MemberAccessNode):
            struct = self.structs.get(_name(self.type_of(node.object_expr, env)))
            member = (
                next(
                    (item for item in struct.members if item.name == node.member), None
                )
                if struct
                else None
            )
            return bool(
                _qualifiers(member) & {"const", "constant", "metal_const"}
            ) or self.readonly(node.object_expr, env)
        return True

    def interval(self, node, env, constants):
        literal = evaluate_literal_int_expression(node, constants)
        if literal is not None:
            return literal, literal
        if isinstance(node, IdentifierNode):
            known = self.ranges.get(id(env.get(node.name)))
            if known is not None:
                return known
            layout = scalar_storage_layout(_name(self.type_of(node, env)))
            if layout is not None and layout.kind == "integer":
                return (
                    (-(1 << (layout.bit_width - 1)), (1 << (layout.bit_width - 1)) - 1)
                    if layout.signed
                    else (0, (1 << layout.bit_width) - 1)
                )
        if isinstance(node, BinaryOpNode):
            left, right = self.interval(node.left, env, constants), self.interval(
                node.right, env, constants
            )
            if left is None or right is None:
                return None
            a, b = left
            c, d = right
            if node.operator == "+":
                return a + c, b + d
            if node.operator == "-":
                return a - d, b - c
            if node.operator == "*":
                products = (a * c, a * d, b * c, b * d)
                return min(products), max(products)
            if node.operator == "&" and c == d and 0 <= c < (1 << 31):
                return 0, c
            if node.operator == "%" and a >= 0 and c == d and c > 0:
                return 0, c - 1
        return None

    def checked_index(self, node, env, constants, maximum, reason):
        bounds = self.interval(node, env, constants)
        if bounds is None or not 0 <= bounds[0] <= bounds[1] <= maximum:
            self.error(reason, node)
        return bounds

    @staticmethod
    def root(node, env):
        while isinstance(node, (MemberAccessNode, ArrayAccessNode)):
            node = (
                node.object_expr
                if isinstance(node, MemberAccessNode)
                else node.array_expr
            )
        return env.get(node.name) if isinstance(node, IdentifierNode) else None

    def substitute(self, node, arguments):
        if isinstance(node, IdentifierNode) and node.name in arguments:
            return arguments[node.name]
        return self.rewrite_children(
            node, lambda child: self.substitute(child, arguments)
        )

    def view(self, node, env, aliases, constants):
        if isinstance(node, IdentifierNode):
            value = aliases.get(node.name)
            if value is not None and env.get(value.root.name) is not value.root:
                self.error("backing-shadowed", node)
            return value
        if (
            isinstance(node, FunctionCallNode)
            and _name(node.function) in self.accessors
        ):
            function, returned = self.accessors[_name(node.function)]
            if len(node.arguments) != len(function.parameters):
                self.error("accessor-arity", node)
            if any(not _pure(argument) for argument in node.arguments):
                self.error("accessor-argument-side-effects", node)
            arguments = {
                parameter.name: argument
                for parameter, argument in zip(function.parameters, node.arguments)
            }
            returned_root = self.root(
                returned.expression,
                {parameter.name: parameter for parameter in function.parameters},
            )
            actual_type = self.type_of(arguments[returned_root.name], env)
            if repr(actual_type) != repr(_unreference(returned_root.param_type)):
                self.error("accessor-backing-type-mismatch", node)
            value = self.view(
                self.substitute(returned, arguments), env, aliases, constants
            )
            if value is None:
                self.error("accessor-backing-unresolved", node)
            return replace(
                value, writable=value.writable and function.return_type.is_mutable
            )
        if isinstance(node, BinaryOpNode) and node.operator in {"+", "-"}:
            value = self.view(node.left, env, aliases, constants)
            if value is not None:
                if not _pure(node.right):
                    self.error("offset-side-effects", node)
                return replace(
                    value, offset=BinaryOpNode(value.offset, node.operator, node.right)
                )
        if not isinstance(node, PointerReinterpretNode) or not _private(
            node.target_type
        ):
            return None
        backing = node.expression
        source = self.type_of(backing, env)
        if not isinstance(source, ArrayType) or not isinstance(
            source.element_type, VectorType
        ):
            return None
        vector = source.element_type
        extent = evaluate_literal_int_expression(source.size, constants)
        source_layout = scalar_storage_layout(_name(vector.element_type))
        target_layout = scalar_storage_layout(_name(node.target_type.pointee_type))
        if (
            vector.size not in {2, 4}
            or extent is None
            or not 0 < extent <= ((1 << 31) - 1) // vector.size
        ):
            self.error("unproven-vector-array-layout", node)
        if (
            source_layout != target_layout
            or source_layout is None
            or source_layout.bit_width != 32
        ):
            self.error("incompatible-scalar-layout", node)
        if not _pure(backing):
            self.error("backing-side-effects", node)
        root = self.root(backing, env)
        if root is None or not isinstance(root, (VariableNode, ParameterNode)):
            self.error("backing-lifetime-unresolved", node)
        qualifiers = _qualifiers(root)
        if qualifiers & {"volatile", "device", "constant", "threadgroup", "workgroup"}:
            self.error("backing-storage-unsupported", node)
        root_type = getattr(root, "param_type", getattr(root, "var_type", None))
        if isinstance(root_type, ReferenceType) and (
            root_type.address_space not in {None, "thread", "private", "function"}
            or root_type.resource_qualifiers
        ):
            self.error("backing-storage-unsupported", node)
        if (
            node.target_type.resource_qualifiers
            or node.target_type.access_mode not in {None, "read", "read_write"}
        ):
            self.error("view-qualifiers-unsupported", node)
        readonly = self.readonly(backing, env)
        if readonly and node.target_type.is_mutable:
            self.error("readonly-view-conversion", node)
        writable = (
            node.target_type.is_mutable
            and node.target_type.access_mode != "read"
            and not readonly
        )
        return _View(
            backing,
            root,
            vector.element_type,
            vector.size,
            extent,
            _integer(0),
            writable,
        )

    def capture(self, value, env, constants, prefix):
        def capture_index(node):
            if isinstance(node, ArrayAccessNode):
                parent = capture_index(node.array_expr)
                parent_type = self.type_of(node.array_expr, env)
                size = evaluate_literal_int_expression(
                    getattr(parent_type, "size", None), constants
                )
                if size is None:
                    self.error("backing-array-extent-unproven", node)
                self.checked_index(
                    node.index, env, constants, size - 1, "backing-index-unproven"
                )
                index = snapshot(node.index)
                return ArrayAccessNode(parent, index)
            if isinstance(node, MemberAccessNode):
                return MemberAccessNode(capture_index(node.object_expr), node.member)
            return node

        def snapshot(node):
            bounds = self.checked_index(
                node, env, constants, (1 << 31) - 1, "captured-index-unproven"
            )
            literal = evaluate_literal_int_expression(node, constants)
            if literal is not None:
                return _integer(literal)
            name = self.fresh()
            declaration = VariableNode(
                name,
                PrimitiveType("int"),
                FunctionCallNode("int", [node]),
                qualifiers=["const"],
            )
            prefix.append(declaration)
            env[name] = declaration
            self.ranges[id(declaration)] = bounds
            return IdentifierNode(name)

        return replace(
            value, backing=capture_index(value.backing), offset=snapshot(value.offset)
        )

    def access(self, value, index, env, constants, write, node):
        if write and not value.writable:
            self.error("write-through-readonly-view", node)
        if not _pure(index):
            self.error("index-side-effects", node)
        combined = BinaryOpNode(value.offset, "+", index)
        self.checked_index(
            combined,
            env,
            constants,
            value.extent * value.lanes - 1,
            "index-range-unproven",
        )
        literal = evaluate_literal_int_expression(combined, constants)
        if literal is not None:
            if not 0 <= literal < value.extent * value.lanes:
                self.error("index-out-of-bounds", node)
            row, lane = _integer(literal // value.lanes), _integer(
                literal % value.lanes
            )
        else:
            combined = BinaryOpNode(
                FunctionCallNode("int", [value.offset]),
                "+",
                FunctionCallNode("int", [index]),
            )
            row = BinaryOpNode(combined, "/", _integer(value.lanes))
            lane = BinaryOpNode(deepcopy(combined), "%", _integer(value.lanes))
        return ArrayAccessNode(
            ArrayAccessNode(value.backing, row),
            lane,
            source_location=getattr(node, "source_location", None),
            annotations=deepcopy(getattr(node, "annotations", {})),
        )

    def expression(self, node, env, aliases, constants, write=False):
        def contains_view(expression):
            return isinstance(expression, ASTNode) and any(
                (isinstance(child, IdentifierNode) and child.name in aliases)
                or (
                    isinstance(child, FunctionCallNode)
                    and _name(child.function) in self.accessors
                )
                or (
                    isinstance(child, PointerReinterpretNode)
                    and self.view(child, env, aliases, constants) is not None
                )
                for child in expression.walk()
            )

        if isinstance(node, ArrayAccessNode):
            value = self.view(node.array_expr, env, aliases, constants)
            if value is not None:
                return self.access(value, node.index, env, constants, write, node)
        if isinstance(node, UnaryOpNode):
            if node.operator == "&" and contains_view(node.operand):
                self.error("view-escape", node)
            if node.operator == "*":
                value = self.view(node.operand, env, aliases, constants)
                if value is not None:
                    return self.access(value, _integer(0), env, constants, write, node)
            if node.operator in {"++", "--"}:
                result = copy(node)
                result.operand = self.expression(
                    node.operand, env, aliases, constants, True
                )
                return result
        if isinstance(node, AssignmentNode):
            result = copy(node)
            result.target = result.left = self.expression(
                node.target, env, aliases, constants, True
            )
            result.value = result.right = self.expression(
                node.value, env, aliases, constants
            )
            return result
        if isinstance(node, IdentifierNode) and node.name in aliases:
            self.error("view-escape", node)
        if (
            isinstance(node, FunctionCallNode)
            and _name(node.function) in self.accessors
        ):
            self.error("view-escape", node)
        if isinstance(node, FunctionCallNode):
            # A scalar view passed by reference needs alias-aware copyout, not an
            # ordinary value argument or a target-generated temporary.
            for position, argument in enumerate(node.arguments):
                if contains_view(argument):
                    candidates = self.functions.get(_name(node.function), ())
                    constructor = (
                        scalar_storage_layout(_name(node.function)) is not None
                    )
                    if (not candidates and not constructor) or any(
                        position >= len(fn.parameters)
                        or isinstance(
                            fn.parameters[position].param_type,
                            (PointerType, ReferenceType),
                        )
                        or _qualifiers(fn.parameters[position]) & {"out", "inout"}
                        for fn in candidates
                    ):
                        self.error("view-call-escape", node)
        return self.rewrite_children(
            node, lambda child: self.expression(child, env, aliases, constants)
        )

    def rewrite_children(self, node, rewrite):
        if not isinstance(node, ASTNode) or isinstance(node, TypeNode):
            return node
        result = copy(node)
        memo = {}

        def transform(value):
            if isinstance(value, ASTNode):
                if id(value) not in memo:
                    memo[id(value)] = rewrite(value)
                return memo[id(value)]
            if isinstance(value, list):
                return [transform(item) for item in value]
            if isinstance(value, tuple):
                return tuple(transform(item) for item in value)
            if isinstance(value, dict):
                return {key: transform(item) for key, item in value.items()}
            return value

        for key, value in vars(node).items():
            if key not in AST_CHILD_FIELD_EXCLUSIONS:
                setattr(result, key, transform(value))
        return result

    def loop_range(self, node, env, constants):
        init = node.init
        update = (
            node.update.expression
            if isinstance(node.update, ExpressionStatementNode)
            else node.update
        )
        condition = node.condition
        if (
            not isinstance(init, VariableNode)
            or not isinstance(condition, BinaryOpNode)
            or not isinstance(condition.left, IdentifierNode)
            or condition.left.name != init.name
            or condition.operator not in {"<", "<="}
        ):
            return
        start = evaluate_literal_int_expression(init.initial_value, constants)
        end = evaluate_literal_int_expression(condition.right, constants)
        if start is None or end is None:
            return
        upper = end - (condition.operator == "<")
        layout = scalar_storage_layout(_name(init.var_type))
        if (
            layout is None
            or layout.kind != "integer"
            or not 0 <= start <= upper < (1 << 31) - 1
        ):
            return
        increment = (
            isinstance(update, UnaryOpNode)
            and update.operator == "++"
            and isinstance(update.operand, IdentifierNode)
            and update.operand.name == init.name
        )
        increment |= (
            isinstance(update, AssignmentNode)
            and update.operator == "+="
            and isinstance(update.target, IdentifierNode)
            and update.target.name == init.name
            and evaluate_literal_int_expression(update.value, constants) == 1
        )
        if not increment:
            return

        for child in node.body.walk():
            if (
                isinstance(child, VariableNode)
                and child.name == init.name
                or isinstance(child, AssignmentNode)
                and isinstance(child.target, IdentifierNode)
                and child.target.name == init.name
                or isinstance(child, UnaryOpNode)
                and child.operator in {"++", "--", "&"}
                and isinstance(child.operand, IdentifierNode)
                and child.operand.name == init.name
                or isinstance(child, FunctionCallNode)
                and any(
                    isinstance(argument, IdentifierNode) and argument.name == init.name
                    for argument in child.arguments
                )
            ):
                return
        self.ranges[id(env[init.name])] = (start, upper)

    def statement(self, node, env, aliases, constants):
        if isinstance(node, BlockNode):
            env, aliases, constants = dict(env), dict(aliases), dict(constants)
            statements = []
            for item in node.statements:
                statements.extend(self.statement(item, env, aliases, constants))
            result = copy(node)
            result.statements = statements
            return [result]
        if isinstance(node, VariableNode):
            prefix = []
            value = self.view(node.initial_value, env, aliases, constants)
            if value is not None and _private(node.var_type):
                if (
                    _qualifiers(node) & {"volatile", "metal_volatile"}
                    or node.var_type.resource_qualifiers
                    or node.var_type.access_mode not in {None, "read", "read_write"}
                ):
                    self.error("view-qualifiers-unsupported", node)
                mutable = (
                    node.var_type.is_mutable
                    and not _qualifiers(node)
                    & {
                        "const",
                        "constant",
                    }
                    and node.var_type.access_mode != "read"
                )
                if mutable and not value.writable:
                    self.error("readonly-view-conversion", node)
                if scalar_storage_layout(
                    _name(node.var_type.pointee_type)
                ) != scalar_storage_layout(_name(value.element)):
                    self.error("incompatible-scalar-layout", node)
                aliases[node.name] = self.capture(
                    replace(value, writable=value.writable and mutable),
                    env,
                    constants,
                    prefix,
                )
                env[node.name] = node
                constants.pop(node.name, None)
                return prefix
            initial = self.expression(node.initial_value, env, aliases, constants)
            result = copy(node)
            result.initial_value = initial
            env[node.name] = node
            aliases.pop(node.name, None)
            constants.pop(node.name, None)
            if {"const", "constant", "constexpr"} & set(node.qualifiers or ()):
                literal = evaluate_literal_int_expression(node.initial_value, constants)
                layout = scalar_storage_layout(_name(node.var_type))
                representable = (
                    layout is not None
                    and layout.kind == "integer"
                    and layout.bit_width in {32, 64}
                )
                limits = (
                    (-(1 << (layout.bit_width - 1)), (1 << (layout.bit_width - 1)) - 1)
                    if representable and layout.signed
                    else (0, (1 << layout.bit_width) - 1) if representable else (1, 0)
                )
                if literal is not None and limits[0] <= literal <= limits[1]:
                    constants[node.name] = literal
                elif representable and _pure(node.initial_value):
                    bounds = self.interval(node.initial_value, env, constants)
                    if (
                        bounds is not None
                        and limits[0] <= bounds[0] <= bounds[1] <= limits[1]
                    ):
                        self.ranges[id(node)] = bounds
            return [result]
        if isinstance(node, IfNode):
            result = copy(node)
            result.condition = result.if_condition = self.expression(
                node.condition, env, aliases, constants
            )
            result.then_branch = result.if_body = self.branch(
                node.then_branch, env, aliases, constants
            )
            result.else_branch = result.else_body = self.branch(
                node.else_branch, env, aliases, constants
            )
            result.else_if_conditions = [
                self.expression(value, env, aliases, constants)
                for value in node.else_if_conditions
            ]
            result.else_if_bodies = [
                self.branch(value, env, aliases, constants)
                for value in node.else_if_bodies
            ]
            return [result]
        if isinstance(node, ForNode):
            scoped, views, values = dict(env), dict(aliases), dict(constants)
            init = self.statement(node.init, scoped, views, values)
            if len(init) != 1:
                self.error("view-in-loop-initializer", node)
            self.loop_range(node, scoped, values)
            result = copy(node)
            result.init = init[0]
            result.condition = self.expression(node.condition, scoped, views, values)
            result.update = self.expression(node.update, scoped, views, values)
            result.body = self.branch(node.body, scoped, views, values)
            return [result]
        if isinstance(node, (WhileNode, DoWhileNode, LoopNode)):
            result = copy(node)
            if hasattr(node, "condition"):
                result.condition = self.expression(
                    node.condition, env, aliases, constants
                )
            result.body = self.branch(node.body, env, aliases, constants)
            return [result]
        if isinstance(node, SwitchNode):
            result = copy(node)
            result.expression = self.expression(
                node.expression, env, aliases, constants
            )
            result.cases = []
            scoped, views, values = dict(env), dict(aliases), dict(constants)
            for case in node.cases:
                statements = []
                for statement in case.statements:
                    statements.extend(self.statement(statement, scoped, views, values))
                rewritten = copy(case)
                rewritten.statements = statements
                result.cases.append(rewritten)
            result.default_case = self.branch(node.default_case, scoped, views, values)
            return [result]
        return [self.expression(node, env, aliases, constants)]

    def branch(self, node, env, aliases, constants):
        if node is None:
            return None
        statements = self.statement(node, dict(env), dict(aliases), dict(constants))
        return statements[0] if len(statements) == 1 else BlockNode(statements)

    def run(self):
        for functions in self.functions.values():
            for function in functions:
                if function.name in self.accessors:
                    continue
                env = {parameter.name: parameter for parameter in function.parameters}
                function.body = self.branch(function.body, env, {}, {})
        removed = {id(function) for function, _ in self.accessors.values()}
        self.ast.functions = [
            function for function in self.ast.functions if id(function) not in removed
        ]
        for stage in self.ast.stages.values():
            stage.local_functions = [
                function
                for function in stage.local_functions
                if id(function) not in removed
            ]
        for node in self.ast.walk():
            if (
                isinstance(node, FunctionCallNode)
                and _name(node.function) in self.accessors
            ):
                self.error("view-escape", node)
        return self.ast


def lower_private_vector_views(ast, target):
    """Retain source ASTs and rewrite only explicit private vector-array views."""
    if not isinstance(ast, ASTNode):
        return ast
    if not any(
        isinstance(node, PointerReinterpretNode) and _private(node.target_type)
        for node in ast.walk()
    ):
        return ast
    return _Lowering(deepcopy(ast), target).run()
