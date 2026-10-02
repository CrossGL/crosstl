"""Lower private aggregates containing storage pointers to resource handles.

Handles are private values, never a buffer ABI. Their identity is drawn from the
selected entry's bindings; access helpers branch over concrete resource arguments
instead of constructing target resource-object arrays. The source AST is retained.
"""

from copy import copy, deepcopy
from dataclasses import dataclass

from ..ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    ParameterNode,
    PointerReinterpretNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    ReturnNode,
    StructMemberNode,
    StructNode,
    TernaryOpNode,
    TypeNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import evaluate_literal_int_expression


class ResourceAggregateError(ValueError):
    """An aggregate resource reference cannot be represented without guessing."""

    project_diagnostic_code = "project.translate.resource-aggregate-unsupported"
    missing_capabilities = ("resource-aggregate-lowering",)

    def __init__(self, reason, node=None):
        super().__init__(f"Cannot preserve aggregate resource references: {reason}")
        self.reason = reason
        self.source_location = getattr(node, "source_location", None)


@dataclass(frozen=True)
class _Pointer:
    element: str
    space: str
    writable: bool
    readable: bool


@dataclass(frozen=True)
class _RootPointer(_Pointer):
    identity: int
    local_name: str


def _name(value):
    return value if isinstance(value, str) else getattr(value, "name", None)


def _id(name):
    return IdentifierNode(name)


def _call(name, args):
    return FunctionCallNode(_id(name), args)


def _integer(value):
    return LiteralNode(value, "int")


def _member(value, name):
    return MemberAccessNode(value, name)


def _pointer_type(value, owner=None):
    qualifiers = set(getattr(owner, "qualifiers", ()) or ())
    qualifiers.update(_name(item) for item in getattr(owner, "attributes", ()) or ())

    def element_name(value):
        return {"int64": "int64_t", "uint64": "uint64_t"}.get(
            _name(value), _name(value)
        )

    if (
        isinstance(value, NamedType)
        and value.name
        in {
            "StructuredBuffer",
            "RWStructuredBuffer",
        }
        and len(value.generic_args) == 1
    ):
        if getattr(owner, "resource_qualifiers", None) or "volatile" in qualifiers:
            raise ResourceAggregateError("qualified-resource-memory", owner)
        element = element_name(value.generic_args[0])
        if element:
            space = "constant" if "constant" in qualifiers else "device"
            return _Pointer(
                element,
                space,
                value.name == "RWStructuredBuffer",
                not bool(qualifiers & {"writeonly", "write"}),
            )
    if not isinstance(value, PointerType):
        return None
    space = value.address_space or next(
        (q for q in ("constant", "device", "global", "storage") if q in qualifiers),
        None,
    )
    if space not in {"constant", "device", "global", "storage"}:
        return None
    element = element_name(value.pointee_type)
    if not element or getattr(value.pointee_type, "generic_args", None):
        raise ResourceAggregateError("non-scalar-pointee", owner)
    writable = not (
        space == "constant"
        or "const" in qualifiers
        or value.access_mode in {"read", "readonly"}
        or not value.is_mutable
    )
    if value.resource_qualifiers or "volatile" in qualifiers:
        raise ResourceAggregateError("qualified-resource-memory", owner)
    readable = value.access_mode not in {"write", "writeonly"} and not bool(
        qualifiers & {"write", "writeonly"}
    )
    return _Pointer(
        element, "constant" if space == "constant" else "device", writable, readable
    )


def _contains_storage_pointer(node):
    return any(
        isinstance(child, PointerType)
        and child.address_space in {"constant", "device", "global", "storage"}
        for child in node.walk()
    )


class _Lowering:
    def __init__(self, ast, entry):
        self.ast = ast
        self.entry = entry
        self.reserved = {
            node.name
            for node in ast.walk()
            if isinstance(getattr(node, "name", None), str)
        }
        self.structs = {}
        for node in ast.walk():
            if isinstance(node, StructNode):
                if node.name in self.structs:
                    raise ResourceAggregateError("ambiguous-aggregate-type", node)
                self.structs[node.name] = node
        self.functions = {}
        for node in ast.walk():
            if isinstance(node, FunctionNode):
                self.functions.setdefault(node.name, []).append(node)
        self.fields = {
            name: {
                member.name: self.source_type(member.member_type, member)
                for member in struct.members
            }
            for name, struct in self.structs.items()
        }
        self.resources = []
        for param in entry.parameters:
            pointer = _pointer_type(param.param_type, param)
            if pointer is not None:
                if self.contains_reference(NamedType(pointer.element)):
                    raise ResourceAggregateError("aggregate-buffer-abi", param)
                self.resources.append((param, pointer, self.fresh("crosstl_resource")))
            elif self.contains_reference(self.source_type(param.param_type, param)):
                raise ResourceAggregateError("aggregate-buffer-abi", param)
        for variable in ast.global_variables:
            if not getattr(
                variable, "is_type_alias", False
            ) and self.contains_reference(
                self.source_type(getattr(variable, "var_type", None), variable)
            ):
                raise ResourceAggregateError("global-resource-reference", variable)
        for stage in ast.stages.values():
            for variable in stage.local_variables:
                if self.contains_reference(
                    self.source_type(variable.var_type, variable)
                ):
                    raise ResourceAggregateError("global-resource-reference", variable)
        cbuffers = list(ast.cbuffers)
        cbuffers.extend(
            buffer for stage in ast.stages.values() for buffer in stage.local_cbuffers
        )
        for buffer in cbuffers:
            if _contains_storage_pointer(buffer):
                raise ResourceAggregateError("aggregate-buffer-abi", buffer)
        self.handles = {}
        self.helpers = {}
        self.generated = []
        self.entry_handles = {
            param.name: self.fresh(f"crosstl_reference_{param.name}")
            for param, _pointer, _root in self.resources
        }
        self.parameter_types = {
            id(function): [
                self.source_type(p.param_type, p) for p in function.parameters
            ]
            for overloads in self.functions.values()
            for function in overloads
        }
        self.returns = {
            id(function): self.source_type(function.return_type, function)
            for overloads in self.functions.values()
            for function in overloads
        }
        self.reachable = set()
        self.current = entry

    def fresh(self, stem):
        name = stem
        index = 2
        while name in self.reserved:
            name = f"{stem}_{index}"
            index += 1
        self.reserved.add(name)
        return name

    def source_type(self, value, owner=None):
        pointer = _pointer_type(value, owner)
        if pointer is not None:
            return pointer
        if isinstance(value, ArrayType):
            return (self.source_type(value.element_type, owner), value.size)
        if isinstance(value, ReferenceType):
            return self.source_type(value.referenced_type, owner)
        return value

    def handle_name(self, pointer):
        if pointer.element not in self.handles:
            name = self.fresh(f"crosstl_resource_ref_{pointer.element}")
            self.handles[pointer.element] = name
            self.generated.append(
                StructNode(
                    name,
                    [
                        StructMemberNode("identity", PrimitiveType("int")),
                        StructMemberNode("offset", PrimitiveType("int64_t")),
                    ],
                )
            )
        return self.handles[pointer.element]

    def target_type(self, source):
        if isinstance(source, _Pointer):
            return NamedType(self.handle_name(source))
        if isinstance(source, tuple):
            return ArrayType(self.target_type(source[0]), deepcopy(source[1]))
        return deepcopy(source)

    def resource_parameters(self):
        return [
            ParameterNode(name, deepcopy(param.param_type))
            for param, _pointer, name in self.resources
        ]

    def resource_arguments(self):
        return [_id(name) for _param, _pointer, name in self.resources]

    def function_for_call(self, node):
        candidates = self.functions.get(_name(node.function), [])
        candidates = [
            f
            for f in candidates
            if len(self.parameter_types[id(f)]) == len(node.arguments)
        ]
        if len(candidates) > 1:
            raise ResourceAggregateError("overloaded-resource-call", node)
        return candidates[0] if candidates else None

    def discover(self, function):
        if id(function) in self.reachable:
            return
        self.reachable.add(id(function))
        if isinstance(function.return_type, ReferenceType):
            raise ResourceAggregateError("reference-return", function)
        if function.body is None:
            raise ResourceAggregateError("external-resource-call", function)
        for node in function.body.walk():
            if isinstance(node, FunctionCallNode):
                callee = self.function_for_call(node)
                if callee is not None:
                    if callee is self.entry:
                        raise ResourceAggregateError("recursive-entry", node)
                    self.discover(callee)

    def infer(self, node, env):
        if isinstance(node, IdentifierNode):
            return env.get(node.name)
        if isinstance(node, MemberAccessNode):
            return self.fields.get(_name(self.infer(node.object_expr, env)), {}).get(
                node.member
            )
        if isinstance(node, ArrayAccessNode):
            owner = self.infer(node.array_expr, env)
            if isinstance(owner, tuple):
                return owner[0]
            if isinstance(owner, _Pointer):
                return PrimitiveType(owner.element)
        if isinstance(node, FunctionCallNode):
            if _name(node.function) in self.structs:
                return NamedType(_name(node.function))
            callee = self.function_for_call(node)
            if callee is not None:
                return self.returns[id(callee)]
        if isinstance(node, BinaryOpNode) and node.operator in {"+", "-"}:
            left = self.infer(node.left, env)
            if isinstance(left, _Pointer):
                if isinstance(self.infer(node.right, env), _Pointer):
                    return PrimitiveType("int64_t")
                return left
        if isinstance(node, UnaryOpNode):
            if node.operator == "&" and isinstance(node.operand, ArrayAccessNode):
                return self.infer(node.operand.array_expr, env)
            if node.operator in {"++", "--"}:
                return self.infer(node.operand, env)
        if isinstance(node, TernaryOpNode):
            left, right = self.infer(node.true_expr, env), self.infer(
                node.false_expr, env
            )
            if isinstance(left, _Pointer) and left == right:
                return left
        return None

    def contains_reference(self, source, visited=()):
        if isinstance(source, _Pointer):
            return True
        if isinstance(source, tuple):
            return self.contains_reference(source[0], visited)
        name = _name(source)
        return name not in visited and any(
            self.contains_reference(field, (*visited, name))
            for field in self.fields.get(name, {}).values()
        )

    def compatible(self, expected, actual, node):
        if isinstance(expected, _Pointer):
            if not isinstance(actual, _Pointer) or (
                expected.element != actual.element
                or expected.writable
                and not actual.writable
                or expected.readable
                and not actual.readable
                or expected.space != actual.space
            ):
                raise ResourceAggregateError("pointer-contract-mismatch", node)

    def helper(self, pointer, operation):
        pointer = _Pointer(
            pointer.element, pointer.space, pointer.writable, pointer.readable
        )
        key = pointer, operation
        if key in self.helpers:
            return self.helpers[key]
        name = self.fresh(f"crosstl_resource_{operation}_{pointer.element}")
        self.helpers[key] = name
        handle = self.target_type(pointer)
        params = [ParameterNode("reference", handle)]
        if operation in {"shift", "load", "store"}:
            params.append(ParameterNode("index", PrimitiveType("int64_t")))
        if operation == "make":
            params = [ParameterNode("identity", PrimitiveType("int"))]
            body = [
                VariableNode("reference", handle),
                AssignmentNode(_member(_id("reference"), "identity"), _id("identity")),
                AssignmentNode(_member(_id("reference"), "offset"), _integer(0)),
                ReturnNode(_id("reference")),
            ]
            result_type = handle
        elif operation == "shift":
            body = [
                AssignmentNode(_member(_id("reference"), "offset"), _id("index"), "+="),
                ReturnNode(_id("reference")),
            ]
            result_type = handle
        else:
            if operation == "load" and not pointer.readable:
                raise ResourceAggregateError("read-through-writeonly-pointer")
            if operation == "store":
                if not pointer.writable:
                    raise ResourceAggregateError("write-through-readonly-pointer")
                params.append(ParameterNode("value", PrimitiveType(pointer.element)))
            params.extend(self.resource_parameters())
            candidates = [
                (index, resource_name)
                for index, (_param, resource, resource_name) in enumerate(
                    self.resources
                )
                if resource.element == pointer.element
                and (operation != "store" or resource.writable)
                and (operation != "load" or resource.readable)
            ]
            if not candidates:
                raise ResourceAggregateError("missing-compatible-resource")
            body = []
            for index, resource_name in candidates:
                access = ArrayAccessNode(
                    _id(resource_name),
                    BinaryOpNode(
                        _member(_id("reference"), "offset"), "+", _id("index")
                    ),
                )
                statements = (
                    [ReturnNode(access)]
                    if operation == "load"
                    else [
                        AssignmentNode(access, _id("value")),
                        ReturnNode(_id("value")),
                    ]
                )
                # Every handle originates at one of these compatible bindings.
                if index == candidates[-1][0]:
                    body.extend(statements)
                else:
                    body.append(
                        IfNode(
                            BinaryOpNode(
                                _member(_id("reference"), "identity"),
                                "==",
                                _integer(index),
                            ),
                            BlockNode(statements),
                        )
                    )
            result_type = PrimitiveType(pointer.element)
        function = FunctionNode(name, result_type, params, BlockNode(body))
        function.resource_aggregate_nonmutating = operation in {"make", "shift", "load"}
        self.generated.append(function)
        return name

    def access(self, node, env):
        if (
            isinstance(node, FunctionCallNode)
            and _name(node.function) == "buffer_load"
            and "buffer_load" not in self.functions
            and len(node.arguments) == 2
        ):
            pointer = self.infer(node.arguments[0], env)
            if isinstance(pointer, _Pointer):
                return pointer, *node.arguments
        if isinstance(node, ArrayAccessNode):
            pointer = self.infer(node.array_expr, env)
            if isinstance(pointer, _Pointer):
                return pointer, node.array_expr, node.index_expr
        if isinstance(node, UnaryOpNode) and node.operator == "*":
            pointer = self.infer(node.operand, env)
            if isinstance(pointer, _Pointer):
                return pointer, node.operand, _integer(0)
        return None

    def expression(self, node, env, expected=None):
        if node is None:
            return None
        if isinstance(node, (BlockNode, VariableNode, IfNode, ForNode, ReturnNode)):
            return self.statement(node, env)
        actual = self.infer(node, env)
        self.compatible(expected, actual, node)
        location = getattr(node, "source_location", None)
        if isinstance(node, IdentifierNode):
            if isinstance(actual, _RootPointer):
                return _id(actual.local_name)
            return node
        if isinstance(node, AssignmentNode):
            access = self.access(node.target, env)
            if access is not None:
                if node.operator != "=":
                    raise ResourceAggregateError("compound-resource-write", node)
                pointer, owner, index = access
                return _call(
                    self.helper(pointer, "store"),
                    [
                        self.expression(owner, env),
                        self.expression(index, env),
                        self.expression(node.value, env),
                        *self.resource_arguments(),
                    ],
                )
            target_type = self.infer(node.target, env)
            target = self.expression(node.target, env)
            value = self.expression(
                node.value, env, target_type if node.operator == "=" else None
            )
            if isinstance(target_type, _Pointer) and node.operator in {"+=", "-="}:
                target = _member(target, "offset")
                value = _call("int64_t", [value])
            elif isinstance(target_type, _Pointer) and node.operator != "=":
                raise ResourceAggregateError("pointer-assignment-operator", node)
            return AssignmentNode(
                target, value, node.operator, source_location=location
            )
        access = self.access(node, env)
        if access is not None:
            pointer, owner, index = access
            return _call(
                self.helper(pointer, "load"),
                [
                    self.expression(owner, env),
                    self.expression(index, env),
                    *self.resource_arguments(),
                ],
            )
        if isinstance(node, BinaryOpNode):
            left_type, right_type = self.infer(node.left, env), self.infer(
                node.right, env
            )
            if isinstance(left_type, _Pointer):
                if node.operator not in {"+", "-"} or isinstance(right_type, _Pointer):
                    raise ResourceAggregateError("pointer-binary-operator", node)
                delta = _call("int64_t", [self.expression(node.right, env)])
                if node.operator == "-":
                    delta = UnaryOpNode("-", delta)
                return _call(
                    self.helper(left_type, "shift"),
                    [self.expression(node.left, env), delta],
                )
            if isinstance(right_type, _Pointer):
                raise ResourceAggregateError("pointer-binary-operator", node)
            return BinaryOpNode(
                self.expression(node.left, env),
                node.operator,
                self.expression(node.right, env),
                source_location=location,
            )
        if isinstance(node, UnaryOpNode):
            if node.operator == "&":
                if self.contains_reference(self.infer(node.operand, env)):
                    raise ResourceAggregateError("aggregate-address-escape", node)
                access = self.access(node.operand, env)
                if access is not None:
                    pointer, owner, index = access
                    return _call(
                        self.helper(pointer, "shift"),
                        [self.expression(owner, env), self.expression(index, env)],
                    )
            if isinstance(self.infer(node.operand, env), _Pointer):
                raise ResourceAggregateError("pointer-unary-operator", node)
            return UnaryOpNode(
                node.operator,
                self.expression(node.operand, env),
                node.is_postfix,
                source_location=location,
            )
        if isinstance(node, MemberAccessNode):
            return MemberAccessNode(
                self.expression(node.object_expr, env),
                node.member,
                source_location=location,
            )
        if isinstance(node, ArrayAccessNode):
            return ArrayAccessNode(
                self.expression(node.array_expr, env),
                self.expression(node.index_expr, env),
                source_location=location,
            )
        if isinstance(node, ArrayLiteralNode):
            fields = self.fields.get(_name(expected))
            if fields is not None:
                types = list(fields.values())
            elif isinstance(expected, tuple):
                if self.contains_reference(expected):
                    extent = evaluate_literal_int_expression(expected[1])
                    if extent is None or extent != len(node.elements):
                        raise ResourceAggregateError(
                            "pointer-array-initialization-arity", node
                        )
                types = [expected[0]] * len(node.elements)
            else:
                types = [None] * len(node.elements)
            if len(types) != len(node.elements):
                raise ResourceAggregateError("aggregate-initialization-arity", node)
            return ArrayLiteralNode(
                [
                    self.expression(value, env, type_)
                    for value, type_ in zip(node.elements, types)
                ],
                source_location=location,
            )
        if isinstance(node, FunctionCallNode):
            if (
                _name(node.function) == "buffer_store"
                and "buffer_store" not in self.functions
                and len(node.arguments) == 3
                and isinstance(self.infer(node.arguments[0], env), _Pointer)
            ):
                return _call(
                    self.helper(self.infer(node.arguments[0], env), "store"),
                    [
                        *(self.expression(value, env) for value in node.arguments),
                        *self.resource_arguments(),
                    ],
                )
            callee = self.function_for_call(node)
            fields = self.fields.get(_name(node.function))
            types = (
                self.parameter_types[id(callee)]
                if callee is not None
                else list(fields.values()) if fields is not None else None
            )
            if types is not None and len(types) != len(node.arguments):
                raise ResourceAggregateError("aggregate-initialization-arity", node)
            if types is None and any(
                self.contains_reference(self.infer(arg, env))
                or self.contains_reference(arg)
                for arg in node.arguments
            ):
                raise ResourceAggregateError("resource-reference-escape", node)
            args = [
                self.expression(arg, env, types[i] if types is not None else None)
                for i, arg in enumerate(node.arguments)
            ]
            if callee is not None:
                args.extend(self.resource_arguments())
            return FunctionCallNode(
                node.function, args, node.generic_args or None, source_location=location
            )
        if isinstance(node, TernaryOpNode):
            if isinstance(actual, _Pointer):
                raise ResourceAggregateError("conditional-pointer-selection", node)
            return TernaryOpNode(
                self.expression(node.condition, env),
                self.expression(node.true_expr, env, expected),
                self.expression(node.false_expr, env, expected),
                source_location=location,
            )
        if isinstance(node, (CastNode, PointerReinterpretNode)) and (
            self.contains_reference(self.infer(node.expression, env))
            or self.contains_reference(self.source_type(node.target_type))
        ):
            raise ResourceAggregateError("resource-reference-cast", node)
        return self.generic(node, env)

    def generic(self, node, env):
        if isinstance(node, TypeNode) or not isinstance(node, ASTNode):
            return node
        result = copy(node)
        memo = {}

        def rewrite(value):
            if isinstance(value, ASTNode):
                if id(value) not in memo:
                    memo[id(value)] = self.expression(value, env)
                return memo[id(value)]
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            return value

        for field, value in vars(node).items():
            if field not in AST_CHILD_FIELD_EXCLUSIONS:
                setattr(result, field, rewrite(value))
        return result

    def statement(self, node, env):
        if isinstance(node, BlockNode):
            scoped = dict(env)
            return BlockNode(
                [self.statement(item, scoped) for item in node.statements],
                source_location=node.source_location,
            )
        if isinstance(node, VariableNode):
            if isinstance(node.var_type, ReferenceType):
                raise ResourceAggregateError("local-reference-alias", node)
            source_type = self.source_type(node.var_type, node)
            if _name(source_type) == "auto":
                source_type = self.infer(node.initial_value, env) or source_type
            if isinstance(source_type, _RootPointer):
                source_type = _Pointer(
                    source_type.element,
                    source_type.space,
                    source_type.writable,
                    source_type.readable,
                )
            value = self.expression(node.initial_value, env, source_type)
            env[node.name] = source_type
            result = copy(node)
            result.var_type = result.vtype = self.target_type(source_type)
            result.initial_value = value
            return result
        if isinstance(node, ReturnNode):
            return ReturnNode(
                self.expression(node.value, env, self.returns[id(self.current)]),
                source_location=node.source_location,
            )
        if isinstance(node, IfNode):
            result = IfNode(
                self.expression(node.condition, env),
                self.statement(node.then_branch, dict(env)),
                self.statement(node.else_branch, dict(env)),
                source_location=node.source_location,
            )
            result.else_if_conditions = [
                self.expression(value, env) for value in node.else_if_conditions
            ]
            result.else_if_bodies = [
                self.statement(value, dict(env)) for value in node.else_if_bodies
            ]
            return result
        if isinstance(node, ForNode):
            scoped = dict(env)
            return ForNode(
                self.statement(node.init, scoped),
                self.expression(node.condition, scoped),
                self.expression(node.update, scoped),
                self.statement(node.body, scoped),
                source_location=node.source_location,
            )
        return self.expression(node, env)

    def run(self):
        self.discover(self.entry)
        for struct in self.structs.values():
            for member in struct.members:
                member.member_type = self.target_type(
                    self.fields[struct.name][member.name]
                )
                if isinstance(self.fields[struct.name][member.name], (_Pointer, tuple)):
                    member.attributes = [
                        a
                        for a in member.attributes
                        if _name(a)
                        not in {"const", "device", "constant", "global", "storage"}
                    ]
        for overloads in self.functions.values():
            for function in overloads:
                if id(function) not in self.reachable:
                    continue
                self.current = function
                types = self.parameter_types[id(function)]
                env = {
                    param.name: type_
                    for param, type_ in zip(function.parameters, types)
                }
                if function is self.entry:
                    for index, (param, pointer, _root) in enumerate(self.resources):
                        env[param.name] = _RootPointer(
                            pointer.element,
                            pointer.space,
                            pointer.writable,
                            pointer.readable,
                            index,
                            self.entry_handles[param.name],
                        )
                function.body = self.statement(function.body, env)
                if function is not self.entry:
                    for param, type_ in zip(function.parameters, types):
                        if isinstance(param.param_type, ReferenceType):
                            reference = copy(param.param_type)
                            reference.referenced_type = self.target_type(type_)
                            param.param_type = reference
                        else:
                            param.param_type = self.target_type(type_)
                    function.parameters.extend(self.resource_parameters())
                    function.return_type = self.target_type(self.returns[id(function)])
        self.entry.body.statements[:0] = [
            VariableNode(name, deepcopy(param.param_type), _id(param.name))
            for param, _pointer, name in self.resources
        ] + [
            VariableNode(
                self.entry_handles[param.name],
                self.target_type(pointer),
                _call(self.helper(pointer, "make"), [_integer(index)]),
            )
            for index, (param, pointer, _name_) in enumerate(self.resources)
        ]
        self.ast.structs = [
            node for node in self.generated if isinstance(node, StructNode)
        ] + self.ast.structs
        self.ast.functions.extend(
            node for node in self.generated if isinstance(node, FunctionNode)
        )
        self.mark_value_helpers()
        return self.ast

    def mark_value_helpers(self):
        """Prove helpers that mutate only unaliased, private scalar values."""
        functions = {
            function.name: function
            for function in self.ast.walk()
            if isinstance(function, FunctionNode)
        }
        globals_ = {node.name for node in self.ast.global_variables}
        scalar_names = {
            "bool",
            "int",
            "uint",
            "int64",
            "uint64",
            "int64_t",
            "uint64_t",
            "float",
            "double",
            "half",
            "float16_t",
        }
        candidates = {}
        for name, function in functions.items():
            if function.body is None:
                continue
            if getattr(function, "resource_aggregate_nonmutating", False):
                continue
            parameters = {p.name: p.param_type for p in function.parameters}
            locals_ = [
                node for node in function.body.walk() if isinstance(node, VariableNode)
            ]
            names = [node.name for node in locals_]
            if len(names) != len(set(names)) or set(names) & (
                globals_ | parameters.keys()
            ):
                continue
            private_scalars = {
                p.name
                for p in function.parameters
                if _name(p.param_type) in scalar_names
                and set(p.qualifiers) <= {"in", "const", "thread"}
                and not p.attributes
            }
            private_scalars.update(
                node.name
                for node in locals_
                if _name(node.var_type) in scalar_names
                and set(node.qualifiers) <= {"const", "thread"}
                and not node.attributes
            )
            dependencies = set()
            valid = True
            for node in function.body.walk():
                target = (
                    node.target
                    if isinstance(node, AssignmentNode)
                    else (
                        node.operand
                        if isinstance(node, UnaryOpNode)
                        and node.operator in {"++", "--"}
                        else None
                    )
                )
                if target is not None and not (
                    isinstance(target, IdentifierNode)
                    and target.name in private_scalars
                ):
                    valid = False
                    break
                if isinstance(node, UnaryOpNode) and node.operator == "&":
                    valid = False
                    break
                if isinstance(node, FunctionCallNode):
                    callee = _name(node.function)
                    if callee in functions:
                        dependencies.add(callee)
                    elif callee not in scalar_names:
                        valid = False
                        break
                if type(node).__name__ in {
                    "AtomicOpNode",
                    "SyncNode",
                    "WaveOpNode",
                    "TextureOpNode",
                    "BufferOpNode",
                    "PointerReinterpretNode",
                    "LambdaNode",
                }:
                    valid = False
                    break
            if valid:
                candidates[name] = dependencies
        while candidates:
            ready = [
                name
                for name, deps in candidates.items()
                if all(
                    getattr(functions[dep], "resource_aggregate_nonmutating", False)
                    for dep in deps
                )
            ]
            if not ready:
                break
            for name in ready:
                functions[name].resource_aggregate_nonmutating = True
                del candidates[name]


def lower_resource_aggregates(ast):
    """Lower a selected compute entry's private resource-reference aggregates."""
    if not isinstance(ast, ASTNode):
        return ast
    if not any(
        isinstance(node, StructNode) and _contains_storage_pointer(node)
        for node in ast.walk()
    ):
        return ast
    stages = list(getattr(ast, "stages", {}).values())
    if len(stages) != 1 or _name(stages[0].stage).lower() != "compute":
        return ast
    result = deepcopy(ast)
    entry = list(result.stages.values())[0].entry_point
    return _Lowering(result, entry).run()
