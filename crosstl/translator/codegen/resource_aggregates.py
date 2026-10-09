"""Lower private aggregates containing storage pointers to resource handles.

Handles are private values, never a buffer ABI. Their identity is drawn from the
selected entry's bindings and fixed workgroup allocations; access helpers branch
over concrete storage instead of constructing target resource-object arrays.
The source AST is retained.
"""

from copy import copy, deepcopy
from dataclasses import dataclass

from ..arithmetic_conversions import (
    ArithmeticScalarKind,
    ArithmeticType,
    UnrepresentableArithmeticConversion,
    arithmetic_type_name,
    resolve_arithmetic_conversion,
    source_integer_shape,
    target_arithmetic_type,
)
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
    CooperativeMatrixType,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MatrixType,
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
    SwitchNode,
    TernaryOpNode,
    TypeNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
    WhileNode,
)
from .array_utils import evaluate_literal_int_expression
from .resource_origins import ResourceOriginAnalysis
from .workgroup_access_contracts import parse_workgroup_access_assertions


class ResourceAggregateError(ValueError):
    """An aggregate resource reference cannot be represented without guessing."""

    project_diagnostic_code = "project.translate.resource-aggregate-unsupported"
    missing_capabilities = ("resource-aggregate-lowering",)

    def __init__(self, reason, node=None, *, detail=None):
        message = f"Cannot preserve aggregate resource references: {reason}"
        super().__init__(f"{message}: {detail}" if detail else message)
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


def _numeric(value):
    if isinstance(value, VectorType):
        scalar = _numeric(value.element_type)
        if scalar is not None and scalar.lanes == 1:
            return ArithmeticType(
                scalar.source_type,
                scalar.target_type,
                scalar.kind,
                scalar.bits,
                value.size,
            )
        return None
    name = _name(value)
    numeric = target_arithmetic_type(name)
    integer = source_integer_shape(name or "")
    if integer is not None:
        kind, bits, lanes = integer
        return ArithmeticType(
            name, name, kind, getattr(value, "size_bits", None) or bits, lanes
        )
    if numeric is not None and getattr(value, "size_bits", None) is not None:
        return ArithmeticType(name, name, numeric.kind, value.size_bits, numeric.lanes)
    return numeric


def _numeric_node(numeric, lanes=1):
    narrow = numeric.kind != ArithmeticScalarKind.BOOLEAN and numeric.bits < 32
    component = PrimitiveType(
        arithmetic_type_name(numeric.kind, 32 if narrow else numeric.bits),
        size_bits=numeric.bits if narrow else None,
    )
    return component if lanes == 1 else VectorType(component, lanes)


def _type_key(value):
    if isinstance(value, _Pointer):
        return ("pointer", value.element, value.space, value.writable, value.readable)
    if isinstance(value, tuple):
        size = evaluate_literal_int_expression(value[1])
        element = _type_key(value[0])
        return (
            ("array", element, size)
            if size is not None and element is not None
            else None
        )
    if isinstance(value, (MatrixType, CooperativeMatrixType)):
        rows = evaluate_literal_int_expression(value.rows)
        cols = evaluate_literal_int_expression(value.cols)
        element = _type_key(value.element_type)
        if rows is None or cols is None or element is None:
            return None
        key = ("matrix", element, rows, cols)
        if isinstance(value, CooperativeMatrixType):
            key = (
                "cooperative",
                *key,
                value.scope,
                value.use,
                value.layout,
                value.fragment_layout,
                value.subgroup_size,
                value.elements_per_lane,
                value.fragment_provenance,
                value.fragment_mapping,
                value.fragment_mapping_provenance,
            )
        return key
    numeric = _numeric(value)
    if numeric is not None:
        return ("numeric", numeric.kind, numeric.bits, numeric.lanes)
    if _name(value) and not getattr(value, "generic_args", None):
        return ("named", _name(value))
    return None


def _conversion_rank(expected, actual):
    if _type_key(expected) is not None and _type_key(expected) == _type_key(actual):
        return 0
    if isinstance(expected, _Pointer) and isinstance(actual, _Pointer):
        if (
            expected.element == actual.element
            and expected.space == actual.space
            and expected.readable == actual.readable
            and not expected.writable
            and actual.writable
        ):
            return 1
        return None
    left, right = _numeric(expected), _numeric(actual)
    if left is None or right is None or left.lanes != 1 or right.lanes != 1:
        return None
    if any(
        type_.kind == ArithmeticScalarKind.FLOATING and type_.bits not in {32, 64}
        for type_ in (left, right)
    ):
        return None
    if (
        left.kind == ArithmeticScalarKind.SIGNED_INTEGER
        and left.bits == 32
        and right.is_integer
        and right.bits < 32
    ):
        return 1
    if (
        left.kind == right.kind == ArithmeticScalarKind.FLOATING
        and left.bits == 64
        and right.bits == 32
    ):
        return 1
    return 2


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
        if isinstance(value, VectorType):
            component = target_arithmetic_type(_name(value.element_type))
            if (
                component is None
                or component.lanes != 1
                or component.bits != 32
                or getattr(value.element_type, "size_bits", None) not in {None, 32}
                or value.size not in {2, 4}
            ):
                raise ResourceAggregateError("unsupported-vector-pointee", owner)
            return arithmetic_type_name(component.kind, component.bits, value.size)
        return {"int64": "int64_t", "uint64": "uint64_t"}.get(
            _name(value), _name(value)
        )

    if (
        isinstance(value, NamedType)
        and value.name in {
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
        (
            q
            for q in (
                "constant",
                "device",
                "global",
                "storage",
                "threadgroup",
                "workgroup",
            )
            if q in qualifiers
        ),
        None,
    )
    if space not in {
        "constant",
        "device",
        "global",
        "storage",
        "threadgroup",
        "workgroup",
    }:
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
        element,
        (
            "threadgroup"
            if space in {"threadgroup", "workgroup"}
            else "constant" if space == "constant" else "device"
        ),
        writable,
        readable,
    )


def _contains_storage_pointer(node):
    return any(
        isinstance(child, PointerType)
        and child.address_space in {
            "constant",
            "device",
            "global",
            "storage",
            "threadgroup",
            "workgroup",
        }
        for child in node.walk()
    )


class _Lowering:
    def __init__(
        self,
        ast,
        entry,
        storage_pointer_parameters=False,
        workgroup_access_assertions=(),
    ):
        self.ast = ast
        self.entry = entry
        self.storage_pointer_parameters = storage_pointer_parameters
        self.workgroup_access_assertions = parse_workgroup_access_assertions(
            workgroup_access_assertions
        )
        self.reserved = {
            node.name
            for node in ast.walk()
            if isinstance(getattr(node, "name", None), str)
        }
        self.workgroup_roots = {}
        self.workgroup_declarations = []
        self.resource_origins = {}
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
                if pointer.space == "threadgroup":
                    raise ResourceAggregateError("dynamic-workgroup-binding", param)
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
        self.bound_calls = {}
        self.overload_names = {}
        self.global_types = {
            variable.name: self.source_type(variable.var_type, variable)
            for variable in ast.global_variables
            if isinstance(variable, VariableNode) and not variable.is_type_alias
        }
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
        if id(owner) in self.workgroup_roots:
            return self.workgroup_roots[id(owner)]
        pointer = _pointer_type(value, owner)
        if pointer is not None:
            return pointer
        if isinstance(value, ArrayType):
            return (self.source_type(value.element_type, owner), value.size)
        if isinstance(value, ReferenceType):
            return self.source_type(value.referenced_type, owner)
        return value

    def collect_workgroup_roots(self, function):
        # Shared arrays have workgroup lifetime even when their source declaration
        # is nested. Hoist each physical allocation once; bind source names only
        # at their lexical declaration, including shadowed names.
        for node in function.body.walk():
            if not isinstance(node, VariableNode) or not isinstance(
                node.var_type, ArrayType
            ):
                continue
            qualifiers = set(node.qualifiers)
            qualifiers.update(_name(a) for a in node.attributes)
            if not qualifiers & {"threadgroup", "workgroup", "shared", "groupshared"}:
                continue
            if id(node) in self.workgroup_roots:
                continue
            extent = evaluate_literal_int_expression(node.var_type.size)
            element = node.var_type.element_type
            # The qualifier belongs to the pointee, not the private array of
            # pointer values (for example ``threadgroup float* choices[2]``).
            if isinstance(element, PointerType):
                continue
            numeric = _numeric(element)
            if (
                extent is None
                or extent <= 0
                or numeric is None
                or numeric.bits != 32
                or numeric.lanes != 1
                or numeric.kind == ArithmeticScalarKind.BOOLEAN
            ):
                raise ResourceAggregateError("workgroup-allocation-layout", node)
            if node.initial_value is not None or qualifiers - {
                "threadgroup",
                "workgroup",
                "shared",
                "groupshared",
            }:
                raise ResourceAggregateError(
                    "workgroup-allocation-initializer-or-qualifier", node
                )
            pointer = _Pointer(_name(element), "threadgroup", True, True)
            root = self.fresh(f"crosstl_workgroup_{node.name}")
            handle = self.fresh(f"crosstl_reference_{node.name}")
            identity = len(self.resources)
            declaration = copy(node)
            declaration.name = root
            declaration.var_type = declaration.vtype = ArrayType(
                deepcopy(element), _integer(extent)
            )
            self.workgroup_declarations.append(declaration)
            parameter = ParameterNode(
                root, PointerType(deepcopy(element), address_space="threadgroup")
            )
            self.resources.append((parameter, pointer, root))
            self.workgroup_roots[id(node)] = _RootPointer(
                pointer.element, pointer.space, True, True, identity, handle
            )

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
            ParameterNode(
                name, **self.resource_declaration(param, pointer, "param_type")
            )
            for param, pointer, name in self.resources
            if pointer.space != "threadgroup"
        ]

    def resource_declaration(self, param, pointer, field):
        if not self.storage_pointer_parameters:
            return {field: deepcopy(param.param_type)}
        # GLSL specializes these references against concrete SSBO bindings;
        # unsized buffer arrays cannot be ordinary function parameters.
        access = (
            "read_write"
            if pointer.readable and pointer.writable
            else ("read" if pointer.readable else "write")
        )
        return {
            field: PointerType(
                PrimitiveType(pointer.element),
                is_mutable=pointer.writable,
                address_space=pointer.space,
                access_mode=access,
            ),
            "qualifiers": [
                pointer.space,
                *([] if pointer.writable else ["const"]),
                *([] if pointer.readable else ["writeonly"]),
            ],
        }

    def resource_arguments(self):
        return [
            _id(name)
            for _param, pointer, name in self.resources
            if pointer.space != "threadgroup"
        ]

    def function_for_call(self, node, env=None):
        if id(node) in self.bound_calls:
            return self.bound_calls[id(node)]
        candidates = self.functions.get(_name(node.function), [])
        candidates = [
            f
            for f in candidates
            if len(self.parameter_types[id(f)]) == len(node.arguments)
        ]
        if len(candidates) > 1:
            actual = [self.infer(arg, env or {}) for arg in node.arguments]
            if any(type_ is None for type_ in actual) or node.generic_args:
                raise ResourceAggregateError("overloaded-resource-call", node)
            ranked = []
            for function in candidates:
                ranks = tuple(
                    _conversion_rank(expected, type_)
                    for expected, type_ in zip(
                        self.parameter_types[id(function)], actual
                    )
                )
                if None not in ranks:
                    # Reference binding needs address, constness and value-category
                    # evidence in addition to the value type. Do not guess it.
                    if any(
                        isinstance(p.param_type, ReferenceType)
                        for p in function.parameters
                    ):
                        raise ResourceAggregateError("overloaded-reference-call", node)
                    if any(
                        rank and set(p.qualifiers) & {"out", "inout"}
                        for rank, p in zip(ranks, function.parameters)
                    ):
                        continue
                    ranked.append((function, ranks))
            winners = [
                function
                for function, ranks in ranked
                if all(
                    other is function
                    or (
                        all(a <= b for a, b in zip(ranks, other_ranks))
                        and any(a < b for a, b in zip(ranks, other_ranks))
                    )
                    for other, other_ranks in ranked
                )
            ]
            if len(winners) != 1:
                raise ResourceAggregateError("overloaded-resource-call", node)
            candidates = winners
        callee = candidates[0] if candidates else None
        self.bound_calls[id(node)] = callee
        return callee

    def bind_calls(self, node, env):
        if isinstance(node, BlockNode):
            scoped = dict(env)
            for item in node.statements:
                self.bind_calls(item, scoped)
        elif isinstance(node, VariableNode):
            self.bind_calls(node.initial_value, env)
            type_ = self.source_type(node.var_type, node)
            env[node.name] = (
                self.infer(node.initial_value, env) if _name(type_) == "auto" else type_
            )
        elif isinstance(node, ForNode):
            scoped = dict(env)
            for item in (node.init, node.condition, node.update, node.body):
                self.bind_calls(item, scoped)
        elif isinstance(node, SwitchNode):
            self.bind_calls(node.expression, env)
            scoped = dict(env)
            for case in node.cases:
                self.bind_calls(case.value, scoped)
                for item in case.statements:
                    self.bind_calls(item, scoped)
            self.bind_calls(node.default_case, scoped)
        elif isinstance(node, FunctionCallNode):
            for argument in node.arguments:
                self.bind_calls(argument, env)
            callee = self.function_for_call(node, env)
            if callee is self.entry:
                raise ResourceAggregateError("recursive-entry", node)
            if callee is not None:
                self.discover(callee)
        elif isinstance(node, ASTNode) and not isinstance(node, TypeNode):
            for child in node.child_nodes():
                self.bind_calls(child, dict(env))

    def discover(self, function):
        if id(function) in self.reachable:
            return
        self.reachable.add(id(function))
        if isinstance(function.return_type, ReferenceType):
            raise ResourceAggregateError("reference-return", function)
        if function.body is None:
            raise ResourceAggregateError("external-resource-call", function)
        self.collect_workgroup_roots(function)
        env = dict(self.global_types)
        env.update(
            zip(
                (p.name for p in function.parameters),
                self.parameter_types[id(function)],
            )
        )
        self.bind_calls(function.body, env)

    def elide_unused_null_parameters(self):
        """Remove proven unused helper formals, never manufacture null handles."""
        functions = {
            id(f): f
            for overloads in self.functions.values()
            for f in overloads
            if id(f) in self.reachable and f is not self.entry and len(overloads) == 1
        }
        candidates = {
            (key, index): param
            for key, function in functions.items()
            for index, param in enumerate(function.parameters)
            if isinstance(self.parameter_types[key][index], _Pointer)
            and not isinstance(param.param_type, ReferenceType)
        }
        calls = [n for n in self.ast.walk() if isinstance(n, FunctionCallNode)]
        owners = {
            id(node): function
            for overloads in self.functions.values()
            for function in overloads
            if function.body is not None
            for node in function.body.walk()
            if isinstance(node, FunctionCallNode)
        }
        targets = {}
        seeds = set()
        unresolved = set()
        for call in calls:
            overloads = self.functions.get(_name(call.function), [])
            if len(overloads) != 1 or id(overloads[0]) not in functions:
                continue
            callee = overloads[0]
            if call.generic_args or len(call.arguments) != len(callee.parameters):
                unresolved.add(id(callee))
                continue
            targets[id(call)] = id(callee)
            for index, argument in enumerate(call.arguments):
                if isinstance(argument, IdentifierNode) and argument.name == "nullptr":
                    seeds.add((id(callee), index))
        call_names = {id(call.function) for call in calls}
        escaped_names = {
            node.name
            for node in self.ast.walk()
            if isinstance(node, IdentifierNode) and id(node) not in call_names
        }
        unresolved.update(
            key for key, function in functions.items() if function.name in escaped_names
        )
        candidates = {
            key: param for key, param in candidates.items() if key[0] not in unresolved
        }
        seeds.intersection_update(candidates)
        if not seeds:
            return

        # Materialization may retain a literal branch around an unused resource.
        # Fold only literal conditions, with the selected branch's scope intact.
        memo = {}

        def fold(value):
            if isinstance(value, list):
                return [fold(child) for child in value]
            if not isinstance(value, ASTNode):
                return value
            if id(value) in memo:
                return memo[id(value)]
            if isinstance(value, IfNode) and isinstance(value.condition, LiteralNode):
                condition = evaluate_literal_int_expression(value.condition)
                if condition is not None and (
                    condition or not value.else_if_conditions
                ):
                    selected = value.then_branch if condition else value.else_branch
                    result = fold(selected) or BlockNode([])
                    if not isinstance(result, BlockNode):
                        result = BlockNode([result])
                    memo[id(value)] = result
                    return result
            memo[id(value)] = value
            for field, child in vars(value).items():
                if field not in AST_CHILD_FIELD_EXCLUSIONS:
                    setattr(value, field, fold(child))
            return value

        for function in functions.values():
            function.body = fold(function.body)

        def can_omit_argument(call, index):
            argument = call.arguments[index]
            if not isinstance(argument, IdentifierNode):
                return False
            if argument.name == "nullptr":
                return True
            owner = owners.get(id(call))
            if owner is None or any(
                isinstance(node, VariableNode) and node.name == argument.name
                for node in owner.body.walk()
            ):
                return False
            actual = next(
                (
                    type_
                    for param, type_ in zip(
                        owner.parameters, self.parameter_types[id(owner)]
                    )
                    if param.name == argument.name
                ),
                None,
            )
            try:
                self.compatible(
                    self.parameter_types[targets[id(call)]][index], actual, argument
                )
            except ResourceAggregateError:
                return False
            return True

        dependencies = {}
        for key, parameter in candidates.items():
            function = functions[key[0]]
            # Without declaration binding, a lexical shadow is not a proof.
            if any(
                isinstance(node, VariableNode) and node.name == parameter.name
                for node in function.body.walk()
            ):
                continue
            if any(
                targets.get(id(call)) == key[0] and not can_omit_argument(call, key[1])
                for call in calls
            ):
                continue
            forwarded = set()

            def used(value):
                if isinstance(value, str):
                    return value == parameter.name
                if isinstance(value, (list, tuple)):
                    return any(used(child) for child in value)
                if isinstance(value, dict):
                    return any(used(child) for child in value.values())
                if not isinstance(value, ASTNode):
                    return False
                if isinstance(value, FunctionCallNode):
                    callee = targets.get(id(value))
                    if used(value.function) or used(value.generic_args):
                        return True
                    for index, argument in enumerate(value.arguments):
                        if (
                            isinstance(argument, IdentifierNode)
                            and argument.name == parameter.name
                            and (callee, index) in candidates
                        ):
                            forwarded.add((callee, index))
                        elif used(argument):
                            return True
                    return False
                return any(
                    used(child)
                    for field, child in vars(value).items()
                    if field not in AST_CHILD_FIELD_EXCLUSIONS
                )

            if not used(function.body):
                dependencies[key] = forwarded

        # Only acyclic forwarding chains terminating in unused formals qualify.
        proven = set()
        while True:
            ready = {
                key for key, deps in dependencies.items() if deps <= proven
            } - proven
            if not ready:
                break
            proven.update(ready)
        selected = set()
        pending = list(seeds & proven)
        while pending:
            key = pending.pop()
            if key not in selected:
                selected.add(key)
                pending.extend(dependencies[key])
        for call in calls:
            callee = targets.get(id(call))
            if callee is not None:
                call.arguments = call.args = [
                    arg
                    for index, arg in enumerate(call.arguments)
                    if (callee, index) not in selected
                ]
        for key, function in functions.items():
            function.parameters = [
                param
                for index, param in enumerate(function.parameters)
                if (key, index) not in selected
            ]
            self.parameter_types[key] = [
                type_
                for index, type_ in enumerate(self.parameter_types[key])
                if (key, index) not in selected
            ]

    def infer(self, node, env):
        if isinstance(node, LiteralNode):
            return node.literal_type
        if isinstance(node, CastNode):
            return self.source_type(node.target_type)
        if isinstance(node, IdentifierNode):
            return env.get(node.name)
        if isinstance(node, MemberAccessNode):
            owner = self.infer(node.object_expr, env)
            numeric = _numeric(owner)
            if (
                numeric is not None
                and numeric.lanes > 1
                and any(
                    node.member
                    and all(c in alphabet[: numeric.lanes] for c in node.member)
                    for alphabet in ("xyzw", "rgba", "stpq")
                )
                and 1 <= len(node.member) <= 4
            ):
                return _numeric_node(numeric, len(node.member))
            return self.fields.get(_name(owner), {}).get(node.member)
        if isinstance(node, ArrayAccessNode):
            owner = self.infer(node.array_expr, env)
            if isinstance(owner, tuple):
                return owner[0]
            if isinstance(owner, _Pointer):
                return PrimitiveType(owner.element)
            numeric = _numeric(owner)
            if numeric is not None and numeric.lanes > 1:
                return _numeric_node(numeric)
        if isinstance(node, FunctionCallNode):
            if _name(node.function) in self.structs:
                return NamedType(_name(node.function))
            if _numeric(_name(node.function)) is not None:
                return PrimitiveType(_name(node.function))
            callee = self.function_for_call(node, env)
            if callee is not None:
                return self.returns[id(callee)]
        if isinstance(node, BinaryOpNode):
            left = self.infer(node.left, env)
            right = self.infer(node.right, env)
            if node.operator in {"+", "-"} and isinstance(left, _Pointer):
                if isinstance(right, _Pointer):
                    return PrimitiveType("int64_t")
                return left
            if node.operator == "+" and isinstance(right, _Pointer):
                return right
            return self.arithmetic_result(left, right, node.operator)
        if isinstance(node, UnaryOpNode):
            if node.operator == "&" and isinstance(node.operand, ArrayAccessNode):
                return self.infer(node.operand.array_expr, env)
            if node.operator in {"++", "--"}:
                return self.infer(node.operand, env)
            if node.operator == "*":
                pointer = self.infer(node.operand, env)
                if isinstance(pointer, _Pointer):
                    return PrimitiveType(pointer.element)
            if node.operator in {"+", "-", "~"}:
                value = self.infer(node.operand, env)
                return self.arithmetic_result(value, PrimitiveType("int"), "+")
            if node.operator == "!":
                numeric = _numeric(self.infer(node.operand, env))
                if numeric is not None and numeric.lanes == 1:
                    return PrimitiveType("bool")
        if isinstance(node, TernaryOpNode):
            left, right = self.infer(node.true_expr, env), self.infer(
                node.false_expr, env
            )
            if isinstance(left, _Pointer) and left == right:
                return left
            if _type_key(left) is not None and _type_key(left) == _type_key(right):
                return left
            return self.arithmetic_result(left, right, "?:")
        return None

    def arithmetic_result(self, left, right, operator):
        left, right = _numeric(left), _numeric(right)
        if left is None or right is None:
            return None
        if any(
            type_.kind == ArithmeticScalarKind.FLOATING and type_.bits not in {32, 64}
            for type_ in (left, right)
        ):
            return None
        if operator in {"&&", "||"} and left.lanes == right.lanes == 1:
            return PrimitiveType("bool")
        try:
            conversion = resolve_arithmetic_conversion(
                left,
                right,
                operator,
                supported_integer_widths=frozenset({8, 16, 32, 64}),
                supported_floating_widths=frozenset({16, 32, 64}),
            )
        except UnrepresentableArithmeticConversion:
            return None
        return PrimitiveType(conversion.result_type) if conversion is not None else None

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

    def helper(self, pointer, operation, origins=None):
        pointer = _Pointer(
            pointer.element, pointer.space, pointer.writable, pointer.readable
        )
        if pointer.space != "threadgroup" or operation not in {"load", "store"}:
            origins = None
        key = pointer, operation, origins
        if key in self.helpers:
            return self.helpers[key]
        storage = "workgroup" if pointer.space == "threadgroup" else "resource"
        name = self.fresh(f"crosstl_{storage}_{operation}_{pointer.element}")
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
                and resource.space == pointer.space
                and (operation != "store" or resource.writable)
                and (operation != "load" or resource.readable)
                and (origins is None or index in origins)
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

    def atomic_call(self, node, env):
        operation = _name(node.function)
        arity = {
            "atomicLoad": 1,
            "atomicCompareExchangeWeak": 3,
            "atomicStore": 2,
            "atomicAdd": 2,
            "atomicMin": 2,
            "atomicMax": 2,
            "atomicAnd": 2,
            "atomicOr": 2,
            "atomicXor": 2,
            "atomicExchange": 2,
            "atomicCompSwap": 3,
            "atomicCompareExchange": 3,
        }.get(operation)
        if arity is None or operation in self.functions or not node.arguments:
            return None
        target = node.arguments[0]
        members = []
        while isinstance(target, MemberAccessNode):
            members.insert(0, target.member)
            target = target.object_expr
        access = self.access(target, env)
        if access is None:
            if any(self.access(child, env) is not None for child in target.walk()):
                raise ResourceAggregateError("atomic-resource-member-path", node)
            return None
        if len(node.arguments) != arity:
            raise ResourceAggregateError("atomic-resource-argument-count", node)
        source_pointer, owner, index = access
        pointer = _Pointer(
            source_pointer.element,
            source_pointer.space,
            source_pointer.writable,
            source_pointer.readable,
        )
        store = operation == "atomicStore"
        compare_expected = operation == "atomicCompareExchangeWeak"
        if not pointer.writable or (not store and not pointer.readable):
            raise ResourceAggregateError("atomic-resource-access", node)
        element = PrimitiveType(pointer.element)
        for member in members:
            element = self.fields.get(_name(element), {}).get(member)
        if _name(element) not in {"int", "uint"} and not (
            _name(element) == "float"
            and operation in {
                "atomicLoad",
                "atomicStore",
                "atomicAdd",
                "atomicExchange",
                "atomicCompareExchangeWeak",
            }
        ):
            raise ResourceAggregateError("atomic-resource-element", node)
        key = pointer, operation, tuple(members)
        name = self.helpers.get(key)
        if name is None:
            name = self.fresh(f"crosstl_resource_{operation}_{pointer.element}")
            self.helpers[key] = name
            params = [
                ParameterNode("reference", self.target_type(pointer)),
                ParameterNode("index", PrimitiveType("int64_t")),
                *(
                    ParameterNode(
                        f"value{i}",
                        deepcopy(element),
                        qualifiers=["inout"] if compare_expected and i == 0 else [],
                    )
                    for i in range(arity - 1)
                ),
                *self.resource_parameters(),
            ]
            candidates = [
                (identity, resource_name)
                for identity, (_param, resource, resource_name) in enumerate(
                    self.resources
                )
                if resource.element == pointer.element
                and resource.space == pointer.space
                and resource.writable
                and (store or resource.readable)
            ]
            if not candidates:
                raise ResourceAggregateError("missing-compatible-resource", node)
            body = []
            for identity, resource_name in candidates:
                destination = ArrayAccessNode(
                    _id(resource_name),
                    BinaryOpNode(
                        _member(_id("reference"), "offset"), "+", _id("index")
                    ),
                )
                for member in members:
                    destination = _member(destination, member)
                call = _call(
                    operation,
                    [destination, *(_id(f"value{i}") for i in range(arity - 1))],
                )
                statements = [call, ReturnNode()] if store else [ReturnNode(call)]
                if identity == candidates[-1][0]:
                    body.extend(statements)
                else:
                    body.append(
                        IfNode(
                            BinaryOpNode(
                                _member(_id("reference"), "identity"),
                                "==",
                                _integer(identity),
                            ),
                            BlockNode(statements),
                        )
                    )
            # Branch on the handle, then perform the atomic on the actual buffer.
            # A value-returning load helper would destroy the storage identity.
            self.generated.append(
                FunctionNode(
                    name,
                    (
                        PrimitiveType("void")
                        if store
                        else (
                            PrimitiveType("bool")
                            if compare_expected
                            else deepcopy(element)
                        )
                    ),
                    params,
                    BlockNode(body),
                )
            )
        return FunctionCallNode(
            _id(name),
            [
                self.expression(owner, env),
                _call("int64_t", [self.expression(index, env)]),
                *(self.expression(value, env) for value in node.arguments[1:]),
                *self.resource_arguments(),
            ],
            source_location=node.source_location,
        )

    def record_copy(self, node, env):
        if not isinstance(node, AssignmentNode):
            return None
        views = []
        for operand in (node.target, node.value):
            if not (
                isinstance(operand, UnaryOpNode)
                and operand.operator == "*"
                and not operand.is_postfix
                and isinstance(operand.operand, PointerReinterpretNode)
            ):
                return None
            views.append(operand.operand)
        destination, source = views
        pointers = [self.infer(view.expression, env) for view in views]
        if not all(isinstance(pointer, _Pointer) for pointer in pointers):
            return None
        dst, src = pointers
        if dst.space != "threadgroup" or src.space not in {"device", "constant"}:
            return None
        if node.operator != "=":
            raise ResourceAggregateError("record-copy-assignment-operator", node)
        if (
            not dst.writable
            or not src.readable
            or not destination.target_type.is_mutable
            or any(view.target_type.resource_qualifiers for view in views)
        ):
            raise ResourceAggregateError("record-copy-access-contract", node)
        view_pointers = [_pointer_type(view.target_type) for view in views]
        if (
            any(pointer is None for pointer in view_pointers)
            or view_pointers[0].space != dst.space
            or view_pointers[1].space != src.space
        ):
            raise ResourceAggregateError("record-copy-address-space", node)
        if not view_pointers[0].writable or not view_pointers[1].readable:
            raise ResourceAggregateError("record-copy-access-contract", node)
        record_name = _name(destination.target_type.pointee_type)
        if record_name != _name(source.target_type.pointee_type):
            raise ResourceAggregateError("record-copy-type-mismatch", node)
        record = self.structs.get(record_name)
        if (
            record is None
            or len(record.members) != 1
            or record.generic_params
            or record.inheritance
        ):
            raise ResourceAggregateError("record-copy-layout", node)
        member = record.members[0]
        array = member.member_type
        numeric = _numeric(array.element_type) if isinstance(array, ArrayType) else None
        extent = (
            evaluate_literal_int_expression(array.size)
            if isinstance(array, ArrayType)
            else None
        )
        alignment = 1
        attributes = list(record.attributes or ())
        if attributes:
            if (
                len(attributes) != 1
                or attributes[0].name != "metal_alignas"
                or len(attributes[0].arguments) != 1
            ):
                raise ResourceAggregateError("record-copy-layout", node)
            alignment = evaluate_literal_int_expression(attributes[0].arguments[0])
        if (
            numeric is None
            or not numeric.is_integer
            or numeric.bits != 8
            or numeric.lanes != 1
            or member.attributes
            or getattr(member, "resource_qualifiers", None)
            or extent is None
            or extent <= 0
            or alignment is None
            or alignment <= 0
            or alignment & (alignment - 1)
            or extent % alignment
        ):
            raise ResourceAggregateError("record-copy-layout", node)
        backing = _numeric(src.element)
        if (
            src.element != dst.element
            or backing is None
            or backing.bits != 32
            or backing.lanes != 1
            or backing.kind == ArithmeticScalarKind.BOOLEAN
            or alignment > 4
            or extent % 4
        ):
            raise ResourceAggregateError("record-copy-backing-layout", node)
        if any(
            isinstance(child, (FunctionCallNode, AssignmentNode))
            or isinstance(child, UnaryOpNode)
            and child.operator in {"++", "--"}
            for view in views
            for child in view.expression.walk()
        ):
            raise ResourceAggregateError("record-copy-address-side-effects", node)

        def path(expression):
            if isinstance(expression, IdentifierNode):
                return expression.name
            if isinstance(expression, MemberAccessNode):
                owner = path(expression.object_expr)
                return f"{owner}.{expression.member}" if owner else None
            if isinstance(expression, UnaryOpNode) and expression.operator == "&":
                return path(expression.operand)
            if isinstance(expression, ArrayAccessNode):
                return path(expression.array_expr)
            if isinstance(expression, BinaryOpNode) and expression.operator in {
                "+",
                "-",
            }:
                return path(expression.left)
            return None

        selector = path(destination.expression)
        matches = [
            assertion
            for assertion in self.workgroup_access_assertions
            if assertion.applies_to(self.entry.name, self.current.name, selector)
        ]
        if not matches:
            raise ResourceAggregateError(
                "record-copy-destination-range-unproven",
                node,
                detail=f"entry '{self.entry.name}', function '{self.current.name}', pointer '{selector}' requires an absolute range covering every copied word",
            )
        minimum = max(assertion.minimum for assertion in matches)
        maximum = min(assertion.maximum for assertion in matches)
        words = extent // 4
        if minimum > maximum or maximum - minimum + 1 < words:
            raise ResourceAggregateError(
                "record-copy-conflicting-access-assertions", node
            )
        destinations = [
            (index, root)
            for index, (_param, pointer, root) in enumerate(self.resources)
            if pointer.space == dst.space
            and pointer.element == dst.element
            and pointer.writable
            and (
                id(destination.expression) not in self.resource_origins
                or index in self.resource_origins[id(destination.expression)]
            )
        ]
        sources = [
            (index, root)
            for index, (_param, pointer, root) in enumerate(self.resources)
            if pointer.space == src.space
            and pointer.element == src.element
            and pointer.readable
            and (
                id(source.expression) not in self.resource_origins
                or index in self.resource_origins[id(source.expression)]
            )
        ]
        extents = {
            declaration.name: evaluate_literal_int_expression(declaration.var_type.size)
            for declaration in self.workgroup_declarations
        }
        if not destinations or not sources:
            raise ResourceAggregateError("missing-compatible-resource", node)
        if (
            minimum < 0
            or maximum > 2147483647
            or any(maximum >= extents[root] for _, root in destinations)
        ):
            raise ResourceAggregateError("record-copy-destination-out-of-bounds", node)
        # Assertions cover every accessed element, including the expanded tail.
        # They therefore prove the destination offset fits a signed target index.
        # Source offsets remain wide and retain the target's ordinary index checks.
        key = (
            "record-copy",
            src,
            dst,
            words,
            minimum,
            maximum,
            tuple(destinations),
            tuple(sources),
        )
        if key not in self.helpers:
            name = self.fresh(f"crosstl_workgroup_copy_{src.element}")
            self.helpers[key] = name
            body = [
                VariableNode(
                    "destination_index",
                    PrimitiveType("int"),
                    _call("int", [_member(_id("destination"), "offset")]),
                )
            ]
            for destination_index, destination_root in destinations:
                for source_index, source_root in sources:
                    statements = [
                        AssignmentNode(
                            ArrayAccessNode(
                                _id(destination_root),
                                BinaryOpNode(
                                    _id("destination_index"), "+", _integer(word)
                                ),
                            ),
                            ArrayAccessNode(
                                _id(source_root),
                                BinaryOpNode(
                                    _member(_id("source"), "offset"),
                                    "+",
                                    _integer(word),
                                ),
                            ),
                        )
                        for word in range(words)
                    ]
                    statements.append(ReturnNode())
                    body.append(
                        IfNode(
                            BinaryOpNode(
                                BinaryOpNode(
                                    _member(_id("destination"), "identity"),
                                    "==",
                                    _integer(destination_index),
                                ),
                                "&&",
                                BinaryOpNode(
                                    _member(_id("source"), "identity"),
                                    "==",
                                    _integer(source_index),
                                ),
                            ),
                            BlockNode(statements),
                        )
                    )
            self.generated.append(
                FunctionNode(
                    name,
                    PrimitiveType("void"),
                    [
                        ParameterNode("destination", self.target_type(dst)),
                        ParameterNode("source", self.target_type(src)),
                        *self.resource_parameters(),
                    ],
                    BlockNode(body),
                )
            )
        return _call(
            self.helpers[key],
            [
                self.expression(destination.expression, env),
                self.expression(source.expression, env),
                *self.resource_arguments(),
            ],
        )

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
                    self.helper(pointer, "store", self.resource_origins.get(id(owner))),
                    [
                        self.expression(owner, env),
                        _call("int64_t", [self.expression(index, env)]),
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
                self.helper(pointer, "load", self.resource_origins.get(id(owner))),
                [
                    self.expression(owner, env),
                    _call("int64_t", [self.expression(index, env)]),
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
                if node.operator != "+":
                    raise ResourceAggregateError("pointer-binary-operator", node)
                return _call(
                    self.helper(right_type, "shift"),
                    [
                        self.expression(node.right, env),
                        _call("int64_t", [self.expression(node.left, env)]),
                    ],
                )
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
                        [
                            self.expression(owner, env),
                            _call("int64_t", [self.expression(index, env)]),
                        ],
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
            atomic = self.atomic_call(node, env)
            if atomic is not None:
                return atomic
            if (
                _name(node.function) == "buffer_store"
                and "buffer_store" not in self.functions
                and len(node.arguments) == 3
                and isinstance(self.infer(node.arguments[0], env), _Pointer)
            ):
                return _call(
                    self.helper(self.infer(node.arguments[0], env), "store"),
                    [
                        self.expression(node.arguments[0], env),
                        _call("int64_t", [self.expression(node.arguments[1], env)]),
                        self.expression(node.arguments[2], env),
                        *self.resource_arguments(),
                    ],
                )
            callee = self.function_for_call(node, env)
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
                (
                    _id(self.overload_names[id(callee)])
                    if id(callee) in self.overload_names
                    else node.function
                ),
                args,
                node.generic_args or None,
                source_location=location,
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
        expression = (
            node.expression if isinstance(node, ExpressionStatementNode) else node
        )
        record_copy = self.record_copy(expression, env)
        if record_copy is not None:
            return ExpressionStatementNode(
                record_copy, source_location=node.source_location
            )
        if isinstance(node, BlockNode):
            scoped = dict(env)
            return BlockNode(
                [self.statement(item, scoped) for item in node.statements],
                source_location=node.source_location,
            )
        if isinstance(node, VariableNode):
            if id(node) in self.workgroup_roots:
                pointer = self.workgroup_roots[id(node)]
                env[node.name] = pointer
                return VariableNode(
                    pointer.local_name,
                    self.target_type(pointer),
                    _call(self.helper(pointer, "make"), [_integer(pointer.identity)]),
                )
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
            pointer = source_type
            while isinstance(pointer, tuple):
                pointer = pointer[0]
            if isinstance(pointer, _Pointer):
                # Address/access qualifiers describe the pointee. The lowered
                # handle (or array of handles) is still private, mutable data.
                pointee_qualifiers = {
                    "const",
                    "device",
                    "constant",
                    "global",
                    "storage",
                    "threadgroup",
                    "workgroup",
                    "readonly",
                    "writeonly",
                }
                result.qualifiers = [
                    q for q in result.qualifiers if q not in pointee_qualifiers
                ]
                result.attributes = [
                    a for a in result.attributes if _name(a) not in pointee_qualifiers
                ]
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
        if isinstance(node, (WhileNode, DoWhileNode)):
            result = copy(node)
            result.condition = self.expression(node.condition, env)
            result.body = self.statement(node.body, dict(env))
            return result
        if isinstance(node, SwitchNode):
            result = copy(node)
            result.expression = self.expression(node.expression, env)
            result.cases = []
            scoped = dict(env)
            for case in node.cases:
                rewritten = copy(case)
                rewritten.value = self.expression(case.value, scoped)
                rewritten.statements = [
                    self.statement(item, scoped) for item in case.statements
                ]
                result.cases.append(rewritten)
            result.default_case = self.statement(node.default_case, scoped)
            return result
        return self.expression(node, env)

    def run(self):
        self.discover(self.entry)
        self.elide_unused_null_parameters()
        if any(isinstance(node, PointerReinterpretNode) for node in self.ast.walk()):
            self.resource_origins = ResourceOriginAnalysis(self, _Pointer).analyze()
        # Bind calls to distinct symbols before pointer handles erase access and
        # address-space differences between source overloads.
        for name, overloads in self.functions.items():
            if len(overloads) > 1:
                for function in overloads:
                    if id(function) in self.reachable and function is not self.entry:
                        self.overload_names[id(function)] = self.fresh(
                            f"{name}_resource_overload"
                        )
        # A changed resource-aggregate type cannot retain an unlowered helper ABI.
        # Prune only unreachable resource helpers; unrelated global initializers
        # may still refer to ordinary value functions outside the entry closure.
        unused = {
            id(function)
            for overloads in self.functions.values()
            for function in overloads
            if id(function) not in self.reachable
            and any(
                self.contains_reference(type_)
                for type_ in [
                    *self.parameter_types[id(function)],
                    self.returns[id(function)],
                ]
            )
        }
        self.ast.functions = [
            function for function in self.ast.functions if id(function) not in unused
        ]
        for stage in self.ast.stages.values():
            stage.local_functions = [
                function
                for function in stage.local_functions
                if id(function) not in unused
            ]
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
                        not in {
                            "const",
                            "device",
                            "constant",
                            "global",
                            "storage",
                            "threadgroup",
                            "workgroup",
                        }
                    ]
        for overloads in self.functions.values():
            for function in overloads:
                if id(function) not in self.reachable:
                    continue
                self.current = function
                types = self.parameter_types[id(function)]
                env = dict(self.global_types)
                env.update(
                    {
                        param.name: type_
                        for param, type_ in zip(function.parameters, types)
                    }
                )
                if function is self.entry:
                    for index, (param, pointer, _root) in enumerate(self.resources):
                        if pointer.space == "threadgroup":
                            continue
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
                    function.name = self.overload_names.get(id(function), function.name)
        self.entry.body.statements[:0] = [
            VariableNode(
                name,
                initial_value=_id(param.name),
                **self.resource_declaration(param, pointer, "var_type"),
            )
            for param, pointer, name in self.resources
            if pointer.space != "threadgroup"
        ] + [
            VariableNode(
                self.entry_handles[param.name],
                self.target_type(pointer),
                _call(self.helper(pointer, "make"), [_integer(index)]),
            )
            for index, (param, pointer, _name_) in enumerate(self.resources)
            if pointer.space != "threadgroup"
        ]
        self.ast.global_variables.extend(self.workgroup_declarations)
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


def lower_resource_aggregates(
    ast, *, storage_pointer_parameters=False, workgroup_access_assertions=()
):
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
    return _Lowering(
        result, entry, storage_pointer_parameters, workgroup_access_assertions
    ).run()
