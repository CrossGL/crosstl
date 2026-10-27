"""Lower private aggregates containing storage pointers to resource handles.

Handles are private values, never a buffer ABI. Their identity is drawn from the
selected entry's bindings and fixed workgroup allocations; access helpers branch
over concrete storage instead of constructing target resource-object arrays.
The source AST is retained.
"""

from copy import copy, deepcopy
from dataclasses import dataclass, replace

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
from .resource_storage_views import (
    combined_load,
    restorable_view,
    storage_view,
    view_load,
)
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


# CrossGL spellings of the 16-bit floating storage types.
_NARROW_FLOAT_ELEMENTS = frozenset({"float16", "bfloat16"})
# Other spellings of binary16 that type aliases may name.
_BINARY16_SPELLINGS = frozenset({"f16", "half"})


@dataclass(frozen=True)
class _RootPointer(_Pointer):
    identity: int
    local_name: str
    backing_name: str = ""


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
    # A typed literal lets target range proofs compare it with the typed identity
    # member, so a view's proven identity decides the accessor branch it takes.
    return LiteralNode(value, PrimitiveType("int"))


def _member(value, name):
    return MemberAccessNode(value, name)


def _pointer_type(value, owner=None, *, allow_thread=False):
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
    allowed_spaces = (
        "constant",
        "device",
        "global",
        "storage",
        "threadgroup",
        "workgroup",
        *(("thread", "private") if allow_thread else ()),
    )
    space = value.address_space or next(
        (q for q in allowed_spaces if q in qualifiers),
        None,
    )
    if space not in set(allowed_spaces):
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
            "thread"
            if space in {"thread", "private"}
            else (
                "threadgroup"
                if space in {"threadgroup", "workgroup"}
                else "constant" if space == "constant" else "device"
            )
        ),
        writable,
        readable,
    )


def _remove_pointee_qualifiers(node):
    # Storage access belongs to the pointee, not its private identity/offset value.
    qualifiers = {
        "const",
        "device",
        "constant",
        "global",
        "storage",
        "threadgroup",
        "workgroup",
        "thread",
        "private",
        "readonly",
        "writeonly",
    }
    node.qualifiers = [q for q in node.qualifiers if q not in qualifiers]
    node.attributes = [a for a in node.attributes if _name(a) not in qualifiers]


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


def _contains_thread_pointer(node):
    return any(
        isinstance(child, PointerType) and child.address_space in {"thread", "private"}
        for child in node.walk()
    )


def _is_used_as_pointer(body, name):
    direct_array_uses = set()
    for node in body.walk():
        if (
            isinstance(node, UnaryOpNode)
            and node.operator == "&"
            and isinstance(node.operand, ArrayAccessNode)
            and isinstance(node.operand.array_expr, IdentifierNode)
            and node.operand.array_expr.name == name
        ):
            return True
        if (
            isinstance(node, ArrayAccessNode)
            and isinstance(node.array_expr, IdentifierNode)
            and node.array_expr.name == name
        ):
            direct_array_uses.add(id(node.array_expr))
    for node in body.walk():
        if (
            isinstance(node, IdentifierNode)
            and node.name == name
            and id(node) not in direct_array_uses
        ):
            return True
    return False


def _static_member(member):
    qualifiers = {
        str(_name(q)).lower() for q in getattr(member, "qualifiers", ()) or ()
    }
    attributes = {str(_name(a)).lower() for a in member.attributes or ()}
    return "static" in qualifiers or "static" in attributes


def _literal_constant(node):
    """Only inline constants whose meaning cannot depend on the use-site scope."""
    if isinstance(node, LiteralNode):
        return True
    if isinstance(node, UnaryOpNode):
        return node.operator in {"+", "-", "!", "~"} and _literal_constant(node.operand)
    if isinstance(node, BinaryOpNode):
        return (
            node.operator in {
                "+",
                "-",
                "*",
                "/",
                "%",
                "<<",
                ">>",
                "&",
                "|",
                "^",
                "&&",
                "||",
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
            }
            and _literal_constant(node.left)
            and _literal_constant(node.right)
        )
    if isinstance(node, TernaryOpNode):
        return all(
            _literal_constant(value)
            for value in (node.condition, node.true_expr, node.false_expr)
        )
    if isinstance(node, FunctionCallNode):
        return _numeric(NamedType(_name(node.function))) is not None and all(
            _literal_constant(argument) for argument in node.arguments
        )
    return False


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
        self.storage_origins = {}
        self.structs = {}
        for node in ast.walk():
            if isinstance(node, StructNode):
                if node.name in self.structs:
                    raise ResourceAggregateError("ambiguous-aggregate-type", node)
                self.structs[node.name] = node
        self.has_thread_aggregate_pointers = any(
            _contains_thread_pointer(struct) for struct in self.structs.values()
        )
        self.functions = {}
        for node in ast.walk():
            if isinstance(node, FunctionNode):
                self.functions.setdefault(node.name, []).append(node)
        self.fields = {
            name: {
                member.name: self.source_type(member.member_type, member)
                for member in struct.members
                if not _static_member(member)
            }
            for name, struct in self.structs.items()
        }
        self.static_fields = {
            name: {
                member.name: self.source_type(member.member_type, member)
                for member in struct.members
                if _static_member(member)
            }
            for name, struct in self.structs.items()
        }
        for name, fields in self.static_fields.items():
            if any(self.contains_reference(type_) for type_ in fields.values()):
                raise ResourceAggregateError(
                    "static-resource-reference", self.structs[name]
                )
        self.resources = []
        for param in entry.parameters:
            pointer = _pointer_type(
                param.param_type,
                param,
                allow_thread=self.has_thread_aggregate_pointers,
            )
            if pointer is not None:
                if pointer.space in {"threadgroup", "thread"}:
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
        # (view element, space) -> storage elements some view reads. The
        # origin analysis infers every reachable expression, and so registers
        # every view, before any helper is built.
        self.storage_views = {}
        # (element, space) -> elements some view of its handles yields.
        self.view_targets = {}
        self.viewed_loads = set()
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
        pointer = _pointer_type(
            value, owner, allow_thread=self.has_thread_aggregate_pointers
        )
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
            is_workgroup = bool(
                qualifiers & {"threadgroup", "workgroup", "shared", "groupshared"}
            )
            is_thread_local = (
                not is_workgroup
                and self.has_thread_aggregate_pointers
                and qualifiers <= {"thread", "private", "const"}
                and _is_used_as_pointer(function.body, node.name)
            )
            if not is_workgroup and not is_thread_local:
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
            # Binary16 and bfloat16 scalars keep their own element type; the
            # target backends already lay out and convert shared 16-bit arrays.
            narrow_float = _name(element) in _NARROW_FLOAT_ELEMENTS
            if (
                extent is None
                or extent <= 0
                or not narrow_float
                and (
                    numeric is None
                    or numeric.bits != 32
                    or numeric.lanes not in {1, 2, 4}
                    or numeric.kind == ArithmeticScalarKind.BOOLEAN
                )
            ):
                raise ResourceAggregateError(
                    (
                        "workgroup-allocation-layout"
                        if is_workgroup
                        else "local-allocation-layout"
                    ),
                    node,
                )
            if is_workgroup and (
                node.initial_value is not None
                or qualifiers
                - {
                    "threadgroup",
                    "workgroup",
                    "shared",
                    "groupshared",
                }
            ):
                raise ResourceAggregateError(
                    "workgroup-allocation-initializer-or-qualifier", node
                )
            if (
                is_thread_local
                and node.initial_value is not None
                and not (
                    isinstance(node.initial_value, ArrayLiteralNode)
                    and len(node.initial_value.elements) == extent
                )
            ):
                raise ResourceAggregateError(
                    "local-allocation-initializer-or-qualifier", node
                )
            element_name = _name(element) or arithmetic_type_name(
                numeric.kind, numeric.bits, numeric.lanes
            )
            space = "threadgroup" if is_workgroup else "thread"
            writable = True if is_workgroup else ("const" not in qualifiers)
            pointer = _Pointer(element_name, space, writable, True)
            prefix = "crosstl_workgroup" if is_workgroup else "crosstl_local"
            root = self.fresh(f"{prefix}_{node.name}")
            handle = self.fresh(f"crosstl_reference_{node.name}")
            identity = len(self.resources)
            if is_workgroup:
                declaration = copy(node)
                declaration.name = root
                declaration.var_type = declaration.vtype = ArrayType(
                    deepcopy(element), _integer(extent)
                )
            else:
                declaration = VariableNode(
                    root,
                    ArrayType(deepcopy(element), _integer(extent)),
                    qualifiers=["static"],
                )
            self.workgroup_declarations.append(declaration)
            parameter = ParameterNode(
                root, PointerType(deepcopy(element), address_space=space)
            )
            self.resources.append((parameter, pointer, root))
            self.workgroup_roots[id(node)] = _RootPointer(
                pointer.element,
                pointer.space,
                writable,
                True,
                identity,
                handle,
                root,
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
            if pointer.space not in {"threadgroup", "thread"}
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

    def storage_read(self, pointer, owner, call):
        """Record the bindings a storage load can read when origins are proven.

        A load helper selects its binding by the handle's identity, so a handle
        whose origins are proven reads only those bindings. Uniformity proofs
        then need only those bindings to be immutable, not every binding of
        the element type. ``call.storage_reads`` names them in the call's
        argument spelling; the helper body itself is unchanged.
        """
        origins = self.storage_origins.get(id(owner))
        if pointer.space == "threadgroup" or not origins:
            return call
        names = []
        for identity in sorted(origins):
            if not 0 <= identity < len(self.resources):
                return call
            _param, resource, name = self.resources[identity]
            if (
                resource.space == "threadgroup"
                or resource.element != pointer.element
                or resource.space != pointer.space
                or not resource.readable
            ):
                return call
            names.append(name)
        call.storage_reads = tuple(names)
        return call

    def resource_arguments(self):
        return [
            _id(name)
            for _param, pointer, name in self.resources
            if pointer.space not in {"threadgroup", "thread"}
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
        if isinstance(node, PointerReinterpretNode):
            return self.pointer_view(node, env)
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
            return self.fields.get(_name(owner), {}).get(
                node.member, self.static_fields.get(_name(owner), {}).get(node.member)
            )
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

    def pointer_view(self, node, env):
        """The pointer a reinterpretation in one address space yields.

        A view keeps the source's allocation and grants at most the access
        both pointers grant, so an access the source lacks fails where it is
        made. A view of the same element changes no byte and is the source
        pointer itself. A view of another element is a narrowing integer view
        of device or constant storage (see ``resource_storage_views``), or a
        widening view whose elements every storage the source names holds
        whole or in adjacent narrower elements; both are read-only. Any other
        reinterpretation stays unknown.
        """
        source = self.infer(node.expression, env)
        if not isinstance(source, _Pointer):
            return None
        try:
            target = _pointer_type(node.target_type)
        except ResourceAggregateError:
            return None
        if not isinstance(target, _Pointer) or target.space != source.space:
            return None
        readable = source.readable and target.readable
        if target.element == source.element:
            return replace(
                source, writable=source.writable and target.writable, readable=readable
            )
        if source.space == "threadgroup":
            return None
        space = source.space
        storages = self.storage_views.get((source.element, space), set())
        if storage_view(source.element, target.element) is None:
            if not self.widening_view(source, target, storages):
                return None
            if self.binds_element(source.element, space):
                # Handles of the narrow element may name its own bindings.
                storages = {source.element, *storages}
        else:
            # A narrowing view reads the source's own bindings too.
            storages = {source.element, *storages}
        self.view_targets.setdefault((source.element, space), set()).add(target.element)
        for storage in sorted(storages):
            self.register_view(storage, target.element, space, node)
        return _Pointer(target.element, space, False, readable)

    def register_view(self, storage, element, space, node):
        """Record that ``element`` handles in ``space`` may name ``storage``.

        Handles viewed from ``element`` handles carry the same identities, so
        the storage reaches every element viewed from it as well.
        """
        if storage == element:
            # The handle's own element: load helpers read it directly.
            return
        if not restorable_view(storage, element):
            # Neither whole nor adjacent storage elements form the element.
            raise ResourceAggregateError("widened-narrow-storage", node)
        key = element, space
        viewed = self.storage_views.setdefault(key, set())
        if storage in viewed:
            return
        if key in self.viewed_loads:
            # A load helper of this element no longer covers every view.
            raise ResourceAggregateError("late-storage-view", node)
        viewed.add(storage)
        for target in sorted(self.view_targets.get(key, ())):
            self.register_view(storage, target, space, node)

    def widening_view(self, source, target, storages):
        """Whether ``target`` handles can restore what ``source`` handles view.

        A ``source`` handle names a binding of its own element or a storage a
        narrowing view of it reads (``storages``). The handle's offset divides
        into ``target`` elements, which each such storage holds whole or as
        adjacent unsigned narrower elements that a load combines. An offset
        that does not divide is a misaligned ``target`` pointer, whose use
        C++, and so MSL, leaves undefined.
        """
        if storage_view(target.element, source.element) is None:
            return False
        if self.binds_element(source.element, source.space):
            storages = {source.element, *storages}
        return bool(storages) and all(
            restorable_view(storage, target.element) for storage in storages
        )

    def binds_element(self, element, space):
        return any(
            resource.element == element and resource.space == space
            for _param, resource, _name in self.resources
        )

    def view_helper(self, source, target):
        """A helper turning a ``source`` handle into a ``target`` view handle.

        A narrowing view counts ``ratio`` view elements per source element; a
        widening view divides the offset back into its wider elements.
        """
        view = storage_view(source.element, target.element)
        if view is None:
            view = storage_view(target.element, source.element)
            operator = "/"
        else:
            operator = "*"
        key = (source.element, target.element), "view", None
        if key in self.helpers:
            return self.helpers[key]
        name = self.fresh(f"crosstl_resource_view_{source.element}_as_{target.element}")
        self.helpers[key] = name
        handle = self.target_type(target)
        reference = _member(_id("reference"), "identity")
        offset = _member(_id("reference"), "offset")
        body = [
            VariableNode("view", handle),
            AssignmentNode(_member(_id("view"), "identity"), reference),
            AssignmentNode(
                _member(_id("view"), "offset"),
                BinaryOpNode(
                    offset, operator, _call("int64_t", [_integer(view.ratio)])
                ),
            ),
            ReturnNode(_id("view")),
        ]
        function = FunctionNode(
            name,
            handle,
            [ParameterNode("reference", self.target_type(source))],
            BlockNode(body),
        )
        function.resource_aggregate_nonmutating = True
        self.generated.append(function)
        return name

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
            if isinstance(node, IdentifierNode) and node.name == "nullptr":
                return
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
        if pointer.space not in {"threadgroup", "thread"} or operation not in {
            "load",
            "store",
        }:
            origins = None
        key = pointer, operation, origins
        if key in self.helpers:
            return self.helpers[key]
        storage = (
            "workgroup"
            if pointer.space == "threadgroup"
            else "local" if pointer.space == "thread" else "resource"
        )
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
            # Read-only loads also read the storage views of this element.
            viewed = set()
            if operation == "load" and not pointer.writable:
                self.viewed_loads.add((pointer.element, pointer.space))
                viewed = self.storage_views.get((pointer.element, pointer.space), set())
            candidates = [
                (index, resource_name, resource.element)
                for index, (_param, resource, resource_name) in enumerate(
                    self.resources
                )
                if (resource.element == pointer.element or resource.element in viewed)
                and resource.space == pointer.space
                and (operation != "store" or resource.writable)
                and (operation != "load" or resource.readable)
                and (origins is None or index in origins)
            ]
            if not candidates:
                raise ResourceAggregateError("missing-compatible-resource")
            body = []
            for index, resource_name, element in candidates:
                position = BinaryOpNode(
                    _member(_id("reference"), "offset"), "+", _id("index")
                )
                if element == pointer.element:
                    access = ArrayAccessNode(_id(resource_name), position)
                elif storage_view(element, pointer.element) is not None:
                    access = view_load(
                        storage_view(element, pointer.element),
                        _id(resource_name),
                        position,
                    )
                else:
                    # Only read-only loads reach the storage of a widening view.
                    access = combined_load(
                        storage_view(pointer.element, element),
                        _id(resource_name),
                        position,
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
        members = (
            [member for member in record.members if not _static_member(member)]
            if record is not None
            else []
        )
        if (
            record is None
            or len(members) != 1
            or record.generic_params
            or record.inheritance
        ):
            raise ResourceAggregateError("record-copy-layout", node)
        member = members[0]
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
        # Copies move whole backing elements: 16-bit floating or 32-bit scalars.
        element_bytes = (
            2
            if src.element in _NARROW_FLOAT_ELEMENTS
            else (
                4
                if backing is not None
                and backing.bits == 32
                and backing.lanes == 1
                and backing.kind != ArithmeticScalarKind.BOOLEAN
                else None
            )
        )
        if (
            src.element != dst.element
            or element_bytes is None
            or alignment > element_bytes
            or extent % element_bytes
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
        words = extent // element_bytes
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

    def initializer_requires_sequence(self, arguments):
        for argument in arguments:
            for node in argument.walk():
                if isinstance(node, AssignmentNode) or (
                    isinstance(node, UnaryOpNode) and node.operator in {"++", "--"}
                ):
                    return True
                if isinstance(node, FunctionCallNode) and (
                    _name(node.function) in self.functions
                    or _numeric(NamedType(_name(node.function))) is None
                ):
                    return True
                if not isinstance(
                    node,
                    (
                        ArrayAccessNode,
                        ArrayLiteralNode,
                        BinaryOpNode,
                        CastNode,
                        FunctionCallNode,
                        IdentifierNode,
                        LiteralNode,
                        MemberAccessNode,
                        PointerReinterpretNode,
                        TernaryOpNode,
                        TypeNode,
                        UnaryOpNode,
                    ),
                ):
                    return True
        return False

    def aggregate_value(self, type_name, arguments, *, sequence=False):
        """Construct private values without target-specific struct expressions."""
        fields = self.fields[type_name]
        values = []

        def flatten(type_, argument, path, target):
            if isinstance(type_, tuple) and isinstance(argument, ArrayLiteralNode):
                for index, element in enumerate(argument.elements):
                    flatten(
                        type_[0],
                        element,
                        (*path, index),
                        ArrayAccessNode(target, _integer(index)),
                    )
            else:
                values.append((path, target, type_, argument))

        for (field, type_), argument in zip(fields.items(), arguments):
            flatten(type_, argument, (field,), MemberAccessNode(_id("value"), field))
        key = ("aggregate-value", type_name, tuple(path for path, *_ in values))
        if key not in self.helpers:
            name = self.fresh(f"crosstl_make_{type_name}")
            self.helpers[key] = name
            parameters = [
                ParameterNode(f"field_{index}", self.target_type(type_))
                for index, (_path, _target, type_, _argument) in enumerate(values)
            ]
            body = [VariableNode("value", NamedType(type_name))]
            body.extend(
                AssignmentNode(target, _id(parameter.name))
                for (_path, target, _type, _argument), parameter in zip(
                    values, parameters
                )
            )
            body.append(ReturnNode(_id("value")))
            self.generated.append(
                FunctionNode(name, NamedType(type_name), parameters, BlockNode(body))
            )
        arguments = [argument for _path, _target, _type, argument in values]
        if not sequence or len(values) < 2:
            return _call(self.helpers[key], arguments)
        # Function arguments need not be evaluated in initializer-list order.
        # Hoist only declarations; captures stay at the original expression,
        # including conditional branches and repeatedly evaluated loop clauses.
        captures = []
        arguments = []
        for _path, _target, type_, argument in values:
            name = self.fresh("crosstl_initializer")
            self.initializer_temporaries.append(
                VariableNode(name, self.target_type(type_))
            )
            captures.append(AssignmentNode(_id(name), argument))
            arguments.append(_id(name))
        result = _call(self.helpers[key], arguments)
        for capture in reversed(captures):
            result = BinaryOpNode(capture, ",", result)
        return result

    def expression(self, node, env, expected=None):
        if node is None:
            return None
        if isinstance(node, (BlockNode, VariableNode, IfNode, ForNode, ReturnNode)):
            return self.statement(node, env)
        actual = self.infer(node, env)
        self.compatible(expected, actual, node)
        location = getattr(node, "source_location", None)
        if isinstance(node, IdentifierNode):
            if node.name == "nullptr" and isinstance(expected, _Pointer):
                return _call(self.helper(expected, "make"), [_integer(-1)])
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
            return self.storage_read(
                pointer,
                owner,
                _call(
                    self.helper(pointer, "load", self.resource_origins.get(id(owner))),
                    [
                        self.expression(owner, env),
                        _call("int64_t", [self.expression(index, env)]),
                        *self.resource_arguments(),
                    ],
                ),
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
            owner_type = _name(self.infer(node.object_expr, env))
            static_type = self.static_fields.get(owner_type, {}).get(node.member)
            if static_type is not None:
                member = next(
                    member
                    for member in self.structs[owner_type].members
                    if member.name == node.member
                )
                qualifiers = {_name(q) for q in getattr(member, "qualifiers", ()) or ()}
                qualifiers.update(_name(a) for a in member.attributes or ())
                initializer = getattr(member, "default_value", None)
                if (
                    initializer is None
                    or not _literal_constant(initializer)
                    or not qualifiers & {"const", "constant", "constexpr"}
                    or _numeric(static_type) is None
                ):
                    raise ResourceAggregateError(
                        "static-member-value-unsupported", node
                    )
                # A static member has no instance slot, but evaluating its
                # object expression can still have observable side effects.
                return _call(
                    _name(static_type),
                    [
                        BinaryOpNode(
                            self.expression(node.object_expr, env),
                            ",",
                            self.expression(initializer, env, static_type),
                            source_location=location,
                        )
                    ],
                )
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
            if fields:
                return self.aggregate_value(
                    _name(expected),
                    [
                        self.expression(value, env, type_)
                        for value, type_ in zip(node.elements, types)
                    ],
                    sequence=self.initializer_requires_sequence(node.elements),
                )
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
            if callee is None and fields:
                return self.aggregate_value(
                    _name(node.function),
                    args,
                    sequence=self.initializer_requires_sequence(node.arguments),
                )
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
        if isinstance(node, PointerReinterpretNode) and isinstance(actual, _Pointer):
            source = self.infer(node.expression, env)
            value = self.expression(node.expression, env)
            if source.element == actual.element:
                return value
            return _call(self.view_helper(source, actual), [value])
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
        step = self.pointer_step(expression, env)
        if step is not None:
            return ExpressionStatementNode(step, source_location=node.source_location)
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
                init_expr = _call(
                    self.helper(pointer, "make"), [_integer(pointer.identity)]
                )
                if pointer.space == "thread" and isinstance(
                    node.initial_value, ArrayLiteralNode
                ):
                    for idx, elem in reversed(
                        list(enumerate(node.initial_value.elements))
                    ):
                        assign = AssignmentNode(
                            ArrayAccessNode(_id(pointer.backing_name), _integer(idx)),
                            self.expression(elem, env, PrimitiveType(pointer.element)),
                        )
                        init_expr = BinaryOpNode(assign, ",", init_expr)
                return VariableNode(
                    pointer.local_name,
                    self.target_type(pointer),
                    init_expr,
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
                _remove_pointee_qualifiers(result)
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
                self.pointer_step(node.update, scoped)
                or self.expression(node.update, scoped),
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

    def pointer_step(self, node, env):
        """A pointer increment or decrement whose value is discarded.

        It advances the handle's offset by one element. An increment whose
        value is used still fails closed in ``expression``.
        """
        if not (
            isinstance(node, UnaryOpNode)
            and node.operator in {"++", "--"}
            and isinstance(self.infer(node.operand, env), _Pointer)
        ):
            return None
        return AssignmentNode(
            _member(self.expression(node.operand, env), "offset"),
            _call("int64_t", [_integer(1)]),
            "+=" if node.operator == "++" else "-=",
            source_location=getattr(node, "source_location", None),
        )

    def default_state(self, type_, valid=False):
        if isinstance(type_, _Pointer):
            return bool(valid)
        if isinstance(type_, tuple):
            extent = evaluate_literal_int_expression(type_[1])
            return (
                [self.default_state(type_[0], valid) for _ in range(extent)]
                if extent is not None and extent >= 0
                else None
            )
        fields = self.fields.get(_name(type_))
        if fields is not None:
            return {
                field: self.default_state(field_type, valid)
                for field, field_type in fields.items()
            }
        return None

    def merge_state(self, left, right):
        if isinstance(left, bool) and isinstance(right, bool):
            return left and right
        if isinstance(left, dict) and isinstance(right, dict):
            return {
                key: self.merge_state(left.get(key), right.get(key))
                for key in set(left) | set(right)
            }
        if (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
        ):
            return [self.merge_state(a, b) for a, b in zip(left, right)]
        return None

    def get_lvalue_state(self, target, state):
        if isinstance(target, IdentifierNode):
            return state.get(target.name)
        if isinstance(target, MemberAccessNode):
            owner_state = self.get_lvalue_state(target.object_expr, state)
            return (
                owner_state.get(target.member)
                if isinstance(owner_state, dict)
                else None
            )
        if isinstance(target, ArrayAccessNode):
            owner_state = self.get_lvalue_state(target.array_expr, state)
            idx = evaluate_literal_int_expression(target.index_expr)
            if (
                isinstance(owner_state, list)
                and idx is not None
                and 0 <= idx < len(owner_state)
            ):
                return owner_state[idx]
            return None
        return None

    def set_lvalue_state(self, target, value_state, state):
        if isinstance(target, IdentifierNode):
            state[target.name] = deepcopy(value_state)
            return True
        if isinstance(target, MemberAccessNode):
            owner_state = self.get_lvalue_state(target.object_expr, state)
            if isinstance(owner_state, dict) and target.member in owner_state:
                owner_state[target.member] = deepcopy(value_state)
                return True
            return False
        if isinstance(target, ArrayAccessNode):
            owner_state = self.get_lvalue_state(target.array_expr, state)
            idx = evaluate_literal_int_expression(target.index_expr)
            if isinstance(owner_state, list):
                if idx is not None and 0 <= idx < len(owner_state):
                    owner_state[idx] = deepcopy(value_state)
                    return True
                for i in range(len(owner_state)):
                    owner_state[i] = self.merge_state(owner_state[i], value_state)
                return True
            return False
        return False

    def verify_lvalue_subexpressions(self, target, env, state):
        if isinstance(target, MemberAccessNode):
            self.verify_lvalue_subexpressions(target.object_expr, env, state)
        elif isinstance(target, ArrayAccessNode):
            self.verify_lvalue_subexpressions(target.array_expr, env, state)
            self.verify_expression(target.index_expr, env, state)

    def verify_call(self, callee, call_node, arg_states, caller_state):
        param_types = self.parameter_types[id(callee)]
        ret_type = self.returns[id(callee)]
        builds_or_mutates_aggregate = self.contains_reference(ret_type) or any(
            (
                isinstance(param.param_type, ReferenceType)
                or set(param.qualifiers) & {"out", "inout"}
            )
            and self.contains_reference(ptype)
            and not isinstance(ptype, _Pointer)
            for param, ptype in zip(callee.parameters, param_types)
        )
        for param, ptype, arg_state in zip(callee.parameters, param_types, arg_states):
            if (
                isinstance(ptype, _Pointer)
                and arg_state is False
                and not builds_or_mutates_aggregate
            ):
                raise ResourceAggregateError("pointer-contract-mismatch", call_node)
        callee_env = dict(self.global_types)
        callee_state = {}
        for param, ptype, arg_state in zip(callee.parameters, param_types, arg_states):
            callee_env[param.name] = ptype
            callee_state[param.name] = (
                deepcopy(arg_state)
                if arg_state is not None
                else self.default_state(ptype, valid=True)
            )
        if id(callee) in self.verifying:
            raise ResourceAggregateError("recursive-call", call_node)
        self.verifying.add(id(callee))
        try:
            ret_state = self.verify_function_body(callee, callee_env, callee_state)
        finally:
            self.verifying.discard(id(callee))
        for param, arg in zip(callee.parameters, call_node.arguments):
            if isinstance(param.param_type, ReferenceType) or (
                set(param.qualifiers) & {"out", "inout"}
            ):
                self.set_lvalue_state(arg, callee_state.get(param.name), caller_state)
        return (
            ret_state
            if ret_state is not None
            else self.default_state(ret_type, valid=True)
        )

    def verify_expression(self, node, env, state, expected=None):
        if node is None:
            return None
        if isinstance(node, (BlockNode, VariableNode, IfNode, ForNode, ReturnNode)):
            self.verify_statement(node, env, state)
            return None
        actual = self.infer(node, env)
        if isinstance(node, IdentifierNode):
            if node.name == "nullptr":
                if not isinstance(expected, _Pointer):
                    raise ResourceAggregateError("pointer-contract-mismatch", node)
                return False
            if isinstance(actual, _RootPointer):
                return True
            return deepcopy(
                state.get(node.name, self.default_state(actual, valid=True))
            )
        if isinstance(node, AssignmentNode):
            access = self.access(node.target, env)
            if access is not None:
                _pointer, owner, index = access
                if self.verify_expression(owner, env, state) is not True:
                    raise ResourceAggregateError("null-pointer-dereference", node)
                self.verify_expression(index, env, state)
                self.verify_expression(node.value, env, state)
                return None
            target_type = self.infer(node.target, env)
            if node.operator == "=":
                val_state = self.verify_expression(node.value, env, state, target_type)
                self.verify_lvalue_subexpressions(node.target, env, state)
                self.set_lvalue_state(node.target, val_state, state)
                return deepcopy(val_state)
            cur_state = self.verify_expression(node.target, env, state)
            self.verify_expression(node.value, env, state)
            if (
                isinstance(target_type, _Pointer)
                and node.operator in {"+=", "-="}
                and cur_state is not True
            ):
                raise ResourceAggregateError("null-pointer-dereference", node)
            return deepcopy(cur_state)
        access = self.access(node, env)
        if access is not None:
            _pointer, owner, index = access
            if self.verify_expression(owner, env, state) is not True:
                raise ResourceAggregateError("null-pointer-dereference", node)
            self.verify_expression(index, env, state)
            return None
        if isinstance(node, BinaryOpNode):
            if node.operator == ",":
                self.verify_expression(node.left, env, state)
                return self.verify_expression(node.right, env, state, expected)
            left_type, right_type = self.infer(node.left, env), self.infer(
                node.right, env
            )
            left_state = self.verify_expression(node.left, env, state)
            right_state = self.verify_expression(node.right, env, state)
            if (
                isinstance(left_type, _Pointer)
                and node.operator in {"+", "-"}
                and not isinstance(right_type, _Pointer)
            ):
                if left_state is not True:
                    raise ResourceAggregateError("null-pointer-dereference", node)
                return True
            if isinstance(right_type, _Pointer) and node.operator == "+":
                if right_state is not True:
                    raise ResourceAggregateError("null-pointer-dereference", node)
                return True
            return None
        if isinstance(node, UnaryOpNode):
            if node.operator == "&":
                access = self.access(node.operand, env)
                if access is not None:
                    _pointer, owner, index = access
                    if self.verify_expression(owner, env, state) is not True:
                        raise ResourceAggregateError("null-pointer-dereference", node)
                    self.verify_expression(index, env, state)
                    return True
            operand_state = self.verify_expression(node.operand, env, state)
            if (
                node.operator in {"++", "--"}
                and isinstance(self.infer(node.operand, env), _Pointer)
                and operand_state is not True
            ):
                raise ResourceAggregateError("null-pointer-dereference", node)
            return operand_state
        if isinstance(node, MemberAccessNode):
            owner_state = self.verify_expression(node.object_expr, env, state)
            if isinstance(owner_state, dict):
                return deepcopy(owner_state.get(node.member))
            return self.default_state(actual, valid=False)
        if isinstance(node, ArrayAccessNode):
            owner_state = self.verify_expression(node.array_expr, env, state)
            self.verify_expression(node.index_expr, env, state)
            idx = evaluate_literal_int_expression(node.index_expr)
            if isinstance(owner_state, list):
                if idx is not None and 0 <= idx < len(owner_state):
                    return deepcopy(owner_state[idx])
                result = (
                    owner_state[0]
                    if owner_state
                    else self.default_state(actual, valid=False)
                )
                for item in owner_state[1:]:
                    result = self.merge_state(result, item)
                return deepcopy(result)
            return self.default_state(actual, valid=False)
        if isinstance(node, ArrayLiteralNode):
            fields = self.fields.get(_name(expected))
            if fields is not None and len(fields) == len(node.elements):
                return {
                    field: self.verify_expression(value, env, state, type_)
                    for (field, type_), value in zip(fields.items(), node.elements)
                }
            if isinstance(expected, tuple):
                return [
                    self.verify_expression(value, env, state, expected[0])
                    for value in node.elements
                ]
            for value in node.elements:
                self.verify_expression(value, env, state)
            return None
        if isinstance(node, FunctionCallNode):
            operation = _name(node.function)
            if (
                operation in {
                    "atomicLoad",
                    "atomicCompareExchangeWeak",
                    "atomicStore",
                    "atomicAdd",
                    "atomicMin",
                    "atomicMax",
                    "atomicAnd",
                    "atomicOr",
                    "atomicXor",
                    "atomicExchange",
                    "atomicCompSwap",
                    "atomicCompareExchange",
                }
                and operation not in self.functions
                and node.arguments
            ):
                target = node.arguments[0]
                while isinstance(target, MemberAccessNode):
                    target = target.object_expr
                access = self.access(target, env)
                if access is not None:
                    _pointer, owner, index = access
                    if self.verify_expression(owner, env, state) is not True:
                        raise ResourceAggregateError("null-pointer-dereference", node)
                    self.verify_expression(index, env, state)
                    for arg in node.arguments[1:]:
                        self.verify_expression(arg, env, state)
                    return None
            if (
                operation == "buffer_store"
                and "buffer_store" not in self.functions
                and len(node.arguments) == 3
                and isinstance(self.infer(node.arguments[0], env), _Pointer)
            ):
                if self.verify_expression(node.arguments[0], env, state) is not True:
                    raise ResourceAggregateError("null-pointer-dereference", node)
                self.verify_expression(node.arguments[1], env, state)
                self.verify_expression(node.arguments[2], env, state)
                return None
            callee = self.function_for_call(node, env)
            fields = self.fields.get(operation)
            types = (
                self.parameter_types[id(callee)]
                if callee is not None
                else list(fields.values()) if fields is not None else None
            )
            arg_states = [
                self.verify_expression(
                    arg,
                    env,
                    state,
                    types[i] if types is not None and i < len(types) else None,
                )
                for i, arg in enumerate(node.arguments)
            ]
            if callee is None and fields is not None and len(fields) == len(arg_states):
                return {field: st for field, st in zip(fields.keys(), arg_states)}
            if callee is not None:
                return self.verify_call(callee, node, arg_states, state)
            return None
        if isinstance(node, TernaryOpNode):
            self.verify_expression(node.condition, env, state)
            then_state = deepcopy(state)
            else_state = deepcopy(state)
            true_val = self.verify_expression(node.true_expr, env, then_state, expected)
            false_val = self.verify_expression(
                node.false_expr, env, else_state, expected
            )
            for key in set(state) | set(then_state) | set(else_state):
                state[key] = self.merge_state(then_state.get(key), else_state.get(key))
            return self.merge_state(true_val, false_val)
        if isinstance(node, PointerReinterpretNode):
            expr_state = self.verify_expression(node.expression, env, state)
            if (
                isinstance(self.infer(node.expression, env), _Pointer)
                and expr_state is not True
            ):
                raise ResourceAggregateError("null-pointer-dereference", node)
            return True
        if isinstance(node, ASTNode) and not isinstance(node, TypeNode):
            for child in node.child_nodes():
                self.verify_expression(child, env, state)
        return None

    def verify_statement(self, node, env, state, return_states=None):
        if node is None:
            return
        if isinstance(node, BlockNode):
            scoped_env = dict(env)
            outer_keys = set(state)
            for item in node.statements:
                self.verify_statement(item, scoped_env, state, return_states)
            for key in list(state):
                if key not in outer_keys:
                    del state[key]
            return
        if isinstance(node, VariableNode):
            if id(node) in self.workgroup_roots:
                pointer = self.workgroup_roots[id(node)]
                env[node.name] = pointer
                if isinstance(node.initial_value, ArrayLiteralNode):
                    for elem in node.initial_value.elements:
                        self.verify_expression(elem, env, state)
                state[node.name] = True
                return
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
            if node.initial_value is not None:
                init_state = self.verify_expression(
                    node.initial_value, env, state, source_type
                )
            else:
                init_state = self.default_state(source_type, valid=False)
            env[node.name] = source_type
            state[node.name] = init_state
            return
        if isinstance(node, ReturnNode):
            val_state = self.verify_expression(
                node.value, env, state, self.returns[id(self.current)]
            )
            if return_states is not None:
                return_states.append(val_state)
            return
        if isinstance(node, IfNode):
            self.verify_expression(node.condition, env, state)
            branch_states = []
            then_state = deepcopy(state)
            self.verify_statement(
                node.then_branch, dict(env), then_state, return_states
            )
            branch_states.append(then_state)
            for cond, body in zip(node.else_if_conditions, node.else_if_bodies):
                self.verify_expression(cond, env, state)
                elif_state = deepcopy(state)
                self.verify_statement(body, dict(env), elif_state, return_states)
                branch_states.append(elif_state)
            else_state = deepcopy(state)
            if node.else_branch is not None:
                self.verify_statement(
                    node.else_branch, dict(env), else_state, return_states
                )
            branch_states.append(else_state)
            for key in list(state):
                merged = branch_states[0].get(key)
                for other in branch_states[1:]:
                    merged = self.merge_state(merged, other.get(key))
                state[key] = merged
            return
        if isinstance(node, ForNode):
            scoped_env = dict(env)
            outer_keys = set(state)
            self.verify_statement(node.init, scoped_env, state, return_states)
            self.verify_loop(
                (node.condition, node.body, node.update),
                scoped_env,
                state,
                return_states,
            )
            for key in list(state):
                if key not in outer_keys:
                    del state[key]
            return
        if isinstance(node, (WhileNode, DoWhileNode)):
            self.verify_loop((node.condition, node.body), env, state, return_states)
            return
        if isinstance(node, SwitchNode):
            self.verify_expression(node.expression, env, state)
            branch_states = []
            for case in node.cases:
                case_state = deepcopy(state)
                if branch_states:
                    # A case without break falls through with the slots the
                    # previous case left behind.
                    for key in case_state:
                        case_state[key] = self.merge_state(
                            case_state[key], branch_states[-1].get(key)
                        )
                case_env = dict(env)
                self.verify_expression(case.value, case_env, case_state)
                for item in case.statements:
                    self.verify_statement(item, case_env, case_state, return_states)
                branch_states.append(case_state)
            default_state = deepcopy(state)
            if node.default_case is not None and branch_states:
                for key in default_state:
                    default_state[key] = self.merge_state(
                        default_state[key], branch_states[-1].get(key)
                    )
            if node.default_case is not None:
                self.verify_statement(
                    node.default_case, dict(env), default_state, return_states
                )
            branch_states.append(default_state)
            for key in list(state):
                merged = branch_states[0].get(key)
                for other in branch_states[1:]:
                    merged = self.merge_state(merged, other.get(key))
                state[key] = merged
            return
        if isinstance(node, ExpressionStatementNode):
            self.verify_expression(node.expression, env, state)
            return
        self.verify_expression(node, env, state)

    def verify_loop(self, parts, env, state, return_states):
        # A null stored late in one iteration reaches the next one, so repeat
        # the loop parts until the merged slot states stop changing.
        while True:
            loop_state = deepcopy(state)
            for part in parts:
                self.verify_statement(part, dict(env), loop_state, return_states)
            merged = {
                key: self.merge_state(state[key], loop_state.get(key)) for key in state
            }
            if merged == state:
                return
            state.update(merged)

    def verify_function_body(self, function, env, state):
        previous = self.current
        self.current = function
        return_states = []
        try:
            self.verify_statement(function.body, env, state, return_states)
        finally:
            self.current = previous
        if not return_states:
            return None
        result = return_states[0]
        for item in return_states[1:]:
            result = self.merge_state(result, item)
        return result

    def verify_definite_assignment(self):
        env = dict(self.global_types)
        state = {}
        types = self.parameter_types[id(self.entry)]
        for param, type_ in zip(self.entry.parameters, types):
            env[param.name] = type_
            state[param.name] = self.default_state(type_, valid=True)
        for index, (param, pointer, _root) in enumerate(self.resources):
            if pointer.space in {"threadgroup", "thread"}:
                continue
            env[param.name] = _RootPointer(
                pointer.element,
                pointer.space,
                pointer.writable,
                pointer.readable,
                index,
                self.entry_handles[param.name],
            )
            state[param.name] = True
        self.verifying = {id(self.entry)}
        self.verify_function_body(self.entry, env, state)

    def run(self):
        self.discover(self.entry)
        self.elide_unused_null_parameters()
        # Only an explicit null needs a proof that it is overwritten before
        # use; programs without one keep the existing lowering unchanged.
        if any(
            isinstance(node, IdentifierNode) and node.name == "nullptr"
            for node in self.ast.walk()
        ):
            self.verify_definite_assignment()
        origins = ResourceOriginAnalysis(self, _Pointer).analyze()
        # Device helpers keep every compatible binding; proven origins only
        # annotate their reads (see ``storage_read``).
        self.storage_origins = origins
        if any(isinstance(node, PointerReinterpretNode) for node in self.ast.walk()):
            self.resource_origins = origins
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
                if _static_member(member):
                    continue
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
                            "thread",
                            "private",
                        }
                    ]
        for overloads in self.functions.values():
            for function in overloads:
                if id(function) not in self.reachable:
                    continue
                self.current = function
                self.initializer_temporaries = []
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
                        if pointer.space in {"threadgroup", "thread"}:
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
                if self.initializer_temporaries:
                    function.body.statements[:0] = self.initializer_temporaries
                if function is not self.entry:
                    for param, type_ in zip(function.parameters, types):
                        if isinstance(param.param_type, ReferenceType):
                            reference = copy(param.param_type)
                            reference.referenced_type = self.target_type(type_)
                            param.param_type = reference
                        else:
                            param.param_type = self.target_type(type_)
                        pointer = type_
                        while isinstance(pointer, tuple):
                            pointer = pointer[0]
                        if isinstance(pointer, _Pointer):
                            _remove_pointee_qualifiers(param)
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
            if pointer.space not in {"threadgroup", "thread"}
        ] + [
            VariableNode(
                self.entry_handles[param.name],
                self.target_type(pointer),
                _call(self.helper(pointer, "make"), [_integer(index)]),
            )
            for index, (param, pointer, _name_) in enumerate(self.resources)
            if pointer.space not in {"threadgroup", "thread"}
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


def _resolve_narrow_float_aliases(ast):
    """Spell aliases of 16-bit floating scalars by their CrossGL type.

    Resource identity compares pointee names, so ``bfloat16_t*`` members must
    match ``bfloat16`` entry buffers, and ``float16_t*`` members, whose alias
    may name binary16 as ``half`` or ``f16``, ``float16`` ones. Other aliases
    keep their spelling.
    """
    declared = {
        variable.name: variable.var_type
        for variable in getattr(ast, "global_variables", ())
        if isinstance(variable, VariableNode) and variable.is_type_alias
    }
    aliases = {}
    for name in sorted(declared):
        target, seen = declared[name], {name}
        while (
            isinstance(target, (PrimitiveType, NamedType))
            and not getattr(target, "generic_args", None)
            and target.name in declared
            and target.name not in seen
        ):
            seen.add(target.name)
            target = declared[target.name]
        if (
            isinstance(target, (PrimitiveType, NamedType))
            and not getattr(target, "generic_args", None)
            and getattr(target, "size_bits", None) is None
        ):
            if target.name in _NARROW_FLOAT_ELEMENTS:
                aliases[name] = target.name
            elif target.name in _BINARY16_SPELLINGS:
                aliases[name] = "float16"
    if not aliases:
        return
    declarations = {
        id(variable.var_type)
        for variable in ast.global_variables
        if isinstance(variable, VariableNode) and variable.is_type_alias
    }
    for node in ast.walk():
        if (
            isinstance(node, (PrimitiveType, NamedType))
            and id(node) not in declarations
            and not getattr(node, "generic_args", None)
            and node.name in aliases
        ):
            node.name = aliases[node.name]


def lower_resource_aggregates(
    ast, *, storage_pointer_parameters=False, workgroup_access_assertions=()
):
    """Lower a selected compute entry's private resource-reference aggregates."""
    if not isinstance(ast, ASTNode):
        return ast
    if not any(
        isinstance(node, StructNode)
        and (_contains_storage_pointer(node) or _contains_thread_pointer(node))
        for node in ast.walk()
    ):
        return ast
    stages = list(getattr(ast, "stages", {}).values())
    if len(stages) != 1 or _name(stages[0].stage).lower() != "compute":
        return ast
    result = deepcopy(ast)
    _resolve_narrow_float_aliases(result)
    entry = list(result.stages.values())[0].entry_point
    return _Lowering(
        result, entry, storage_pointer_parameters, workgroup_access_assertions
    ).run()
