"""Shared contracts for storage-backed pointer reinterpretation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarStorageLayout:
    """Logical scalar layout retained across a pointer view change."""

    name: str
    kind: str
    bit_width: int
    signed: bool = False

    @property
    def byte_width(self):
        return self.bit_width // 8


_SCALAR_LAYOUTS = {
    "char": ScalarStorageLayout("int8", "integer", 8, True),
    "i8": ScalarStorageLayout("int8", "integer", 8, True),
    "int8": ScalarStorageLayout("int8", "integer", 8, True),
    "int8_t": ScalarStorageLayout("int8", "integer", 8, True),
    "uchar": ScalarStorageLayout("uint8", "integer", 8),
    "u8": ScalarStorageLayout("uint8", "integer", 8),
    "uint8": ScalarStorageLayout("uint8", "integer", 8),
    "uint8_t": ScalarStorageLayout("uint8", "integer", 8),
    "short": ScalarStorageLayout("int16", "integer", 16, True),
    "i16": ScalarStorageLayout("int16", "integer", 16, True),
    "int16": ScalarStorageLayout("int16", "integer", 16, True),
    "int16_t": ScalarStorageLayout("int16", "integer", 16, True),
    "ushort": ScalarStorageLayout("uint16", "integer", 16),
    "u16": ScalarStorageLayout("uint16", "integer", 16),
    "uint16": ScalarStorageLayout("uint16", "integer", 16),
    "uint16_t": ScalarStorageLayout("uint16", "integer", 16),
    "int": ScalarStorageLayout("int", "integer", 32, True),
    "i32": ScalarStorageLayout("int", "integer", 32, True),
    "int32": ScalarStorageLayout("int", "integer", 32, True),
    "int32_t": ScalarStorageLayout("int", "integer", 32, True),
    "uint": ScalarStorageLayout("uint", "integer", 32),
    "u32": ScalarStorageLayout("uint", "integer", 32),
    "uint32": ScalarStorageLayout("uint", "integer", 32),
    "uint32_t": ScalarStorageLayout("uint", "integer", 32),
    "float": ScalarStorageLayout("float", "floating", 32),
    "float32": ScalarStorageLayout("float", "floating", 32),
    "float32_t": ScalarStorageLayout("float", "floating", 32),
    "long": ScalarStorageLayout("int64", "integer", 64, True),
    "i64": ScalarStorageLayout("int64", "integer", 64, True),
    "int64": ScalarStorageLayout("int64", "integer", 64, True),
    "int64_t": ScalarStorageLayout("int64", "integer", 64, True),
    "ulong": ScalarStorageLayout("uint64", "integer", 64),
    "u64": ScalarStorageLayout("uint64", "integer", 64),
    "uint64": ScalarStorageLayout("uint64", "integer", 64),
    "uint64_t": ScalarStorageLayout("uint64", "integer", 64),
    "double": ScalarStorageLayout("double", "floating", 64),
}


def scalar_storage_layout(type_name):
    """Return the logical scalar layout for a source type spelling."""

    normalized = str(type_name or "").strip()
    while normalized.endswith(("*", "&")):
        normalized = normalized[:-1].strip()
    if normalized.startswith("metal::"):
        normalized = normalized[len("metal::") :]
    qualifiers = {
        "const",
        "constant",
        "device",
        "thread",
        "threadgroup",
        "volatile",
        "restrict",
    }
    parts = normalized.split()
    while parts and parts[0].lower() in qualifiers:
        parts.pop(0)
    return _SCALAR_LAYOUTS.get(" ".join(parts).lower())


class PointerReinterpretationError(ValueError):
    """Raised when a target cannot preserve a storage pointer view change."""

    project_diagnostic_code = "project.translate.pointer-reinterpret-unsupported"
    missing_capabilities = ("pointer.reinterpretation",)

    def __init__(
        self,
        message,
        *,
        source_type=None,
        target_type=None,
        address_space=None,
        alignment=None,
        access=None,
        target_backend=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.source_type = source_type
        self.target_type = target_type
        self.address_space = address_space
        self.alignment = alignment
        self.access = access
        self.target_backend = target_backend
        self.reason = reason
        self.source_location = source_location


METAL_LOCAL_SINGLE_FIELD_READ_ANNOTATION = "metalLocalSingleFieldReinterpretRead"


def metal_local_single_field_reinterpret_contract(expression):
    """Return the validated Metal-local value-view contract, if present."""

    annotations = getattr(expression, "annotations", None) or {}
    return annotations.get(METAL_LOCAL_SINGLE_FIELD_READ_ANNOTATION)


def _crossgl_type_identity(type_node):
    if type_node is None:
        return None
    name = getattr(type_node, "name", None)
    if name is not None:
        generic_args = tuple(
            _crossgl_type_identity(argument)
            for argument in getattr(type_node, "generic_args", ()) or ()
        )
        return (type(type_node).__name__, str(name), generic_args)
    element_type = getattr(type_node, "element_type", None)
    if element_type is not None:
        return (
            type(type_node).__name__,
            _crossgl_type_identity(element_type),
            getattr(type_node, "size", None),
        )
    return (type(type_node).__name__, str(type_node))


def _node_operator(node):
    return getattr(node, "operator", getattr(node, "op", None))


def _metal_local_single_field_read_contract(
    expression,
    parent,
    grandparent,
    function,
    structs_by_name,
):
    from ..ast import AssignmentNode, IdentifierNode, PointerType, UnaryOpNode

    if not (
        isinstance(parent, UnaryOpNode)
        and _node_operator(parent) == "*"
        and getattr(parent, "operand", None) is expression
    ):
        return None
    if (
        isinstance(grandparent, AssignmentNode)
        and getattr(grandparent, "target", None) is parent
    ):
        return None
    if isinstance(grandparent, UnaryOpNode) and _node_operator(grandparent) in {
        "&",
        "++",
        "--",
    }:
        return None

    target_pointer = getattr(expression, "target_type", None)
    if not isinstance(target_pointer, PointerType) or getattr(
        target_pointer, "address_space", None
    ) not in {None, "thread"}:
        return None
    pointee = getattr(target_pointer, "pointee_type", None)
    target_name = getattr(pointee, "name", None)
    target_struct = structs_by_name.get(target_name)
    if target_struct is None:
        return None
    if (
        getattr(target_struct, "generic_params", None)
        or getattr(target_struct, "attributes", None)
        or getattr(target_struct, "inheritance", None)
    ):
        return None
    members = list(getattr(target_struct, "members", ()) or ())
    if len(members) != 1:
        return None

    address = getattr(expression, "expression", None)
    if not isinstance(address, UnaryOpNode) or _node_operator(address) != "&":
        return None
    source = getattr(address, "operand", None)
    if not isinstance(source, IdentifierNode) or function is None:
        return None
    source_name = getattr(source, "name", None)
    parameters = [
        parameter
        for parameter in getattr(function, "parameters", ()) or ()
        if getattr(parameter, "name", None) == source_name
    ]
    if len(parameters) != 1:
        return None
    source_type = getattr(parameters[0], "param_type", None)
    member_type = getattr(members[0], "member_type", None)
    if _crossgl_type_identity(source_type) != _crossgl_type_identity(member_type):
        return None
    if isinstance(source_type, PointerType) or getattr(
        source_type, "address_space", None
    ):
        return None

    return {
        "sourceName": source_name,
        "sourceType": source_type,
        "targetType": pointee,
        "targetTypeName": target_name,
    }


def validate_pointer_reinterpretation_target(ast, target):
    """Reject targets that cannot preserve the shared pointer-view contract."""

    if target in {"directx", "opengl", "vulkan"}:
        return

    from ..ast import FunctionNode, PointerReinterpretNode

    structs_by_name = {
        getattr(struct, "name", None): struct
        for struct in getattr(ast, "structs", ()) or ()
        if getattr(struct, "name", None)
    }
    seen = set()
    unsupported = []

    def walk(value, parent=None, grandparent=None, function=None):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        if isinstance(value, FunctionNode):
            function = value
        if isinstance(value, PointerReinterpretNode):
            contract = None
            if target == "metal":
                contract = _metal_local_single_field_read_contract(
                    value,
                    parent,
                    grandparent,
                    function,
                    structs_by_name,
                )
            if contract is None:
                unsupported.append(value)
            else:
                value.annotations[METAL_LOCAL_SINGLE_FIELD_READ_ANNOTATION] = contract
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set)):
            children = value
        elif hasattr(value, "__dict__"):
            children = (
                child
                for key, child in vars(value).items()
                if key not in {"parent", "annotations"}
            )
        else:
            return
        for child in children:
            walk(child, value, parent, function)

    walk(ast)
    if not unsupported:
        return
    expression = unsupported[0]
    pointee_type = getattr(expression.target_type, "pointee_type", None)
    target_type = getattr(pointee_type, "name", str(pointee_type))
    raise PointerReinterpretationError(
        f"{target} does not implement storage-backed pointer reinterpretation",
        target_type=target_type,
        target_backend=target,
        reason="target-lowering-unavailable",
        source_location=getattr(expression, "source_location", None),
    )
