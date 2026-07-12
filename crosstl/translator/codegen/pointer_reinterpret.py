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


def validate_pointer_reinterpretation_target(ast, target):
    """Reject targets that cannot preserve the shared pointer-view contract."""

    if target in {"directx", "opengl", "vulkan"}:
        return

    from ..ast import PointerReinterpretNode

    seen = set()

    def walk(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return None
        value_id = id(value)
        if value_id in seen:
            return None
        seen.add(value_id)
        if isinstance(value, PointerReinterpretNode):
            return value
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set)):
            children = value
        elif hasattr(value, "__dict__"):
            children = vars(value).values()
        else:
            return None
        for child in children:
            found = walk(child)
            if found is not None:
                return found
        return None

    expression = walk(ast)
    if expression is None:
        return
    pointee_type = getattr(expression.target_type, "pointee_type", None)
    target_type = getattr(pointee_type, "name", str(pointee_type))
    raise PointerReinterpretationError(
        f"{target} does not implement storage-backed pointer reinterpretation",
        target_type=target_type,
        target_backend=target,
        reason="target-lowering-unavailable",
        source_location=getattr(expression, "source_location", None),
    )
