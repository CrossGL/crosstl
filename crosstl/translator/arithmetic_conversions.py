"""Shared source arithmetic conversion contracts.

This module models C-family integer promotions and usual arithmetic conversions.
Backends provide mapped scalar/vector types and target width capabilities, then
remain responsible for rendering the required conversions.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional, Tuple


class ArithmeticScalarKind(str, Enum):
    """Scalar categories used by source arithmetic conversion rules."""

    BOOLEAN = "boolean"
    SIGNED_INTEGER = "signed-integer"
    UNSIGNED_INTEGER = "unsigned-integer"
    FLOATING = "floating"


@dataclass(frozen=True)
class ArithmeticType:
    """One numeric source operand and its mapped target representation."""

    source_type: str
    target_type: str
    kind: ArithmeticScalarKind
    bits: int
    lanes: int = 1

    @property
    def is_integer(self) -> bool:
        return self.kind in {
            ArithmeticScalarKind.BOOLEAN,
            ArithmeticScalarKind.SIGNED_INTEGER,
            ArithmeticScalarKind.UNSIGNED_INTEGER,
        }


@dataclass(frozen=True)
class ArithmeticConversion:
    """Target operand types and result selected for one source operation."""

    left_target_type: str
    right_target_type: str
    common_type: str
    result_type: str


class UnrepresentableArithmeticConversion(ValueError):
    """Raised when recognized numeric operands have no faithful target form."""

    def __init__(self, reason: str, attempted_common_type: Optional[str]):
        super().__init__(reason)
        self.reason = reason
        self.attempted_common_type = attempted_common_type


_ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/"})
_INTEGRAL_OPERATORS = frozenset({"%", "&", "|", "^"})
_SHIFT_OPERATORS = frozenset({"<<", ">>"})
_RELATIONAL_OPERATORS = frozenset({"<", "<=", ">", ">=", "==", "!="})
_CONDITIONAL_OPERATORS = frozenset({"?:"})
SUPPORTED_ARITHMETIC_OPERATORS = (
    _ARITHMETIC_OPERATORS
    | _INTEGRAL_OPERATORS
    | _SHIFT_OPERATORS
    | _RELATIONAL_OPERATORS
    | _CONDITIONAL_OPERATORS
)

_TARGET_SCALAR_TYPES = {
    "bool": (ArithmeticScalarKind.BOOLEAN, 1),
    "int": (ArithmeticScalarKind.SIGNED_INTEGER, 32),
    "uint": (ArithmeticScalarKind.UNSIGNED_INTEGER, 32),
    "int64_t": (ArithmeticScalarKind.SIGNED_INTEGER, 64),
    "uint64_t": (ArithmeticScalarKind.UNSIGNED_INTEGER, 64),
    "float": (ArithmeticScalarKind.FLOATING, 32),
    "double": (ArithmeticScalarKind.FLOATING, 64),
}
_TARGET_VECTOR_TYPES = {
    "bvec": (ArithmeticScalarKind.BOOLEAN, 1),
    "ivec": (ArithmeticScalarKind.SIGNED_INTEGER, 32),
    "uvec": (ArithmeticScalarKind.UNSIGNED_INTEGER, 32),
    "i64vec": (ArithmeticScalarKind.SIGNED_INTEGER, 64),
    "u64vec": (ArithmeticScalarKind.UNSIGNED_INTEGER, 64),
    "vec": (ArithmeticScalarKind.FLOATING, 32),
    "dvec": (ArithmeticScalarKind.FLOATING, 64),
}

_SOURCE_INTEGER_TYPES = {
    "char": (ArithmeticScalarKind.SIGNED_INTEGER, 8),
    "signedchar": (ArithmeticScalarKind.SIGNED_INTEGER, 8),
    "i8": (ArithmeticScalarKind.SIGNED_INTEGER, 8),
    "int8": (ArithmeticScalarKind.SIGNED_INTEGER, 8),
    "int8_t": (ArithmeticScalarKind.SIGNED_INTEGER, 8),
    "uchar": (ArithmeticScalarKind.UNSIGNED_INTEGER, 8),
    "unsignedchar": (ArithmeticScalarKind.UNSIGNED_INTEGER, 8),
    "u8": (ArithmeticScalarKind.UNSIGNED_INTEGER, 8),
    "uint8": (ArithmeticScalarKind.UNSIGNED_INTEGER, 8),
    "uint8_t": (ArithmeticScalarKind.UNSIGNED_INTEGER, 8),
    "short": (ArithmeticScalarKind.SIGNED_INTEGER, 16),
    "signedshort": (ArithmeticScalarKind.SIGNED_INTEGER, 16),
    "i16": (ArithmeticScalarKind.SIGNED_INTEGER, 16),
    "int16": (ArithmeticScalarKind.SIGNED_INTEGER, 16),
    "int16_t": (ArithmeticScalarKind.SIGNED_INTEGER, 16),
    "ushort": (ArithmeticScalarKind.UNSIGNED_INTEGER, 16),
    "unsignedshort": (ArithmeticScalarKind.UNSIGNED_INTEGER, 16),
    "u16": (ArithmeticScalarKind.UNSIGNED_INTEGER, 16),
    "uint16": (ArithmeticScalarKind.UNSIGNED_INTEGER, 16),
    "uint16_t": (ArithmeticScalarKind.UNSIGNED_INTEGER, 16),
    "int": (ArithmeticScalarKind.SIGNED_INTEGER, 32),
    "signedint": (ArithmeticScalarKind.SIGNED_INTEGER, 32),
    "i32": (ArithmeticScalarKind.SIGNED_INTEGER, 32),
    "int32": (ArithmeticScalarKind.SIGNED_INTEGER, 32),
    "int32_t": (ArithmeticScalarKind.SIGNED_INTEGER, 32),
    "uint": (ArithmeticScalarKind.UNSIGNED_INTEGER, 32),
    "unsignedint": (ArithmeticScalarKind.UNSIGNED_INTEGER, 32),
    "u32": (ArithmeticScalarKind.UNSIGNED_INTEGER, 32),
    "uint32": (ArithmeticScalarKind.UNSIGNED_INTEGER, 32),
    "uint32_t": (ArithmeticScalarKind.UNSIGNED_INTEGER, 32),
    "i64": (ArithmeticScalarKind.SIGNED_INTEGER, 64),
    "int64": (ArithmeticScalarKind.SIGNED_INTEGER, 64),
    "int64_t": (ArithmeticScalarKind.SIGNED_INTEGER, 64),
    "u64": (ArithmeticScalarKind.UNSIGNED_INTEGER, 64),
    "uint64": (ArithmeticScalarKind.UNSIGNED_INTEGER, 64),
    "uint64_t": (ArithmeticScalarKind.UNSIGNED_INTEGER, 64),
}
_SOURCE_VECTOR_COMPONENTS = {
    "i8vec": "int8_t",
    "u8vec": "uint8_t",
    "i16vec": "int16_t",
    "u16vec": "uint16_t",
    "i64vec": "int64_t",
    "u64vec": "uint64_t",
}


def _normalized_source_type(type_name: str) -> str:
    normalized = re.sub(
        r"\b(?:const|volatile|restrict|device|constant|thread|threadgroup)\b",
        "",
        str(type_name or ""),
    )
    return re.sub(r"\s+", "", normalized).lower().rstrip("&")


def source_integer_shape(
    type_name: str,
) -> Optional[Tuple[ArithmeticScalarKind, int, int]]:
    """Return ``(kind, bits, lanes)`` for a recognized source integer type."""

    compact = _normalized_source_type(type_name)
    generic = re.fullmatch(
        r"(?:[a-z_][a-z0-9_:]*(?:vec|vector)|vec|vector)<([^,<>]+),([234])>",
        compact,
    )
    if generic is not None:
        component_name, lanes_text = generic.groups()
        component = _SOURCE_INTEGER_TYPES.get(component_name.rsplit("::", 1)[-1])
        if component is not None:
            return component[0], component[1], int(lanes_text)

    unqualified = compact.rsplit("::", 1)[-1]
    for prefix in ("packed_", "simd_"):
        if unqualified.startswith(prefix):
            unqualified = unqualified[len(prefix) :]
            break

    scalar = _SOURCE_INTEGER_TYPES.get(unqualified)
    if scalar is not None:
        return scalar[0], scalar[1], 1

    vector = re.fullmatch(r"(.+)([234])", unqualified)
    if vector is None:
        return None
    component_name, lanes_text = vector.groups()
    component_name = _SOURCE_VECTOR_COMPONENTS.get(component_name, component_name)
    component = _SOURCE_INTEGER_TYPES.get(component_name)
    if component is None:
        return None
    return component[0], component[1], int(lanes_text)


def narrow_integer_shape(
    type_name: str,
) -> Optional[Tuple[ArithmeticScalarKind, int, int]]:
    """Return source integer metadata when integer promotion is required."""

    shape = source_integer_shape(type_name)
    if shape is None or shape[1] >= 32:
        return None
    return shape


def target_arithmetic_type(type_name: str) -> Optional[ArithmeticType]:
    """Parse a canonical GLSL-family numeric scalar or vector type."""

    target_type = str(type_name or "").strip()
    scalar = _TARGET_SCALAR_TYPES.get(target_type)
    if scalar is not None:
        return ArithmeticType(target_type, target_type, scalar[0], scalar[1])

    vector = re.fullmatch(
        r"(bvec|ivec|uvec|i64vec|u64vec|vec|dvec)([234])", target_type
    )
    if vector is None:
        return None
    prefix, lanes_text = vector.groups()
    kind, bits = _TARGET_VECTOR_TYPES[prefix]
    return ArithmeticType(target_type, target_type, kind, bits, int(lanes_text))


def arithmetic_type(source_type: str, target_type: str) -> Optional[ArithmeticType]:
    """Combine source width metadata with a backend's mapped numeric type."""

    target = target_arithmetic_type(target_type)
    if target is None:
        return None
    source_integer = source_integer_shape(source_type)
    if (
        source_integer is not None
        and target.kind
        in {
            ArithmeticScalarKind.SIGNED_INTEGER,
            ArithmeticScalarKind.UNSIGNED_INTEGER,
        }
        and source_integer[0] == target.kind
        and source_integer[2] == target.lanes
    ):
        return ArithmeticType(
            str(source_type),
            target.target_type,
            source_integer[0],
            source_integer[1],
            source_integer[2],
        )
    return ArithmeticType(
        str(source_type),
        target.target_type,
        target.kind,
        target.bits,
        target.lanes,
    )


def arithmetic_type_name(kind: ArithmeticScalarKind, bits: int, lanes: int = 1) -> str:
    """Render a canonical GLSL-family type for semantic arithmetic metadata."""

    scalar_name = {
        (ArithmeticScalarKind.BOOLEAN, 1): "bool",
        (ArithmeticScalarKind.SIGNED_INTEGER, 32): "int",
        (ArithmeticScalarKind.UNSIGNED_INTEGER, 32): "uint",
        (ArithmeticScalarKind.SIGNED_INTEGER, 64): "int64_t",
        (ArithmeticScalarKind.UNSIGNED_INTEGER, 64): "uint64_t",
        (ArithmeticScalarKind.FLOATING, 32): "float",
        (ArithmeticScalarKind.FLOATING, 64): "double",
    }.get((kind, bits))
    if scalar_name is None:
        raise ValueError(
            f"Unsupported arithmetic scalar kind/width: {kind.value}/{bits}"
        )
    if lanes == 1:
        return scalar_name
    vector_prefix = {
        "bool": "bvec",
        "int": "ivec",
        "uint": "uvec",
        "int64_t": "i64vec",
        "uint64_t": "u64vec",
        "float": "vec",
        "double": "dvec",
    }[scalar_name]
    return f"{vector_prefix}{lanes}"


def promoted_integer_type_name(operand: ArithmeticType) -> Optional[str]:
    """Return the source integer-promotion result for one operand."""

    promoted = _promoted_integer(operand)
    if promoted is None:
        return None
    return arithmetic_type_name(promoted[0], promoted[1], operand.lanes)


def _promoted_integer(
    operand: ArithmeticType,
) -> Optional[Tuple[ArithmeticScalarKind, int]]:
    if not operand.is_integer:
        return None
    if operand.kind is ArithmeticScalarKind.BOOLEAN or operand.bits < 32:
        return ArithmeticScalarKind.SIGNED_INTEGER, 32
    return operand.kind, operand.bits


def _common_integer_component(
    left: ArithmeticType, right: ArithmeticType
) -> Optional[Tuple[ArithmeticScalarKind, int]]:
    left_promoted = _promoted_integer(left)
    right_promoted = _promoted_integer(right)
    if left_promoted is None or right_promoted is None:
        return None
    if left_promoted == right_promoted:
        return left_promoted

    left_kind, left_bits = left_promoted
    right_kind, right_bits = right_promoted
    if left_kind == right_kind:
        return left_promoted if left_bits >= right_bits else right_promoted

    if left_kind is ArithmeticScalarKind.SIGNED_INTEGER:
        signed_bits, unsigned_bits = left_bits, right_bits
    else:
        signed_bits, unsigned_bits = right_bits, left_bits
    if unsigned_bits >= signed_bits:
        return ArithmeticScalarKind.UNSIGNED_INTEGER, unsigned_bits
    return ArithmeticScalarKind.SIGNED_INTEGER, signed_bits


def _common_component(
    left: ArithmeticType, right: ArithmeticType, operator: str
) -> Optional[Tuple[ArithmeticScalarKind, int]]:
    if (
        operator in {"==", "!=", "?:"}
        and left.kind is ArithmeticScalarKind.BOOLEAN
        and right.kind is ArithmeticScalarKind.BOOLEAN
    ):
        return ArithmeticScalarKind.BOOLEAN, 1
    if (
        left.kind is ArithmeticScalarKind.FLOATING
        or right.kind is ArithmeticScalarKind.FLOATING
    ):
        floating_bits = max(
            operand.bits
            for operand in (left, right)
            if operand.kind is ArithmeticScalarKind.FLOATING
        )
        return ArithmeticScalarKind.FLOATING, floating_bits
    return _common_integer_component(left, right)


def _required_target_width(operand: ArithmeticType) -> Optional[Tuple[str, int]]:
    if operand.kind is ArithmeticScalarKind.BOOLEAN:
        return None
    if operand.kind is ArithmeticScalarKind.FLOATING:
        return "floating", operand.bits
    return "integer", max(32, operand.bits)


def _validate_target_widths(
    operands: Tuple[ArithmeticType, ...],
    supported_integer_widths: FrozenSet[int],
    supported_floating_widths: FrozenSet[int],
    attempted_common_type: str,
) -> None:
    for operand in operands:
        _validate_required_target_width(
            _required_target_width(operand),
            supported_integer_widths,
            supported_floating_widths,
            attempted_common_type,
        )


def _validate_required_target_width(
    required: Optional[Tuple[str, int]],
    supported_integer_widths: FrozenSet[int],
    supported_floating_widths: FrozenSet[int],
    attempted_common_type: str,
) -> None:
    if required is None:
        return
    category, bits = required
    if category == "integer" and bits not in supported_integer_widths:
        raise UnrepresentableArithmeticConversion(
            "target-integer-width-unsupported", attempted_common_type
        )
    if category == "floating" and bits not in supported_floating_widths:
        raise UnrepresentableArithmeticConversion(
            "target-floating-width-unsupported", attempted_common_type
        )


def resolve_arithmetic_conversion(
    left: ArithmeticType,
    right: ArithmeticType,
    operator: str,
    *,
    supported_integer_widths: FrozenSet[int],
    supported_floating_widths: FrozenSet[int],
) -> Optional[ArithmeticConversion]:
    """Resolve operand conversions or reject a recognized unsupported operation."""

    if operator not in SUPPORTED_ARITHMETIC_OPERATORS:
        return None

    if operator in _SHIFT_OPERATORS:
        left_promoted = _promoted_integer(left)
        right_promoted = _promoted_integer(right)
        attempted = (
            arithmetic_type_name(left_promoted[0], left_promoted[1], left.lanes)
            if left_promoted is not None
            else None
        )
        if left_promoted is None or right_promoted is None:
            raise UnrepresentableArithmeticConversion(
                "integer-operands-required", attempted
            )
        if left.lanes == 1 and right.lanes > 1:
            raise UnrepresentableArithmeticConversion(
                "scalar-left-vector-shift-unsupported", attempted
            )
        if left.lanes > 1 and right.lanes > 1 and left.lanes != right.lanes:
            raise UnrepresentableArithmeticConversion(
                "vector-width-mismatch", attempted
            )
        left_target = arithmetic_type_name(
            left_promoted[0], left_promoted[1], left.lanes
        )
        right_target = arithmetic_type_name(
            right_promoted[0], right_promoted[1], right.lanes
        )
        _validate_target_widths(
            (left, right),
            supported_integer_widths,
            supported_floating_widths,
            left_target,
        )
        for _promoted_kind, promoted_bits in (left_promoted, right_promoted):
            _validate_required_target_width(
                ("integer", promoted_bits),
                supported_integer_widths,
                supported_floating_widths,
                left_target,
            )
        return ArithmeticConversion(
            left_target,
            right_target,
            left_target,
            left_target,
        )

    common_component = _common_component(left, right, operator)
    attempted = (
        arithmetic_type_name(common_component[0], common_component[1])
        if common_component is not None
        else None
    )
    if common_component is None:
        raise UnrepresentableArithmeticConversion(
            "numeric-common-type-unavailable", attempted
        )
    if operator in _INTEGRAL_OPERATORS and (
        not left.is_integer or not right.is_integer
    ):
        raise UnrepresentableArithmeticConversion(
            "integer-operands-required", attempted
        )
    if left.lanes > 1 and right.lanes > 1 and left.lanes != right.lanes:
        raise UnrepresentableArithmeticConversion("vector-width-mismatch", attempted)

    common_lanes = max(left.lanes, right.lanes)
    common_type = arithmetic_type_name(
        common_component[0], common_component[1], common_lanes
    )
    _validate_target_widths(
        (left, right),
        supported_integer_widths,
        supported_floating_widths,
        common_type,
    )
    common_required_width = (
        None
        if common_component[0] is ArithmeticScalarKind.BOOLEAN
        else (
            (
                "floating"
                if common_component[0] is ArithmeticScalarKind.FLOATING
                else "integer"
            ),
            common_component[1],
        )
    )
    _validate_required_target_width(
        common_required_width,
        supported_integer_widths,
        supported_floating_widths,
        common_type,
    )

    require_common_shape = operator in _RELATIONAL_OPERATORS | _CONDITIONAL_OPERATORS
    left_lanes = common_lanes if require_common_shape else left.lanes
    right_lanes = common_lanes if require_common_shape else right.lanes
    left_target = arithmetic_type_name(
        common_component[0], common_component[1], left_lanes
    )
    right_target = arithmetic_type_name(
        common_component[0], common_component[1], right_lanes
    )
    if operator in _RELATIONAL_OPERATORS:
        result_type = arithmetic_type_name(
            ArithmeticScalarKind.BOOLEAN, 1, common_lanes
        )
    else:
        result_type = common_type
    return ArithmeticConversion(
        left_target,
        right_target,
        common_type,
        result_type,
    )
