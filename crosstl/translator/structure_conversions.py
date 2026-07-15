"""Shared semantic contracts for structure-backed value conversions.

The registry in this module describes source-language conversion semantics only.
Backends remain responsible for rendering target syntax, but they must consume the
same destination shape and field-initialization plan instead of inferring a
one-argument structure constructor independently.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Sequence, Tuple


class ScalarKind(str, Enum):
    """Backend-neutral scalar categories accepted by conversion contracts."""

    BOOLEAN = "boolean"
    SIGNED_INTEGER = "signed-integer"
    UNSIGNED_INTEGER = "unsigned-integer"
    FLOATING = "floating"


class StructureConversionKind(str, Enum):
    """Stable conversion kinds shared by lowering and diagnostics."""

    COPY = "copy"
    SCALAR_PROMOTION = "scalar-promotion"
    CONTEXTUAL_SCALAR_CONVERSION = "contextual-scalar-conversion"
    DEFAULT_CONSTRUCTION = "default-construction"
    STRUCTURE_CONVERSION = "structure-conversion"
    VALUE_CONVERSION = "value-conversion"
    UNKNOWN = "unknown"


class StructureFieldValue(str, Enum):
    """Value source for one destination field."""

    CONVERTED_SOURCE = "converted-source"
    ZERO = "zero"


@dataclass(frozen=True)
class StructureConversionField:
    """One destination field and its scalar/default initialization semantics."""

    name: str
    type_name: str
    scalar_value: StructureFieldValue
    default_value: Optional[StructureFieldValue] = None


@dataclass(frozen=True)
class ScalarToStructureConversion:
    """Registered scalar conversion for one structure-backed value type."""

    destination_type: str
    source_kinds: FrozenSet[ScalarKind]
    fields: Tuple[StructureConversionField, ...]

    def __post_init__(self):
        if not self.destination_type:
            raise ValueError("Structure conversion destination type cannot be empty")
        if not self.source_kinds:
            raise ValueError(
                f"Structure conversion '{self.destination_type}' needs source kinds"
            )
        if not self.fields:
            raise ValueError(
                f"Structure conversion '{self.destination_type}' needs fields"
            )
        field_names = [field.name for field in self.fields]
        if any(not name for name in field_names) or len(set(field_names)) != len(
            field_names
        ):
            raise ValueError(
                f"Structure conversion '{self.destination_type}' has invalid fields"
            )
        if any(not field.type_name for field in self.fields):
            raise ValueError(
                f"Structure conversion '{self.destination_type}' has untyped fields"
            )
        if self.scalar_source_use_count == 0:
            raise ValueError(
                f"Structure conversion '{self.destination_type}' does not use its source"
            )

    @property
    def destination_shape(self) -> Tuple[Tuple[str, str], ...]:
        """Return the required ordered destination fields."""
        return tuple((field.name, field.type_name) for field in self.fields)

    @property
    def scalar_source_use_count(self) -> int:
        """Return how many output fields consume the scalar expression."""
        return sum(
            field.scalar_value is StructureFieldValue.CONVERTED_SOURCE
            for field in self.fields
        )

    @property
    def supports_default_construction(self) -> bool:
        """Return whether every field has registered default semantics."""
        return all(field.default_value is not None for field in self.fields)

    def matches_destination_shape(self, fields: Sequence[Tuple[str, str]]) -> bool:
        """Return whether ordered ``(name, type)`` fields match this contract."""
        return tuple(fields) == self.destination_shape


COMPLEX64_SCALAR_CONVERSION = ScalarToStructureConversion(
    destination_type="complex64_t",
    source_kinds=frozenset(
        {
            ScalarKind.BOOLEAN,
            ScalarKind.SIGNED_INTEGER,
            ScalarKind.UNSIGNED_INTEGER,
            ScalarKind.FLOATING,
        }
    ),
    fields=(
        StructureConversionField(
            name="real",
            type_name="float",
            scalar_value=StructureFieldValue.CONVERTED_SOURCE,
            default_value=StructureFieldValue.ZERO,
        ),
        StructureConversionField(
            name="imag",
            type_name="float",
            scalar_value=StructureFieldValue.ZERO,
            default_value=StructureFieldValue.ZERO,
        ),
    ),
)


REGISTERED_SCALAR_TO_STRUCTURE_CONVERSIONS: Mapping[
    str, ScalarToStructureConversion
] = MappingProxyType(
    {COMPLEX64_SCALAR_CONVERSION.destination_type: COMPLEX64_SCALAR_CONVERSION}
)


def registered_scalar_to_structure_conversion(
    destination_type: str,
) -> Optional[ScalarToStructureConversion]:
    """Return the exact registered contract for ``destination_type``."""
    return REGISTERED_SCALAR_TO_STRUCTURE_CONVERSIONS.get(destination_type)
