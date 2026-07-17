"""Target-independent range contracts for source index normalization."""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


INDEX_RANGE_CONSTANT = "constant"
INDEX_RANGE_STATIC = "statically-bounded"
INDEX_RANGE_ASSERTED = "asserted"
INDEX_RANGE_UNPROVEN = "unproven"
INDEX_RANGE_OUT_OF_RANGE = "out-of-range"


@dataclass(frozen=True)
class IntegerRange:
    """An inclusive integer interval and the proof that established it."""

    minimum: int
    maximum: int
    status: str = INDEX_RANGE_STATIC
    provenance: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.minimum, bool) or isinstance(self.maximum, bool):
            raise ValueError("integer range bounds must be integers")
        if self.minimum > self.maximum:
            raise ValueError("integer range minimum must not exceed maximum")

    @classmethod
    def exact(cls, value: int, *, provenance: str | None = None) -> "IntegerRange":
        return cls(value, value, INDEX_RANGE_CONSTANT, provenance)

    @property
    def is_exact(self) -> bool:
        return self.minimum == self.maximum

    def is_within(self, other: "IntegerRange") -> bool:
        return self.minimum >= other.minimum and self.maximum <= other.maximum

    def with_status(self, status: str, provenance: str | None = None) -> "IntegerRange":
        return replace(
            self,
            status=status,
            provenance=self.provenance if provenance is None else provenance,
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "status": self.status,
        }
        if self.provenance:
            payload["provenance"] = self.provenance
        return payload


@dataclass(frozen=True)
class IndexScalarType:
    """One scalar index representation accepted by a target profile."""

    name: str
    signed: bool
    bits: int

    @property
    def value_range(self) -> IntegerRange:
        if self.signed:
            return IntegerRange(-(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1)
        return IntegerRange(0, (1 << self.bits) - 1)


@dataclass(frozen=True)
class IndexTargetProfile:
    """Legal scalar index types for a concrete target language profile."""

    name: str
    scalar_types: tuple[IndexScalarType, ...]

    def scalar_type(self, *, signed: bool) -> IndexScalarType | None:
        return next(
            (scalar for scalar in self.scalar_types if scalar.signed is signed),
            None,
        )


OPENGL_INDEX_PROFILE = IndexTargetProfile(
    "OpenGL desktop GLSL",
    (
        IndexScalarType("int", True, 32),
        IndexScalarType("uint", False, 32),
    ),
)

WEBGL_INDEX_PROFILE = IndexTargetProfile(
    "WebGL 2 / GLSL ES 3.00",
    (
        IndexScalarType("int", True, 32),
        IndexScalarType("uint", False, 32),
    ),
)


@dataclass(frozen=True)
class IndexRangeAssertion:
    """A project-supplied range assertion for one source expression."""

    expression: str
    value_range: IntegerRange
    source: str = "*"
    function: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise ValueError("index range assertion expression must be non-empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("index range assertion source must be non-empty")
        if self.function is not None and (
            not isinstance(self.function, str) or not self.function.strip()
        ):
            raise ValueError("index range assertion function must be non-empty")
        object.__setattr__(self, "expression", self.expression.strip())
        object.__setattr__(self, "source", self.source.strip())
        if self.function is not None:
            object.__setattr__(self, "function", self.function.strip())

    def applies_to(self, expression: str, function: str | None) -> bool:
        if _normalize_expression(expression) != _normalize_expression(self.expression):
            return False
        return self.function is None or self.function == function

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "expression": self.expression,
            "minimum": self.value_range.minimum,
            "maximum": self.value_range.maximum,
        }
        if self.function is not None:
            payload["function"] = self.function
        return payload


@dataclass(frozen=True)
class IndexNarrowingDecision:
    """The result of checking a source index against a target profile."""

    action: str
    target_type: IndexScalarType | None
    range_status: str
    reason: str | None
    source_range: IntegerRange | None
    accepted_range: IntegerRange | None

    @property
    def accepted(self) -> bool:
        return self.action in {"identity", "convert"}


def decide_index_narrowing(
    *,
    source_signed: bool,
    source_bits: int,
    source_width: int,
    profile: IndexTargetProfile,
    value_range: IntegerRange | None,
    indexed_extent: int | None = None,
) -> IndexNarrowingDecision:
    """Prove an identity or narrowing conversion without target-side checks."""

    if source_width != 1:
        return IndexNarrowingDecision(
            "reject",
            None,
            INDEX_RANGE_UNPROVEN,
            "vector-index-unsupported",
            value_range,
            None,
        )

    target_type = profile.scalar_type(signed=source_signed)
    if target_type is None:
        return IndexNarrowingDecision(
            "reject",
            None,
            INDEX_RANGE_UNPROVEN,
            "target-signedness-unsupported",
            value_range,
            None,
        )

    target_range = target_type.value_range
    maximum_index = target_range.maximum
    if indexed_extent is not None:
        maximum_index = min(maximum_index, indexed_extent - 1)
    accepted_range = (
        IntegerRange(0, maximum_index) if maximum_index >= 0 else None
    )

    # A source index that is already legal for the target needs no range proof.
    # Still reject an exact invalid subscript because constants are diagnosed
    # before target emission instead of relying on undefined target behavior.
    if source_bits <= target_type.bits:
        if value_range is not None and value_range.is_exact:
            if value_range.minimum < 0:
                return IndexNarrowingDecision(
                    "reject",
                    target_type,
                    INDEX_RANGE_OUT_OF_RANGE,
                    "negative-index",
                    value_range,
                    accepted_range,
                )
            if indexed_extent is not None and value_range.maximum >= indexed_extent:
                return IndexNarrowingDecision(
                    "reject",
                    target_type,
                    INDEX_RANGE_OUT_OF_RANGE,
                    "constant-index-out-of-range",
                    value_range,
                    accepted_range,
                )
        return IndexNarrowingDecision(
            "identity",
            target_type,
            value_range.status if value_range is not None else INDEX_RANGE_UNPROVEN,
            None,
            value_range,
            accepted_range,
        )

    if value_range is not None:
        if value_range.minimum < 0:
            return IndexNarrowingDecision(
                "reject",
                target_type,
                INDEX_RANGE_OUT_OF_RANGE,
                "negative-index",
                value_range,
                accepted_range,
            )
        if indexed_extent is not None and value_range.maximum >= indexed_extent:
            return IndexNarrowingDecision(
                "reject",
                target_type,
                INDEX_RANGE_OUT_OF_RANGE,
                (
                    "constant-index-out-of-range"
                    if value_range.is_exact
                    else "index-range-out-of-bounds"
                ),
                value_range,
                accepted_range,
            )
        if not value_range.is_within(target_range):
            return IndexNarrowingDecision(
                "reject",
                target_type,
                INDEX_RANGE_OUT_OF_RANGE,
                (
                    "constant-index-out-of-range"
                    if value_range.is_exact
                    else "index-range-out-of-target-range"
                ),
                value_range,
                accepted_range,
            )

    if value_range is None:
        return IndexNarrowingDecision(
            "reject",
            target_type,
            INDEX_RANGE_UNPROVEN,
            "index-range-unproven",
            None,
            accepted_range,
        )

    return IndexNarrowingDecision(
        "convert",
        target_type,
        value_range.status,
        None,
        value_range,
        accepted_range,
    )


def parse_index_range_assertions(
    value: Any,
    *,
    field_name: str = "index_range_assertions",
) -> tuple[IndexRangeAssertion, ...]:
    """Validate assertion records from API objects or project configuration."""

    if value is None:
        return ()
    if isinstance(value, IndexRangeAssertion):
        return (value,)
    if isinstance(value, Mapping):
        records: Sequence[Any] = [
            {"expression": expression, **dict(bounds)}
            if isinstance(bounds, Mapping)
            else bounds
            for expression, bounds in value.items()
        ]
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        records = value
    else:
        raise ValueError(f"{field_name} must be an array of assertion tables")

    assertions = []
    for index, record in enumerate(records):
        record_path = f"{field_name}[{index}]"
        if isinstance(record, IndexRangeAssertion):
            assertions.append(record)
            continue
        if not isinstance(record, Mapping):
            raise ValueError(f"{record_path} must be a table")
        expression = record.get("expression")
        source = record.get("source", "*")
        function = record.get("function")
        minimum = _range_bound(record, "minimum", "min", record_path)
        maximum = _range_bound(record, "maximum", "max", record_path)
        assertions.append(
            IndexRangeAssertion(
                expression=str(expression) if expression is not None else "",
                source=str(source) if source is not None else "",
                function=(str(function) if function is not None else None),
                value_range=IntegerRange(
                    minimum,
                    maximum,
                    INDEX_RANGE_ASSERTED,
                    "project-config",
                ),
            )
        )
    return tuple(assertions)


def _range_bound(
    record: Mapping[str, Any], primary: str, alias: str, field_name: str
) -> int:
    if primary in record and alias in record:
        raise ValueError(f"{field_name} must not define both {primary} and {alias}")
    if primary not in record and alias not in record:
        raise ValueError(f"{field_name}.{primary} is required")
    value = record.get(primary, record.get(alias))
    if isinstance(value, bool):
        raise ValueError(f"{field_name}.{primary} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field_name}.{primary} must be an integer") from exc


def _normalize_expression(expression: str) -> str:
    return "".join(str(expression).split())
