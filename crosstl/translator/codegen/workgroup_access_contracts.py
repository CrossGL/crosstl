"""Project contracts for workgroup-storage access ranges."""

from __future__ import annotations

import fnmatch
import operator
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class WorkgroupAccessAssertion:
    """An asserted absolute element range for one workgroup pointer use."""

    entry_point: str
    minimum: int
    maximum: int
    source: str = "*"
    function: str = "*"
    parameter: str = "*"

    def __post_init__(self) -> None:
        for field_name in ("entry_point", "source", "function", "parameter"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"workgroup access assertion {field_name} must be non-empty"
                )
            object.__setattr__(self, field_name, value.strip())
        if (
            isinstance(self.minimum, bool)
            or isinstance(self.maximum, bool)
            or not isinstance(self.minimum, int)
            or not isinstance(self.maximum, int)
        ):
            raise ValueError("workgroup access assertion bounds must be integers")
        if self.minimum > self.maximum:
            raise ValueError(
                "workgroup access assertion minimum must not exceed maximum"
            )

    def applies_to(
        self,
        entry_point: str | None,
        function: str | None,
        parameter: str | None,
    ) -> bool:
        if not entry_point or not function or not parameter:
            return False
        return (
            fnmatch.fnmatchcase(entry_point, self.entry_point)
            and fnmatch.fnmatchcase(function, self.function)
            and fnmatch.fnmatchcase(parameter, self.parameter)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "entryPoint": self.entry_point,
            "function": self.function,
            "parameter": self.parameter,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def parse_workgroup_access_assertions(
    value: Any,
    *,
    field_name: str = "workgroup_access_assertions",
) -> tuple[WorkgroupAccessAssertion, ...]:
    """Validate workgroup access assertions from API or project records."""

    if value is None:
        return ()
    if isinstance(value, WorkgroupAccessAssertion):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be an array of assertion tables")

    assertions = []
    for index, record in enumerate(value):
        record_path = f"{field_name}[{index}]"
        if isinstance(record, WorkgroupAccessAssertion):
            assertions.append(record)
            continue
        if not isinstance(record, Mapping):
            raise ValueError(f"{record_path} must be a table")
        entry_point = _selector(
            record,
            "entry_point",
            "entryPoint",
            record_path,
            required=True,
        )
        assertions.append(
            WorkgroupAccessAssertion(
                source=_selector(record, "source", None, record_path, default="*"),
                entry_point=entry_point,
                function=_selector(
                    record,
                    "function",
                    None,
                    record_path,
                    default="*",
                ),
                parameter=_selector(
                    record,
                    "parameter",
                    None,
                    record_path,
                    default="*",
                ),
                minimum=_bound(record, "minimum", "min", record_path),
                maximum=_bound(record, "maximum", "max", record_path),
            )
        )
    return tuple(assertions)


def _selector(
    record: Mapping[str, Any],
    primary: str,
    alias: str | None,
    field_name: str,
    *,
    required: bool = False,
    default: str | None = None,
) -> str:
    if alias is not None and primary in record and alias in record:
        raise ValueError(f"{field_name} must not define both {primary} and {alias}")
    value = record.get(primary, record.get(alias) if alias is not None else default)
    if value is None and required:
        raise ValueError(f"{field_name}.{primary} is required")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}.{primary} must be a non-empty string")
    return value


def _bound(
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
