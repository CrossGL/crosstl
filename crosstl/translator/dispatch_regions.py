"""Partition exact thread grids without creating inactive invocations."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _dimensions(
    value: Sequence[int], name: str, *, zero: bool = False
) -> tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 3:
        raise ValueError(f"{name} must contain one to three integer dimensions")
    minimum = 0 if zero else 1
    if any(type(n) is not int or not minimum <= n <= 0xFFFFFFFF for n in value):
        raise ValueError(f"{name} dimensions must be uint32 values >= {minimum}")
    return tuple(value) + (minimum,) * (3 - len(value))


@dataclass(frozen=True)
class DispatchRegion:
    """A rectangular set of source groups with the same active group shape.

    Offsets are measured in source workgroups, not physical threads. Device
    limits and shared allocation ownership are checked by the runtime adapter.
    """

    thread_grid_size: tuple[int, int, int]
    source_workgroup_size: tuple[int, int, int]
    workgroup_offset: tuple[int, int, int]
    workgroup_count: tuple[int, int, int]
    workgroup_size: tuple[int, int, int]

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _dimensions(getattr(self, name), name, zero=name == "workgroup_offset"),
            )
        for grid, nominal, offset, count, active in zip(
            self.thread_grid_size,
            self.source_workgroup_size,
            self.workgroup_offset,
            self.workgroup_count,
            self.workgroup_size,
        ):
            groups = (grid + nominal - 1) // nominal
            if offset + count > groups:
                raise ValueError("Dispatch region exceeds the source workgroup grid")
            first = min(nominal, grid - offset * nominal)
            last = min(nominal, grid - (offset + count - 1) * nominal)
            if first != active or last != active:
                raise ValueError(
                    "Dispatch region must have a uniform active workgroup shape"
                )

    @property
    def thread_offset(self) -> tuple[int, int, int]:
        return tuple(
            g * n for g, n in zip(self.workgroup_offset, self.source_workgroup_size)
        )

    @property
    def source_workgroup_count(self) -> tuple[int, int, int]:
        return tuple(
            (n + w - 1) // w
            for n, w in zip(self.thread_grid_size, self.source_workgroup_size)
        )

    def to_json(self) -> dict[str, list[int]]:
        return {
            "threadGridSize": list(self.thread_grid_size),
            "sourceWorkgroupSize": list(self.source_workgroup_size),
            "workgroupOffset": list(self.workgroup_offset),
            "workgroupCount": list(self.workgroup_count),
            "workgroupSize": list(self.workgroup_size),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> DispatchRegion:
        fields = {
            "threadGridSize": "thread_grid_size",
            "sourceWorkgroupSize": "source_workgroup_size",
            "workgroupOffset": "workgroup_offset",
            "workgroupCount": "workgroup_count",
            "workgroupSize": "workgroup_size",
        }
        if not isinstance(value, Mapping) or set(value) != set(fields):
            raise ValueError(
                "Dispatch region must contain exactly " + ", ".join(sorted(fields))
            )
        return cls(**{name: value[key] for key, name in fields.items()})


def plan_dispatch_regions(
    thread_grid_size: Sequence[int], workgroup_size: Sequence[int]
) -> tuple[DispatchRegion, ...]:
    """Return at most eight non-overlapping full/edge regions, in X-major order.

    Each region launches only real source invocations, including on targets
    without nonuniform workgroups. This is a geometry plan, not a dispatch:
    kernels must also be specialized to retain source IDs and grid extents.
    """
    grid = _dimensions(thread_grid_size, "thread_grid_size")
    nominal = _dimensions(workgroup_size, "workgroup_size")
    axes = []
    for extent, width in zip(grid, nominal):
        full, tail = divmod(extent, width)
        axis = []
        if full:
            axis.append((0, full, width))
        if tail:
            axis.append((full, 1, tail))
        axes.append(axis)
    regions = []
    for parts in itertools.product(*reversed(axes)):
        x, y, z = reversed(parts)
        regions.append(
            DispatchRegion(
                grid,
                nominal,
                (x[0], y[0], z[0]),
                (x[1], y[1], z[1]),
                (x[2], y[2], z[2]),
            )
        )
    return tuple(regions)
