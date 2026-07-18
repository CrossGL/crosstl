"""Coordinate mappings for lane-distributed cooperative-matrix fragments."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

FragmentCoordinate = Tuple[int, int]
LaneCoordinates = Tuple[FragmentCoordinate, ...]
FragmentCoordinates = Tuple[LaneCoordinates, ...]
FragmentOwner = Tuple[int, int]
FragmentMappingKey = Tuple[str, int, int, int, int]


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _normalize_lookup_name(name: object) -> Optional[str]:
    if not isinstance(name, str):
        return None
    normalized = name.strip()
    return normalized or None


@dataclass(frozen=True)
class CooperativeMatrixFragmentMapping:
    """An exact mapping from lane-local elements to logical matrix coordinates."""

    name: str
    rows: int
    columns: int
    subgroup_size: int
    lane_coordinates: FragmentCoordinates
    description: str = ""
    _owners: Tuple[FragmentOwner, ...] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise ValueError(
                "Cooperative-matrix fragment mapping name must be a string"
            )
        name = self.name.strip()
        if not name:
            raise ValueError(
                "Cooperative-matrix fragment mapping name must be non-empty"
            )
        object.__setattr__(self, "name", name)

        for field_name in ("rows", "columns", "subgroup_size"):
            value = getattr(self, field_name)
            if not _is_positive_integer(value):
                raise ValueError(
                    f"Cooperative-matrix fragment mapping {field_name} must be "
                    f"a positive integer, got {value!r}"
                )

        if not isinstance(self.description, str):
            raise ValueError(
                "Cooperative-matrix fragment mapping description must be a string"
            )

        if not isinstance(self.lane_coordinates, (list, tuple)):
            raise ValueError(
                f"Cooperative-matrix fragment mapping '{name}' lane_coordinates "
                "must be a sequence"
            )

        normalized_lanes = []
        for lane_index, lane in enumerate(self.lane_coordinates):
            if not isinstance(lane, (list, tuple)):
                raise ValueError(
                    f"Cooperative-matrix fragment mapping '{name}' lane "
                    f"{lane_index} must be a coordinate sequence"
                )
            normalized_lane = []
            for element_index, coordinate in enumerate(lane):
                if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                    raise ValueError(
                        f"Cooperative-matrix fragment mapping '{name}' lane "
                        f"{lane_index} element {element_index} must be a "
                        "(row, column) pair"
                    )
                row, column = coordinate
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (row, column)
                ):
                    raise ValueError(
                        f"Cooperative-matrix fragment mapping '{name}' lane "
                        f"{lane_index} element {element_index} coordinates must "
                        "be integers"
                    )
                normalized_lane.append((row, column))
            normalized_lanes.append(tuple(normalized_lane))
        coordinates = tuple(normalized_lanes)
        object.__setattr__(self, "lane_coordinates", coordinates)

        if len(coordinates) != self.subgroup_size:
            raise ValueError(
                f"Cooperative-matrix fragment mapping '{name}' declares "
                f"subgroup_size={self.subgroup_size}, but provides "
                f"{len(coordinates)} lanes"
            )
        if not coordinates or not coordinates[0]:
            raise ValueError(
                f"Cooperative-matrix fragment mapping '{name}' must provide "
                "at least one coordinate per lane"
            )

        elements_per_lane = len(coordinates[0])
        if any(len(lane) != elements_per_lane for lane in coordinates):
            raise ValueError(
                f"Cooperative-matrix fragment mapping '{name}' must provide the "
                "same number of coordinates for every lane"
            )

        flattened = [coordinate for lane in coordinates for coordinate in lane]
        expected_count = self.rows * self.columns
        if len(flattened) != expected_count:
            raise ValueError(
                f"Cooperative-matrix fragment mapping '{name}' provides "
                f"{len(flattened)} coordinates for a {self.rows}x{self.columns} "
                f"matrix with {expected_count} elements"
            )

        out_of_range = sorted(
            {
                coordinate
                for coordinate in flattened
                if not (
                    0 <= coordinate[0] < self.rows and 0 <= coordinate[1] < self.columns
                )
            }
        )
        if out_of_range:
            raise ValueError(
                f"Cooperative-matrix fragment mapping '{name}' contains "
                f"out-of-range coordinates: {out_of_range!r}"
            )

        coordinate_counts = Counter(flattened)
        duplicates = sorted(
            coordinate for coordinate, count in coordinate_counts.items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"Cooperative-matrix fragment mapping '{name}' contains "
                f"duplicate coordinates: {duplicates!r}"
            )

        unique_coordinates = set(coordinate_counts)
        expected_coordinates = {
            (row, column) for row in range(self.rows) for column in range(self.columns)
        }
        missing = sorted(expected_coordinates - unique_coordinates)
        if missing:
            raise ValueError(
                f"Cooperative-matrix fragment mapping '{name}' omits "
                f"coordinates: {missing!r}"
            )

        owners = [(-1, -1)] * expected_count
        for lane_index, lane in enumerate(coordinates):
            for element_index, (row, column) in enumerate(lane):
                owners[row * self.columns + column] = (lane_index, element_index)
        object.__setattr__(
            self,
            "_owners",
            tuple(owners),
        )

    @property
    def elements_per_lane(self) -> int:
        return len(self.lane_coordinates[0])

    def coordinate(self, lane: int, element: int) -> FragmentCoordinate:
        """Return the logical coordinate owned by one lane-local element."""
        if isinstance(lane, bool) or not isinstance(lane, int):
            raise TypeError("lane must be an integer")
        if isinstance(element, bool) or not isinstance(element, int):
            raise TypeError("element must be an integer")
        if lane < 0 or lane >= self.subgroup_size:
            raise IndexError(f"lane index out of range: {lane}")
        if element < 0 or element >= self.elements_per_lane:
            raise IndexError(f"element index out of range: {element}")
        return self.lane_coordinates[lane][element]

    def owner(self, row: int, column: int) -> Tuple[int, int]:
        """Return the lane and lane-local index for a logical coordinate."""
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (row, column)
        ):
            raise TypeError("row and column must be integers")
        coordinate = (row, column)
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise KeyError(coordinate)
        return self._owners[row * self.columns + column]


_FRAGMENT_MAPPINGS: Dict[FragmentMappingKey, CooperativeMatrixFragmentMapping] = {}


def register_cooperative_matrix_fragment_mapping(
    mapping: CooperativeMatrixFragmentMapping,
    *,
    overwrite: bool = False,
) -> CooperativeMatrixFragmentMapping:
    """Register an exact mapping contract for parser and target lookup."""
    if not isinstance(mapping, CooperativeMatrixFragmentMapping):
        raise TypeError("mapping must be a CooperativeMatrixFragmentMapping instance")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    key = (
        mapping.name,
        mapping.rows,
        mapping.columns,
        mapping.subgroup_size,
        mapping.elements_per_lane,
    )
    if key in _FRAGMENT_MAPPINGS and not overwrite:
        raise ValueError(
            "Cooperative-matrix fragment mapping is already registered for "
            f"{mapping.name!r} with contract "
            f"{mapping.rows}x{mapping.columns}/"
            f"{mapping.subgroup_size}x{mapping.elements_per_lane}"
        )
    _FRAGMENT_MAPPINGS[key] = mapping
    return mapping


def cooperative_matrix_fragment_mapping_names() -> Tuple[str, ...]:
    """Return registered mapping profile names in stable order."""
    return tuple(sorted({key[0] for key in _FRAGMENT_MAPPINGS}))


def has_cooperative_matrix_fragment_mapping(name: str) -> bool:
    """Return whether any exact contract is registered under ``name``."""
    normalized_name = _normalize_lookup_name(name)
    if normalized_name is None:
        return False
    return any(key[0] == normalized_name for key in _FRAGMENT_MAPPINGS)


def get_cooperative_matrix_fragment_mapping(
    name: str,
    rows: int,
    columns: int,
    subgroup_size: int,
    elements_per_lane: int,
) -> Optional[CooperativeMatrixFragmentMapping]:
    """Resolve an exact mapping contract without applying target defaults."""
    normalized_name = _normalize_lookup_name(name)
    dimensions = (rows, columns, subgroup_size, elements_per_lane)
    if normalized_name is None or not all(
        _is_positive_integer(value) for value in dimensions
    ):
        return None
    return _FRAGMENT_MAPPINGS.get(
        (
            normalized_name,
            *dimensions,
        )
    )


def _tile_4x4_row_pair_coordinates() -> Iterable[LaneCoordinates]:
    for lane in range(32):
        qid = lane // 4
        fm = (qid & 4) + ((lane // 2) % 4)
        fn = (qid & 2) * 2 + (lane % 2) * 2
        yield ((fm, fn), (fm, fn + 1))


register_cooperative_matrix_fragment_mapping(
    CooperativeMatrixFragmentMapping(
        name="tile_4x4_row_pair",
        rows=8,
        columns=8,
        subgroup_size=32,
        lane_coordinates=tuple(_tile_4x4_row_pair_coordinates()),
        description=("Row-pair ownership within row-major 4x4 tiles of an 8x8 matrix"),
    )
)
