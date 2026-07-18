import pytest

from crosstl.translator.cooperative_matrix import (
    CooperativeMatrixFragmentMapping,
    cooperative_matrix_fragment_mapping_names,
    get_cooperative_matrix_fragment_mapping,
    has_cooperative_matrix_fragment_mapping,
    register_cooperative_matrix_fragment_mapping,
)


def test_tile_4x4_row_pair_mapping_matches_exact_lane_coordinates():
    mapping = get_cooperative_matrix_fragment_mapping(
        "tile_4x4_row_pair",
        8,
        8,
        32,
        2,
    )

    assert mapping is not None
    assert mapping.name == "tile_4x4_row_pair"
    assert mapping.rows == 8
    assert mapping.columns == 8
    assert mapping.subgroup_size == 32
    assert mapping.elements_per_lane == 2

    expected_coordinates = []
    for lane in range(32):
        qid = lane // 4
        fm = (qid & 4) + ((lane // 2) % 4)
        fn = (qid & 2) * 2 + (lane % 2) * 2
        lane_coordinates = ((fm, fn), (fm, fn + 1))
        expected_coordinates.append(lane_coordinates)

        for element, coordinate in enumerate(lane_coordinates):
            assert mapping.coordinate(lane, element) == coordinate
            assert mapping.owner(*coordinate) == (lane, element)

    assert mapping.lane_coordinates == tuple(expected_coordinates)
    assert {
        coordinate
        for lane_coordinates in mapping.lane_coordinates
        for coordinate in lane_coordinates
    } == {(row, column) for row in range(8) for column in range(8)}


@pytest.mark.parametrize(
    "contract",
    [
        (4, 8, 32, 2),
        (8, 4, 32, 2),
        (8, 8, 16, 2),
        (8, 8, 32, 1),
    ],
)
def test_fragment_mapping_registry_resolves_exact_contracts_only(contract):
    assert "tile_4x4_row_pair" in cooperative_matrix_fragment_mapping_names()
    assert has_cooperative_matrix_fragment_mapping("tile_4x4_row_pair")
    assert get_cooperative_matrix_fragment_mapping(
        "  tile_4x4_row_pair  ",
        8,
        8,
        32,
        2,
    ) is get_cooperative_matrix_fragment_mapping(
        "tile_4x4_row_pair",
        8,
        8,
        32,
        2,
    )
    assert (
        get_cooperative_matrix_fragment_mapping(
            "tile_4x4_row_pair",
            *contract,
        )
        is None
    )


def test_fragment_mapping_registry_leaves_opaque_names_unresolved():
    names_before_lookup = cooperative_matrix_fragment_mapping_names()

    assert not has_cooperative_matrix_fragment_mapping("vendor_opaque_profile")
    assert not has_cooperative_matrix_fragment_mapping(1)
    assert (
        get_cooperative_matrix_fragment_mapping(
            "vendor_opaque_profile",
            8,
            8,
            32,
            2,
        )
        is None
    )
    assert (
        get_cooperative_matrix_fragment_mapping(
            "tile_4x4_row_pair",
            True,
            8,
            32,
            2,
        )
        is None
    )
    assert cooperative_matrix_fragment_mapping_names() == names_before_lookup
    assert (
        get_cooperative_matrix_fragment_mapping(
            "unregistered_profile",
            8,
            8,
            32,
            2,
        )
        is None
    )


def test_fragment_mapping_registry_supports_multiple_exact_contracts_per_name():
    first_mapping = CooperativeMatrixFragmentMapping(
        name="test_multiple_exact_contracts",
        rows=1,
        columns=1,
        subgroup_size=1,
        lane_coordinates=(((0, 0),),),
    )
    second_mapping = CooperativeMatrixFragmentMapping(
        name="test_multiple_exact_contracts",
        rows=2,
        columns=2,
        subgroup_size=2,
        lane_coordinates=(((0, 0), (0, 1)), ((1, 0), (1, 1))),
        description="Test-only two-row mapping",
    )

    assert register_cooperative_matrix_fragment_mapping(first_mapping) is first_mapping
    assert (
        register_cooperative_matrix_fragment_mapping(second_mapping) is second_mapping
    )
    assert (
        get_cooperative_matrix_fragment_mapping(
            "test_multiple_exact_contracts", 1, 1, 1, 1
        )
        is first_mapping
    )
    assert (
        get_cooperative_matrix_fragment_mapping(
            "test_multiple_exact_contracts", 2, 2, 2, 2
        )
        is second_mapping
    )
    with pytest.raises(ValueError, match="already registered"):
        register_cooperative_matrix_fragment_mapping(
            CooperativeMatrixFragmentMapping(
                name="test_multiple_exact_contracts",
                rows=1,
                columns=1,
                subgroup_size=1,
                lane_coordinates=(((0, 0),),),
                description="A duplicate exact contract",
            )
        )

    replacement = CooperativeMatrixFragmentMapping(
        name="test_multiple_exact_contracts",
        rows=1,
        columns=1,
        subgroup_size=1,
        lane_coordinates=(((0, 0),),),
        description="Replacement for the exact contract",
    )
    assert (
        register_cooperative_matrix_fragment_mapping(replacement, overwrite=True)
        is replacement
    )
    assert (
        get_cooperative_matrix_fragment_mapping(
            "test_multiple_exact_contracts", 1, 1, 1, 1
        )
        is replacement
    )

    names = cooperative_matrix_fragment_mapping_names()
    assert names == tuple(sorted(set(names)))
    assert names.count("test_multiple_exact_contracts") == 1


def test_fragment_mapping_registry_rejects_invalid_registration_arguments():
    with pytest.raises(TypeError, match="mapping must be"):
        register_cooperative_matrix_fragment_mapping(None)

    mapping = CooperativeMatrixFragmentMapping(
        name="test_invalid_overwrite",
        rows=1,
        columns=1,
        subgroup_size=1,
        lane_coordinates=(((0, 0),),),
    )
    with pytest.raises(TypeError, match="overwrite must be a boolean"):
        register_cooperative_matrix_fragment_mapping(mapping, overwrite=1)


def test_fragment_mapping_rejects_duplicate_coordinates():
    with pytest.raises(ValueError, match="duplicate coordinates"):
        CooperativeMatrixFragmentMapping(
            name="duplicate",
            rows=2,
            columns=2,
            subgroup_size=2,
            lane_coordinates=(((0, 0), (0, 1)), ((1, 0), (0, 0))),
        )


def test_fragment_mapping_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError, match="out-of-range coordinates"):
        CooperativeMatrixFragmentMapping(
            name="out_of_range",
            rows=2,
            columns=2,
            subgroup_size=2,
            lane_coordinates=(((0, 0), (0, 1)), ((1, 0), (2, 1))),
        )


def test_fragment_mapping_rejects_non_integer_coordinates():
    with pytest.raises(ValueError, match="coordinates must be integers"):
        CooperativeMatrixFragmentMapping(
            name="non_integer",
            rows=1,
            columns=1,
            subgroup_size=1,
            lane_coordinates=((("0", 0),),),
        )


def test_fragment_mapping_rejects_inconsistent_lane_widths():
    with pytest.raises(ValueError, match="same number of coordinates"):
        CooperativeMatrixFragmentMapping(
            name="inconsistent_width",
            rows=2,
            columns=2,
            subgroup_size=2,
            lane_coordinates=(((0, 0),), ((0, 1), (1, 0), (1, 1))),
        )


def test_fragment_mapping_rejects_subgroup_lane_count_mismatch():
    with pytest.raises(ValueError, match="provides 1 lanes"):
        CooperativeMatrixFragmentMapping(
            name="lane_count",
            rows=1,
            columns=1,
            subgroup_size=2,
            lane_coordinates=(((0, 0),),),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": None}, "name must be a string"),
        ({"name": "   "}, "name must be non-empty"),
        ({"rows": 0}, "rows must be a positive integer"),
        ({"columns": -1}, "columns must be a positive integer"),
        ({"subgroup_size": True}, "subgroup_size must be a positive integer"),
        ({"description": None}, "description must be a string"),
        ({"lane_coordinates": None}, "lane_coordinates must be a sequence"),
        (
            {"lane_coordinates": ("not-a-lane", ((1, 0), (1, 1)))},
            "lane 0 must be a coordinate sequence",
        ),
        (
            {"lane_coordinates": (((0, 0, 1), (0, 1)), ((1, 0), (1, 1)))},
            "must be a \\(row, column\\) pair",
        ),
        (
            {"lane_coordinates": (((0, False), (0, 1)), ((1, 0), (1, 1)))},
            "coordinates must be integers",
        ),
        (
            {"lane_coordinates": ((), ())},
            "must provide at least one coordinate per lane",
        ),
        (
            {"columns": 3},
            "provides 4 coordinates for a 2x3 matrix with 6 elements",
        ),
    ],
)
def test_fragment_mapping_rejects_malformed_profiles(overrides, message):
    arguments = {
        "name": "malformed_profile",
        "rows": 2,
        "columns": 2,
        "subgroup_size": 2,
        "lane_coordinates": (((0, 0), (0, 1)), ((1, 0), (1, 1))),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        CooperativeMatrixFragmentMapping(**arguments)


def test_fragment_mapping_normalizes_names_and_coordinate_sequences():
    mapping = CooperativeMatrixFragmentMapping(
        name="  test_normalized_profile  ",
        rows=1,
        columns=2,
        subgroup_size=1,
        lane_coordinates=[[(0, 0), [0, 1]]],
    )

    assert mapping.name == "test_normalized_profile"
    assert mapping.lane_coordinates == (((0, 0), (0, 1)),)


def test_fragment_mapping_coordinate_access_rejects_invalid_indices():
    mapping = get_cooperative_matrix_fragment_mapping("tile_4x4_row_pair", 8, 8, 32, 2)

    assert mapping is not None
    with pytest.raises(IndexError, match="lane index out of range"):
        mapping.coordinate(-1, 0)
    with pytest.raises(IndexError, match="element index out of range"):
        mapping.coordinate(0, 2)
    with pytest.raises(TypeError, match="lane must be an integer"):
        mapping.coordinate(True, 0)
    with pytest.raises(KeyError):
        mapping.owner(8, 0)
    with pytest.raises(TypeError, match="row and column must be integers"):
        mapping.owner(False, 0)
