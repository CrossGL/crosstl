"""Contracts the Metal importer gives SIMD-group matrices no lane-level view constrains."""

from crosstl.backend.Metal.opaque_matrix_contracts import (
    complete_opaque_matrix_contracts,
)

UNCONTRACTED = "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified>"
CONTRACTED = (
    "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified,"
    "metal_thread_elements,32,2,metal_opaque_simdgroup_matrix,"
    "tile_4x4_row_pair,unit_lane_coordinates>"
)


def _complete(code, replacements=None, mapping="tile_4x4_row_pair", provenance=None):
    replacements = {} if replacements is None else replacements
    complete_opaque_matrix_contracts(
        code, replacements, mapping, provenance or "unit_lane_coordinates"
    )
    return replacements


def test_uncontracted_matrices_take_the_configured_mapping():
    assert _complete(f"{UNCONTRACTED} A; {UNCONTRACTED} B;") == {
        UNCONTRACTED: CONTRACTED
    }


def test_each_component_type_gets_its_own_contract():
    half = UNCONTRACTED.replace("float", "half")
    assert _complete(f"{UNCONTRACTED} A; {half} H;") == {
        UNCONTRACTED: CONTRACTED,
        half: CONTRACTED.replace("float", "half"),
    }


def test_a_contract_from_a_lane_view_is_kept():
    viewed = CONTRACTED.replace(
        "metal_opaque_simdgroup_matrix", "metal_thread_elements_reference_view"
    )
    assert _complete(UNCONTRACTED, {UNCONTRACTED: viewed}) == {UNCONTRACTED: viewed}


def test_without_a_mapping_nothing_is_inferred():
    replacements = {}
    complete_opaque_matrix_contracts(UNCONTRACTED, replacements, None, None)
    assert replacements == {}


def test_a_mapping_without_provenance_leaves_it_unspecified():
    replacements = {}
    complete_opaque_matrix_contracts(
        UNCONTRACTED, replacements, "tile_4x4_row_pair", None
    )
    assert replacements == {
        UNCONTRACTED: CONTRACTED.replace(",unit_lane_coordinates", "")
    }


def test_shapes_the_mapping_does_not_register_keep_an_unspecified_contract():
    # The registered row-pair mapping covers 8x8 matrices only, 16x2 does not
    # spread evenly over 32 lanes, and symbolic dimensions have no value yet.
    code = " ".join(
        UNCONTRACTED.replace(",8,8,", f",{shape},")
        for shape in ("16,16", "4,8", "16,2", "N,8")
    )
    assert _complete(code) == {}


def test_types_that_already_carry_a_contract_are_not_matched():
    assert _complete(f"{CONTRACTED} A;") == {}


def test_an_unknown_mapping_registers_no_contract():
    assert _complete(UNCONTRACTED, mapping="not_registered") == {}
