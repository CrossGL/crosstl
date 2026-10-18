"""HLSL constant- and structured-buffer layout rules (DXC packing)."""

import pytest

from crosstl.translator.ast import AttributeNode, IdentifierNode
from crosstl.translator.codegen.hlsl_buffer_layout import (
    HLSL_CBUFFER_LAYOUT,
    HLSL_STRUCTURED_LAYOUT,
    HlslLayoutError,
    array_type,
    buffer_layout_from_attributes,
    cbuffer_struct_placements,
    matrix_type,
    scalar_type,
    struct_placements,
    struct_type,
    vector_type,
)

UINT = scalar_type("uint")
INT = scalar_type("int")
FLOAT = scalar_type("float")
BOOL = scalar_type("bool")


def offsets(members, layout=HLSL_CBUFFER_LAYOUT):
    placements, size = struct_placements(struct_type("S", members), layout)
    return [placement.offset for placement in placements], size


def test_scalars_and_vectors_pack_without_straddling_registers():
    members = [
        ("x", UINT),
        ("y", UINT),
        ("z", UINT),
        ("a", BOOL),
        ("b", BOOL),
        ("c", INT),
        ("d", BOOL),
        ("e", BOOL),
        ("f", vector_type("int", 3)),
        ("g", BOOL),
        ("h", vector_type("bool", 2)),
        ("i", INT),
        ("j", BOOL),
    ]

    assert offsets(members) == (
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 44, 48, 56, 60],
        64,
    )


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (FLOAT, vector_type("float", 2), 4),
        (FLOAT, vector_type("float", 3), 4),
        (vector_type("float", 2), vector_type("float", 3), 16),
        (vector_type("float", 3), FLOAT, 12),
        (vector_type("float", 3), vector_type("float", 2), 16),
        (FLOAT, vector_type("float", 4), 16),
    ],
)
def test_vector_offsets_follow_register_boundaries(first, second, expected):
    assert offsets([("a", first), ("b", second)])[0] == [0, expected]


def test_mixed_vectors_after_dispatch_constants():
    members = [
        ("x", UINT),
        ("y", UINT),
        ("z", UINT),
        ("f2", vector_type("float", 2)),
        ("i4", vector_type("int", 4)),
        ("i2", vector_type("int", 2)),
        ("i3", vector_type("int", 3)),
    ]

    assert offsets(members) == ([0, 4, 8, 16, 32, 48, 64], 76)


def test_array_elements_start_registers_and_the_last_one_is_not_padded():
    members = [("a", array_type(INT, 2)), ("b", FLOAT), ("c", array_type(FLOAT, 3))]

    assert offsets(members) == ([0, 20, 32], 68)


def test_structs_start_registers_and_following_members_pack_after_them():
    cell = struct_type("Cell", [(name, FLOAT) for name in "abcdefg"])
    members = [
        ("x", UINT),
        ("y", UINT),
        ("z", UINT),
        ("cell", cell),
        ("t", FLOAT),
        ("u", FLOAT),
        ("v", INT),
        ("other", cell),
    ]

    assert offsets(members) == ([0, 4, 8, 16, 44, 48, 52, 64], 92)


def test_matrices_place_each_stored_vector_in_its_own_register():
    members = [("s", FLOAT), ("m", matrix_type("float", 3, 3)), ("t", FLOAT)]

    assert offsets(members) == ([0, 16, 60], 64)


def test_structured_layout_uses_component_alignment():
    members = [("p", vector_type("int", 2)), ("b", BOOL)]
    placements, size = struct_placements(
        struct_type("S", members), HLSL_STRUCTURED_LAYOUT
    )

    assert [placement.offset for placement in placements] == [0, 8]
    assert size == 12


def test_structured_layout_rounds_structs_to_their_alignment():
    members = [("d", scalar_type("double")), ("f", FLOAT)]

    assert offsets(members, HLSL_STRUCTURED_LAYOUT) == ([0, 8], 16)


def test_nested_cbuffer_struct_offsets_are_relative_to_the_struct():
    inner = struct_type("Inner", [("a", INT), ("b", FLOAT)])
    outer = struct_type("Outer", [("inner", inner), ("c", FLOAT)])
    placements, size = cbuffer_struct_placements(outer)

    assert [placement.offset for placement in placements] == [0, 8]
    assert size == 12


def test_unknown_scalars_and_unsized_arrays_are_rejected():
    with pytest.raises(HlslLayoutError):
        scalar_type("half")
    with pytest.raises(HlslLayoutError):
        array_type(FLOAT, 0)


def test_buffer_layout_attribute_parsing():
    attributes = [
        AttributeNode("register", [IdentifierNode("b0")]),
        AttributeNode("buffer_layout", [IdentifierNode("hlsl_cbuffer")]),
    ]

    assert buffer_layout_from_attributes(attributes) == HLSL_CBUFFER_LAYOUT
    assert buffer_layout_from_attributes(attributes[:1]) is None
    assert (
        buffer_layout_from_attributes(
            [AttributeNode("buffer_layout", [IdentifierNode("std140")])]
        )
        is None
    )
