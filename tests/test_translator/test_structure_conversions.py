import pytest

from crosstl.translator.structure_conversions import (
    COMPLEX64_SCALAR_CONVERSION,
    REGISTERED_SCALAR_TO_STRUCTURE_CONVERSIONS,
    ScalarKind,
    StructureFieldValue,
    registered_scalar_to_structure_conversion,
)


def test_complex64_scalar_conversion_contract_is_explicit_and_single_use():
    contract = registered_scalar_to_structure_conversion("complex64_t")

    assert contract is COMPLEX64_SCALAR_CONVERSION
    assert contract.destination_shape == (
        ("real", "float"),
        ("imag", "float"),
    )
    assert contract.source_kinds == {
        ScalarKind.BOOLEAN,
        ScalarKind.SIGNED_INTEGER,
        ScalarKind.UNSIGNED_INTEGER,
        ScalarKind.FLOATING,
    }
    assert [field.scalar_value for field in contract.fields] == [
        StructureFieldValue.CONVERTED_SOURCE,
        StructureFieldValue.ZERO,
    ]
    assert contract.scalar_source_use_count == 1
    assert contract.supports_default_construction
    assert contract.matches_destination_shape((("real", "float"), ("imag", "float")))
    assert not contract.matches_destination_shape((("real", "float"), ("imag", "int")))


def test_scalar_structure_conversion_registry_is_exact_and_read_only():
    assert registered_scalar_to_structure_conversion("Pair") is None
    assert set(REGISTERED_SCALAR_TO_STRUCTURE_CONVERSIONS) == {"complex64_t"}

    with pytest.raises(TypeError):
        REGISTERED_SCALAR_TO_STRUCTURE_CONVERSIONS["Pair"] = COMPLEX64_SCALAR_CONVERSION
