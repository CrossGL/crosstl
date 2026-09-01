import pytest

from crosstl.translator.structure_conversions import (
    COMPLEX64_SCALAR_CONVERSION,
    REGISTERED_SCALAR_TO_STRUCTURE_CONVERSIONS,
    REGISTERED_STRUCTURE_CONVERSION_IDENTITIES,
    ScalarKind,
    ScalarToStructureConversion,
    StructureConversionField,
    StructureFieldValue,
    registered_scalar_to_structure_conversion,
    registered_structure_conversion_for_identity,
)


def test_structure_conversion_representation_identities_are_exact_and_validated():
    assert COMPLEX64_SCALAR_CONVERSION.representation_types == {"complex_t_float"}
    assert (
        registered_structure_conversion_for_identity("complex64_t")
        is COMPLEX64_SCALAR_CONVERSION
    )
    assert (
        registered_structure_conversion_for_identity("complex_t_float")
        is COMPLEX64_SCALAR_CONVERSION
    )
    assert registered_structure_conversion_for_identity("Pair") is None
    assert set(REGISTERED_STRUCTURE_CONVERSION_IDENTITIES) == {
        "complex64_t",
        "complex_t_float",
    }

    with pytest.raises(ValueError, match="invalid representation type"):
        ScalarToStructureConversion(
            destination_type="Pair",
            source_kinds=frozenset({ScalarKind.FLOATING}),
            fields=(
                StructureConversionField(
                    name="value",
                    type_name="float",
                    scalar_value=StructureFieldValue.CONVERTED_SOURCE,
                ),
            ),
            representation_types=frozenset({""}),
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
