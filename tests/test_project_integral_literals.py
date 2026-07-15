import pytest

from crosstl.project.integral_literals import (
    CFamilyIntegralLiteralError,
    parse_c_family_integral_literal,
)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("0", 0),
        ("00", 0),
        ("01", 1),
        ("42", 42),
        ("077", 63),
        ("0x2a", 42),
        ("0X2Aul", 42),
        ("0b101010", 42),
        ("0B10'1010ULL", 42),
        ("1'000L", 1000),
    ],
)
def test_parse_c_family_integral_literal(spelling, expected):
    assert parse_c_family_integral_literal(spelling) == expected


@pytest.mark.parametrize(
    ("spelling", "reason"),
    [
        ("", "missing"),
        ("08", "invalid-octal-digit"),
        ("0b102", "invalid-digit"),
        ("0x", "invalid-digit"),
        ("1''0", "invalid-digit-separator"),
        ("1uu", "invalid-suffix"),
        ("-1", "negative"),
        ("1 + 1", "unsupported-expression"),
    ],
)
def test_parse_c_family_integral_literal_rejects_invalid_spelling(spelling, reason):
    with pytest.raises(CFamilyIntegralLiteralError) as exc_info:
        parse_c_family_integral_literal(spelling)

    assert exc_info.value.spelling == spelling
    assert exc_info.value.reason == reason
