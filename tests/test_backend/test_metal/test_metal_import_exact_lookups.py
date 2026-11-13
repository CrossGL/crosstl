"""The Metal importer looks scalar spellings, template bindings and unsigned types up exactly."""

import pytest

import crosstl
from crosstl.backend.Metal.MetalParser import UNSIGNED_INTEGER_TYPES
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.template_text import substitute_template_identifiers


def test_template_substitution_replaces_only_bound_identifiers():
    bindings = {"T": "float", "N": 4}
    assert (
        substitute_template_identifiers("vec<T, N> Tx(T)", bindings)
        == "vec<float, 4> Tx(float)"
    )
    assert substitute_template_identifiers("U::value", bindings) == "U::value"


def test_every_metal_scalar_spelling_names_one_scalar():
    table = MetalPreprocessor._METAL_SCALAR_BY_SPELLING
    for name in MetalPreprocessor._METAL_SCALAR_TYPE_SIZES:
        assert table[name] == name
    assert table["bfloat"] == table["bfloat16"] == "bfloat16_t"
    assert set(table.values()) == set(MetalPreprocessor._METAL_SCALAR_TYPE_SIZES)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("bfloat4", ("bfloat16_t", 4)),
        ("float3", ("float", 3)),
        ("vec<bfloat, 2>", ("bfloat16_t", 2)),
        ("metal::vec<int, 3>", ("int", 3)),
        ("int64_t", ("int64_t", 1)),
        ("mystery4", None),
        ("vec<mystery, 2>", None),
    ],
)
def test_scalar_and_width_accepts_only_metal_scalars(text, expected):
    assert MetalPreprocessor()._scalar_and_width(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("const bfloat16", "bfloat16_t"),
        ("metal::half", "half"),
        ("mystery", None),
        ("float2", None),
    ],
)
def test_static_constexpr_scalar_type_accepts_only_metal_scalars(text, expected):
    assert MetalPreprocessor()._static_constexpr_scalar_type(text) == expected


def test_unsigned_integer_spellings_import_as_unsigned_types(tmp_path):
    assert UNSIGNED_INTEGER_TYPES == {
        "char": "uchar",
        "short": "ushort",
        "int": "uint",
        "long": "ulong",
    }
    path = tmp_path / "kernel.metal"
    path.write_text("""
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* o [[buffer(0)]]) {
    unsigned char a = 1;
    unsigned short b = 2;
    unsigned c = 3;
    unsigned long d = 4;
    signed int e = 5;
    o[0] = uint(a) + uint(b) + c + uint(d) + uint(e);
}
""")
    output = crosstl.translate(str(path), backend="cgl")
    for declaration in (
        "uint8 a = 1;",
        "uint16 b = 2;",
        "uint c = 3;",
        "uint64 d = 4;",
        "int e = 5;",
    ):
        assert declaration in output
