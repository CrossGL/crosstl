"""Metal static assertions over qualified namespace constants, integral casts, member lists and ``auto`` locals.

Each case is a construct from real Metal headers that left a true assertion unresolved, so the kernel was refused
with ``MetalStaticAssertionError``. A false assertion must still fail, and what the evaluator cannot prove must stay
unresolved. The expected ``sizeof`` values are the ones the Metal compiler (``xcrun metal``) accepts.
"""

import pytest

import crosstl
from crosstl.backend.Metal.auto_integral_constants import deduced_integral_auto_type
from crosstl.backend.Metal.integral_casts import (
    fold_integral_cast,
    integral_cast_target,
    rewrite_integral_casts,
)
from crosstl.backend.Metal.member_declarator_lists import split_member_declarator_list
from crosstl.backend.Metal.namespace_constants import NamespaceConstantValues
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalStaticAssertionError,
)
from crosstl.backend.Metal.struct_head_specifiers import (
    alignment_is_natural,
    skip_struct_head_specifiers,
)

KERNEL = "kernel void k(device int* out [[buffer(0)]]) { out[0] = 1; }\n"
NAMESPACE = """
namespace c10 {
namespace metal {
constant constexpr unsigned simdgroup_size = 32;
} // namespace metal
} // namespace c10
constant constexpr uint32_t INNER_TG_SIZE = 256;
"""


def _evaluate(code: str) -> str:
    return MetalPreprocessor()._evaluate_static_assertions(code)


def _fails(code: str) -> MetalStaticAssertionError:
    with pytest.raises(MetalStaticAssertionError) as excinfo:
        _evaluate(code)
    return excinfo.value


@pytest.mark.parametrize(
    "reference", ["::c10::metal::simdgroup_size", "c10::metal::simdgroup_size"]
)
def test_qualified_namespace_constant_resolves(reference):
    code = (
        NAMESPACE
        + f'static_assert(INNER_TG_SIZE % {reference} == 0, "whole simdgroups");\n'
        + KERNEL
    )

    assert "static_assert" not in _evaluate(code)


def test_qualified_namespace_constant_keeps_false_assertions_failing():
    code = (
        NAMESPACE
        + 'static_assert(INNER_TG_SIZE % ::c10::metal::simdgroup_size == 1, "odd");\n'
        + KERNEL
    )

    error = _fails(code)

    assert error.reason == "assertion-failed"
    assert "256 % (32) == 1" in str(error)


def test_unqualified_namespace_member_is_not_visible_outside_its_namespace():
    code = NAMESPACE + 'static_assert(simdgroup_size == 32, "hidden");\n' + KERNEL

    error = _fails(code)

    assert error.reason == "condition-unresolved"
    assert error.unresolved_dependencies == ("simdgroup_size",)


def test_namespace_constant_declared_after_the_assertion_is_not_used():
    code = (
        "namespace a { constant constexpr int early = 1; }\n"
        'static_assert(a::late == 2, "later");\n'
        "namespace a { constant constexpr int late = 2; }\n" + KERNEL
    )

    assert _fails(code).reason == "condition-unresolved"


@pytest.mark.parametrize(
    "condition",
    [
        "256 <= static_cast<uint32_t>(1024)",
        "static_cast<unsigned int>(7) == 7",
        "uint(5) + int(-1) == 4",
        "static_cast<bool>(9) == 1",
        "static_cast<metal::ushort>(65535) == 65535",
    ],
)
def test_exact_integral_casts_fold(condition):
    code = f'static_assert({condition}, "cast");\n' + KERNEL

    assert "static_assert" not in _evaluate(code)


@pytest.mark.parametrize(
    "condition",
    [
        "static_cast<uint8_t>(300) == 44",
        "static_cast<uint>(-1) > 0",
        "static_cast<float>(2) == 2",
    ],
)
def test_inexact_or_non_integral_casts_stay_unresolved(condition):
    code = f'static_assert({condition}, "cast");\n' + KERNEL

    assert _fails(code).reason == "condition-unresolved"


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        (
            "struct P { uint32_t m, n, k; long a; bool r; };",
            32,
        ),
        (
            "struct P { uint32_t m; uint32_t n; uint32_t k; long a; bool r; };",
            32,
        ),
        ("struct P { metal::float2 a, b[2]; uint c; };", 32),
    ],
)
def test_sizeof_counts_every_member_of_a_declarator_list(declaration, expected):
    code = (
        f'{declaration}\nstatic_assert(sizeof(P) == {expected}, "layout");\n' + KERNEL
    )

    assert "static_assert" not in _evaluate(code)


def test_sizeof_declarator_list_keeps_wrong_sizes_failing():
    code = (
        "struct P { uint32_t m, n, k; long a; bool r; };\n"
        'static_assert(sizeof(P) == 24, "layout");\n' + KERNEL
    )

    error = _fails(code)

    assert error.reason == "assertion-failed"
    assert "32 == 24" in str(error)


def test_constexpr_auto_locals_follow_their_initializer():
    code = """
    constant constexpr uint32_t tile_size = 32;
    kernel void k(device uint* out [[buffer(0)]]) {
        constexpr auto tile = tile_size;
        constexpr uint half_tile = tile / 2;
        constexpr auto literal = 8;
        static_assert(half_tile * 2 == tile, "split");
        static_assert(literal == 8, "literal");
        out[0] = half_tile;
    }
    """

    assert "static_assert" not in _evaluate(code)


def test_constexpr_auto_with_a_non_evident_type_stays_unresolved():
    code = """
    kernel void k(device uint* out [[buffer(0)]]) {
        constexpr auto mixed = 2 * 3u;
        static_assert(mixed == 6, "mixed");
        out[0] = mixed;
    }
    """

    assert _fails(code).reason == "condition-unresolved"


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("uint32_t m, n, k", ["uint32_t m", "uint32_t n", "uint32_t k"]),
        ("vec<float, 4> a, b[2]", ["vec<float, 4> a", "vec<float, 4> b[2]"]),
        ("metal::float4 a, b", ["metal::float4 a", "metal::float4 b"]),
        ("uint a : 4, b : 4", None),
        ("int a = 1, b", None),
        ("device float *a, *b", None),
        ("static constexpr int a, b", None),
        ("float a [[user(x)]], b", None),
        ("float x", None),
    ],
)
def test_split_member_declarator_list(declaration, expected):
    assert split_member_declarator_list(declaration) == expected


def test_integral_cast_helpers():
    rewritten = rewrite_integral_casts(
        "static_cast<unsigned int>(3) + static_cast<float>(1)"
    )

    assert rewritten.startswith("__crossgl_integral_cast_unsigned_int(3)")
    assert "static_cast<float>(1)" in rewritten
    assert integral_cast_target("__crossgl_integral_cast_uint32_t") == "uint32_t"
    assert integral_cast_target("ushort") == "ushort"
    assert integral_cast_target("float") is None
    assert fold_integral_cast("bool", 7) == 1
    assert fold_integral_cast("uint8_t", 255) == 255
    assert fold_integral_cast("uint8_t", 256) is None
    assert fold_integral_cast("int", -(2**31)) == -(2**31)


@pytest.mark.parametrize(
    ("initializer", "expected"),
    [
        ("tile", "uint32_t"),
        ("(tile)", "uint32_t"),
        ("8", "int"),
        ("8u", "uint"),
        ("4294967296", "long"),
        ("1ul", "ulong"),
        ("true", "bool"),
        ("tile * 2", None),
        ("unknown", None),
    ],
)
def test_deduced_integral_auto_type(initializer, expected):
    assert deduced_integral_auto_type(initializer, {"tile": "uint32_t"}) == expected


def test_namespace_constant_values_respect_declaration_order():
    class Binding:
        def __init__(self, position, start, end, value):
            self.declaration_position = position
            self.scope_start = start
            self.scope_end = end
            self.value = value
            self.type_alias = False

    values = NamespaceConstantValues(
        [(10, 50, "outer::inner"), (0, 60, "outer")],
        {"width": [Binding(20, 10, 50, "4")], "local": [Binding(30, 0, 100, "1")]},
    )

    assert values.visible_at(25) == {"outer::inner::width": "4"}
    assert values.substitute("::outer::inner::width * 2", 25) == "(4) * 2"
    assert values.substitute("outer::inner::width", 15) == "outer::inner::width"
    assert values.substitute("outer::inner::width(3)", 25) == "outer::inner::width(3)"


TEMPLATE_CONSTRUCTOR = """
#include <metal_stdlib>
namespace c10 {
namespace metal {
struct alignas(1) Byte {
  uchar x;
  Byte() = default;
  template <typename T>
  Byte(T value) : x(uchar(float(value) * 2.0f)) {}
  operator float() const { return float(x); }
};
} // namespace metal
} // namespace c10
using c10::metal::Byte;
using namespace metal;
kernel void k(device const Byte* in [[buffer(0)]], device float* out [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  const auto v = i < 4 ? in[i] : Byte(0.0f);
  out[i] = float(v);
}
"""


def test_template_constructor_of_an_alignas_struct_is_lowered(tmp_path):
    # ``struct alignas(1) Byte`` used to be scanned as a struct named
    # ``alignas``; its template constructor then became an invalid free function.
    source = tmp_path / "byte.metal"
    source.write_text(TEMPLATE_CONSTRUCTOR, encoding="utf-8")

    hlsl = crosstl.translate(str(source), backend="directx", source_backend="metal")

    assert "crosstl_ctor_Byte" in hlsl
    assert " : x(" not in hlsl


def test_natural_alignas_keeps_the_layout_and_stricter_alignas_refuses_it():
    natural = (
        "struct alignas(1) B { uchar x; uchar y; };\n"
        'static_assert(sizeof(B) == 2, "natural");\n' + KERNEL
    )
    stricter = (
        "struct alignas(16) R { float a; float b; };\n"
        'static_assert(sizeof(R) == 8, "not the MSL size");\n' + KERNEL
    )

    assert "static_assert" not in _evaluate(natural)
    # MSL lays R out in 16 bytes; the evaluator does not model alignas(16), so it
    # must not report the natural 8.
    assert _fails(stricter).reason == "condition-unresolved"


@pytest.mark.parametrize(
    ("head", "rest", "natural"),
    [
        (" alignas(1) Byte {", "Byte {", True),
        (" alignas(0) Z {", "Z {", True),
        (" [[deprecated]] alignas(16) X {", "X {", False),
        (" __attribute__((packed)) Y {", "Y {", False),
        (" alignas(sizeof(float)) W {", "W {", False),
    ],
)
def test_struct_head_specifiers(head, rest, natural):
    index, specifiers = skip_struct_head_specifiers(head, 0)

    assert head[index:] == rest
    assert alignment_is_natural(specifiers) is natural
