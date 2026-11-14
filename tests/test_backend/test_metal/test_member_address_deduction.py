"""Constrained overloads deduce ``T`` from addresses of private members.

``bump(&(word.bits))`` and ``bump(&box.bits)`` pass a ``thread uint*`` and
``bump(&box)`` passes a ``thread word_box*``. Overload resolution among
``enable_if_t`` alternatives needs those argument types. The address of one
member of a local value has the member's field type in the local's address
space, and a struct local keeps the storage provenance of its declaration.
Array members and other unknown shapes stay un-inferable and fail closed.
"""

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.preprocessor import MetalTemplateSpecializationError
from tests.test_translator.test_metal_builtin_ownership import _compile

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T>
constexpr constant bool is_word = is_same_v<T, uint>;

union packed_word { uchar val[4]; uint bits; };
struct word_box { uint bits; uint other; };

template <typename T, enable_if_t<is_word<T>, bool> = true>
METAL_FUNC void bump(thread T* value) { value[0] += T(1); }

template <typename T, enable_if_t<!is_word<T>, bool> = true>
METAL_FUNC void bump(thread T* value) { value[0].bits += 2u; }

kernel void member_deduction(device uint* results [[buffer(0)]]) {
  packed_word word;
  word.bits = 0x04030201u;
  BUMP_WORD;
  word_box box;
  box.bits = 5u;
  box.other = 9u;
  bump(&box.bits);
  bump(&box);
  results[0] = word.bits;
  results[1] = box.bits;
  results[2] = box.other;
}
"""


def _translate(tmp_path, backend, bump_word="bump(&(word.bits))"):
    source = tmp_path / "member_deduction.metal"
    source.write_text(SOURCE.replace("BUMP_WORD", bump_word), encoding="utf-8")
    return translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


@pytest.mark.parametrize(
    ("target", "word_call", "box_call"),
    (
        ("directx", "bump_uint(word.CrossGLUnionStorage)", "bump_word_box(box)"),
        ("opengl", "bump_uint(word.CrossGLUnionStorage)", "bump_word_box(box)"),
    ),
)
def test_member_addresses_select_constrained_overloads(
    tmp_path, target, word_call, box_call
):
    generated = _translate(tmp_path, target)
    compact = " ".join(generated.split())
    assert word_call in compact
    assert "bump_uint(box.bits)" in compact
    assert box_call in compact
    _compile(generated, target, tmp_path)


def test_member_address_deduction_round_trips_to_metal(tmp_path):
    generated = _translate(tmp_path, "metal")
    _compile(generated, "metal", tmp_path)


def test_array_member_address_stays_uninferable(tmp_path):
    with pytest.raises(MetalTemplateSpecializationError, match="inferred"):
        _translate(tmp_path, "directx", bump_word="bump(&(word.val))")
