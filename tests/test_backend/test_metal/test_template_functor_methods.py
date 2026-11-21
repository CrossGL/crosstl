"""A struct-template functor used inside a function template keeps its members in Metal input (#4337).

``Less<T> op; return op(a, b);`` in a function template names no concrete struct, and CrossGL structs carry data
only, so the importer used to write ``generic<T> struct Less {}`` without its operator and keep ``op(a, b)``: Metal
output then declared ``struct Less {};`` and instantiated ``Less<float>``, which the Metal compiler rejects. The
importer now instantiates such a function template at each call, as it does for a template whose result type is a
``decltype`` of its parameters, so member lowering sees ``Less<float>`` and lowers its members.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from tests.metal_compute_dispatch import compile_metal_library, dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_TEMPLATE_FUNCTOR_EXECUTION"

# The reproducer of #4337, on one word buffer.
OPERATOR_ONLY = """#include <metal_stdlib>
using namespace metal;
template <typename T>
struct Less {
  bool operator()(T a, T b) const thread { return a < b; }
};
template <typename T>
bool less(T a, T b) { Less<T> op; return op(a, b); }
kernel void run(device int* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  words[0] = less(words[1], words[2]) ? 1 : 0;
  words[3] = less(float(words[4]), 2.5f) ? 7 : 9;
}
"""

# A data member, a named method, a method call inside the call operator's arguments, two specializations and a
# function template that reaches the functor only through another function template.
MEMBERS_AND_CHAINS = """#include <metal_stdlib>
using namespace metal;
template <typename T>
struct Less {
  T bias;
  bool operator()(T a, T b) const thread { return a < b + bias; }
  T twice(T a) const { return a + a; }
};
template <typename T>
bool less(T a, T b) { Less<T> op; op.bias = T(1); return op(a, op.twice(b)); }
template <typename T>
int count_less(T a, T b, T c) { return int(less(a, b)) + int(less(b, c)); }
kernel void run(device int* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  words[0] = count_less(words[1], words[2], words[3]);
  words[4] = less(float(words[5]), 2.5f) ? 7 : 9;
}
"""

# A concrete function whose call-operator argument calls another method of the same receiver: `return op(...)` is
# a statement, not a declaration of `op`.
RETURNED_CALL_ARGUMENT = """#include <metal_stdlib>
using namespace metal;
template <typename T>
struct Less {
  T bias;
  bool operator()(T a, T b) const thread { return a < b + bias; }
  T twice(T a) const { return a + a; }
};
bool less_int(int a, int b) { Less<int> op; op.bias = 1; return op(a, op.twice(b)); }
kernel void run(device int* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  words[0] = less_int(words[1], words[2]) ? 1 : 0;
  words[4] = less_int(words[5], words[6]) ? 7 : 9;
}
"""

SOURCES = {
    "operator-only": OPERATOR_ONLY,
    "members-and-chains": MEMBERS_AND_CHAINS,
    "returned-call-argument": RETURNED_CALL_ARGUMENT,
}
WORDS = [0, 5, 2, 1, 0, 3, 2, 0]


def _write(tmp_path, name):
    source = tmp_path / f"{name}.metal"
    source.write_text(SOURCES[name], encoding="utf-8")
    return source


def _round_trip(tmp_path, name):
    crossgl = tmp_path / f"{name}.cgl"
    crossgl.write_text(
        crosstl.translate(str(_write(tmp_path, name)), backend="cgl"),
        encoding="utf-8",
    )
    return crosstl.translate(str(crossgl), backend="metal", format_output=False)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_functor_members_reach_crossgl_as_functions(tmp_path, name):
    crossgl = crosstl.translate(str(_write(tmp_path, name)), backend="cgl")

    # Every call on the functor is a call of its lowered member; no generic body keeps `op(...)` or `op.twice(...)`.
    assert re.search(r"\bLess_\w+__operator_call\(op, ", crossgl), crossgl
    assert not re.search(r"\bop\s*\(|\bop\.twice\b", crossgl), crossgl
    assert "Less<T> op" not in crossgl, crossgl


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_functor_metal_output_compiles(tmp_path, name):
    for label, metal in (
        ("direct", crosstl.translate(str(_write(tmp_path, name)), backend="metal")),
        ("round-trip", _round_trip(tmp_path, name)),
    ):
        assert not re.search(r"\bLess<", metal), metal
        (tmp_path / label).mkdir()
        compile_metal_library(metal, tmp_path / label, REQUIRE_ENV)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_functor_computes_like_the_source_on_metal(tmp_path, name):
    (tmp_path / "native").mkdir()
    (tmp_path / "translated").mkdir()
    native = dispatch_word_buffer(
        SOURCES[name], tmp_path / "native", WORDS, REQUIRE_ENV
    )
    translated = dispatch_word_buffer(
        _round_trip(tmp_path, name), tmp_path / "translated", WORDS, REQUIRE_ENV
    )
    assert translated == native


@pytest.mark.parametrize(
    "name, expected",
    [
        ("operator-only", {0: 0, 3: 7}),
        ("members-and-chains", {0: 1, 4: 7}),
        ("returned-call-argument", {0: 0, 4: 7}),
    ],
)
def test_functor_translation_writes_the_expected_words(tmp_path, name, expected):
    # Worked by hand, so the parity test cannot pass with a kernel that writes nothing. operator-only: 5 < 2 is
    # false, 0.0 < 2.5 true. members-and-chains (bias 1): 5 < 4 + 1 false and 2 < 2 + 1 true give 1; 3.0 < 6.0.
    # returned-call-argument: 5 < 4 + 1 false; 3 < 4 + 1 true.
    words = dispatch_word_buffer(
        _round_trip(tmp_path, name), tmp_path, WORDS, REQUIRE_ENV
    )
    assert {index: words[index] for index in expected} == expected


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_functor_translates_for_other_targets(tmp_path, backend):
    output = crosstl.translate(
        str(_write(tmp_path, "members-and-chains")), backend=backend
    )
    # GLSL reserves `__` in identifiers, so its output spells the lowered members with one underscore.
    assert re.search(r"\bLess_float__?operator_call\(op, a, Less_float__?twice\(op, b\)\)", output)
    assert re.search(r"\bLess_int__?operator_call\(op, a, Less_int__?twice\(op, b\)\)", output)


def test_functor_spirv_output_validates(tmp_path):
    spirv = crosstl.translate(
        str(_write(tmp_path, "members-and-chains")), backend="vulkan"
    )
    assert "OpEntryPoint GLCompute" in spirv
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        pytest.skip("spirv-tools is not installed")
    (tmp_path / "functor.spvasm").write_text(spirv, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", "functor.spvasm", "-o", "f.spv"],
        ["spirv-val", "--target-env", "vulkan1.0", "f.spv"],
    ):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
