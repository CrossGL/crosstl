"""Functor locals of a function template first instantiated from a member template keep their members (#3530).

``combine<T, Add<T>>(a, b)`` in the member template ``Summer<U>::update<T>`` is concrete only once member lowering
instantiates ``update<uint>``. Its instantiation declares ``apply_helper<uint, Add<uint>> helper;`` and calls
``helper(a, b)``, whose operator calls ``Op op; op(a, b)``. Member lowering used to lower ``Add<uint>`` before the
instantiation existed, so Metal output kept ``apply_helper<uint, Add_uint> helper;`` (which the Metal compiler rejects)
or ``op(a, b)`` on a struct without a call operator. The structs a pending call names in its template arguments now
keep their members until the call is instantiated.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from tests.metal_compute_dispatch import compile_metal_library, dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_TEMPLATE_FUNCTOR_EXECUTION"

# The reproducer of #3530 on one word buffer.
MEMBER_TEMPLATE = """#include <metal_stdlib>
using namespace metal;
template <typename T>
struct Add {
  T operator()(T a, T b) thread { return a + b; }
};
template <typename T, typename Op>
struct apply_helper {
  T operator()(T a, T b) thread {
    Op op;
    return op(a, b);
  }
};
template <typename T, typename Op>
METAL_FUNC T combine(T a, T b) {
  apply_helper<T, Op> helper;
  return helper(a, b);
}
template <typename U>
struct Summer {
  template <typename T>
  T update(T a, T b) thread {
    return combine<T, Add<T>>(a, b);
  }
};
kernel void run(device uint* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  Summer<uint> op;
  words[0] = op.update(words[1], words[2]);
}
"""

# Two operators reach the same helper, one through a member template of a concrete struct; the helper reads a data
# member that the function template sets.
TWO_OPERATORS = """#include <metal_stdlib>
using namespace metal;
template <typename T>
struct Add {
  T operator()(T a, T b) thread { return a + b; }
};
template <typename T>
struct Scaled {
  T operator()(T a, T b) thread { return a * T(3) + b; }
};
template <typename T, typename Op>
struct apply_helper {
  T bias;
  T operator()(T a, T b) thread {
    Op op;
    return op(a, b) + bias;
  }
};
template <typename T, typename Op>
T combine(T a, T b, T bias) {
  apply_helper<T, Op> helper;
  helper.bias = bias;
  return helper(a, b);
}
struct Mixer {
  template <typename T>
  T scaled(T a, T b) thread { return combine<T, Scaled<T>>(a, b, T(1)); }
};
kernel void run(device uint* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  Mixer mixer;
  words[0] = mixer.scaled(words[1], words[2]);
  words[3] = combine<uint, Add<uint>>(words[4], words[5], 0u);
}
"""

SOURCES = {"member-template": MEMBER_TEMPLATE, "two-operators": TWO_OPERATORS}
WORDS = [0, 5, 2, 0, 7, 11, 0, 0]
# member-template: 5 + 2. two-operators: 5 * 3 + 2 + 1 and 7 + 11 + 0.
EXPECTED = {"member-template": {0: 7}, "two-operators": {0: 18, 3: 18}}


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
def test_functor_locals_reach_crossgl_as_member_calls(tmp_path, name):
    crossgl = crosstl.translate(str(_write(tmp_path, name)), backend="cgl")

    # Each functor call is a call of the lowered operator; no instantiation keeps a template spelling or a raw call.
    assert re.search(r"\bapply_helper_\w+__operator_call\(helper, ", crossgl), crossgl
    assert re.search(
        r"\b(Add|Scaled)_uint__operator_call\(op, a, b\)", crossgl
    ), crossgl
    assert not re.search(r"\bapply_helper<\s*uint", crossgl), crossgl
    concrete = re.sub(r"generic<[^>]*>[^{]*\{[^}]*\}", "", crossgl)
    assert not re.search(r"\b(helper|op)\s*\(", concrete), concrete


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_functor_locals_metal_output_compiles(tmp_path, name):
    for label, metal in (
        ("direct", crosstl.translate(str(_write(tmp_path, name)), backend="metal")),
        ("round-trip", _round_trip(tmp_path, name)),
    ):
        (tmp_path / label).mkdir()
        compile_metal_library(metal, tmp_path / label, REQUIRE_ENV)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_functor_locals_compute_like_the_source_on_metal(tmp_path, name):
    (tmp_path / "native").mkdir()
    (tmp_path / "translated").mkdir()
    native = dispatch_word_buffer(
        SOURCES[name], tmp_path / "native", WORDS, REQUIRE_ENV
    )
    translated = dispatch_word_buffer(
        _round_trip(tmp_path, name), tmp_path / "translated", WORDS, REQUIRE_ENV
    )
    assert translated == native
    assert {index: native[index] for index in EXPECTED[name]} == EXPECTED[name]


_VALIDATORS = {
    "opengl": ("functor.comp", [["glslangValidator", "-S", "comp", "functor.comp"]]),
    "vulkan": (
        "functor.spvasm",
        [
            ["spirv-as", "--target-env", "vulkan1.0", "functor.spvasm", "-o", "f.spv"],
            ["spirv-val", "--target-env", "vulkan1.0", "f.spv"],
        ],
    ),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_functor_locals_translate_for_other_targets(tmp_path, backend):
    file_name, commands = _VALIDATORS[backend]
    code = crosstl.translate(str(_write(tmp_path, "two-operators")), backend=backend)
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command, check=False, cwd=tmp_path, capture_output=True, text=True
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
