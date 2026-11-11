"""Metal function templates that name a member type an explicit specialization redefines.

``template <> struct OpMathType<half> { using type = float; };`` makes ``opmath_t<half>`` ``float`` while
``opmath_t<float>`` stays ``float`` and ``opmath_t<int>`` stays ``int``. A function template whose parameter or
body-local alias names ``opmath_t<T>`` therefore computes in another type for each argument, which one CrossGL generic
function cannot express. Each call is materialized for the arguments it deduces, so member lookup selects the
specialization they match; written as a generic over ``T``, ``scale<half>`` would compute in ``half`` and overflow
where the source computes in ``float``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl
from crosstl.backend.Metal.generic_local_aliases import (
    MetalLocalTypeAliasResolutionError,
)

ROOT = Path(__file__).resolve().parents[3]
REQUIRE_ENV = "CROSSTL_REQUIRE_SPECIALIZED_MEMBER_TYPE_EXECUTION"

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

TRAIT = """
template <typename T> struct OpMathType { using type = T; };
template <> struct OpMathType<half> { using type = float; };
template <typename T> using opmath_t = typename OpMathType<T>::type;
"""

NAMESPACED_TRAIT = """
namespace lib { namespace math {
namespace detail {
template <typename T> struct OpMathType { using type = T; };
template <> struct OpMathType<half> { using type = float; };
template <> struct OpMathType<bfloat> { using type = float; };
}
template <typename T> using opmath_t = typename detail::OpMathType<T>::type;
}}
using lib::math::opmath_t;
"""

SCALE = """
template <typename T>
inline T scale(T x, opmath_t<T> factor, T w) {
  using op_T = opmath_t<T>;
  return static_cast<T>((static_cast<op_T>(x) * factor) * static_cast<op_T>(w));
}
"""

DEDUCED_CALLS = TRAIT + SCALE + """
kernel void k(device half* out [[buffer(0)]], device float* outf [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  out[i] = scale(out[i], 0.5f, half(2));
  outf[i] = scale(outf[i], 0.25f, 3.0f);
}
"""

# The source computes 60000 * 2 in float, where half overflows to infinity.
EXECUTED = TRAIT + SCALE + """
kernel void k(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[0] = scale(half(60000.0h), 2.0f, half(0.25h));
  out[1] = scale(half(-60000.0h), 1.5f, half(0.5h));
}
"""
EXPECTED = (30000.0, -44992.0)


def _translate(tmp_path, source, backend="cgl", name="unit.metal"):
    path = tmp_path / name
    path.write_text(PREFIX + source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _signature(code, name):
    match = re.search(rf"^\s*([^\n=]*\b{name}\([^)]*\))\s*(?:@\w+\s*)*\{{", code, re.M)
    assert match is not None, code
    return re.sub(r"\s+", " ", match.group(1)).strip()


def test_deduced_call_is_materialized_with_the_specialized_member_type(tmp_path):
    code = _translate(tmp_path, DEDUCED_CALLS)

    assert "scale(" not in code
    assert _signature(code, "scale_half") == (
        "float16 scale_half(float16 x, float factor, float16 w)"
    )
    assert _signature(code, "scale_float") == (
        "float scale_float(float x, float factor, float w)"
    )
    assert "(float16)(float(x) * factor * float(w));" in code
    assert "op_T" not in code


def test_template_without_a_specialized_member_type_stays_generic(tmp_path):
    source = """
template <typename T> struct Limits { static constant constexpr float max = 1.0f; };
template <> struct Limits<half> { static constant constexpr float max = 2.0f; };
template <typename T> struct Plain { using type = T; };

template <typename T>
T twice(T x) {
  using U = typename Plain<T>::type;
  return U(x) + U(x);
}

kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = twice(out[i]);
}
"""
    code = _translate(tmp_path, source)

    assert "generic<T> T twice(T x)" in code
    assert "twice_float" not in code


def test_alias_template_named_by_a_using_declaration_is_resolved(tmp_path):
    source = NAMESPACED_TRAIT + SCALE + """
kernel void k(device bfloat* out [[buffer(0)]], device half* outh [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  out[i] = scale(out[i], 0.5f, bfloat(2));
  outh[i] = scale(outh[i], 0.5f, half(2));
}
"""
    for backend in ("cgl", "metal"):
        code = _translate(tmp_path, source, backend)

        assert "opmath_t" not in code, backend
        assert "op_T" not in code, backend
    assert _signature(code, "scale_bfloat") == (
        "inline bfloat scale_bfloat(bfloat x, float factor, bfloat w)"
    )
    assert _signature(code, "scale_half") == (
        "inline half scale_half(half x, float factor, half w)"
    )


def test_unresolved_generic_local_alias_has_a_project_diagnostic_code(tmp_path):
    source = """
template <typename T>
T pick(T v) {
  using U = typename T::value_type;
  U x = U(0);
  return v;
}

kernel void k(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  out[id] = pick(float(id));
}
"""
    with pytest.raises(MetalLocalTypeAliasResolutionError) as caught:
        _translate(tmp_path, source)

    assert caught.value.project_diagnostic_code == (
        "project.translate.metal-local-type-alias-unresolved"
    )


def _metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def _run(command):
    completed = subprocess.run(
        [str(item) for item in command], capture_output=True, text=True, timeout=180
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _execute(source, tmp_path, label):
    xcrun = _metal_compiler()
    if xcrun is None or sys.platform != "darwin" or shutil.which("swiftc") is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{REQUIRE_ENV}=1 needs macOS with the Metal toolchain")
        pytest.skip("Metal dispatch needs macOS with the Metal toolchain")
    shader = tmp_path / f"{label}.metal"
    shader.write_text(source, encoding="utf-8")
    air, library = tmp_path / f"{label}.air", tmp_path / f"{label}.metallib"
    _run([xcrun, "-sdk", "macosx", "metal", "-c", shader, "-o", air])
    _run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
    kernel = re.search(r"kernel void (\w+)\(", source)
    assert kernel is not None, source
    payload = tmp_path / f"{label}-buffer-0.in"
    payload.write_bytes(b"\0" * 8)
    runner = tmp_path / "runner"
    if not runner.exists():
        _run(
            [
                "swiftc",
                ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
                "-o",
                runner,
            ]
        )
    request = tmp_path / f"{label}-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": [str(payload)],
                "workgroupCount": [1, 1, 1],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / f"{label}-output"
    _run([runner, library, kernel.group(1), request, output])
    return struct.unpack("<2e", (output / "buffer-0.bin").read_bytes()[:4])


def test_specialized_member_type_computes_in_float_on_metal(tmp_path):
    translated = _translate(tmp_path, EXECUTED, "metal")

    assert _execute(PREFIX + EXECUTED, tmp_path, "source") == EXPECTED
    assert _execute(translated, tmp_path, "translated") == EXPECTED
