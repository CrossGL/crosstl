"""Members of class template specializations that a sibling member calls, and templates only they call.

A member body may call a sibling through the implicit object (``post_in(x)``). When the sibling is overloaded, member
lowering leaves the call for code generation to resolve among the lowered overloads. The pass that drops unused
specialization members ([temp.inst]/4) has to count that call as a use of every overload; it used to count only the
lowered names, so ``RW_float__post_in`` was dropped and the call reached the output as an undeclared ``post_in``.

A function template whose body names a type through an expression of its parameters, such as
``using phase_T = decltype(M_PI_F * T(0));``, has no CrossGL generic spelling, but each instantiation folds the
``decltype``. A template that only members of class template specializations call is instantiated for each of them,
so its generic definition is unused and is left out instead of refusing the program. A generic definition the output
still names stays refused.
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
REQUIRE_ENV = "CROSSTL_REQUIRE_CLASS_TEMPLATE_MEMBER_EXECUTION"

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

OVERLOADED_SIBLINGS = """
template <typename T>
struct RW {
  using complex_T = vec<T, 2>;
  device complex_T* buf;
  bool inv;
  METAL_FUNC complex_T post_in(complex_T elem) const thread {
    return inv ? complex_T(elem.x, -elem.y) : elem;
  }
  METAL_FUNC complex_T post_in(T elem) const thread {
    return complex_T(elem, 0);
  }
  METAL_FUNC void load(T scale) const thread {
    buf[0] = post_in(buf[0]) * scale;
    buf[1] = post_in(scale);
  }
};

kernel void k(device float2* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  RW<float> rw{out, i > 3};
  rw.load(2.0f);
}
"""

PHASE = """
template <typename T>
METAL_FUNC vec<T, 2> twiddle(int k, int p) {
  using phase_T = decltype(M_PI_F * T(0));
  phase_T theta = -phase_T(2) * phase_T(k) * phase_T(M_PI_F) / phase_T(p);
  return {T(metal::fast::cos(theta)), T(metal::fast::sin(theta))};
}
"""

MEMBER_CALLERS = PHASE + """
template <typename in_T>
struct Writer {
  using scalar_T = in_T;
  int n;
  METAL_FUNC vec<scalar_T, 2> get(int k) const thread {
    return twiddle<scalar_T>(k, n);
  }
};

kernel void k(device half2* out_h [[buffer(0)]], device float2* out_f [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  Writer<half> wh{8};
  Writer<float> wf{8};
  out_h[i] = wh.get(int(i));
  out_f[i] = wf.get(int(i));
}
"""

GENERIC_CALLER = PHASE + """
template <typename T>
METAL_FUNC vec<T, 2> use(T x) {
  return twiddle<T>(int(x), 8);
}

kernel void k(device float2* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = use(float(i));
}
"""

# buf[0] = post_in(float2(5, 2)) + phase<float>(2, 4) = (5 - 2) + 2 * pi / 4; buf[1] = post_in(7.0f) = 21.
EXECUTED = """
template <typename T>
METAL_FUNC T phase(int k, int p) {
  using phase_T = decltype(M_PI_F * T(0));
  phase_T theta = phase_T(k) * phase_T(M_PI_F) / phase_T(p);
  return T(theta);
}

template <typename T>
struct RW {
  device T* buf;
  bool inv;
  METAL_FUNC T post_in(vec<T, 2> elem) const thread {
    return inv ? elem.x - elem.y : elem.x + elem.y;
  }
  METAL_FUNC T post_in(T elem) const thread {
    return elem * T(3);
  }
  METAL_FUNC void load(int k) const thread {
    buf[0] = post_in(vec<T, 2>(T(5), T(2))) + phase<T>(k, 4);
    buf[1] = post_in(T(7));
  }
};

kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  RW<float> rw{out, true};
  rw.load(2);
}
"""


def _translate(tmp_path, source, backend="cgl", name="unit.metal"):
    path = tmp_path / name
    path.write_text(PREFIX + source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _definitions(code, name):
    return [
        re.sub(r"\s+", " ", match.group(1)).strip()
        for match in re.finditer(
            rf"^\s*([^\n=;]*\b{name}\([^)]*\))\s*(?:@\w+\s*)*\{{", code, re.M
        )
    ]


def test_overloaded_sibling_members_of_a_specialization_are_kept(tmp_path):
    code = _translate(tmp_path, OVERLOADED_SIBLINGS)

    assert _definitions(code, "RW_float__post_in") == [
        "vec2 RW_float__post_in(in thread RW_float self, vec2 elem)",
        "vec2 RW_float__post_in(in thread RW_float self @maybe_unused, float elem)",
    ]
    assert "self.buf[0] = RW_float__post_in(self, self.buf[0]) * scale;" in code
    assert "self.buf[1] = RW_float__post_in(self, scale);" in code
    assert re.search(r"(?<![\w.])post_in\(", code) is None


def test_overloaded_sibling_members_reach_metal_output(tmp_path):
    code = _translate(tmp_path, OVERLOADED_SIBLINGS, "metal")

    assert code.count("static inline float2 RW_float__post_in(") == 2
    assert re.search(r"(?<![\w.])post_in\(", code) is None


def test_template_called_only_by_specialization_members_has_no_generic_definition(
    tmp_path,
):
    code = _translate(tmp_path, MEMBER_CALLERS)

    assert _definitions(code, "twiddle_half") == ["f16vec2 twiddle_half(int k, int p)"]
    assert _definitions(code, "twiddle_float") == ["vec2 twiddle_float(int k, int p)"]
    assert re.search(r"\btwiddle\b", code) is None
    assert "phase_T" not in code
    assert "float theta = (-float(2)) * float(k) * float(M_PI_F) / float(p);" in code


def test_generic_use_of_an_unspellable_definition_is_refused(tmp_path):
    with pytest.raises(MetalLocalTypeAliasResolutionError) as caught:
        _translate(tmp_path, GENERIC_CALLER)

    assert caught.value.alias_name == "phase_T"
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
    return struct.unpack("<2f", (output / "buffer-0.bin").read_bytes()[:8])


def test_specialization_members_and_their_callee_compute_like_the_source_on_metal(
    tmp_path,
):
    translated = _translate(tmp_path, EXECUTED, "metal")

    source_values = _execute(PREFIX + EXECUTED, tmp_path, "source")
    assert source_values[1] == 21.0
    assert source_values[0] == pytest.approx(3.0 + 3.14159265 / 2, abs=1e-6)
    assert _execute(translated, tmp_path, "translated") == source_values
