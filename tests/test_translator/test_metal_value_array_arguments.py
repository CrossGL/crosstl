"""Read-only arrays and array call results passed to by-value array parameters in Metal output.

A by-value CrossGL array parameter is a C array parameter in MSL, a non-``const`` ``thread`` pointer. A function-scope
``const`` array is ``const`` thread storage, whose pointer does not convert, and the emitter refused the call. It is
now passed as a thread copy. A user function returns a CrossGL array as ``metal::array<T, N>``, which does not
convert to ``T*`` either, so the Metal compiler rejected the call; it is now passed as ``f(...).data()``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

SOURCE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Outp { float o[]; };
float dot4(float v[4], float k) {
    v[0] = 100.0;
    return v[0] + v[1] * k + v[3];
}
float[4] make(float k) { return float[4](k, k + 1.0, k + 2.0, k + 3.0); }
void main() {
    const float L[4] = float[4](13.0, 14.0, 15.0, 16.0);
    float k = o[8];
    o[0] = dot4(L, k);
    o[1] = dot4(make(k), 2.0);
    o[2] = L[0] + L[3];
}
"""

NESTED = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Outp { float o[]; };
float first(float m[2][2]) { return m[1][0]; }
float[2][2] make2(float k) { return float[2][2](float[2](k, k), float[2](k, k)); }
void main() { o[0] = first(make2(o[1])); }
"""


def _translate(tmp_path, source):
    path = tmp_path / "values.comp"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def _compact(text):
    return re.sub(r"\s+", "", text)


def test_const_local_array_is_passed_as_a_thread_copy(tmp_path):
    code = _compact(_translate(tmp_path, SOURCE))
    assert "dot4(crosstl_array_value<float,4>{{L[0],L[1],L[2],L[3]}}.v,k)" in code


IN_PARAMETERS = """
shader main {
    float dot4(in float v[4], float k) { v[0] = 100.0; return v[0] + v[1] * k + v[3]; }
    float read4(in float v[4]) { return v[0] + v[3]; }
    compute {
        buffer float* o;
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            const float L[4] = float[4](13.0, 14.0, 15.0, 16.0);
            o[0] = dot4(L, o[8]) + read4(L);
        }
    }
}
"""


def test_const_local_array_is_copied_for_an_in_parameter(tmp_path):
    # An ``in`` array parameter is a non-const C array in MSL whether or not the callee writes it.
    path = tmp_path / "in_parameters.cgl"
    path.write_text(IN_PARAMETERS, encoding="utf-8")
    code = _compact(crosstl.translate(str(path), backend="metal"))
    copy = "crosstl_array_value<float,4>{{L[0],L[1],L[2],L[3]}}.v"
    assert f"dot4({copy},o[8])" in code
    assert f"read4({copy})" in code


def test_array_call_result_is_passed_through_its_data(tmp_path):
    code = _compact(_translate(tmp_path, SOURCE))
    assert "array<float,4>make(floatk)" in code
    assert "dot4(make(k).data(),2.0)" in code


def test_nested_array_call_result_is_refused(tmp_path):
    with pytest.raises(MetalLoweringRefused, match="array<array<T, M>, N>"):
        _translate(tmp_path, NESTED)


def test_metal_compiles_the_arguments(tmp_path):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "values.metal").write_text(
        _translate(tmp_path, SOURCE), encoding="utf-8"
    )
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "values.metal", "-o", "values.air"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr
