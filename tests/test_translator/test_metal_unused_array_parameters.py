"""The Metal generator drops an unused array parameter only where every caller agrees.

Metal output omits a value-array parameter that a helper never reads, and drops the
matching argument at each call site.  Call sites resolve the callee by name, so the
decision has to hold for every signature declared under that name: a prototype has
no body and must defer to its definition, and one overload's unused array must not
remove a parameter of another overload.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            %s
        }
    }
"""

BUFFER_PROTOTYPE = """
shader A {
    buffer uint data[] @binding(0);
    void put(buffer uint values[], uint i);
    void go(uint i) { put(data, i); }
    void put(buffer uint values[], uint i) {
        values[i] = 1u;
    }
%s}
""" % (COMPUTE % "go(0u);")

VALUE_ARRAY_PROTOTYPE = """
shader A {
    buffer float outv[] @binding(0);
    float g(float a[2], float x);
    float h(float x) { float arr[2]; arr[0] = x; arr[1] = x; return g(arr, x); }
    float g(float a[2], float x) { return a[1] + x; }
%s}
""" % (COMPUTE % "outv[0] = h(1.0);")

OVERLOADS = """
shader A {
    buffer float outv[] @binding(0);
    float f(float a[2], float x) { return x; }
    float f(float x, float y) { return x + y; }
%s}
""" % (
    COMPUTE
    % "float arr[2]; arr[0] = 1.0; arr[1] = 2.0; outv[0] = f(arr, 3.0) + f(4.0, 5.0);"
)

PROTOTYPE_ONLY = """
shader A {
    buffer float outv[] @binding(0);
    float e(float a[2], float x);
%s}
""" % (COMPUTE % "float arr[2]; arr[0] = 1.0; arr[1] = 2.0; outv[0] = 0.0;")

UNUSED = """
shader A {
    buffer float outv[] @binding(0);
    float u(float a[2], float x) { return x; }
%s}
""" % (COMPUTE % "float arr[2]; arr[0] = 1.0; arr[1] = 2.0; outv[0] = u(arr, 3.0);")


def _metal(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def _compile(tmp_path, code):
    xcrun = _metal_compiler()
    if xcrun is None:
        pytest.skip("the Metal compiler is not available")
    source = tmp_path / "shader.metal"
    source.write_text(code, encoding="utf-8")
    completed = subprocess.run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.2",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "shader.air"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


def test_buffer_parameter_survives_prototype(tmp_path):
    code = _flat(_metal(tmp_path, BUFFER_PROTOTYPE))
    signatures = re.findall(r"void put\(([^)]*)\)", code)
    assert len(signatures) == 2
    for signature in signatures:
        assert re.search(r"device uint\s*\*\s*values", signature), signature
        assert signature.rstrip().endswith("uint i")
    assert "put(data, i)" in code
    _compile(tmp_path, code)


def test_value_array_parameter_survives_prototype(tmp_path):
    code = _flat(_metal(tmp_path, VALUE_ARRAY_PROTOTYPE))
    assert code.count("float g(float a[2], float x)") == 2
    assert "return g(arr, x);" in code
    _compile(tmp_path, code)


def test_unused_array_of_one_overload_keeps_the_other_overloads_parameters(tmp_path):
    code = _flat(_metal(tmp_path, OVERLOADS))
    assert "float f(float a[2], float x)" in code
    assert "float f(float x, float y)" in code
    assert "f(arr, 3.0) + f(4.0, 5.0)" in code
    _compile(tmp_path, code)


def test_prototype_without_definition_keeps_its_array_parameter(tmp_path):
    code = _flat(_metal(tmp_path, PROTOTYPE_ONLY))
    assert "float e(float a[2], float x);" in code


def test_unused_array_parameter_of_a_single_definition_is_still_dropped(tmp_path):
    code = _flat(_metal(tmp_path, UNUSED))
    assert "float u(float x)" in code
    assert "u(3.0)" in code
    _compile(tmp_path, code)
