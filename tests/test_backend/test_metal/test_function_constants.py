"""Metal function constants with a default value import as CrossGL specialization constants.

Metal gives a function constant a default through a second constant,
``constant T x = is_function_constant_defined(raw) ? raw : D;`` (Metal Shading Language Specification, section 5.8.1).
CrossGL has no ``is_function_constant_defined``, so the importer refused the pair as a call to an undeclared function.
It now writes the pair as CrossGL's ``const T x @constant_id(N) = D;``, which GLSL writes as
``layout(constant_id = N)`` and Metal output writes back as the same pair.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.declared_functions import UndeclaredFunctionError

KERNEL = """#include <metal_stdlib>
using namespace metal;
DECLARATIONS
kernel void k(device float* fin [[buffer(0)]], device float* outp [[buffer(1)]],
              uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    outp[i] = BODY;
}
"""

SCALE = (
    "constant int scale_value [[function_constant(0)]];\n"
    "constant int scale = is_function_constant_defined(scale_value) ? scale_value : 3;\n"
)


def _source(tmp_path, declarations, body="fin[i] * float(scale)"):
    path = tmp_path / "probe.metal"
    path.write_text(
        KERNEL.replace("DECLARATIONS", declarations).replace("BODY", body),
        encoding="utf-8",
    )
    return str(path)


def _compile_metal(tmp_path, code):
    if not shutil.which("xcrun"):
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code


def test_defaulted_function_constant_is_a_specialization_constant(tmp_path):
    crossgl = crosstl.translate(_source(tmp_path, SCALE), backend="cgl")

    assert "const int scale @constant_id(0) = 3;" in crossgl
    assert "is_function_constant_defined" not in crossgl
    assert "scale_value" not in crossgl


@pytest.mark.parametrize(
    "declarations, expected",
    [
        (
            "constant bool negate_value [[function_constant(2)]];\n"
            "constant bool negate = is_function_constant_defined(negate_value) ? negate_value : true;\n",
            "const bool negate @constant_id(2) = true;",
        ),
        (
            "constant float gain_value [[function_constant(5)]];\n"
            "constant float gain = is_function_constant_defined(gain_value) ? gain_value : 1.5;\n",
            "const float gain @constant_id(5) = 1.5;",
        ),
    ],
)
def test_bool_and_float_defaults(tmp_path, declarations, expected):
    crossgl = crosstl.translate(_source(tmp_path, declarations), backend="cgl")

    assert expected in crossgl


def test_constants_derived_from_the_default_keep_reading_it(tmp_path):
    declarations = SCALE + "constant int2 steps = int2(scale, 1);\n"
    crossgl = crosstl.translate(
        _source(tmp_path, declarations, "fin[i] * float(steps.x + steps.y)"),
        backend="cgl",
    )

    assert "const int scale @constant_id(0) = 3;" in crossgl
    assert "steps = ivec2(scale, 1);" in crossgl


def test_glsl_writes_a_constant_id_layout(tmp_path):
    glsl = crosstl.translate(_source(tmp_path, SCALE), backend="opengl")

    assert "layout(constant_id = 0) const int scale = 3;" in glsl


def test_metal_round_trip_keeps_the_function_constant(tmp_path):
    metal = crosstl.translate(_source(tmp_path, SCALE), backend="metal")

    assert "[[function_constant(0)]]" in metal
    assert "is_function_constant_defined(" in metal
    _compile_metal(tmp_path, metal)


@pytest.mark.parametrize(
    "declarations, body",
    [
        # The raw constant is read on its own, where CrossGL has no value for an unset constant.
        (SCALE, "fin[i] * float(scale + scale_value)"),
        # The default's type differs from the function constant's.
        (
            "constant int scale_value [[function_constant(0)]];\n"
            "constant float scale = is_function_constant_defined(scale_value) ? scale_value : 3.0;\n",
            "fin[i] * scale",
        ),
    ],
)
def test_other_queries_stay_refused(tmp_path, declarations, body):
    with pytest.raises(UndeclaredFunctionError, match="is_function_constant_defined"):
        crosstl.translate(_source(tmp_path, declarations, body), backend="metal")
