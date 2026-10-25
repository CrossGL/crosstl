"""Metal and C float limit constants lower to exact values on every target.

``INFINITY``, ``HUGE_VALF``, ``NAN``, ``MAXFLOAT``, ``FLT_MAX``, ``FLT_MIN`` and ``FLT_EPSILON`` are predefined in
Metal. CrossGL keeps them as identifiers. The HLSL and GLSL generators emitted all but ``NAN`` verbatim, which left
undeclared identifiers in the output, and the SPIR-V generator refused all of them (and ``M_PI_F``).
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_standard_constants import (
    spirv_non_finite_float_literal,
)
from crosstl.translator.float_limit_constants import (
    FLOAT_LIMIT_CONSTANTS,
    float_limit_constant,
    render_float_limit_constant,
)

MACROS = (
    "INFINITY",
    "HUGE_VALF",
    "NAN",
    "MAXFLOAT",
    "FLT_MAX",
    "FLT_MIN",
    "FLT_EPSILON",
)

SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void limits(device float *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    float lowest = -INFINITY;
    float quiet = NAN;
    lowest = max(lowest, out[i]);
    out[i] = lowest + M_PI_F + MAXFLOAT + HUGE_VALF + FLT_MAX + FLT_MIN + FLT_EPSILON + quiet;
}
"""

SHADOWING_SOURCE = """
shader S {
    compute {
        RWStructuredBuffer<float> out_ @buffer(0);
        @numthreads(1, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            float FLT_MAX = 2.0;
            buffer_store(out_, i, FLT_MAX + INFINITY);
        }
    }
}
"""


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"limits.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_registry_records_exact_binary32_encodings():
    assert set(FLOAT_LIMIT_CONSTANTS) == set(MACROS)
    assert float_limit_constant("FLT_MAX").bits == 0x7F7FFFFF
    assert float_limit_constant("FLT_MIN").bits == 0x00800000
    assert float_limit_constant("FLT_EPSILON").bits == 0x34000000
    assert float_limit_constant("INFINITY").bits == 0x7F800000
    assert float_limit_constant("NAN").bits == 0x7FC00000
    for constant in FLOAT_LIMIT_CONSTANTS.values():
        assert _bits(constant.value) == constant.bits
        if constant.is_finite:
            # The decimal spelling rounds to exactly the recorded encoding.
            assert _bits(float(constant.decimal)) == constant.bits
    assert float_limit_constant("NAN").value is float_limit_constant("NAN").value
    assert float_limit_constant("DBL_MAX") is None
    assert float_limit_constant(None) is None


@pytest.mark.parametrize(
    ("name", "target", "expected"),
    [
        ("INFINITY", "directx", "asfloat(0x7f800000u)"),
        ("HUGE_VALF", "hlsl", "asfloat(0x7f800000u)"),
        ("NAN", "opengl", "uintBitsToFloat(0x7fc00000u)"),
        ("FLT_MAX", "glsl", "3.40282346638528859811704183484516925e+38f"),
        ("FLT_EPSILON", " HLSL ", "1.19209289550781250000000000000000000e-7f"),
    ],
)
def test_render_uses_target_spellings(name, target, expected):
    assert render_float_limit_constant(name, target) == expected


def test_render_rejects_unknown_names_and_targets():
    assert render_float_limit_constant("M_PI_F", "hlsl") is None
    assert render_float_limit_constant("INFINITY", "metal") is None
    assert render_float_limit_constant("INFINITY", None) is None


def test_spirv_writes_non_finite_values_as_hex_floats():
    infinity = float_limit_constant("INFINITY").value
    assert spirv_non_finite_float_literal(infinity, "float") == "0x1p+128"
    assert spirv_non_finite_float_literal(-infinity, "float") == "-0x1p+128"
    assert spirv_non_finite_float_literal(infinity, "double") == "0x1p+1024"
    assert spirv_non_finite_float_literal(float("nan"), "float") == "0x1.8p+128"
    assert spirv_non_finite_float_literal(1.5, "float") is None
    assert spirv_non_finite_float_literal(3, "int") is None


@pytest.mark.parametrize(
    ("backend", "spellings"),
    [
        (
            "directx",
            (
                "asfloat(0x7f800000u)",
                "asfloat(0x7fc00000u)",
                "3.40282346638528859811704183484516925e+38f",
            ),
        ),
        (
            "opengl",
            (
                "uintBitsToFloat(0x7f800000u)",
                "uintBitsToFloat(0x7fc00000u)",
                "1.17549435082228750796873653722224568e-38f",
            ),
        ),
        ("vulkan", ("0x1p+128", "0x1.8p+128")),
    ],
)
def test_metal_limits_lower_on_every_target(tmp_path, backend, spellings):
    code = _translate(tmp_path, SOURCE, "metal", backend)
    for macro in MACROS:
        assert macro not in code
    assert "M_PI_F" not in code
    for spelling in spellings:
        assert spelling in code


def test_metal_limits_round_trip_unchanged_to_metal(tmp_path):
    code = _translate(tmp_path, SOURCE, "metal", "metal")
    for macro in ("INFINITY", "NAN", "FLT_MAX", "FLT_EPSILON"):
        assert macro in code


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_declared_names_shadow_the_limits(tmp_path, backend):
    code = _translate(tmp_path, SHADOWING_SOURCE, "cgl", backend)
    assert "3.40282346638528859811704183484516925e+38f" not in code
    if backend == "vulkan":
        # Formatted output names the type %float; raw output keeps numeric ids.
        assert re.search(r"= OpConstant %\w+ 2(\.0)?$", code, re.M)
    else:
        assert "FLT_MAX = 2.0" in code


def _run(command, tmp_path):
    return subprocess.run(
        command, capture_output=True, text=True, timeout=120, cwd=tmp_path
    )


def test_spirv_output_assembles_and_validates(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not installed")
    (tmp_path / "limits.spvasm").write_text(
        _translate(tmp_path, SOURCE, "metal", "vulkan"), encoding="utf-8"
    )
    assembled = _run(
        ["spirv-as", "--target-env", "vulkan1.1", "limits.spvasm", "-o", "limits.spv"],
        tmp_path,
    )
    assert assembled.returncode == 0, assembled.stderr[-2000:]
    validated = _run(["spirv-val", "--target-env", "vulkan1.1", "limits.spv"], tmp_path)
    assert validated.returncode == 0, validated.stderr[-2000:]


def test_glsl_output_compiles(tmp_path):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "limits.comp").write_text(
        _translate(tmp_path, SOURCE, "metal", "opengl"), encoding="utf-8"
    )
    compiled = _run(
        ["glslangValidator", "-V", "limits.comp", "-o", "limits.spv"], tmp_path
    )
    assert compiled.returncode == 0, compiled.stdout[-2000:]
