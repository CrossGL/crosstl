"""CrossGL ``powi``: calls become CrossGL helpers on every target but CUDA and HIP.

``crosstl/translator/semantics/powi_helpers.py`` replaces each ``powi``/``powif`` call with a call of a helper that
multiplies by repeated squaring. ``tests/conformance/cases/x_integer_powers`` executes the helpers against the exact
powers; these tests pin the rewriting, its conversions and its refusals.
"""

from __future__ import annotations

import re

import pytest

import crosstl
from crosstl.translator.semantics.builtin_function_names import BUILTIN_FUNCTION_NAMES
from crosstl.translator.semantics.powi_helpers import (
    FUNCTIONS,
    KEEPING_TARGETS,
    PREFIX,
    PowiUnsupportedError,
    helper_program_source,
)

LOWERING_TARGETS = (
    "metal",
    "directx",
    "opengl",
    "vulkan",
    "wgsl",
    "slang",
    "rust",
    "mojo",
)


def _program(body, functions=""):
    return f"""
shader IntegerPowers {{
    {functions}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* iin;
        buffer uint* uin;
        buffer float* outp;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float x = fin[i];
            int n = iin[i];
            {body}
        }}
    }}
}}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "integer_powers.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


_CALL = re.compile(r"(?<![\w.])powif?\s*\(")


def test_both_names_are_crossgl_builtins():
    assert set(FUNCTIONS) <= BUILTIN_FUNCTION_NAMES


@pytest.mark.parametrize("backend", LOWERING_TARGETS)
def test_calls_become_helpers(tmp_path, backend):
    body = (
        "vec2 v = powi(vec2(x, -x), ivec2(n, 1 - n)); "
        "outp[i] = powi(x, n) + powif(x, uin[i]) + v.y;"
    )
    generated = _translate(tmp_path, _program(body), backend)
    if backend == "vulkan":
        # SPIR-V names no function; the helpers are functions the entry point calls.
        assert generated.count("OpFunctionCall") >= 3
        assert not _CALL.search(generated)
        return
    assert not _CALL.search(generated)
    assert f"{PREFIX}f32(" in generated
    assert f"{PREFIX}f32_v2(" in generated


@pytest.mark.parametrize(
    ("backend", "written"),
    [
        ("hip", "powif(x, n)"),
        ("cuda", "(float)pow((double)(x), (double)(n))"),
        ("cgl", "powi(x, n)"),
    ],
)
def test_cuda_hip_and_crossgl_keep_the_call(tmp_path, backend, written):
    generated = _translate(tmp_path, _program("outp[i] = powi(x, n);"), backend)
    assert written in generated
    assert PREFIX not in generated
    assert backend in KEEPING_TARGETS


def test_the_helper_squares_over_a_bounded_loop():
    source = helper_program_source([(32, 1), (64, 1)])
    assert source.count("for (int i = 0; i < 32; i++)") == 2
    assert "uint m = n < 0 ? uint(-(n + 1)) + 1u : uint(n);" in source
    assert "return float(1) / r;" in source
    assert "return double(1) / r;" in source


def test_each_precision_has_its_helper(tmp_path):
    body = (
        "double d = powi(double(x), n); half h = powi(half(x), n); "
        "outp[i] = float(d) + float(h);"
    )
    generated = _translate(tmp_path, _program(body), "opengl")
    assert f"{PREFIX}f64(double(x), n)" in generated
    assert f"{PREFIX}f16(" in generated
    # A binary16 base computes in binary32 and the result rounds to binary16.
    assert re.search(rf"{PREFIX}f32\(float\(x\), n\)", generated)


def test_powif_computes_a_float_base(tmp_path):
    generated = _translate(
        tmp_path, _program("outp[i] = powif(double(x), n);"), "opengl"
    )
    assert f"{PREFIX}f32(float(double(x)), n)" in generated
    assert f"{PREFIX}f64" not in generated


def test_exponents_convert_to_int_and_splat(tmp_path):
    body = "vec3 v = powi(vec3(x), uin[i]); outp[i] = powi(x, uin[i]) + v.z;"
    generated = _translate(tmp_path, _program(body), "opengl")
    assert f"{PREFIX}f32(x, int(uin[i]))" in generated
    assert f"{PREFIX}f32_v3(vec3(x), ivec3(int(uin[i])))" in generated


def test_a_scalar_base_splats_to_the_exponent_lanes(tmp_path):
    generated = _translate(
        tmp_path,
        _program("vec2 v = powi(x, ivec2(n, 2)); outp[i] = v.x;"),
        "opengl",
    )
    assert f"{PREFIX}f32_v2(vec2(x), ivec2(n, 2))" in generated


def test_helpers_are_added_once(tmp_path):
    generated = _translate(
        tmp_path,
        _program("outp[i] = powi(x, n) + powi(x, 2) + powif(x, n);"),
        "metal",
    )
    assert generated.count(f"float {PREFIX}f32(float x, int n)") == 1


def test_a_program_function_of_the_same_name_is_called(tmp_path):
    functions = "float powi(float a, int b) { return a + float(b); }"
    generated = _translate(
        tmp_path, _program("outp[i] = powi(x, n);", functions), "opengl"
    )
    assert PREFIX not in generated


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("outp[i] = float(powi(n, 2));", "floating-point base"),
        ("outp[i] = powi(x, 2.0);", "integer exponent"),
        ("outp[i] = powi(vec3(x), ivec2(n)).x;", "no lane pairing"),
        ("outp[i] = powi(x, n, n);", "takes 2 operands, not 3"),
    ],
)
def test_other_operands_are_refused(tmp_path, body, message):
    with pytest.raises(PowiUnsupportedError, match=message):
        _translate(tmp_path, _program(body), "metal")


def test_the_reserved_prefix_is_refused(tmp_path):
    functions = f"float {PREFIX}f32(float a, int b) {{ return a; }}"
    with pytest.raises(PowiUnsupportedError, match="reserved"):
        _translate(tmp_path, _program("outp[i] = powi(x, n);", functions), "directx")
