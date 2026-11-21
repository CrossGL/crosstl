"""Binary64 exp, exp2, log, log2 and pow on targets whose built-ins take binary32 at most.

The OpenGL, Vulkan and DirectX outputs call the CrossGL helpers of crosstl/translator/semantics/libm_binary64 instead
of the built-ins, which have no binary64 form there; targets with binary64 math libraries keep the call. The helpers'
accuracy and special values are measured on their CrossGL text by tests/conformance/libm64_accuracy.py.
"""

import math
import shutil
import subprocess

import numpy as np
import pytest

import crosstl
from crosstl.translator.semantics import Binary64MathError
from crosstl.translator.semantics.libm_binary64 import FUNCTIONS
from tests.conformance import libm64_accuracy

TARGETS = ("opengl", "vulkan", "directx")

CUDA = """extern "C" __global__ void probe(const double* a, const double* b, double* o) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = exp(a[i]) + exp2(a[i]) + log(b[i]) + log2(b[i]) + pow(a[i], b[i]);
}
"""

CROSSGL = """@compute
@workgroup_size(64, 1, 1)
fn probe(
    @group(0) @binding(0) var<storage, read_write> data: array<f64>,
    @group(0) @binding(1) var<storage, read_write> values: array<f32>
) {
    var i: u32 = gl_GlobalInvocationID.x;
    var x: f64 = data[i];
    BODY
}
"""


def translate(tmp_path, source, backend, suffix):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def crossgl(tmp_path, body, backend):
    return translate(tmp_path, CROSSGL.replace("BODY", body), backend, ".cgl")


def _bits(values):
    return np.asarray(values, np.float64).view(np.uint64)


# Accuracy -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("function", sorted(FUNCTIONS))
def test_each_helper_is_within_one_ulp_on_its_samples(function):
    # fdlibm documents exp, log and pow within one ulp; CUDA's binary64 bounds are one ulp for exp, exp2, log and
    # log2 and two for pow. The samples cover every exponent, the reduction boundaries and subnormal results.
    operands = libm64_accuracy.samples(function, count=400)
    worst, where = libm64_accuracy.max_error(function, *operands)
    assert worst < 1.0, (function, worst, where)


def test_exact_cases_are_exact():
    integers = np.arange(-1074, 1024, dtype=np.float64)
    assert np.array_equal(
        _bits(libm64_accuracy.evaluate("exp2", integers)),
        _bits(np.ldexp(1.0, integers.astype(int))),
    )
    powers = np.ldexp(1.0, np.arange(-1074, 1024))
    assert np.array_equal(
        libm64_accuracy.evaluate("log2", powers),
        np.arange(-1074, 1024, dtype=np.float64),
    )
    x = np.array([3.0, -2.0, 10.0, 2.0, 7.0, -3.0, 0.5])
    y = np.array([5.0, 3.0, 22.0, -1074.0, 3.0, 4.0, -3.0])
    want = np.array([243.0, -8.0, 1e22, 2.0**-1074, 343.0, 81.0, 8.0])
    assert np.array_equal(libm64_accuracy.evaluate("pow", x, y), want)
    assert np.array_equal(libm64_accuracy.evaluate("exp", [0.0, -0.0]), [1.0, 1.0])


INF, NAN = math.inf, math.nan

UNARY_SPECIAL_VALUES = {
    "exp": [
        (NAN, NAN),
        (INF, INF),
        (-INF, 0.0),
        (710.0, INF),
        (-746.0, 0.0),
        (1e-300, 1.0),
    ],
    "exp2": [
        (NAN, NAN),
        (INF, INF),
        (-INF, 0.0),
        (1024.0, INF),
        (-1075.0, 0.0),
        (-1074.0, 2.0**-1074),
    ],
    "log": [
        (NAN, NAN),
        (0.0, -INF),
        (-0.0, -INF),
        (-1.0, NAN),
        (-INF, NAN),
        (INF, INF),
        (1.0, 0.0),
    ],
    "log2": [
        (NAN, NAN),
        (0.0, -INF),
        (-0.0, -INF),
        (-1.0, NAN),
        (-INF, NAN),
        (INF, INF),
        (1.0, 0.0),
    ],
}


@pytest.mark.parametrize("function", sorted(UNARY_SPECIAL_VALUES))
def test_unary_special_values_follow_c99_annex_f(function):
    cases = UNARY_SPECIAL_VALUES[function]
    got = libm64_accuracy.evaluate(function, [operand for operand, _ in cases])
    for (operand, want), result in zip(cases, got):
        if math.isnan(want):
            assert math.isnan(result), (function, operand, result)
        else:
            # Bit equality: the sign of a zero result is part of the value.
            assert _bits(result) == _bits(want), (function, operand, result, want)


# C99 Annex F.9.4.4, which CUDA and HIP follow.
POW_SPECIAL_VALUES = [
    (0.0, -3.0, INF),
    (-0.0, -3.0, -INF),
    (0.0, -2.0, INF),
    (-0.0, -2.5, INF),
    (-0.0, 3.0, -0.0),
    (0.0, 3.0, 0.0),
    (-0.0, 2.5, 0.0),
    (-0.0, -INF, INF),
    (-1.0, INF, 1.0),
    (-1.0, -INF, 1.0),
    (1.0, NAN, 1.0),
    (1.0, -INF, 1.0),
    (NAN, 0.0, 1.0),
    (NAN, -0.0, 1.0),
    (-2.0, 0.5, NAN),
    (-1.0, 0.5, NAN),
    (0.5, -INF, INF),
    (2.0, -INF, 0.0),
    (0.5, INF, 0.0),
    (2.0, INF, INF),
    (-INF, -3.0, -0.0),
    (-INF, -2.0, 0.0),
    (-INF, 3.0, -INF),
    (-INF, 2.0, INF),
    (INF, -0.5, 0.0),
    (INF, 0.5, INF),
    (NAN, 2.0, NAN),
    (2.0, NAN, NAN),
    (10.0, 400.0, INF),
    (-10.0, 401.0, -INF),
    (10.0, -400.0, 0.0),
    (-10.0, -401.0, -0.0),
    (1.0 - 2.0**-53, 2.0**70, 0.0),
    (1.0 + 2.0**-52, 2.0**70, INF),
    (-2.0, 2.0**60 + 1.0, INF),
]


def test_pow_special_values_follow_c99_annex_f():
    x = [case[0] for case in POW_SPECIAL_VALUES]
    y = [case[1] for case in POW_SPECIAL_VALUES]
    got = libm64_accuracy.evaluate("pow", x, y)
    for (base, exponent, want), result in zip(POW_SPECIAL_VALUES, got):
        if math.isnan(want):
            assert math.isnan(result), (base, exponent, result)
        else:
            assert _bits(result) == _bits(want), (base, exponent, result, want)


def test_the_evaluator_refuses_a_literal_binary32_cannot_hold():
    from crosstl.translator.source_registry import (
        SOURCE_REGISTRY,
        register_default_sources,
    )

    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(
        "shader T {\n double f(double x) { return x * 0.1; }\n}\n"
    )
    evaluator = libm64_accuracy.Evaluator(program.functions)
    with pytest.raises(libm64_accuracy.Unmodelled, match="not exact in binary32"):
        evaluator.call("f", [np.array([1.0])], np.ones(1, bool))


# Translation ----------------------------------------------------------------------------------------------------
def _calls(text, name):
    import re

    return re.findall(rf"(?<![\w.]){name}\s*\(", text)


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_cuda_binary64_math_calls_the_helpers_on_glsl_and_hlsl(tmp_path, backend):
    code = translate(tmp_path, CUDA, backend, ".cu")
    for function, (helper, _) in FUNCTIONS.items():
        assert _calls(code, helper), (function, backend)
        assert not _calls(code, function), (function, backend)
    assert "Sun Microsystems" in code


def test_cuda_binary64_math_is_valid_spirv(tmp_path):
    code = translate(tmp_path, CUDA, "vulkan", ".cu")
    for name in ("Exp", "Exp2", "Log", "Log2", "Pow"):
        assert f"%1 {name} " not in code
    assert "PackDouble2x32" in code and "UnpackDouble2x32" in code
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        source = tmp_path / "out.spvasm"
        source.write_text(code, encoding="utf-8")
        subprocess.run(
            ["spirv-as", str(source), "-o", str(tmp_path / "out.spv")], check=True
        )
        subprocess.run(["spirv-val", str(tmp_path / "out.spv")], check=True)


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator is not installed"
)
def test_cuda_binary64_math_compiles_with_glslang(tmp_path):
    code = translate(tmp_path, CUDA, "opengl", ".cu")
    shader = tmp_path / "out.comp"
    shader.write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["glslangValidator", "-V", str(shader), "-o", str(tmp_path / "out.spv")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_targets_with_binary64_math_keep_the_built_ins(tmp_path, backend):
    code = translate(tmp_path, CUDA, backend, ".cu")
    assert "cgl_rule_libm64" not in code
    assert _calls(code, "exp") and _calls(code, "pow")


@pytest.mark.parametrize("backend", TARGETS)
def test_binary32_math_keeps_the_built_ins(tmp_path, backend):
    code = crossgl(
        tmp_path, "values[i] = exp(values[i]) + pow(values[i], 2.0);", backend
    )
    assert "cgl_rule_libm64" not in code


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_vector_operands_call_a_per_component_lift(tmp_path, backend):
    body = (
        "var v: vec3<f64> = vec3<f64>(x, x + 1.0, x + 2.0);\n"
        "    var w: vec3<f64> = log(v) + pow(v, x);\n"
        "    data[i] = w.x + w.y + w.z;"
    )
    code = crossgl(tmp_path, body, backend)
    assert _calls(code, "cgl_rule_libm64_log_v3")
    assert _calls(code, "cgl_rule_libm64_pow_v3")
    assert not _calls(code, "log") and not _calls(code, "pow")


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_mixed_operands_convert_to_binary64(tmp_path, backend):
    body = "data[i] = pow(x, values[i]) + pow(x, 3) + pow(2.0, x);"
    code = crossgl(tmp_path, body, backend)
    if backend == "opengl":
        assert "cgl_rule_libm64_pow(x, double(values[i]))" in code
        assert "cgl_rule_libm64_pow(x, 3)" in code
        assert "cgl_rule_libm64_pow(double(2.0), x)" in code
        assert not _calls(code, "pow")
        return
    assert "%1 Pow " not in code
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        source = tmp_path / "out.spvasm"
        source.write_text(code, encoding="utf-8")
        subprocess.run(
            ["spirv-as", str(source), "-o", str(tmp_path / "out.spv")], check=True
        )
        subprocess.run(["spirv-val", str(tmp_path / "out.spv")], check=True)


def test_a_program_function_named_like_the_built_in_keeps_its_calls(tmp_path):
    source = "fn exp(v: f64) -> f64 {\n    return v + 1.0;\n}\n" + CROSSGL.replace(
        "BODY", "data[i] = exp(x);"
    )
    code = translate(tmp_path, source, "opengl", ".cgl")
    assert "cgl_rule_libm64" not in code


def test_a_program_function_with_a_reserved_name_is_refused(tmp_path):
    source = (
        "fn cgl_rule_libm64_exp(v: f64) -> f64 {\n    return v;\n}\n"
        + CROSSGL.replace("BODY", "data[i] = exp(x) + cgl_rule_libm64_exp(x);")
    )
    with pytest.raises(
        Binary64MathError, match="reserved for the binary64 math helpers"
    ):
        translate(tmp_path, source, "opengl", ".cgl")
