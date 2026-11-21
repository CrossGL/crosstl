"""The libm rule: C-family sinf and cosf keep their 2 ulp bound over the whole binary32 range on every target.

CUDA, HIP and OpenCL C ``sinf``/``sin`` and ``cosf``/``cos`` used to become CrossGL ``sin`` and ``cos``, whose precision
is the target's: Vulkan and WGSL bound it on [-pi, pi] only, GLSL not at all, and MoltenVK, KosmicKrisp and WARP
return results far from sin(x) for |x| up to 1e5. See the libm family in crosstl/translator/semantics/rule_table.py,
its helpers in libm_trig.py, tests/conformance/libm_exhaustive.py, and the conformance cells c_sin, u_sinf,
u_sinf_range, u_cosf_range and c_cos_range.
"""

import shutil
import subprocess

import numpy as np
import pytest

import crosstl
from crosstl.translator.semantics import RuleInexpressibleError
from crosstl.translator.semantics.helpers import helper_program_source
from crosstl.translator.semantics.libm_trig import SUPPORT_FUNCTIONS
from crosstl.translator.semantics.rule_table import (
    HELPERS,
    LIBM,
    LIBM_ULP_BOUND,
    TARGET_ENTRIES,
    source_families,
)
from tests.conformance import libm_exhaustive

CUDA = """__global__ void k(const float *a, float *o) {{
    unsigned int i = threadIdx.x;
    o[i] = {call};
}}
"""

OPENCL = """__kernel void k(__global const float *a, __global float *o, __global float4 *v) {
    size_t i = get_global_id(0);
    o[i] = sin(a[i]) + cos(a[i]);
    v[i] = sin(v[i]);
}
"""

SIN = HELPERS["__cgl_libm_sin"].name
COS = HELPERS["__cgl_libm_cos"].name


def translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def cuda(tmp_path, call, backend="cgl", suffix=".cu"):
    return translate(tmp_path, CUDA.format(call=call), suffix, backend)


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
@pytest.mark.parametrize(
    "call, intrinsic",
    [
        ("sinf(a[i])", "__cgl_libm_sin"),
        ("cosf(a[i])", "__cgl_libm_cos"),
        ("sin(a[i])", "__cgl_libm_sin"),
        ("cos(a[i])", "__cgl_libm_cos"),
    ],
)
def test_c_sine_and_cosine_carry_the_libm_intrinsics(tmp_path, suffix, call, intrinsic):
    crossgl = cuda(tmp_path, call, suffix=suffix)
    assert f"{intrinsic}(a[i])" in crossgl
    assert "libm" in crossgl.splitlines()[0]


def test_the_fast_intrinsic_keeps_the_target_sine(tmp_path):
    crossgl = cuda(tmp_path, "__sinf(a[i])")
    assert "__cgl_libm" not in crossgl
    assert "sin(a[i])" in crossgl


def test_sources_declare_the_family():
    for source in ("cuda", "hip", "opencl"):
        assert LIBM in source_families(source)
    for source in ("opengl", "directx", "metal", "vulkan"):
        assert LIBM not in source_families(source)


@pytest.mark.parametrize(
    "backend", ["metal", "directx", "opengl", "vulkan", "wgsl", "slang", "rust", "mojo"]
)
def test_targets_without_a_full_range_bound_use_the_helper(tmp_path, backend):
    assert TARGET_ENTRIES[(LIBM, backend)].kind == "polyfill"
    output = cuda(tmp_path, "sinf(a[i]) + cosf(a[i])", backend=backend)
    if backend == "vulkan":
        # The helpers have no floating-point arithmetic: no Sin or Cos, and no float multiply.
        assert "OpFunctionCall" in output
        assert " Sin " not in output and " Cos " not in output
        assert "OpFMul" not in output
    else:
        assert SIN in output and COS in output
        for name in SUPPORT_FUNCTIONS:
            assert output.count(f"{name}(") >= 2, name


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_cuda_and_hip_keep_their_own_functions(tmp_path, backend):
    output = cuda(tmp_path, "sinf(a[i]) + cosf(a[i])", backend=backend)
    assert "sinf(a[i])" in output and "cosf(a[i])" in output
    assert SIN not in output


def test_opencl_vector_sine_uses_the_lane_helper(tmp_path):
    crossgl = translate(tmp_path, OPENCL, ".cl", "cgl")
    assert "__cgl_libm_sin(a[i])" in crossgl and "__cgl_libm_cos(a[i])" in crossgl
    metal = translate(tmp_path, OPENCL, ".cl", "metal")
    assert f"{SIN}4(" in metal
    # The support functions are defined once, ahead of both helpers.
    assert metal.count("cgl_rule_libm_trig(float") == 1


def test_a_binary64_operand_without_a_helper_is_refused(tmp_path):
    source = """__global__ void k(const double *a, double *o) {
    unsigned int i = threadIdx.x;
    o[i] = sin(a[i]);
}
"""
    with pytest.raises(RuleInexpressibleError, match="binary32"):
        translate(tmp_path, source, ".cu", "directx")
    assert "sin(a[i])" in translate(tmp_path, source, ".cu", "cuda")


def test_a_program_function_with_a_reserved_support_name_is_refused(tmp_path):
    source = """__device__ unsigned int cgl_rule_libm_mulhi(unsigned int a, unsigned int b) { return a; }
__global__ void k(const float *a, float *o) {
    unsigned int i = threadIdx.x;
    o[i] = sinf(a[i]) + (float)cgl_rule_libm_mulhi(i, i);
}
"""
    with pytest.raises(RuleInexpressibleError, match="reserved"):
        translate(tmp_path, source, ".cu", "metal")


def test_the_helpers_stay_within_the_bound_on_a_stratified_sample():
    bits = libm_exhaustive.sample_bits(32)
    for intrinsic in ("__cgl_libm_sin", "__cgl_libm_cos"):
        worst, _ = libm_exhaustive.check(intrinsic, bits)
        assert worst <= LIBM_ULP_BOUND
        # The integer evaluation is nearly correctly rounded.
        assert worst < 0.52


def test_the_helpers_are_integer_only():
    text = helper_program_source("__cgl_libm_sin")
    evaluator, name = libm_exhaustive.evaluator_for("__cgl_libm_sin")
    operand = np.array([1e5], np.float32)
    # The evaluator refuses floating-point arithmetic, so a successful run shows the helper has none.
    evaluator.call(name, [operand], np.ones(1, bool))
    assert "1.0" in text  # the only float literal: cos(x) = 1 for |x| < 2^-12


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="needs the Metal toolchain")
def test_the_metal_helpers_compile(tmp_path):
    metal = translate(tmp_path, OPENCL, ".cl", "metal")
    shader = tmp_path / "k.metal"
    shader.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "k.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("spirv-val") is None, reason="needs SPIRV-Tools")
def test_the_spirv_helpers_validate(tmp_path):
    spvasm = cuda(tmp_path, "sinf(a[i]) + cosf(a[i])", backend="vulkan")
    source = tmp_path / "k.spvasm"
    source.write_text(spvasm, encoding="utf-8")
    binary = tmp_path / "k.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
