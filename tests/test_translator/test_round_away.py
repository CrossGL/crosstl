"""The round_away rule: C11 ``round`` keeps halfway cases away from zero, and ``rint`` rounds them to even.

CUDA, HIP and OpenCL C ``round``/``roundf`` used to become CrossGL ``round``, whose halfway direction is the target's:
HLSL and WGSL round to even and GLSL leaves it to the implementation, so ``roundf(2.5f)`` could give 2. ``rintf`` and
``nearbyintf`` used to become ``round`` too, which Metal rounds away from zero. See the round_away family in
crosstl/translator/semantics/rule_table.py and the conformance cells u_roundf_half, c_round_half and u_rintf_half.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import RuleInexpressibleError
from crosstl.translator.semantics.rule_table import (
    HELPERS,
    ROUND_AWAY,
    TARGET_ENTRIES,
    source_families,
)

CUDA = """__global__ void k(const float *a, int *n, float *o) {{
    unsigned int i = threadIdx.x;
    o[i] = {call};
}}
"""

OPENCL = """__kernel void k(__global const float *a, __global float *o, __global float4 *v) {
    size_t i = get_global_id(0);
    o[i] = round(a[i]) + rint(a[i]);
    v[i] = round(v[i]);
}
"""

HELPER = HELPERS["__cgl_round_away"].name


def translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def cuda(tmp_path, call, backend="cgl", suffix=".cu"):
    return translate(tmp_path, CUDA.format(call=call), suffix, backend)


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
@pytest.mark.parametrize("call", ["roundf(a[i])", "round(a[i])"])
def test_c_round_is_carried_as_the_round_away_intrinsic(tmp_path, suffix, call):
    crossgl = cuda(tmp_path, call, suffix=suffix)
    assert "__cgl_round_away(a[i])" in crossgl
    assert "round_away" in crossgl.splitlines()[0]


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
@pytest.mark.parametrize(
    "call", ["rintf(a[i])", "nearbyintf(a[i])", "rint(a[i])", "nearbyint(a[i])"]
)
def test_rint_and_nearbyint_round_to_even(tmp_path, suffix, call):
    crossgl = cuda(tmp_path, call, suffix=suffix)
    assert "roundEven(a[i])" in crossgl
    assert "round(a[i])" not in crossgl


@pytest.mark.parametrize("unsigned", ["", "u"])
def test_cuda_round_to_nearest_integer_conversions_round_to_even(tmp_path, unsigned):
    crossgl = cuda(tmp_path, f"(float)__float2{unsigned}int_rn(a[i])")
    assert "roundEven(a[i])" in crossgl


def test_sources_declare_the_family():
    for source in ("cuda", "hip", "opencl"):
        assert ROUND_AWAY in source_families(source)
    assert ROUND_AWAY not in source_families("opengl")


@pytest.mark.parametrize(
    "backend", ["metal", "directx", "opengl", "vulkan", "wgsl", "slang", "rust"]
)
def test_targets_without_a_proven_round_use_the_helper(tmp_path, backend):
    assert TARGET_ENTRIES[(ROUND_AWAY, backend)].kind == "polyfill"
    output = cuda(tmp_path, "roundf(a[i])", backend=backend)
    if backend == "vulkan":
        # The SPIR-V helper has no OpName; it calls Trunc and never the implementation-defined Round.
        assert "OpFunctionCall" in output and " Trunc " in output
        assert " Round " not in output
    else:
        assert HELPER in output


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_cuda_and_hip_keep_their_own_roundf(tmp_path, backend):
    output = cuda(tmp_path, "roundf(a[i])", backend=backend)
    assert "roundf(a[i])" in output
    assert HELPER not in output


def test_opencl_vector_round_uses_the_lane_helper(tmp_path):
    crossgl = translate(tmp_path, OPENCL, ".cl", "cgl")
    assert "__cgl_round_away(a[i])" in crossgl
    assert "roundEven(a[i])" in crossgl
    metal = translate(tmp_path, OPENCL, ".cl", "metal")
    assert f"{HELPER}4(" in metal
    assert "rint(a[i])" in metal


def test_a_binary64_operand_without_a_helper_is_refused(tmp_path):
    source = """__global__ void k(const double *a, double *o) {
    unsigned int i = threadIdx.x;
    o[i] = round(a[i]);
}
"""
    with pytest.raises(RuleInexpressibleError, match="binary32"):
        translate(tmp_path, source, ".cu", "directx")
    assert "round(a[i])" in translate(tmp_path, source, ".cu", "cuda")


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="needs the Metal toolchain")
def test_the_metal_helper_compiles(tmp_path):
    metal = cuda(tmp_path, "roundf(a[i])", backend="metal")
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
