"""Precise evaluation on CUDA, HIP, WGSL, Rust and Mojo (crosstl/translator/codegen/precise_evaluation.py).

CUDA emits every floating multiply of a precisely evaluated function as ``__fmul_rn``, ``__dmul_rn`` or
``__hmul_rn``, which nvcc never merges into a multiply-add. HIP starts the function, and each vector helper it calls,
with ``#pragma clang fp contract(off)``. WGSL, Rust and Mojo cannot keep the evaluation order and refuse.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.precise_evaluation import (
    PreciseEvaluationUnsupportedError,
)

PRECISE_KERNEL = """@compute
@precise
@workgroup_size(64, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> a: array<vec4<f32>>,
    @group(0) @binding(1) var<storage, read_write> o: array<f32>
) {
    var i: i32 = gl_GlobalInvocationID.x;
    var acc: f32 = 1.0f;
    acc *= a[i].x;
    acc += a[i].y * a[i].z;
    var v: vec4<f32> = a[i] * a[i + 1] + a[i + 2];
    v *= 2.0f;
    var d: f64 = f64(acc) * f64(acc) - 1.0;
    o[i] = acc + v.x + f32(d);
}
"""

CONTRACTED_KERNEL = PRECISE_KERNEL.replace("@precise\n", "")

SCALAR_KERNEL = """@compute
@precise
@workgroup_size(64, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> a: array<f32>,
    @group(0) @binding(1) var<storage, read_write> o: array<f32>
) {
    var i: i32 = gl_GlobalInvocationID.x;
    var acc: f32 = 0.0f;
    acc = acc + a[i] * a[i + 2];
    o[i] = a[i] * a[i] - acc;
}
"""

HALF_KERNEL = """@compute
@precise
@workgroup_size(64, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> a: array<f16>,
    @group(0) @binding(1) var<storage, read_write> o: array<f16>
) {
    var i: i32 = gl_GlobalInvocationID.x;
    o[i] = a[i] * a[i + 1] + a[i + 2];
}
"""

MATRIX_KERNEL = """@compute
@precise
@workgroup_size(64, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> a: array<mat4>,
    @group(0) @binding(1) var<storage, read_write> o: array<vec4>
) {
    var i: i32 = gl_GlobalInvocationID.x;
    o[i] = a[i] * o[i + 1];
}
"""

SIDE_EFFECT_TARGET_KERNEL = """@compute
@precise
@workgroup_size(64, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> a: array<f32>,
    @group(0) @binding(1) var<storage, read_write> o: array<f32>
) {
    var i: i32 = gl_GlobalInvocationID.x;
    var j: i32 = i;
    o[j++] *= a[i];
}
"""

# An HLSL ``precise`` local makes its function evaluate precisely.
PRECISE_LOCAL_SHADER = """shader main {
    float g(float a, float b) {
        return (a * b) + 1.0;
    }
    compute {
    @ numthreads(64, 1, 1)
    void main(uvec3 id @ gl_GlobalInvocationID) {
        float y = 2.0 * 3.0;
        precise float x = y * y + 1.0;
    }
    }
}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "k.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def function_text(code, name):
    start = re.search(rf"\b{name}\(", code).start()
    end = code.index("\n}\n", start)
    return code[start:end]


def test_cuda_precise_function_multiplies_without_contraction(tmp_path):
    cuda = translate(tmp_path, PRECISE_KERNEL, "cuda")
    body = function_text(cuda, "k")
    assert "acc = __fmul_rn(acc, a[i].x);" in body
    assert "acc += __fmul_rn(a[i].y, a[i].z);" in body
    assert "__dmul_rn(static_cast<double>(acc), static_cast<double>(acc))" in body
    assert "cgl_float4_mul_precise(a[i], a[(i + 1)])" in body
    assert "v = cgl_float4_mul_scalar_precise(v, 2.0f)" in body
    helper = function_text(cuda, "cgl_float4_mul_precise")
    assert helper.count("__fmul_rn(") == 4
    # A plain floating multiply would let nvcc fuse it with the following add; the index computation is integral.
    assert not re.search(
        r"[\w\]\)] \* [\w(]", body.replace("blockIdx.x * blockDim.x", "")
    )


def test_cuda_function_that_is_not_precise_is_unchanged(tmp_path):
    cuda = translate(tmp_path, CONTRACTED_KERNEL, "cuda")
    assert "__fmul_rn" not in cuda
    assert "__dmul_rn" not in cuda
    assert "_precise" not in cuda
    assert "acc *= a[i].x;" in cuda


def test_cuda_precise_half_multiply_uses_hmul_rn(tmp_path):
    cuda = translate(tmp_path, HALF_KERNEL, "cuda")
    assert "o[i] = (__hmul_rn(a[i], a[(i + 1)]) + a[(i + 2)]);" in cuda


def test_cuda_precise_local_makes_its_function_precise(tmp_path):
    cuda = translate(tmp_path, PRECISE_LOCAL_SHADER, "cuda")
    assert "__fmul_rn(y, y)" in function_text(cuda, "compute_main")
    # ``g`` declares nothing precise.
    assert "__fmul_rn" not in function_text(cuda, "g")


@pytest.mark.parametrize(
    ("backend", "error"),
    [
        ("cuda", "CudaLoweringRefused"),
        ("hip", "HipLoweringRefused"),
    ],
)
def test_precise_matrix_arithmetic_is_refused(tmp_path, backend, error):
    with pytest.raises(Exception) as caught:
        translate(tmp_path, MATRIX_KERNEL, backend)
    assert type(caught.value).__name__ == error
    assert "precise '*' on mat4x4" in str(caught.value)
    assert "no uncontracted form" in str(caught.value)


def test_hip_precise_half_arithmetic_is_refused(tmp_path):
    with pytest.raises(Exception) as caught:
        translate(tmp_path, HALF_KERNEL, "hip")
    assert type(caught.value).__name__ == "HipLoweringRefused"
    assert "precise '*' on f16 and f16" in str(caught.value)


def test_cuda_precise_multiply_assignment_with_side_effects_is_refused(tmp_path):
    with pytest.raises(Exception) as caught:
        translate(tmp_path, SIDE_EFFECT_TARGET_KERNEL, "cuda")
    assert type(caught.value).__name__ == "CudaLoweringRefused"
    assert "target has side effects" in str(caught.value)


def test_hip_precise_function_disables_contraction_for_its_body(tmp_path):
    hip = translate(tmp_path, PRECISE_KERNEL, "hip")
    body = function_text(hip, "k")
    assert re.match(r"k\([^)]*\)\n\{\n\s*#pragma clang fp contract\(off\)\n", body)
    assert (
        "cgl_float4_add_precise(cgl_float4_mul_precise(a[i], a[(i + 1)]), a[(i + 2)])"
        in body
    )
    for helper in ("cgl_float4_mul_precise", "cgl_float4_add_precise"):
        assert re.match(
            rf"{helper}\([^)]*\)\n\{{\n\s*#pragma clang fp contract\(off\)\n",
            function_text(hip, helper),
        )
    # The side-effecting ``*=`` needs no rewrite: the pragma covers it.
    side_effect = translate(tmp_path, SIDE_EFFECT_TARGET_KERNEL, "hip")
    assert "o[j++] *= a[i];" in side_effect


def test_hip_function_that_is_not_precise_is_unchanged(tmp_path):
    hip = translate(tmp_path, CONTRACTED_KERNEL, "hip")
    assert "fp contract" not in hip
    assert "_precise" not in hip


@pytest.mark.parametrize(
    ("backend", "label", "reason"),
    [
        ("wgsl", "WGSL", "WGSL section 15.7.5"),
        ("rust", "Rust", "NoContraction"),
        ("mojo", "Mojo", "no control over floating-point contraction"),
    ],
)
def test_targets_without_precise_evaluation_refuse_it(tmp_path, backend, label, reason):
    with pytest.raises(PreciseEvaluationUnsupportedError) as caught:
        translate(tmp_path, SCALAR_KERNEL, backend)
    error = caught.value
    assert error.code == "project.translate.precise-evaluation-unsupported"
    assert error.missing_capabilities == (f"{backend}.precise-evaluation",)
    assert error.constructs == ("function k",)
    assert str(error).startswith(f"{label} cannot evaluate")
    assert reason in str(error)
    # Without the marker the same program translates.
    assert translate(tmp_path, SCALAR_KERNEL.replace("@precise\n", ""), backend)


def test_precise_local_is_refused_on_wgsl(tmp_path):
    with pytest.raises(PreciseEvaluationUnsupportedError) as caught:
        translate(tmp_path, PRECISE_LOCAL_SHADER, "wgsl")
    assert caught.value.constructs == ("function main",)


def test_nvcc_emits_no_multiply_add_for_a_precise_function(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    for source, fused in (
        (SCALAR_KERNEL, False),
        (SCALAR_KERNEL.replace("@precise\n", ""), True),
    ):
        cuda_path = tmp_path / "k.cu"
        cuda_path.write_text(translate(tmp_path, source, "cuda"))
        ptx_path = tmp_path / "k.ptx"
        subprocess.run(
            [nvcc, "-arch=sm_80", "-O3", "-ptx", str(cuda_path), "-o", str(ptx_path)],
            check=True,
            capture_output=True,
        )
        ptx = ptx_path.read_text()
        assert ("fma.rn.f32" in ptx) is fused
        if not fused:
            assert "mul.rn.f32" in ptx
