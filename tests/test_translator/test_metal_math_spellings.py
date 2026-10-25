"""MSL math functions without a spelling in the other shading languages.

``powr`` and ``exp10`` reached HLSL and GLSL verbatim, as calls to undeclared functions, and SPIR-V refused them.
See :mod:`crosstl.translator.semantics.metal_math_spellings`.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.metal_math_spellings import (
    MetalMathFunctionUnsupportedError,
)

POWERS = """
#include <metal_stdlib>
using namespace metal;

kernel void powers(device float4 *values [[buffer(0)]],
                   uint tid [[thread_position_in_grid]]) {
    float4 x = values[tid];
    values[tid] = powr(x, float4(2.0f)) + exp10(x) + metal::powr(x.z, 3.0f) + fast::exp10(x.w);
}
"""

USER_POWR = """
#include <metal_stdlib>
using namespace metal;

static float powr(float x, float y) { return x * y; }

kernel void powers(device float *values [[buffer(0)]],
                   uint tid [[thread_position_in_grid]]) {
    values[tid] = powr(values[tid], 2.0f);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "powers.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, source_options={"threadgroup_size": (64, 1, 1)}
    )


def _run(command, **kwargs):
    completed = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-2000:]


def _validate(tmp_path, backend, code):
    if backend == "directx" and shutil.which("dxc"):
        path = tmp_path / "powers.hlsl"
        path.write_text(code, encoding="utf-8")
        _run(
            [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(path)],
            timeout=180,
        )
    elif backend == "opengl" and shutil.which("glslangValidator"):
        path = tmp_path / "powers.comp"
        path.write_text(code, encoding="utf-8")
        _run([shutil.which("glslangValidator"), str(path)])
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly, module = tmp_path / "powers.spvasm", tmp_path / "powers.spv"
        assembly.write_text(code, encoding="utf-8")
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(module)]
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", str(module)])


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_powr_and_exp10_take_the_target_spellings(tmp_path, backend):
    code = _translate(tmp_path, POWERS, backend)
    assert "powr" not in code
    assert "exp10" not in code
    assert "pow(x, " in code
    assert "exp2((x * 3.321928094887362))" in code
    _validate(tmp_path, backend, code)


def test_spirv_lowers_powr_and_exp10(tmp_path):
    code = _translate(tmp_path, POWERS, "vulkan")
    assert " Pow " in code
    assert " Exp2 " in code
    _validate(tmp_path, "vulkan", code)


def test_metal_keeps_powr_and_exp10(tmp_path):
    code = _translate(tmp_path, POWERS, "metal")
    assert "powr(x, float4(2.0))" in code
    assert "exp10(x)" in code


def test_a_user_function_named_powr_is_kept(tmp_path):
    code = _translate(tmp_path, USER_POWR, "directx")
    assert "powr(" in code
    assert "pow(" not in code


@pytest.mark.parametrize("function", ["sinpi", "cospi", "tanpi", "asinpi", "atanpi"])
@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_pi_functions_are_refused_until_they_reduce_exactly(
    tmp_path, function, backend
):
    source = POWERS.replace("exp10(x)", f"{function}(x)")
    with pytest.raises(MetalMathFunctionUnsupportedError) as raised:
        _translate(tmp_path, source, backend)
    assert (
        raised.value.project_diagnostic_code
        == f"project.translate.{backend}-metal-math-function-unsupported"
    )
    assert function in str(raised.value)


PI_PROBE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = sinpi(fin[i]) + cospi(fin[i]);
        }
    }
}
"""


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_cuda_and_hip_keep_sinpi_and_cospi(tmp_path, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(PI_PROBE, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    assert "sinpi(fin[i])" in code
    assert "cospi(fin[i])" in code


@pytest.mark.parametrize("function", ["tanpi", "asinpi", "acospi", "atanpi"])
@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_cuda_and_hip_still_refuse_the_pi_functions_they_lack(
    tmp_path, function, backend
):
    path = tmp_path / "probe.cgl"
    path.write_text(PI_PROBE.replace("cospi(", f"{function}("), encoding="utf-8")
    with pytest.raises(MetalMathFunctionUnsupportedError) as raised:
        crosstl.translate(str(path), backend=backend)
    assert function in str(raised.value)
    assert not str(raised.value).startswith("MSL ")
