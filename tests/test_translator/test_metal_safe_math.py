"""Metal output keeps the IEEE floating-point rule of CUDA, HIP and OpenCL C under Metal's default fast math.

See crosstl/translator/codegen/metal_safe_math.py and the fp_ieee family of rule_table.py. Execution evidence: the
conformance cells c_sin, u_sinf, u_atan2f_zero and u_fmodf_sign on metal-apple under the relaxed configuration.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_safe_math import (
    METAL_SAFE_MATH_PRAGMA,
    precise_namespace_eligible,
)
from crosstl.translator.semantics import apply_rule_carrying
from crosstl.translator.semantics.rule_table import (
    FP_IEEE,
    TARGET_ENTRIES,
    source_families,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

CUDA = """__global__ void k(const float *a, float *o) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = sinf(a[i]) + fmodf(a[i], 3.0f) + atan2f(a[i], a[i + 1]) + fmaxf(a[i], a[i + 2]);
}
"""

CROSSGL = """#pragma cgl_rules(fp_ieee)
shader S {
    half f(half x) { return sin(x); }
    float g(float x, int n) { return sin(x) + float(max(n, 2)); }
    compute {
        void main() { }
    }
}
"""

OPENCL = """__kernel void k(__global const float *a, __global float *o) {
    size_t i = get_global_id(0);
    o[i] = sin(a[i]) * sqrt(a[i]);
}
"""

GLSL = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer A { float a[]; };
layout(std430, binding = 1) buffer O { float o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    o[i] = sin(a[i]);
}
"""

METAL = """#include <metal_stdlib>
using namespace metal;
kernel void k(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    o[i] = sin(a[i]);
}
"""


def _translate(tmp_path, name, text, backend):
    path = tmp_path / name
    path.write_text(text)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _body(code):
    return code.split("kernel void", 1)[1]


def test_cuda_hip_and_opencl_define_fp_ieee_and_every_target_discharges_it():
    for source in ("cuda", "hip", "opencl"):
        assert FP_IEEE in source_families(source), source
    for source in ("metal", "directx", "opengl", "vulkan"):
        assert FP_IEEE not in source_families(source), source
    assert TARGET_ENTRIES[(FP_IEEE, "metal")].kind == "polyfill"
    assert TARGET_ENTRIES[(FP_IEEE, "directx")].configuration == "dxc -Gis"


def test_cuda_to_metal_selects_safe_math_and_precise_functions(tmp_path):
    metal = _translate(tmp_path, "k.cu", CUDA, "metal")

    assert metal.lstrip().startswith(METAL_SAFE_MATH_PRAGMA + "\n#include <metal_stdlib>")
    body = _body(metal)
    for name in ("sin", "fmod", "atan2", "fmax"):
        assert f"metal::precise::{name}(" in body, name


def test_half_and_integer_operands_keep_the_plain_call(tmp_path):
    metal = _translate(tmp_path, "k.cgl", CROSSGL, "metal")

    assert METAL_SAFE_MATH_PRAGMA in metal
    # metal::precise declares float overloads only.
    assert "half f(half x) {\n    return sin(x);" in metal
    assert "metal::precise::sin(x) + float(max(n, 2))" in metal


def test_opencl_to_metal_selects_safe_math(tmp_path):
    metal = _translate(tmp_path, "k.cl", OPENCL, "metal")

    assert METAL_SAFE_MATH_PRAGMA in metal
    body = _body(metal)
    assert "metal::precise::sin(" in body and "metal::precise::sqrt(" in body


@pytest.mark.parametrize(
    ("name", "text"), [("k.comp", GLSL), ("k.metal", METAL)], ids=["glsl", "metal"]
)
def test_sources_without_fp_ieee_keep_plain_metal_calls(tmp_path, name, text):
    metal = _translate(tmp_path, name, text, "metal")

    assert "math_mode" not in metal
    assert "metal::precise::" not in metal
    assert "sin(" in _body(metal)


def test_the_crossgl_of_a_cuda_program_carries_fp_ieee(tmp_path):
    crossgl = _translate(tmp_path, "k.cu", CUDA, "cgl")

    assert crossgl.startswith("#pragma cgl_rules(fp_ieee)\n")
    saved = tmp_path / "saved.cgl"
    saved.write_text(crossgl)
    carried = crosstl.translate(str(saved), backend="metal", format_output=False)
    direct = _translate(tmp_path, "k.cu", CUDA, "metal")
    # The direct route also reorders the entry's parameters; the math is the same.
    for metal in (carried, direct):
        assert metal.lstrip().startswith(METAL_SAFE_MATH_PRAGMA)
    assert _body(carried).split("{", 1)[1] == _body(direct).split("{", 1)[1]


def test_native_targets_record_the_configuration_they_rely_on():
    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(
        "#pragma cgl_rules(fp_ieee)\nshader S { compute { void main() { } } }\n"
    )
    report = apply_rule_carrying(program, "cgl", "directx")

    assert report.configurations == {"dxc -Gis"}
    assert "metal.safe_math" not in program.annotations


@pytest.mark.parametrize(
    ("name", "types", "eligible"),
    [
        ("sin", ["float"], True),
        ("fmod", ["float3", "float3"], True),
        ("clamp", ["float4", "float4", "float4"], True),
        ("sin", ["half"], False),
        ("max", ["int", "int"], False),
        ("max", ["float3", "float"], False),
        ("ldexp", ["float", "int"], False),
        ("mix", ["float", "float", "float"], False),
        ("sin", [], False),
    ],
)
def test_precise_namespace_needs_one_binary32_type(name, types, eligible):
    assert precise_namespace_eligible(name, types) is eligible


def _metal_compiler():
    if not shutil.which("xcrun"):
        return False
    probe = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True)
    return probe.returncode == 0


@pytest.mark.parametrize("option", ["-ffast-math", "-fno-fast-math"])
def test_safe_math_output_compiles_in_either_math_mode(tmp_path, option):
    if not _metal_compiler():
        pytest.skip("the Metal compiler is not available")
    source = tmp_path / "k.metal"
    source.write_text(_translate(tmp_path, "k.cu", CUDA, "metal"))
    result = subprocess.run(
        ["xcrun", "metal", "-c", str(source), "-o", str(tmp_path / "k.air"), option],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
