"""Program-scope Metal ``auto`` constants take the type of their initializer ([dcl.type.auto.deduct])."""

import shutil
import subprocess

import pytest

import crosstl

UNIT = """
#include <metal_stdlib>
using namespace metal;
struct P { float a; int b; };
constant constexpr auto base = 16;
constexpr constant auto scale = 2.0f;
constant constexpr auto mask = 0xffu;
constant constexpr auto dbl = base * 2;
constant auto offset = float2(0.5f, 1.5f);
constant constexpr auto params = P{1.0f, 3};
constant constexpr auto flag = true;
kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = float(dbl) * scale + offset.y + params.a + float(params.b)
        + float(tid & mask) + (flag ? 1.0f : 0.0f);
}
"""

DEDUCED = {
    "base": ("int", "int", "int"),
    "scale": ("float", "float", "float"),
    "mask": ("uint", "uint", "uint"),
    "dbl": ("int", "int", "int"),
    "offset": ("vec2", "vec2", "float2"),
    "params": ("P", "P", "P"),
    "flag": ("bool", "bool", "bool"),
}


def _translate(tmp_path, backend):
    source = tmp_path / "unit.metal"
    source.write_text(UNIT, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend, format_output=False)


def _run(tool, *args):
    executable = shutil.which(tool)
    if executable is None:
        pytest.skip(f"{tool} is not installed")
    result = subprocess.run([executable, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_crossgl_declares_the_deduced_types(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "auto" not in crossgl
    for name, (cgl_type, _glsl_type, _hlsl_type) in DEDUCED.items():
        assert f"constant {cgl_type} {name} = " in crossgl


def test_glsl_declares_compile_time_constants_and_compiles(tmp_path):
    glsl = _translate(tmp_path, "opengl")
    for name, (_cgl_type, glsl_type, _hlsl_type) in DEDUCED.items():
        assert f"const {glsl_type} {name} = " in glsl
    shader = tmp_path / "unit.comp"
    shader.write_text(glsl, encoding="utf-8")
    _run("glslangValidator", "-S", "comp", str(shader))


def test_hlsl_declares_static_constants(tmp_path):
    hlsl = _translate(tmp_path, "directx")
    for name, (_cgl_type, _glsl_type, hlsl_type) in DEDUCED.items():
        assert f"static const {hlsl_type} {name} = " in hlsl


def test_spirv_assembles_and_validates(tmp_path):
    assembly = tmp_path / "unit.spvasm"
    assembly.write_text(_translate(tmp_path, "vulkan"), encoding="utf-8")
    binary = tmp_path / "unit.spv"
    _run("spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary))
    _run("spirv-val", "--target-env", "vulkan1.1", str(binary))


def test_metal_round_trip_keeps_deduced_types(tmp_path):
    metal = _translate(tmp_path, "metal")
    assert "auto" not in metal
    assert "constant int base = 16;" in metal
    assert "constant float2 offset = " in metal
