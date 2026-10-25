"""``device T&`` structure arguments of Metal entry points are buffers, not stage inputs."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import (
    DeviceReferenceUnsupportedError,
    apply_rule_carrying,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

GRAPHICS = """#include <metal_stdlib>
using namespace metal;
struct Params { float4x4 transform; float3 color; uint index; };
struct VOut { float4 position [[position]]; float3 color; };
vertex VOut v(const device Params& params [[buffer(1)]], uint vid [[vertex_id]]) {
    VOut out;
    out.position = params.transform * float4(float(vid), 0.0, 0.0, 1.0);
    out.color = params.color;
    return out;
}
fragment float4 f(VOut in [[stage_in]], device Params& state [[buffer(1)]]) {
    state.index = 3u;
    return float4(in.color * state.color, 1.0);
}
"""
COMPUTE = """#include <metal_stdlib>
using namespace metal;
struct Params { float scale; uint count; };
void accumulate(device Params& target, float value) { target.scale += value; }
kernel void k(device float& total [[buffer(0)]], device Params& params [[buffer(1)]],
              device float* values [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    values[i] = values[i] * params.scale;
    if (i == 0) {
        total = float(params.count);
        accumulate(params, 1.0);
    }
}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "shader.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_binds_structure_references_as_structured_buffers(tmp_path):
    hlsl = translate(tmp_path, GRAPHICS, "directx")
    assert "StructuredBuffer<Params> params : register(t1);" in hlsl
    assert "RWStructuredBuffer<Params> state : register(u1);" in hlsl
    assert "params[0u].transform" in hlsl
    assert "state[0u].index = 3u;" in hlsl
    # The references used to be stage inputs: `const Params params` and `inout Params state`.
    assert "const Params params" not in hlsl and "inout Params" not in hlsl
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        _run(["dxc", "-T", "vs_6_0", "-E", "VSMain", "out.hlsl"], tmp_path)
        _run(["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"], tmp_path)


def test_glsl_binds_structure_references_as_storage_blocks(tmp_path):
    glsl = translate(tmp_path, GRAPHICS, "opengl")
    assert "readonly buffer paramsBuffer { Params params[]; };" in glsl
    assert "state[0u].index = 3u;" in glsl
    # The members used to be vertex inputs, and `state.color` read the varying.
    assert "in mat4 transform;" not in glsl
    assert "(color * state[0u].color)" in glsl
    if shutil.which("glslangValidator"):
        (tmp_path / "out.glsl").write_text(glsl)
        _run(["glslangValidator", "-S", "vert", "out.glsl"], tmp_path)
        _run(["glslangValidator", "-S", "frag", "out.glsl"], tmp_path)


@pytest.mark.parametrize("source", [GRAPHICS, COMPUTE], ids=["graphics", "compute"])
def test_spirv_reads_structure_references_from_storage_buffers(tmp_path, source):
    path = tmp_path / "shader.metal"
    path.write_text(source)
    spirv = crosstl.translate(str(path), backend="vulkan")
    # The structure used to be read from Input variables, one per member.
    assert "OpTypePointer Input %Params" not in spirv
    assert "OpTypePointer Input %mat4v4float" not in spirv
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(spirv)
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "o.spv"],
            tmp_path,
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", "o.spv"], tmp_path)


def test_helpers_receive_the_buffer_element(tmp_path):
    hlsl = translate(tmp_path, COMPUTE, "directx")
    assert "RWStructuredBuffer<Params> params : register(u1);" in hlsl
    assert "accumulate(params[0u], 1.0)" in hlsl
    # Scalar references keep their existing lowering.
    assert "RWStructuredBuffer<float> total : register(u0);" in hlsl


def test_metal_keeps_the_references(tmp_path):
    metal = translate(tmp_path, GRAPHICS, "metal")
    assert "device Params& state [[buffer(1)]]" in metal


def test_a_local_with_the_argument_name_is_refused():
    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(
        "shader s {\n"
        "    struct P { float x; };\n"
        "    compute {\n"
        "        void main(device P& p @buffer(0)) {\n"
        "            { float p = 1.0; }\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    with pytest.raises(DeviceReferenceUnsupportedError):
        apply_rule_carrying(program, "cgl", "directx")


def test_argument_buffers_are_left_to_the_generators():
    # A structure with resource or [[id(n)]] members is an argument buffer, not plain data.
    from crosstl.translator import ast as A
    from crosstl.translator.semantics.device_references import (
        lower_device_struct_references,
    )

    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(
        "shader s {\n"
        "    struct Args { texture2D t @id(0); uint k @id(1); };\n"
        "    struct Plain { vec4 v; mat3 m; float w[2]; };\n"
        "    struct Nested { Plain p; uint n; };\n"
        "    fragment {\n"
        "        vec4 main(device Args& a @buffer(0), device Nested& n @buffer(1)) @gl_FragColor {\n"
        "            return n.p.v;\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    assert lower_device_struct_references(program, "directx") == 1
    (stage,) = program.stages.values()
    args, nested = stage.entry_point.parameters
    assert isinstance(args.param_type, A.ReferenceType)
    assert nested.param_type.name == "RWStructuredBuffer"
