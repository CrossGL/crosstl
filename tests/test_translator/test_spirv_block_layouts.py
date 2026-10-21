"""SPIR-V block layouts: push constants, and layout decorations kept off function and workgroup types.

* A ``cbuffer`` member type is a ``std140`` clone, so an ``ArrayStride``/``Offset`` decoration never reaches the
  type of a ``shared`` or function variable of the same shape (VUID-StandaloneSpirv-None-10684).
* ``@push_constant`` blocks (GLSL ``layout(push_constant)``, HLSL ``[[vk::push_constant]]``, SPIR-V
  ``PushConstant``) are ``PushConstant`` variables with ``std430`` offsets and no descriptor slot.
* A GLSL buffer block whose only member is a sized or multi-dimensional array keeps its block form instead of
  becoming an element buffer of the innermost scalar.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import UnsupportedSPIRVFeatureError
from tests.spirv_friendly_names import friendly_spirv

HAS_SPIRV_TOOLS = bool(shutil.which("spirv-as") and shutil.which("spirv-val"))

SHARED_ARRAY_GLSL = """#version 450
layout(local_size_x = 4) in;
struct Light { vec3 pos; float k[2]; mat3 m; };
layout(std140, binding = 0) uniform Params { float k[3]; Light lights[2]; bool flag; mat4 ms[2]; } params;
layout(std430, binding = 1) buffer Out { float data[]; } outb;
shared float s[3];
shared Light sl;
void main() {
    float f[3] = params.k;
    Light l = params.lights[1];
    mat4 mm[2] = params.ms;
    uint i = gl_LocalInvocationID.x;
    if (i < 3u) { s[i] = params.k[i]; }
    if (i == 0u) { sl = params.lights[0]; }
    barrier();
    float v = s[i % 3u] + f[i % 3u] + l.k[1] + sl.pos.x + l.m[1][1] + mm[1][2][2];
    if (params.flag) v += 1.0;
    outb.data[i] = v;
}
"""

SHARED_ARRAY_HLSL = """
cbuffer Params : register(b0) { float4 k[3]; float w[2]; };
RWStructuredBuffer<float4> outb : register(u0);
groupshared float4 s[3];
[numthreads(4,1,1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    float4 f[3] = k;
    float wl[2] = w;
    if (tid.x < 3) s[tid.x] = k[tid.x];
    GroupMemoryBarrierWithGroupSync();
    outb[tid.x] = s[tid.x % 3] + f[tid.x % 3] + wl[1];
}
"""

PUSH_CONSTANT_GLSL = """#version 450
layout(local_size_x = 4) in;
struct S { float a[2]; vec4 b; };
layout(push_constant) uniform PC { float k[3]; S st; } pc;
layout(std430, binding = 0) buffer Out { float data[]; } outb;
shared float s[3];
void main() {
    uint i = gl_LocalInvocationID.x;
    S local = pc.st;
    if (i < 3u) { s[i] = pc.k[i]; }
    barrier();
    outb.data[i] = s[i % 3u] + local.a[1] + pc.st.b.y;
}
"""

PUSH_CONSTANT_STRUCT_HLSL = """
struct PCData { float4 tint; float k[2]; };
[[vk::push_constant]] ConstantBuffer<PCData> pc;
RWStructuredBuffer<float4> o : register(u0);
[numthreads(1,1,1)] void CSMain(uint3 t : SV_DispatchThreadID) { o[t.x] = pc.tint * pc.k[1]; }
"""

PUSH_CONSTANT_GLOBAL_HLSL = """
struct PushConstants { float4x4 model; float4 color; };
[[vk::push_constant]] PushConstants push_constants;
struct VSOutput { float4 Pos : SV_POSITION; float3 Color : COLOR0; };
VSOutput main(float3 pos : POSITION0) {
    VSOutput output = (VSOutput) 0;
    output.Pos = mul(push_constants.model, float4(pos, 1.0));
    output.Color = push_constants.color.rgb;
    return output;
}
"""

PUSH_CONSTANT_CBUFFER_CROSSGL = """
shader main {
    cbuffer A @push_constant { float a; };
    cbuffer B @push_constant { float b; };
    RWStructuredBuffer<float> o @binding(0);
    compute {
        layout(local_size_x = 1) in;
        void main() { buffer_store(o, 0u, a + b); }
    }
}
"""

FIXED_ARRAY_BUFFER_GLSL = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Fixed { float d[4]; } fixedb;
layout(std430, binding = 1) buffer Rows { float r[][4]; } rows;
void main() {
    float t[4] = fixedb.d;
    rows.r[1][2] = rows.r[0][3] + t[2];
    rows.r[2] = t;
}
"""


def _translate(tmp_path, source, backend, name):
    path = tmp_path / name
    path.write_text(source)
    if backend == "vulkan":
        return friendly_spirv(
            crosstl.translate(str(path), backend=backend, format_output=False)
        )
    return crosstl.translate(str(path), backend=backend)


def _validate_spirv(tmp_path, spirv, env="vulkan1.1"):
    if not HAS_SPIRV_TOOLS:
        return
    assembly = tmp_path / "module.spvasm"
    binary = tmp_path / "module.spv"
    assembly.write_text(spirv)
    for command in (
        ["spirv-as", "--target-env", env, str(assembly), "-o", str(binary)],
        ["spirv-val", "--target-env", env, str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _decorated_types(spirv, decoration):
    return set(re.findall(rf"OpDecorate %(\w+) {decoration}\b", spirv)) | set(
        re.findall(rf"OpMemberDecorate %(\w+) \d+ {decoration}\b", spirv)
    )


def _pointee_types(spirv, storage_class):
    return set(re.findall(rf"= OpTypePointer {storage_class} %(\w+)", spirv))


def _member_offsets(spirv, type_name):
    return [
        int(offset)
        for offset in re.findall(
            rf"OpMemberDecorate %{type_name} \d+ Offset (\d+)", spirv
        )
    ]


@pytest.mark.parametrize(
    "source,name",
    [(SHARED_ARRAY_GLSL, "shader.comp"), (SHARED_ARRAY_HLSL, "shader.comp.hlsl")],
)
def test_block_layout_decorations_stay_off_workgroup_and_function_types(
    tmp_path, source, name
):
    spirv = _translate(tmp_path, source, "vulkan", name)

    laid_out = _decorated_types(spirv, "ArrayStride") | _decorated_types(
        spirv, "Offset"
    )
    for storage_class in ("Workgroup", "Function", "Private"):
        assert not laid_out & _pointee_types(spirv, storage_class), storage_class
    _validate_spirv(tmp_path, spirv)


def test_bool_block_member_loads_as_uint_and_converts(tmp_path):
    spirv = _translate(tmp_path, SHARED_ARRAY_GLSL, "vulkan", "shader.comp")

    assert re.search(r"OpAccessChain %_ptr_Uniform_uint ", spirv)
    assert "OpINotEqual %bool" in spirv


def test_glsl_push_constant_block_is_a_push_constant_variable(tmp_path):
    spirv = _translate(tmp_path, PUSH_CONSTANT_GLSL, "vulkan", "shader.comp")

    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', spirv))
    push_variables = re.findall(r"%(\w+) = OpVariable %\w+ PushConstant\b", spirv)
    assert [names.get(var, var) for var in push_variables] == ["PC"]
    for var in push_variables:
        assert f"OpDecorate %{var} DescriptorSet" not in spirv
        assert f"OpDecorate %{var} Binding" not in spirv
    # std430 (as glslang lays the block out): k[3] is tightly packed and S starts at its vec4 alignment.
    block = _pointee_types(spirv, "PushConstant") & set(
        re.findall(r"OpDecorate %(\w+) Block\b", spirv)
    )
    assert len(block) == 1
    assert _member_offsets(spirv, block.pop()) == [0, 16]
    assert "ArrayStride 16" not in spirv
    _validate_spirv(tmp_path, spirv)


def test_hlsl_vk_push_constant_imports_as_crossgl_push_constant(tmp_path):
    crossgl = _translate(tmp_path, PUSH_CONSTANT_STRUCT_HLSL, "cgl", "shader.comp.hlsl")

    assert "push_constant" in crossgl
    assert "vk_push_constant" not in crossgl


def test_hlsl_push_constant_struct_matches_dxc_layout(tmp_path):
    spirv = _translate(
        tmp_path, PUSH_CONSTANT_STRUCT_HLSL, "vulkan", "shader.comp.hlsl"
    )

    push_variables = re.findall(r"%(\w+) = OpVariable (%\w+) PushConstant\b", spirv)
    assert len(push_variables) == 1
    var, _pointer = push_variables[0]
    assert f"OpDecorate %{var} Binding" not in spirv
    assert re.search(r"OpAccessChain %_ptr_PushConstant_v4float ", spirv)
    # DXC lays [[vk::push_constant]] out with vector-relaxed std430: offsets 0 and 16, float[2] stride 4.
    block = (
        _pointee_types(spirv, "PushConstant") & _decorated_types(spirv, "Offset")
    ).pop()
    assert _member_offsets(spirv, block) == [0, 16]
    assert "ArrayStride 4" in spirv
    _validate_spirv(tmp_path, spirv)


def test_hlsl_plain_push_constant_global_is_not_a_private_variable(tmp_path):
    crossgl = _translate(tmp_path, PUSH_CONSTANT_GLOBAL_HLSL, "cgl", "shader.vert.hlsl")
    assert "ConstantBuffer<PushConstants> push_constants;" in crossgl
    assert "hlsl_program_constant" not in crossgl

    spirv = _translate(
        tmp_path, PUSH_CONSTANT_GLOBAL_HLSL, "vulkan", "shader.vert.hlsl"
    )

    assert re.search(r"%push_constants = OpVariable %\w+ PushConstant\b", spirv)
    assert "OpTypePointer Private" not in spirv
    _validate_spirv(tmp_path, spirv)


def test_spirv_push_constant_round_trips(tmp_path):
    if not shutil.which("glslangValidator") or not shutil.which("spirv-dis"):
        pytest.skip("glslangValidator and spirv-dis are required")
    source = tmp_path / "pc.comp"
    source.write_text(PUSH_CONSTANT_GLSL)
    binary = tmp_path / "pc.spv"
    subprocess.run(
        ["glslangValidator", "-V", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    disassembly = subprocess.run(
        ["spirv-dis", str(binary)], check=True, capture_output=True, text=True
    ).stdout

    spirv = _translate(tmp_path, disassembly, "vulkan", "pc.spvasm")

    assert len(re.findall(r"= OpVariable %\w+ PushConstant\b", spirv)) == 1
    _validate_spirv(tmp_path, spirv)


def test_second_push_constant_block_is_refused(tmp_path):
    with pytest.raises(UnsupportedSPIRVFeatureError, match="push-constant"):
        _translate(tmp_path, PUSH_CONSTANT_CBUFFER_CROSSGL, "vulkan", "shader.cgl")


def test_glsl_fixed_and_nested_array_buffer_blocks_keep_their_block_form(tmp_path):
    crossgl = _translate(tmp_path, FIXED_ARRAY_BUFFER_GLSL, "cgl", "shader.comp")

    assert "RWStructuredBuffer<float> fixedb" not in crossgl
    assert "RWStructuredBuffer<float> rows" not in crossgl
    assert "float r[][4];" in crossgl
    assert "rows.r[1][2] = " in crossgl
    assert "buffer_load(" not in crossgl

    spirv = _translate(tmp_path, FIXED_ARRAY_BUFFER_GLSL, "vulkan", "shader.comp")
    _validate_spirv(tmp_path, spirv)
