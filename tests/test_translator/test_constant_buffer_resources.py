"""HLSL ``ConstantBuffer<T>`` resources reach SPIR-V and Metal as one constant buffer each.

Regression tests for issue #4269: Metal codegen must pass ``constant Buf& buf`` to
generic-function specializations, not just to plain (non-generic) functions.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

CONSTANT_BUFFERS_HLSL = """
struct Camera { float4x4 projection; float4 eye; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera : register(b0);
struct Material { float4 tint; float roughness; };
[[vk::binding(1, 0)]] ConstantBuffer<Material> material : register(b1);
struct VSOutput { float4 Pos : SV_POSITION; float3 Tint : TEXCOORD0; float Rough : TEXCOORD1; };
VSOutput main(float3 pos : POSITION0) {
    VSOutput output = (VSOutput) 0;
    output.Pos = mul(camera.projection, float4(pos, 1.0)) + camera.eye;
    output.Tint = material.tint.xyz * material.tint.w;
    output.Rough = material.roughness;
    return output;
}
"""

CONSTANT_BUFFER_ARRAY_HLSL = """
struct Light { float4 color; };
ConstantBuffer<Light> lights[4] : register(b0);
float4 main(float2 uv : TEXCOORD0) : SV_Target { return lights[1].color; }
"""


def _translate(tmp_path, source, backend, name="shader.vert.hlsl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_keeps_constant_buffers_out_of_the_program_constants(tmp_path):
    crossgl = _translate(tmp_path, CONSTANT_BUFFERS_HLSL, "cgl")

    assert "ConstantBuffer<Camera> camera;" in crossgl
    assert "hlsl_program_constant" not in crossgl


def test_spirv_declares_each_constant_buffer_as_a_uniform_block(tmp_path):
    spirv = _translate(tmp_path, CONSTANT_BUFFERS_HLSL, "vulkan")

    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', spirv))
    variables = {
        names.get(var, var): pointer
        for var, pointer in re.findall(r"%(\w+) = OpVariable %(\w+) Uniform\b", spirv)
    }
    assert set(variables) == {"camera", "material"}
    decorations = {
        names.get(var, var): (int(dset), int(binding))
        for var, dset, binding in re.findall(
            r"OpDecorate %(\w+) DescriptorSet (\d+)\s+OpDecorate %\1 Binding (\d+)",
            spirv,
        )
    }
    assert decorations == {"camera": (0, 0), "material": (0, 1)}
    assert len(re.findall(r"OpDecorate %\w+ Block\b", spirv)) == 2
    assert "Unknown type" not in spirv

    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(spirv)
        _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_metal_binds_each_constant_buffer_as_a_constant_reference(tmp_path):
    metal = _translate(tmp_path, CONSTANT_BUFFERS_HLSL, "metal")

    assert re.search(r"constant Camera\s*&\s*camera \[\[buffer\(0\)\]\]", metal)
    assert re.search(r"constant Material\s*&\s*material \[\[buffer\(1\)\]\]", metal)
    assert "ConstantBuffer" not in metal
    assert "HlslProgramConstants" not in metal
    assert "material.roughness" in metal

    if shutil.which("xcrun") is None:
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(metal)
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")]
    )


def test_hlsl_round_trip_keeps_the_constant_buffer_declarations(tmp_path):
    hlsl = _translate(tmp_path, CONSTANT_BUFFERS_HLSL, "directx")

    assert "ConstantBuffer<Camera> camera : register(b0);" in hlsl
    assert "ConstantBuffer<Material> material : register(b1);" in hlsl


def test_spirv_declares_an_array_of_constant_buffers_as_a_descriptor_array(tmp_path):
    spirv = _translate(
        tmp_path, CONSTANT_BUFFER_ARRAY_HLSL, "vulkan", "shader.frag.hlsl"
    )

    # One Uniform variable of four Block structs: four descriptors, not one buffer holding an array.
    assert len(re.findall(r"= OpVariable %\w+ Uniform\b", spirv)) == 1
    assert re.search(r"OpTypeArray %\w+ %\w+", spirv)
    assert len(re.findall(r"OpDecorate %\w+ Block\b", spirv)) == 1
    assert "Unknown type" not in spirv


def test_metal_refuses_arrays_of_constant_buffers(tmp_path):
    reason = "array of constant buffers 'lights': Metal binds an array of buffers only through an argument buffer"
    with pytest.raises(MetalLoweringRefused, match=re.escape(reason)):
        _translate(tmp_path, CONSTANT_BUFFER_ARRAY_HLSL, "metal", "shader.frag.hlsl")


# ---------------------------------------------------------------------------
# Regression tests for issue #4269 — cbuffer param missing from generic-function
# specializations on the Metal backend.
# ---------------------------------------------------------------------------

# A generic function that reads a cbuffer member, plus a plain non-generic
# function that reads the same member.  Both must receive ``constant Params& params``
# in the emitted Metal, and every call site must pass the argument.
_GENERIC_CBUFFER_COMPUTE_HLSL = """\
cbuffer Params : register(b0) { float gain; }

template<typename T> T scaled(T v) { return v * T(gain); }
float plainScaled(float v) { return v * gain; }

RWStructuredBuffer<float> output : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    output[id.x] = scaled(output[id.x]) + plainScaled(output[id.x]);
}
"""

# Same generic function used from a fragment shader.
_GENERIC_CBUFFER_FRAG_HLSL = """\
cbuffer Params : register(b0) { float gain; }

template<typename T> T scaled(T v) { return v * T(gain); }

float4 main(float2 uv : TEXCOORD0) : SV_Target {
    return float4(scaled(uv.x), scaled(uv.y), 0.0, 1.0);
}
"""

# A plain (non-generic) wrapper that calls a generic function.  The wrapper
# itself does not directly touch the cbuffer; it must still receive the param
# via transitive dependency propagation.
_GENERIC_CBUFFER_TRANSITIVE_HLSL = """\
cbuffer Params : register(b0) { float gain; }

template<typename T> T scaled(T v) { return v * T(gain); }
float applyScale(float v) { return scaled(v); }

RWStructuredBuffer<float> output : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    output[id.x] = applyScale(output[id.x]);
}
"""


def _xcrun_validate_metal(tmp_path, metal, stem="out"):
    """Compile *metal* with xcrun if the toolchain is available; skip otherwise."""
    if shutil.which("xcrun") is None:
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / f"{stem}.metal"
    source.write_text(metal)
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / f"{stem}.air")]
    )


def test_metal_passes_cbuffer_to_generic_function_specialization_compute(tmp_path):
    """Issue #4269: ``scaled_float`` must receive ``constant Params& params`` (compute)."""
    metal = _translate(
        tmp_path, _GENERIC_CBUFFER_COMPUTE_HLSL, "metal", "shader.comp.hlsl"
    )

    # The generic specialization must carry the cbuffer parameter.
    assert re.search(
        r"float scaled_float\(float v,\s*constant Params\s*&\s*params\)", metal
    ), f"scaled_float missing cbuffer param:\n{metal}"
    # The body must reference params.gain, not a bare 'gain'.
    assert (
        "params.gain" in metal
    ), f"params.gain not found in scaled_float body:\n{metal}"
    # The plain non-generic function must also still have the param (baseline).
    assert re.search(
        r"float plainScaled\(float v,\s*constant Params\s*&\s*params\)", metal
    ), f"plainScaled missing cbuffer param:\n{metal}"
    # The kernel call site must pass params to scaled_float.
    assert (
        "scaled_float(" in metal and ", params)" in metal
    ), f"call site does not pass params to scaled_float:\n{metal}"

    _xcrun_validate_metal(tmp_path, metal, "compute")


def test_metal_passes_cbuffer_to_generic_function_specialization_fragment(tmp_path):
    """Issue #4269: ``scaled_float`` must receive ``constant Params& params`` (fragment)."""
    metal = _translate(
        tmp_path, _GENERIC_CBUFFER_FRAG_HLSL, "metal", "shader.frag.hlsl"
    )

    # Specialization signature.
    assert re.search(
        r"float scaled_float\(float v,\s*constant Params\s*&\s*params\)", metal
    ), f"scaled_float missing cbuffer param in fragment shader:\n{metal}"
    # Body uses params.gain.
    assert "params.gain" in metal, f"params.gain not in scaled_float body:\n{metal}"
    # Fragment entry point receives the cbuffer binding.
    assert re.search(
        r"constant Params\s*&\s*params\s*\[\[buffer\(0\)\]\]", metal
    ), f"fragment entry missing constant Params binding:\n{metal}"
    # Calls inside the fragment pass params.
    assert (
        "scaled_float(" in metal and ", params)" in metal
    ), f"fragment call site does not pass params to scaled_float:\n{metal}"

    _xcrun_validate_metal(tmp_path, metal, "fragment")


def test_metal_passes_cbuffer_transitively_through_non_generic_wrapper(tmp_path):
    """Issue #4269: a plain wrapper calling a generic specialization must also get the cbuffer param."""
    metal = _translate(
        tmp_path, _GENERIC_CBUFFER_TRANSITIVE_HLSL, "metal", "shader.comp.hlsl"
    )

    # The generic specialization itself carries the param.
    assert re.search(
        r"float scaled_float\(float v,\s*constant Params\s*&\s*params\)", metal
    ), f"scaled_float missing cbuffer param:\n{metal}"
    # The plain wrapper that calls scaled<float> must also receive the param
    # via transitive dependency propagation.
    assert re.search(
        r"float applyScale\(float v,\s*constant Params\s*&\s*params\)", metal
    ), f"applyScale missing transitive cbuffer param:\n{metal}"
    # applyScale must pass params through to scaled_float.
    assert "scaled_float(v, params)" in metal or re.search(
        r"scaled_float\(\s*v\s*,\s*params\s*\)", metal
    ), f"applyScale does not forward params to scaled_float:\n{metal}"
    # Kernel call site passes params to applyScale.
    assert (
        "applyScale(" in metal and ", params)" in metal
    ), f"kernel call site does not pass params to applyScale:\n{metal}"

    _xcrun_validate_metal(tmp_path, metal, "transitive")
