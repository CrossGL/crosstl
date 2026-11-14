"""A function the program defines under the name of a DXR intrinsic is the program's own function.

HLSL resolves ``WorldRayOrigin()`` to a function the shader defines with that name. The OpenGL and Vulkan
outputs replaced the call with the GL_EXT_ray_tracing built-in, so the shader read the ray state instead of
calling the function, Metal refused the call as a ray-state read, and DXC read a ``float4 ObjectToWorld()`` call
as the ``float3x4`` intrinsic (#4571).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.dxr_ray_interfaces import DXR_SYSTEM_VALUE_CALLS
from tests.spirv_friendly_names import friendly_spirv

NAMES = sorted(DXR_SYSTEM_VALUE_CALLS)

CGL_CLOSEST_HIT = """
shader main {{
    struct Payload {{
        vec4 color @ ray_payload_read(caller) @ ray_payload_write(anyhit, closesthit);
    }}
    vec4 {name}() {{ return vec4(5.0, 6.0, 7.0, 8.0); }}
    ray_closest_hit {{
        void main(inout Payload payload @ payload, in BuiltInTriangleIntersectionAttributes attributes @ hit_attribute) {{
            payload.color = {name}();
        }}
    }}
}}
"""

HLSL_CLOSEST_HIT = """
struct [raypayload] Payload { float4 color : read(caller) : write(anyhit, closesthit); };
float3 WorldRayOrigin() { return float3(5.0, 6.0, 7.0); }
[shader("closesthit")]
void ClosestHitMain(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    payload.color = float4(WorldRayOrigin(), RayTCurrent());
}
"""

CGL_RAYGEN = """
shader main {
    image2D image @binding(1);
    uvec3 DispatchRaysIndex() { return uvec3(1u, 2u, 3u); }
    ray_generation {
        void main() {
            uvec3 index = DispatchRaysIndex();
            imageStore(image, ivec2(index.xy), vec4(0.0, 0.0, 0.0, 1.0));
        }
    }
}
"""

# GL_EXT_ray_tracing built-ins (and the GLSL names of the Metal launch values) that a DXR intrinsic lowers to.
_BUILTIN = re.compile(r"\bgl_\w+(EXT|ID)\b")


def _translate(tmp_path, source, name, backend, source_backend=None):
    path = tmp_path / name
    path.write_text(source)
    code = crosstl.translate(
        str(path),
        backend=backend,
        source_backend=source_backend,
        format_output=False,
    )
    return friendly_spirv(code) if backend == "vulkan" else code


def _run(tmp_path, *commands):
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    (tmp_path / f"out.{stage}").write_text(code)
    _run(
        tmp_path,
        [
            "glslangValidator",
            "--target-env",
            "vulkan1.2",
            "-o",
            "out.spv",
            f"out.{stage}",
        ],
    )


def _spirv_val(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    (tmp_path / "out.spvasm").write_text(assembly)
    _run(
        tmp_path,
        ["spirv-as", "--target-env", "vulkan1.2", "out.spvasm", "-o", "out.spv"],
        ["spirv-val", "--target-env", "vulkan1.2", "out.spv"],
    )


@pytest.mark.parametrize("name", NAMES)
def test_opengl_calls_a_user_function_named_like_an_intrinsic(tmp_path, name):
    code = _translate(tmp_path, CGL_CLOSEST_HIT.format(name=name), "hit.cgl", "opengl")
    assert re.search(rf"\bvec4 {name}\(\) \{{", code), code
    assert re.search(rf"payload\.color = {name}\(\);", code), code
    assert not _BUILTIN.search(code), code
    _glslang(tmp_path, code, "rchit")


@pytest.mark.parametrize("name", NAMES)
def test_vulkan_calls_a_user_function_named_like_an_intrinsic(tmp_path, name):
    code = _translate(tmp_path, CGL_CLOSEST_HIT.format(name=name), "hit.cgl", "vulkan")
    assert "OpFunctionCall %v4float" in code, code
    # Only the payload and the hit attributes are interface variables; no ray built-in is read.
    assert " BuiltIn " not in code, code
    _spirv_val(tmp_path, code)


@pytest.mark.parametrize("name", NAMES)
def test_metal_calls_a_user_function_named_like_an_intrinsic(tmp_path, name):
    code = _translate(tmp_path, CGL_CLOSEST_HIT.format(name=name), "hit.cgl", "metal")
    assert re.search(rf"\bfloat4 {name}\(\)", code), code
    assert re.search(rf"payload\.color = {name}\(\);", code), code


@pytest.mark.parametrize("name", NAMES)
def test_hlsl_renames_a_user_function_named_like_an_intrinsic(tmp_path, name):
    code = _translate(tmp_path, CGL_CLOSEST_HIT.format(name=name), "hit.cgl", "directx")
    # DXC reads a call to the intrinsic's name as the intrinsic when the return types differ.
    assert re.search(rf"\bfloat4 {name}_\(\) \{{", code), code
    assert re.search(rf"payload\.color = {name}_\(\);", code), code
    assert not re.search(rf"\b{name}\(", code), code


@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal"])
def test_ray_generation_keeps_a_user_dispatch_rays_index(tmp_path, backend):
    code = _translate(tmp_path, CGL_RAYGEN, "raygen.cgl", backend)
    if backend == "vulkan":
        assert "OpFunctionCall %v3uint" in code, code
        assert "BuiltIn LaunchIdKHR" not in code, code
        _spirv_val(tmp_path, code)
        return
    assert re.search(r"index = DispatchRaysIndex\(\);", code), code
    assert "gl_LaunchIDEXT" not in code and "gl_GlobalInvocationID" not in code, code
    if backend == "opengl":
        _glslang(tmp_path, code, "rgen")


def test_hlsl_import_calls_the_user_function_and_reads_the_real_intrinsic(tmp_path):
    glsl = _translate(tmp_path, HLSL_CLOSEST_HIT, "hit.hlsl", "opengl", "directx")
    # The program's WorldRayOrigin is called; RayTCurrent, which it does not define, is still the built-in.
    assert re.search(r"vec4\(WorldRayOrigin\(\), gl_RayTmaxEXT\)", glsl), glsl
    assert "gl_WorldRayOriginEXT" not in glsl, glsl
    _glslang(tmp_path, glsl, "rchit")

    spirv = _translate(tmp_path, HLSL_CLOSEST_HIT, "hit.hlsl", "vulkan", "directx")
    assert "OpFunctionCall %v3float" in spirv, spirv
    assert "BuiltIn RayTmaxKHR" in spirv, spirv
    assert "BuiltIn WorldRayOriginKHR" not in spirv, spirv
    _spirv_val(tmp_path, spirv)
