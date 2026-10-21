"""Ray tracing system values reach SPIR-V as the decorated inputs GL_EXT_ray_tracing defines.

DXR reads them through intrinsics (``PrimitiveIndex()``, ``RayTCurrent()``) and GLSL through built-in variables
(``gl_PrimitiveID``, ``gl_RayTmaxEXT``). In a hit stage ``gl_PrimitiveID`` and ``gl_InstanceID`` are the hit
primitive and its instance, decorated ``PrimitiveId`` and ``InstanceId`` (Vulkan 1.3 section 15.9), and the hit's
geometry index is ``RayGeometryIndexKHR`` (SPIR-V section 3.2.20, BuiltIn). Every module is checked with spirv-val under
Vulkan 1.2 rules, and the GLSL ones also with glslang.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.builtin_variable_types import (
    crossgl_builtin_variable_type,
)


def _translate(tmp_path, source, name, source_backend, backend="vulkan"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, source_backend=source_backend)


def _spirv_val(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    (tmp_path / "out.spvasm").write_text(assembly)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.2", "out.spvasm", "-o", "out.spv"],
        ["spirv-val", "--target-env", "vulkan1.2", "out.spv"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "--target-env", "vulkan1.2", "-o", "out.spv", str(path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _builtins(assembly):
    """``{variable name: BuiltIn}`` of every built-in the module decorates."""
    return dict(re.findall(r"OpDecorate %(\w+) BuiltIn (\w+)", assembly))


HLSL_CLOSEST_HIT = """
struct Payload { float4 color; uint4 ids; };

[shader("closesthit")]
void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.ids = uint4(PrimitiveIndex(), InstanceIndex(), InstanceID(), GeometryIndex());
    float3 world = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 object = ObjectRayOrigin() + ObjectRayDirection() * RayTMin();
    uint state = RayFlags() + HitKind() + DispatchRaysIndex().x + DispatchRaysDimensions().y;
    payload.color = float4(world + object, float(state) + attribs.barycentrics.x);
}
"""

HLSL_INTERSECTION = """
struct Attributes { float2 bary; };

[shader("intersection")]
void main()
{
    Attributes attribs;
    attribs.bary = float2(float(PrimitiveIndex()), float(InstanceIndex() + InstanceID()));
    ReportHit(RayTCurrent() * 0.5 + float(RayFlags()), 0, attribs);
}
"""

HLSL_MISS = """
struct Payload { float4 color; };

[shader("miss")]
void main(inout Payload payload)
{
    payload.color = float4(WorldRayDirection() * RayTCurrent() + WorldRayOrigin(), float(RayFlags()));
}
"""

GLSL_CLOSEST_HIT = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec4 payload;
hitAttributeEXT vec2 bary;
void main()
{
    vec3 p = gl_ObjectToWorldEXT * vec4(gl_ObjectRayOriginEXT, 1.0);
    vec3 q = gl_WorldToObjectEXT * vec4(gl_WorldRayOriginEXT, 1.0);
    uint ids = uint(gl_PrimitiveID + gl_InstanceID + gl_GeometryIndexEXT);
    payload = vec4(p + q * gl_RayTmaxEXT, float(gl_IncomingRayFlagsEXT + ids) + bary.x);
}
"""


def test_dxr_hit_system_values_are_ray_stage_builtins(tmp_path):
    code = _translate(tmp_path, HLSL_CLOSEST_HIT, "values.rchit.hlsl", "directx")
    builtins = _builtins(code)
    assert builtins["gl_PrimitiveID"] == "PrimitiveId"
    # The instance index is InstanceId in a ray stage, not the vertex stage's InstanceIndex.
    assert builtins["gl_InstanceID"] == "InstanceId"
    assert builtins["gl_InstanceCustomIndexEXT"] == "InstanceCustomIndexKHR"
    assert builtins["gl_GeometryIndexEXT"] == "RayGeometryIndexKHR"
    assert builtins["gl_RayTmaxEXT"] == "RayTmaxKHR"
    assert builtins["gl_IncomingRayFlagsEXT"] == "IncomingRayFlagsKHR"
    assert builtins["gl_HitKindEXT"] == "HitKindKHR"
    # DXR returns the indices as uint; the built-ins are int, so each one is reinterpreted.
    assert len(re.findall(r"OpBitcast %uint", code)) >= 4
    _spirv_val(tmp_path, code)


def test_dxr_intersection_and_miss_read_the_incoming_ray(tmp_path):
    intersection = _translate(tmp_path, HLSL_INTERSECTION, "box.rint.hlsl", "directx")
    assert "OpEntryPoint IntersectionKHR" in intersection
    builtins = _builtins(intersection)
    assert builtins["gl_PrimitiveID"] == "PrimitiveId"
    assert builtins["gl_InstanceID"] == "InstanceId"
    assert builtins["gl_RayTmaxEXT"] == "RayTmaxKHR"
    _spirv_val(tmp_path, intersection)

    miss = _translate(tmp_path, HLSL_MISS, "sky.rmiss.hlsl", "directx")
    assert "OpEntryPoint MissKHR" in miss
    builtins = _builtins(miss)
    assert builtins["gl_RayTmaxEXT"] == "RayTmaxKHR"
    assert builtins["gl_IncomingRayFlagsEXT"] == "IncomingRayFlagsKHR"
    _spirv_val(tmp_path, miss)


def test_glsl_hit_transforms_reach_spirv_and_glsl(tmp_path):
    code = _translate(tmp_path, GLSL_CLOSEST_HIT, "hit.rchit", "opengl")
    builtins = _builtins(code)
    assert builtins["gl_ObjectToWorldEXT"] == "ObjectToWorldKHR"
    assert builtins["gl_WorldToObjectEXT"] == "WorldToObjectKHR"
    assert builtins["gl_PrimitiveID"] == "PrimitiveId"
    assert builtins["gl_InstanceID"] == "InstanceId"
    # Four columns of three floats, the shape Vulkan requires of both transforms.
    assert re.search(r"%mat4v3float = OpTypeMatrix %v3float 4", code), code
    _spirv_val(tmp_path, code)

    glsl = _translate(tmp_path, GLSL_CLOSEST_HIT, "hit.rchit", "opengl", "opengl")
    assert "gl_ObjectToWorldEXT * vec4(gl_ObjectRayOriginEXT, 1.0)" in glsl
    _glslang(tmp_path, glsl, "rchit")


def test_ray_builtin_types_follow_gl_ext_ray_tracing():
    assert crossgl_builtin_variable_type("gl_ObjectToWorldEXT") == "mat4x3"
    assert crossgl_builtin_variable_type("gl_WorldToObject3x4EXT") == "mat3x4"
    assert crossgl_builtin_variable_type("gl_IncomingRayFlagsEXT") == "uint"
    assert crossgl_builtin_variable_type("gl_RayTmaxEXT") == "float"


def test_ray_builtin_outside_a_ray_stage_is_refused(tmp_path):
    source = """#version 460
layout(location = 0) out vec4 color;
void main()
{
    color = vec4(gl_RayTmaxEXT);
}
"""
    with pytest.raises(Exception, match="gl_RayTmaxEXT"):
        _translate(tmp_path, source, "bad.frag", "opengl")


def test_fragment_primitive_id_keeps_its_rasterisation_meaning(tmp_path):
    source = """#version 460
layout(location = 0) out vec4 color;
void main()
{
    color = vec4(float(gl_PrimitiveID));
}
"""
    code = _translate(tmp_path, source, "prim.frag", "opengl")
    assert "OpEntryPoint Fragment" in code
    assert "OpCapability Geometry" in code
    assert _builtins(code)["gl_PrimitiveID"] == "PrimitiveId"
    _spirv_val(tmp_path, code)
