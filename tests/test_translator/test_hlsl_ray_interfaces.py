"""GLSL-style ray tracing stages reach HLSL with DXR entry parameters, struct payloads and intrinsics.

See :mod:`crosstl.translator.semantics.hlsl_ray_interfaces`. DXC validates the output when it is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.hlsl_ray_interfaces import (
    RayInterfaceUnsupportedError,
)

RAYGEN = """#version 460
#extension GL_EXT_ray_tracing : require
layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0, rgba8) uniform image2D image;
layout(location = 0) rayPayloadEXT vec3 hitValue;
void main()
{
    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy);
    hitValue = vec3(0.0);
    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(uv, 0.0), 0.001, vec3(0.0, 0.0, 1.0), 100.0, 0);
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 1.0));
}
"""

CLOSEST_HIT = """#version 460
#extension GL_EXT_ray_tracing : require
struct Shadow { bool hit; float distance; };
layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 2) rayPayloadEXT Shadow shadow;
hitAttributeEXT vec2 attribs;
layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
void darken() { hitValue *= 0.5; }
void main()
{
    hitValue = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    if (gl_PrimitiveID == 3) {
        return;
    }
    vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    shadow.hit = true;
    traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xff, 0, 0, 1,
                origin, 0.001, vec3(0.0, 1.0, 0.0), 100.0, 2);
    if (shadow.hit) {
        darken();
    }
}
"""

MISS = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec3 hitValue;
void main()
{
    hitValue = vec3(0.0, 0.0, 0.2);
}
"""

CALLABLE = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) callableDataInEXT vec3 outColor;
void main()
{
    outColor = vec3(1.0, 0.0, 0.0);
}
"""

CALLER = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 0) callableDataEXT vec3 outColor;
hitAttributeEXT vec2 attribs;
void main()
{
    executeCallableEXT(gl_InstanceCustomIndexEXT, 0);
    hitValue = outColor;
}
"""

INTERSECTION = """#version 460
#extension GL_EXT_ray_tracing : require
void main()
{
    reportIntersectionEXT(gl_RayTminEXT + 1.0, 0u);
}
"""

TRANSFORM = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec2 attribs;
void main()
{
    hitValue = gl_ObjectToWorldEXT * vec4(attribs, 0.0, 1.0);
}
"""


def _hlsl(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _validate(tmp_path, hlsl):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    shader = tmp_path / "out.hlsl"
    shader.write_text(hlsl)
    result = subprocess.run(
        [
            dxc,
            "-T",
            "lib_6_6",
            "-HV",
            "2021",
            str(shader),
            "-Fo",
            str(tmp_path / "out.dxil"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_raygen_traces_with_a_ray_desc_and_the_payload_variable(tmp_path):
    hlsl = _hlsl(tmp_path, "shader.rgen", RAYGEN)
    assert (
        "struct [raypayload] CrossGLRayPayload_hitValue {\n"
        "    float3 value : read(caller, closesthit, miss, anyhit) "
        ": write(caller, closesthit, miss, anyhit);\n};" in hlsl
    )
    assert "static CrossGLRayPayload_hitValue hitValue;" in hlsl
    assert (
        "RayDesc crossgl_ray_desc(float3 origin, float tmin, float3 direction, float tmax)"
        in hlsl
    )
    assert re.search(
        r"TraceRay\(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 255, 0, 0, 0, crossgl_ray_desc\(.*\), hitValue\);",
        hlsl,
    )
    assert "hitValue.value = float3(0.0, 0.0, 0.0);" in hlsl
    assert "DispatchRaysIndex().xy" in hlsl and "DispatchRaysDimensions().xy" in hlsl
    assert not re.search(r"\bgl_", hlsl)
    _validate(tmp_path, hlsl)


def test_closest_hit_receives_payload_and_attributes_and_copies_the_payload_back(
    tmp_path,
):
    hlsl = _hlsl(tmp_path, "shader.rchit", CLOSEST_HIT)
    signature = re.search(r'\[shader\("closesthit"\)\]\s*void (\w+)\((.*)\) \{', hlsl)
    assert signature.group(2) == (
        "inout CrossGLRayPayload_hitValue crossgl_payload, "
        "in CrossGLHitAttributes_attribs crossgl_hit_attribute"
    )
    body = hlsl[signature.end() :]
    assert body.index("hitValue = crossgl_payload;") < body.index(
        "attribs = crossgl_hit_attribute;"
    )
    # The early return and the end of the entry both copy the payload back.
    assert body.count("crossgl_payload = hitValue;") == 2
    assert re.search(r"\{\s*crossgl_payload = hitValue;\s*return;\s*\}", body)
    # A struct payload keeps its type; the helper reads the module copy.
    assert "static Shadow shadow;" in hlsl
    assert (
        "TraceRay(topLevelAS, (RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER)"
        in hlsl
    )
    assert hlsl.rstrip().endswith("}") and ", shadow);" in hlsl
    assert "hitValue.value *= 0.5;" in hlsl
    assert "int(PrimitiveIndex()) == 3" in hlsl
    assert "WorldRayOrigin() + (WorldRayDirection() * RayTCurrent())" in hlsl
    _validate(tmp_path, hlsl)


def test_miss_and_callable_stages_receive_their_inout_parameter(tmp_path):
    miss = _hlsl(tmp_path, "shader.rmiss", MISS)
    assert "(inout CrossGLRayPayload_hitValue crossgl_payload)" in miss
    _validate(tmp_path, miss)
    callable_stage = _hlsl(tmp_path, "shader.rcall", CALLABLE)
    assert (
        "(inout CrossGLCallableData_outColor crossgl_callable_data)" in callable_stage
    )
    assert "crossgl_callable_data = outColor;" in callable_stage
    _validate(tmp_path, callable_stage)


def test_call_shader_passes_the_callable_data_of_its_location(tmp_path):
    hlsl = _hlsl(tmp_path, "shader.rchit", CALLER)
    assert "CallShader(int(InstanceID()), outColor);" in hlsl
    assert "hitValue.value = outColor.value;" in hlsl
    _validate(tmp_path, hlsl)


def test_intersection_without_attributes_reports_a_placeholder(tmp_path):
    hlsl = _hlsl(tmp_path, "shader.rint", INTERSECTION)
    assert "static CrossGLHitAttributesUnused crossgl_hit_attributes;" in hlsl
    assert "ReportHit((RayTMin() + 1.0), 0u, crossgl_hit_attributes);" in hlsl
    _validate(tmp_path, hlsl)


def test_glsl_targets_keep_the_glsl_spelling(tmp_path):
    path = tmp_path / "shader.rchit"
    path.write_text(CLOSEST_HIT)
    glsl = crosstl.translate(str(path), backend="glsl", format_output=False)
    assert "rayPayloadInEXT vec3 hitValue;" in glsl
    assert "crossgl_payload" not in glsl and "CrossGLRayPayload" not in glsl


def test_transform_matrices_are_refused(tmp_path):
    with pytest.raises(RayInterfaceUnsupportedError) as caught:
        _hlsl(tmp_path, "shader.rchit", TRANSFORM)
    assert caught.value.missing_capabilities == ("directx.ray-builtin",)
    assert "gl_ObjectToWorldEXT" in str(caught.value)
