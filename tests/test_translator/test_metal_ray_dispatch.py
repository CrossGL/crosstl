"""Ray generation launch values and ray flags on Metal; traces whose payload Metal cannot fill are refused.

A ray generation stage becomes a Metal ``kernel``. Its launch index and size (``gl_LaunchIDEXT``,
``DispatchRaysIndex()``, ``gl_LaunchSizeEXT``, ``DispatchRaysDimensions()``) and the GLSL ray flag constants were
printed verbatim, which MSL does not declare; a ``traceRayEXT`` naming a ``rayPayloadEXT`` variable lowered to an
``intersect`` call while the payload became a zero constant. See :mod:`crosstl.translator.semantics.metal_ray_dispatch`.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.semantics import MetalRayDispatchUnsupportedError
from tests.conformance.corpus import checks

GLSL_RAYGEN = """#version 460
#extension GL_EXT_ray_tracing : require
layout(binding = 1, set = 0, rgba8) uniform image2D image;
void main()
{
    uint flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipAABBEXT;
    vec2 uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(gl_LaunchSizeEXT.xy);
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(uv, float(flags), 1.0));
}
"""

HLSL_RAYGEN = """RWTexture2D<float4> image : register(u0);
[shader("raygeneration")]
void main()
{
    uint2 id = DispatchRaysIndex().xy;
    uint2 size = DispatchRaysDimensions().xy;
    image[id] = float4(float2(id) / float2(size), 0.0, 1.0);
}
"""

CGL_RAYGEN = """
shader LaunchShader {
    image2D image @binding(1);
    ray_generation {
        void main() {
            uvec3 index = gl_LaunchIDEXT;
            vec2 uv = vec2(index.xy) / vec2(gl_LaunchSizeEXT.xy);
            imageStore(image, ivec2(index.xy), vec4(uv, 0.0, 1.0));
        }
    }
}
"""

GLSL_TRACED_PAYLOAD = """#version 460
#extension GL_EXT_ray_tracing : require
layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0, rgba8) uniform image2D image;
layout(location = 0) rayPayloadEXT vec3 hitValue;
void main()
{
    vec2 uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(gl_LaunchSizeEXT.xy);
    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(uv, 0.0), 0.001, vec3(0.0, 0.0, 1.0), 100.0, 0);
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 1.0));
}
"""

GLSL_CLOSEST_HIT_LAUNCH = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec3 hitValue;
void main() { hitValue = vec3(gl_LaunchIDEXT.xy, 0.0); }
"""

GLSL_HELPER_LAUNCH = """#version 460
#extension GL_EXT_ray_tracing : require
layout(binding = 1, set = 0, rgba8) uniform image2D image;
ivec2 pixel() { return ivec2(gl_LaunchIDEXT.xy); }
void main() { imageStore(image, pixel(), vec4(1.0)); }
"""

SOURCES = {
    "glsl": ("shader.rgen", GLSL_RAYGEN),
    "hlsl": ("shader.hlsl", HLSL_RAYGEN),
    "cgl": ("shader.cgl", CGL_RAYGEN),
}
LAUNCH_SPELLINGS = (
    "gl_LaunchIDEXT",
    "gl_LaunchSizeEXT",
    "gl_GlobalInvocationID",
    "DispatchRaysIndex",
    "DispatchRaysDimensions",
)


def translate(tmp_path, name, text, backend="metal"):
    path = tmp_path / name
    path.write_text(text)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("language", sorted(SOURCES))
def test_launch_values_become_kernel_grid_attributes(tmp_path, language):
    code = translate(tmp_path, *SOURCES[language])
    assert "kernel void" in code
    assert "uint3 thread_position_in_grid [[thread_position_in_grid]]" in code
    assert "uint3 threads_per_grid [[threads_per_grid]]" in code
    for spelling in LAUNCH_SPELLINGS:
        assert spelling not in code


@pytest.mark.parametrize("language", sorted(SOURCES))
def test_ray_generation_kernel_compiles(tmp_path, language):
    artifact = tmp_path / "out.metal"
    artifact.write_text(translate(tmp_path, *SOURCES[language]))
    verdict = checks.check(artifact, "metal", ["compute"])
    if verdict["verdict"] == "unavailable":
        pytest.skip("no metal checker installed")
    assert verdict["verdict"] == "accepted", verdict.get("log", "")


def test_glsl_ray_flags_become_their_values(tmp_path):
    code = translate(tmp_path, *SOURCES["glsl"])
    # gl_RayFlagsOpaqueEXT, gl_RayFlagsTerminateOnFirstHitEXT and gl_RayFlagsSkipAABBEXT (GL_EXT_ray_tracing 8.19).
    assert "1u | 4u | 512u" in code
    assert "gl_RayFlags" not in code


def test_traced_payload_is_refused(tmp_path):
    with pytest.raises(MetalRayDispatchUnsupportedError) as caught:
        translate(tmp_path, "shader.rgen", GLSL_TRACED_PAYLOAD)
    assert caught.value.code == "project.translate.metal-ray-dispatch-unsupported"
    assert "'hitValue'" in caught.value.reason
    assert "shader binding table" in caught.value.reason


def test_launch_value_outside_ray_generation_is_refused(tmp_path):
    with pytest.raises(MetalRayDispatchUnsupportedError) as caught:
        translate(tmp_path, "shader.rchit", GLSL_CLOSEST_HIT_LAUNCH)
    assert "ray_closest_hit" in str(caught.value)
    assert "[[visible]]" in caught.value.reason


def test_launch_value_in_helper_is_refused(tmp_path):
    with pytest.raises(MetalRayDispatchUnsupportedError) as caught:
        translate(tmp_path, "shader.rgen", GLSL_HELPER_LAUNCH)
    assert "helper 'pixel'" in caught.value.reason


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("opengl", "gl_LaunchIDEXT"), ("directx", "DispatchRaysIndex()")],
)
def test_other_targets_keep_their_launch_spelling(tmp_path, backend, expected):
    code = translate(tmp_path, "shader.rgen", GLSL_RAYGEN, backend)
    assert expected in code
    assert "threads_per_grid" not in code
