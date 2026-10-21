"""DXR ray tracing stages reach GLSL through the GL_EXT_ray_tracing interface.

HLSL passes the payload, hit attributes and callable data as entry parameters and hands TraceRay, CallShader and
ReportHit struct values; GLSL declares module-scope rayPayloadEXT/rayPayloadInEXT/hitAttributeEXT/callableDataEXT/
callableDataInEXT variables and names payloads by location (GLSL_EXT_ray_tracing). The rewrite keeps every value
and is checked with glslang under Vulkan 1.2 rules, the only API GL_EXT_ray_tracing exists in.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import DxrRayInterfaceUnsupportedError


def _translate(tmp_path, source, name, backend="opengl"):
    path = tmp_path / f"{name}.hlsl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, source_backend="directx")


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        [
            "glslangValidator",
            "--target-env",
            "vulkan1.2",
            "-o",
            str(tmp_path / f"{stage}.spv"),
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _squash(code):
    return re.sub(r"\s+", " ", code)


RAYGEN = """
struct Payload { float3 color; };
RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<float4> image : register(u1);

[shader("raygeneration")]
void main()
{
    uint2 pixel = DispatchRaysIndex().xy;
    RayDesc ray;
    ray.Origin = float3(pixel, 0.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.001;
    ray.TMax = 100.0;
    Payload payload;
    payload.color = float3(0.0, 0.0, 0.0);
    TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);
    image[pixel.y * 64 + pixel.x] = float4(payload.color, 1.0);
}
"""

CLOSEST_HIT = """
struct Payload { float3 color; };
struct CallData { float3 tint; };
struct Attributes { float2 bary; };

[shader("closesthit")]
void main(inout Payload payload, in Attributes attribs)
{
    attribs.bary *= 0.5;
    CallData data;
    data.tint = float3(attribs.bary, 1.0);
    CallShader(PrimitiveIndex(), data);
    payload.color = data.tint * RayTCurrent();
}
"""

CALLABLE = """
struct CallData { float3 tint; };

[shader("callable")]
void main(inout CallData data)
{
    float2 pos = float2(DispatchRaysIndex().x / 8, DispatchRaysIndex().y / 8);
    data.tint = float3(pos, 1.0);
}
"""

INTERSECTION = """
struct Attributes { float2 bary; };

[shader("intersection")]
void main()
{
    Attributes attribs;
    attribs.bary = float2(0.25, 0.5);
    if (ReportHit(RayTMin(), 0, attribs)) {
        return;
    }
}
"""


def test_raygen_traces_through_a_located_payload(tmp_path):
    code = _translate(tmp_path, RAYGEN, "trace.rgen")
    flat = _squash(code)
    assert (
        "struct RayDesc { vec3 Origin; float TMin; vec3 Direction; float TMax; };"
        in flat
    )
    assert "layout(location = 0) rayPayloadEXT Payload crossgl_ray_payload;" in flat
    assert "gl_LaunchIDEXT.xy" in code
    # The payload goes in and comes back around the trace, which names it by location.
    assert (
        "crossgl_ray_payload = payload; traceRayEXT(scene, gl_RayFlagsOpaqueEXT, 255, 0, 0, 0, ray.Origin, "
        "ray.TMin, ray.Direction, ray.TMax, 0); payload = crossgl_ray_payload;"
    ) in flat
    _glslang(tmp_path, code, "rgen")


def test_closest_hit_reads_its_payload_attributes_and_callable_data(tmp_path):
    code = _translate(tmp_path, CLOSEST_HIT, "hit.rchit")
    flat = _squash(code)
    assert "layout(location = 0) rayPayloadInEXT Payload payload;" in flat
    assert "hitAttributeEXT Attributes crossgl_hit_attributes;" in flat
    # GLSL hit attributes are read-only in hit shaders; the HLSL `in` parameter is a writable copy.
    assert "Attributes attribs = crossgl_hit_attributes;" in flat
    assert (
        "layout(location = 0) callableDataEXT CallData crossgl_callable_data;" in flat
    )
    assert "executeCallableEXT(uint(gl_PrimitiveID), 0);" in flat
    assert "gl_RayTmaxEXT" in code
    assert "void main()" in code
    _glslang(tmp_path, code, "rchit")


def test_callable_data_parameter_becomes_the_incoming_callable_data(tmp_path):
    code = _translate(tmp_path, CALLABLE, "call.rcall")
    flat = _squash(code)
    assert "layout(location = 0) callableDataInEXT CallData data;" in flat
    assert "DispatchRaysIndex" not in code
    _glslang(tmp_path, code, "rcall")


def test_intersection_reports_its_attributes_through_the_hit_attribute_variable(
    tmp_path,
):
    code = _translate(tmp_path, INTERSECTION, "box.rint")
    flat = _squash(code)
    assert "hitAttributeEXT Attributes crossgl_hit_attributes;" in flat
    assert (
        "crossgl_hit_attributes = attribs; if (reportIntersectionEXT(gl_RayTminEXT, 0))"
        in flat
    )
    _glslang(tmp_path, code, "rint")


def test_object_to_world_matrices_are_refused(tmp_path):
    source = CLOSEST_HIT.replace(
        "payload.color = data.tint * RayTCurrent();",
        "payload.color = mul(WorldToObject4x3(), float4(data.tint, 1.0));",
    )
    with pytest.raises(DxrRayInterfaceUnsupportedError, match="world-to-object"):
        _translate(tmp_path, source, "matrix.rchit")


def test_payload_without_a_known_struct_type_is_refused(tmp_path):
    source = RAYGEN.replace(
        "TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);",
        "TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payloads[0]);",
    ).replace("Payload payload;", "Payload payload; Payload payloads[1];")
    with pytest.raises(DxrRayInterfaceUnsupportedError, match="payload argument"):
        _translate(tmp_path, source, "array.rgen")


BUILTIN_ATTRIBUTES_HIT = """
struct Payload { float3 color; };

[shader("closesthit")]
void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    float2 uv = attribs.barycentrics;
    payload.color = float3(1.0 - uv.x - uv.y, uv.x, uv.y) * RayTCurrent();
}
"""

BUILTIN_ATTRIBUTES_ANY_HIT = """
struct Payload { float3 color; };

[shader("anyhit")]
void main(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    if (attribs.barycentrics.x > 0.5) {
        IgnoreHit();
    }
    payload.color = float3(attribs.barycentrics, 0.0);
}
"""


def _spirv_val(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    source = tmp_path / "out.spvasm"
    source.write_text(assembly)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.2", str(source), "-o", "out.spv"],
        ["spirv-val", "--target-env", "vulkan1.2", "out.spv"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "source, name, stage",
    [
        (BUILTIN_ATTRIBUTES_HIT, "builtin.rchit", "rchit"),
        (BUILTIN_ATTRIBUTES_ANY_HIT, "builtin.rahit", "rahit"),
    ],
)
def test_builtin_triangle_attributes_are_two_barycentrics_in_glsl(
    tmp_path, source, name, stage
):
    code = _translate(tmp_path, source, name)
    flat = _squash(code)
    # DXR declares the struct itself; GLSL reads the same two floats from a vec2 hit attribute.
    assert (
        "struct BuiltInTriangleIntersectionAttributes { vec2 barycentrics; };" in flat
    )
    assert (
        "hitAttributeEXT BuiltInTriangleIntersectionAttributes crossgl_hit_attributes;"
        in flat
    )
    assert (
        "BuiltInTriangleIntersectionAttributes attribs = crossgl_hit_attributes;"
        in flat
    )
    _glslang(tmp_path, code, stage)


@pytest.mark.parametrize(
    "source, name, model",
    [
        (BUILTIN_ATTRIBUTES_HIT, "builtin.rchit", "ClosestHitKHR"),
        (BUILTIN_ATTRIBUTES_ANY_HIT, "builtin.rahit", "AnyHitKHR"),
    ],
)
def test_builtin_triangle_attributes_reach_spirv_as_a_hit_attribute(
    tmp_path, source, name, model
):
    code = _translate(tmp_path, source, name, backend="vulkan")
    assert f"OpEntryPoint {model} " in code
    assert re.search(
        r"%crossgl_hit_attributes = OpVariable %\S+ HitAttributeKHR", code
    ), code
    assert (
        'OpMemberName %BuiltInTriangleIntersectionAttributes 0 "barycentrics"' in code
    )
    if model == "ClosestHitKHR":
        assert "BuiltIn RayTmaxKHR" in code
    _spirv_val(tmp_path, code)
