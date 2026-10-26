"""Ray payload structures carry the access qualifiers HLSL shader model 6.7 and later require.

DXC rejects a payload type that is not declared ``struct [raypayload]`` with access qualifiers on every field
(DirectX Raytracing specification, "Payload access qualifiers"). The DirectX generator printed payload structures
as plain structures, so every ray-tracing program failed to compile for ``lib_6_7`` and later.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.diagnostics import CODE_PREFIX
from crosstl.translator.semantics import HlslRayPayloadUnsupportedError

ALL_STAGES = "caller, closesthit, miss, anyhit"
EVERY_ACCESS = f": read({ALL_STAGES}) : write({ALL_STAGES});"

# The coverage probe shape for a closest-hit stage (tools/coverage/crossgl.py).
CLOSEST_HIT_CGL = """
shader Probe {
    struct Payload {
        vec4 color;
        uint depth;
    }
    ray_closest_hit {
        void main(Payload payload @ payload, BuiltInTriangleIntersectionAttributes attributes @ hit_attribute) {
            payload.color = vec4(attributes.barycentrics, 0.0, 1.0);
        }
    }
}
"""

QUALIFIED_HLSL = """
struct [raypayload] Payload {
    float4 color : write(caller, closesthit, miss) : read(caller);
    uint depth : write(caller) : read(closesthit);
};
RaytracingAccelerationStructure scene : register(t0);
RWTexture2D<float4> output : register(u0);
[shader("raygeneration")]
void rg() {
    RayDesc ray;
    ray.Origin = float3(0, 0, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.0;
    ray.TMax = 100.0;
    Payload p;
    p.color = float4(0, 0, 0, 0);
    p.depth = 0;
    TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, p);
    output[DispatchRaysIndex().xy] = p.color;
}
[shader("miss")]
void ms(inout Payload p) { p.color = float4(1, 0, 0, 1); }
[shader("closesthit")]
void ch(inout Payload p, in BuiltInTriangleIntersectionAttributes attr) {
    p.color = float4(attr.barycentrics, float(p.depth), 1);
}
"""

RAYGEN_GLSL = """#version 460
#extension GL_EXT_ray_tracing : require
struct Payload { vec4 color; uint depth; };
layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1, rgba8) uniform image2D outputImage;
layout(location = 0) rayPayloadEXT Payload payload;
void main() {
    payload.color = vec4(0.0);
    payload.depth = 0u;
    traceRayEXT(scene, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(0.0), 0.001, vec3(0.0, 0.0, 1.0), 100.0, 0);
    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), payload.color);
}
"""

MISS_GLSL = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec4 payload;
void main() { payload = vec4(0.0, 0.0, 1.0, 1.0); }
"""


def _translate(tmp_path, source, name, backend="directx"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _dxc(tmp_path, code, profile="lib_6_8"):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / f"out_{profile}.hlsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", profile, "-HV", "2021", str(path), "-Fo", str(tmp_path / "o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _payload_fields(code, struct_name):
    match = re.search(
        rf"struct\s*\[raypayload\]\s*{struct_name}\s*\{{(.*?)\}};", code, re.S
    )
    assert match, code
    return {
        name: re.sub(r"\s+", " ", qualifiers).strip()
        for name, qualifiers in re.findall(r"\b(\w+)\s*(:[^;]*;)", match.group(1))
    }


def test_an_unqualified_payload_is_read_and_written_by_every_stage(tmp_path):
    code = _translate(tmp_path, CLOSEST_HIT_CGL, "probe.cgl")

    assert _payload_fields(code, "Payload") == {
        "color": EVERY_ACCESS,
        "depth": EVERY_ACCESS,
    }
    _dxc(tmp_path, code)


def test_hlsl_payload_qualifiers_survive_the_round_trip(tmp_path):
    code = _translate(tmp_path, QUALIFIED_HLSL, "shader.hlsl")

    assert _payload_fields(code, "Payload") == {
        "color": ": write(caller, closesthit, miss) : read(caller);",
        "depth": ": write(caller) : read(closesthit);",
    }
    _dxc(tmp_path, code)


def test_crossgl_spells_the_qualifiers_as_ray_payload_attributes(tmp_path):
    cgl = _translate(tmp_path, QUALIFIED_HLSL, "shader.hlsl", "cgl")

    assert (
        "vec4 color @ ray_payload_write(caller, closesthit, miss) "
        "@ ray_payload_read(caller);" in cgl
    )
    assert "uint depth @ ray_payload_write(caller) @ ray_payload_read(closesthit);" in (
        cgl
    )
    assert "hlsl_read" not in cgl and "hlsl_write" not in cgl


def test_a_field_without_qualifiers_gets_every_stage(tmp_path):
    source = CLOSEST_HIT_CGL.replace(
        "vec4 color;",
        "vec4 color @ ray_payload_write(closesthit) @ ray_payload_read(caller);",
    )
    code = _translate(tmp_path, source, "probe.cgl")

    assert _payload_fields(code, "Payload") == {
        "color": ": write(closesthit) : read(caller);",
        "depth": EVERY_ACCESS,
    }
    _dxc(tmp_path, code)


HIT_OBJECT_HLSL = """
struct [raypayload] Payload {
    float4 color : read(caller) : write(anyhit, closesthit);
};
RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<uint> outp : register(u0);
[shader("raygeneration")]
void main() {
    RayDesc ray;
    ray.Origin = float3(0, 0, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.0;
    ray.TMax = 100.0;
    Payload payload;
    dx::HitObject hit = dx::HitObject::TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
    dx::HitObject::Invoke(hit, payload);
    outp[0] = 0u;
}
"""


@pytest.mark.parametrize(
    "call",
    [
        "dx::HitObject hit = dx::HitObject::TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);",
        "dx::HitObject::Invoke(hit, payload);",
    ],
)
def test_a_hit_object_payload_stays_a_payload(tmp_path, call):
    # dx::HitObject::TraceRay and dx::HitObject::Invoke take the payload as their
    # last argument, like TraceRay, so DXC needs it declared [raypayload].
    source = HIT_OBJECT_HLSL
    if call.startswith("dx::HitObject::Invoke"):
        source = source.replace(
            "dx::HitObject hit = dx::HitObject::TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);",
            "dx::HitObject hit;",
        )
    else:
        source = source.replace("    dx::HitObject::Invoke(hit, payload);\n", "")
    code = _translate(tmp_path, source, "shader.hlsl")

    assert _payload_fields(code, "Payload") == {
        "color": ": read(caller) : write(anyhit, closesthit);",
    }


def test_a_trace_ray_payload_from_glsl_is_declared_a_payload(tmp_path):
    code = _translate(tmp_path, RAYGEN_GLSL, "shader.rgen")

    assert _payload_fields(code, "Payload") == {
        "color": EVERY_ACCESS,
        "depth": EVERY_ACCESS,
    }
    _dxc(tmp_path, code)


def test_a_glsl_vector_payload_is_wrapped_in_a_payload_structure(tmp_path):
    code = _translate(tmp_path, MISS_GLSL, "shader.rmiss")

    fields = _payload_fields(code, r"\w+")
    assert list(fields.values()) == [EVERY_ACCESS]
    _dxc(tmp_path, code)


def test_a_qualifier_naming_an_unknown_stage_is_refused(tmp_path):
    source = CLOSEST_HIT_CGL.replace(
        "vec4 color;", "vec4 color @ ray_payload_write(intersection);"
    )
    with pytest.raises(HlslRayPayloadUnsupportedError) as raised:
        _translate(tmp_path, source, "probe.cgl")

    assert raised.value.code == f"{CODE_PREFIX}directx-ray-payload-unsupported"
    assert "'intersection'" in str(raised.value)


@pytest.mark.parametrize("backend", ["metal", "opengl", "slang"])
def test_targets_without_payload_qualifiers_drop_them(tmp_path, backend):
    source = CLOSEST_HIT_CGL.replace(
        "vec4 color;",
        "vec4 color @ ray_payload_write(closesthit) @ ray_payload_read(caller);",
    )
    code = _translate(tmp_path, source, "probe.cgl", backend)

    assert "ray_payload" not in code
    assert "raypayload" not in code
    assert re.search(r"\bcolor\s*;", code), code
