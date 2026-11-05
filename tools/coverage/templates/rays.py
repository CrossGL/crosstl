"""Ray-tracing interface probes: the data a ray-tracing stage hands another stage, and the record it reads.

Each probe is a stage program of :mod:`.graphics` with one interface added. A ray-generation stage traces a ray with
a payload and reads back what the hit or miss stage wrote, or calls a callable shader with data and reads it back. An
intersection stage reports a hit with an attribute structure of its own, and a closest-hit stage reads that structure
in place of the triangle's barycentrics. A miss stage reads a value from its shader binding table record.

GLSL declares the outgoing payload ``rayPayloadEXT``, the callable data ``callableDataEXT``, the attributes
``hitAttributeEXT`` and the record as a ``shaderRecordEXT`` buffer block (GL_EXT_ray_tracing). HLSL and Slang pass a
local payload to ``TraceRay``, local data to ``CallShader`` and local attributes to ``ReportHit``, take the attributes
as a hit stage's second parameter, and read the record through a ``ConstantBuffer`` that DXC marks
``[[vk::shader_record_ext]]`` and slangc ``[[vk::shader_record]]``. rust-gpu takes each as an attributed parameter.
CrossGL writes them as its HLSL importer does: locals passed to the calls, attributes as a ``hit_attribute``
parameter, and the record as a buffer block whose layout names ``shaderRecordEXT``
(:mod:`crosstl.translator.semantics.shader_records`).

Metal traces rays from kernels through an intersector whose result the kernel reads, so it has no ray-generation,
hit, miss or callable stage to hand values between and no shader binding table (Metal Shading Language
Specification, "Ray-Tracing Functions"). WGSL, GLSL ES 3.00 and the kernel languages have no ray-tracing stages.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "ray-payload",
    "ray-callable-data",
    "ray-reported-attributes",
    "ray-custom-attributes",
    "ray-shader-record",
)

_GLSL_HEADER = "#version 460\n#extension GL_EXT_ray_tracing : require\n"
_GLSL_PAYLOAD = "struct Payload {\n    vec4 color;\n};\n"
_GLSL_INCOMING = (
    _GLSL_PAYLOAD + "layout(location = 0) rayPayloadInEXT Payload payload;\n"
)
_GLSL_OUTPUT = "layout(std430, binding = 0) buffer Output {\n    float outp[];\n};\n"
_GLSL_ATTRIBUTES = "struct Attributes {\n    vec3 normal;\n};\nhitAttributeEXT Attributes attributes;\n"

GLSL = {
    "ray-payload": Template(
        program=_GLSL_HEADER
        + _GLSL_PAYLOAD
        + "layout(location = 0) rayPayloadEXT Payload payload;\n"
        "layout(binding = 1) uniform accelerationStructureEXT scene;\n"
        + _GLSL_OUTPUT
        + "void main() {\n    payload.color = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "    traceRayEXT(scene, gl_RayFlagsNoneEXT, 0xFF, 0u, 1u, 0u, vec3(0.0, 0.0, 0.0), 0.0, "
        "vec3(0.0, 0.0, 1.0), 100.0, 0);\n"
        "    outp[0] = payload.color.x;\n}\n"
    ),
    "ray-callable-data": Template(
        program=_GLSL_HEADER
        + "struct CallData {\n    vec4 color;\n};\nlayout(location = 0) callableDataEXT CallData data;\n"
        + _GLSL_OUTPUT
        + "void main() {\n    data.color = vec4(0.0, 0.0, 0.0, 0.0);\n    executeCallableEXT(0u, 0);\n"
        "    outp[0] = data.color.x;\n}\n"
    ),
    "ray-reported-attributes": Template(
        program=_GLSL_HEADER
        + _GLSL_ATTRIBUTES
        + "void main() {\n    attributes.normal = vec3(0.0, 0.0, 1.0);\n    reportIntersectionEXT(1.0, 0u);\n}\n"
    ),
    "ray-custom-attributes": Template(
        program=_GLSL_HEADER
        + _GLSL_INCOMING
        + _GLSL_ATTRIBUTES
        + "void main() {\n    payload.color = vec4(attributes.normal, 1.0);\n}\n"
    ),
    "ray-shader-record": Template(
        program=_GLSL_HEADER
        + _GLSL_INCOMING
        + "layout(shaderRecordEXT, std430) buffer Record {\n    vec4 color;\n} record;\n"
        "void main() {\n    payload.color = record.color;\n}\n"
    ),
}
WEBGL = dict.fromkeys(NAMES, ABSENT)
WGSL = dict.fromkeys(NAMES, ABSENT)
METAL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)

# Shader model 6.7 and later require a payload to say which stages read and write each field; a ray-generation stage
# that fills its payload before it traces writes it as the caller.
_HLSL_PAYLOAD = (
    "struct [raypayload] Payload {\n"
    "    float4 color : read(caller) : write({writers}anyhit, closesthit, miss);\n};\n\n"
)
_HLSL_ATTRIBUTES = "struct Attributes {\n    float3 normal;\n};\n\n"
_HLSL_OUTPUT = "RWStructuredBuffer<float> outp : register(u0);\n\n"


def _hlsl(record: str) -> dict[str, Template]:
    """The HLSL programs; ``record`` is the attribute that marks a constant buffer as the shader record."""
    payload = _HLSL_PAYLOAD.replace("{writers}", "")
    return {
        "ray-payload": Template(
            program=_HLSL_PAYLOAD.replace("{writers}", "caller, ")
            + "RaytracingAccelerationStructure scene : register(t0);\n"
            + _HLSL_OUTPUT
            + '[shader("raygeneration")]\nvoid main() {\n    RayDesc ray;\n'
            "    ray.Origin = float3(0.0, 0.0, 0.0);\n    ray.Direction = float3(0.0, 0.0, 1.0);\n"
            "    ray.TMin = 0.0;\n    ray.TMax = 100.0;\n    Payload payload;\n"
            "    payload.color = float4(0.0, 0.0, 0.0, 0.0);\n"
            "    TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);\n"
            "    outp[0] = payload.color.x;\n}\n"
        ),
        "ray-callable-data": Template(
            program="struct CallData {\n    float4 color;\n};\n\n"
            + _HLSL_OUTPUT
            + '[shader("raygeneration")]\nvoid main() {\n    CallData data;\n'
            "    data.color = float4(0.0, 0.0, 0.0, 0.0);\n    CallShader(0, data);\n"
            "    outp[0] = data.color.x;\n}\n"
        ),
        "ray-reported-attributes": Template(
            program=_HLSL_ATTRIBUTES
            + '[shader("intersection")]\nvoid main() {\n    Attributes attributes;\n'
            "    attributes.normal = float3(0.0, 0.0, 1.0);\n    ReportHit(1.0, 0, attributes);\n}\n"
        ),
        "ray-custom-attributes": Template(
            program=payload
            + _HLSL_ATTRIBUTES
            + '[shader("closesthit")]\nvoid main(inout Payload payload, in Attributes attributes) {\n'
            "    payload.color = float4(attributes.normal, 1.0);\n}\n"
        ),
        "ray-shader-record": Template(
            program=payload + "struct Record {\n    float4 color;\n};\n\n"
            f"[[{record}]] ConstantBuffer<Record> record;\n\n"
            '[shader("miss")]\nvoid main(inout Payload payload) {\n    payload.color = record.color;\n}\n'
        ),
    }


HLSL = _hlsl("vk::shader_record_ext")
SLANG = _hlsl("vk::shader_record")

_RUST_HEADER = "#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
_RUST_PAYLOAD = "pub struct Payload {\n    pub color: Vec4,\n}\n\n"
_RUST_ATTRIBUTES = "pub struct Attributes {\n    pub normal: Vec3,\n}\n\n"
_RUST_OUTPUT = (
    "#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] outp: &mut [f32]"
)


def _rust(declarations: str, stage: str, params: str, body: str) -> Template:
    return Template(
        program=f"{_RUST_HEADER}{declarations}#[spirv({stage})]\npub fn main({params}) {{\n{body}}}\n",
        entry="main",
    )


RUST = {
    "ray-payload": _rust(
        _RUST_PAYLOAD,
        "ray_generation",
        "#[spirv(descriptor_set = 0, binding = 1)] scene: &spirv_std::ray_tracing::AccelerationStructure, "
        f"#[spirv(ray_payload)] payload: &mut Payload, {_RUST_OUTPUT}",
        "    payload.color = Vec4::ZERO;\n    unsafe {\n        scene.trace_ray(\n"
        "            spirv_std::ray_tracing::RayFlags::NONE, 0xFF, 0, 1, 0, Vec3::ZERO, 0.0,\n"
        "            Vec3::new(0.0, 0.0, 1.0), 100.0, payload,\n        );\n    }\n"
        "    outp[0] = payload.color.x;\n",
    ),
    "ray-callable-data": _rust(
        "pub struct CallData {\n    pub color: Vec4,\n}\n\n",
        "ray_generation",
        f"#[spirv(callable_data)] data: &mut CallData, {_RUST_OUTPUT}",
        "    data.color = Vec4::ZERO;\n    unsafe {\n"
        "        spirv_std::arch::execute_callable::<_, 0>(data);\n    }\n"
        "    outp[0] = data.color.x;\n",
    ),
    "ray-reported-attributes": _rust(
        _RUST_ATTRIBUTES,
        "intersection",
        "#[spirv(hit_attribute)] attributes: &mut Attributes",
        "    attributes.normal = Vec3::new(0.0, 0.0, 1.0);\n    unsafe {\n"
        "        spirv_std::arch::report_intersection(1.0, 0);\n    }\n",
    ),
    "ray-custom-attributes": _rust(
        _RUST_PAYLOAD + _RUST_ATTRIBUTES,
        "closest_hit",
        "#[spirv(incoming_ray_payload)] payload: &mut Payload, "
        "#[spirv(hit_attribute)] attributes: &mut Attributes",
        "    payload.color = attributes.normal.extend(1.0);\n",
    ),
    "ray-shader-record": _rust(
        _RUST_PAYLOAD + "pub struct Record {\n    pub color: Vec4,\n}\n\n",
        "miss",
        "#[spirv(incoming_ray_payload)] payload: &mut Payload, "
        "#[spirv(shader_record_buffer)] record: &Record",
        "    payload.color = record.color;\n",
    ),
}


def _crossgl(
    structs: str, block: str, body: str, params: str = "", scope: str = ""
) -> Template:
    """A CrossGL stage program: ``structs`` and the shader-scope ``scope`` ahead of the stage ``block``."""
    return Template(
        program=f"shader Probe {{\n{structs}{scope}    {block} {{\n{body.replace('{params}', params)}"
        "    }\n}\n"
    )


def _struct(name: str, member: str) -> str:
    return f"    struct {name} {{\n        {member};\n    }}\n"


_PAYLOAD = _struct("Payload", "vec4 color")
_ATTRIBUTES = _struct("Attributes", "vec3 normal")

CROSSGL = {
    "ray-payload": _crossgl(
        _PAYLOAD,
        "ray_generation",
        "        buffer float* outp;\n        uniform accelerationStructureEXT scene;\n"
        "        void main() {\n            RayDesc ray;\n"
        "            ray.Origin = vec3(0.0, 0.0, 0.0);\n            ray.Direction = vec3(0.0, 0.0, 1.0);\n"
        "            ray.TMin = 0.0;\n            ray.TMax = 100.0;\n            Payload payload;\n"
        "            payload.color = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "            TraceRay(scene, RAY_FLAG_NONE, 255u, 0u, 1u, 0u, ray, payload);\n"
        "            outp[0] = payload.color.x;\n        }\n",
    ),
    "ray-callable-data": _crossgl(
        _struct("CallData", "vec4 color"),
        "ray_generation",
        "        buffer float* outp;\n        void main() {\n            CallData data;\n"
        "            data.color = vec4(0.0, 0.0, 0.0, 0.0);\n            CallShader(0u, data);\n"
        "            outp[0] = data.color.x;\n        }\n",
    ),
    "ray-reported-attributes": _crossgl(
        _ATTRIBUTES,
        "ray_intersection",
        "        void main() {\n            Attributes attributes;\n"
        "            attributes.normal = vec3(0.0, 0.0, 1.0);\n            ReportHit(1.0, 0u, attributes);\n"
        "        }\n",
    ),
    "ray-custom-attributes": _crossgl(
        _PAYLOAD + _ATTRIBUTES,
        "ray_closest_hit",
        "        void main({params}) {\n            payload.color = vec4(attributes.normal, 1.0);\n        }\n",
        "Payload payload @ payload, Attributes attributes @ hit_attribute",
    ),
    "ray-shader-record": _crossgl(
        _PAYLOAD + _struct("Record", "vec4 color"),
        "ray_miss",
        "        void main({params}) {\n            payload.color = record.color;\n        }\n",
        "Payload payload @ payload",
        "    Record record @glsl_buffer_block(shaderRecordEXT, std430);\n",
    ),
}
