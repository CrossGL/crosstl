"""DXR ray tracing stages reach SPIR-V through the SPV_KHR_ray_tracing storage classes.

HLSL passes a ray payload, hit attributes and callable data as entry parameters. SPIR-V declares them as
module-scope variables in the ``RayPayloadKHR``, ``IncomingRayPayloadKHR``, ``HitAttributeKHR``,
``CallableDataKHR`` and ``IncomingCallableDataKHR`` storage classes, and ``OpTraceRayKHR`` and
``OpExecuteCallableKHR`` take the payload variable itself. A payload declared as a stage ``Input`` validates
but never returns its value to the caller, so these tests check the storage classes, not only spirv-val.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import DxrRayInterfaceUnsupportedError
from tests.test_translator.test_dxr_glsl_ray_interfaces import (
    CALLABLE,
    CLOSEST_HIT,
    INTERSECTION,
    RAYGEN,
)

MISS = """
struct Payload { float3 color; };

[shader("miss")]
void main(inout Payload payload)
{
    payload.color = float3(0.0, 0.0, RayTCurrent());
}
"""

# Every ray state intrinsic a hit shader reads; DXR's InstanceIndex() is SPIR-V's InstanceId, InstanceID() its
# InstanceCustomIndexKHR, and RayTCurrent() its RayTmaxKHR (DXR specification, "Ray tracing intrinsics").
HIT_STATE = """
struct Payload { float3 color; };
struct Attributes { float2 bary; };

[shader("closesthit")]
void main(inout Payload payload, in Attributes attribs)
{
    float3 p = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 local = mul(float4(p, 1.0), WorldToObject4x3());
    float3 back = mul(ObjectToWorld3x4(), float4(local, 1.0));
    float ids = (float)PrimitiveIndex() + (float)InstanceIndex() + (float)InstanceID() + (float)GeometryIndex();
    payload.color = back + float3(attribs.bary, ids);
    if (RayFlags() & RAY_FLAG_CULL_BACK_FACING_TRIANGLES) {
        payload.color = local;
    }
}
"""


def _translate(tmp_path, source, name):
    path = tmp_path / f"{name}.hlsl"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend="vulkan", source_backend="directx", format_output=False
    )


def _spirv_val(tmp_path, code):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "out.spvasm"
    asm.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.2", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.2", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _lines(code):
    return [line.strip() for line in code.splitlines()]


def _variable(code, storage_class):
    """The result id of the single module-scope variable in ``storage_class``."""
    ids = re.findall(rf"^\s*%(\w+) = OpVariable %\w+ {storage_class}$", code, re.M)
    assert len(ids) == 1, (storage_class, ids)
    return ids[0]


def _builtins(code):
    return set(re.findall(r"OpDecorate %\w+ BuiltIn (\w+)", code))


def _assert_no_located_input(code):
    located = set(re.findall(r"OpDecorate %(\w+) Location", code))
    inputs = set(re.findall(r"^\s*%(\w+) = OpVariable %\w+ Input$", code, re.M))
    assert not located & inputs, "a ray interface became a located stage input"


def test_raygen_traces_with_its_payload_variable(tmp_path):
    code = _translate(tmp_path, RAYGEN, "trace.rgen")
    payload = _variable(code, "RayPayloadKHR")
    trace = next(line for line in _lines(code) if line.startswith("OpTraceRayKHR"))
    # OpTraceRayKHR's last operand is the payload variable, not GLSL's location number.
    assert trace.split()[-1] == f"%{payload}"
    assert len(trace.split()) == 12
    _spirv_val(tmp_path, code)


def test_closest_hit_payload_attributes_and_callable_data_use_ray_storage(
    tmp_path,
):
    code = _translate(tmp_path, CLOSEST_HIT, "hit.rchit")
    _variable(code, "IncomingRayPayloadKHR")
    _variable(code, "HitAttributeKHR")
    data = _variable(code, "CallableDataKHR")
    call = next(
        line for line in _lines(code) if line.startswith("OpExecuteCallableKHR")
    )
    assert call.split()[-1] == f"%{data}"
    _assert_no_located_input(code)
    _spirv_val(tmp_path, code)


def test_miss_payload_reaches_the_caller(tmp_path):
    code = _translate(tmp_path, MISS, "miss.rmiss")
    payload = _variable(code, "IncomingRayPayloadKHR")
    assert re.search(rf"OpAccessChain %\w+ %{payload} ", code)
    assert "RayTmaxKHR" in _builtins(code)
    _assert_no_located_input(code)
    _spirv_val(tmp_path, code)


def test_callable_and_intersection_interfaces(tmp_path):
    callable_code = _translate(tmp_path, CALLABLE, "call.rcall")
    _variable(callable_code, "IncomingCallableDataKHR")
    assert "LaunchIdKHR" in _builtins(callable_code)
    _spirv_val(tmp_path, callable_code)

    intersection = _translate(tmp_path, INTERSECTION, "box.rint")
    attributes = _variable(intersection, "HitAttributeKHR")
    assert re.search(rf"OpStore %{attributes} ", intersection)
    assert any("OpReportIntersectionKHR" in line for line in _lines(intersection))
    _spirv_val(tmp_path, intersection)


def test_hit_state_intrinsics_read_the_ray_builtins(tmp_path):
    code = _translate(tmp_path, HIT_STATE, "state.rchit")
    assert {
        "WorldRayOriginKHR",
        "WorldRayDirectionKHR",
        "RayTmaxKHR",
        "PrimitiveId",
        "InstanceId",
        "InstanceCustomIndexKHR",
        "RayGeometryIndexKHR",
        "IncomingRayFlagsKHR",
        "ObjectToWorldKHR",
        "WorldToObjectKHR",
    } <= _builtins(code)
    # InstanceIndex() is the instance's index in the acceleration structure, not the vertex stage's InstanceIndex.
    assert "InstanceIndex" not in _builtins(code)
    assert "GeometryIndexKHR" not in _builtins(code)
    # RAY_FLAG_CULL_BACK_FACING_TRIANGLES is RayFlags bit 0x10.
    assert re.search(r"OpConstant %\w+ 16$", code, re.M)
    _spirv_val(tmp_path, code)


def test_transforms_keep_their_orientation(tmp_path):
    code = _translate(tmp_path, HIT_STATE, "state.rchit")
    world_to_object = re.search(
        r"OpDecorate %(\w+) BuiltIn WorldToObjectKHR", code
    ).group(1)
    object_to_world = re.search(
        r"OpDecorate %(\w+) BuiltIn ObjectToWorldKHR", code
    ).group(1)
    # Both built-ins are four columns of three floats, the 3x4 affine transform.
    matrix = re.search(r"%(\w+) = OpTypeMatrix %(\w+) 4", code)
    assert matrix and re.search(rf"%{matrix.group(2)} = OpTypeVector %\w+ 3", code)
    loads = {
        match.group(2): match.group(1)
        for match in re.finditer(r"%(\w+) = OpLoad %\w+ %(\w+)$", code, re.M)
    }
    # WorldToObject4x3() is the transpose: the row vector times it is OpVectorTimesMatrix on the transpose.
    transposed = re.search(
        rf"%(\w+) = OpTranspose %\w+ %{loads[world_to_object]}$", code, re.M
    )
    assert transposed
    assert re.search(
        rf"OpVectorTimesMatrix %\w+ %\w+ %{transposed.group(1)}$", code, re.M
    )
    # ObjectToWorld3x4() is the built-in itself, applied to a column vector.
    assert re.search(
        rf"OpMatrixTimesVector %\w+ %{loads[object_to_world]} %\w+$", code, re.M
    )
    _spirv_val(tmp_path, code)


def test_raygen_and_miss_in_one_library_get_their_own_interfaces(tmp_path):
    library = RAYGEN + MISS.replace("void main(", "void miss_main(").replace(
        "struct Payload { float3 color; };", ""
    )
    code = _translate(tmp_path, library, "library.rgen")
    outgoing = _variable(code, "RayPayloadKHR")
    incoming = _variable(code, "IncomingRayPayloadKHR")
    entries = {
        line.split()[1]: line.split()[4:]
        for line in _lines(code)
        if line.startswith("OpEntryPoint")
    }
    assert f"%{outgoing}" in entries["RayGenerationKHR"]
    assert f"%{incoming}" not in entries["RayGenerationKHR"]
    assert f"%{incoming}" in entries["MissKHR"]
    _spirv_val(tmp_path, code)


def test_several_stages_with_ray_parameters_are_refused_for_vulkan(tmp_path):
    library = MISS + CLOSEST_HIT.replace("void main(", "void hit_main(").replace(
        "struct Payload { float3 color; };", ""
    )
    with pytest.raises(DxrRayInterfaceUnsupportedError) as raised:
        _translate(tmp_path, library, "library.rmiss")
    assert raised.value.missing_capabilities == ("vulkan.ray-library",)
    assert "vulkan-ray-interface-unsupported" in str(raised.value)


GLSL_HIT = """#version 460
#extension GL_EXT_ray_tracing : require
struct Payload { vec3 color; };
layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec2 bary;
void main()
{
    vec4 p4 = gl_ObjectToWorld3x4EXT * vec3(1.0, 2.0, 3.0);
    vec3 w = gl_WorldToObjectEXT * vec4(p4.xyz, 1.0);
    if ((gl_IncomingRayFlagsEXT & gl_RayFlagsCullOpaqueEXT) != 0u
        && gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT) {
        payload.color = w * gl_RayTmaxEXT + vec3(float(gl_PrimitiveID + gl_InstanceID));
    }
}
"""

GLSL_VERTEX = """#version 450
layout(location = 0) out float instance;
void main()
{
    instance = float(gl_InstanceID);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""


def _translate_glsl(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path), backend="vulkan", source_backend="opengl", format_output=False
    )


def test_glsl_ray_builtins_and_constants(tmp_path):
    code = _translate_glsl(tmp_path, GLSL_HIT, "hit.rchit")
    builtins = _builtins(code)
    # GLSL_EXT_ray_tracing section 7.1: gl_PrimitiveID and gl_InstanceID are the hit's primitive and instance.
    assert {
        "PrimitiveId",
        "InstanceId",
        "IncomingRayFlagsKHR",
        "HitKindKHR",
    } <= builtins
    assert {"ObjectToWorldKHR", "WorldToObjectKHR", "RayTmaxKHR"} <= builtins
    assert "InstanceIndex" not in builtins
    # gl_ObjectToWorld3x4EXT has no built-in of its own; it is the transpose of ObjectToWorldKHR.
    assert "OpTranspose" in code
    # gl_RayFlagsCullOpaqueEXT is 0x40 and gl_HitKindBackFacingTriangleEXT 0xFF.
    assert re.search(r"OpConstant %\w+ 64$", code, re.M)
    assert re.search(r"OpConstant %\w+ 255$", code, re.M)
    _spirv_val(tmp_path, code)


def test_vertex_instance_id_stays_the_instance_index(tmp_path):
    code = _translate_glsl(tmp_path, GLSL_VERTEX, "instance.vert")
    assert "InstanceIndex" in _builtins(code)
    assert "InstanceId" not in _builtins(code)
