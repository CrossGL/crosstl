"""Inline ray queries translate between HLSL, GLSL and SPIR-V.

Previously none of these directions produced code that compiled:

* the HLSL importer dropped the ``RayQuery<FLAGS>`` flags;
* GLSL output used DXR's ``RayDesc``, ``RAY_FLAG_*`` and ``COMMITTED_*`` names,
  none of which GLSL declares;
* HLSL output kept GLSL's ``rayQueryEXT`` type, its seven-operand
  initialization and its ``gl_*`` constants;
* SPIR-V output refused the ``RayDesc`` type.

Now:

* the type flags are combined into the initialization flags wherever the
  target has no typed query;
* each constant gets the target's spelling, with the same value;
* GLSL gets a ``RayDesc`` struct, and HLSL builds one with a generated helper.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl

HLSL = textwrap.dedent("""
    RaytracingAccelerationStructure tlas : register(t0);
    RWStructuredBuffer<uint> outv : register(u0);
    [numthreads(1, 1, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID)
    {
        RayDesc ray;
        ray.Origin = float3(0, 0, 0);
        ray.Direction = float3(0, 0, 1);
        ray.TMin = 0.01;
        ray.TMax = 1000.0;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(tlas, RAY_FLAG_CULL_NON_OPAQUE, 0xFF, ray);
        q.Proceed();
        outv[id.x] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1 : 0;
    }
    """)

GLSL = textwrap.dedent("""
    #version 460
    #extension GL_EXT_ray_query : require
    layout(local_size_x = 1) in;
    layout(binding = 0) uniform accelerationStructureEXT tlas;
    layout(std430, binding = 1) buffer Out { uint values[]; };
    void main() {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, tlas, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF,
                              vec3(0.0), 0.01, vec3(0.0, 0.0, 1.0), 1000.0);
        rayQueryProceedEXT(rq);
        if (rayQueryGetIntersectionTypeEXT(rq, true)
                == gl_RayQueryCommittedIntersectionTriangleEXT) {
            values[0] = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
        }
    }
    """)

CROSSGL_TWO_QUERIES = textwrap.dedent("""
    shader S {
        compute {
            accelerationStructureEXT tlas @binding(0);
            buffer uint data[];
            layout(local_size_x = 1) in;
            void main() {
                RayQuery<RAY_FLAG_FORCE_OPAQUE> opaque;
                RayQuery<RAY_FLAG_CULL_OPAQUE | RAY_FLAG_SKIP_TRIANGLES> culled;
                opaque.TraceRayInline(tlas, RAY_FLAG_NONE, 255, vec3(0.0), 0.0,
                                      vec3(0.0, 0.0, 1.0), 10.0);
                culled.TraceRayInline(tlas, RAY_FLAG_NONE, 255, vec3(0.0), 0.0,
                                      vec3(0.0, 0.0, 1.0), 10.0);
                data[0] = opaque.Proceed() && culled.Proceed() ? 1u : 0u;
            }
        }
    }
    """)


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"query{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return re.sub(r"\(\s+", "(", re.sub(r"\s+", " ", code))


def _run(commands, tmp_path):
    for command in commands:
        if shutil.which(command[0]) is None:
            pytest.skip(f"{command[0]} is not installed")
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def test_hlsl_import_keeps_the_query_flags(tmp_path):
    source = HLSL.replace(
        "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH",
        "RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_NON_OPAQUE",
    )
    code = _translate(tmp_path, source, ".hlsl", "cgl")
    assert "RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_NON_OPAQUE> q;" in code


def test_hlsl_to_glsl(tmp_path):
    code = _translate(tmp_path, HLSL, ".hlsl", "opengl")
    flat = _flat(code)
    assert "struct RayDesc {" in code
    assert (
        "rayQueryInitializeEXT(q, tlas, (gl_RayFlagsCullNoOpaqueEXT) | "
        "gl_RayFlagsTerminateOnFirstHitEXT, 255," in flat
    )
    assert "gl_RayQueryCommittedIntersectionTriangleEXT" in code
    assert not re.search(r"\b(RAY_FLAG_|COMMITTED_)", code)
    (tmp_path / "query.comp").write_text(code, encoding="utf-8")
    _run(
        [["glslangValidator", "--target-env", "vulkan1.2", "-S", "comp", "query.comp"]],
        tmp_path,
    )


def test_hlsl_to_spirv(tmp_path):
    code = _translate(tmp_path, HLSL, ".hlsl", "vulkan")
    # RAY_FLAG_CULL_NON_OPAQUE (0x80) | ACCEPT_FIRST_HIT_AND_END_SEARCH (0x4).
    initialize = re.search(r"OpRayQueryInitializeKHR %\w+ %\w+ %(\w+)", code)
    assert initialize
    assert re.search(rf"%{initialize.group(1)} = OpConstant %\w+ 132$", code, re.M)
    (tmp_path / "query.spvasm").write_text(code, encoding="utf-8")
    _run(
        [
            ["spirv-as", "--target-env", "vulkan1.1", "query.spvasm", "-o", "q.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "q.spv"],
        ],
        tmp_path,
    )


def test_glsl_to_hlsl(tmp_path):
    code = _translate(tmp_path, GLSL, ".comp", "directx")
    assert "RayQuery<RAY_FLAG_NONE> rq;" in code
    assert "RayDesc __crossgl_make_ray_desc(" in code
    flat = _flat(code)
    assert (
        "rq.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 255, "
        "__crossgl_make_ray_desc(" in flat
    )
    assert "rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT" in code
    assert not re.search(r"\bgl_", code)
    entry = re.search(r"void (\w+)\(", code.split("[numthreads", 1)[1]).group(1)
    (tmp_path / "query.hlsl").write_text(code, encoding="utf-8")
    _run([["dxc", "-T", "cs_6_5", "-E", entry, "query.hlsl"]], tmp_path)


def test_each_query_type_keeps_its_own_flags_in_spirv(tmp_path):
    code = _translate(tmp_path, CROSSGL_TWO_QUERIES, ".cgl", "vulkan")
    assert code.count("OpTypeRayQueryKHR") == 1
    flags = re.findall(r"OpRayQueryInitializeKHR %\w+ %\w+ %(\w+)", code)
    values = [
        re.search(rf"%{flag} = OpConstant %\w+ (\d+)$", code, re.M).group(1)
        for flag in flags
    ]
    # FORCE_OPAQUE = 1; CULL_OPAQUE | SKIP_TRIANGLES = 0x40 | 0x100.
    assert values == ["1", "320"]


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_query_types_with_unknown_flags_are_refused(tmp_path, backend):
    source = CROSSGL_TWO_QUERIES.replace(
        "RayQuery<RAY_FLAG_FORCE_OPAQUE>", "RayQuery<SOME_FLAG>"
    )
    with pytest.raises(ValueError, match="template flags that are not ray flags"):
        _translate(tmp_path, source, ".cgl", backend)
