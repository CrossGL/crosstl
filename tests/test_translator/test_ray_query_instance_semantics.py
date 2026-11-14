"""Ray query instance getters keep HLSL semantics on every target.

DXR's ``InstanceID()`` is the application-defined custom index of an instance.
``InstanceIndex()`` is the instance's position in the top-level acceleration
structure. In GLSL and SPIR-V the same values are
``rayQueryGetIntersectionInstanceCustomIndexEXT`` (``...CustomIndexKHR``) and
``rayQueryGetIntersectionInstanceIdEXT`` (``...InstanceIdKHR``).

The GLSL importer and the GLSL and SPIR-V generators gave CrossGL's
``CommittedInstanceID`` the GLSL meaning (the index), while the HLSL importer
and generator gave it the DXR meaning (the custom index). An HLSL ray query
translated to GLSL or SPIR-V therefore read the other value. The Slang
generator also swapped ``gl_InstanceID`` and ``gl_InstanceCustomIndexEXT``
in ray stages.

CrossGL now uses the DXR meaning everywhere.
"""

import textwrap

import pytest

import crosstl

CROSSGL = textwrap.dedent("""
    shader S {
        compute {
            buffer uint data[];
            layout(local_size_x = 1) in;
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                data[0] = rq.CommittedInstanceID();
                data[1] = rq.CommittedInstanceIndex();
                data[2] = rq.CandidateInstanceID();
                data[3] = rq.CandidateInstanceIndex();
            }
        }
    }
    """)

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
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(tlas, 0, 0xFF, ray);
        q.Proceed();
        outv[0] = q.CommittedInstanceID();
        outv[1] = q.CommittedInstanceIndex();
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
        rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, 0xFF,
                              vec3(0.0), 0.01, vec3(0.0, 0.0, 1.0), 1000.0);
        rayQueryProceedEXT(rq);
        values[0] = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
        values[1] = rayQueryGetIntersectionInstanceIdEXT(rq, true);
    }
    """)


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"query{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_glsl_lowers_instance_id_to_the_custom_index(tmp_path):
    code = _translate(tmp_path, CROSSGL, ".cgl", "opengl")
    assert "data[0] = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);" in code
    assert "data[1] = rayQueryGetIntersectionInstanceIdEXT(rq, true);" in code
    assert "data[2] = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false);" in code
    assert "data[3] = rayQueryGetIntersectionInstanceIdEXT(rq, false);" in code


def test_spirv_lowers_instance_id_to_the_custom_index(tmp_path):
    code = _translate(tmp_path, CROSSGL, ".cgl", "vulkan")
    assert code.count("OpRayQueryGetIntersectionInstanceCustomIndexKHR") == 2
    assert code.count("OpRayQueryGetIntersectionInstanceIdKHR") == 2


def test_hlsl_instance_getters_keep_their_meaning_in_glsl(tmp_path):
    code = _translate(tmp_path, HLSL, ".hlsl", "opengl")
    assert "= rayQueryGetIntersectionInstanceCustomIndexEXT(q, true);" in code
    assert "= rayQueryGetIntersectionInstanceIdEXT(q, true);" in code


def test_glsl_instance_getters_import_with_dxr_names(tmp_path):
    code = _translate(tmp_path, GLSL, ".comp", "cgl")
    assert "values[0] = rq.CommittedInstanceID();" in code
    assert "values[1] = rq.CommittedInstanceIndex();" in code


@pytest.mark.parametrize(
    ("builtin", "intrinsic"),
    [
        ("gl_InstanceCustomIndexEXT", "InstanceID()"),
        ("gl_InstanceID", "InstanceIndex()"),
    ],
)
def test_slang_ray_stage_instance_builtins(tmp_path, builtin, intrinsic):
    program = textwrap.dedent(f"""
        shader S {{
            struct Payload {{ uint value; }};
            ray_closest_hit {{
                void main(inout Payload payload @ray_payload) {{
                    payload.value = {builtin};
                }}
            }}
        }}
        """)
    code = _translate(tmp_path, program, ".cgl", "slang")
    assert f"payload.value = {intrinsic};" in code
