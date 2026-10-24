"""CrossGL spells ray query methods the DXR way on every importer and target.

The GLSL importer produced CrossGL-only names such as ``CommittedType``,
``CandidateRayT``, ``ConfirmIntersection``, ``GenerateIntersection``,
``...ShaderBindingTableRecordOffset`` and ``CandidateAABBOpaque``. The GLSL and
SPIR-V generators accepted only those names, so DXR-spelled IR from the HLSL
importer could not be lowered. ``CommittedStatus`` and
``CandidateTriangleRayT``, for example, reached the GLSL output as method calls.

CrossGL now uses the DXR names. ``CandidateProceduralPrimitiveNonOpaque`` asks
the DXR question (is the candidate non-opaque?), so it is the negation of the
GLSL and SPIR-V opaque query.
"""

import re
import textwrap

import crosstl

CROSSGL = textwrap.dedent("""
    shader S {
        compute {
            buffer uint data[];
            layout(local_size_x = 1) in;
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                while (rq.Proceed()) {
                    if (rq.CandidateType() == 0u) {
                        rq.CommitNonOpaqueTriangleHit();
                    } else if (rq.CandidateProceduralPrimitiveNonOpaque()) {
                        rq.CommitProceduralPrimitiveHit(1.0);
                    }
                    data[4] = uint(rq.CandidateTriangleRayT());
                    data[5] = rq.CandidateInstanceContributionToHitGroupIndex();
                }
                data[0] = rq.CommittedStatus();
                data[1] = rq.CommittedInstanceContributionToHitGroupIndex();
                rq.Abort();
            }
        }
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
        while (rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionCandidateAABBOpaqueEXT(rq)) {
                rayQueryGenerateIntersectionEXT(rq, 1.0);
            }
            rayQueryConfirmIntersectionEXT(rq);
            values[2] = uint(rayQueryGetIntersectionTEXT(rq, false));
        }
        values[0] = rayQueryGetIntersectionTypeEXT(rq, true);
        values[1] = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, true);
        rayQueryTerminateEXT(rq);
    }
    """)


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"query{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    # The GLSL formatter wraps long calls.
    return re.sub(r"\(\s+", "(", re.sub(r"\s+", " ", code))


def test_glsl_lowers_the_dxr_names(tmp_path):
    code = _flat(_translate(tmp_path, CROSSGL, ".cgl", "opengl"))
    for expected in (
        "rayQueryConfirmIntersectionEXT(rq);",
        "(!rayQueryGetIntersectionCandidateAABBOpaqueEXT(rq))",
        "rayQueryGenerateIntersectionEXT(rq, 1.0);",
        "rayQueryGetIntersectionTEXT(rq, false)",
        "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, false)",
        "data[0] = rayQueryGetIntersectionTypeEXT(rq, true);",
        "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, true)",
        "rayQueryTerminateEXT(rq);",
    ):
        assert expected in code, expected
    # No DXR method survives as a method call.
    assert not re.search(r"\brq\.\w+\(", code)


def test_spirv_lowers_the_dxr_names(tmp_path):
    code = _translate(tmp_path, CROSSGL, ".cgl", "vulkan")
    for opcode, count in (
        ("OpRayQueryConfirmIntersectionKHR", 1),
        ("OpRayQueryGenerateIntersectionKHR", 1),
        ("OpRayQueryGetIntersectionCandidateAABBOpaqueKHR", 1),
        ("OpRayQueryGetIntersectionTKHR", 1),
        ("OpRayQueryGetIntersectionTypeKHR", 2),
        ("OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR", 2),
        ("OpRayQueryTerminateKHR", 1),
    ):
        assert code.count(opcode) == count, opcode
    opaque = re.search(
        r"%(\w+) = OpRayQueryGetIntersectionCandidateAABBOpaqueKHR", code
    ).group(1)
    assert re.search(rf"= OpLogicalNot %\w+ %{opaque}\b", code)


def test_glsl_imports_with_dxr_names(tmp_path):
    code = _translate(tmp_path, GLSL, ".comp", "cgl")
    for expected in (
        "(!rq.CandidateProceduralPrimitiveNonOpaque())",
        "rq.CommitProceduralPrimitiveHit(1.0);",
        "rq.CommitNonOpaqueTriangleHit();",
        "rq.CandidateTriangleRayT()",
        "values[0] = rq.CommittedStatus();",
        "values[1] = rq.CommittedInstanceContributionToHitGroupIndex();",
        "rq.Abort();",
    ):
        assert expected in code, expected


def test_glsl_round_trip_keeps_the_queries(tmp_path):
    imported = _translate(tmp_path, GLSL, ".comp", "cgl")
    code = _flat(_translate(tmp_path, imported, ".cgl", "opengl"))
    assert "rayQueryGetIntersectionTypeEXT(rq, true)" in code
    assert "rayQueryGenerateIntersectionEXT(rq, 1.0);" in code
    assert "(!(!rayQueryGetIntersectionCandidateAABBOpaqueEXT(rq)))" in code
