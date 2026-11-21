"""GLSL ray tracing stages reach Slang in the DXR form: entry parameters, ``static`` copies and attribute arguments.

GL_EXT_ray_tracing declares a stage's payload, hit attributes and callable data as module-scope variables, and
``reportIntersectionEXT(t, kind)`` reports the hit attribute variable implicitly. The Slang generator copied that form:
slangc treats a non-``static`` global as a uniform, so a write to it failed with E30011, and ``ReportHit`` without
its attribute argument with E30439. ``crosstl/translator/codegen/slang_glsl_ray_interfaces.py`` now runs the DXR
rewrite of :mod:`crosstl.translator.semantics.hlsl_ray_interfaces` for a Slang program with such variables, and gives
``ReportHit(t, kind)`` placeholder attributes in any other; a CrossGL ray stage without them keeps the Slang
generator's own forms.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import SlangRefusal

HEADER = "#version 460\n#extension GL_EXT_ray_tracing : require\n"

INTERSECTION = (
    HEADER
    + "hitAttributeEXT vec2 attribs;\n"
    + "void main() { attribs = vec2(0.5); reportIntersectionEXT(1.0, 0u); }\n"
)

INTERSECTION_WITHOUT_ATTRIBUTES = (
    HEADER + "void main() { reportIntersectionEXT(1.0, 0u); }\n"
)

CLOSEST_HIT = (
    HEADER
    + "layout(location = 0) rayPayloadInEXT vec3 hitValue;\n"
    + "hitAttributeEXT vec2 attribs;\n"
    + "void main() {\n"
    + "    vec3 b = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);\n"
    + "    hitValue = b * gl_HitTEXT + vec3(float(gl_PrimitiveID));\n"
    + "}\n"
)

ANY_HIT = (
    HEADER
    + "layout(location = 0) rayPayloadInEXT vec3 hitValue;\n"
    + "hitAttributeEXT vec2 attribs;\n"
    + "void main() {\n"
    + "    hitValue = vec3(attribs, 0.0);\n"
    + "    if (attribs.x > 0.5) ignoreIntersectionEXT;\n"
    + "}\n"
)

MISS = (
    HEADER
    + "layout(location = 0) rayPayloadInEXT vec3 hitValue;\n"
    + "void main() { hitValue = gl_WorldRayDirectionEXT * 0.5; }\n"
)

RAYGEN = (
    HEADER
    + "layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;\n"
    + "layout(set = 0, binding = 1, rgba8) uniform image2D img;\n"
    + "layout(location = 0) rayPayloadEXT vec3 hitValue;\n"
    + "layout(location = 1) callableDataEXT vec4 cdata;\n"
    + "void main() {\n"
    + "    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy);\n"
    + "    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(uv, 0.0), 0.001,\n"
    + "                vec3(0.0, 0.0, 1.0), 100.0, 0);\n"
    + "    executeCallableEXT(0, 1);\n"
    + "    imageStore(img, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 1.0) + cdata);\n"
    + "}\n"
)

CALLABLE = (
    HEADER
    + "layout(location = 1) callableDataInEXT vec4 cdata;\n"
    + "void main() { cdata = vec4(1.0, 0.5, 0.25, 1.0); }\n"
)

# A closest-hit stage that traces a second ray with a struct payload of its own.
RECURSIVE_HIT = (
    HEADER
    + "struct P { vec3 c; uint depth; };\n"
    + "layout(location = 0) rayPayloadInEXT P pl;\n"
    + "layout(location = 1) rayPayloadEXT P sec;\n"
    + "layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;\n"
    + "void main() {\n"
    + "    if (pl.depth < 2u) {\n"
    + "        sec.depth = pl.depth + 1u;\n"
    + "        traceRayEXT(tlas, 0u, 0xffu, 0u, 1u, 0u, gl_WorldRayOriginEXT, 0.01,\n"
    + "                    gl_WorldRayDirectionEXT, 10.0, 1);\n"
    + "        pl.c = sec.c * 0.5;\n"
    + "    } else {\n"
    + "        pl.c = vec3(gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT ? 1.0 : 0.0);\n"
    + "    }\n"
    + "}\n"
)

STAGES = {
    "rint": (INTERSECTION, "intersection"),
    "rint-unattributed": (INTERSECTION_WITHOUT_ATTRIBUTES, "intersection"),
    "rchit": (CLOSEST_HIT, "closesthit"),
    "rahit": (ANY_HIT, "anyhit"),
    "rmiss": (MISS, "miss"),
    "rgen": (RAYGEN, "raygeneration"),
    "rcall": (CALLABLE, "callable"),
}


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader.{suffix.split('-')[0]}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="slang", format_output=False)


def _compact(code):
    return re.sub(r"\s+", " ", code)


def test_intersection_reports_the_static_hit_attribute_variable(tmp_path):
    code = _compact(_translate(tmp_path, INTERSECTION, "rint"))
    assert "struct CrossGLHitAttributes_attribs { float2 value; };" in code, code
    assert "static CrossGLHitAttributes_attribs attribs;" in code, code
    assert "attribs.value = float2(0.5);" in code, code
    assert "ReportHit(1.0, 0u, attribs);" in code, code
    assert "float2 attribs;" not in code, code


def test_an_intersection_without_attributes_reports_a_placeholder(tmp_path):
    code = _compact(_translate(tmp_path, INTERSECTION_WITHOUT_ATTRIBUTES, "rint"))
    assert "struct CrossGLHitAttributesUnused { uint unused; };" in code, code
    assert "static CrossGLHitAttributesUnused crossgl_hit_attributes;" in code, code
    assert "ReportHit(1.0, 0u, crossgl_hit_attributes);" in code, code


def test_a_crossgl_stage_without_glsl_interfaces_keeps_the_slang_forms(tmp_path):
    source = """
        shader S {
            struct Payload { vec3 color; };
            ray_closest_hit {
                void main(inout Payload payload @ray_payload) {
                    payload.color = vec3(float(gl_InstanceCustomIndexEXT), gl_HitTEXT, 0.0);
                }
            }
        }
        """
    code = _compact(_translate(tmp_path, source, "cgl"))
    assert "void main(inout Payload payload : ray_payload)" in code, code
    assert "float3(float(InstanceID()), RayTCurrent(), 0.0)" in code, code
    assert "crossgl" not in code.lower(), code


def test_closest_hit_receives_the_payload_and_attributes_as_parameters(tmp_path):
    code = _compact(_translate(tmp_path, CLOSEST_HIT, "rchit"))
    assert re.search(
        r'\[shader\("closesthit"\)\] void main\(inout CrossGLRayPayload_hitValue (\w+), '
        r"in CrossGLHitAttributes_attribs (\w+)\)",
        code,
    ), code
    payload, attributes = re.search(
        r"void main\(inout \w+ (\w+), in \w+ (\w+)\)", code
    ).groups()
    assert f"hitValue = {payload}; attribs = {attributes};" in code, code
    assert "RayTCurrent()" in code and "PrimitiveIndex()" in code, code
    # The payload is copied back to the caller at the end of the entry.
    assert code.rstrip().endswith(f"{payload} = hitValue; }}"), code


def test_any_hit_copies_the_payload_back_before_ignoring_the_hit(tmp_path):
    code = _compact(_translate(tmp_path, ANY_HIT, "rahit"))
    payload = re.search(r"void main\(inout \w+ (\w+),", code).group(1)
    assert re.search(rf"{payload} = hitValue; IgnoreHit\(\);", code), code


def test_ray_generation_traces_with_a_ray_description_and_the_payload(tmp_path):
    code = _compact(_translate(tmp_path, RAYGEN, "rgen"))
    assert "static CrossGLRayPayload_hitValue hitValue;" in code, code
    assert "static CrossGLCallableData_cdata cdata;" in code, code
    trace = re.search(r"TraceRay\((.*?)\);", code).group(1)
    assert trace == (
        "tlas, RAY_FLAG_FORCE_OPAQUE, 255, 0, 0, 0, crossgl_ray_desc(float3(uv, 0.0), 0.001, "
        "float3(0.0, 0.0, 1.0), 100.0), hitValue"
    ), trace
    assert "RayDesc crossgl_ray_desc(" in code, code
    assert "CallShader(0, cdata);" in code, code
    assert "hitValue.value" in code and "cdata.value" in code, code


def test_callable_receives_its_data_as_an_inout_parameter(tmp_path):
    code = _compact(_translate(tmp_path, CALLABLE, "rcall"))
    data = re.search(r"void main\(inout CrossGLCallableData_cdata (\w+)\)", code)
    assert data, code
    assert f"cdata = {data.group(1)};" in code, code
    assert f"{data.group(1)} = cdata;" in code, code


def test_a_construct_without_a_dxr_form_is_refused_for_slang(tmp_path):
    source = (
        HEADER
        + "layout(location = 0) rayPayloadInEXT vec3 hitValue;\n"
        + "void main() { hitValue = gl_ObjectToWorldEXT * vec4(gl_ObjectRayOriginEXT, 1.0); }\n"
    )
    with pytest.raises(SlangRefusal) as raised:
        _translate(tmp_path, source, "rchit")
    message = str(raised.value)
    assert message.startswith(
        "Slang ray_closest_hit stage: gl_ObjectToWorldEXT"
    ), message
    assert "DirectX" not in message, message
    assert raised.value.target == "slang"


def _slangc():
    return shutil.which("slangc")


@pytest.mark.skipif(_slangc() is None, reason="slangc is not available")
@pytest.mark.parametrize(
    ("suffix", "source", "stage"),
    [
        *((suffix, source, stage) for suffix, (source, stage) in STAGES.items()),
        ("rchit", RECURSIVE_HIT, "closesthit"),
    ],
    ids=[*STAGES, "rchit-recursive"],
)
def test_slangc_compiles_the_stage_to_valid_spirv(tmp_path, suffix, source, stage):
    shader = tmp_path / "out.slang"
    shader.write_text(_translate(tmp_path, source, suffix))
    binary = tmp_path / "out.spv"
    result = subprocess.run(
        [
            _slangc(),
            str(shader),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            stage,
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + shader.read_text()
    if shutil.which("spirv-val") is not None:
        validated = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(binary)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert validated.returncode == 0, validated.stderr
