"""Metal-style intersection results become hit reports on the targets whose intersection shaders call ReportHit.

A CrossGL ``ray_intersection`` entry may return Metal's result structure (a ``bool @accept_intersection`` and a
``float @distance`` member) and read its inputs from ``@min_distance`` and the other intersection function input
parameters. Metal output keeps that form. For HLSL, GLSL, SPIR-V and Slang the entry becomes ``void`` and reports
an accepted hit with ``ReportHit``; the inputs come from the ray state intrinsics.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import IntersectionResultUnsupportedError
from crosstl.translator.semantics.intersection_results import (
    lower_intersection_results,
)

RESULT = """
shader Probe {
    struct BoxResult {
        bool accept @ accept_intersection;
        float distance @ distance;
    }
    ray_intersection {
        BoxResult main() @bounding_box {
            BoxResult r;
            r.accept = true;
            r.distance = 1.0;
            return r;
        }
    }
}
"""

INPUTS = """
shader Probe {
    struct BoxResult {
        bool accept @ accept_intersection;
        float distance @ distance;
    }
    ray_intersection {
        BoxResult main(vec3 origin @ origin, float tmin @ min_distance, float tmax @ max_distance,
                       uint prim @ primitive_id, mat4x3 toWorld @ object_to_world_transform) @bounding_box {
            BoxResult r;
            r.accept = prim == 0u;
            r.distance = max(tmin, 1.0) + origin.x + toWorld[0].x;
            if (tmax < 1.0) {
                return r;
            }
            r.accept = false;
            return r;
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _squash(code):
    return re.sub(r"\s+", " ", code)


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_hlsl_style_targets_report_the_accepted_hit_with_empty_attributes(
    tmp_path, backend
):
    code = _squash(_translate(tmp_path, RESULT, backend))
    assert "struct IntersectionAttributes" in code
    assert re.search(r"\[shader\(\"intersection\"\)\] void \w+\(\)", code)
    assert (
        "BoxResult intersection_result = r; if (intersection_result.accept) "
        "{ IntersectionAttributes hit_attributes; "
        "ReportHit(intersection_result.distance, 0u, hit_attributes); } return;"
    ) in code
    # The result members lose Metal's roles, which HLSL would print as semantics.
    assert "accept_intersection" not in code
    assert "bounding_box" not in code


def test_glsl_reports_the_accepted_hit_with_report_intersection(tmp_path):
    code = _squash(_translate(tmp_path, RESULT, "opengl"))
    assert "void main()" in code
    assert (
        "BoxResult intersection_result = r; if (intersection_result.accept) "
        "{ reportIntersectionEXT(intersection_result.distance, 0u); } return;"
    ) in code
    assert "hitAttributeEXT" not in code


def test_spirv_reports_the_hit_and_declares_no_output_variable(tmp_path):
    code = _translate(tmp_path, RESULT, "vulkan")
    assert "OpReportIntersectionKHR" in code
    assert " Output" not in code


def test_metal_keeps_the_result_structure(tmp_path):
    code = _squash(_translate(tmp_path, RESULT, "metal"))
    assert "[[intersection(bounding_box)]] BoxResult" in code
    assert "bool accept [[accept_intersection]];" in code
    assert "return r;" in code


def test_inputs_read_the_ray_state_and_every_return_reports(tmp_path):
    hlsl = _squash(_translate(tmp_path, INPUTS, "directx"))
    for declaration in (
        "float3 origin = ObjectRayOrigin();",
        "float tmin = RayTMin();",
        "float tmax = RayTCurrent();",
        "uint prim = PrimitiveIndex();",
        "float3x4 toWorld = ObjectToWorld3x4();",
    ):
        assert declaration in hlsl
    assert hlsl.count("ReportHit(") == 2
    glsl = _squash(_translate(tmp_path, INPUTS, "opengl"))
    for declaration in (
        "vec3 origin = gl_ObjectRayOriginEXT;",
        "float tmin = gl_RayTminEXT;",
        "float tmax = gl_RayTmaxEXT;",
        "mat4x3 toWorld = gl_ObjectToWorldEXT;",
    ):
        assert declaration in glsl
    assert "uint prim = uint(gl_PrimitiveID);" in glsl
    assert glsl.count("reportIntersectionEXT(") == 2


def _refusal(source, detail):
    with pytest.raises(IntersectionResultUnsupportedError, match=detail):
        lower_intersection_results(_parse(source), "directx")


def test_triangle_intersection_functions_are_refused():
    source = """
    shader Probe {
        ray_intersection {
            bool main() @triangle {
                return true;
            }
        }
    }
    """
    _refusal(source, "triangle intersection function")
    _refusal(source.replace("@triangle", ""), "triangle intersection function")


@pytest.mark.parametrize(
    "members, detail",
    [
        (
            "bool accept @ accept_intersection; float distance @ distance; "
            "bool more @ continue_search;",
            "@continue_search",
        ),
        ("bool accept @ accept_intersection;", "float @distance member"),
        ("int accept @ accept_intersection; float distance @ distance;", "bool @acc"),
        (
            "bool accept @ accept_intersection; float distance @ distance; float extra;",
            "neither @accept_intersection nor @distance",
        ),
    ],
)
def test_result_structures_without_a_report_are_refused(members, detail):
    source = f"""
    shader Probe {{
        struct BoxResult {{ {members} }}
        ray_intersection {{
            BoxResult main() @bounding_box {{
                BoxResult r;
                return r;
            }}
        }}
    }}
    """
    _refusal(source, detail)


@pytest.mark.parametrize(
    "parameter, detail",
    [
        ("vec3 color @ payload", "not one intersection input"),
        ("float n", "not one intersection input"),
        ("int tmin @ min_distance", "must be a float"),
        ("vec3 toWorld @ object_to_world_transform", "must be a mat4x3"),
    ],
)
def test_parameters_without_an_intrinsic_are_refused(parameter, detail):
    source = RESULT.replace("BoxResult main()", f"BoxResult main({parameter})")
    _refusal(source, detail)


def test_a_program_function_named_like_the_intrinsic_is_refused():
    source = RESULT.replace(
        "    ray_intersection {",
        "    float RayTMin() { return 2.0; }\n    ray_intersection {",
    ).replace("BoxResult main()", "BoxResult main(float t @ min_distance)")
    _refusal(source, "defines a function named RayTMin")


def test_void_entries_and_metal_are_left_alone():
    program = _parse(RESULT)
    assert lower_intersection_results(program, "metal") == 0
    assert lower_intersection_results(program, "opengl") == 1
    assert lower_intersection_results(program, "opengl") == 0


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", [RESULT, INPUTS], ids=["result", "inputs"])
def test_glsl_and_spirv_outputs_validate(tmp_path, source):
    if shutil.which("glslangValidator"):
        (tmp_path / "probe.rint").write_text(_translate(tmp_path, source, "opengl"))
        _run(
            [
                "glslangValidator",
                "--target-env",
                "vulkan1.2",
                "-o",
                "g.spv",
                "probe.rint",
            ],
            tmp_path,
        )
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "probe.spvasm").write_text(_translate(tmp_path, source, "vulkan"))
        _run(
            ["spirv-as", "--target-env", "vulkan1.2", "probe.spvasm", "-o", "s.spv"],
            tmp_path,
        )
        _run(["spirv-val", "--target-env", "vulkan1.2", "s.spv"], tmp_path)
