"""Ray flags of a ``TraceRay`` call become Metal intersector settings.

The Metal generator lowered ``TraceRay`` to ``intersector<>::intersect`` and dropped its ray flags, so culling,
forced opacity and first-hit termination were lost. See :mod:`crosstl.translator.codegen.metal_ray_flags`.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.metal_ray_flags import (
    RAY_FLAG_CONSTANTS,
    invalid_ray_flags_reason,
)
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from tests.conformance.corpus import checks

INTERSECT = "__crossgl_intersector_1.intersect(__crossgl_ray_0, topLevelAS, 255"


def program(flags, helper=False):
    trace = f"""
                TraceRay(topLevelAS, {flags}, 0xff, 0, 1, 0, vec3(0.0), 0.001,
                         vec3(0.0, 0.0, 1.0), 1000.0, 0);"""
    if helper:
        return f"""
shader rt {{
    accelerationStructureEXT topLevelAS @binding(0);
    ray_generation {{
        void trace(uint flags) {{{trace}
        }}
        void main() {{
            trace(RAY_FLAG_CULL_BACK_FACING_TRIANGLES);
        }}
    }}
}}
"""
    return f"""
shader rt {{
    accelerationStructureEXT topLevelAS @binding(0);
    ray_generation {{
        void main() {{{trace}
        }}
    }}
}}
"""


def translate(tmp_path, text):
    path = tmp_path / "shader.cgl"
    path.write_text(text)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def setters(code):
    """The intersector setter calls in emission order."""
    return [
        line.strip().rstrip(";")
        for line in code.splitlines()
        if line.strip().startswith("__crossgl_intersector_1.")
        and ".intersect(" not in line
    ]


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("RAY_FLAG_NONE", []),
        ("RAY_FLAG_FORCE_OPAQUE", ["force_opacity(forced_opacity::opaque)"]),
        ("RAY_FLAG_FORCE_NON_OPAQUE", ["force_opacity(forced_opacity::non_opaque)"]),
        (
            "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH",
            ["accept_any_intersection(true)"],
        ),
        ("RAY_FLAG_SKIP_CLOSEST_HIT_SHADER", []),
        (
            "RAY_FLAG_CULL_BACK_FACING_TRIANGLES",
            ["set_triangle_cull_mode(triangle_cull_mode::back)"],
        ),
        (
            "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES",
            ["set_triangle_cull_mode(triangle_cull_mode::front)"],
        ),
        ("RAY_FLAG_CULL_OPAQUE", ["set_opacity_cull_mode(opacity_cull_mode::opaque)"]),
        (
            "RAY_FLAG_CULL_NON_OPAQUE",
            ["set_opacity_cull_mode(opacity_cull_mode::non_opaque)"],
        ),
        (
            "RAY_FLAG_SKIP_TRIANGLES",
            ["set_geometry_cull_mode(geometry_cull_mode::triangle)"],
        ),
        (
            "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES",
            ["set_geometry_cull_mode(geometry_cull_mode::bounding_box)"],
        ),
        ("RAY_FLAG_FORCE_OMM_2_STATE", []),
    ],
)
def test_each_constant_flag_selects_its_intersector_setter(tmp_path, flag, expected):
    code = translate(tmp_path, program(flag))
    assert setters(code) == [f"__crossgl_intersector_1.{call}" for call in expected]
    assert INTERSECT in code
    assert "__crossgl_ray_flags" not in code


def test_combined_constant_flags_configure_the_intersector_before_it_traces(tmp_path):
    flags = (
        "RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE | "
        "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES"
    )
    code = translate(tmp_path, program(flags))
    assert setters(code) == [
        "__crossgl_intersector_1.force_opacity(forced_opacity::opaque)",
        "__crossgl_intersector_1.set_triangle_cull_mode(triangle_cull_mode::back)",
        "__crossgl_intersector_1.set_geometry_cull_mode(geometry_cull_mode::bounding_box)",
        "__crossgl_intersector_1.accept_any_intersection(true)",
    ]
    declaration = code.index(
        "intersector<triangle_data, instancing> __crossgl_intersector_1;"
    )
    assert declaration < code.index("accept_any_intersection") < code.index(INTERSECT)


def test_glsl_flag_spellings_and_integer_literals_fold(tmp_path):
    code = translate(tmp_path, program("gl_RayFlagsCullFrontFacingTrianglesEXT | 0x4"))
    assert setters(code) == [
        "__crossgl_intersector_1.set_triangle_cull_mode(triangle_cull_mode::front)",
        "__crossgl_intersector_1.accept_any_intersection(true)",
    ]


def test_runtime_flags_select_every_mode_from_one_evaluation(tmp_path):
    code = translate(tmp_path, program("flags", helper=True))
    assert "uint __crossgl_ray_flags_3 = uint(flags);" in code
    calls = setters(code)
    assert [call.split("(")[0] for call in calls] == [
        "__crossgl_intersector_1.force_opacity",
        "__crossgl_intersector_1.set_opacity_cull_mode",
        "__crossgl_intersector_1.set_triangle_cull_mode",
        "__crossgl_intersector_1.set_geometry_cull_mode",
        "__crossgl_intersector_1.accept_any_intersection",
    ]
    assert calls[0] == (
        "__crossgl_intersector_1.force_opacity(((__crossgl_ray_flags_3 & 1u) != 0u ? "
        "forced_opacity::opaque : ((__crossgl_ray_flags_3 & 2u) != 0u ? "
        "forced_opacity::non_opaque : forced_opacity::none)))"
    )
    assert calls[-1] == (
        "__crossgl_intersector_1.accept_any_intersection("
        "(__crossgl_ray_flags_3 & 4u) != 0u)"
    )
    assert code.count("uint(flags)") == 1


@pytest.mark.parametrize(
    ("flags", "fragment"),
    [
        (
            "RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_OPAQUE",
            "more than one of FORCE_OPAQUE",
        ),
        (
            "RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_CULL_NON_OPAQUE",
            "more than one of FORCE_OPAQUE",
        ),
        (
            "RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_CULL_FRONT_FACING_TRIANGLES",
            "cull both front-facing and back-facing",
        ),
        (
            "RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES",
            "combine SKIP_TRIANGLES",
        ),
        (
            "RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_CULL_BACK_FACING_TRIANGLES",
            "combine SKIP_TRIANGLES",
        ),
        ("0x800", "bits outside"),
        ("-1", "bits outside"),
    ],
)
def test_undefined_constant_flags_are_refused(tmp_path, flags, fragment):
    with pytest.raises(MetalLoweringRefused) as caught:
        translate(tmp_path, program(flags))
    assert caught.value.code == "project.translate.metal-lowering-unsupported"
    assert "TraceRay ray flags" in str(caught.value)
    assert fragment in str(caught.value)


def test_glsl_and_dxr_spellings_share_values():
    for name, value in RAY_FLAG_CONSTANTS.items():
        if name.startswith("gl_"):
            continue
        assert invalid_ray_flags_reason(value) is None, name
    pairs = {
        "gl_RayFlagsOpaqueEXT": "RAY_FLAG_FORCE_OPAQUE",
        "gl_RayFlagsNoOpaqueEXT": "RAY_FLAG_FORCE_NON_OPAQUE",
        "gl_RayFlagsTerminateOnFirstHitEXT": "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH",
        "gl_RayFlagsSkipClosestHitShaderEXT": "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER",
        "gl_RayFlagsCullBackFacingTrianglesEXT": "RAY_FLAG_CULL_BACK_FACING_TRIANGLES",
        "gl_RayFlagsCullFrontFacingTrianglesEXT": (
            "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES"
        ),
        "gl_RayFlagsCullOpaqueEXT": "RAY_FLAG_CULL_OPAQUE",
        "gl_RayFlagsCullNoOpaqueEXT": "RAY_FLAG_CULL_NON_OPAQUE",
        "gl_RayFlagsSkipTrianglesEXT": "RAY_FLAG_SKIP_TRIANGLES",
        "gl_RayFlagsSkipAABBEXT": "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES",
    }
    for glsl, dxr in pairs.items():
        assert RAY_FLAG_CONSTANTS[glsl] == RAY_FLAG_CONSTANTS[dxr]


@pytest.mark.parametrize(
    ("flags", "helper"),
    [
        (
            "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES | RAY_FLAG_CULL_NON_OPAQUE | "
            "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH",
            False,
        ),
        ("RAY_FLAG_SKIP_TRIANGLES", False),
        ("flags", True),
    ],
)
def test_configured_intersector_compiles(tmp_path, flags, helper):
    artifact = tmp_path / "out.metal"
    artifact.write_text(translate(tmp_path, program(flags, helper)))
    verdict = checks.check(artifact, "metal", ["compute"])
    if verdict["verdict"] == "unavailable":
        pytest.skip("no metal checker installed")
    assert verdict["verdict"] == "accepted", verdict.get("log", "")
