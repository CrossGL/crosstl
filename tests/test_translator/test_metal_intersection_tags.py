"""Intersection tags on Metal intersection functions generated from CrossGL ray_intersection entries."""

from __future__ import annotations

import shutil

import pytest

import crosstl
from crosstl.translator.refusals import MetalRefusal
from tests.test_translator.test_metal_builtin_ownership import _compile

RESULT = """    struct BoxResult {
        bool accept @accept_intersection;
        float distance @distance;
    }
"""


def _program(parameters, attributes="@bounding_box", body=None):
    returns = "bool" if "@triangle" in attributes else "BoxResult"
    if body is None:
        body = (
            "return true;"
            if returns == "bool"
            else "BoxResult r; r.accept = true; r.distance = 1.0; return r;"
        )
    return (
        "shader Probe {\n"
        + RESULT
        + "    ray_intersection {\n"
        + f"        {returns} main({parameters}) {attributes} {{\n"
        + f"            {body}\n"
        + "        }\n    }\n}\n"
    )


def _metal(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


@pytest.mark.parametrize(
    ("parameters", "attributes", "expected"),
    [
        (
            "uint instance @instance_id, uint user @user_instance_id",
            "@bounding_box",
            "[[intersection(bounding_box, instancing)]]",
        ),
        (
            "vec3 origin @world_space_origin, mat4x3 to_object @world_to_object_transform",
            "@bounding_box",
            "[[intersection(bounding_box, instancing, world_space_data)]]",
        ),
        (
            "vec2 uv @barycentric_coord, bool front @front_facing, float t @distance",
            "@triangle",
            "[[intersection(triangle, triangle_data)]]",
        ),
        (
            "float t @time, uint primitive @primitive_id",
            "@bounding_box @instance_motion @extended_limits",
            "[[intersection(bounding_box, instancing, primitive_motion, instance_motion, extended_limits)]]",
        ),
        (
            "float t @time",
            "@bounding_box",
            "[[intersection(bounding_box, primitive_motion)]]",
        ),
        (
            "uint geometry @geometry_id",
            "@bounding_box @max_levels(3)",
            "[[intersection(bounding_box, instancing, max_levels<3>)]]",
        ),
        (
            "uint primitive @primitive_id",
            "@world_space_data @instancing @bounding_box @curve_data",
            "[[intersection(bounding_box, instancing, world_space_data, curve_data)]]",
        ),
    ],
)
def test_entry_gets_the_tags_it_states_and_the_tags_its_inputs_need(
    tmp_path, parameters, attributes, expected
):
    generated = _metal(tmp_path, _program(parameters, attributes))

    assert expected in generated
    if shutil.which("xcrun"):
        _compile(generated, "metal", tmp_path)


def test_input_semantic_reaches_the_metal_attribute(tmp_path):
    generated = _metal(
        tmp_path,
        _program(
            "uint instance @instance_id, uint user @user_instance_id, uint primitive @primitive_id",
            body="BoxResult r; r.accept = instance + user + primitive > 0u; r.distance = 1.0; return r;",
        ),
    )

    assert "uint instance [[instance_id]]" in generated
    assert "uint user [[user_instance_id]]" in generated
    assert "uint primitive [[primitive_id]]" in generated


@pytest.mark.parametrize(
    ("parameters", "attributes", "reason"),
    [
        (
            "uint geometry @geometry_id",
            "@bounding_box @max_levels(0)",
            "positive integer",
        ),
        (
            "uint instance @instance_id",
            "@bounding_box @max_levels(2)",
            "multi-level instancing",
        ),
        (
            "vec2 uv @barycentric_coord",
            "@bounding_box",
            "not a supported intersection parameter",
        ),
    ],
)
def test_inputs_metal_cannot_take_are_refused(tmp_path, parameters, attributes, reason):
    with pytest.raises(MetalRefusal, match=reason):
        _metal(tmp_path, _program(parameters, attributes))
