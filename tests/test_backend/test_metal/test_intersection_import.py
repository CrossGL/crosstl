"""Metal intersection functions imported as CrossGL intersection stages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused
from tests.test_translator.test_metal_builtin_ownership import _compile

ROOT = Path(__file__).resolve().parents[3]
REQUIRE_ENV = "CROSTL_REQUIRE_METAL_INTERSECTION"
HARNESS = (
    ROOT / "tests/fixtures/runtime_verification/metal_bounding_box_intersection.swift"
)
HEADER = (
    "#include <metal_stdlib>\nusing namespace metal;\nusing namespace raytracing;\n\n"
)

SPHERE = HEADER + """struct BoxResult {
    bool accept [[accept_intersection]];
    bool more [[continue_search]];
    float distance [[distance]];
};

[[intersection(bounding_box)]]
BoxResult sphere(float3 origin [[origin]], float3 direction [[direction]], float lo [[min_distance]],
                 float hi [[max_distance]], device float* outp [[buffer(0)]]) {
    float b = dot(origin, direction);
    float c = dot(origin, origin) - 1.0;
    float disc = b * b - c;
    BoxResult result;
    result.accept = false;
    result.more = true;
    result.distance = hi;
    if (disc >= 0.0) {
        float t = -b - sqrt(disc);
        if (t >= lo && t <= hi) {
            result.accept = true;
            result.distance = t;
            outp[0] += 1.0;
        }
    }
    return result;
}
"""

TRIANGLE = HEADER + """struct Payload {
    float3 color;
};

[[intersection(triangle)]]
bool probe(ray_data Payload& payload [[payload]], uint prim [[primitive_id]], float lo [[min_distance]],
           float3 origin [[origin]], device float* outp [[buffer(0)]]) {
    payload.color = origin;
    outp[prim] = lo;
    return lo > 0.25;
}
"""


def _translate(tmp_path, source, backend, name="intersection.metal"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_bounding_box_function_imports_as_an_intersection_stage(tmp_path):
    crossgl = _translate(tmp_path, SPHERE, "cgl")

    assert "ray_intersection {" in crossgl
    assert "bool accept @accept_intersection;" in crossgl
    assert "bool more @continue_search;" in crossgl
    assert "float distance @distance;" in crossgl
    assert (
        "BoxResult sphere(vec3 origin @origin, vec3 direction @direction, "
        "float lo @min_distance, float hi @max_distance, "
        "RWStructuredBuffer<float> outp @buffer(0)) @bounding_box {"
    ) in crossgl
    assert "@intersection" not in crossgl


def test_bounding_box_function_round_trips_to_metal(tmp_path):
    generated = _translate(tmp_path, SPHERE, "metal")

    assert "[[intersection(bounding_box)]] BoxResult sphere(" in generated
    assert "bool more [[continue_search]];" in generated
    assert "float lo [[min_distance]]" in generated
    assert "device float* outp [[buffer(0)]]" in generated
    if shutil.which("xcrun"):
        _compile(generated, "metal", tmp_path)


def test_triangle_function_round_trips_with_its_payload(tmp_path):
    crossgl = _translate(tmp_path, TRIANGLE, "cgl")
    assert "inout Payload payload @ray_data @payload" in crossgl
    assert ") @triangle {" in crossgl

    generated = _translate(tmp_path, TRIANGLE, "metal")
    assert "[[intersection(triangle)]] bool probe(" in generated
    assert "ray_data Payload& payload [[payload]]" in generated
    assert "uint prim [[primitive_id]]" in generated
    assert "device float* outp [[buffer(0)]]" in generated
    if shutil.which("xcrun"):
        _compile(generated, "metal", tmp_path)


INSTANCED = HEADER + """struct BoxResult {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

[[intersection(bounding_box, instancing, world_space_data)]]
BoxResult probe(float3 origin [[origin]], float lo [[min_distance]], uint prim [[primitive_id]],
                uint geometry [[geometry_id]], uint instance [[instance_id]], uint user [[user_instance_id]],
                float3 world [[world_space_origin]], float4x3 to_world [[object_to_world_transform]],
                device float* outp [[buffer(0)]]) {
    outp[0] = float(prim + geometry + instance + user) + world.x + to_world[3].x + origin.y;
    BoxResult result;
    result.accept = true;
    result.distance = lo;
    return result;
}
"""

TRIANGLE_DATA = (
    HEADER + """[[intersection(triangle, triangle_data, instancing, max_levels<2>)]]
bool probe(float2 uv [[barycentric_coord]], bool front [[front_facing]], float t [[distance]],
           uint geometry [[geometry_id]]) {
    return front && uv.x < 0.5 && t > 0.0 && geometry != 3;
}
"""
)


def test_intersection_tags_and_ray_inputs_keep_their_metal_names(tmp_path):
    crossgl = _translate(tmp_path, INSTANCED, "cgl")

    assert (
        "BoxResult probe(vec3 origin @origin, float lo @min_distance, uint prim @primitive_id, "
        "uint geometry @geometry_id, uint instance @instance_id, uint user @user_instance_id, "
        "vec3 world @world_space_origin, mat4x3 to_world @object_to_world_transform, "
        "RWStructuredBuffer<float> outp @buffer(0)) @bounding_box @instancing @world_space_data {"
    ) in crossgl
    assert "gl_PrimitiveID" not in crossgl
    assert "gl_InstanceIndex" not in crossgl

    triangle = _translate(tmp_path, TRIANGLE_DATA, "cgl", name="triangle.metal")
    assert (
        "bool probe(vec2 uv @barycentric_coord, bool front @front_facing, float t @distance, "
        "uint geometry @geometry_id) @triangle @triangle_data @instancing @max_levels(2) {"
    ) in triangle
    assert "gl_BaryCoord" not in triangle
    assert "gl_FrontFacing" not in triangle


@pytest.mark.parametrize(
    ("source", "attribute"),
    [
        (INSTANCED, "[[intersection(bounding_box, instancing, world_space_data)]]"),
        (
            TRIANGLE_DATA,
            "[[intersection(triangle, triangle_data, instancing, max_levels<2>)]]",
        ),
    ],
)
def test_tagged_functions_round_trip_to_metal(tmp_path, source, attribute):
    generated = _translate(tmp_path, source, "metal")

    assert attribute in generated
    if shutil.which("xcrun"):
        native = tmp_path / "native"
        native.mkdir()
        _compile(source, "metal", native)
        _compile(generated, "metal", tmp_path)


@pytest.mark.parametrize(
    ("attribute", "reason"),
    [
        (
            "bounding_box, instancing, unknown_tag",
            "names intersection tag 'unknown_tag'",
        ),
        ("bounding_box, max_levels", "names intersection tag 'max_levels'"),
        ("curve", "has primitive type 'curve'"),
    ],
)
def test_unrepresented_intersection_forms_are_refused(tmp_path, attribute, reason):
    source = (
        HEADER
        + f"[[intersection({attribute})]]\n"
        + "bool probe(float lo [[min_distance]]) {\n    return lo > 0.0;\n}\n"
    )
    with pytest.raises(MetalImportRefused, match=reason):
        _translate(tmp_path, source, "cgl")


@pytest.fixture(scope="module")
def intersection_runner(tmp_path_factory):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native Metal ray tracing")
    assert sys.platform == "darwin", "Native Metal execution requires macOS"
    runner = tmp_path_factory.mktemp("metal-intersection") / "trace"
    subprocess.run(
        ["swiftc", str(HARNESS), "-o", str(runner)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return runner


def _trace(runner, path, function="sphere", floats="1", tags=""):
    completed = subprocess.run(
        [str(runner), str(path), function, floats, "0:100", "4.5:100", "0:3.9"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CROSSGL_INTERSECTOR_TAGS": tags},
    )
    return json.loads(completed.stdout)


def test_round_tripped_function_matches_the_native_function_on_the_gpu(
    intersection_runner, tmp_path
):
    native = tmp_path / "native.metal"
    native.write_text(SPHERE, encoding="utf-8")
    translated = tmp_path / "translated.metal"
    translated.write_text(_translate(tmp_path, SPHERE, "metal"), encoding="utf-8")

    expected = _trace(intersection_runner, native)
    assert expected["hits"] == [
        {"distance": 4.0, "hit": True},
        {"distance": -1.0, "hit": False},
        {"distance": -1.0, "hit": False},
    ]
    assert expected["output"] == [1.0]
    assert _trace(intersection_runner, translated) == expected


INSTANCED_SLAB = HEADER + """struct BoxResult {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

[[intersection(bounding_box, instancing, world_space_data)]]
BoxResult slab(float3 origin [[origin]], float3 direction [[direction]], float lo [[min_distance]],
               float hi [[max_distance]], uint prim [[primitive_id]], uint geometry [[geometry_id]],
               uint instance [[instance_id]], uint user [[user_instance_id]],
               float3 world [[world_space_origin]], float3 world_direction [[world_space_direction]],
               float4x3 to_world [[object_to_world_transform]], float4x3 to_object [[world_to_object_transform]],
               device float* outp [[buffer(0)]]) {
    float t = (-1.0 - origin.z) / direction.z;
    BoxResult result;
    result.accept = t >= lo && t <= hi;
    result.distance = t;
    if (result.accept) {
        outp[0] = float(user);
        outp[1] = float(instance + prim + geometry);
        outp[2] = origin.z;
        outp[3] = world.z;
        outp[4] = to_world[3].z;
        outp[5] = to_object[3].z;
        outp[6] = world_direction.z;
    }
    return result;
}
"""


def test_round_tripped_instanced_function_matches_the_native_function_on_the_gpu(
    intersection_runner, tmp_path
):
    native = tmp_path / "native.metal"
    native.write_text(INSTANCED_SLAB, encoding="utf-8")
    translated = tmp_path / "translated.metal"
    translated.write_text(
        _translate(tmp_path, INSTANCED_SLAB, "metal"), encoding="utf-8"
    )
    trace = {"function": "slab", "floats": "7", "tags": "instancing,world_space_data"}

    expected = _trace(intersection_runner, native, **trace)
    # The harness moves the box's only instance by +1 along z and gives it user instance ID 7.
    assert expected["hits"] == [
        {"distance": 5.0, "hit": True},
        {"distance": 5.0, "hit": True},
        {"distance": -1.0, "hit": False},
    ]
    assert expected["output"] == [7.0, 0.0, -6.0, -5.0, 1.0, -1.0, 1.0]
    assert _trace(intersection_runner, translated, **trace) == expected
