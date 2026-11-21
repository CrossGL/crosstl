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


@pytest.mark.parametrize(
    ("attribute", "reason"),
    [
        ("bounding_box, instancing", "names intersection tags instancing"),
        (
            "bounding_box, instancing, world_space_data",
            "names intersection tags instancing, world_space_data",
        ),
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


def _trace(runner, path):
    completed = subprocess.run(
        [str(runner), str(path), "sphere", "1", "0:100", "4.5:100", "0:3.9"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
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
