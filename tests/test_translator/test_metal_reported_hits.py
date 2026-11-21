"""Intersection stages that report hits with ReportHit, lowered to Metal bounding-box intersection functions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from tests.metal_refusal_helpers import refused_metal
from tests.test_translator.test_metal_builtin_ownership import _compile

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_ENV = "CROSTL_REQUIRE_METAL_INTERSECTION"
HARNESS = (
    ROOT / "tests/fixtures/runtime_verification/metal_bounding_box_intersection.swift"
)

REPORTS = """shader Probe {
    struct Attributes {
        vec3 normal;
    }
    ray_intersection {
        buffer float* outp;
        void main() {
            Attributes attributes;
            attributes.normal = vec3(0.0, 0.0, 1.0);
            bool first = ReportHit(5.5, 0u, attributes);
            outp[0] = first ? 1.0 : 0.0;
            if (ReportHit(4.5, 1u, attributes)) {
                outp[1] = 1.0;
            }
            outp[2] = ReportHit(5.0, 2u, attributes) ? 1.0 : 0.0;
            if (outp[2] == 0.0) {
                return;
            }
            outp[3] = 7.0;
        }
    }
}
"""


def _metal(source):
    return MetalCodeGen().generate(parse(source))


def _entry(generated):
    return generated[generated.index("[[intersection(bounding_box)]]") :]


def test_reported_hits_become_the_bounding_box_result():
    generated = _metal(REPORTS)

    assert "bool accept [[accept_intersection]];" in generated
    assert "float distance [[distance]];" in generated
    entry = _entry(generated)
    assert entry.startswith("[[intersection(bounding_box)]] CrossGLIntersectionHit")
    assert "float crossgl_min_distance [[min_distance]]" in entry
    assert "float crossgl_max_distance [[max_distance]]" in entry
    assert "device float* outp [[buffer(0)]]" in entry
    assert "crossgl_hit.accept = false;" in entry
    assert "crossgl_hit.distance = crossgl_max_distance;" in entry
    assert (
        "bool first = crossgl_report_hit(crossgl_hit, 5.5, crossgl_min_distance);"
        in entry
    )
    assert "if (crossgl_report_hit(crossgl_hit, 4.5, crossgl_min_distance))" in entry
    assert entry.count("return crossgl_hit;") == 2
    assert "ReportHit" not in generated
    assert "t >= minDistance && t <= hit.distance" in generated


def test_reported_hits_compile_with_the_metal_compiler(tmp_path):
    if not shutil.which("xcrun"):
        pytest.skip("the Metal compiler is not installed")
    _compile(_metal(REPORTS), "metal", tmp_path)


def test_declared_distance_parameters_are_reused():
    generated = _metal("""shader Probe {
    struct Attributes { vec3 normal; }
    ray_intersection {
        void main(float lo @ min_distance, float hi @ max_distance, uint id @ primitive_id) {
            Attributes attributes;
            attributes.normal = vec3(float(id));
            ReportHit(lo + 0.5, 0u, attributes);
        }
    }
}
""")
    entry = _entry(generated)
    assert "crossgl_min_distance" not in entry
    assert "crossgl_max_distance" not in entry
    assert "crossgl_hit.distance = hi;" in entry
    assert "crossgl_report_hit(crossgl_hit, lo + 0.5, lo);" in entry
    assert "uint id [[primitive_id]]" in entry


def test_entry_without_reports_accepts_nothing():
    generated = _metal("""shader Probe {
    ray_intersection {
        buffer float* outp;
        void main() {
            outp[0] = 1.0;
        }
    }
}
""")
    entry = _entry(generated)
    assert "crossgl_hit.accept = false;" in entry
    assert entry.rstrip().endswith("return crossgl_hit;\n}")
    assert "crossgl_report_hit(" not in entry


def test_generated_names_avoid_program_names():
    generated = _metal("""shader Probe {
    struct CrossGLIntersectionHit { float value; }
    struct Attributes { vec3 normal; }
    float crossgl_report_hit(float x) { return x; }
    ray_intersection {
        buffer float* outp;
        void main() {
            Attributes attributes;
            attributes.normal = vec3(0.0);
            float crossgl_hit = crossgl_report_hit(2.0);
            outp[0] = crossgl_hit;
            ReportHit(crossgl_hit, 0u, attributes);
        }
    }
}
""")
    entry = _entry(generated)
    assert entry.startswith("[[intersection(bounding_box)]] CrossGLIntersectionHit_1")
    assert "float crossgl_hit = crossgl_report_hit(2.0);" in entry
    assert (
        "crossgl_report_hit_1(crossgl_hit_1, crossgl_hit, crossgl_min_distance)"
        in entry
    )


def test_other_ray_stages_keep_their_signatures():
    generated = _metal("""shader Probe {
    struct Attributes { vec3 normal; }
    ray_generation {
        buffer float* outp;
        void main() {
            outp[0] = 1.0;
        }
    }
    ray_intersection {
        void main() {
            Attributes attributes;
            attributes.normal = vec3(0.0);
            ReportHit(1.0, 0u, attributes);
        }
    }
}
""")
    assert "kernel void kernel_main(" in generated
    assert generated.count("CrossGLIntersectionHit crossgl_hit;") == 1


def test_input_program_is_not_modified():
    program = parse(REPORTS)
    before = deepcopy(program)
    MetalCodeGen().generate(program)
    entry = next(iter(program.stages.values())).entry_point
    assert str(entry.return_type) == str(
        next(iter(before.stages.values())).entry_point.return_type
    )
    assert not entry.parameters
    assert all(struct.name != "CrossGLIntersectionHit" for struct in program.structs)


def test_report_with_side_effecting_attributes_is_refused():
    source = """shader Probe {
    struct Attributes { vec3 normal; }
    Attributes make(inout float counter) { counter += 1.0; Attributes a; a.normal = vec3(counter); return a; }
    ray_intersection {
        void main() {
            float counter = 0.0;
            ReportHit(1.0, 0u, make(counter));
        }
    }
}
"""
    refusal = refused_metal(MetalCodeGen().generate, parse(source))
    assert "ReportHit hit kind or attributes with side effects" in refusal


def test_constructed_attributes_are_not_side_effects(tmp_path):
    generated = _metal("""shader Probe {
    struct Attributes { vec3 normal; }
    ray_intersection {
        void main() {
            ReportHit(1.0, uint(2), Attributes(vec3(0.0, 0.0, 1.0)));
        }
    }
}
""")
    assert "crossgl_report_hit(crossgl_hit, 1.0, crossgl_min_distance);" in generated


def test_reported_attributes_read_by_a_hit_stage_are_refused():
    source = """shader Probe {
    struct Payload { vec4 color; }
    struct Attributes { vec3 normal; }
    ray_intersection {
        void main() {
            Attributes attributes;
            attributes.normal = vec3(0.0, 0.0, 1.0);
            ReportHit(1.0, 0u, attributes);
        }
    }
    ray_closest_hit {
        void main(Payload payload @ payload, Attributes attributes @ hit_attribute) {
            payload.color = vec4(attributes.normal, 1.0);
        }
    }
}
"""
    refusal = refused_metal(MetalCodeGen().generate, parse(source))
    assert "@hit_attribute parameter 'attributes'" in refusal


def test_triangle_entry_returning_void_stays_refused():
    source = """shader Probe {
    struct Attributes { vec3 normal; }
    ray_intersection {
        void main() @triangle {
            Attributes attributes;
            attributes.normal = vec3(0.0);
            ReportHit(1.0, 0u, attributes);
        }
    }
}
"""
    refusal = refused_metal(MetalCodeGen().generate, parse(source))
    assert "triangle intersection stage" in refusal


def test_report_in_a_helper_stays_refused():
    source = """shader Probe {
    struct Attributes { vec3 normal; }
    bool report(Attributes attributes) { return ReportHit(1.0, 0u, attributes); }
    ray_intersection {
        void main() {
            Attributes attributes;
            attributes.normal = vec3(0.0);
            report(attributes);
        }
    }
}
"""
    refusal = refused_metal(MetalCodeGen().generate, parse(source))
    assert "ReportHit" in refusal


@pytest.mark.parametrize(
    ("suffix", "source"),
    [
        (
            ".hlsl",
            (
                "struct Attributes {\n    float3 normal;\n};\n"
                "RWStructuredBuffer<float> outp : register(u0);\n\n"
                '[shader("intersection")]\nvoid main() {\n    Attributes attributes;\n'
                "    attributes.normal = float3(0.0, 0.0, 1.0);\n"
                "    if (ReportHit(5.5, 0, attributes)) {\n        outp[0] = 1.0;\n    }\n}\n"
            ),
        ),
        (
            ".rint",
            (
                "#version 460\n#extension GL_EXT_ray_tracing : require\n"
                "struct Attributes {\n    vec3 normal;\n};\nhitAttributeEXT Attributes attributes;\n"
                "layout(std430, binding = 0) buffer Output {\n    float outp[];\n};\n"
                "void main() {\n    attributes.normal = vec3(0.0, 0.0, 1.0);\n"
                "    if (reportIntersectionEXT(5.5, 0u)) {\n        outp[0] = 1.0;\n    }\n}\n"
            ),
        ),
    ],
)
def test_imported_intersection_shader_compiles_for_metal(tmp_path, suffix, source):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source, encoding="utf-8")
    generated = crosstl.translate(str(path), backend="metal")
    entry = _entry(generated)
    assert "crossgl_report_hit(crossgl_hit, 5.5, crossgl_min_distance)" in entry
    if shutil.which("xcrun"):
        _compile(generated, "metal", tmp_path)


def test_hit_attribute_variable_becomes_an_entry_local():
    generated = _metal("""shader Probe {
    struct Attributes { vec3 normal; }
    Attributes attributes @hitAttributeEXT;
    ray_intersection {
        void main() {
            attributes.normal = vec3(0.0, 0.0, 1.0);
            ReportHit(1.0, 0u);
        }
    }
}
""")
    entry = _entry(generated)
    assert "Attributes attributes;" in entry
    assert "attributes.normal = float3(0.0, 0.0, 1.0);" in entry
    assert "constant Attributes" not in generated


def test_hit_attribute_variable_used_by_a_helper_is_refused():
    source = """shader Probe {
    struct Attributes { vec3 normal; }
    Attributes attributes @hitAttributeEXT;
    void fill() { attributes.normal = vec3(1.0); }
    ray_intersection {
        void main() {
            fill();
            ReportHit(1.0, 0u);
        }
    }
}
"""
    refusal = refused_metal(MetalCodeGen().generate, parse(source))
    assert "hit attribute variable 'attributes' is used by 'fill'" in refusal


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


def _trace(runner, tmp_path, generated, outputs, *intervals):
    source = tmp_path / "intersection.metal"
    source.write_text(generated, encoding="utf-8")
    completed = subprocess.run(
        [str(runner), str(source), "intersection_main", str(outputs), *intervals],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def test_reported_hits_run_on_the_gpu(intersection_runner, tmp_path):
    # ReportHit(5.5) is accepted, ReportHit(4.5) is closer and accepted, ReportHit(5.0) is farther than the
    # current hit and rejected, so the body returns before writing outp[3] and the hit is at 4.5.
    report = _trace(intersection_runner, tmp_path, _metal(REPORTS), 4, "0:100")
    assert report["hits"] == [{"distance": 4.5, "hit": True}]
    assert report["output"] == [1.0, 1.0, 0.0, 0.0]


def test_ray_interval_limits_the_reported_hits_on_the_gpu(
    intersection_runner, tmp_path
):
    generated = _metal("""shader Probe {
    struct Attributes { vec3 normal; }
    ray_intersection {
        void main() {
            Attributes attributes;
            attributes.normal = vec3(0.0, 0.0, 1.0);
            ReportHit(4.5, 0u, attributes);
            ReportHit(5.5, 1u, attributes);
        }
    }
}
""")
    report = _trace(
        intersection_runner, tmp_path, generated, 0, "0:100", "5:100", "0:4", "6:100"
    )
    assert report["hits"] == [
        {"distance": 4.5, "hit": True},
        {"distance": 5.5, "hit": True},
        {"distance": -1.0, "hit": False},
        {"distance": -1.0, "hit": False},
    ]
