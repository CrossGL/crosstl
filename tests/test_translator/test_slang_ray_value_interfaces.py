"""Slang ray payloads, hit attributes and callable data of a scalar, vector, matrix or array type.

DXC requires a user-defined structure for each of them, and the Slang generator copied that rule, so
``ReportHit(t, kind, attribute)`` with an ``int`` attribute (the ReportHit coverage probe in both Slang rows)
was refused. Slang accepts any value type there and wraps a non-structure in a structure itself when it writes
HLSL, so the generator now writes these arguments and parameters as they are.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import SlangRefusal

RAY_GENERATION = """
shader Probe {
    ray_generation {
        buffer float* outp;
        uniform accelerationStructureEXT scene;
        void main() {
            RayDesc ray;
            ray.Origin = vec3(0.0, 0.0, 0.0);
            ray.Direction = vec3(0.0, 0.0, 1.0);
            ray.TMin = 0.0;
            ray.TMax = 100.0;
            vec4 payload = vec4(0.0, 0.0, 0.0, 0.0);
            TraceRay(scene, RAY_FLAG_NONE, 255u, 0u, 1u, 0u, ray, payload);
            int data = 3;
            CallShader(0u, data);
            float pair[2];
            pair[0] = 1.0;
            pair[1] = 2.0;
            TraceRay(scene, RAY_FLAG_NONE, 255u, 0u, 1u, 0u, ray, pair);
            outp[0] = payload.x + float(data) + pair[1];
        }
    }
}
"""

INTERSECTION = """
shader Probe {
    ray_intersection {
        buffer float* fin;
        buffer int* iin;
        buffer uint* uin;
        buffer int* outp;
        void main() {
            float t = fin[0];
            uint kind = uin[0];
            int attribute = iin[0];
            bool reported = ReportHit(t, kind, attribute);
            outp[0] = reported ? 1 : 0;
        }
    }
}
"""

CALL_RESULT_ATTRIBUTE = """
shader Probe {
    uint makeAttribute() {
        return 1u;
    }

    ray_intersection {
        void main() {
            ReportHit(1.0, 0u, makeAttribute());
        }
    }
}
"""

CLOSEST_HIT = """
shader Probe {
    ray_closest_hit {
        void main(vec4 payload @ payload, vec2 attributes @ hit_attribute) {
            payload = vec4(attributes, 0.0, 1.0);
        }
    }
}
"""

ANY_HIT = """
shader Probe {
    ray_any_hit {
        void main(uint payload @ payload, float attributes @ hit_attribute) {
            payload = uint(attributes);
        }
    }
}
"""

MISS = """
shader Probe {
    ray_miss {
        void main(float payload @ payload) {
            payload = 7.0;
        }
    }
}
"""

CALLABLE = """
shader Probe {
    ray_callable {
        void main(mat2 data @ callable_data) {
            data[0][0] = 1.0;
        }
    }
}
"""

STAGES = [
    ("raygeneration", RAY_GENERATION),
    ("intersection", INTERSECTION),
    ("intersection", CALL_RESULT_ATTRIBUTE),
    ("closesthit", CLOSEST_HIT),
    ("anyhit", ANY_HIT),
    ("miss", MISS),
    ("callable", CALLABLE),
]


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="slang")


def _compact(code):
    return re.sub(r"\s+", " ", code)


def test_ray_generation_passes_value_payloads_and_callable_data(tmp_path):
    code = _compact(_translate(tmp_path, RAY_GENERATION))
    assert "float4 payload = float4(0.0, 0.0, 0.0, 0.0);" in code
    assert "TraceRay(scene, RAY_FLAG_NONE, 255u, 0u, 1u, 0u, ray, payload);" in code
    assert "CallShader(0u, data);" in code
    assert "TraceRay(scene, RAY_FLAG_NONE, 255u, 0u, 1u, 0u, ray, pair);" in code


def test_intersection_reports_a_scalar_hit_attribute(tmp_path):
    code = _compact(_translate(tmp_path, INTERSECTION))
    assert "bool reported = ReportHit(t, kind, attribute);" in code


def test_intersection_reports_a_call_result_as_the_hit_attribute(tmp_path):
    code = _compact(_translate(tmp_path, CALL_RESULT_ATTRIBUTE))
    assert "ReportHit(1.0, 0u, makeAttribute());" in code


@pytest.mark.parametrize(
    ("source", "signature"),
    [
        (CLOSEST_HIT, "void main(inout float4 payload, in float2 attributes)"),
        (ANY_HIT, "void main(inout uint payload, in float attributes)"),
        (MISS, "void main(inout float payload)"),
        (CALLABLE, "void main(inout float2x2 data)"),
    ],
)
def test_value_typed_entry_parameters_keep_their_type(tmp_path, source, signature):
    assert signature in _compact(_translate(tmp_path, source))


def test_a_resource_payload_is_still_refused(tmp_path):
    source = MISS.replace("float payload @ payload", "sampler2D payload @ payload")
    source = source.replace("payload = 7.0;", "")
    with pytest.raises(SlangRefusal, match="must use a structure, scalar, vector"):
        _translate(tmp_path, source)


def _slangc():
    return shutil.which("slangc")


@pytest.mark.skipif(_slangc() is None, reason="slangc is not available")
@pytest.mark.parametrize(
    ("stage", "source"),
    STAGES,
    ids=[f"{stage}-{index}" for index, (stage, _) in enumerate(STAGES)],
)
def test_slangc_compiles_the_stage_to_valid_spirv(tmp_path, stage, source):
    shader = tmp_path / "out.slang"
    shader.write_text(_translate(tmp_path, source))
    binary = tmp_path / "out.spv"
    result = subprocess.run(
        [
            _slangc(),
            str(shader),
            "-target",
            "spirv",
            "-stage",
            stage,
            "-entry",
            "main",
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    if shutil.which("spirv-val") is not None:
        validation = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(binary)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert validation.returncode == 0, validation.stderr
