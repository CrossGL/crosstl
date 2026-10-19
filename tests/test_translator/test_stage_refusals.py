"""Stages and stage interfaces a target cannot declare are typed refusals, and half color outputs stay valid."""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.stage_refusals import (
    RayStageInterfaceUnsupportedError,
    StageUnsupportedError,
)

RAY_MISS_WITH_GLOBAL_PAYLOAD = """
shader main {
    vec3 hitValue @location(0) @rayPayloadInEXT;
    ray_miss {
        void main() {
            hitValue = vec3(0.0, 0.0, 0.2);
        }
    }
}
"""

RAY_MISS_WITHOUT_PAYLOAD = """
shader main {
    vec3 color @location(0);
    ray_miss {
        void main() {
            color = vec3(0.0);
        }
    }
}
"""

HALF_COLOR = """
shader main {
    struct VOut {
        vec4 position @gl_Position;
    }
    struct FragData {
        f16vec4 color @gl_FragColor;
    }
    fragment {
        FragData main(VOut input) {
            FragData output;
            output.color = f16vec4(1.0);
            return output;
        }
    }
}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def test_webgl_refuses_stages_beyond_vertex_and_fragment(tmp_path):
    with pytest.raises(StageUnsupportedError) as raised:
        translate(tmp_path, RAY_MISS_WITH_GLOBAL_PAYLOAD, "webgl")

    assert (
        raised.value.project_diagnostic_code
        == "project.translate.webgl-stage-unsupported"
    )
    assert "ray_miss" in str(raised.value)


def test_directx_ray_entry_without_payload_parameter_is_a_typed_refusal(tmp_path):
    # A global rayPayloadInEXT becomes the entry's payload parameter (semantics/hlsl_ray_interfaces.py); a miss
    # shader with no payload at all has nothing DXR can pass.
    with pytest.raises(RayStageInterfaceUnsupportedError) as raised:
        translate(tmp_path, RAY_MISS_WITHOUT_PAYLOAD, "directx")

    assert (
        raised.value.project_diagnostic_code
        == "project.translate.directx-ray-stage-interface-unsupported"
    )
    assert "requires a payload parameter" in str(raised.value)


@pytest.mark.parametrize(
    ("backend", "declaration"),
    [
        ("directx", "float16_t4 color : SV_TARGET"),
        ("metal", "half4 color [[color(0)]]"),
    ],
)
def test_half_precision_color_outputs_are_render_targets(
    tmp_path, backend, declaration
):
    assert declaration in translate(tmp_path, HALF_COLOR, backend)
