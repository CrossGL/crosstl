"""Metal gives each resource without a binding the lowest free slots of its binding space, in declaration order.

MSL binds every buffer argument of an entry, constant or device, through one ``[[buffer(n)]]`` index space, and
textures and samplers through their own (MSL 4.1 section 5.2.1). An explicit binding elsewhere in the program does
not move the unbound resources past it: a kernel whose constants block is bound at buffer 3 takes its unbound storage
buffers at buffers 0 and 1, where a host binds them for every other target. The generator used to start the unbound
resources one past the highest explicit binding, so that kernel read its storage buffers at 4 and 5. A structured
buffer's length and counter companions still take the first free slots after their buffer.
"""

from __future__ import annotations

import re
import textwrap

import crosstl
from tests.metal_compute_dispatch import compile_metal_library

REQUIRE_ENV = "CROSSTL_REQUIRE_METAL_UNBOUND_RESOURCE_SLOTS"


def _metal(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _slots(metal, kind):
    """``{name: index}`` for each entry parameter bound with ``[[kind(index)]]``."""
    return {
        name: int(index)
        for name, index in re.findall(
            rf"(\w+)\s*\[\[\s*{kind}\s*\(\s*(\d+)\s*\)\s*\]\]", metal
        )
    }


def test_unbound_buffers_fill_the_slots_below_an_explicit_constants_block(tmp_path):
    metal = _metal(
        tmp_path,
        """
        shader Probe {
            cbuffer Params @binding(3) {
                float a;
            };
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* fin;
                buffer float* outp;
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    outp[i] = fin[i] * a;
                }
            }
        }
        """,
    )

    assert _slots(metal, "buffer") == {"params": 3, "fin": 0, "outp": 1}
    compile_metal_library(metal, tmp_path, REQUIRE_ENV)


def test_unbound_resources_skip_every_explicit_slot(tmp_path):
    metal = _metal(
        tmp_path,
        """
        shader Probe {
            cbuffer First @binding(1) {
                float a;
            };
            cbuffer Second {
                float b;
            };
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* fin @binding(0);
                buffer float* outp;
                buffer float* extra @binding(3);
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    outp[i] = fin[i] * a + b + extra[i];
                }
            }
        }
        """,
    )

    assert _slots(metal, "buffer") == {
        "first": 1,
        "second": 2,
        "fin": 0,
        "outp": 4,
        "extra": 3,
    }
    compile_metal_library(metal, tmp_path, REQUIRE_ENV)


def test_unbound_textures_take_the_lowest_free_texture_slots(tmp_path):
    metal = _metal(
        tmp_path,
        """
        shader Probe {
            sampler samp;
            sampler2D bound @texture(1);
            sampler2D first;
            sampler2D second;
            fragment {
                vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                    return texture(bound, samp, uv) + texture(first, samp, uv) + texture(second, samp, uv);
                }
            }
        }
        """,
    )

    assert _slots(metal, "texture") == {"bound": 1, "first": 0, "second": 2}
    assert _slots(metal, "sampler") == {"samp": 0}
    compile_metal_library(metal, tmp_path, REQUIRE_ENV)


def test_structured_buffer_companions_follow_their_buffer(tmp_path):
    metal = _metal(
        tmp_path,
        """
        shader Probe {
            RWStructuredBuffer<int> values @binding(3);
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer int* outp;
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    uint len;
                    buffer_dimensions(values, len);
                    outp[i] = buffer_load(values, i) + int(len);
                }
            }
        }
        """,
    )

    assert _slots(metal, "buffer") == {"values": 3, "valuesLength": 4, "outp": 0}
    compile_metal_library(metal, tmp_path, REQUIRE_ENV)
