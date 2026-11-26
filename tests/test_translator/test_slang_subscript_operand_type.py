"""Slang output of a subscript whose operand has a wider type than the subscript.

``float d = textureLod(tex, uv, 0.0)[0];`` reads one component of a ``float4``. Slang output passed the declared
``float`` down to the subscripted call as its expected type, so the sampled-texture lowering refused the read as
"textureLod returns float4 but target expects float".
"""

import re
import shutil
import subprocess

import crosstl

SAMPLE_COMPONENT = """#version 450
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    float d = textureLod(tex, uv, 0.0)[0];
    color = vec4(d);
}
"""


def test_component_of_a_sampled_texel_is_read_from_the_vector(tmp_path):
    path = tmp_path / "shader.frag"
    path.write_text(SAMPLE_COMPONENT)
    slang = crosstl.translate(str(path), backend="slang")
    assert re.search(r"float d = \w+\.SampleLevel\([^;]*\)\[0\];", slang), slang
    slangc = shutil.which("slangc")
    if slangc is None:
        return
    # The entry carries [shader("fragment")], so slangc finds it without -entry.
    source = tmp_path / "out.slang"
    source.write_text(slang)
    result = subprocess.run(
        [
            slangc,
            str(source),
            "-target",
            "spirv-asm",
            "-o",
            str(tmp_path / "out.spvasm"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + slang
