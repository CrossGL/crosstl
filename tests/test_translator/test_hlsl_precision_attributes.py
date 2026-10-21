"""GLSL precision qualifiers on structure members are not HLSL semantics.

``@mediump``/``@lowp``/``@highp`` set a minimum precision (GLSL ES 3.00 section 4.5.2) that HLSL's 32-bit types
meet. The HLSL emitter used to take the first attribute of a member as its semantic, so a precision-qualified
member was declared ``float3 dir: mediump;``, which for a stage interface breaks linkage by semantic name.
"""

import shutil
import subprocess

import crosstl

SOURCE = """
shader main {
    struct Light {
        vec3 dir @mediump;
        float power @lowp;
    }
    struct VertexInput {
        vec3 position @mediump @location(0);
    }
    struct VertexOutput {
        vec4 position @gl_Position;
        vec3 shade @highp @location(0);
    }
    cbuffer Params {
        Light light;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = vec4(input.position, 1.0);
            output.shade = light.dir * light.power;
            return output;
        }
    }
}
"""


def test_precision_attributes_are_not_hlsl_semantics(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE)

    code = crosstl.translate(str(path), backend="directx", format_output=False)

    for qualifier in ("highp", "mediump", "lowp"):
        assert qualifier not in code
    assert "float3 dir;" in code
    assert "float power;" in code
    assert "float3 position: TEXCOORD0;" in code
    assert "float3 shade: TEXCOORD0;" in code
    compiler = shutil.which("dxc")
    if compiler is not None:
        source = tmp_path / "shader.hlsl"
        source.write_text(code)
        subprocess.run(
            [compiler, "-T", "vs_6_0", "-E", "VSMain", str(source)],
            check=True,
            capture_output=True,
        )
