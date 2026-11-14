"""GLSL output requires the extensions of the extension built-ins it names.

HLSL's ``SV_Barycentrics`` and SPIR-V's ``BaryCoordKHR`` become ``gl_BaryCoordEXT``, which GLSL declares only under
``GL_EXT_fragment_shader_barycentric``; SPIR-V's ``ShadingRateKHR`` becomes ``gl_ShadingRateEXT`` under
``GL_EXT_fragment_shading_rate``. Without the directive glslang rejects the program ("required extension not
requested").
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_builtin_extensions import (
    require_builtin_extensions,
)

GLSLANG = shutil.which("glslangValidator")

BARYCENTRIC_HLSL = """
float4 main(float3 bary : SV_Barycentrics) : SV_TARGET { return float4(bary, 1.0); }
"""


def test_hlsl_barycentrics_require_the_glsl_extension(tmp_path):
    path = tmp_path / "shader.frag.hlsl"
    path.write_text(BARYCENTRIC_HLSL)

    glsl = crosstl.translate(str(path), backend="opengl")

    assert "gl_BaryCoordEXT" in glsl
    lines = glsl.splitlines()
    assert lines[1] == "#extension GL_EXT_fragment_shader_barycentric : require"
    if GLSLANG is not None:
        out = tmp_path / "out.frag"
        out.write_text(glsl)
        result = subprocess.run([GLSLANG, str(out)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_each_extension_is_required_once_after_the_version():
    code = (
        "#version 450 core\n"
        "void main() {\n"
        "    int rate = gl_ShadingRateEXT & gl_ShadingRateFlag2VerticalPixelsEXT;\n"
        "    vec3 b = gl_BaryCoordEXT + gl_BaryCoordNoPerspEXT;\n"
        "}\n"
    )

    lines = require_builtin_extensions(code).splitlines()

    assert lines[:3] == [
        "#version 450 core",
        "#extension GL_EXT_fragment_shading_rate : require",
        "#extension GL_EXT_fragment_shader_barycentric : require",
    ]


@pytest.mark.parametrize(
    "code",
    [
        "#version 450\n#extension GL_EXT_fragment_shader_barycentric : enable\nvec3 b = gl_BaryCoordEXT;\n",
        "#version 450\n// gl_BaryCoordEXT in a comment\nvoid main() {}\n",
        '#version 450\nvoid main() { debugPrintfEXT("gl_BaryCoordEXT"); }\n',
    ],
    ids=["declared", "comment", "string"],
)
def test_declared_or_unused_builtins_add_nothing(code):
    assert require_builtin_extensions(code) == code
