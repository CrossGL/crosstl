"""GLSL ES 3.00 spellings of desktop GLSL that WebGL 2 rejects, and GLSL globals that stay private.

ESSL 3.00 has no implicit conversions (section 4.1.10), so float built-ins called with integer arguments and
arithmetic mixing a float value with an integer literal convert explicitly; it has no scalar swizzles; and WebGL 2
exposes none of the fragment density, shading rate or ray query extensions. A GLSL global without a storage
qualifier is private to each invocation (GLSL 4.60 section 4.3), not a uniform.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic


def _translate(tmp_path, source, name, backend, **options):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, **options)


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


BUILTIN_ARGUMENTS = """#version 450
layout(location = 0) flat in uint index;
layout(location = 1) in float hue;
layout(location = 0) out vec4 color;
void main() {
    float wrapped = 1 - abs(mod(hue, 2) - 1);
    float rounded = round(index);
    vec3 specular = pow(max(hue, 0.0), 8.0).xxx;
    color = vec4(specular * wrapped, rounded);
}
"""


def test_float_builtins_convert_integer_arguments_for_webgl(tmp_path):
    code = _translate(tmp_path, BUILTIN_ARGUMENTS, "args.frag", "webgl")
    assert "#version 300 es" in code
    assert "mod(cgl_location_1, 2.0)" in code or "mod(hue, 2.0)" in code
    assert "(1.0 - abs(" in code and " - 1.0)" in code
    assert "round(float(" in code
    assert ".xxx" not in code
    _glslang(tmp_path, code, "frag")


def test_desktop_glsl_keeps_the_implicit_conversions(tmp_path):
    code = _translate(tmp_path, BUILTIN_ARGUMENTS, "args.frag", "opengl")
    assert "mod(hue, 2)" in code
    _glslang(tmp_path, code, "frag")


PRIVATE_GLOBALS = """#version 450
vec2 positions[3] = vec2[](vec2(0.0, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));
float counter;
void main() {
    counter += 1.0;
    gl_Position = vec4(positions[gl_VertexIndex], counter, 1.0);
}
"""


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_unqualified_glsl_globals_stay_private(tmp_path, backend):
    cgl = _translate(tmp_path, PRIVATE_GLOBALS, "private.vert", "cgl")
    assert "static vec2 positions[3]" in cgl
    assert "static float counter;" in cgl
    code = _translate(tmp_path, PRIVATE_GLOBALS, "private.vert", backend)
    assert "uniform" not in code
    assert "vec2 positions[3] = vec2[3](" in " ".join(code.split())
    _glslang(tmp_path, code, "vert")


@pytest.mark.parametrize(
    "declaration, use",
    [
        (
            "#extension GL_EXT_fragment_invocation_density : require",
            "float(gl_FragSizeEXT.x)",
        ),
        (
            "#extension GL_EXT_fragment_shading_rate : require",
            "float(gl_ShadingRateEXT)",
        ),
    ],
)
def test_vulkan_only_fragment_builtins_are_refused_for_webgl(
    tmp_path, declaration, use
):
    source = (
        f"#version 450\n{declaration}\nlayout(location = 0) out vec4 color;\n"
        f"void main() {{ color = vec4({use}); }}\n"
    )
    with pytest.raises(TranslationDiagnostic, match="GLSL ES 3.00 has no"):
        _translate(tmp_path, source, "density.frag", "webgl")


def test_ray_queries_are_refused_for_webgl(tmp_path):
    source = """#version 460
#extension GL_EXT_ray_query : require
layout(binding = 0) uniform accelerationStructureEXT scene;
layout(location = 0) out vec4 color;
void main() {
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, vec3(0.0), 0.0, vec3(0.0, 0.0, 1.0), 1.0);
    color = vec4(1.0);
}
"""
    with pytest.raises(TranslationDiagnostic, match="ray queries"):
        _translate(tmp_path, source, "query.frag", "webgl")
