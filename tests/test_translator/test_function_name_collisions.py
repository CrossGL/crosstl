"""A function that shares its name with a uniform block member keeps working once the member is module-scope."""

import re
import shutil
import subprocess

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.function_name_collisions import (
    rename_colliding_functions,
    rename_functions,
)
from crosstl.translator.semantics.hub_types import call_name

# GLSL reaches the member through the block instance, so it may share its name with the function.
GLSL_COLLISION = """#version 450
layout(binding = 2) uniform UBO {
    vec2 brightnessContrast;
    float gain;
} ubo;
layout(location = 0) in vec3 color;
layout(location = 0) out vec4 outColor;
vec3 brightnessContrast(vec3 c, float brightness, float contrast) {
    return (c - 0.5) * contrast + 0.5 + brightness;
}
vec3 brightnessContrast(vec3 c) { return brightnessContrast(c, 0.0, 1.0); }
void main() {
    outColor = vec4(brightnessContrast(color, ubo.brightnessContrast[0], ubo.brightnessContrast[1])
                    + brightnessContrast(color) * ubo.gain, 1.0);
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "shader.frag"
    path.write_text(GLSL_COLLISION)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_renamed(code):
    # The member keeps its name (the host binds it by name); the function and every call to it are renamed.
    assert re.search(r"\bbrightnessContrast_fn\(", code), code
    assert not re.search(r"\bbrightnessContrast\(", code), code
    assert re.search(r"\bbrightnessContrast\s*\[0\]", code), code


def test_glsl_renames_the_function_not_the_block_member(tmp_path):
    glsl = _translate(tmp_path, "opengl")

    _assert_renamed(glsl)
    assert re.search(r"vec2 brightnessContrast;", glsl)
    if shutil.which("glslangValidator"):
        _run(["glslangValidator", "--stdin", "-S", "frag"], input=glsl)


def test_glsl_es_renames_the_function_not_the_block_member(tmp_path):
    essl = _translate(tmp_path, "webgl")

    _assert_renamed(essl)
    if shutil.which("glslangValidator"):
        _run(["glslangValidator", "--stdin", "-S", "frag"], input=essl)


def test_hlsl_renames_the_function_not_the_cbuffer_member(tmp_path):
    hlsl = _translate(tmp_path, "directx")

    _assert_renamed(hlsl)
    assert re.search(r"float2 brightnessContrast;", hlsl)


def test_metal_renames_the_function(tmp_path):
    metal = _translate(tmp_path, "metal")

    assert re.search(r"\bbrightnessContrast_fn\(", metal), metal
    assert not re.search(r"\bbrightnessContrast\(", metal), metal


def test_entry_points_and_unrelated_functions_keep_their_names():
    program = Parser(Lexer("""
            shader main {
                cbuffer Params { float scale; float main_value; };
                float twice(float x) { return x * 2.0; }
                fragment {
                    vec4 main() @ gl_FragColor { return vec4(twice(scale)); }
                }
            }
            """).get_tokens()).parse()

    assert rename_colliding_functions(program) == {}


def _call_names(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen or not hasattr(node, "__dict__"):
        return []
    seen.add(id(node))
    names = [call_name(node)] if isinstance(node, A.FunctionCallNode) else []
    for value in vars(node).values():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            names.extend(_call_names(item, seen))
    return names


def test_rename_keeps_the_type_arguments_of_an_explicit_generic_call():
    program = Parser(Lexer("""
            shader main {
                generic<T> T twice(T x) { return x + x; }
                generic<T> T quad(T x) { return twice<T>(x); }
                float caller(float x) { return quad<float>(x) + quad(x); }
                compute {
                    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                    buffer float* data;
                    void main() { data[0] = caller(data[0]); }
                }
            }
            """).get_tokens()).parse()

    assert rename_functions(program, {"quad"}) == {"quad": "quad_fn"}
    names = _call_names(program)
    assert "quad_fn<float>" in names and "quad_fn" in names
    assert not any(name.partition("<")[0] == "quad" for name in names)
    assert "twice<T>" in names
