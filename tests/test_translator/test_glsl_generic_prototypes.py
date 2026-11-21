"""GLSL prototypes for explicit-type-argument calls to generics defined later (#4816).

GLSL has no forward references, so a helper that calls a function defined after it needs that function's
prototype first. ``twice(x)`` got ``float twice_float(float x);``, but ``twice<float>(x)`` got none: the parser
keeps the type arguments in the callee name, so the prototype list looked for ``twice<float>`` among the generic
definitions and skipped ``twice``. glslang then rejected the call with "no matching overloaded function found".
"""

import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """
shader main {
    vec2 caller(vec2 x, int n) {
        return scale<vec2, int>(x, n) + twice<vec2>(x) + vec2(float(twice<int>(n)));
    }
    generic<T, U> T scale(T x, U n) { return x * T(n); }
    generic<T> T twice(T x) { return x + x; }
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer vec2* inp;
        buffer vec2* out_;
        vec2 stage_helper(vec2 v) { return late<vec2>(v); }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            out_[i] = caller(inp[i], 3) + stage_helper(inp[i]);
        }
    }
    generic<T> T late(T x) { return x * x; }
}
"""

FRAGMENT = """
shader main {
    float caller(float x) { return twice<float>(x); }
    generic<T> T twice(T x) { return x + x; }
    fragment {
        vec4 main(vec4 color @ COLOR) @ gl_FragColor {
            return vec4(caller(color.x));
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["glslangValidator", "-S", stage, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_declared_before_use(glsl, prototype, caller_definition):
    assert prototype in glsl, glsl
    assert glsl.index(prototype) < glsl.index(caller_definition), glsl


def test_explicit_type_argument_calls_get_prototypes(tmp_path):
    glsl = _translate(tmp_path, COMPUTE, "opengl")

    caller = "vec2 caller(vec2 x, int n) {"
    for prototype in (
        "vec2 scale_vec2_int(vec2 x, int n);",
        "vec2 twice_vec2(vec2 x);",
        "int twice_int(int x);",
    ):
        _assert_declared_before_use(glsl, prototype, caller)
    _assert_declared_before_use(
        glsl, "vec2 late_vec2(vec2 x);", "vec2 stage_helper(vec2 v) {"
    )
    _glslang(tmp_path, glsl, "comp")


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_fragment_helper_calls_later_generic_with_type_argument(tmp_path, backend):
    glsl = _translate(tmp_path, FRAGMENT, backend)

    _assert_declared_before_use(
        glsl, "float twice_float(float x);", "float caller(float x) {"
    )
    assert "twice<" not in glsl
    _glslang(tmp_path, glsl, "frag")
