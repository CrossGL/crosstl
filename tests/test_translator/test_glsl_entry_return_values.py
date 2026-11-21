"""GLSL output refuses a stage entry return value that no GLSL output receives.

A GLSL entry is ``void main()``: a value the CrossGL entry returns reaches the pipeline only through the stage
outputs its return type maps to. Where none does, the generator wrote the value as an expression statement
before ``return;``, so the result was evaluated and discarded. A ray intersection entry returning
``accept_intersection`` and ``distance`` then never reported a hit, and glslang accepted the shader.

A vertex entry that returns a call to a helper declared in its stage block keeps its outputs, and the output
structure the helper returns is declared: the check for structure uses outside the entries read only the
stage's ``functions``, while the parser keeps a stage block's helpers in ``local_functions``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import OpenGLRefusal

INTERSECTION_RESULT = """
shader Probe {
    struct BoxResult {
        bool accept @ accept_intersection;
        float distance @ distance;
    }
    ray_intersection {
        BoxResult main() @bounding_box {
            BoxResult r;
            r.accept = true;
            r.distance = 1.0;
            return r;
        }
    }
}
"""

INTERSECTION_HELPER_RESULT = """
shader Probe {
    struct BoxResult {
        bool accept @ accept_intersection;
        float distance @ distance;
    }
    ray_intersection {
        BoxResult make(float t) {
            BoxResult r;
            r.accept = t > 0.0;
            r.distance = t;
            return r;
        }
        BoxResult main() @bounding_box {
            return make(2.0);
        }
    }
}
"""

CLOSEST_HIT_VALUE = """
shader Probe {
    ray_closest_hit {
        float main() {
            return 1.0;
        }
    }
}
"""

COMPUTE_VALUE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        float main() {
            data[0] = 2.0;
            return data[0];
        }
    }
}
"""

# Returns the stage outputs do receive, and a void entry returning a void call, keep their GLSL.
VERTEX_HELPER_RESULT = """
shader Probe {
    struct VSOut {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    }
    vertex {
        VSOut make(vec2 p) {
            VSOut o;
            o.position = vec4(p, 0.0, 1.0);
            o.uv = p;
            return o;
        }
        VSOut main(vec2 p @ POSITION) {
            return make(p);
        }
    }
}
"""

FRAGMENT_VALUE = """
shader Probe {
    fragment {
        vec4 main(vec4 color @ COLOR) @ gl_FragColor {
            return color * 2.0;
        }
    }
}
"""

COMPUTE_VOID_CALL = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        void store(float v) {
            data[0] = v;
        }
        void main() {
            return store(3.0);
        }
    }
}
"""


def _translate(tmp_path, source, backend="opengl"):
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


@pytest.mark.parametrize(
    ("source", "stage", "return_type"),
    [
        (INTERSECTION_RESULT, "ray_intersection", "BoxResult"),
        (INTERSECTION_HELPER_RESULT, "ray_intersection", "BoxResult"),
        (CLOSEST_HIT_VALUE, "ray_closest_hit", "float"),
        (COMPUTE_VALUE, "compute", "float"),
    ],
    ids=["intersection-local", "intersection-call", "closest-hit", "compute"],
)
def test_entry_return_value_without_output_is_refused(
    tmp_path, source, stage, return_type
):
    with pytest.raises(OpenGLRefusal) as refused:
        _translate(tmp_path, source)

    message = str(refused.value)
    assert f"OpenGL {stage} entry point returns '{return_type}'" in message
    assert "no stage output that receives the value" in message


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_vertex_entry_returning_a_helper_call_keeps_its_outputs(tmp_path, backend):
    glsl = _translate(tmp_path, VERTEX_HELPER_RESULT, backend)

    assert "gl_Position = " in glsl
    assert all(line.strip() != "make(p);" for line in glsl.splitlines()), glsl
    _glslang(tmp_path, glsl, "vert")


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_fragment_entry_return_value_reaches_its_output(tmp_path, backend):
    glsl = _translate(tmp_path, FRAGMENT_VALUE, backend)

    assert " = (color * 2.0);" in glsl or " = color * 2.0;" in glsl
    _glslang(tmp_path, glsl, "frag")


def test_void_entry_returning_a_void_call_keeps_the_call(tmp_path):
    glsl = _translate(tmp_path, COMPUTE_VOID_CALL)

    assert "store(3.0);\n" in glsl
    _glslang(tmp_path, glsl, "comp")
