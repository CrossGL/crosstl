"""GLSL output without a ``#version`` line of its own writes the lowest version that expresses the program.

The program starts from GLSL 3.30 (GLSL ES 3.00 for the gles target) and rises per construct. Every case is
compiled with glslangValidator when it is installed, at the emitted version and at the desktop version just below
it, which glslang must reject: that proves the emitted version is the lowest one.
"""

import functools
import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

DESKTOP_VERSIONS = (330, 400, 410, 420, 430, 440, 450, 460)
VERSION_LINE = re.compile(r"^#version .*$", re.M)


def generate(source, codegen=GLSLCodeGen):
    return codegen().generate(crosstl.translator.parse(source))


def glslang_accepts(tmp_path, code, stage):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    source = tmp_path / f"program.{stage}"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(source)], capture_output=True, text=True, timeout=60
    )
    return result.returncode == 0, result.stdout + result.stderr


def assert_lowest_desktop_version(tmp_path, code, stage, version):
    accepted, log = glslang_accepts(tmp_path, code, stage)
    assert accepted, log
    lower = [number for number in DESKTOP_VERSIONS if number < version]
    if lower:
        below = VERSION_LINE.sub(f"#version {lower[-1]} core", code, count=1)
        accepted, _ = glslang_accepts(tmp_path, below, stage)
        assert not accepted, f"GLSL {lower[-1]} also accepts:\n{below}"


FRAGMENT = """
shader Plain {
    fragment {
        vec4 main() @gl_FragColor { return vec4(1.0, 0.5, 0.25, 1.0); }
    }
}
"""

SAMPLE_ID = """
shader SampleIndex {
    struct FragmentInput { vec2 uv; int sid @ gl_SampleID; }
    struct FragmentOutput { vec4 color; }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, float(input.sid), 1.0);
            return output;
        }
    }
}
"""

EARLY_FRAGMENT_TESTS = """
shader EarlyTests {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    fragment {
        layout(early_fragment_tests) in;
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, 0.0, 1.0);
            return output;
        }
    }
}
"""

SAMPLER = """
shader Sampled {
    sampler2D tex;
    fragment {
        vec4 main() @gl_FragColor { return texture(tex, vec2(0.5)); }
    }
}
"""

HELPER_INVOCATION = """
shader Helper {
    struct FragmentInput { vec2 uv; bool helper @ gl_HelperInvocation; }
    struct FragmentOutput { vec4 color; }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, input.helper ? 1.0 : 0.0, 1.0);
            return output;
        }
    }
}
"""

BASE_VERTEX = """
shader BaseVertex {
    struct VertexInput { vec4 position; int base @ gl_BaseVertex; }
    struct VertexOutput { vec4 position @ gl_Position; }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position * 2.0 + vec4(float(input.base), 0.0, 0.0, 0.0);
            return output;
        }
    }
}
"""

COMPUTE = """
shader Doubler {
    buffer float* data;
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            data[i] = data[i] * 2.0;
        }
    }
}
"""


@pytest.mark.parametrize(
    "source,stage,version",
    [
        # GLSL 3.30 has fragment outputs with locations.
        (FRAGMENT, "frag", 330),
        # gl_SampleID: GLSL 4.00 (section 7.1).
        (SAMPLE_ID, "frag", 400),
        # layout(early_fragment_tests): GLSL 4.20 (section 4.4.1.3).
        (EARLY_FRAGMENT_TESTS, "frag", 420),
        # The binding CrossTL gives each resource: GLSL 4.20 (section 4.4.5).
        (SAMPLER, "frag", 420),
        # The compute stage: GLSL 4.30.
        (COMPUTE, "comp", 430),
        # gl_BaseVertexARB: GL_ARB_shader_draw_parameters, which glslang declares from GLSL 4.40.
        (BASE_VERTEX, "vert", 440),
        # gl_HelperInvocation: GLSL 4.50 (section 7.1).
        (HELPER_INVOCATION, "frag", 450),
    ],
)
def test_program_without_version_writes_lowest_version(
    tmp_path, source, stage, version
):
    code = generate(source)

    assert code.startswith(f"#version {version} core\n")
    assert_lowest_desktop_version(tmp_path, code, stage, version)


def test_uint_literals_are_explicit_below_glsl_400(tmp_path):
    # GLSL 3.30 converts no int to a uint implicitly (GLSL 4.60 section 4.1.10 lists the conversion from 4.00).
    source = """
    shader Unsigned {
        fragment {
            vec4 main() @gl_FragColor {
                uint a = 3;
                uint b = a + 1;
                return vec4(float(b));
            }
        }
    }
    """
    code = generate(source)

    assert code.startswith("#version 330 core\n")
    assert "uint a = 3u;" in code
    assert "(a + 1u)" in code
    assert_lowest_desktop_version(tmp_path, code, "frag", 330)


def test_program_version_line_is_kept_as_its_floor(tmp_path):
    # A program's own #version line is the starting point: output rises from it but never goes below it.
    code = generate("#version 450 core\n" + FRAGMENT)

    assert code.startswith("#version 450 core\n")
    accepted, log = glslang_accepts(tmp_path, code, "frag")
    assert accepted, log


def test_gles_target_starts_from_glsl_es_300(tmp_path):
    code = generate(FRAGMENT, functools.partial(GLSLCodeGen, target_profile="gles"))

    assert code.startswith("#version 300 es\n")
    accepted, log = glslang_accepts(tmp_path, code, "frag")
    assert accepted, log
