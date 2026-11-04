"""``gl_`` built-in variables a function body names directly are refused in stages GLSL does not declare them in."""

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.builtin_stage_identifiers import (
    refuse_misplaced_builtin_identifiers,
)
from crosstl.translator.semantics.builtin_values import BuiltinValueUnsupportedError

GRAPHICS_TARGETS = ["vulkan", "metal", "directx", "opengl", "webgl", "slang"]

FRAGMENT_READ = """shader Probe {
    fragment {
        vec4 main(vec4 color @ COLOR) @ gl_FragColor {
            return color + vec4(float(BUILTIN));
        }
    }
}
"""

FRAGMENT_LOCAL_HELPER = """shader Probe {
    fragment {
        float helper() {
            return float(gl_InstanceID);
        }
        vec4 main(vec4 color @ COLOR) @ gl_FragColor {
            return color * helper();
        }
    }
}
"""

SHARED_HELPER = """shader Probe {
    float vertexIndex() {
        return float(gl_VertexID);
    }
    vertex {
        vec4 main(vec3 position @ POSITION) @ gl_Position {
            return vec4(position, vertexIndex());
        }
    }
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(CALL);
        }
    }
}
"""

VERTEX_READ = """shader Probe {
    vertex {
        vec4 main(vec3 position @ POSITION) @ gl_Position {
            return vec4(position, float(gl_VertexID + gl_InstanceID));
        }
    }
}
"""

FRAGMENT_COORD = """shader Probe {
    fragment {
        vec4 main(vec4 color @ COLOR) @ gl_FragColor {
            return color * gl_FragCoord.x;
        }
    }
}
"""

COMPUTE_READ = """shader Probe {
    compute {
        buffer float data[];
        void main() {
            uint index = gl_GlobalInvocationID.x;
            data[index] = BUILTIN;
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


@pytest.mark.parametrize("backend", GRAPHICS_TARGETS)
@pytest.mark.parametrize("builtin", ["gl_VertexID", "gl_DrawID", "gl_BaseInstance"])
def test_vertex_builtin_read_in_fragment_is_refused(tmp_path, backend, builtin):
    source = FRAGMENT_READ.replace("BUILTIN", builtin)
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, source, backend)
    message = str(raised.value)
    assert f"{builtin} in 'main' of the fragment stage" in message
    assert "as an input of" in message and "vertex" in message
    assert "GLSL 4.60 section 7.1" in message


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx"])
def test_stage_local_helper_is_checked(tmp_path, backend):
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, FRAGMENT_LOCAL_HELPER, backend)
    assert "gl_InstanceID in 'helper' of the fragment stage" in str(raised.value)


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan"])
def test_program_helper_reached_from_fragment_is_refused(tmp_path, backend):
    source = SHARED_HELPER.replace("CALL", "vertexIndex()")
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, source, backend)
    assert "gl_VertexID in 'vertexIndex' of the fragment stage" in str(raised.value)


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan"])
def test_program_helper_reached_only_from_vertex_is_accepted(tmp_path, backend):
    assert _translate(tmp_path, SHARED_HELPER.replace("CALL", "1.0"), backend)


@pytest.mark.parametrize("backend", GRAPHICS_TARGETS)
def test_builtins_of_their_own_stage_are_accepted(tmp_path, backend):
    assert _translate(tmp_path, VERTEX_READ, backend)
    assert _translate(tmp_path, FRAGMENT_COORD, backend)


@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal", "directx"])
def test_fragment_builtin_read_in_compute_is_refused(tmp_path, backend):
    source = COMPUTE_READ.replace("BUILTIN", "gl_FragCoord.x")
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, source, backend)
    assert "gl_FragCoord in 'main' of the compute stage" in str(raised.value)


def test_compute_builtin_read_in_compute_is_accepted(tmp_path):
    source = COMPUTE_READ.replace("BUILTIN", "float(gl_LocalInvocationIndex)")
    assert _translate(tmp_path, source, "vulkan")


def test_checker_counts_builtins_and_ignores_user_names():
    program = _program(VERTEX_READ)
    assert refuse_misplaced_builtin_identifiers(program, "opengl") == 2
    plain = _program(
        FRAGMENT_READ.replace("float(BUILTIN)", "1.0").replace("@ gl_FragColor", "")
    )
    assert refuse_misplaced_builtin_identifiers(plain, "opengl") == 0
