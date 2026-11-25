"""GLSL subgroup built-ins are refused when their stage rules do not permit them."""

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.builtin_stage_functions import (
    refuse_misplaced_builtin_functions,
)
from crosstl.translator.semantics.builtin_values import BuiltinValueUnsupportedError

FRAGMENT = """shader Probe {
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            subgroupMemoryBarrierShared();
            return vec4(pos.x);
        }
    }
}
"""

FRAGMENT_HELPER = """shader Probe {
    void sharedBarrier() {
        subgroupMemoryBarrierShared();
    }
    fragment {
        vec4 main() @ gl_FragColor {
            sharedBarrier();
            return vec4(1.0);
        }
    }
}
"""

COMPUTE = """shader Probe {
    compute {
        void main() {
            subgroupMemoryBarrierShared();
        }
    }
}
"""

MESH_GLSL = """#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
void main() {
    subgroupMemoryBarrierShared();
    SetMeshOutputsEXT(3, 1);
}
"""

TASK_GLSL = """#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 1) in;
void main() {
    subgroupMemoryBarrierShared();
    EmitMeshTasksEXT(1, 1, 1);
}
"""


def _translate(tmp_path, source, suffix=".cgl", backend="opengl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_shared_subgroup_barrier_is_refused_in_fragment_stage(tmp_path):
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, FRAGMENT)

    message = str(raised.value)
    assert "subgroupMemoryBarrierShared" in message
    assert "fragment stage" in message
    assert "compute, mesh, task stages" in message


def test_shared_subgroup_barrier_is_checked_in_reachable_helpers(tmp_path):
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, FRAGMENT_HELPER)

    assert (
        "subgroupMemoryBarrierShared in 'sharedBarrier' of the fragment stage" in str(
            raised.value
        )
    )


VERTEX_HELPER = """shader Probe {
    vertex {
        void subgroupMemoryBarrierShared() { }
        vec4 main(vec4 p @ POSITION) @ gl_Position {
            subgroupMemoryBarrierShared();
            return p;
        }
    }
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            subgroupMemoryBarrierShared();
            return vec4(pos.x);
        }
    }
}
"""


def test_helper_declared_in_another_stage_does_not_hide_the_builtin(tmp_path):
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, VERTEX_HELPER)

    assert "in 'main' of the fragment stage" in str(raised.value)


def test_helper_declared_in_the_calling_stage_is_called(tmp_path):
    source = FRAGMENT.replace(
        "    fragment {\n",
        "    fragment {\n        void subgroupMemoryBarrierShared() { }\n",
    )
    output = _translate(tmp_path, source)
    assert "subgroupMemoryBarrierShared_();" in output


def test_shared_subgroup_barrier_is_allowed_in_compute_stage(tmp_path):
    output = _translate(tmp_path, COMPUTE)
    assert "subgroupMemoryBarrierShared();" in output


@pytest.mark.parametrize(
    "source, suffix",
    [(MESH_GLSL, ".mesh"), (TASK_GLSL, ".task")],
    ids=["mesh", "task"],
)
def test_shared_subgroup_barrier_is_allowed_in_mesh_and_task_stages(
    tmp_path, source, suffix
):
    output = _translate(tmp_path, source, suffix=suffix)
    assert "subgroupMemoryBarrierShared();" in output


@pytest.mark.parametrize(
    "name",
    [
        "subgroupBarrier",
        "subgroupMemoryBarrier",
        "subgroupMemoryBarrierBuffer",
        "subgroupMemoryBarrierImage",
    ],
)
def test_other_subgroup_barriers_remain_available_in_fragment(tmp_path, name):
    source = FRAGMENT.replace("subgroupMemoryBarrierShared", name)
    output = _translate(tmp_path, source)
    assert f"{name}();" in output


def test_stage_restriction_does_not_apply_to_non_glsl_targets():
    program = Parser(Lexer(FRAGMENT).get_tokens()).parse()
    assert refuse_misplaced_builtin_functions(program, "metal") == 0
