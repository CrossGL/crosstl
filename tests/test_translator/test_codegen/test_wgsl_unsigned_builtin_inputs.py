"""WGSL output of the ``u32`` scalar built-in inputs.

WGSL declares ``vertex_index``, ``instance_index``, ``sample_index`` and the subgroup inputs ``subgroup_size``,
``subgroup_invocation_id``, ``num_subgroups`` and ``subgroup_id`` as ``u32`` (WGSL section 13.3.1.1). An ``int``
stage-input member bound to one of them was declared ``i32``, which naga rejects, and ``gl_SubgroupSize`` and its
siblings were printed verbatim, which WGSL does not define.
"""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.codegen.wgsl_stage_io import WGSLStageIOUnsupportedError
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.refusals import WGSLRefusal


def generate_wgsl(source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return WGSLCodeGen().generate(ast)


def validate_with_naga(tmp_path, code):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "shader.wgsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


VERTEX = """
shader Probe {
    struct VertexInput {
        vec4 position;
        MEMBER
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            BODY
            return output;
        }
    }
}
"""


@pytest.mark.parametrize(
    "semantic, builtin",
    [("gl_VertexID", "vertex_index"), ("gl_InstanceID", "instance_index")],
)
def test_int_vertex_member_is_u32_and_read_as_i32(tmp_path, semantic, builtin):
    code = generate_wgsl(
        VERTEX.replace("MEMBER", f"int id @ {semantic};").replace(
            "BODY",
            "output.position = input.position + vec4(float(input.id - 1), 0.0, 0.0, 0.0);",
        )
    )
    assert f"    @builtin({builtin}) id: u32,\n" in code
    assert "f32((i32(input.id) - 1))" in code
    validate_with_naga(tmp_path, code)


def test_uint_vertex_member_is_read_unchanged(tmp_path):
    code = generate_wgsl(
        VERTEX.replace("MEMBER", "uint id @ gl_VertexID;").replace(
            "BODY",
            "output.position = input.position + vec4(float(input.id), 0.0, 0.0, 0.0);",
        )
    )
    assert "    @builtin(vertex_index) id: u32,\n" in code
    assert "f32(input.id)" in code
    validate_with_naga(tmp_path, code)


def test_write_to_int_vertex_member_is_refused():
    with pytest.raises(
        WGSLStageIOUnsupportedError,
        match="cannot write member VertexInput.id, the vertex_index input",
    ):
        generate_wgsl(
            VERTEX.replace("MEMBER", "int id @ gl_VertexID;").replace(
                "BODY", "input.id = 2; output.position = input.position;"
            )
        )


def test_float_vertex_member_is_refused():
    with pytest.raises(
        WGSLStageIOUnsupportedError,
        match="VertexInput.id of type f32 to vertex_index",
    ):
        generate_wgsl(
            VERTEX.replace("MEMBER", "float id @ gl_VertexID;").replace(
                "BODY", "output.position = input.position;"
            )
        )


def test_int_sample_member_is_u32_and_read_as_i32(tmp_path):
    code = generate_wgsl("""
shader Probe {
    struct FragmentInput {
        vec2 uv;
        int sid @ gl_SampleID;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, float(input.sid), 1.0);
            return output;
        }
    }
}
""")
    assert "    @builtin(sample_index) sid: u32,\n" in code
    assert "f32(i32(input.sid))" in code
    validate_with_naga(tmp_path, code)


COMPUTE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = float(VALUE);
        }
    }
}
"""


@pytest.mark.parametrize(
    "identifier, builtin",
    [
        ("gl_SubgroupSize", "subgroup_size"),
        ("gl_SubgroupInvocationID", "subgroup_invocation_id"),
        ("gl_NumSubgroups", "num_subgroups"),
        ("gl_SubgroupID", "subgroup_id"),
    ],
)
def test_subgroup_identifier_is_a_compute_builtin_input(tmp_path, identifier, builtin):
    code = generate_wgsl(COMPUTE.replace("VALUE", identifier))
    assert f"@builtin({builtin}) {builtin}: u32" in code
    assert f"outp[i] = f32({builtin});" in code
    assert identifier not in code
    validate_with_naga(tmp_path, code)


def test_subgroup_size_is_a_fragment_builtin_input(tmp_path):
    code = generate_wgsl("""
shader Probe {
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(float(gl_SubgroupSize), float(gl_SubgroupInvocationID), 0.0, 1.0);
        }
    }
}
""")
    assert "@builtin(subgroup_size) subgroup_size: u32" in code
    assert "@builtin(subgroup_invocation_id) subgroup_invocation_id: u32" in code
    validate_with_naga(tmp_path, code)


def test_subgroup_id_in_a_fragment_shader_is_refused():
    with pytest.raises(WGSLRefusal, match="in fragment stage: gl_SubgroupID"):
        generate_wgsl("""
shader Probe {
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(float(gl_SubgroupID), 0.0, 0.0, 1.0);
        }
    }
}
""")
