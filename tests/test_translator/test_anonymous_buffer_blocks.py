"""CrossGL anonymous buffer blocks put their members in the enclosing scope (GLSL 4.60 section 4.3.9).

The parser names the instance and binds every free member reference to it, honouring GLSL scoping: parameters and
locals shadow a member from the end of their declarator (section 4.2.2).
"""

import shutil
import subprocess

from crosstl.translator import parse
from crosstl.translator.ast import ArrayAccessNode, MemberAccessNode
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

ANONYMOUS = """shader Anonymous {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(std430, binding = 3) buffer Data { uint count; float data[]; };
        buffer float* outp;
        float first(float count) { return count + data[0]; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float data2 = data[i];
            outp[i] = float(count) + first(data2) + float(data.length());
            for (int count = 0; count < 2; count++) { outp[i] += float(count); }
            float data = data[1];
            outp[i] += data;
        }
    }
}"""


def test_members_bind_to_a_named_instance():
    shader = parse(ANONYMOUS)

    (block,) = (
        variable
        for variable in shader.walk()
        if getattr(getattr(variable, "var_type", None), "name", None) == "Data"
    )
    # The instance may not hide either member.
    assert block.name == "data_"
    reads = [
        node.array_expr
        for node in shader.walk()
        if isinstance(node, ArrayAccessNode)
        and isinstance(node.array_expr, MemberAccessNode)
    ]
    assert {(access.object_expr.name, access.member) for access in reads} == {
        ("data_", "data")
    }


def test_shadowing_follows_glsl_scopes(tmp_path):
    glsl = GLSLCodeGen().generate(parse(ANONYMOUS))

    assert "return (count + data_.data[0]);" in glsl
    assert "float data2 = data_.data[i];" in glsl
    assert "float(data_.count)" in glsl
    assert "float(data_.data.length())" in glsl
    assert "outp[i] += float(count);" in glsl
    # The initializer reads the member; the declared local shadows it afterwards.
    assert "float data = data_.data[1];" in glsl
    assert "outp[i] += data;" in glsl
    if shutil.which("glslangValidator"):
        (tmp_path / "anon.comp").write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            ["glslangValidator", "-S", "comp", "anon.comp"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout


def test_hlsl_loads_bound_members(tmp_path):
    hlsl = HLSLCodeGen().generate(parse(ANONYMOUS))

    assert "data_.Load(0)" in hlsl
    dxc = shutil.which("dxc")
    if dxc:
        (tmp_path / "anon.hlsl").write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", "anon.hlsl", "-Fo", "anon.o"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout + result.stderr
