"""SPIR-V output writes the quad votes QuadAll and QuadAny with SPV_KHR_quad_control.

CrossGL's ``QuadAll`` and ``QuadAny`` (HLSL SM 6.7, MSL ``quad_all`` and ``quad_any``, GL_EXT_shader_quad_control's
``subgroupQuadAll`` and ``subgroupQuadAny``) lower to ``OpGroupNonUniformQuadAllKHR`` and
``OpGroupNonUniformQuadAnyKHR``, which take no execution scope, with the ``QuadControlKHR`` capability and
``GroupNonUniformVote``, as glslang declares them (``spirv_wave_extended.py``).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

CROSSGL = """shader main {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            bool p = fin[i] > 0.0;
            outp[i] = (QuadAll(p) ? 1 : 0) + (QuadAny(p) ? 2 : 0);
        }
    }
}
"""

HLSL = """RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<int> outp : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    bool p = fin[id.x] > 0.0;
    outp[id.x] = (QuadAll(p) ? 1 : 0) + (QuadAny(p) ? 2 : 0);
}
"""


def translate(tmp_path, code, suffix):
    path = tmp_path / f"quad{suffix}"
    path.write_text(code, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan")


def validate(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools is not installed")
    source, binary = tmp_path / "quad.spvasm", tmp_path / "quad.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("code, suffix", [(CROSSGL, ".cgl"), (HLSL, ".hlsl")])
def test_quad_votes_lower_to_quad_control_instructions(tmp_path, code, suffix):
    assembly = translate(tmp_path, code, suffix)
    assert "OpCapability QuadControlKHR" in assembly
    assert "OpCapability GroupNonUniformVote" in assembly
    assert 'OpExtension "SPV_KHR_quad_control"' in assembly
    lines = [line.split("=", 1)[-1].split() for line in assembly.splitlines()]
    votes = [
        line for line in lines if line and line[0].startswith("OpGroupNonUniformQuadA")
    ]
    # Result type and predicate only: the quad votes have no execution scope operand.
    assert sorted(line[0] for line in votes) == [
        "OpGroupNonUniformQuadAllKHR",
        "OpGroupNonUniformQuadAnyKHR",
    ]
    assert all(len(line) == 3 for line in votes), votes
    validate(tmp_path, assembly)


def test_quad_vote_of_a_float_is_refused(tmp_path):
    code = CROSSGL.replace("QuadAll(p)", "QuadAll(fin[i])")
    with pytest.raises(SpirvLoweringRefused, match="QuadAll requires one scalar bool"):
        translate(tmp_path, code, ".cgl")
