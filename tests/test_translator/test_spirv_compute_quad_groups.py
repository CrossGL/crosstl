"""SPIR-V output forms the quads of compute quad operations as HLSL does (#4840).

HLSL Shader Model 6.6 forms the quads of a compute, mesh or amplification shader from 2x2 blocks of threads when
the thread group's x and y dimensions are even, and otherwise from four consecutive ``SV_GroupIndex`` values. In a
Vulkan compute-like entry the quads of ``OpGroupNonUniformQuadSwap``, ``OpGroupNonUniformQuadBroadcast`` and the
SPV_KHR_quad_control votes follow the entry's derivative group (``DerivativeGroupQuadsKHR`` or
``DerivativeGroupLinearKHR``), and without one they are four consecutive subgroup invocations. DXC declares the mode
for an entry that uses a quad operation. CrossTL declared it only for derivatives, so
``QuadReadAcrossY(fin[gi])`` in an 8x8 group read ``gi ^ 2`` on lavapipe instead of DXC's ``gi ^ 8``.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

REQUIRE_ENV = "CROSSTL_REQUIRE_COMPUTE_QUAD_EXECUTION"

SWAP = """RWStructuredBuffer<int> fin : register(u0);
RWStructuredBuffer<int> outp : register(u1);
[numthreads({size})]
void main(uint gi : SV_GroupIndex) {{
    outp[gi] = QuadReadAcrossY(fin[gi]);
}}
"""

VOTES = """RWStructuredBuffer<int> fin : register(u0);
RWStructuredBuffer<int> outp : register(u1);
[numthreads(8, 8, 1)]
void main(uint gi : SV_GroupIndex) {
    bool a = fin[gi] > 0;
    outp[gi] = (QuadAll(a) ? 2 : 0) + (QuadAny(a) ? 1 : 0);
}
"""

HELPER = """shader QuadHelper {
    uint across(uint value) {
        return QuadReadAcrossDiagonal(value);
    }

    compute {
        layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
        buffer uint* outp;

        void main() {
            uint i = gl_LocalInvocationIndex;
            outp[i] = across(i);
        }
    }
}
"""

TWO_STAGES = """shader QuadStages {
    fragment {
        vec4 main(vec4 color @ COLOR) @ gl_FragColor {
            return vec4(QuadReadAcrossX(color.x), color.yzw);
        }
    }

    compute {
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        buffer uint* outp;

        void main() {
            outp[gl_LocalInvocationIndex] = gl_LocalInvocationIndex;
        }
    }
}
"""

VECTOR_VOTE = """shader QuadVectorVote {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer int* fin;
        buffer int* outp;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            bvec2 a = bvec2(fin[i] > 0, fin[i] > 1);
            outp[i] = QuadAny(a) ? 1 : 0;
        }
    }
}
"""

SPECIALIZED = """shader QuadSpecialized {
    compute {
        layout(local_size_x = 8, local_size_x_id = 3, local_size_y = 8, local_size_z = 1) in;
        buffer uint* outp;

        void main() {
            uint i = gl_LocalInvocationIndex;
            outp[i] = QuadReadAcrossX(i);
        }
    }
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _assemble(tmp_path, assembly):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "probe.spvasm"
    binary = tmp_path / "probe.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        [assembler, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        [validator, "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout
    return binary


def _modes(assembly):
    return re.findall(r"OpExecutionMode %\w+ (DerivativeGroup\w+)", assembly)


@pytest.mark.parametrize(
    ("size", "mode"),
    [
        ("8, 8, 1", "DerivativeGroupQuadsKHR"),
        ("2, 2, 1", "DerivativeGroupQuadsKHR"),
        ("64, 1, 1", "DerivativeGroupLinearKHR"),
        ("4, 1, 2", "DerivativeGroupLinearKHR"),
        ("1, 1, 1", None),
        ("6, 1, 1", None),
    ],
)
def test_compute_quad_operations_declare_the_hlsl_quad_layout(tmp_path, size, mode):
    assembly = _translate(tmp_path, SWAP.format(size=size), ".hlsl")

    x, y, z = (part.strip() for part in size.split(","))
    assert re.search(rf"OpExecutionMode %\w+ LocalSize {x} {y} {z}\n", assembly)
    assert "OpGroupNonUniformQuadSwap" in assembly
    if mode is None:
        # No HLSL quad layout exists for the size, so the quads stay Vulkan's subgroup quads.
        assert _modes(assembly) == []
        assert "SPV_KHR_compute_shader_derivatives" not in assembly
    else:
        assert _modes(assembly) == [mode]
        assert f"OpCapability Compute{mode}\n" in assembly
        assert 'OpExtension "SPV_KHR_compute_shader_derivatives"' in assembly
    _assemble(tmp_path, assembly)


def test_quad_operation_in_a_called_function_declares_the_entry_group(tmp_path):
    assembly = _translate(tmp_path, HELPER, ".cgl")

    assert "OpFunctionCall" in assembly
    assert _modes(assembly) == ["DerivativeGroupQuadsKHR"]
    _assemble(tmp_path, assembly)


def test_compute_entry_without_quad_operations_declares_no_group(tmp_path):
    assembly = _translate(tmp_path, TWO_STAGES, ".cgl")

    assert "OpGroupNonUniformQuadSwap" in assembly
    assert _modes(assembly) == []
    assert "ComputeDerivativeGroup" not in assembly
    _assemble(tmp_path, assembly)


def test_quad_votes_lower_to_quad_control(tmp_path):
    assembly = _translate(tmp_path, VOTES, ".hlsl")

    assert re.search(r"= OpGroupNonUniformQuadAllKHR %\w+ %\w+\n", assembly)
    assert re.search(r"= OpGroupNonUniformQuadAnyKHR %\w+ %\w+\n", assembly)
    assert "OpCapability QuadControlKHR\n" in assembly
    assert 'OpExtension "SPV_KHR_quad_control"' in assembly
    assert _modes(assembly) == ["DerivativeGroupQuadsKHR"]
    _assemble(tmp_path, assembly)


def test_vector_quad_vote_is_refused(tmp_path):
    with pytest.raises(
        SpirvLoweringRefused, match="QuadAny requires a scalar bool argument"
    ):
        _translate(tmp_path, VECTOR_VOTE, ".cgl")


def test_quad_operation_in_a_specialized_workgroup_is_refused(tmp_path):
    with pytest.raises(
        SpirvLoweringRefused,
        match="quad operations in compute, mesh and task entries need even workgroup",
    ):
        _translate(tmp_path, SPECIALIZED, ".cgl")


def _lavapipe():
    """The host lavapipe runtime's environment with ``vkrun``, or ``None``.

    MoltenVK and KosmicKrisp ignore the derivative group of a compute entry (DXC's own module reads ``gi ^ 2`` on
    them), so only lavapipe can check the quads.
    """
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    try:
        provisioned = provision("vulkan-lavapipe")
    except Exception:
        return None
    if not (provisioned.available and provisioned.mode == "host"):
        return None
    env = dict(provisioned.env)
    return env if env.get("CROSSTL_CONFORMANCE_VKRUN") else None


def _quad(gi):
    """The ``SV_GroupIndex`` values of the 2x2 quad of thread ``gi`` of an 8x8 group, in quad lane order."""
    x, y = (gi % 8) & ~1, (gi // 8) & ~1
    return [(y + dy) * 8 + x + dx for dy in (0, 1) for dx in (0, 1)]


def _run(tmp_path, env, assembly, values):
    binary = _assemble(tmp_path, assembly)
    entry = re.search(r'OpEntryPoint GLCompute %\w+ "(\w+)"', assembly).group(1)
    source = tmp_path / "fin.bin"
    result = tmp_path / "outp.bin"
    source.write_bytes(struct.pack("<64i", *values))
    result.write_bytes(struct.pack("<64i", *([-7] * 64)))
    command = [env["CROSSTL_CONFORMANCE_VKRUN"], str(binary), entry, "1", "1", "1"]
    command += [f"0:inout:{source}:256", f"1:inout:{result}:256"]
    run = subprocess.run(
        command, capture_output=True, text=True, timeout=120, env={**os.environ, **env}
    )
    assert run.returncode == 0 and run.stdout.startswith("ok"), run.stdout + run.stderr
    return list(struct.unpack("<64i", result.read_bytes()))


@pytest.fixture
def lavapipe():
    env = _lavapipe()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{REQUIRE_ENV}=1 but no host lavapipe runtime with vkrun")
        pytest.skip(f"no host lavapipe runtime; set {REQUIRE_ENV}=1 to require one")
    return env


def test_compute_quad_swap_reads_the_2x2_quad_on_lavapipe(tmp_path, lavapipe):
    assembly = _translate(tmp_path, SWAP.format(size="8, 8, 1"), ".hlsl")

    values = _run(tmp_path, lavapipe, assembly, list(range(64)))

    # QuadReadAcrossY reads quad lane ^ 2: thread (x, y) reads (x, y ^ 1).
    assert values == [_quad(gi)[_quad(gi).index(gi) ^ 2] for gi in range(64)]


def test_compute_quad_votes_read_the_2x2_quad_on_lavapipe(tmp_path, lavapipe):
    assembly = _translate(tmp_path, VOTES, ".hlsl")
    inputs = [1 if gi in {0, 4, 5, 12, 13} else 0 for gi in range(64)]

    values = _run(tmp_path, lavapipe, assembly, inputs)

    expected = []
    for gi in range(64):
        votes = [inputs[lane] > 0 for lane in _quad(gi)]
        expected.append((2 if all(votes) else 0) + (1 if any(votes) else 0))
    assert values == expected
