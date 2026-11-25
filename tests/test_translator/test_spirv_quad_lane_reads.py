"""SPIR-V output reads a quad lane that the program computes at run time.

``OpGroupNonUniformQuadBroadcast`` needs a constant index before SPIR-V 1.5 and a dynamically uniform one after it,
but HLSL's ``QuadReadLaneAt``, Metal's ``quad_broadcast``/``quad_shuffle`` (which Metal input imports as
``QuadReadLaneAt``) and CrossGL's ``QuadReadLaneAt`` take a lane that differs per invocation. SPIR-V output used to
refuse such a lane. It now broadcasts each of the four quad lanes with a constant index and selects the one
``lane & 3`` names, as GLSL output's ``crossglQuadReadLaneAt`` does.
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

REQUIRE_ENV = "CROSSTL_REQUIRE_QUAD_LANE_EXECUTION"

DYNAMIC_LANE = """shader QuadLane {{
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer int* fin;
        buffer int* outp;

        void main() {{
            uint gi = gl_LocalInvocationIndex;
            {body}
        }}
    }}
}}
"""

FRAGMENT_LANE = """shader QuadLaneFragment {
    struct FragmentInput {
        vec4 color @ COLOR0;
        flat uint lane @ TEXCOORD0;
    };

    fragment {
        vec4 main(FragmentInput input) @ gl_FragColor {
            return QuadReadLaneAt(input.color, input.lane);
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _lane_source(body):
    return DYNAMIC_LANE.format(body=body)


def _assemble(tmp_path, assembly, environment="vulkan1.1"):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "probe.spvasm"
    binary = tmp_path / "probe.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        [assembler, "--target-env", environment, str(source), "-o", str(binary)],
        [validator, "--target-env", environment, str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout
    return binary


def _broadcast_indices(assembly):
    """The constant indices of every ``OpGroupNonUniformQuadBroadcast``, in emission order."""
    constants = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r"%(\w+) = OpConstant %\w+ (\d+)\b", assembly)
    }
    indices = re.findall(
        r"OpGroupNonUniformQuadBroadcast %\w+ %\w+ %\w+ %(\w+)", assembly
    )
    return [constants.get(index) for index in indices]


@pytest.mark.parametrize(
    "body",
    [
        "outp[gi] = int(QuadReadLaneAt(gi, uint(fin[gi])));",
        "outp[gi] = QuadReadLaneAt(fin[gi], fin[gi]);",
        "outp[gi] = int(QuadReadLaneAt(vec2(float(gi)), uint(fin[gi]) & 3u).y);",
        "outp[gi] = QuadReadLaneAt(ivec3(fin[gi]), gl_SubgroupInvocationID ^ 1u).z;",
        "outp[gi] = int(QuadReadLaneAt(fin[gi] > 2, uint(fin[gi])));",
    ],
    ids=["uint-value", "int-lane", "float-vector", "int-vector", "bool-value"],
)
def test_a_run_time_lane_reads_each_quad_lane_and_selects_one(tmp_path, body):
    assembly = _translate(tmp_path, _lane_source(body))

    _assemble(tmp_path, assembly)
    assert _broadcast_indices(assembly) == [0, 1, 2, 3]
    assert "OpBitwiseAnd" in assembly
    assert assembly.count("OpSelect") >= 3
    assert "OpCapability GroupNonUniformQuad" in assembly


def test_a_literal_lane_keeps_one_broadcast(tmp_path):
    assembly = _translate(
        tmp_path, _lane_source("outp[gi] = QuadReadLaneAt(fin[gi], 2u);")
    )

    _assemble(tmp_path, assembly)
    assert _broadcast_indices(assembly) == [2]
    assert "OpSelect" not in assembly


def test_a_run_time_lane_validates_in_a_fragment_stage(tmp_path):
    assembly = _translate(tmp_path, FRAGMENT_LANE)

    _assemble(tmp_path, assembly)
    assert _broadcast_indices(assembly) == [0, 1, 2, 3]


def test_a_literal_lane_above_three_is_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="must be between 0 and 3"):
        _translate(tmp_path, _lane_source("outp[gi] = QuadReadLaneAt(fin[gi], 5u);"))


def test_a_vector_lane_is_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="requires a scalar lane index"):
        _translate(
            tmp_path,
            _lane_source("outp[gi] = QuadReadLaneAt(fin[gi], uvec2(1u, 2u)).x;"),
        )


def _vulkan_runtime():
    """The environment of a host Vulkan runtime with ``vkrun`` (lavapipe, else MoltenVK), or ``None``.

    Without a derivative group layout, the quads of a compute stage are four consecutive subgroup invocations on
    both runtimes.
    """
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    for runtime in ("vulkan-lavapipe", "vulkan-moltenvk"):
        try:
            provisioned = provision(runtime)
        except Exception:
            continue
        if not (provisioned.available and provisioned.mode == "host"):
            continue
        env = dict(provisioned.env)
        if env.get("CROSSTL_CONFORMANCE_VKRUN"):
            return env
    return None


@pytest.fixture
def vulkan_runtime():
    env = _vulkan_runtime()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{REQUIRE_ENV}=1 but no host Vulkan runtime with vkrun")
        pytest.skip(f"no host Vulkan runtime; set {REQUIRE_ENV}=1 to require one")
    return env


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


def test_a_run_time_lane_reads_the_lane_it_names_on_vulkan(tmp_path, vulkan_runtime):
    body = (
        "uint lane = gl_SubgroupInvocationID;\n"
        "            uint source = QuadReadLaneAt(lane, uint(fin[gi]));\n"
        "            outp[gi] = int(source) * 256 + int(lane);"
    )
    assembly = _translate(tmp_path, _lane_source(body))
    # Each invocation names its own lane, including lanes above 3 and the invocation itself.
    selectors = [(gi * 5 + gi // 4) % 7 for gi in range(64)]

    values = _run(tmp_path, vulkan_runtime, assembly, selectors)

    for gi, value in enumerate(values):
        source, lane = divmod(value, 256)
        assert source == (lane & ~3) + (selectors[gi] & 3), (gi, lane, source)
