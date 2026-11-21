"""SPIR-V output forms compute quads as the CrossGL stage declares them (#4840).

A CrossGL compute, mesh or task stage declares how its invocations form quads with
``layout(derivative_group_quadsKHR) in;`` (2x2 quads of the x and y local invocation ids) or
``layout(derivative_group_linearKHR) in;`` (four consecutive local invocation indices). Without the layout, the quads
of its quad operations are four consecutive subgroup invocations, and a stage that takes derivatives forms the group
its size allows. In SPIR-V the group is the ``DerivativeGroupQuadsKHR`` or ``DerivativeGroupLinearKHR`` execution
mode, which forms the quads of both the derivatives and the quad operations.

CrossTL ignored the layout: a stage that declared 2x2 quads read subgroup quads in ``QuadReadAcrossY``, and a stage
that declared the linear group and took derivatives in an 8x8 workgroup declared the 2x2 group instead.
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

QUAD_READ = """shader QuadRead {{
    compute {{
        layout(local_size_x = {x}, local_size_y = {y}, local_size_z = 1) in;
        {layout}
        buffer int* fin;
        buffer int* outp;

        void main() {{
            uint gi = gl_LocalInvocationIndex;
            outp[gi] = QuadReadAcrossY(fin[gi]);
        }}
    }}
}}
"""

SUBGROUP_LANES = """shader QuadLanes {
    compute {
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        buffer int* fin;
        buffer int* outp;

        void main() {
            uint gi = gl_LocalInvocationIndex;
            uint lane = gl_SubgroupInvocationID;
            outp[gi] = int(QuadReadAcrossY(lane)) * 256 + int(lane) + fin[gi] * 0;
        }
    }
}
"""

DERIVATIVES = """shader Derivatives {{
    compute {{
        layout({size}) in;
        {layout}
        buffer int* fin;
        buffer int* outp;

        void main() {{
            uint gi = gl_LocalInvocationIndex;
            float v = float(fin[gi]);
            outp[gi] = int(ddy(v));
        }}
    }}
}}
"""

QUADS = "layout(derivative_group_quadsKHR) in;"
LINEAR = "layout(derivative_group_linearKHR) in;"


def _translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
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
    ("layout", "x", "y", "mode"),
    [
        (QUADS, 8, 8, "DerivativeGroupQuadsKHR"),
        (LINEAR, 8, 8, "DerivativeGroupLinearKHR"),
        (LINEAR, 64, 1, "DerivativeGroupLinearKHR"),
        ("layout(derivative_group_quadsNV) in;", 4, 2, "DerivativeGroupQuadsKHR"),
        ("layout(derivative_group_linearNV) in;", 4, 1, "DerivativeGroupLinearKHR"),
    ],
)
def test_a_declared_group_forms_the_quads(tmp_path, layout, x, y, mode):
    assembly = _translate(tmp_path, QUAD_READ.format(x=x, y=y, layout=layout))

    assert "OpGroupNonUniformQuadSwap" in assembly
    assert _modes(assembly) == [mode]
    assert f"OpCapability Compute{mode}\n" in assembly
    assert 'OpExtension "SPV_KHR_compute_shader_derivatives"' in assembly
    _assemble(tmp_path, assembly)


@pytest.mark.parametrize(("x", "y"), [(8, 8), (64, 1), (1, 1)])
def test_quad_operations_without_a_layout_keep_subgroup_quads(tmp_path, x, y):
    assembly = _translate(tmp_path, QUAD_READ.format(x=x, y=y, layout=""))

    assert "OpGroupNonUniformQuadSwap" in assembly
    assert _modes(assembly) == []
    assert "ComputeDerivativeGroup" not in assembly
    assert "SPV_KHR_compute_shader_derivatives" not in assembly
    _assemble(tmp_path, assembly)


@pytest.mark.parametrize(
    ("size", "layout", "mode"),
    [
        ("local_size_x = 8, local_size_y = 8, local_size_z = 1", "", "Quads"),
        ("local_size_x = 64, local_size_y = 1, local_size_z = 1", "", "Linear"),
        ("local_size_x = 8, local_size_y = 8, local_size_z = 1", LINEAR, "Linear"),
        ("local_size_x = 4, local_size_y = 4, local_size_z = 2", QUADS, "Quads"),
    ],
)
def test_derivatives_take_the_declared_group_over_the_derived_one(
    tmp_path, size, layout, mode
):
    assembly = _translate(tmp_path, DERIVATIVES.format(size=size, layout=layout))

    assert "OpDPdy" in assembly
    assert _modes(assembly) == [f"DerivativeGroup{mode}KHR"]
    _assemble(tmp_path, assembly)


@pytest.mark.parametrize(
    ("layout", "x", "y", "message"),
    [
        (QUADS, 4, 1, "need even workgroup x and y dimensions; the 4x1x1"),
        (QUADS, 3, 2, "need even workgroup x and y dimensions; the 3x2x1"),
        (
            LINEAR,
            6,
            1,
            "need a multiple of four invocations; the 6x1x1 workgroup has 6",
        ),
        (QUADS + "\n" + LINEAR, 8, 8, "declares both"),
    ],
)
def test_a_group_the_workgroup_cannot_form_is_refused(tmp_path, layout, x, y, message):
    with pytest.raises(SpirvLoweringRefused, match=re.escape(message)):
        _translate(tmp_path, QUAD_READ.format(x=x, y=y, layout=layout))


def test_a_specialized_size_with_a_declared_group_keeps_the_group(tmp_path):
    size = "local_size_x = 8, local_size_x_id = 3, local_size_y = 8, local_size_z = 1"
    assembly = _translate(tmp_path, DERIVATIVES.format(size=size, layout=QUADS))

    assert re.search(r"OpDecorate %\w+ SpecId 3\n", assembly)
    assert _modes(assembly) == ["DerivativeGroupQuadsKHR"]
    _assemble(tmp_path, assembly)


def test_a_specialized_size_without_a_declared_group_cannot_take_derivatives(
    tmp_path,
):
    size = "local_size_x = 8, local_size_x_id = 3, local_size_y = 8, local_size_z = 1"
    with pytest.raises(SpirvLoweringRefused, match="does not guarantee"):
        _translate(tmp_path, DERIVATIVES.format(size=size, layout=""))


def _lavapipe():
    """The host lavapipe runtime's environment with ``vkrun``, or ``None``.

    MoltenVK and KosmicKrisp ignore the derivative group of a compute entry, so only lavapipe can check the quads.
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


@pytest.fixture
def lavapipe():
    env = _lavapipe()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{REQUIRE_ENV}=1 but no host lavapipe runtime with vkrun")
        pytest.skip(f"no host lavapipe runtime; set {REQUIRE_ENV}=1 to require one")
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


def _quad(gi):
    """The local invocation indices of the 2x2 quad of invocation ``gi`` of an 8x8 group, in quad lane order."""
    x, y = (gi % 8) & ~1, (gi // 8) & ~1
    return [(y + dy) * 8 + x + dx for dy in (0, 1) for dx in (0, 1)]


def _linear(gi):
    """The local invocation indices of the linear quad of invocation ``gi``, in quad lane order."""
    return [gi & ~3, (gi & ~3) + 1, (gi & ~3) + 2, (gi & ~3) + 3]


@pytest.mark.parametrize(("layout", "quad"), [(QUADS, _quad), (LINEAR, _linear)])
def test_quad_swap_reads_the_declared_quad_on_lavapipe(
    tmp_path, lavapipe, layout, quad
):
    assembly = _translate(tmp_path, QUAD_READ.format(x=8, y=8, layout=layout))

    values = _run(tmp_path, lavapipe, assembly, list(range(64)))

    # QuadReadAcrossY reads quad lane ^ 2.
    assert values == [quad(gi)[quad(gi).index(gi) ^ 2] for gi in range(64)]


def test_quad_swap_without_a_layout_reads_the_subgroup_quad_on_lavapipe(
    tmp_path, lavapipe
):
    assembly = _translate(tmp_path, SUBGROUP_LANES)

    values = _run(tmp_path, lavapipe, assembly, [0] * 64)

    for value in values:
        partner, lane = divmod(value, 256)
        assert partner == lane ^ 2


def test_derivatives_read_the_declared_linear_quad_on_lavapipe(tmp_path, lavapipe):
    size = "local_size_x = 8, local_size_y = 8, local_size_z = 1"
    assembly = _translate(tmp_path, DERIVATIVES.format(size=size, layout=LINEAR))
    inputs = [gi * gi for gi in range(64)]

    values = _run(tmp_path, lavapipe, assembly, inputs)

    # ddy is quad lane 2 minus quad lane 0 for the top row and lane 3 minus lane 1 for the bottom row; the 2x2
    # group would read the invocation eight indices on instead.
    expected = []
    for gi in range(64):
        lanes = _linear(gi)
        column = lanes.index(gi) & 1
        expected.append(inputs[lanes[2 + column]] - inputs[lanes[column]])
    assert values == expected
