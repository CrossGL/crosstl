"""Compute derivatives keep the workgroup size and pick the derivative group it allows.

SPV_KHR_compute_shader_derivatives forms quads either as 2x2 blocks of the x and y dimensions
(``DerivativeGroupQuadsKHR``, both even) or as four consecutive local invocation indices
(``DerivativeGroupLinearKHR``, invocation count a multiple of four); HLSL Shader Model 6.6 forms compute quads by
the same rule. The generator used to round the x and y dimensions up to even numbers so that it could always use
quads, which changed the dispatch: a ``numthreads(4, 1, 1)`` kernel ran eight invocations per workgroup, with
``gl_LocalInvocationID.y`` reaching 1 and ``gl_GlobalInvocationID`` addressing rows the host never sized. A size
that fits neither group is now refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_compute_derivative_groups import (
    LINEAR,
    QUADS,
    compute_derivative_group,
)
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

DERIVATIVE = """shader S {{
    compute {{
        buffer float data[];
        @ numthreads({size})
        void main() {{
            float x = data[0];
            data[0] = dFdx(x);
        }}
    }}
}}
"""

QUERY_LOD = """shader S {{
    sampler2D colorMap;
    sampler querySampler;
    compute {{
        buffer vec2 data[];
        @ numthreads({size})
        void main() {{
            data[0] = textureQueryLod(colorMap, querySampler, vec2(0.25, 0.75));
        }}
    }}
}}
"""


def _translate(tmp_path, template, size):
    path = tmp_path / "shader.cgl"
    path.write_text(template.format(size=size), encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _validate(tmp_path, assembly):
    missing = [name for name in ("spirv-as", "spirv-val") if shutil.which(name) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not on PATH")
    version = re.search(r"; Version: 1\.(\d)", assembly).group(1)
    environment = {
        "0": "vulkan1.0",
        "3": "vulkan1.1",
        "5": "vulkan1.2",
        "6": "vulkan1.3",
    }.get(version, f"spv1.{version}")
    source = tmp_path / "out.spvasm"
    binary = tmp_path / "out.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", environment, str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", environment, str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("size", "group"),
    [
        ((2, 2, 1), QUADS),
        ((8, 8, 1), QUADS),
        ((2, 4, 3), QUADS),
        ((4, 1, 1), LINEAR),
        ((64, 1, 1), LINEAR),
        ((3, 4, 1), LINEAR),
        ((6, 1, 2), LINEAR),
        ((1, 1, 1), None),
        ((2, 3, 1), None),
        ((3, 1, 1), None),
        ((6, 1, 1), None),
    ],
)
def test_derivative_group_follows_the_workgroup_shape(size, group):
    assert compute_derivative_group(size) == group


@pytest.mark.parametrize(
    ("size", "mode", "capability"),
    [
        ("4, 1, 1", "DerivativeGroupLinearKHR", "ComputeDerivativeGroupLinearKHR"),
        ("2, 2, 1", "DerivativeGroupQuadsKHR", "ComputeDerivativeGroupQuadsKHR"),
        ("3, 4, 1", "DerivativeGroupLinearKHR", "ComputeDerivativeGroupLinearKHR"),
    ],
)
def test_derivatives_keep_the_declared_workgroup_size(tmp_path, size, mode, capability):
    assembly = _translate(tmp_path, DERIVATIVE, size)
    x, y, z = (part.strip() for part in size.split(","))
    assert re.search(rf"OpExecutionMode %\d+ LocalSize {x} {y} {z}\n", assembly)
    assert len(re.findall(r"OpExecutionMode %\d+ LocalSize ", assembly)) == 1
    assert re.search(rf"OpExecutionMode %\d+ {mode}\n", assembly)
    assert f"OpCapability {capability}\n" in assembly
    other = (
        {"ComputeDerivativeGroupQuadsKHR", "ComputeDerivativeGroupLinearKHR"}
        - {capability}
    ).pop()
    assert f"OpCapability {other}" not in assembly
    assert 'OpExtension "SPV_KHR_compute_shader_derivatives"' in assembly
    _validate(tmp_path, assembly)


@pytest.mark.parametrize("size", ["1, 1, 1", "2, 3, 1", "6, 1, 1"])
@pytest.mark.parametrize(
    "template", [DERIVATIVE, QUERY_LOD], ids=["dFdx", "textureQueryLod"]
)
def test_workgroup_without_a_derivative_group_is_refused(tmp_path, template, size):
    with pytest.raises(
        SpirvLoweringRefused, match="compute derivatives need a workgroup"
    ):
        _translate(tmp_path, template, size)


def test_query_lod_in_compute_keeps_the_workgroup_size(tmp_path):
    assembly = _translate(tmp_path, QUERY_LOD, "8, 1, 1")
    assert re.search(r"OpExecutionMode %\d+ LocalSize 8 1 1\n", assembly)
    assert re.search(r"OpExecutionMode %\d+ DerivativeGroupLinearKHR\n", assembly)
    _validate(tmp_path, assembly)
