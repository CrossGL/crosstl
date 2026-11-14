"""SPIR-V input keeps the domain, spacing, winding and point mode of a tessellation control entry point.

SPIR-V allows those execution modes on either tessellation stage (SPIR-V specification, Execution Mode), and DXC
writes a hull shader's on the ``TessellationControl`` entry point. The importer used to keep only
``OutputVertices``. It now spells the others as the hull attributes a CrossGL ``tessellation_control`` stage uses,
and SPIR-V output states them again on the control entry point.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.tessellation_control_import import (
    tessellation_control_mode_attributes,
)

CONTROL_TEMPLATE = """               OpCapability Tessellation
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationControl %main "main" %gl_InvocationID %out_pos
{modes}
               OpExecutionMode %main OutputVertices 3
               OpName %main "main"
               OpName %out_pos "out_pos"
               OpDecorate %gl_InvocationID BuiltIn InvocationId
               OpDecorate %out_pos Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
     %uint_3 = OpConstant %uint 3
    %float_1 = OpConstant %float 1
        %one = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
        %arr = OpTypeArray %v4float %uint_3
%ptr_out_arr = OpTypePointer Output %arr
 %ptr_in_uint = OpTypePointer Input %uint
 %ptr_out_v4 = OpTypePointer Output %v4float
%gl_InvocationID = OpVariable %ptr_in_uint Input
    %out_pos = OpVariable %ptr_out_arr Output
       %main = OpFunction %void None %fn
      %label = OpLabel
         %id = OpLoad %uint %gl_InvocationID
       %slot = OpAccessChain %ptr_out_v4 %out_pos %id
               OpStore %slot %one
               OpReturn
               OpFunctionEnd
"""

# The mode sets DXC writes on the control entry point for [domain], [partitioning] and [outputtopology].
DXC_MODE_SETS = {
    "tri-integer-point": ("Triangles", "SpacingEqual", "PointMode"),
    "tri-fractional-odd-ccw": ("Triangles", "SpacingFractionalOdd", "VertexOrderCcw"),
    "tri-fractional-even-cw": ("Triangles", "SpacingFractionalEven", "VertexOrderCw"),
    "quad-fractional-even-ccw": ("Quads", "SpacingFractionalEven", "VertexOrderCcw"),
    "isoline-integer": ("Isolines", "SpacingEqual"),
}

EXPECTED_ATTRIBUTES = {
    "tri-integer-point": [
        '@domain("tri")',
        '@partitioning("integer")',
        '@outputtopology("point")',
    ],
    "tri-fractional-odd-ccw": [
        '@domain("tri")',
        '@partitioning("fractional_odd")',
        '@outputtopology("triangle_ccw")',
    ],
    "tri-fractional-even-cw": [
        '@domain("tri")',
        '@partitioning("fractional_even")',
        '@outputtopology("triangle_cw")',
    ],
    "quad-fractional-even-ccw": [
        '@domain("quad")',
        '@partitioning("fractional_even")',
        '@outputtopology("triangle_ccw")',
    ],
    "isoline-integer": ['@domain("isoline")', '@partitioning("integer")'],
}


def _module(modes):
    return CONTROL_TEMPLATE.format(
        modes="\n".join(
            f"               OpExecutionMode %main {mode}" for mode in modes
        )
    )


def _write(tmp_path, source, name="control.spvasm"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate(tmp_path, assembly, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    source = _write(tmp_path, assembly, f"{name}.spvasm")
    binary = tmp_path / f"{name}.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", source, "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def _crossgl(tmp_path, modes):
    return crosstl.translate(
        _write(tmp_path, _module(modes)), backend="cgl", format_output=False
    )


def _control_modes(spirv):
    entry = re.search(r"OpEntryPoint TessellationControl %(\w+)", spirv)
    assert entry is not None, spirv
    return sorted(
        re.findall(
            rf"OpExecutionMode %{re.escape(entry.group(1))} (\w+(?: \d+)?)", spirv
        )
    )


def _round_trip(tmp_path, modes):
    crossgl = _crossgl(tmp_path, modes)
    path = tmp_path / "control.cgl"
    path.write_text(crossgl, encoding="utf-8")
    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    _validate(tmp_path, spirv, "round-trip")
    return spirv


@pytest.mark.parametrize("name", sorted(DXC_MODE_SETS))
def test_fixtures_are_valid_spirv(tmp_path, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    _validate(tmp_path, _module(DXC_MODE_SETS[name]), name)


@pytest.mark.parametrize("name", sorted(DXC_MODE_SETS))
def test_control_modes_import_as_hull_attributes(tmp_path, name):
    crossgl = _crossgl(tmp_path, DXC_MODE_SETS[name])

    assert "tessellation_control {" in crossgl
    assert "layout(vertices = 3) out;" in crossgl
    main = re.search(r"void main\(\)([^{]*)\{", crossgl)
    assert main is not None, crossgl
    assert main.group(1).split() == EXPECTED_ATTRIBUTES[name]


@pytest.mark.parametrize("name", sorted(DXC_MODE_SETS))
def test_spirv_output_states_the_control_modes_again(tmp_path, name):
    modes = DXC_MODE_SETS[name]
    spirv = _round_trip(tmp_path, modes)

    assert _control_modes(spirv) == sorted([*modes, "OutputVertices 3"])


def test_glslang_control_entry_point_keeps_only_output_vertices(tmp_path):
    crossgl = _crossgl(tmp_path, ())

    assert re.search(r"void main\(\) \{", crossgl), crossgl
    assert _control_modes(_round_trip(tmp_path, ())) == ["OutputVertices 3"]


def test_control_entry_point_without_a_domain_states_only_its_own_modes(tmp_path):
    modes = ("SpacingFractionalOdd", "VertexOrderCw")
    crossgl = _crossgl(tmp_path, modes)

    assert "@domain" not in crossgl
    assert '@partitioning("fractional_odd")' in crossgl
    assert '@outputtopology("triangle_cw")' in crossgl
    assert _control_modes(_round_trip(tmp_path, modes)) == [
        "OutputVertices 3",
        "SpacingFractionalOdd",
        "VertexOrderCw",
    ]


def test_control_entry_point_with_only_point_mode(tmp_path):
    spirv = _round_trip(tmp_path, ("PointMode",))

    assert _control_modes(spirv) == ["OutputVertices 3", "PointMode"]


@pytest.mark.parametrize(
    "modes, attributes",
    [
        (
            ("Triangles", "SpacingEqual", "PointMode", "VertexOrderCcw"),
            ['@domain("tri")', '@partitioning("integer")', '@outputtopology("point")'],
        ),
        (
            ("Isolines", "SpacingFractionalOdd", "VertexOrderCw"),
            [
                '@domain("isoline")',
                '@partitioning("fractional_odd")',
                '@outputtopology("line")',
            ],
        ),
        (("OutputVertices",), []),
        ((), []),
    ],
)
def test_winding_is_not_carried_where_it_has_no_effect(modes, attributes):
    assert (
        tessellation_control_mode_attributes([{"mode": mode} for mode in modes])
        == attributes
    )
