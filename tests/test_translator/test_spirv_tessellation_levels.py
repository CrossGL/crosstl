"""SPIR-V output of tessellation evaluation stages that read the patch constants as a structure.

HLSL's domain shader and Metal's post-tessellation vertex function take the patch constants as a structure: its
tessellation level members are narrower than Vulkan's ``float[4]`` ``TessLevelOuter`` and ``float[2]``
``TessLevelInner`` built-ins (``float edges[3] : SV_TessFactor``, ``float inside : SV_InsideTessFactor``), and its
other members are per-patch values. SPIR-V output used to build the structure from the whole built-in arrays, which
spirv-val rejects ("Expected Constituent type to be equal to the corresponding member type"), and gave a per-patch
member the control points' location without a ``Patch`` decoration, which it refused as a location conflict. Element
``i`` of a narrower level now reads element ``i`` of the built-in, as DXC reads it, and a per-patch member is a
``Patch`` input after the control-point inputs, where the tessellation control stage writes it. Metal input writes
``[[position_in_patch]]`` as ``SV_DomainLocation``, which is the ``TessCoord`` built-in.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

CONTROL_POINT = """struct ControlPoint {
    float4 position : POSITION;
};
"""

TRIANGLE_LEVELS = CONTROL_POINT + """
struct PatchConstants {
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

[domain("tri")]
float4 main(PatchConstants constants, float3 coord : SV_DomainLocation,
            const OutputPatch<ControlPoint, 3> patch) : SV_Position {
    return patch[0].position * constants.edges[2] + patch[1].position * constants.inside
        + patch[2].position * coord.z;
}
"""

ISOLINE_LEVELS = CONTROL_POINT + """
struct PatchConstants {
    float edges[2] : SV_TessFactor;
};

[domain("isoline")]
float4 main(PatchConstants constants, float2 coord : SV_DomainLocation,
            const OutputPatch<ControlPoint, 2> patch) : SV_Position {
    return patch[0].position * constants.edges[0] + patch[1].position * constants.edges[1];
}
"""

PATCH_INPUT = CONTROL_POINT + """
struct PatchConstants {
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
    float4 extra : EXTRA;
};

[domain("tri")]
float4 main(PatchConstants constants, float3 coord : SV_DomainLocation,
            const OutputPatch<ControlPoint, 3> patch) : SV_Position {
    return patch[0].position * coord.x + constants.extra * coord.y;
}
"""

PATCH_OUTPUT = CONTROL_POINT + """
struct PatchConstants {
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
    float4 extra : EXTRA;
};

PatchConstants constants(InputPatch<ControlPoint, 3> patch) {
    PatchConstants output;
    output.edges[0] = 1.0;
    output.edges[1] = 1.0;
    output.edges[2] = 1.0;
    output.inside = 1.0;
    output.extra = patch[0].position;
    return output;
}

[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("constants")]
ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id : SV_OutputControlPointID) {
    ControlPoint output;
    output.position = patch[id].position * 2.0;
    return output;
}
"""

METAL_PATCH_INPUT = """#include <metal_stdlib>
using namespace metal;

struct ControlPoint {
    float4 position [[attribute(0)]];
};

struct Patch {
    patch_control_point<ControlPoint> points;
    float4 extra [[attribute(1)]];
};

[[patch(triangle, 3)]]
vertex float4 probe(Patch patch [[stage_in]], float3 coord [[position_in_patch]]) {
    return patch.points[0].position * coord.x + patch.extra * coord.y;
}
"""


def _translate(tmp_path, source, suffix=".hlsl"):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _validate(tmp_path, assembly):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "probe.spvasm"
    binary = tmp_path / "probe.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        [assembler, "--target-env", "vulkan1.3", str(source), "-o", str(binary)],
        [validator, "--target-env", "vulkan1.3", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout


def _named(assembly, name):
    """The result id ``OpName`` gives ``name``."""
    match = re.search(rf'OpName (%\S+) "{re.escape(name)}"', assembly)
    assert match, f"no variable named {name}"
    return match.group(1)


def _decorations(assembly, variable):
    return re.findall(rf"OpDecorate {re.escape(variable)} (\w+(?: \d+)?)", assembly)


def _extracted_elements(assembly, built_in):
    """The element indices read from a whole load of the ``built_in`` variable."""
    variable = _named(assembly, built_in)
    loads = re.findall(rf"(%\S+) = OpLoad %\S+ {re.escape(variable)}\b", assembly)
    assert len(loads) == 1, f"{built_in} is loaded {len(loads)} times"
    return [
        int(index)
        for index in re.findall(
            rf"= OpCompositeExtract %\S+ {re.escape(loads[0])} (\d+)", assembly
        )
    ]


def test_triangle_levels_read_the_leading_built_in_elements(tmp_path):
    assembly = _translate(tmp_path, TRIANGLE_LEVELS)

    assert _extracted_elements(assembly, "gl_TessLevelOuter") == [0, 1, 2]
    assert _extracted_elements(assembly, "gl_TessLevelInner") == [0]
    _validate(tmp_path, assembly)


def test_isoline_levels_read_the_two_leading_outer_elements(tmp_path):
    assembly = _translate(tmp_path, ISOLINE_LEVELS)

    assert _extracted_elements(assembly, "gl_TessLevelOuter") == [0, 1]
    assert re.search(r"OpExecutionMode %\S+ Isolines", assembly)
    _validate(tmp_path, assembly)


def test_per_patch_member_is_a_patch_input_after_the_control_points(tmp_path):
    assembly = _translate(tmp_path, PATCH_INPUT)

    extra = _named(assembly, "CrossGL_tessellationevaluation_input_constants_extra")
    assert _decorations(assembly, extra) == ["Location 1", "Patch"]
    assert _decorations(assembly, _named(assembly, "patch")) == ["Location 0"]
    _validate(tmp_path, assembly)


def test_patch_input_location_matches_the_control_stage_output(tmp_path):
    """The domain stage reads ``extra`` where SPIR-V output of the hull stage writes it."""
    control = _translate(tmp_path, PATCH_OUTPUT)
    written = _decorations(control, _named(control, "tc_patch_extra"))
    evaluation = _translate(tmp_path, PATCH_INPUT)
    read = _decorations(
        evaluation,
        _named(evaluation, "CrossGL_tessellationevaluation_input_constants_extra"),
    )

    assert sorted(written) == sorted(read) == ["Location 1", "Patch"]


def test_metal_patch_attribute_is_a_patch_input(tmp_path):
    assembly = _translate(tmp_path, METAL_PATCH_INPUT, suffix=".metal")

    patch_inputs = [
        _decorations(assembly, variable)
        for variable in re.findall(r"OpDecorate (%\S+) Patch", assembly)
    ]
    assert patch_inputs == [["Location 1", "Patch"]], assembly
    # [[position_in_patch]] arrives as SV_DomainLocation and is the TessCoord built-in, not a user input.
    assert re.search(r"OpDecorate %\S+ BuiltIn TessCoord", assembly)
    _validate(tmp_path, assembly)
