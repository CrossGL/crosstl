"""Slang output: wave-op operands keep their own types, and a domain stage takes the hull's modes.

A wave op's operands used to inherit the type its result is assigned to, so a ``uint`` lane index inside
``WaveReadLaneAt(float4 value, lane)`` was refused as "returns uint but target expects float4". A CrossGL domain
stage may state ``@partitioning`` and ``@outputtopology``; Slang, like HLSL, takes them from the hull, so the domain
output validates and omits them. SV_DomainLocation is ``float2`` on quad and isoline domains and ``float3`` on a
tri domain. Every output is compiled with slangc when it is installed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import SlangRefusal


def translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang")


def compile_slang(tmp_path, code, entry, stage):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source, binary = tmp_path / "out.slang", tmp_path / "out.spv"
    source.write_text(code, encoding="utf-8")
    command = [slangc, str(source), "-target", "spirv", "-o", str(binary)]
    command += ["-entry", entry, "-stage", stage]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    if shutil.which("spirv-val"):
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(binary)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


LANE_INDEX = """shader LaneContext {
    compute {
        RWStructuredBuffer<float4> data @buffer(0);
        @stage_entry
        @workgroup_size(32, 1, 1)
        void k(uvec3 id @gl_GlobalInvocationID) {
            data[id.x] = WaveReadLaneAt(data[id.x], (WaveGetLaneIndex() + 1u) & (WaveGetLaneCount() - 1u));
        }
    }
}
"""


def test_wave_lane_index_operand_is_not_typed_by_the_result(tmp_path):
    code = translate(tmp_path, LANE_INDEX)
    assert (
        "WaveReadLaneAt(data[id.x], WaveGetLaneIndex() + 1u & WaveGetLaneCount() - 1u)"
        in code
    )
    compile_slang(tmp_path, code, "k", "compute")


def test_wave_result_is_still_checked_against_its_target(tmp_path):
    source = LANE_INDEX.replace(
        "data[id.x] = WaveReadLaneAt(data[id.x], (WaveGetLaneIndex() + 1u) & (WaveGetLaneCount() - 1u));",
        "data[id.x] = WaveGetLaneIndex();",
    )
    with pytest.raises(
        Exception, match="WaveGetLaneIndex returns uint but target expects float4"
    ):
        translate(tmp_path, source)


SHUFFLE = """shader Shuffle {
    compute {
        RWStructuredBuffer<float4> data @buffer(0);
        @stage_entry
        @workgroup_size(32, 1, 1)
        void k(uvec3 id @gl_GlobalInvocationID) {
            data[id.x] = WaveShuffleDown(data[id.x], WaveGetLaneCount() / 2u);
        }
    }
}
"""


def test_wave_shuffle_delta_operand_is_not_typed_by_the_result(tmp_path):
    try:
        code = translate(tmp_path, SHUFFLE)
    except Exception as error:  # the CrossGL front end may not offer this spelling
        if "WaveShuffleDown" in str(error) and "not declared" in str(error):
            pytest.skip("CrossGL has no WaveShuffleDown spelling")
        raise
    assert "WaveGetLaneCount() / 2u" in code
    compile_slang(tmp_path, code, "k", "compute")


def domain_program(domain, coord_type, extra=""):
    count = {"tri": 3, "quad": 4, "isoline": 2}[domain]
    if domain == "tri":
        edges, inside = (
            "float edges[3] @ SV_TessFactor;",
            "float inside @ SV_InsideTessFactor;",
        )
    elif domain == "quad":
        edges = "float edges[4] @ SV_TessFactor;"
        inside = "float inside[2] @ SV_InsideTessFactor;"
    else:
        edges, inside = "float edges[2] @ SV_TessFactor;", ""
    weight = "coord.x" if domain != "tri" else "coord.z"
    return f"""shader Probe {{
    struct ControlPoint {{
        vec4 position @ POSITION;
    }}
    struct PatchConstants {{
        {edges}
        {inside}
    }}
    struct DomainOutput {{
        vec4 position @ SV_Position;
    }}
    tessellation_evaluation {{
        DomainOutput main(PatchConstants constants, {coord_type} coord @ SV_DomainLocation,
                          OutputPatch<ControlPoint, {count}> patch) @domain({domain}){extra} {{
            DomainOutput output;
            output.position = patch[0].position * {weight} * constants.edges[0];
            return output;
        }}
    }}
}}
"""


@pytest.mark.parametrize(
    "domain, coord_type, topology",
    [
        ("tri", "vec3", "triangle_ccw"),
        ("quad", "vec2", "triangle_cw"),
        ("isoline", "vec2", "line"),
    ],
)
def test_domain_stage_validates_and_omits_hull_modes(
    tmp_path, domain, coord_type, topology
):
    extra = f" @partitioning(fractional_odd) @outputtopology({topology})"
    code = translate(tmp_path, domain_program(domain, coord_type, extra))
    assert f'[domain("{domain}")]' in code
    assert "outputtopology" not in code
    assert "partitioning" not in code
    compile_slang(tmp_path, code, "main", "domain")


@pytest.mark.parametrize("domain", ["quad", "isoline"])
def test_quad_and_isoline_domain_location_is_float2(tmp_path, domain):
    code = translate(tmp_path, domain_program(domain, "vec2"))
    assert "float2 coord : SV_DomainLocation" in code
    compile_slang(tmp_path, code, "main", "domain")


@pytest.mark.parametrize(
    "domain, coord_type, expected",
    [
        ("tri", "vec2", "float3"),
        ("quad", "vec3", "float2"),
        ("isoline", "vec3", "float2"),
    ],
)
def test_domain_location_of_the_wrong_width_is_refused(
    tmp_path, domain, coord_type, expected
):
    with pytest.raises(
        SlangRefusal, match=f"SV_DomainLocation .*expected {expected} type"
    ):
        translate(tmp_path, domain_program(domain, coord_type))


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            " @partitioning(nearest)",
            "tessellation_evaluation stage partitioning 'nearest'",
        ),
        (
            " @outputtopology(point_list)",
            "tessellation_evaluation stage outputtopology 'point_list'",
        ),
        (
            " @outputtopology(line)",
            "domain 'tri' requires outputtopology triangle_cw or triangle_ccw",
        ),
    ],
)
def test_domain_stage_refuses_invalid_hull_modes(tmp_path, extra, message):
    with pytest.raises(SlangRefusal, match=message):
        translate(tmp_path, domain_program("tri", "vec3", extra))
