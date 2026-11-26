"""HLSL-form tessellation control (hull) stages become Metal compute kernels.

Metal reads tessellation factors from a buffer and the post-tessellation vertex function reads the control points and
the per-patch values as ``[[stage_in]]`` attributes, so the generator writes a hull stage as a kernel that runs one
thread per patch: it calls the control-point function for every output control point and then the patch-constant
function, as HLSL runs them, and writes the factors as ``MTLTriangleTessellationFactorsHalf`` or
``MTLQuadTessellationFactorsHalf`` records (:mod:`crosstl.translator.codegen.metal_tessellation_control`). The GPU
tests run the translated kernel on this Mac's Metal device and compare every word it writes with a model of the
HLSL hull.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_stage_refusals import MetalStageUnsupportedError
from crosstl.translator.codegen.metal_tessellation_control import (
    MetalTessellationControlError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PATCHES = 4

TRIANGLE_HULL = """
cbuffer Params : register(b0) {
    float bias;
};

struct VSOut {
    float4 position : POSITION;
    float weight : WEIGHT;
};

struct HSOut {
    float4 position : POSITION;
    uint index : INDEX;
};

struct Constants {
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
    float4 centre : CENTRE;
    uint tag : TAG;
};

Constants patch_constants(InputPatch<VSOut, 3> patch, OutputPatch<HSOut, 3> points, uint pid : SV_PrimitiveID) {
    Constants c;
    for (int i = 0; i < 3; i++) {
        c.edges[i] = 1.0 + patch[i].weight + float(i);
    }
    c.inside = points[0].position.x + points[1].position.y + bias;
    c.centre = (points[0].position + points[1].position + points[2].position) * 0.25;
    c.tag = pid * 10u + points[2].index;
    return c;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("patch_constants")]
[maxtessfactor(16.0)]
HSOut main(InputPatch<VSOut, 3> patch, uint id : SV_OutputControlPointID, uint pid : SV_PrimitiveID) {
    HSOut o;
    o.position = patch[(id + 1u) % 3u].position * patch[id].weight + float4(float(pid), 0.0, 0.0, 0.0);
    o.index = id + pid * 3u;
    return o;
}
"""

QUAD_HULL = """
struct Point {
    float4 position : POSITION;
};

struct Constants {
    float edges[4] : SV_TessFactor;
    float inside[2] : SV_InsideTessFactor;
};

Constants quad_constants(InputPatch<Point, 4> patch, OutputPatch<Point, 2> points) {
    Constants c;
    c.edges[0] = patch[0].position.x;
    c.edges[1] = patch[1].position.y;
    c.edges[2] = patch[2].position.z;
    c.edges[3] = patch[3].position.w;
    c.inside[0] = points[0].position.x;
    c.inside[1] = points[1].position.y;
    return c;
}

[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_ccw")]
[outputcontrolpoints(2)]
[patchconstantfunc("quad_constants")]
Point quad_hull(InputPatch<Point, 4> patch, uint id : SV_OutputControlPointID) {
    Point o;
    o.position = patch[id].position + patch[id + 2u].position;
    return o;
}
"""

# The coverage system's HLSL domain probe with the hull a pipeline pairs it with.
HULL_AND_DOMAIN = """
struct ControlPoint {
    float4 position : POSITION;
};

struct PatchConstants {
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
    float4 extra : EXTRA;
};

PatchConstants patch_constants(InputPatch<ControlPoint, 3> patch) {
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
[outputtopology("triangle_ccw")]
[outputcontrolpoints(3)]
[patchconstantfunc("patch_constants")]
ControlPoint hull_main(InputPatch<ControlPoint, 3> patch, uint id : SV_OutputControlPointID) {
    ControlPoint output;
    output.position = patch[id].position * 2.0;
    return output;
}

[domain("tri")]
float4 main(PatchConstants constants, float3 coord : SV_DomainLocation,
            const OutputPatch<ControlPoint, 3> patch) : SV_Position {
    return patch[0].position * coord.x + constants.extra * coord.y;
}
"""

# CrossGL's own spelling: identifier arguments and a stage-local patch-constant function.
CROSSGL_HULL = """
shader TessControl {
    struct PatchInput { vec3 position @ POSITION; };
    struct PatchOutput { vec3 position @ POSITION; };
    struct PatchConstants {
        vec3 edgeFactors @ SV_TessFactor;
        float insideFactor @ SV_InsideTessFactor;
    };

    tessellation_control {
        PatchConstants patchConstants(InputPatch<PatchInput, 3> patch) {
            PatchConstants constants;
            constants.edgeFactors = vec3(patch[0].position.x, 2.0, 3.0);
            constants.insideFactor = 1.0;
            return constants;
        }

        PatchOutput main(InputPatch<PatchInput, 3> patch, uint pointID @ SV_OutputControlPointID)
            @domain(tri) @partitioning(integer) @outputtopology(triangle_cw)
            @outputcontrolpoints(3) @patchconstantfunc(patchConstants) {
            PatchOutput output;
            output.position = patch[pointID].position;
            return output;
        }
    }
}
"""


def parse(code):
    return Parser(Lexer(code).get_tokens()).parse()


def translate(tmp_path, source, name="hull.hlsl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def compile_metal(tmp_path, source):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    path = tmp_path / "translated.metal"
    path.write_text(source)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "-Werror", str(path)]
        + ["-o", str(tmp_path / "translated.air")],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr + source


def buffer_indices(source):
    """``{argument name: [[buffer(n)]]}`` of the generated kernel's arguments."""
    return {
        name: int(index)
        for name, index in re.findall(r"[*&]\s*(\w+)\s*\[\[buffer\((\d+)\)\]\]", source)
    }


def kernel_name(source):
    found = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", source)
    assert found, source
    return found.group(1)


def run_kernel(source, inputs, output_sizes, threads):
    """Run the generated kernel over ``threads`` threads; returns the bytes of each output buffer."""
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        source, metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(kernel_name(source)), None
    )
    assert pipeline is not None, error
    indices = buffer_indices(source)
    assert set(indices) == set(inputs) | set(output_sizes), indices
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    buffers = {}
    for name, data in inputs.items():
        buffers[name] = device.newBufferWithBytes_length_options_(data, len(data), 0)
    for name, size in output_sizes.items():
        buffers[name] = device.newBufferWithLength_options_(size, 0)
    for name, buffer in buffers.items():
        encoder.setBuffer_offset_atIndex_(buffer, 0, indices[name])
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(threads, 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return {
        name: bytes(buffers[name].contents().as_buffer(size))
        for name, size in output_sizes.items()
    }


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def triangle_model(bias):
    """What the HLSL hull of TRIANGLE_HULL writes for each patch: points, factors and patch data."""
    inputs, points, factors, data = [], [], [], []
    for pid in range(PATCHES):
        patch = [
            (
                (pid + i + 1.0, 2.0 * i, -float(pid), 0.5 * i),
                0.5 + 0.25 * i + 0.125 * pid,
            )
            for i in range(3)
        ]
        inputs += patch
        outputs = []
        for point in range(3):
            position = [
                component * patch[point][1] for component in patch[(point + 1) % 3][0]
            ]
            position[0] += pid
            outputs.append((tuple(position), point + pid * 3))
        points += outputs
        edges = [1.0 + patch[i][1] + i for i in range(3)]
        inside = outputs[0][0][0] + outputs[1][0][1] + bias
        centre = tuple(
            (outputs[0][0][axis] + outputs[1][0][axis] + outputs[2][0][axis]) * 0.25
            for axis in range(4)
        )
        factors.append(tuple(half(value) for value in edges + [inside]))
        data.append((centre, pid * 10 + outputs[2][1]))
    return inputs, points, factors, data


def test_hlsl_triangle_hull_runs_as_a_metal_compute_kernel(tmp_path):
    metal = translate(tmp_path, TRIANGLE_HULL)
    assert "kernel void" in metal and "MTLTriangleTessellationFactorsHalf" in metal
    # The hull's partitioning, winding and maximum factor are render pipeline state in Metal.
    assert "fractional" not in metal and "16.0" not in metal
    compile_metal(tmp_path, metal)

    bias = 0.25
    inputs, points, factors, data = triangle_model(bias)
    # Metal aligns a structure that holds a float4 to 16 bytes: each record here is 32 bytes.
    vertices = b"".join(
        struct.pack("<4ff12x", *position, weight) for position, weight in inputs
    )
    written = run_kernel(
        metal,
        {
            "main_input_points": vertices,
            "params": struct.pack("<f12x", bias),
        },
        {
            "main_output_points": 32 * 3 * PATCHES,
            "main_patch_data": 32 * PATCHES,
            "main_tessellation_factors": 8 * PATCHES,
        },
        PATCHES,
    )
    got_points = [
        (values[:4], values[4])
        for values in struct.iter_unpack("<4fI12x", written["main_output_points"])
    ]
    assert got_points == points
    assert (
        list(struct.iter_unpack("<4e", written["main_tessellation_factors"])) == factors
    )
    got_data = [
        (values[:4], values[4])
        for values in struct.iter_unpack("<4fI12x", written["main_patch_data"])
    ]
    assert got_data == data
    assert len({point for point, _ in points}) == len(points)


def test_hlsl_quad_hull_writes_quad_factors_on_metal(tmp_path):
    metal = translate(tmp_path, QUAD_HULL)
    assert "MTLQuadTessellationFactorsHalf" in metal
    compile_metal(tmp_path, metal)

    positions = [
        tuple(float(4 * (pid * 4 + i) + axis + 1) * 0.25 for axis in range(4))
        for pid in range(PATCHES)
        for i in range(4)
    ]
    written = run_kernel(
        metal,
        {"quad_hull_input_points": b"".join(struct.pack("<4f", *p) for p in positions)},
        {
            "quad_hull_output_points": 16 * 2 * PATCHES,
            "quad_hull_tessellation_factors": 12 * PATCHES,
        },
        PATCHES,
    )
    expected_points, expected_factors = [], []
    for pid in range(PATCHES):
        patch = positions[4 * pid : 4 * pid + 4]
        outputs = [
            tuple(a + b for a, b in zip(patch[point], patch[point + 2]))
            for point in range(2)
        ]
        expected_points += outputs
        edges = [patch[0][0], patch[1][1], patch[2][2], patch[3][3]]
        expected_factors.append(
            tuple(half(v) for v in edges + [outputs[0][0], outputs[1][1]])
        )
    got = list(struct.iter_unpack("<4f", written["quad_hull_output_points"]))
    assert got == expected_points
    factors = list(struct.iter_unpack("<6e", written["quad_hull_tessellation_factors"]))
    assert factors == expected_factors


@pytest.mark.parametrize("language", ["hlsl", "slang"])
def test_hull_and_domain_become_a_kernel_and_a_patch_function(tmp_path, language):
    source = HULL_AND_DOMAIN
    if language == "slang":
        source = source.replace(
            '[domain("tri")]\n[partitioning',
            '[shader("hull")]\n[domain("tri")]\n[partitioning',
        ).replace(
            '[domain("tri")]\nfloat4 main',
            '[shader("domain")]\n[domain("tri")]\nfloat4 main',
        )
    metal = translate(tmp_path, source, f"probe.{language}")
    assert re.search(r"kernel\s+void\s+hull_main\s*\(", metal), metal
    assert "[[patch(triangle, 3)]]" in metal
    # The kernel writes the per-patch value the post-tessellation function reads as attribute 1.
    data = re.search(r"struct hull_main_PatchData \{\s*float4 extra;\s*\};", metal)
    assert data, metal
    assert re.search(r"float4 extra \[\[attribute\(1\)\]\];", metal), metal
    flat = " ".join(metal.split())
    assert (
        "hull_main_patch_data[hull_main_patch].extra = hull_main_constants.extra;"
        in flat
    )
    compile_metal(tmp_path, metal)


def test_crossgl_hull_with_a_stage_local_patch_function(tmp_path):
    metal = MetalCodeGen().generate(parse(CROSSGL_HULL))
    assert "MTLTriangleTessellationFactorsHalf" in metal
    assert "edgeTessellationFactor[2] = half(main_constants.edgeFactors[2]);" in metal
    compile_metal(tmp_path, metal)

    stage = MetalCodeGen().generate_stage(parse(CROSSGL_HULL), "tessellation_control")
    assert "kernel void" in stage


def _hull(attributes="", parameters="", constants_members=None, body_constants=""):
    members = constants_members or (
        "float edges[3] : SV_TessFactor;\n    float inside : SV_InsideTessFactor;"
    )
    return f"""
struct Point {{ float4 position : POSITION; }};
struct Constants {{
    {members}
}};
Constants constants(InputPatch<Point, 3> patch) {{
    Constants c;
    {body_constants}
    return c;
}}
{attributes or '[domain("tri")]'}
[outputcontrolpoints(3)]
[patchconstantfunc("constants")]
Point main(InputPatch<Point, 3> patch, uint id : SV_OutputControlPointID{parameters}) {{
    Point o;
    o.position = patch[id].position;
    return o;
}}
"""


@pytest.mark.parametrize(
    ("source", "phrase"),
    [
        (_hull('[domain("isoline")]'), "not the 'isoline' domain"),
        (
            _hull('[domain("tri")]\n[outputtopology("point")]'),
            "not the 'point' output topology",
        ),
        (
            _hull(parameters=", float4 extra : COLOR"),
            "parameter 'extra' is neither",
        ),
        (
            _hull(constants_members="float edges[3] : SV_TessFactor;"),
            "lack the edge or the inside tessellation factors",
        ),
        (
            _hull(
                constants_members=(
                    "float edges[4] : SV_TessFactor;\n    float inside : SV_InsideTessFactor;"
                )
            ),
            "is not an array of 3 floats",
        ),
    ],
    ids=["isoline", "point-topology", "parameter", "missing-factor", "edge-count"],
)
def test_hull_shapes_metal_cannot_write_are_refused(tmp_path, source, phrase):
    with pytest.raises(MetalTessellationControlError) as raised:
        translate(tmp_path, source)
    assert (
        raised.value.code == "project.translate.metal-tessellation-control-unsupported"
    )
    assert phrase in str(raised.value)


def test_glsl_form_tessellation_control_stays_refused(tmp_path):
    path = tmp_path / "control.tesc"
    path.write_text(
        "#version 450\nlayout(vertices = 3) out;\nlayout(location = 0) in vec4 inPosition[];\n"
        "layout(location = 0) out vec4 outPosition[];\nvoid main() {\n"
        "    outPosition[gl_InvocationID] = inPosition[gl_InvocationID] * 2.0;\n"
        "    barrier();\n    gl_TessLevelOuter[0] = 1.0;\n    gl_TessLevelInner[0] = 1.0;\n}\n"
    )
    with pytest.raises(MetalStageUnsupportedError) as raised:
        crosstl.translate(str(path), backend="metal")
    assert raised.value.stage == "tessellation_control"
    assert "GLSL-form" in str(raised.value)
