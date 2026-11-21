"""CrossGL subgroup rotations and quad votes, and the lowering of rotations to reads of the rotated lane."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator import parse
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused
from crosstl.translator.semantics import WaveRotationUnsupportedError
from crosstl.translator.semantics.hub_types import call_name
from crosstl.translator.semantics.scoped_walk import program_stages
from crosstl.translator.semantics.subgroup_rotations import (
    lower_wave_rotations,
    native_wave_rotations,
    register_native_wave_rotations,
)
from crosstl.translator.wave_operations import (
    GLSL_SUBGROUP_FUNCTIONS,
    WAVE_ROTATE_OPERATIONS,
)

GLSL_ROTATIONS = """#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_rotate : require
#extension GL_EXT_shader_quad_control : require
layout(local_size_x = 32) in;
layout(std430, binding = 0) buffer Inp { int a[]; };
layout(std430, binding = 1) buffer Outp { int o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    int x = a[i];
    o[i] = subgroupRotate(x, 3u) + subgroupClusteredRotate(x, 1u, 4u);
    if (subgroupQuadAll(x > 0) || subgroupQuadAny(x < -5)) {
        o[i] += 1;
    }
}
"""

ALL_ROTATIONS = """shader s {
    compute {
        RWStructuredBuffer<float4> data @buffer(0);
        RWStructuredBuffer<int> ints @buffer(1);
        @stage_entry
        @workgroup_size(32, 1, 1)
        void k(uvec3 id @gl_GlobalInvocationID) {
            int x = ints[id.x];
            uint d = uint(ints[0]);
            ints[id.x] = WaveRotate(x, 3u) + WaveClusteredRotate(x, d, 4u)
                + WaveRotate(WaveRotate(x, d), 1u);
            data[id.x] = WaveRotate(data[id.x], d) + WaveClusteredRotate(data[id.x], 2u, 8u);
        }
    }
}
"""


def crossgl(body):
    return f"""shader s {{
    compute {{
        RWStructuredBuffer<int> ints @buffer(0);
        @stage_entry
        @workgroup_size(32, 1, 1)
        void k(uvec3 id @gl_GlobalInvocationID) {{
            int x = ints[id.x];
            {body}
        }}
    }}
}}
"""


def translate(tmp_path, source, backend, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def wave_operations(node):
    return [n.operation for n in node.walk() if isinstance(n, A.WaveOpNode)]


def lowered(source, target="directx"):
    program = parse(source)
    count = lower_wave_rotations(program, target)
    return program, count


def _run(command, cwd):
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=120, check=False
    )


def test_glsl_rotations_and_quad_votes_have_crossgl_operations():
    assert GLSL_SUBGROUP_FUNCTIONS["subgroupRotate"] == "WaveRotate"
    assert GLSL_SUBGROUP_FUNCTIONS["subgroupClusteredRotate"] == "WaveClusteredRotate"
    assert GLSL_SUBGROUP_FUNCTIONS["subgroupQuadAll"] == "QuadAll"
    assert GLSL_SUBGROUP_FUNCTIONS["subgroupQuadAny"] == "QuadAny"
    assert WAVE_ROTATE_OPERATIONS == {"WaveRotate": 2, "WaveClusteredRotate": 3}


def test_rotations_parse_as_wave_operations():
    program = parse(
        crossgl("int r = WaveRotate(x, 3u) + WaveClusteredRotate(x, 1u, 4u);")
    )
    assert sorted(set(wave_operations(program))) == [
        "WaveClusteredRotate",
        "WaveRotate",
    ]


def test_glsl_names_parse_as_crossgl_operations():
    program = parse(
        crossgl(
            "int r = subgroupRotate(x, 3u) + subgroupClusteredRotate(x, 1u, 4u);\n"
            "bool q = subgroupQuadAll(x > 0) || subgroupQuadAny(x < 0);"
        )
    )
    assert sorted(set(wave_operations(program))) == [
        "QuadAll",
        "QuadAny",
        "WaveClusteredRotate",
        "WaveRotate",
    ]


def test_glsl_import_keeps_rotations_and_quad_votes_in_crossgl(tmp_path):
    output = translate(tmp_path, GLSL_ROTATIONS, "cgl", name="shader.comp")
    for name in ("WaveRotate(", "WaveClusteredRotate(", "QuadAll(", "QuadAny("):
        assert name in output
    assert "subgroupRotate" not in output
    assert "subgroupQuad" not in output


def helpers(program):
    return {
        f.name: f
        for f in program.functions
        if isinstance(f, A.FunctionNode) and f.name.startswith("crossgl_Wave")
    }


def entry(program):
    (stage,) = program_stages(program)
    return stage


def call_names(node):
    return [call_name(n) for n in node.walk() if isinstance(n, A.FunctionCallNode)]


def test_rotation_reads_the_lane_at_the_distance_modulo_the_subgroup():
    program, count = lowered(crossgl("int r = WaveRotate(x, 3u);"))
    assert count == 1
    (helper,) = helpers(program).values()
    assert helper.name == "crossgl_WaveRotate_int"
    assert [p.name for p in helper.parameters] == ["value", "delta"]
    declaration, result = helper.body.statements
    lane = declaration.initial_value
    assert isinstance(lane, A.BinaryOpNode) and lane.op == "&"
    assert lane.left.op == "+" and lane.left.left.operation == "WaveGetLaneIndex"
    assert lane.right.op == "-" and lane.right.left.operation == "WaveGetLaneCount"
    assert result.value.operation == "WaveReadLaneAt"
    assert "crossgl_WaveRotate_int" in call_names(entry(program))
    assert "WaveRotate" not in wave_operations(program)


def test_clustered_rotation_stays_in_the_aligned_cluster():
    program, count = lowered(crossgl("int r = WaveClusteredRotate(x, 1u, 4u);"))
    assert count == 1
    (helper,) = helpers(program).values()
    assert [p.name for p in helper.parameters] == ["value", "delta", "clusterSize"]
    lane = helper.body.statements[0].initial_value
    rotated, base = lane.left, lane.right
    assert lane.op == "+" and rotated.op == "&" and rotated.left.op == "+"
    assert base.op == "&" and base.right.op == "~"


def test_one_helper_per_operation_and_type_and_vectors_read_per_component():
    program, count = lowered(
        crossgl(
            "vec4 f = vec4(float(x));\n"
            "vec4 a = WaveRotate(f, 1u) + WaveRotate(f, 2u);\n"
            "int b = WaveRotate(x, 1u);\n"
            "bool c = WaveClusteredRotate(x > 0, 1u, 2u);"
        )
    )
    assert count == 4
    found = helpers(program)
    assert sorted(found) == [
        "crossgl_WaveClusteredRotate_bool",
        "crossgl_WaveRotate_int",
        "crossgl_WaveRotate_vec4",
    ]
    assert (
        wave_operations(found["crossgl_WaveRotate_vec4"]).count("WaveReadLaneAt") == 4
    )
    # Direct3D and Metal read numeric values only, so a bool travels as a uint.
    (read,) = (
        n
        for n in found["crossgl_WaveClusteredRotate_bool"].walk()
        if isinstance(n, A.WaveOpNode) and n.operation == "WaveReadLaneAt"
    )
    assert isinstance(read.arguments[0], A.TernaryOpNode)


def test_non_uint_deltas_are_converted_at_the_call():
    program, _ = lowered(crossgl("int r = WaveRotate(x, x) + WaveRotate(x, 2u);"))
    calls = [
        n
        for n in entry(program).walk()
        if isinstance(n, A.FunctionCallNode)
        and call_name(n) == "crossgl_WaveRotate_int"
    ]
    deltas = sorted(type(c.arguments[1]).__name__ for c in calls)
    assert deltas == ["FunctionCallNode", "LiteralNode"]
    converted = [
        c.arguments[1] for c in calls if isinstance(c.arguments[1], A.FunctionCallNode)
    ]
    assert call_name(converted[0]) == "uint"


def test_nested_rotations_are_all_lowered():
    program, count = lowered(crossgl("int r = WaveRotate(WaveRotate(x, 2u), 1u);"))
    assert count == 2
    assert "WaveRotate" not in wave_operations(program)
    assert call_names(entry(program)).count("crossgl_WaveRotate_int") == 2


def test_crossgl_and_registered_targets_keep_the_calls():
    assert native_wave_rotations("cgl") == frozenset(WAVE_ROTATE_OPERATIONS)
    program, count = lowered(crossgl("int r = WaveRotate(x, 3u);"), "cgl")
    assert count == 0 and "WaveRotate" in wave_operations(program)

    target = "wave-rotation-native-test"
    register_native_wave_rotations(target, ["WaveRotate"])
    program, count = lowered(
        crossgl("int r = WaveRotate(x, 3u) + WaveClusteredRotate(x, 1u, 4u);"),
        target,
    )
    assert count == 1
    assert "WaveRotate" in wave_operations(program)
    assert "WaveClusteredRotate" not in wave_operations(program)
    with pytest.raises(ValueError):
        register_native_wave_rotations(target, ["WaveReadLaneAt"])


@pytest.mark.parametrize(
    "body, reason",
    [
        ("int r = WaveRotate(x);", "arity"),
        ("int r = WaveClusteredRotate(x, 1u);", "arity"),
        ("bvec2 r = WaveRotate(bvec2(x > 0, x < 0), 1u);", "operand-type"),
    ],
)
def test_unsupported_calls_are_refused(body, reason):
    with pytest.raises(WaveRotationUnsupportedError) as caught:
        lowered(crossgl(body))
    assert caught.value.reason_code == reason
    assert caught.value.code.endswith("directx-wave-rotation-unsupported")


@pytest.mark.parametrize(
    "backend", ["directx", "metal", "cuda", "hip", "opengl", "slang"]
)
def test_targets_read_the_rotated_lane(tmp_path, backend):
    output = translate(tmp_path, ALL_ROTATIONS, backend)
    assert "crossgl_WaveRotate_vec4" in output
    assert "crossgl_WaveClusteredRotate_int" in output
    assert not re.search(
        r"\b(WaveRotate|WaveClusteredRotate|subgroupRotate)\s*\(", output
    )


def test_rust_output_refuses_the_rotations(tmp_path):
    # Rust output writes rust-gpu kernels, which do not lower subgroup operations yet: a typed refusal.
    with pytest.raises(RustLoweringRefused, match="subgroup operations"):
        translate(tmp_path, ALL_ROTATIONS, "rust")


def test_crossgl_output_keeps_the_rotations(tmp_path):
    output = translate(tmp_path, ALL_ROTATIONS, "cgl")
    assert output.count("WaveRotate(") == 4
    assert output.count("WaveClusteredRotate(") == 2


@pytest.mark.skipif(
    not (shutil.which("spirv-as") and shutil.which("spirv-val")),
    reason="SPIRV-Tools not installed",
)
def test_spirv_output_validates(tmp_path):
    assembly = translate(tmp_path, ALL_ROTATIONS, "vulkan")
    assert "OpGroupNonUniformShuffle" in assembly
    source, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    source.write_text(assembly, encoding="utf-8")
    assembled = _run(
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        tmp_path,
    )
    assert assembled.returncode == 0, assembled.stderr
    validated = _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)], tmp_path)
    assert validated.returncode == 0, validated.stderr


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslang not installed"
)
def test_glsl_output_compiles(tmp_path):
    glsl = translate(tmp_path, ALL_ROTATIONS, "opengl")
    path = tmp_path / "out.comp"
    path.write_text(glsl, encoding="utf-8")
    result = _run(
        ["glslangValidator", "-S", "comp", "--target-env", "vulkan1.1", str(path)],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    shutil.which("xcrun") is None, reason="Metal compiler not installed"
)
def test_metal_output_compiles(tmp_path):
    metal = translate(tmp_path, ALL_ROTATIONS, "metal")
    path = tmp_path / "out.metal"
    path.write_text(metal, encoding="utf-8")
    result = _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "out.air"),
        ],
        tmp_path,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("Metal toolchain not installed")
    assert result.returncode == 0, result.stderr
