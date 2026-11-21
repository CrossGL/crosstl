"""CrossGL partitioned wave reductions and scans, and their lowering to loops over the subgroup's lanes."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator import parse
from crosstl.translator.semantics import PartitionedWaveUnsupportedError
from crosstl.translator.semantics.hub_types import call_name
from crosstl.translator.semantics.partitioned_waves import (
    lower_partitioned_wave_operations,
    native_partitioned_wave_operations,
    register_native_partitioned_wave_operations,
)
from crosstl.translator.wave_operations import (
    GLSL_SUBGROUP_FUNCTIONS,
    WAVE_PARTITIONED_OPERATIONS,
)

GLSL_PARTITIONED = """#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_NV_shader_subgroup_partitioned : require
layout(local_size_x = 32) in;
layout(std430, binding = 0) buffer Inp { int a[]; };
layout(std430, binding = 1) buffer Outp { int o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    int x = a[i];
    uvec4 m = subgroupPartitionNV(x & 3);
    o[i] = subgroupPartitionedAddNV(x, m) + subgroupPartitionedInclusiveMinNV(x, m)
        + subgroupPartitionedExclusiveMaxNV(x, m) + subgroupPartitionedXorNV(x, m);
}
"""


def crossgl(body, declarations=""):
    return f"""shader s {{
    {declarations}
    compute {{
        RWStructuredBuffer<float4> data @buffer(0);
        RWStructuredBuffer<int> ints @buffer(1);
        @stage_entry
        @workgroup_size(32, 1, 1)
        void k(uvec3 id @gl_GlobalInvocationID) {{
            uvec4 m = WaveActiveBallot(ints[id.x] > 0);
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
    count = lower_partitioned_wave_operations(program, target)
    return program, count


def test_every_glsl_partitioned_function_has_a_crossgl_operation():
    partitioned = {
        name: operation
        for name, operation in GLSL_SUBGROUP_FUNCTIONS.items()
        if name.startswith("subgroupPartitioned")
    }
    assert len(partitioned) == 21
    for name, operation in partitioned.items():
        if "Exclusive" in name and not re.search(r"(Min|Max)NV$", name):
            assert operation.startswith("WaveMultiPrefix")
            assert operation not in WAVE_PARTITIONED_OPERATIONS
        else:
            assert operation in WAVE_PARTITIONED_OPERATIONS


def test_operations_parse_as_wave_operations():
    names = sorted(WAVE_PARTITIONED_OPERATIONS)
    body = "\n".join(f"float v{i} = {name}(1.0, m);" for i, name in enumerate(names))
    assert sorted(set(wave_operations(parse(crossgl(body))))) == sorted(
        names + ["WaveActiveBallot"]
    )


def test_glsl_import_keeps_partitioned_operations_in_crossgl(tmp_path):
    output = translate(tmp_path, GLSL_PARTITIONED, "cgl", name="shader.comp")
    for name in (
        "WaveMultiSum",
        "WaveMultiPrefixInclusiveMin",
        "WaveMultiPrefixExclusiveMax",
        "WaveMultiBitXor",
    ):
        assert f"{name}(" in output
    assert "subgroupPartitioned" not in output


def test_one_helper_per_operation_and_operand_type():
    program, count = lowered(
        crossgl(
            "vec4 a = WaveMultiSum(data[id.x], m);\n"
            "vec4 b = WaveMultiSum(data[id.x] * 2.0, m);\n"
            "int c = WaveMultiSum(ints[id.x], m);\n"
            "int d = WaveMultiMin(WaveMultiMax(ints[id.x], m), m);"
        )
    )
    assert count == 5
    names = [f.name for f in program.functions]
    assert names == [
        "crossgl_WaveMultiSum_vec4",
        "crossgl_WaveMultiSum_int",
        # The walk is post-order, so the inner call's helper comes first.
        "crossgl_WaveMultiMax_int",
        "crossgl_WaveMultiMin_int",
    ]
    assert not set(wave_operations(program)) & set(WAVE_PARTITIONED_OPERATIONS)
    calls = [
        call_name(n)
        for n in program.walk()
        if isinstance(n, A.FunctionCallNode) and call_name(n).startswith("crossgl_")
    ]
    assert sorted(calls) == sorted(
        [
            "crossgl_WaveMultiSum_vec4",
            "crossgl_WaveMultiSum_vec4",
            "crossgl_WaveMultiSum_int",
            "crossgl_WaveMultiMin_int",
            "crossgl_WaveMultiMax_int",
        ]
    )


def test_helper_names_avoid_program_names():
    program, _ = lowered(
        crossgl(
            "int a = WaveMultiSum(ints[id.x], m);",
            declarations="int crossgl_WaveMultiSum_int(int v) { return v; }",
        )
    )
    names = [f.name for f in program.functions]
    assert names[0] == "crossgl_WaveMultiSum_int_2"
    assert "crossgl_WaveMultiSum_int" in names[1:]


@pytest.mark.parametrize(
    "operation, operand, result_type, identity, scope",
    [
        (
            "WaveMultiPrefixExclusiveMin",
            "data[id.x].x",
            "float",
            "asfloat(2139095040u)",
            "<",
        ),
        ("WaveMultiPrefixExclusiveMax", "ints[id.x]", "int", "(-2147483647 - 1)", "<"),
        (
            "WaveMultiPrefixInclusiveBitAnd",
            "uint(ints[id.x])",
            "uint",
            "4294967295u",
            "<=",
        ),
        ("WaveMultiProduct", "data[id.x].xy", "float2", "float2(1.0, 1.0)", None),
    ],
)
def test_helpers_start_from_the_identity_and_scan_lower_lanes(
    tmp_path, operation, operand, result_type, identity, scope
):
    hlsl = translate(
        tmp_path,
        crossgl(f"data[id.x].y = float({result_type}({operation}({operand}, m)).x);"),
        "directx",
    )
    body = re.search(
        rf"crossgl_{operation}_\w+\([^)]*\)\s*\{{(.*?)\n\}}\n", hlsl, re.S
    ).group(1)
    assert f"{result_type} result = {identity};" in body
    assert re.findall(r"\(lane (<=?) self_lane\)", body) == (
        [] if scope is None else [scope]
    )


def test_vector_reads_are_per_component():
    program, _ = lowered(crossgl("data[id.x] = WaveMultiMax(data[id.x], m);"))
    reads = [
        n
        for n in program.functions[0].walk()
        if isinstance(n, A.WaveOpNode) and n.operation == "WaveReadLaneAt"
    ]
    assert len(reads) == 4


def test_crossgl_and_registered_targets_keep_the_calls():
    assert native_partitioned_wave_operations("cgl") == frozenset(
        WAVE_PARTITIONED_OPERATIONS
    )
    program, count = lowered(crossgl("int a = WaveMultiSum(ints[id.x], m);"), "cgl")
    assert count == 0 and "WaveMultiSum" in wave_operations(program)

    target = "partitioned-native-test"
    register_native_partitioned_wave_operations(target, ["WaveMultiSum"])
    program, count = lowered(
        crossgl("int a = WaveMultiSum(ints[id.x], m) + WaveMultiMin(ints[id.x], m);"),
        target,
    )
    assert count == 1
    assert "WaveMultiSum" in wave_operations(program)
    assert "WaveMultiMin" not in wave_operations(program)
    with pytest.raises(ValueError):
        register_native_partitioned_wave_operations(target, ["WaveActiveSum"])


@pytest.mark.parametrize(
    "body, reason",
    [
        ("bvec2 b = WaveMultiBitAnd(bvec2(true, false), m);", "operand-type"),
        ("bool b = WaveMultiSum(true, m);", "operand-type"),
        ("int64_t b = WaveMultiSum(int64_t(ints[id.x]), m);", "operand-type"),
        ("int b = WaveMultiSum(ints[id.x]);", "arity"),
    ],
)
def test_unsupported_operands_are_refused(body, reason):
    with pytest.raises(PartitionedWaveUnsupportedError) as caught:
        lowered(crossgl(body), "metal")
    assert caught.value.reason_code == reason
    assert caught.value.code == "project.translate.metal-partitioned-wave-unsupported"


def test_boolean_operands_use_logical_operators(tmp_path):
    hlsl = translate(
        tmp_path,
        crossgl("ints[id.x] = WaveMultiBitXor(ints[id.x] > 1, m) ? 1 : 0;"),
        "directx",
    )
    assert "bool result = false;" in hlsl
    assert "result = (result != other);" in hlsl


def _run(command, cwd):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)


ALL_OPERATIONS = crossgl(
    "\n".join(
        [
            "vec4 acc = vec4(0.0);",
            *(
                f"acc += vec4(float({name}(ints[id.x], m)), {name}(data[id.x].x, m), "
                f"float({name}(uint(ints[id.x]), m)), 0.0);"
                for name in sorted(WAVE_PARTITIONED_OPERATIONS)
                if WAVE_PARTITIONED_OPERATIONS[name][0] not in ("and", "or", "xor")
            ),
            *(
                f"acc.w += float({name}(ints[id.x], m) + int({name}(uint(ints[id.x]), m)))"
                f" + ({name}(ints[id.x] > 2, m) ? 1.0 : 0.0);"
                for name in sorted(WAVE_PARTITIONED_OPERATIONS)
                if WAVE_PARTITIONED_OPERATIONS[name][0] in ("and", "or", "xor")
            ),
            "data[id.x] = acc + WaveMultiPrefixInclusiveMin(data[id.x], m);",
        ]
    )
)


@pytest.mark.skipif(
    not (shutil.which("spirv-as") and shutil.which("spirv-val")),
    reason="SPIRV-Tools not installed",
)
def test_spirv_output_of_every_operation_validates(tmp_path):
    assembly = translate(tmp_path, ALL_OPERATIONS, "vulkan")
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
def test_glsl_output_of_every_operation_compiles(tmp_path):
    glsl = translate(tmp_path, ALL_OPERATIONS, "opengl")
    assert "subgroupPartitioned" not in glsl
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
def test_metal_output_of_every_operation_compiles(tmp_path):
    metal = translate(tmp_path, ALL_OPERATIONS, "metal")
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


@pytest.mark.parametrize("backend", ["directx", "cuda", "hip", "slang", "rust"])
def test_other_targets_translate_every_operation(tmp_path, backend):
    output = translate(tmp_path, ALL_OPERATIONS, backend)
    assert "crossgl_WaveMultiPrefixInclusiveMin_vec4" in output
    assert not re.search(r"\bWaveMulti(Sum|Min|Max|Product|Bit\w+)\s*\(", output)
