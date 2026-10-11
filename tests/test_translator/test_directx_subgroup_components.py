"""Component-level workgroup uniformity for explicit software collectives."""

import json
import math

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_software_reductions import (
    _codegen,
    _execute_words,
)
from tests.test_translator.test_directx_software_reductions import (
    directx_runtime as _directx_runtime,
)

directx_runtime = _directx_runtime
COUNTS = (3, 3, 2)
CASES = (
    ((32, 1, 1), "y"),
    ((32, 1, 4), "y"),
    ((32, 4, 1), "z"),
    ((64, 1, 1), "y"),
    ((1, 32, 1), "z"),
    ((1, 1, 32), "y"),
)


def _source(shape, body, helpers=""):
    x, y, z = shape
    return f"""shader UniformComponents {{
        StructuredBuffer<uint> inputWords @register(t0);
        RWStructuredBuffer<uint> outputWords @register(u1);
        {helpers}
        compute {{
            @numthreads({x}, {y}, {z})
            void main(uvec3 global @gl_GlobalInvocationID,
                      uvec3 local @gl_LocalInvocationID) {{
                uint index = global.x + {x * COUNTS[0]}u * (global.y + {y * COUNTS[1]}u * global.z);
                {body}
            }}
        }}
    }}"""


def _row_source(shape, axis):
    limit = 1 if axis == "z" else 2
    return _source(
        shape,
        f"""
        const uint row = uint(global.{axis});
        if (row >= {limit}u) {{ return; }}
        const uint repeats = row + 1u;
        uint value = inputWords[index];
        uint result = 0u;
        for (uint step = 0u; step < repeats; ++step) {{
            result += WaveActiveSum(value) + WaveShuffleDown(value, 1u);
        }}
        outputWords[index] = result;
    """,
    )


@pytest.mark.parametrize("shape,axis", CASES)
def test_uniform_rows_compile(tmp_path, shape, axis):
    generated = _codegen().generate_stage(parse(_row_source(shape, axis)), "compute")
    assert "if (row >= " in generated
    assert "return;" in generated
    assert "GroupMemoryBarrierWithGroupSync();" in generated
    assert "WaveActiveSum" not in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("shape,axis", [((1, 32, 1), "y"), ((1, 1, 32), "z")])
def test_component_proof_rejects_varying_axes_in_non_x_major_groups(shape, axis):
    with pytest.raises(DirectXSoftwareSubgroupError) as failure:
        _codegen().generate_stage(parse(_row_source(shape, axis)), "compute")
    assert failure.value.reason == "early-return-unproven"


@pytest.mark.parametrize("shape,axis", CASES)
def test_uniform_rows_execute(tmp_path, directx_runtime, shape, axis):
    generated = _codegen().generate_stage(parse(_row_source(shape, axis)), "compute")
    width, height, depth = (size * count for size, count in zip(shape, COUNTS))
    words = [index * 17 + 3 for index in range(width * height * depth)]
    expected = [0xDEADBEEF] * len(words)
    component = "xyz".index(axis)
    for gz in range(COUNTS[2]):
        for gy in range(COUNTS[1]):
            for gx in range(COUNTS[0]):
                row = (gx, gy, gz)[component]
                if row >= (1 if axis == "z" else 2):
                    continue
                indices = [
                    gx * shape[0]
                    + x
                    + width * (gy * shape[1] + y + height * (gz * shape[2] + z))
                    for z in range(shape[2])
                    for y in range(shape[1])
                    for x in range(shape[0])
                ]
                for start in range(0, len(indices), 32):
                    lanes = indices[start : start + 32]
                    total = sum(words[index] for index in lanes)
                    for lane, index in enumerate(lanes):
                        neighbor = words[lanes[min(lane + 1, 31)]]
                        expected[index] = (row + 1) * (total + neighbor)
    (tmp_path / "case.json").write_text(
        json.dumps({"shape": shape, "counts": COUNTS, "axis": axis})
    )
    assert len(words) == math.prod(shape) * math.prod(COUNTS)
    assert (
        _execute_words(
            tmp_path,
            generated,
            words,
            expected,
            workgroup_size=shape,
            workgroup_count=COUNTS,
        )
        == expected
    )


@pytest.mark.parametrize(
    "expression",
    [
        "global.y",
        "global.z",
        "local.y",
        "local.z",
        "global.g",
        "global.yz.x",
        "int(global.y)",
        "uint(global.y + local.z)",
    ],
)
def test_uniform_components_compose_with_scalar_expressions(tmp_path, expression):
    source = _source(
        (32, 1, 1),
        f"""
        const uint row = uint({expression});
        if (row >= 2u) {{ return; }}
        outputWords[index] = WaveActiveSum(index);
    """,
    )
    _compile(_codegen().generate_stage(parse(source), "compute"), tmp_path)


@pytest.mark.parametrize(
    "shape,expression",
    [
        ((32, 1, 1), "global.x"),
        ((32, 4, 1), "global.y"),
        ((32, 1, 4), "global.z"),
        ((32, 4, 1), "local.y"),
        ((32, 1, 4), "local.z"),
        ((32, 1, 1), "global.yx.y"),
        ((32, 1, 1), "global.y + global.x"),
        ((32, 1, 1), "local.y + local.x"),
        ((32, 1, 1), "inputWords[global.y]"),
    ],
)
def test_varying_components_cannot_select_collectives(shape, expression):
    source = _source(
        shape,
        f"if ({expression} != 0u) {{ return; }} outputWords[index] = WaveActiveSum(index);",
    )
    with pytest.raises(DirectXSoftwareSubgroupError) as failure:
        _codegen().generate_stage(parse(source), "compute")
    assert failure.value.reason == "early-return-unproven"


@pytest.mark.parametrize(
    "mutation",
    [
        "global.y = local.x;",
        "global.x += 1u;",
        "change(global.y, local.x);",
        "unknown(global);",
        "uvec3& alias = global; alias.y = local.x;",
        "uvec3* alias = &global;",
        "uvec3 global = uvec3(local.x);",
    ],
)
def test_component_proof_rejects_mutation_and_shadowing(mutation):
    source = _source(
        (32, 1, 1),
        f"{{ {mutation} if (global.y != 0u) {{ return; }} outputWords[index] = WaveActiveSum(index); }}",
        "void change(inout uint value, uint lane) { value = lane; }",
    )
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(source), "compute")


def test_component_proof_is_lexically_scoped(tmp_path):
    source = _source(
        (32, 1, 1),
        """
        { uvec3 global = uvec3(local.x); outputWords[index] = global.y; }
        if (global.y != 0u) { return; }
        outputWords[index] = WaveActiveSum(index);
    """,
    )
    _compile(_codegen().generate_stage(parse(source), "compute"), tmp_path)


def test_helper_semantics_do_not_prove_physical_components():
    source = _source(
        (32, 1, 1),
        "outputWords[index] = helper(uvec3(local.x), index);",
        """
        uint helper(uvec3 global @gl_GlobalInvocationID, uint value) {
            if (global.y != 0u) { return value; }
            return WaveActiveSum(value);
        }
    """,
    )
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(source), "compute")


def test_component_proof_is_reset_between_generations(tmp_path):
    codegen = _codegen()
    safe = _row_source((32, 1, 1), "y")
    codegen.generate_stage(parse(safe), "compute")
    with pytest.raises(DirectXSoftwareSubgroupError):
        codegen.generate_stage(parse(_row_source((32, 4, 1), "y")), "compute")
    result = codegen.generate_stage(parse(safe), "compute")
    assert result == _codegen().generate_stage(parse(safe), "compute")
    _compile(result, tmp_path)
