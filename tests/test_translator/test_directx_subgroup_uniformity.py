"""Conservative workgroup-uniform proofs for software subgroup barriers."""

import struct

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_software_reductions import (
    _codegen,
    _execute_words,
    _source,
)
from tests.test_translator.test_directx_software_reductions import (
    directx_runtime as _directx_runtime,
)

directx_runtime = _directx_runtime
CHECKPOINT_CASES = ((1, 0), (4, 1), (8, 3))


def _checkpoint_source(checkpoint, offset):
    return _source(
        "uint",
        f"""const int length = int(group.x) * 5 + {offset};
        const int checkpoints = (length + {checkpoint} - 1) / {checkpoint};
        uint value = inputWords[index];
        uint total = 0u;
        for (int segment = checkpoints - 1; segment >= 0; --segment) {{
            const int start = segment * {checkpoint};
            const int stop = min({checkpoint}, max(length - start, 0));
            for (int step = 0; step < stop; ++step) {{
                total += WaveActiveSum(value);
            }}
            for (int reverse = stop - 1; reverse >= 0; --reverse) {{
                total += WaveShuffleDown(value, 1u);
            }}
        }}
        outputWords[index] = total;
        """,
    )


@pytest.mark.parametrize("checkpoint,offset", CHECKPOINT_CASES)
def test_uniform_checkpoint_loops_compile(tmp_path, checkpoint, offset):
    generated = _codegen().generate_stage(
        parse(_checkpoint_source(checkpoint, offset)), "compute"
    )
    assert "for (int segment" in generated
    assert "for (int step" in generated and "for (int reverse" in generated
    assert "WaveActiveSum" not in generated
    assert "GroupMemoryBarrierWithGroupSync();" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("checkpoint,offset", CHECKPOINT_CASES)
def test_uniform_checkpoint_loops_execute(
    tmp_path, directx_runtime, checkpoint, offset
):
    generated = _codegen().generate_stage(
        parse(_checkpoint_source(checkpoint, offset)), "compute"
    )
    words = [index * 3 + 1 for index in range(384)]
    expected = []
    for index in range(len(words)):
        start = index // 32 * 32
        length = index // 128 * 5 + offset
        neighbor = words[min(index + 1, start + 31)]
        expected.append(length * (sum(words[start : start + 32]) + neighbor))
    assert _execute_words(tmp_path, generated, words, expected) == expected


@pytest.mark.parametrize(
    "expression",
    ["min(length, 3u)", "max(length, 1u)", "clamp(length, 1u, 3u)", "uint(length)"],
)
def test_uniform_constant_reference_bounds_compile(tmp_path, expression):
    source = tmp_path / "kernel.metal"
    source.write_text(f"""#include <metal_stdlib>
    using namespace metal;
    kernel void collect(device uint* output [[buffer(0)]],
                        constant uint& length [[buffer(1)]],
                        uint index [[thread_index_in_threadgroup]]) {{
        const uint stop = {expression};
        uint value = 0u;
        for (uint step = 0u; step < stop; ++step) {{
            value += simd_shuffle_down(index, 1u);
        }}
        output[index] = value;
    }}""")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=(source.name,),
            targets=("directx",),
            output_dir="out",
            entry_points={source.name: ("collect",)},
            workgroup_size=(32, 4, 1),
            source_options={
                "metal": {
                    "target_options": {
                        "directx": {
                            "software_subgroup_width": 32,
                            "relative_wave_shuffle_out_of_range": "self",
                        }
                    }
                }
            },
        ),
        format_output=False,
    )
    report.write_json(tmp_path / "report.json")
    payload = report.to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    _compile((tmp_path / payload["artifacts"][0]["path"]).read_text(), tmp_path)


@pytest.mark.parametrize(
    "body,helpers",
    [
        ("uint stop = invocation;", ""),
        ("uint stop = inputWords[0];", ""),
        ("uint stop = WaveActiveSum(1u);", ""),
        ("uint stop = group.x; stop = invocation;", ""),
        ("uint stop = group.x; stop += invocation;", ""),
        ("uint stop = group.x; ++stop;", ""),
        (
            "uint stop = group.x; change(stop, invocation);",
            "void change(inout uint value, uint lane) { value = lane; }",
        ),
        ("uint stop = group.x; unknown(stop);", ""),
        ("uint stop = unknown(group.x);", ""),
        ("uint stop = min(group.x, invocation);", ""),
        (
            "uint stop = min(group.x, 3u);",
            "uint min(uint left, uint right) { return inputWords[left]; }",
        ),
        (
            "uint stop = group.x; uint ignored = min(stop, invocation);",
            "uint min(inout uint left, uint right) { left = right; return left; }",
        ),
        ("uint stop = group.x; if (invocation == 0u) { stop = 3u; }", ""),
        ("uint stop = group.x; uint& alias = stop; alias = invocation;", ""),
        ("uint stop = group.x; uint* alias = &stop; *alias = invocation;", ""),
        ("uint stop = group.x; { uint stop = invocation; BODY }", ""),
        ("uint stop = invocation; { uint stop = group.x; }", ""),
    ],
)
def test_uniform_local_proof_rejects_unproven_bounds(body, helpers):
    loop = "for (uint step = 0u; step < stop; ++step) { outputWords[index] = WaveActiveSum(index); }"
    body = body.replace("BODY", loop) if "BODY" in body else body + loop
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(_source("uint", body, helpers)), "compute")


@pytest.mark.parametrize(
    "body",
    [
        "uint stop = group.x; for (uint step = 0u; step < stop; ++step) { if (invocation == 0u) { break; } outputWords[index] = WaveActiveSum(index); }",
        "uint stop = group.x; for (uint step = 0u; step < stop; ++step) { if (invocation == 0u) { continue; } outputWords[index] = WaveActiveSum(index); }",
        "uint stop = group.x; for (uint step = 0u; step < stop; ++step) { if (invocation == 0u) { return; } outputWords[index] = WaveActiveSum(index); }",
        "uint stop = group.x; for (uint step = 0u; step < stop; ++step) { stop = invocation; outputWords[index] = WaveActiveSum(index); }",
        "uint stop = group.x; for (uint step = 0u; step < stop; ++step) { uint step = invocation; if (step != 0u) { return; } outputWords[index] = WaveActiveSum(index); }",
    ],
)
def test_uniform_local_proof_retains_loop_exit_checks(body):
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(_source("uint", body)), "compute")


def test_uniform_local_proof_preserves_lexical_scope(tmp_path):
    body = """uint stop = group.x;
        { uint stop = invocation; outputWords[index] = stop; }
        for (uint step = 0u; step < stop; ++step) {
            const uint nested = min(step, stop);
            if (nested == 0u) { outputWords[index] = WaveActiveSum(index); }
        }
        const bool empty = stop == 0u;
        if (empty) { return; }
        outputWords[index] = WaveActiveSum(index);
    """
    generated = _codegen().generate_stage(parse(_source("uint", body)), "compute")
    _compile(generated, tmp_path)


def test_uniform_builtin_facts_do_not_survive_generator_reuse(tmp_path):
    codegen = _codegen()
    codegen.generate_stage(parse(_checkpoint_source(4, 1)), "compute")
    unsafe = _source(
        "uint",
        "uint stop = min(group.x, 1u); for (uint i = 0u; i < stop; ++i) { outputWords[index] = WaveActiveSum(index); }",
        "uint min(uint left, uint right) { return inputWords[left]; }",
    )
    with pytest.raises(DirectXSoftwareSubgroupError):
        codegen.generate_stage(parse(unsafe), "compute")
    clean = _checkpoint_source(8, 3)
    generated = codegen.generate_stage(parse(clean), "compute")
    assert generated == _codegen().generate_stage(parse(clean), "compute")
    _compile(generated, tmp_path)


# MLX's dot product reduces SIMD-group partials under ``if (tid < SIMD_GROUPS)``.
# The masked lowering runs the reduction in every invocation, with inactive
# invocations contributing the sum identity.
MASKED_SUMS = {
    "uint": (
        """uint value = inputWords[index];
        uint total = 7u;
        if (invocation < 16u) {
            total = value * 2u;
            total = WaveActiveSum(total);
            outputWords[index] = total;
        }"""
    ),
    "float": (
        """float value = asfloat(inputWords[index]);
        float total = 1.0;
        if (invocation >= 8u && invocation < 24u) {
            total = WaveActiveSum(value);
            outputWords[index] = asuint(total);
        }"""
    ),
}


def _masked_sum_words(value_type):
    if value_type == "uint":
        return [index * 3 + 1 for index in range(384)]
    # The first group is all -0.0, whose sum stays -0.0 only if inactive
    # invocations contribute -0.0 rather than +0.0.
    return [0x80000000] * 128 + [
        struct.unpack("<I", struct.pack("<f", float(index % 7 - 3)))[0]
        for index in range(128, 384)
    ]


def _masked_sum_expected(value_type, words):
    expected = []
    for index in range(len(words)):
        base, local = index // 128 * 128, index % 128
        if value_type == "uint":
            active = range(16)
            if local not in active:
                expected.append(0xDEADBEEF)
                continue
            expected.append(sum(2 * words[base + lane] for lane in active) % 2**32)
            continue
        active = range(8, 24)
        if local not in active:
            expected.append(0xDEADBEEF)
            continue
        values = [
            struct.unpack("<f", struct.pack("<I", words[base + lane]))[0]
            for lane in active
        ]
        total = -0.0
        for value in values:
            total += value
        expected.append(struct.unpack("<I", struct.pack("<f", total))[0])
    return expected


@pytest.mark.parametrize("value_type", ["uint", "float"])
def test_masked_sum_reaches_every_barrier(tmp_path, value_type):
    generated = _codegen().generate_stage(
        parse(_source(value_type, MASKED_SUMS[value_type])), "compute"
    )
    assert "WaveActiveSum" not in generated
    identity = "asfloat(0x80000000u)" if value_type == "float" else "0u"
    assert f"{value_type} crossglSoftwareSubgroupInput = {identity};" in generated
    call = f"__crossgl_software_subgroup_sum_{value_type}(crossglSoftwareSubgroupInput"
    assert generated.count(call) == 1
    # The reduction sits outside both predicated blocks.
    lines = [line.strip() for line in generated.splitlines()]
    reduction = next(index for index, line in enumerate(lines) if call in line)
    assert lines[reduction - 1] == "}" and lines[reduction + 1].startswith("if (")
    _compile(generated, tmp_path)


@pytest.mark.parametrize("value_type", ["uint", "float"])
def test_masked_sum_executes(tmp_path, directx_runtime, value_type):
    generated = _codegen().generate_stage(
        parse(_source(value_type, MASKED_SUMS[value_type])), "compute"
    )
    words = _masked_sum_words(value_type)
    expected = _masked_sum_expected(value_type, words)
    assert expected[8] == (0x80000000 if value_type == "float" else expected[8])
    assert _execute_words(tmp_path, generated, words, expected) == expected


@pytest.mark.parametrize(
    "body,helpers",
    [
        (
            "if (invocation < 16u) { total = WaveActiveSum(value); } else { total = 1u; }",
            "",
        ),
        (
            "if (invocation < 16u) { uint doubled = value * 2u; total = WaveActiveSum(doubled); }",
            "",
        ),
        (
            "if (invocation < 16u) { total = WaveActiveSum(value); total += reduce_value(value); }",
            "uint reduce_value(uint value) { return WaveActiveSum(value); }",
        ),
        ("if (invocation < 16u) { total = WaveActiveSum(value); return; }", ""),
        ("if (invocation < 16u) { total = WaveActiveMin(value); }", ""),
        (
            "if (invocation < 16u) { total = WaveActiveSum(value); total = WaveActiveSum(total); }",
            "",
        ),
        ("if (invocation < 16u) { total += WaveActiveSum(value); }", ""),
        ("if ((total++) < 16u) { total = WaveActiveSum(value); }", ""),
        ("{ if (invocation < 16u) { total = WaveActiveSum(value); } }", ""),
    ],
)
def test_masked_sum_rejects_unplanned_divergence(body, helpers):
    source = _source(
        "uint",
        f"uint value = inputWords[index]; uint total = 0u; {body} "
        "outputWords[index] = total;",
        helpers,
    )
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(source), "compute")
