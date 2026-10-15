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
        (
            "if (invocation < 16u) { uint low = value; total = WaveActiveMin(low); }",
            "",
        ),
        ("if (invocation < 16u) { total = WaveActiveMax(value) + 1u; }", ""),
        (
            "if (invocation < 16u) { total = WaveActiveSum(value); total = WaveActiveSum(total) + 1u; }",
            "",
        ),
        (
            "if (invocation < 16u) { total = WaveActiveSum(value); uint kept = total; total = WaveActiveSum(kept); }",
            "",
        ),
        ("if (invocation < 16u) { total += WaveActiveSum(value); }", ""),
        ("if ((total++) < 16u) { total = WaveActiveSum(value); }", ""),
        (
            "if (invocation < 64u) { if (invocation < 16u) { total = WaveActiveSum(value); } }",
            "",
        ),
        (
            "for (uint step = 0u; step < invocation; ++step) { if (invocation < 16u) { total = WaveActiveSum(value); } }",
            "",
        ),
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


# MLX's RMS VJP reduces two partials under one ``simd_group_id == 0`` guard
# inside its row loop. Each reduction runs in every invocation in source
# order, so the second payload sees the first result, and a guarded reduction
# nested in a block or a uniform loop is predicated where it stands.
PREDICATED_SEQUENCES = {
    "sequence": (
        "if (invocation < 16u) { total = WaveActiveSum(value); "
        "total = WaveActiveSum(total); }",
        lambda partial: 16 * partial,
    ),
    "block": (
        "{ if (invocation < 16u) { total = WaveActiveSum(value); } }",
        lambda partial: partial,
    ),
    "loop": (
        "uint stop = group.x + 2u; if (stop > 2u) { stop = 2u; } "
        "for (uint step = 0u; step < stop; ++step) { "
        "if (invocation < 16u) { total = WaveActiveSum(total + value); } }",
        lambda partial: 17 * partial,
    ),
}


def _predicated_sequence(name):
    return _source(
        "uint",
        "uint value = inputWords[index]; uint total = 0u; "
        f"{PREDICATED_SEQUENCES[name][0]} outputWords[index] = total;",
    )


def _predicated_sequence_expected(name, words):
    expected = []
    for index in range(len(words)):
        base, local = index // 128 * 128, index % 128
        partial = sum(words[base : base + 16])
        expected.append(
            PREDICATED_SEQUENCES[name][1](partial) % 2**32 if local < 16 else 0
        )
    return expected


@pytest.mark.parametrize("name", sorted(PREDICATED_SEQUENCES))
def test_masked_sequences_reach_every_barrier(tmp_path, name):
    generated = _codegen().generate_stage(parse(_predicated_sequence(name)), "compute")
    assert "WaveActiveSum" not in generated
    calls = [
        line.strip()
        for line in generated.splitlines()
        if "= __crossgl_software_subgroup_sum_uint(crossglSoftwareSubgroupInput" in line
    ]
    assert len(calls) == (2 if name == "sequence" else 1)
    if name == "sequence":
        # The first result is assigned under the guard before the second
        # payload reads it.
        lines = [line.strip() for line in generated.splitlines()]
        start = lines.index("total = crossglSoftwareSubgroupResult;")
        assert lines[start + 1] == "crossglSoftwareSubgroupInput_ = total;"
    _compile(generated, tmp_path)


@pytest.mark.parametrize("name", sorted(PREDICATED_SEQUENCES))
def test_masked_sequences_execute(tmp_path, directx_runtime, name):
    generated = _codegen().generate_stage(parse(_predicated_sequence(name)), "compute")
    words = [index * 3 + 1 for index in range(384)]
    expected = _predicated_sequence_expected(name, words)
    assert _execute_words(tmp_path, generated, words, expected) == expected


# MLX's block logsumexp reduces SIMD-group maxima under
# ``if (simd_group_id == 0)``. Inactive invocations contribute the extremum's
# identity, and the active lanes below hold values that would win if inactive
# invocations took part with their own inputs.
MASKED_EXTREMA = {
    "max-float": (
        "float",
        "WaveActiveMax",
        "asfloat(0xff800000u)",
        """float value = asfloat(inputWords[index]);
        float largest = -1.0;
        if ((invocation & 31u) >= 4u && invocation < 96u) {
            largest = WaveActiveMax(value);
            outputWords[index] = asuint(largest);
        }""",
    ),
    "min-int": (
        "int",
        "WaveActiveMin",
        "2147483647",
        """int value = asint(inputWords[index]);
        int smallest = 0;
        if ((invocation & 31u) < 20u) {
            smallest = WaveActiveMin(value);
            outputWords[index] = asuint(smallest);
        }""",
    ),
    "max-uint": (
        "uint",
        "WaveActiveMax",
        "0u",
        """uint value = inputWords[index];
        uint largest = 5u;
        if (invocation >= 32u && (invocation & 1u) == 0u) {
            largest = WaveActiveMax(value);
            outputWords[index] = largest;
        }""",
    ),
}


def _masked_extremum_active(name, local):
    lane = local % 32
    if name == "max-float":
        return lane >= 4 and local < 96
    if name == "min-int":
        return lane < 20
    return local >= 32 and local % 2 == 0


def _masked_extremum_words(name):
    words = []
    for index in range(384):
        local = index % 128
        active = _masked_extremum_active(name, local)
        if name == "max-float":
            # Inactive lanes hold +1000, active lanes stay within [-50, 50], and
            # the second subgroup's active lanes are all -inf.
            value = float(index % 101 - 50)
            if not active:
                value = 1000.0
            elif local // 32 == 1:
                value = float("-inf")
            words.append(struct.unpack("<I", struct.pack("<f", value))[0])
        elif name == "min-int":
            value = -(2**31) if not active else (index * 7919) % 2001 - 1000
            words.append(value % 2**32)
        else:
            words.append(0xFFFFFFFF if not active else (index * 2654435761) % 2**31)
    return words


def _masked_extremum_expected(name, value_type, words):
    def decode(word):
        if value_type == "float":
            return struct.unpack("<f", struct.pack("<I", word))[0]
        if value_type == "int":
            return word - 2**32 if word >= 2**31 else word
        return word

    expected = []
    for index in range(len(words)):
        local = index % 128
        if not _masked_extremum_active(name, local):
            expected.append(0xDEADBEEF)
            continue
        base = index - local % 32
        values = [
            decode(words[base + lane])
            for lane in range(32)
            if _masked_extremum_active(name, base % 128 + lane)
        ]
        result = min(values) if name == "min-int" else max(values)
        if value_type == "float":
            expected.append(struct.unpack("<I", struct.pack("<f", result))[0])
        else:
            expected.append(result % 2**32)
    return expected


@pytest.mark.parametrize("name", MASKED_EXTREMA)
def test_masked_extremum_reaches_every_barrier(tmp_path, name):
    value_type, operation, identity, body = MASKED_EXTREMA[name]
    generated = _codegen().generate_stage(parse(_source(value_type, body)), "compute")
    assert operation not in generated
    assert f"{value_type} crossglSoftwareSubgroupInput = {identity};" in generated
    helper = "max" if operation == "WaveActiveMax" else "min"
    call = (
        f"__crossgl_software_subgroup_{helper}_{value_type}"
        "(crossglSoftwareSubgroupInput"
    )
    assert generated.count(call) == 1
    lines = [line.strip() for line in generated.splitlines()]
    reduction = next(index for index, line in enumerate(lines) if call in line)
    assert lines[reduction - 1] == "}" and lines[reduction + 1].startswith("if (")
    _compile(generated, tmp_path)


def test_masked_extremum_cases_detect_inactive_participation():
    for name, (value_type, _, _, _) in MASKED_EXTREMA.items():
        words = _masked_extremum_words(name)
        expected = _masked_extremum_expected(name, value_type, words)
        # Every subgroup with an active lane has an inactive lane whose value
        # would replace the result if it took part.
        assert len(set(expected)) > 4
        assert any(word != 0xDEADBEEF for word in expected)
        for index, word in enumerate(expected):
            if word != 0xDEADBEEF:
                assert word not in {0x447A0000, 0x80000000, 0xFFFFFFFF}, name
    float_words = _masked_extremum_words("max-float")
    float_expected = _masked_extremum_expected("max-float", "float", float_words)
    assert float_expected[36] == 0xFF800000  # the all -inf subgroup


@pytest.mark.parametrize("name", MASKED_EXTREMA)
def test_masked_extremum_executes(tmp_path, directx_runtime, name):
    value_type, _, _, body = MASKED_EXTREMA[name]
    generated = _codegen().generate_stage(parse(_source(value_type, body)), "compute")
    words = _masked_extremum_words(name)
    expected = _masked_extremum_expected(name, value_type, words)
    assert _execute_words(tmp_path, generated, words, expected) == expected
