"""Runtime scalar obligations bound collective induction without substitution."""

import copy
from dataclasses import replace

import pytest

from crosstl._crosstl import translate
from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.translator import parse
from crosstl.translator.ast import (
    AssignmentNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.loop_induction import fixed_stride_loop_terminates
from crosstl.translator.metadata_preconditions import (
    MetadataPreconditionError,
    MetadataRangePrecondition,
    parse_metadata_precondition_header,
    resolve_metadata_preconditions,
)
from tests.test_translator.test_project_metadata_entry_scope import _config, _packages

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void scan_loop(device uint* output [[buffer(0)]],
                      constant size_t& axis_size [[buffer(1)]],
                      uint lane [[thread_index_in_threadgroup]],
                      uint3 group [[threadgroup_position_in_grid]]) {
    uint result = 0;
    for (uint j = 0; j < axis_size; j += 32) {
        result += simd_prefix_inclusive_sum(lane + j);
    }
    output[group.x * 64 + lane + 4] = result;
}
"""
CONDITION = dict(
    entryPoint="scan_loop", parameter="axis_size", minimum=0, maximum=65535
)
TARGETS = ("directx", "opengl")
GUARD = 0xABCDEF01


def _loop_config(root, target, *, source=SOURCE, conditions=(CONDITION,), **kwargs):
    config = _config(
        root, target, source=source, conditions=conditions, entries=("scan_loop",)
    )
    options = dict(preserve_resource_origins=True)
    if target in TARGETS:
        target_options = dict(software_subgroup_width=32)
        if target == "directx":
            target_options["relative_wave_shuffle_out_of_range"] = "self"
        options["target_options"] = {target: target_options}
    return replace(
        config,
        workgroup_size=(32, 2, 1),
        source_options={"metal": options},
        **kwargs,
    )


def _request(descriptor, package, count):
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        origin = binding.get("provenance", {}).get("sourceResource")
        if origin is None:
            continue
        name = binding["name"]
        parameter = origin["parameter"]
        dtype = binding["scalarLayout"]["elementType"]
        words = [count] if parameter == "axis_size" else [GUARD] * 136
        inputs[name] = dict(dtype=dtype, shape=[len(words)], values=words)
        if parameter == "axis_size":
            observed.append(name)
        else:
            result = []
            for _group in range(2):
                for lane in range(64):
                    result.append(
                        sum(
                            sum(range((lane // 32) * 32 + j, lane + j + 1))
                            for j in range(0, count, 32)
                        )
                        & 0xFFFFFFFF
                    )
            outputs[name] = dict(
                dtype=dtype, shape=[136], values=[GUARD] * 4 + result + [GUARD] * 4
            )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [2, 1, 1], "workgroupSize": [32, 2, 1]},
        observe_inputs=observed,
    )
    return request, outputs


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "header",
    [
        "uint j = 0; j < axis_size; j += 32",
        "uint j = 0; axis_size > j; j += 32",
        "uint j = 0; j <= axis_size; j += 32",
        "uint j = 65535; j > axis_size; j -= 1",
        "uint j = 0; j < axis_size; ++j",
        "uint j = 65535; j > axis_size; --j",
    ],
)
def test_checked_runtime_bound_retains_loop_and_package_contract(
    tmp_path, target, header
):
    source = SOURCE.replace("uint j = 0; j < axis_size; j += 32", header)
    _, package, (descriptor,) = _packages(_loop_config(tmp_path, target, source=source))
    artifact = (package / descriptor["artifact"]["packagePath"]).read_text()
    records = parse_metadata_precondition_header(artifact)
    assert len(records) == 1 and records[0]["dtype"] == "uint64"
    assert records[0]["maximum"] == 65535
    loop = next(
        line
        for line in artifact.splitlines()
        if "for (" in line and "axis_size" in line
    )
    assert "axis_size" in loop
    for count in (0, 1, 31, 32, 33, 64, 65, 65535):
        request, _ = _request(descriptor, package, count)
        assert not request.execution_plan.diagnostics
    with pytest.raises(NativeLoaderDispatchError, match="outside"):
        _request(descriptor, package, 65536)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("maximum", [4294967295, 4294967296, 18446744073709551615])
def test_metadata_bound_does_not_hide_counter_overflow(tmp_path, target, maximum):
    config = _loop_config(
        tmp_path, target, conditions=({**CONDITION, "maximum": maximum},)
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert any(item["severity"] == "error" for item in payload["diagnostics"])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "body",
    [
        "uint axis_size = lane;",
        "uint axis_size = 4294967295u;",
        "uint j = lane;",
        "if (lane == 0) { continue; }",
        "if (lane == 0) { break; }",
        "if (lane == 0) { return; }",
        "j += lane;",
    ],
)
def test_runtime_proof_does_not_accept_shadowing_or_divergence(tmp_path, target, body):
    source = SOURCE.replace("result += simd", body + " result += simd")
    payload = translate_project(
        _loop_config(tmp_path, target, source=source), format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload


@pytest.mark.parametrize("target", TARGETS)
def test_uncontracted_runtime_bound_remains_rejected(tmp_path, target):
    payload = translate_project(
        _loop_config(tmp_path, target, conditions=()), format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert any(
        item.get("details", {}).get("reason") == "potentially-divergent-control-flow"
        or "statically uniform" in item["message"]
        for item in payload["diagnostics"]
    ), payload["diagnostics"]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "rewrite",
    [
        ("j < axis_size", "j < axis_size + 1"),
        ("j < axis_size", "j < lane"),
        ("j += 32", "j += axis_size"),
        ("j = 0", "j = lane"),
        ("j += 32", "j *= 2"),
        ("uint j = 0", "volatile uint j = 0"),
        ("constant size_t&", "constant volatile size_t&"),
        ("uint result = 0;", "uint result = 0; uint axis_size = lane;"),
        ("uint result = 0;", "uint result = 0; const uint axis_size = 4294967295u;"),
    ],
)
def test_metadata_does_not_prove_other_expressions_or_mutable_values(
    tmp_path, target, rewrite
):
    source = SOURCE.replace(*rewrite)
    payload = translate_project(
        _loop_config(tmp_path, target, source=source), format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("start", ["axis_size", "uint(group.x)"])
def test_uniform_starts_need_no_value_with_a_bounded_stop(tmp_path, target, start):
    # Every visited value precedes axis_size <= 65535, so no start of the
    # counter's type reaches past 65535 + 31 before the loop exits.
    source = SOURCE.replace("uint j = 0", f"uint j = {start}")
    payload = translate_project(
        _loop_config(tmp_path, target, source=source), format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    unbounded = translate_project(
        _loop_config(tmp_path / "unbounded", target, source=source, conditions=()),
        format_output=False,
    ).to_json()
    assert unbounded["summary"]["translatedCount"] == 0, unbounded


@pytest.mark.parametrize("target", TARGETS)
def test_dispatch_region_identity_binds_runtime_loop_obligations(tmp_path, target):
    from crosstl.translator.dispatch_regions import plan_dispatch_regions

    region = plan_dispatch_regions([64, 2, 1], [32, 2, 1])[0]
    identities = []
    for maximum in (65535, 1023):
        root = tmp_path / str(maximum)
        config = _loop_config(
            root, target, conditions=({**CONDITION, "maximum": maximum},)
        )
        options = copy.deepcopy(config.source_options)
        options["metal"]["target_options"][target]["dispatch_region"] = region.to_json()
        payload, _, _ = _packages(replace(config, source_options=options))
        identities.append(
            payload["artifacts"][0]["provenance"]["dispatchRegionProgram"]
        )
    assert identities[0]["intermediateHash"] == identities[1]["intermediateHash"]
    assert identities[0]["settingsHash"] != identities[1]["settingsHash"]


def _generator_with_metadata(root, target, source_text=SOURCE):
    source = root / "loop.metal"
    source.write_text(source_text)
    ast = parse(
        translate(
            source,
            format_output=False,
            source_options={"preserve_resource_origins": True, "threadgroup_size": 32},
        )
    )
    generator = HLSLCodeGen() if target == "directx" else GLSLCodeGen()
    entries = (
        generator.hlsl_software_subgroup_stage_entries(ast)
        if target == "directx"
        else generator.glsl_software_subgroup_stage_entries(ast)
    )
    records = resolve_metadata_preconditions(
        entries[0][1],
        [
            MetadataRangePrecondition(
                entry_point="scan_loop", parameter="axis_size", minimum=0, maximum=65535
            )
        ],
    )
    return generator, ast, records


@pytest.mark.parametrize("target", TARGETS)
def test_helper_provenance_does_not_inherit_entry_bounds(tmp_path, target):
    from crosstl.translator.ast import AttributeNode
    from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
    from crosstl.translator.codegen.GLSL_codegen import OpenGLSoftwareSubgroupError

    helper = """uint collect(constant size_t& limit, uint lane) {
        uint result = 0;
        for (uint j = 0; j < limit; j += 32) {
            result += simd_prefix_inclusive_sum(lane + j);
        }
        return result;
    }
    """
    start = SOURCE.index("    uint result = 0;")
    # The helper receives an unbounded input, so only the copied provenance,
    # never its call, could name the entry's bounded one.
    source = (
        SOURCE[:start]
        + "    output[group.x * 64 + lane + 4] = collect(other, lane);\n}\n"
    )
    source = source.replace(
        "constant size_t& axis_size [[buffer(1)]],",
        "constant size_t& axis_size [[buffer(1)]],\n"
        "                      constant size_t& other [[buffer(2)]],",
    )
    assert "other [[buffer(2)]]" in source
    source = source.replace("kernel void", helper + "kernel void")
    generator, ast, records = _generator_with_metadata(tmp_path, target, source)
    entries = (
        generator.hlsl_software_subgroup_stage_entries(ast)
        if target == "directx"
        else generator.glsl_software_subgroup_stage_entries(ast)
    )
    entry = entries[0][1]
    parameter = next(p for p in entry.parameters if p.name == "axis_size")
    function = next(f for f in ast.functions if f.name == "collect")
    function.parameters[0].attributes = copy.deepcopy(parameter.attributes)
    entry.attributes.append(
        AttributeNode(
            "numthreads", [LiteralNode(n, PrimitiveType("int")) for n in (32, 2, 1)]
        )
    )
    generator.set_software_subgroup_width(32)
    generator.set_metadata_preconditions(records)
    with pytest.raises((DirectXSoftwareSubgroupError, OpenGLSoftwareSubgroupError)):
        generator.generate_program(ast)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("method", ["generate", "generate_program", "generate_stage"])
def test_codegen_entry_paths_preserve_checked_obligations(tmp_path, target, method):
    generator, ast, records = _generator_with_metadata(tmp_path, target)
    generator.set_metadata_preconditions(records)
    arguments = (ast, "compute") if method == "generate_stage" else (ast,)
    generated = getattr(generator, method)(*arguments)
    assert parse_metadata_precondition_header(generated) == records
    assert getattr(generator, method)(*arguments) == generated
    generator.set_metadata_preconditions([])
    assert not parse_metadata_precondition_header(
        getattr(generator, method)(*arguments)
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("fault", ["dtype", "parameterIndex", "backend", "entryPoint"])
def test_codegen_rejects_facts_from_another_source_parameter(tmp_path, target, fault):
    generator, ast, records = _generator_with_metadata(tmp_path, target)
    records = copy.deepcopy(records)
    if fault == "dtype":
        records[0]["dtype"] = "uint32"
    else:
        records[0]["sourceResource"][fault] = {
            "parameterIndex": 0,
            "backend": "cuda",
            "entryPoint": "other",
        }[fault]
    generator.set_metadata_preconditions(records)
    with pytest.raises(MetadataPreconditionError):
        generator.generate_program(ast)


@pytest.mark.parametrize("kind", ["short", "ushort", "int", "uint"])
@pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
def test_interval_proof_checks_every_bound_and_terminating_update(kind, operator):
    unsigned = kind.startswith("u")
    bits = 16 if "short" in kind else 32
    low, high = (0 if unsigned else -(1 << (bits - 1))), (
        1 << (bits - (not unsigned))
    ) - 1
    direction = 1 if operator in ("<", "<=") else -1
    bound = IdentifierNode("limit")
    for start in (low, low + 1, 0, high - 1, high):
        for amount in (1, 2, 7, 32):
            update = AssignmentNode(
                IdentifierNode("j"),
                LiteralNode(amount, PrimitiveType("int")),
                "+=" if direction > 0 else "-=",
            )
            for first in (low, low + 1, 0, high - 7, high):
                last = min(high, first + 7)
                record = dict(
                    dtype=("u" if unsigned else "") + f"int{bits}",
                    minimum=first,
                    maximum=last,
                )
                actual = fixed_stride_loop_terminates(
                    kind,
                    LiteralNode(start, PrimitiveType("int")),
                    bound,
                    update,
                    {},
                    operator,
                    metadata_bounds={"limit": record},
                )
                # Evaluate at most eight concrete bounds, not the proof's endpoints alone.
                expected = True
                for limit in range(first, last + 1):
                    distance = limit - start if direction > 0 else start - limit
                    trips = max(
                        0,
                        (distance + (operator in ("<=", ">=")) + amount - 1) // amount,
                    )
                    final = start + trips * direction * amount
                    expected &= low <= final <= high
                # Unsuffixed source literals outside signed32 are intentionally unproved.
                if start > 2147483647:
                    expected = False
                assert actual == expected, (kind, operator, start, amount, first, last)
