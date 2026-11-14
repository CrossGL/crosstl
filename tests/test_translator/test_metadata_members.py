"""Bind record metadata requirements to the exact reflected record fields.

MLX passes shapes such as ``MLXConvParams<2>`` as one constant record and
loops to its fields (``for (bc = 0; bc < params.C; bc += 32)``). A member
precondition names one integer field of such a record; it travels with the
artifact, is located at the field's reflected offset in every target's layout
(a Metal record buffer, an OpenGL std140 block or a DirectX constant buffer)
and is checked against the uploaded words before dispatch.
"""

import copy
import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    inspect_runtime_package,
    pack_storage_records,
    pack_uniform_block,
    translate_project,
)
from crosstl.project.metadata_preconditions import check_native_metadata_snapshots
from crosstl.project.native_loader_abi import (
    NativeLoaderABIError,
    generate_native_loader_execution_abi,
)
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.runtime_verification import (
    NativeRuntimeParityAdapter,
    RuntimeExecutionState,
)
from crosstl.translator.metadata_preconditions import (
    checked_metadata_preconditions,
    metadata_member_offset,
    parse_metadata_precondition_header,
    parse_metadata_preconditions,
)
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metadata_loop_bounds import _loop_config
from tests.test_translator.test_metadata_preconditions import TARGETS, _package
from tests.test_translator.test_project_metadata_entry_scope import _packages

REQUIRE_ENV = "CROSTL_REQUIRE_METADATA_PRECONDITIONS"
SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Shape {
    int N;
    int C;
    uint O;
    int pads[2];
    float scale;
};
kernel void read_members(constant Shape& shape [[buffer(0)]],
                         device int* results [[buffer(1)]],
                         uint lane [[thread_position_in_grid]]) {
    if (lane == 0) results[0] = shape.C;
    if (lane == 1) results[1] = int(shape.O);
}
"""
CONDITIONS = [
    dict(
        entryPoint="read_members", parameter="shape", member="C", minimum=1, maximum=64
    ),
    dict(
        entryPoint="read_members", parameter="shape", member="O", minimum=0, maximum=8
    ),
]
RECORD = dict(N=2, C=48, O=5, pads=[3, 4], scale=1.5)
LAYOUTS = {
    "metal": "metal-buffer",
    "opengl": "std140",
    "directx": "hlsl-constant-buffer",
}
GUARD = 123456


@pytest.fixture(scope="module")
def packages(tmp_path_factory):
    """One packaged artifact per target, copied for every test that edits it."""
    built = {}

    def build(target):
        if target not in built:
            built[target] = _package(
                tmp_path_factory.mktemp(f"members-{target}"),
                target,
                source=SOURCE,
                conditions=CONDITIONS,
            )
        descriptor, package = built[target]
        return copy.deepcopy(descriptor), package

    return build


def _binding(descriptor):
    return next(
        item
        for item in descriptor["bindings"]
        if "metadataMemberPreconditions" in item["provenance"]
    )


def _fields(layout):
    return layout["structMembers" if "structMembers" in layout else "blockMembers"]


def _record_words(layout, record):
    """The record's words in the binding's own reflected layout."""
    if layout["storageLayout"] == "metal-buffer":
        return pack_storage_records(layout, [record])
    return pack_uniform_block(layout, record)


def _values(descriptor, record=None):
    record = RECORD if record is None else record
    inputs, outputs = {}, {}
    checked = _binding(descriptor)
    for binding in descriptor["bindings"]:
        if binding is checked:
            words = _record_words(binding["scalarLayout"], record)
            inputs[binding["name"]] = dict(
                dtype="uint32", shape=[len(words)], values=words
            )
            continue
        inputs[binding["name"]] = dict(dtype="int32", shape=[8], values=[GUARD] * 8)
        outputs[binding["name"]] = dict(
            dtype="int32",
            shape=[8],
            values=[record["C"], record["O"]] + [GUARD] * 6,
        )
    return inputs, outputs


def _request(descriptor, package, record=None, *, values=None):
    inputs, outputs = values or _values(descriptor, record)
    return build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
        observe_inputs=[_binding(descriptor)["name"]],
    )


def _buffers(request, target):
    adapter = NativeRuntimeParityAdapter()
    adapter.target = target
    return adapter._native_buffer_bindings(
        RuntimeExecutionState(request=request, plan=request.execution_plan)
    )


@pytest.mark.parametrize("target", TARGETS)
def test_member_requirements_survive_package_and_snapshot(packages, target):
    descriptor, package = packages(target)
    binding = _binding(descriptor)
    provenance = binding["provenance"]
    records = provenance["metadataMemberPreconditions"]
    assert [(item["member"], item["dtype"]) for item in records] == [
        ("C", "int32"),
        ("O", "uint32"),
    ]
    assert not {"metadataPrecondition", "metadataElementPreconditions"} & set(
        provenance
    )
    layout = binding["scalarLayout"]
    assert layout["storageLayout"] == LAYOUTS[target]
    # MLX-style records keep their leading int fields at bytes 0, 4 and 8.
    assert [metadata_member_offset(layout, item) for item in records] == [4, 8]
    artifact = package / descriptor["artifact"]["packagePath"]
    assert parse_metadata_precondition_header(artifact.read_text()) == records
    assert inspect_runtime_package(package / "runtime-package.json")["success"]
    receipts = check_native_metadata_snapshots(
        artifact, target, _buffers(_request(descriptor, package), target)
    )
    assert [(row["requirement"]["member"], row["value"]) for row in receipts] == [
        ("C", 48),
        ("O", 5),
    ]
    with pytest.raises(NativeLoaderABIError, match="metadata"):
        generate_native_loader_execution_abi(descriptor)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "fault",
    [
        "misaligned",
        "outside",
        "type",
        "array",
        "vector",
        "encoded",
        "missing",
        "duplicate",
        "member-view",
        "record-offset",
        "words",
    ],
)
def test_member_offsets_require_one_reflected_integer_field(packages, target, fault):
    descriptor, _ = packages(target)
    binding = _binding(descriptor)
    layout = binding["scalarLayout"]
    requirement = binding["provenance"]["metadataMemberPreconditions"][0]
    assert metadata_member_offset(layout, requirement) == 4
    fields = _fields(layout)
    field = next(item for item in fields if item["name"] == "C")
    if fault == "misaligned":
        field["offsetBytes"] += 2
    elif fault == "outside":
        field["offsetBytes"] = 4096
    elif fault == "type":
        field["elementType"] = "uint32"
    elif fault == "array":
        field.update(arrayLength=1, arrayStrideBytes=16)
    elif fault == "vector":
        field["vectorWidth"] = 2
    elif fault == "encoded":
        layout["storageEncoding"] = {"encoding": "float16-bits"}
    elif fault == "missing":
        fields.remove(field)
    elif fault == "duplicate":
        fields.append(dict(field))
    elif fault == "member-view":
        layout["memberName"] = "shape"
    elif fault == "record-offset":
        layout["memberOffsetBytes"] = 4
    else:
        layout["payloadEncoding"] = "float32"
    with pytest.raises(ValueError):
        metadata_member_offset(layout, requirement)


@pytest.mark.parametrize("member", ["", "1C", "C.D", "C ", "params->C", 3, True])
def test_invalid_member_selectors_are_rejected(member):
    with pytest.raises(ValueError):
        parse_metadata_preconditions([{**CONDITIONS[0], "member": member}])


def test_explicit_null_member_and_element_with_member_are_rejected():
    with pytest.raises(ValueError):
        parse_metadata_preconditions([{**CONDITIONS[0], "member": None}])
    with pytest.raises(ValueError):
        parse_metadata_preconditions([{**CONDITIONS[0], "element": 0}])


@pytest.mark.parametrize("other", ["same-member", "scalar", "element"])
def test_member_requirements_cannot_overlap_other_selectors(other):
    second = dict(CONDITIONS[0])
    if other != "same-member":
        second.pop("member")
    if other == "element":
        second["element"] = 0
    with pytest.raises(ValueError):
        parse_metadata_preconditions([CONDITIONS[0], second])
    parsed = parse_metadata_preconditions(CONDITIONS)
    assert [item.member for item in parsed] == ["C", "O"]
    assert [item.to_json()["member"] for item in parsed] == ["C", "O"]


def test_resolved_members_keep_their_own_field_types():
    source = {
        "backend": "metal",
        "entryPoint": "read_members",
        "parameter": "shape",
        "parameterIndex": 0,
        "schemaVersion": 1,
    }
    # Unlike array elements, two fields of one record may differ in type.
    records = [
        dict(sourceResource=source, dtype="uint32", member="O", minimum=0, maximum=8),
        dict(sourceResource=source, dtype="int32", member="C", minimum=1, maximum=64),
    ]
    assert [item["member"] for item in checked_metadata_preconditions(records)] == [
        "C",
        "O",
    ]
    for fault in ("duplicate", "element", "scalar", "null"):
        changed = copy.deepcopy(records)
        if fault == "duplicate":
            changed[0]["member"] = "C"
        elif fault == "element":
            changed[0].pop("member")
            changed[0]["element"] = 1
        elif fault == "scalar":
            changed[0].pop("member")
        else:
            changed[0]["member"] = None
        with pytest.raises(ValueError):
            checked_metadata_preconditions(changed)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "fault", ["low", "negative", "high", "unsigned-high", "dtype", "truncated"]
)
def test_bad_member_values_fail_before_dispatch(packages, target, fault):
    descriptor, package = packages(target)
    record = dict(RECORD)
    if fault in {"low", "negative", "high"}:
        record["C"] = {"low": 0, "negative": -1, "high": 65}[fault]
    elif fault == "unsigned-high":
        record["O"] = 9
    inputs, outputs = _values(descriptor, record)
    name = _binding(descriptor)["name"]
    if fault == "dtype":
        inputs[name] = {**inputs[name], "dtype": "int32"}
    elif fault == "truncated":
        inputs[name] = {
            **inputs[name],
            "shape": [1],
            "values": inputs[name]["values"][:1],
        }
    with pytest.raises(NativeLoaderDispatchError):
        _request(descriptor, package, values=(inputs, outputs))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "fault", ["device", "volatile", "float", "array", "missing", "scalar"]
)
def test_member_contract_requires_readonly_integer_field(tmp_path, target, fault):
    from crosstl.project import ProjectConfig

    source, conditions = SOURCE, copy.deepcopy(CONDITIONS)
    if fault == "device":
        source = source.replace("constant Shape&", "device Shape&")
    elif fault == "volatile":
        source = source.replace("constant Shape&", "constant volatile Shape&")
    elif fault == "float":
        conditions[0]["member"] = "scale"
    elif fault == "array":
        conditions[0]["member"] = "pads"
    elif fault == "missing":
        conditions[0]["member"] = "H"
    else:
        # A scalar reference has no fields to select.
        source = source.replace("constant Shape& shape", "constant int& shape")
        source = source.replace("shape.C", "shape").replace("int(shape.O)", "shape")
    (tmp_path / "members.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            metadata_preconditions=conditions,
            workgroup_size=(4, 1, 1),
            source_options={"metal": {"preserve_resource_origins": True}},
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(item["severity"] == "error" for item in report["diagnostics"])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "fault", ["remove", "member", "range", "duplicate", "layout", "selector"]
)
def test_descriptor_cannot_reassign_member_obligations(packages, target, fault):
    descriptor, package = packages(target)
    values = _values(descriptor)
    binding = _binding(descriptor)
    records = binding["provenance"]["metadataMemberPreconditions"]
    if fault == "remove":
        binding["provenance"].pop("metadataMemberPreconditions")
    elif fault == "member":
        records[1]["member"] = "N"
    elif fault == "range":
        records[1]["maximum"] = 99
    elif fault == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif fault == "layout":
        # Point the checked field at N, which holds 2, not C.
        next(item for item in _fields(binding["scalarLayout"]) if item["name"] == "C")[
            "offsetBytes"
        ] = 0
    else:
        records[1].pop("member")
        records[1]["element"] = 2
    with pytest.raises(NativeLoaderDispatchError):
        build_native_loader_dispatch_request(
            descriptor,
            package,
            *values,
            {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
        )


@pytest.mark.parametrize("target", TARGETS)
def test_member_checks_read_the_owned_upload_snapshot(packages, target):
    descriptor, package = packages(target)
    name = _binding(descriptor)["name"]
    inputs, outputs = _values(descriptor)
    request = _request(descriptor, package, values=(inputs, outputs))
    buffers = _buffers(request, target)
    path = package / descriptor["artifact"]["packagePath"]
    # Later edits to the caller's words do not reach the owned snapshot.
    inputs[name]["values"][1] = 99
    assert [
        row["value"] for row in check_native_metadata_snapshots(path, target, buffers)
    ] == [48, 5]
    key = next(key for key, item in buffers.items() if item.binding.name == name)
    native = buffers[key]
    payload = native.upload_snapshot.payload
    altered = payload[:4] + (99).to_bytes(4, "little", signed=True) + payload[8:]
    buffers[key] = replace(
        native, upload_snapshot=replace(native.upload_snapshot, payload=altered)
    )
    with pytest.raises(ValueError, match="outside"):
        check_native_metadata_snapshots(path, target, buffers)


@pytest.mark.parametrize(
    "record",
    [dict(RECORD), dict(RECORD, C=1, O=0), dict(RECORD, C=64, O=8, N=-7)],
)
def test_record_members_execute_natively(tmp_path, record):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native metadata checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    descriptor, package = _package(
        tmp_path, target, source=SOURCE, conditions=CONDITIONS
    )
    request = _request(descriptor, package, record)
    _, expected = _values(descriptor, record)
    work = tmp_path / "native"
    work.mkdir()
    _execute(
        request,
        expected,
        work,
        validate=_validate,
        original_source=SOURCE,
        original_entry="read_members",
    )
    details = json.loads((work / "evidence.json").read_text())["records"]["generated"][
        "details"
    ]
    assert [
        (row["requirement"]["member"], row["value"])
        for row in details["metadataPreconditions"]
    ] == [("C", record["C"]), ("O", record["O"])]
    assert all(row["unchanged"] for row in details["inputObservations"].values())


# ---------------------------------------------------------------------------
# A checked record field bounds a software-subgroup collective loop.
# ---------------------------------------------------------------------------

LOOP_SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Extent {
    int N;
    uint count;
};
kernel void scan_loop(device uint* output [[buffer(0)]],
                      constant Extent& extent [[buffer(1)]],
                      uint lane [[thread_index_in_threadgroup]],
                      uint3 group [[threadgroup_position_in_grid]]) {
    uint result = 0;
    for (uint j = 0; j < extent.count; j += 32) {
        result += simd_prefix_inclusive_sum(lane + j);
    }
    output[group.x * 64 + lane + 4] = result;
}
"""
LOOP_CONDITION = dict(
    entryPoint="scan_loop",
    parameter="extent",
    member="count",
    minimum=0,
    maximum=65535,
)
LOOP_TARGETS = ("directx", "opengl")
LOOP_GUARD = 0xABCDEF01


def _loop_request(descriptor, package, count):
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        origin = binding.get("provenance", {}).get("sourceResource")
        if origin is None:
            continue
        name = binding["name"]
        if origin["parameter"] == "extent":
            words = _record_words(binding["scalarLayout"], dict(N=-1, count=count))
            inputs[name] = dict(dtype="uint32", shape=[len(words)], values=words)
            observed.append(name)
            continue
        inputs[name] = dict(dtype="uint32", shape=[136], values=[LOOP_GUARD] * 136)
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
            dtype="uint32",
            shape=[136],
            values=[LOOP_GUARD] * 4 + result + [LOOP_GUARD] * 4,
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


@pytest.mark.parametrize("target", LOOP_TARGETS)
def test_member_bound_retains_loop_and_package_contract(tmp_path, target):
    _, package, (descriptor,) = _packages(
        _loop_config(tmp_path, target, source=LOOP_SOURCE, conditions=(LOOP_CONDITION,))
    )
    artifact = (package / descriptor["artifact"]["packagePath"]).read_text()
    (record,) = parse_metadata_precondition_header(artifact)
    assert (record["member"], record["dtype"], record["maximum"]) == (
        "count",
        "uint32",
        65535,
    )
    loop = next(
        line for line in artifact.splitlines() if "for (" in line and "count" in line
    )
    assert "extent" in loop
    for count in (0, 1, 31, 32, 33, 64, 65, 65535):
        request, _ = _loop_request(descriptor, package, count)
        assert not request.execution_plan.diagnostics
    with pytest.raises(NativeLoaderDispatchError, match="outside"):
        _loop_request(descriptor, package, 65536)


@pytest.mark.parametrize("target", LOOP_TARGETS)
@pytest.mark.parametrize(
    "change",
    [
        "uncontracted",
        "other-member",
        "offset-bound",
        "cast-bound",
        "overflow",
        "written-record",
        "scalar-requirement",
    ],
)
def test_member_bound_proves_only_its_own_field(tmp_path, target, change):
    source, conditions = LOOP_SOURCE, [dict(LOOP_CONDITION)]
    if change == "uncontracted":
        conditions = []
    elif change == "other-member":
        conditions[0]["member"] = "N"
        conditions[0]["minimum"] = 0
    elif change == "offset-bound":
        source = source.replace("j < extent.count", "j < extent.count + 1")
    elif change == "cast-bound":
        source = source.replace("j < extent.count", "j < uint(extent.N)")
        conditions[0]["member"] = "N"
    elif change == "overflow":
        conditions[0]["maximum"] = 4294967295
    elif change == "written-record":
        source = source.replace("constant Extent&", "device Extent&").replace(
            "uint result = 0;", "uint result = 0; extent.count = lane;"
        )
    else:
        conditions[0].pop("member")
    payload = translate_project(
        _loop_config(tmp_path, target, source=source, conditions=tuple(conditions)),
        format_output=False,
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert any(item["severity"] == "error" for item in payload["diagnostics"])


@pytest.mark.parametrize("count", [33, 64])
def test_member_bound_loop_executes_natively(tmp_path, count):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native metadata checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    _, package, (descriptor,) = _packages(
        _loop_config(tmp_path, target, source=LOOP_SOURCE, conditions=(LOOP_CONDITION,))
    )
    request, expected = _loop_request(descriptor, package, count)
    work = tmp_path / "native"
    work.mkdir()
    _execute(
        request,
        expected,
        work,
        validate=_validate,
        original_source=LOOP_SOURCE,
        original_entry="scan_loop",
    )
    details = json.loads((work / "evidence.json").read_text())["records"]["generated"][
        "details"
    ]
    assert [
        (row["requirement"]["member"], row["value"])
        for row in details["metadataPreconditions"]
    ] == [("count", count)]


def test_ci_requires_record_member_execution():
    from pathlib import Path

    from tests import workflow_text
    from tests.ci_helpers import assert_paths_covered

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert "tests/test_translator/test_metadata_members.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_metadata_members.py",
        )
