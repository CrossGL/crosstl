"""Bind array metadata requirements to the exact reflected upload elements."""

import copy
import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    inspect_runtime_package,
)
from crosstl.project.metadata_preconditions import check_native_metadata_snapshots
from crosstl.project.native_loader_abi import (
    NativeLoaderABIError,
    generate_native_loader_execution_abi,
)
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.runtime_verification import (
    NativeRuntimeParityAdapter,
    RuntimeAllocationView,
    RuntimeExecutionState,
    RuntimeValue,
)
from crosstl.translator.metadata_preconditions import parse_metadata_preconditions
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metadata_preconditions import TARGETS, _package

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void read_metadata(constant long* metadata [[buffer(0)]],
                          device int* results [[buffer(1)]],
                          uint lane [[thread_position_in_grid]]) {
    if (lane == 0) results[0] = int(metadata[0]);
    if (lane == 1) results[1] = int(metadata[5]);
}
"""
CONDITIONS = [
    dict(
        entryPoint="read_metadata",
        parameter="metadata",
        element=0,
        minimum=-2,
        maximum=-1,
    ),
    dict(
        entryPoint="read_metadata",
        parameter="metadata",
        element=5,
        minimum=7,
        maximum=9,
    ),
]
GUARD = 123456


def _binding(descriptor):
    return next(
        item
        for item in descriptor["bindings"]
        if "metadataElementPreconditions" in item["provenance"]
    )


def _values(descriptor, words=None):
    words = [-2, 1 << 40, 0, 0, 0, 8] if words is None else words
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        metadata = binding is _binding(descriptor)
        values = words if metadata else [GUARD] * 8
        dtype = "int64" if metadata else "int32"
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(values)], values=values)
        if not metadata:
            outputs[binding["name"]] = dict(
                dtype=dtype,
                shape=[8],
                values=[words[0], words[5] if len(words) > 5 else 0] + [GUARD] * 6,
            )
    return inputs, outputs


def _request(descriptor, package, words=None, *, values=None):
    inputs, outputs = values or _values(descriptor, words)
    return build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
        observe_inputs=[_binding(descriptor)["name"]],
    )


@pytest.mark.parametrize("target", TARGETS)
def test_array_requirements_survive_package_and_snapshot(tmp_path, target):
    descriptor, package = _package(
        tmp_path, target, source=SOURCE, conditions=CONDITIONS
    )
    records = _binding(descriptor)["provenance"]["metadataElementPreconditions"]
    assert [item["element"] for item in records] == [0, 5]
    assert {item["dtype"] for item in records} == {"int64"}
    assert inspect_runtime_package(package / "runtime-package.json")["success"]
    request = _request(descriptor, package)
    adapter = NativeRuntimeParityAdapter()
    adapter.target = target
    buffers = adapter._native_buffer_bindings(
        RuntimeExecutionState(request=request, plan=request.execution_plan)
    )
    receipts = check_native_metadata_snapshots(
        package / descriptor["artifact"]["packagePath"], target, buffers
    )
    assert [(row["requirement"]["element"], row["value"]) for row in receipts] == [
        (0, -2),
        (5, 8),
    ]
    with pytest.raises(NativeLoaderABIError, match="metadata"):
        generate_native_loader_execution_abi(descriptor)


@pytest.mark.parametrize("element", [-1, True, 1.5, None, 1 << 61])
def test_invalid_element_selectors_are_rejected(element):
    with pytest.raises(ValueError):
        parse_metadata_preconditions([{**CONDITIONS[0], "element": element}])


@pytest.mark.parametrize("mixed", [False, True])
def test_scalar_and_element_requirements_cannot_overlap(mixed):
    duplicate = dict(CONDITIONS[0])
    if mixed:
        duplicate.pop("element")
    with pytest.raises(ValueError):
        parse_metadata_preconditions([CONDITIONS[0], duplicate])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "words", [[-3, 0, 0, 0, 0, 8], [-2, 0, 0, 0, 0, 10], [-2, 0, 0, 0, 0]]
)
def test_bad_metadata_values_fail_before_dispatch(tmp_path, target, words):
    descriptor, package = _package(
        tmp_path, target, source=SOURCE, conditions=CONDITIONS
    )
    with pytest.raises(NativeLoaderDispatchError):
        _request(descriptor, package, words)


@pytest.mark.parametrize("target", TARGETS)
def test_checked_array_cannot_alias_another_binding(tmp_path, target):
    descriptor, package = _package(
        tmp_path, target, source=SOURCE, conditions=CONDITIONS
    )
    inputs, outputs = _values(descriptor)
    for name, value in list(inputs.items()):
        width = 8 if value["dtype"] == "int64" else 4
        size = width * len(value["values"])
        inputs[name] = RuntimeValue(
            name=name,
            **value,
            allocation=RuntimeAllocationView("shared-metadata", 0, size, 64),
        )
    with pytest.raises(NativeLoaderDispatchError, match="allocation"):
        _request(descriptor, package, values=(inputs, outputs))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("declaration", ["device long*", "constant float*"])
def test_element_contract_requires_readonly_integer_source(
    tmp_path, target, declaration
):
    from crosstl.project import ProjectConfig, translate_project

    (tmp_path / "metadata.metal").write_text(
        SOURCE.replace("constant long*", declaration)
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            metadata_preconditions=CONDITIONS,
            workgroup_size=(4, 1, 1),
            source_options={"metal": {"preserve_resource_origins": True}},
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(item["severity"] == "error" for item in report["diagnostics"])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("fault", ["remove", "element", "range", "duplicate", "layout"])
def test_descriptor_cannot_reassign_element_obligations(tmp_path, target, fault):
    descriptor, package = _package(
        tmp_path, target, source=SOURCE, conditions=CONDITIONS
    )
    values = _values(descriptor)
    binding = _binding(descriptor)
    if fault == "remove":
        binding["provenance"].pop("metadataElementPreconditions")
    elif fault == "layout":
        binding["scalarLayout"]["memberOffsetBytes"] = 8
    elif fault == "duplicate":
        binding["provenance"]["metadataElementPreconditions"].append(
            copy.deepcopy(binding["provenance"]["metadataElementPreconditions"][0])
        )
    else:
        binding["provenance"]["metadataElementPreconditions"][1][
            "element" if fault == "element" else "maximum"
        ] = 4 if fault == "element" else 99
    with pytest.raises(NativeLoaderDispatchError):
        build_native_loader_dispatch_request(
            descriptor,
            package,
            *values,
            {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
        )


@pytest.mark.parametrize("target", TARGETS)
def test_array_checks_use_offset_view_and_owned_snapshot(tmp_path, target):
    descriptor, package = _package(
        tmp_path, target, source=SOURCE, conditions=CONDITIONS
    )
    name = _binding(descriptor)["name"]
    inputs, outputs = _values(descriptor)
    inputs[name] = RuntimeValue(
        name=name,
        **inputs[name],
        allocation=RuntimeAllocationView("metadata-view", 16, 48, 64),
    )
    request = _request(descriptor, package, values=(inputs, outputs))
    adapter = NativeRuntimeParityAdapter()
    adapter.target = target
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    buffers = adapter._native_buffer_bindings(state)
    path = package / descriptor["artifact"]["packagePath"]
    inputs[name].values[5] = 99
    assert [
        row["value"] for row in check_native_metadata_snapshots(path, target, buffers)
    ] == [-2, 8]
    key = next(key for key, item in buffers.items() if item.binding.name == name)
    native = buffers[key]
    altered = native.upload_snapshot.payload[:40] + (99).to_bytes(
        8, "little", signed=True
    )
    buffers[key] = replace(
        native, upload_snapshot=replace(native.upload_snapshot, payload=altered)
    )
    with pytest.raises(ValueError, match="outside"):
        check_native_metadata_snapshots(path, target, buffers)


@pytest.mark.parametrize(
    "words", [[-2, 1 << 40, 0, 0, 0, 7], [-1, 1 << 40, 0, 0, 0, 9]]
)
def test_array_metadata_executes_natively(tmp_path, words):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    descriptor, package = _package(
        tmp_path, target, source=SOURCE, conditions=CONDITIONS
    )
    request = _request(descriptor, package, words)
    _, expected = _values(descriptor, words)
    work = tmp_path / "native"
    work.mkdir()
    _execute(
        request,
        expected,
        work,
        validate=_validate,
        original_source=SOURCE,
        original_entry="read_metadata",
    )
    evidence = json.loads((work / "evidence.json").read_text())
    details = evidence["records"]["generated"]["details"]
    assert [row["value"] for row in details["metadataPreconditions"]] == [
        words[0],
        words[5],
    ]
    assert all(row["unchanged"] for row in details["inputObservations"].values())


INDEX_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void indexed_copy(constant ulong* indices [[buffer(0)]],
                         const device uint* values [[buffer(1)]],
                         device uint* results [[buffer(2)]],
                         uint lane [[thread_position_in_grid]]) {
    if (lane < 2) results[lane] = values[indices[lane]];
}
"""
INDEX_CONDITIONS = [
    dict(
        entryPoint="indexed_copy", parameter="indices", element=0, minimum=1, maximum=2
    ),
    dict(
        entryPoint="indexed_copy", parameter="indices", element=1, minimum=4, maximum=5
    ),
]


@pytest.mark.parametrize("target", TARGETS)
def test_checked_elements_bound_dynamic_index_reads(tmp_path, target):
    descriptor, package = _package(
        tmp_path, target, source=INDEX_SOURCE, conditions=INDEX_CONDITIONS
    )
    assert inspect_runtime_package(package / "runtime-package.json")["success"]
    assert len(_binding(descriptor)["provenance"]["metadataElementPreconditions"]) == 2


@pytest.mark.parametrize("indices", [[1, 5], [2, 4]])
def test_checked_element_indices_execute_natively(tmp_path, indices):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.test_translator.test_boolean_buffer_runtime import _bound_values

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    descriptor, package = _package(
        tmp_path, target, source=INDEX_SOURCE, conditions=INDEX_CONDITIONS
    )
    inputs = _bound_values(
        descriptor,
        {
            "indices": dict(dtype="uint64", shape=[2], values=indices),
            "values": dict(
                dtype="uint32", shape=[6], values=[101, 103, 107, 109, 113, 127]
            ),
            "results": dict(dtype="uint32", shape=[8], values=[GUARD] * 8),
        },
    )
    outputs = _bound_values(
        descriptor,
        {
            "results": dict(
                dtype="uint32",
                shape=[8],
                values=[[101, 103, 107, 109, 113, 127][index] for index in indices]
                + [GUARD] * 6,
            )
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
        observe_inputs=[
            item["name"] for item in descriptor["bindings"] if item["access"] == "read"
        ],
    )
    work = tmp_path / "native"
    work.mkdir()
    _execute(
        request,
        outputs,
        work,
        validate=_validate,
        original_source=INDEX_SOURCE,
        original_entry="indexed_copy",
    )
    details = json.loads((work / "evidence.json").read_text())["records"]["generated"][
        "details"
    ]
    assert [
        (item["requirement"]["element"], item["value"])
        for item in details["metadataPreconditions"]
    ] == list(enumerate(indices))
    assert len(details["inputObservations"]) == 2
    assert all(item["unchanged"] for item in details["inputObservations"].values())


@pytest.mark.parametrize(
    "fault",
    ["missing", "negative", "wide", "shadow", "rebased", "volatile", "asserted"],
)
def test_array_index_proofs_reject_uncovered_or_invalid_values(tmp_path, fault):
    from crosstl.project import ProjectConfig, translate_project

    conditions = copy.deepcopy(INDEX_CONDITIONS)
    source = INDEX_SOURCE
    if fault in {"missing", "asserted"}:
        conditions.pop()
        if fault == "asserted":
            source = source.replace("indices[lane]", "indices[lane + 0]")
    elif fault == "negative":
        source = source.replace("ulong*", "long*")
        conditions[1].update(minimum=-3, maximum=5)
    elif fault == "wide":
        conditions[1]["maximum"] = 1 << 40
    elif fault == "shadow":
        source = source.replace(
            "if (lane < 2) results[lane] = values[indices[lane]];",
            "{ ulong indices[2] = {0, 1099511627776ul}; if (lane < 2) results[lane] = values[indices[lane]]; }",
        )
    elif fault == "volatile":
        source = source.replace("constant ulong*", "constant volatile ulong*")
    else:
        source = source.replace("if (lane < 2)", "indices += 1; if (lane < 2)")
    (tmp_path / "index.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=("opengl",),
            metadata_preconditions=conditions,
            workgroup_size=(4, 1, 1),
            source_options={"metal": {"preserve_resource_origins": True}},
            index_range_assertions=(
                [dict(expression="lane", minimum=0, maximum=0)]
                if fault == "asserted"
                else []
            ),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(item["severity"] == "error" for item in report["diagnostics"])


def test_ci_requires_array_metadata_execution():
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
    assert "test_metadata_elements.py" in step
    assert 'CROSTL_REQUIRE_METADATA_PRECONDITIONS: "1"' in step
    assert "continue-on-error" not in step
    assert "--timeout-seconds" in step and "--junitxml" in step and "-n auto" in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_metadata_elements.py",
        )
