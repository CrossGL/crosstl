"""Source metadata obligations follow project entries, not neighboring kernels."""

import json
from dataclasses import replace

import pytest

from crosstl.project import (
    DISPATCH_CONTRACT_KIND,
    DISPATCH_CONTRACT_SCHEMA_VERSION,
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    inspect_runtime_package,
    translate_project,
    validate_project_report,
)
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.translation_checkpoint import ProjectTranslationCheckpointError
from crosstl.translator.metadata_preconditions import parse_metadata_precondition_header

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"
FIRST = """kernel void first(constant uint& count [[buffer(0)]],
                         device uint* output [[buffer(1)]],
                         uint lane [[thread_position_in_grid]]) {
    if (lane < count) { output[lane] = 100u + count + lane; }
}
"""
SECOND = (
    FIRST.replace("first", "second").replace("count", "length").replace("100u", "200u")
)
SOURCE = HEADER + FIRST + SECOND
TEMPLATE_SOURCE = HEADER + """template<typename T>
kernel void fill(constant uint& count [[buffer(0)]], device T* output [[buffer(1)]], uint lane [[thread_position_in_grid]]) {
    if (lane < count) { output[lane] = T(count + lane); }
}
template [[host_name("first")]] [[kernel]] decltype(fill<uint>) fill<uint>;
template [[host_name("second")]] [[kernel]] decltype(fill<int>) fill<int>;
"""
CONDITIONS = (
    dict(entryPoint="first", parameter="count", minimum=1, maximum=4),
    dict(entryPoint="second", parameter="length", minimum=1, maximum=8),
)
TARGETS = ("metal", "directx", "opengl")
GUARD = 0xABCDEF01


def _config(
    tmp_path,
    target,
    *,
    conditions=CONDITIONS,
    entries=("first", "second"),
    source=SOURCE,
    **kwargs,
):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "multi.metal").write_text(source)
    return ProjectConfig(
        root=tmp_path,
        include_patterns=("*.metal",),
        targets=(target,),
        workgroup_size=(4, 1, 1),
        metadata_preconditions=conditions,
        entry_points={} if entries is None else {"multi.metal": entries},
        source_options={"metal": {"preserve_resource_origins": True}},
        **kwargs,
    )


def _packages(config, *, workers=1, checkpoint=None):
    report = translate_project(
        config, format_output=False, max_workers=workers, checkpoint_path=checkpoint
    )
    payload = report.to_json()
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    assert not [item for item in payload["diagnostics"] if item["severity"] == "error"]
    report_path = config.root / "report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"]
    manifest = build_runtime_artifact_manifest(report_path)
    assert manifest["success"], manifest
    manifest_path = config.root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    package = config.root / "package"
    result = build_runtime_package(manifest_path, package)
    assert result["success"], result
    package_path = package / "runtime-package.json"
    assert inspect_runtime_package(package_path)["success"]
    loader = build_runtime_loader_manifest(package_path)
    assert loader["success"], loader
    descriptors = [
        build_native_loader_abi_descriptor(loader, load_unit_id=unit["id"])
        for unit in loader["loadUnits"]
    ]
    return payload, package, descriptors


def _entry(descriptor):
    entries = {
        binding["provenance"]["sourceResource"]["entryPoint"]
        for binding in descriptor["bindings"]
    }
    assert len(entries) == 1
    return entries.pop()


def _request(descriptor, package, count):
    entry = _entry(descriptor)
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        values = [GUARD] * 8 if parameter == "output" else [count]
        inputs[binding["name"]] = dict(
            dtype="uint32", shape=[len(values)], values=values
        )
        if parameter == "output":
            bias = 100 if entry == "first" else 200
            outputs[binding["name"]] = dict(
                dtype="uint32",
                shape=[8],
                values=[
                    bias + count + lane if lane < min(count, 4) else GUARD
                    for lane in range(8)
                ],
            )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
    )
    return request, outputs


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("conditions", [CONDITIONS, CONDITIONS[:1], CONDITIONS[1:], ()])
def test_each_selected_entry_retains_only_its_metadata(tmp_path, target, conditions):
    payload, package, descriptors = _packages(
        _config(tmp_path, target, conditions=conditions)
    )
    assert payload["summary"]["translatedCount"] == 2
    assert len(descriptors) == 2
    for descriptor in descriptors:
        entry = _entry(descriptor)
        declared = [item for item in conditions if item["entryPoint"] == entry]
        records = parse_metadata_precondition_header(
            (package / descriptor["artifact"]["packagePath"]).read_text()
        )
        assert len(records) == len(declared)
        for record, expected in zip(records, declared):
            assert record["sourceResource"]["entryPoint"] == entry
            assert record["sourceResource"]["parameter"] == expected["parameter"]
            assert (record["minimum"], record["maximum"]) == (1, expected["maximum"])
            _request(descriptor, package, expected["maximum"])
            with pytest.raises(NativeLoaderDispatchError, match="outside"):
                _request(descriptor, package, expected["maximum"] + 1)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", ["first", "second"])
def test_unselected_known_entry_does_not_block_translation(tmp_path, target, entry):
    payload, _package, descriptors = _packages(
        _config(tmp_path, target, entries=(entry,))
    )
    assert payload["summary"]["translatedCount"] == 1
    assert [_entry(item) for item in descriptors] == [entry]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("workers", [1, 2])
def test_discovered_entry_jobs_scope_metadata(tmp_path, target, workers):
    config = _config(
        tmp_path, target, entries=None, translate_discovered_entry_points=("*.metal",)
    )
    payload, _package, descriptors = _packages(config, workers=workers)
    assert payload["summary"]["translatedCount"] == 2
    assert {_entry(item) for item in descriptors} == {"first", "second"}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("workers", [1, 2])
def test_source_glob_does_not_apply_metadata_to_another_files_entry(
    tmp_path, target, workers
):
    config = _config(
        tmp_path, target, conditions=CONDITIONS[:1], entries=None, source=HEADER + FIRST
    )
    (tmp_path / "second.metal").write_text(HEADER + SECOND)
    payload, package, descriptors = _packages(config, workers=workers)
    assert payload["summary"]["translatedCount"] == 2
    for item in descriptors:
        records = parse_metadata_precondition_header(
            (package / item["artifact"]["packagePath"]).read_text()
        )
        assert len(records) == int(_entry(item) == "first")


@pytest.mark.parametrize("target", TARGETS)
def test_variant_only_entries_are_resolved_before_scoping(tmp_path, target):
    source = HEADER + "#if USE_SECOND\n" + SECOND + "#else\n" + FIRST + "#endif\n"
    config = _config(
        tmp_path,
        target,
        entries=None,
        source=source,
        variants={"first": {"USE_SECOND": "0"}, "second": {"USE_SECOND": "1"}},
    )
    payload, package, descriptors = _packages(config)
    assert payload["summary"]["translatedCount"] == 2
    assert {_entry(item) for item in descriptors} == {"first", "second"}
    for item in descriptors:
        records = parse_metadata_precondition_header(
            (package / item["artifact"]["packagePath"]).read_text()
        )
        assert len(records) == 1
        assert records[0]["sourceResource"]["entryPoint"] == _entry(item)


@pytest.mark.parametrize("target", TARGETS)
def test_template_host_names_remain_source_metadata_identities(tmp_path, target):
    conditions = [dict(item, parameter="count") for item in CONDITIONS]
    payload, package, descriptors = _packages(
        _config(tmp_path, target, source=TEMPLATE_SOURCE, conditions=conditions)
    )
    assert payload["summary"]["translatedCount"] == 2
    for item in descriptors:
        records = parse_metadata_precondition_header(
            (package / item["artifact"]["packagePath"]).read_text()
        )
        assert len(records) == 1
        assert records[0]["sourceResource"]["entryPoint"] == _entry(item)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "field,value,failed", [("entryPoint", "misspelled", 2), ("parameter", "missing", 1)]
)
def test_unresolved_metadata_is_not_silently_dropped(
    tmp_path, target, field, value, failed
):
    conditions = [dict(CONDITIONS[0], **{field: value})]
    payload = translate_project(
        _config(tmp_path, target, conditions=conditions), format_output=False
    ).to_json()
    assert payload["summary"]["failedCount"] == failed
    errors = [item for item in payload["diagnostics"] if item["severity"] == "error"]
    assert len(errors) == failed
    assert all(
        item["code"] == "project.translate.metadata-precondition-unsupported"
        for item in errors
    )


def test_unmatched_metadata_source_is_a_configuration_error(tmp_path):
    conditions = [dict(CONDITIONS[0], source="missing/*.metal")]
    report = translate_project(
        _config(tmp_path, "metal", conditions=conditions), format_output=False
    )
    assert any(
        item["code"] == "project.config.metadata-precondition-pattern-unmatched"
        for item in report.to_json()["diagnostics"]
    )


@pytest.mark.parametrize("target", TARGETS)
def test_missing_resource_provenance_still_rejects_selected_requirements(
    tmp_path, target
):
    config = replace(_config(tmp_path, target), source_options={})
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["failedCount"] == 2


@pytest.mark.parametrize("target", TARGETS)
def test_changed_metadata_bounds_cannot_restore_old_checkpoint(tmp_path, target):
    config = _config(tmp_path, target)
    checkpoint = tmp_path / "checkpoint.json"
    _packages(config, checkpoint=checkpoint)
    changed = replace(
        config, metadata_preconditions=[dict(CONDITIONS[0], maximum=3), CONDITIONS[1]]
    )
    with pytest.raises(ProjectTranslationCheckpointError):
        translate_project(
            changed, format_output=False, checkpoint_path=checkpoint, resume=True
        )


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_dispatch_contract_jobs_keep_original_entry_requirements(tmp_path, target):
    manifest = dict(
        kind=DISPATCH_CONTRACT_KIND,
        schemaVersion=DISPATCH_CONTRACT_SCHEMA_VERSION,
        inputs=[dict(name="elementCount", role="shape", type="integer")],
        workloads=[dict(id="count-4", values={"elementCount": 4})],
        capabilities=[],
        devices=[dict(id="portable", values={})],
        contracts=[
            dict(
                id=entry,
                source="multi.metal",
                entryPoint=entry,
                branches=[
                    dict(
                        id="default",
                        when=True,
                        workgroupSize=[4, 1, 1],
                        dispatch={"workgroupCount": [1, 1, 1]},
                    )
                ],
            )
            for entry in ("first", "second")
        ],
    )
    (tmp_path / "dispatch.json").write_text(json.dumps(manifest))
    config = replace(
        _config(tmp_path, target, entries=None, dispatch_contracts=("dispatch.json",)),
        workgroup_size=None,
    )
    payload, package, descriptors = _packages(config)
    assert payload["summary"]["translatedCount"] == 2
    for item in descriptors:
        records = parse_metadata_precondition_header(
            (package / item["artifact"]["packagePath"]).read_text()
        )
        assert len(records) == 1
        assert records[0]["sourceResource"]["entryPoint"] == _entry(item)


def test_automatic_opengl_entry_split_scopes_metadata(tmp_path):
    conditions = [dict(item, parameter="count") for item in CONDITIONS]
    payload, package, descriptors = _packages(
        _config(
            tmp_path,
            "opengl",
            entries=None,
            source=TEMPLATE_SOURCE,
            conditions=conditions,
        )
    )
    assert payload["summary"]["translatedCount"] == 2
    for item in descriptors:
        records = parse_metadata_precondition_header(
            (package / item["artifact"]["packagePath"]).read_text()
        )
        assert len(records) == 1
        assert records[0]["sourceResource"]["entryPoint"] == _entry(item)


@pytest.mark.parametrize("target", TARGETS)
def test_overlapping_source_selectors_cannot_hide_duplicate_obligations(
    tmp_path, target
):
    conditions = [CONDITIONS[0], dict(CONDITIONS[0], source="multi.metal", maximum=3)]
    payload = translate_project(
        _config(tmp_path, target, conditions=conditions), format_output=False
    ).to_json()
    assert payload["summary"]["failedCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["artifacts"][1]["entryPoint"]["source"] == "second"
