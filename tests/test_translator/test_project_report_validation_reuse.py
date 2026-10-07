"""Report consumers share validation only within one public operation."""

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from crosstl.project import pipeline

CONSUMERS = (
    "validate_project_report",
    "plan_runtime_integration",
    "build_runtime_artifact_manifest",
)
SOURCE = """#include <metal_stdlib>
#include "constants.h"
using namespace metal;
kernel void first(device uint* result [[buffer(0)]],
                  uint index [[thread_position_in_grid]]) { result[index] = VALUE; }
kernel void second(device uint* result [[buffer(0)]],
                   uint index [[thread_position_in_grid]]) { result[index] = VALUE + 1u; }
"""


def _report(root, entries=("first", "second"), value=1):
    shaders = root / "shaders"
    shaders.mkdir(parents=True)
    (shaders / "kernels.metal").write_text(SOURCE, encoding="utf-8")
    (shaders / "constants.h").write_text(f"#define VALUE {value}u\n", encoding="utf-8")
    (root / "crosstl.toml").write_text(
        '[project]\nsource_roots = ["shaders"]\ninclude = ["**/*"]\n'
        'exclude = ["**/*.h"]\n'
        'targets = ["metal", "directx", "opengl"]\n'
        'output_dir = "out"\nworkgroup_size = [1, 1, 1]\n'
        '[project.entry_points]\n"shaders/kernels.metal" = '
        + json.dumps(entries)
        + "\n",
        encoding="utf-8",
    )
    report = pipeline.translate_project(root, format_output=False)
    assert not report.to_json()["diagnostics"]
    assert report.to_json()["summary"]["translatedCount"] == len(entries) * 3
    path = root / "out" / "report.json"
    report.write_json(path)
    return path


def _count_scans(monkeypatch):
    original = pipeline.scan_project
    scans = []

    def scan(config):
        result = original(config)
        scans.append(result)
        return result

    monkeypatch.setattr(pipeline, "scan_project", scan)
    return scans


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("entries", (("first",), ("first", "second")))
def test_report_consumers_read_and_scan_once_per_call(
    tmp_path, monkeypatch, consumer, entries
):
    path = _report(tmp_path, entries)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    scans = _count_scans(monkeypatch)
    read_bytes = Path.read_bytes
    reads = []

    def read(file):
        if file == path:
            reads.append(file)
        return read_bytes(file)

    monkeypatch.setattr(Path, "read_bytes", read)
    operation = getattr(pipeline, consumer)
    for count in (1, 2):
        result = operation(path)
        assert result["success"], result["diagnostics"]
        assert result["sourceReportHash"] == {
            "algorithm": "sha256",
            "value": expected_hash,
        }
        assert len(scans) == len(reads) == count
    assert scans[0] is not scans[1]


def test_report_validation_shares_scan_between_unit_and_configuration_checks(
    tmp_path, monkeypatch
):
    path = _report(tmp_path)
    scans = _count_scans(monkeypatch)
    identities = []
    original_units = pipeline._current_project_scan_contract_reasons
    original_diagnostics = pipeline._current_project_config_diagnostic_contract_reasons

    def units(scan, *args, **kwargs):
        identities.append(scan)
        return original_units(scan, *args, **kwargs)

    def diagnostics(*args, **kwargs):
        identities.append(kwargs["current_scan"])
        return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_current_project_scan_contract_reasons", units)
    monkeypatch.setattr(
        pipeline, "_current_project_config_diagnostic_contract_reasons", diagnostics
    )
    result = pipeline.build_runtime_artifact_manifest(path)
    assert result["success"]
    assert len(scans) == 1 and len(identities) == 2
    assert all(scan is scans[0] for scan in identities)


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize(
    "change", ("source", "include", "config", "artifact", "new-source", "new-skipped")
)
def test_separate_report_calls_revalidate_current_files(
    tmp_path, monkeypatch, consumer, change
):
    path = _report(tmp_path)
    scans = _count_scans(monkeypatch)
    operation = getattr(pipeline, consumer)
    assert operation(path)["success"]
    if change in {"source", "include", "config"}:
        file = (
            tmp_path
            / {
                "source": "shaders/kernels.metal",
                "include": "shaders/constants.h",
                "config": "crosstl.toml",
            }[change]
        )
        previous_stat = file.stat()
        file.write_text(file.read_text().replace("1", "2", 1), encoding="utf-8")
        os.utime(file, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    elif change == "artifact":
        artifact = json.loads(path.read_text())["artifacts"][0]
        (tmp_path / artifact["path"]).write_text(
            "// replaced artifact\n", encoding="utf-8"
        )
    elif change == "new-source":
        (tmp_path / "shaders/extra.metal").write_text(SOURCE, encoding="utf-8")
    else:
        (tmp_path / "shaders/notes.txt").write_text(
            "new scan input\n", encoding="utf-8"
        )
    invalid = operation(path)
    assert not invalid["success"]
    assert invalid["diagnostics"]
    assert len(scans) == 2
    standalone = pipeline.validate_project_report(path)
    assert invalid["diagnostics"] == standalone["diagnostics"]
    if consumer == "build_runtime_artifact_manifest":
        assert invalid["artifacts"] == []
    elif consumer == "plan_runtime_integration":
        assert invalid["targetPlans"] == []


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("contents", (None, b"{", b"[]", b"17", b"\xff"))
def test_invalid_report_snapshots_return_diagnostics(
    tmp_path, monkeypatch, consumer, contents
):
    path = tmp_path / "report.json"
    if contents is not None:
        path.write_bytes(contents)
    scans = _count_scans(monkeypatch)
    result = getattr(pipeline, consumer)(path)
    assert not result["success"]
    assert result["diagnostics"][0]["code"] == "project.validate.invalid-report"
    assert result["sourceReportHash"] == (
        {"algorithm": "sha256", "value": hashlib.sha256(contents).hexdigest()}
        if contents is not None
        else None
    )
    assert not scans


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_consumers_do_not_read_unvalidated_replacement_report(
    tmp_path, monkeypatch, consumer
):
    path = _report(tmp_path)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    original = pipeline._report_contract_diagnostics

    def replace_after_validation(report_path, report):
        diagnostics = original(report_path, report)
        path.write_text('{"kind":"replacement"}', encoding="utf-8")
        return diagnostics

    monkeypatch.setattr(
        pipeline, "_report_contract_diagnostics", replace_after_validation
    )
    result = getattr(pipeline, consumer)(path)
    assert result["success"], result["diagnostics"]
    assert result["sourceReportHash"]["value"] == expected_hash
    assert not getattr(pipeline, consumer)(path)["success"]


def test_invalid_skipped_metadata_keeps_configuration_diagnostic_check(
    tmp_path, monkeypatch
):
    config = pipeline.ProjectConfig(
        root=tmp_path, include_patterns=("../outside/*.metal",)
    )
    report = pipeline.scan_project(config).to_report().to_json()
    report["skipped"] = {}
    report["diagnostics"] = []
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    scans = _count_scans(monkeypatch)
    result = pipeline.validate_project_report(path)
    assert not result["success"]
    message = result["diagnostics"][0]["message"]
    assert "skipped must be a list" in message
    assert "diagnostics must include current project config diagnostic" in message
    assert len(scans) == 1


def test_concurrent_manifests_have_independent_validation_snapshots(
    tmp_path, monkeypatch
):
    paths = [_report(tmp_path / str(index), value=index + 1) for index in range(2)]
    baseline = [pipeline.build_runtime_artifact_manifest(path) for path in paths]
    original = pipeline.scan_project
    barrier = Barrier(2)
    counts = Counter()

    def scan(config):
        result = original(config)
        counts[str(config.root)] += 1
        barrier.wait(timeout=15)
        return result

    monkeypatch.setattr(pipeline, "scan_project", scan)
    with ThreadPoolExecutor(max_workers=2) as executor:
        manifests = list(executor.map(pipeline.build_runtime_artifact_manifest, paths))
    assert counts == {str(path.parents[1]): 1 for path in paths}
    for result, expected in zip(manifests, baseline):
        assert result["success"]
        assert result["artifacts"] == expected["artifacts"]
        assert result["runtimePlan"] == expected["runtimePlan"]
        assert result["sourceReportHash"] == expected["sourceReportHash"]
