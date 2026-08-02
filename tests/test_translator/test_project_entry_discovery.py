import copy
import json
import textwrap
from dataclasses import replace

import pytest

import crosstl._crosstl as crosstl_cli
from crosstl.project import (
    ProjectConfig,
    load_project_config,
    scan_project,
    translate_project,
    validate_project_report,
)
from crosstl.project.translation_checkpoint import (
    ProjectTranslationCheckpointError,
    load_project_translation_checkpoint,
)
from crosstl.translator.entry_discovery import (
    ENTRY_DISCOVERY_AVAILABLE,
    ENTRY_DISCOVERY_UNAVAILABLE,
)

MULTI_ENTRY_METAL = textwrap.dedent("""
    kernel void first(device float* output [[buffer(0)]]) {
      output[0] = 1.0f;
    }

    kernel void second(device float* output [[buffer(0)]]) {
      output[0] = 2.0f;
    }
    """).strip()


def _write_multi_entry_metal(repo, name="multi.metal"):
    source = repo / name
    source.write_text(MULTI_ENTRY_METAL + "\n", encoding="utf-8")
    return source


def test_project_scan_reports_metal_entries_and_drives_artifact_jobs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_multi_entry_metal(repo)

    scan = scan_project(repo)

    assert len(scan.units) == 1
    discovery = scan.units[0].entry_discovery
    assert discovery is not None
    assert discovery.status == ENTRY_DISCOVERY_AVAILABLE
    assert discovery.source_backend == "metal"
    assert discovery.source_path == "multi.metal"
    assert [entry.name for entry in discovery.entries] == ["first", "second"]
    assert scan.discovered_entry_points() == {"multi.metal": ("first", "second")}

    scan_report = scan.to_report(targets=["opengl"])
    scan_payload = scan_report.to_json()
    assert scan_payload["units"][0]["entryDiscovery"] == discovery.to_json()
    scan_report_path = repo / "scan-report.json"
    scan_report.write_json(scan_report_path)
    assert validate_project_report(scan_report_path)["success"] is True

    translation_config = replace(
        scan.config,
        targets=("opengl",),
        output_dir="translated",
        entry_points=scan.discovered_entry_points(),
    )
    report = translate_project(
        translation_config,
        format_output=False,
    )
    payload = report.to_json()

    assert payload["summary"]["translatedCount"] == 2
    assert payload["summary"]["failedCount"] == 0
    assert [artifact["entryPoint"]["source"] for artifact in payload["artifacts"]] == [
        "first",
        "second",
    ]
    assert all((repo / artifact["path"]).is_file() for artifact in payload["artifacts"])
    report_path = repo / "translated" / "report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True


def test_project_scan_reports_unavailable_discovery_for_other_frontends(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(
        "shader Simple { void compute main() {} }\n",
        encoding="utf-8",
    )

    payload = scan_project(repo).to_report(targets=["opengl"]).to_json()

    discovery = payload["units"][0]["entryDiscovery"]
    assert discovery == {
        "status": ENTRY_DISCOVERY_UNAVAILABLE,
        "sourceBackend": "cgl",
        "sourcePath": "simple.cgl",
        "entryCount": 0,
        "entries": [],
        "diagnosticCount": 0,
        "diagnostics": [],
    }


def test_project_scan_promotes_metal_entry_discovery_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "dynamic.metal").write_text(
        textwrap.dedent("""
            template <typename T>
            [[kernel]] void convert(device T* output [[buffer(0)]]) {
              output[0] = T(1);
            }

            template [[host_name(EXPORTED_NAME)]] [[kernel]]
            decltype(convert<float>) convert<float>;
            """),
        encoding="utf-8",
    )

    report = scan_project(repo).to_report(targets=["opengl"])
    payload = report.to_json()

    discovery = payload["units"][0]["entryDiscovery"]
    assert discovery["entries"] == []
    assert discovery["diagnosticCount"] == 1
    assert discovery["diagnostics"][0]["code"] == (
        "source.entry-discovery.unresolved-host-name"
    )
    diagnostics = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "source.entry-discovery.unresolved-host-name"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0]["sourceBackend"] == "metal"
    assert diagnostics[0]["checkKind"] == "source-entry-discovery"
    assert diagnostics[0]["location"]["file"] == "dynamic.metal"
    report_path = repo / "diagnostic-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    assert validation["success"] is True, json.dumps(validation, indent=2)


def test_project_report_rejects_malformed_entry_discovery_contracts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "entry.metal").write_text(
        "kernel void entry(device float* output [[buffer(0)]]) "
        "{ output[0] = 1.0f; }\n",
        encoding="utf-8",
    )
    payload = scan_project(repo).to_report(targets=["opengl"]).to_json()
    cases = []

    wrong_count = copy.deepcopy(payload)
    wrong_count["units"][0]["entryDiscovery"]["entryCount"] = 2
    cases.append((wrong_count, "entryCount must match"))

    wrong_source = copy.deepcopy(payload)
    wrong_source["units"][0]["entryDiscovery"]["sourcePath"] = "other.metal"
    cases.append((wrong_source, "sourcePath must match"))

    duplicate_entry = copy.deepcopy(payload)
    entry = duplicate_entry["units"][0]["entryDiscovery"]["entries"][0]
    duplicate_entry["units"][0]["entryDiscovery"]["entries"].append(
        copy.deepcopy(entry)
    )
    duplicate_entry["units"][0]["entryDiscovery"]["entryCount"] = 2
    cases.append((duplicate_entry, "name must be unique"))

    unavailable_with_entry = copy.deepcopy(payload)
    unavailable_with_entry["units"][0]["entryDiscovery"]["status"] = "unavailable"
    cases.append((unavailable_with_entry, "when unavailable"))

    unexpected_field = copy.deepcopy(payload)
    unexpected_field["units"][0]["entryDiscovery"]["unexpected"] = True
    cases.append((unexpected_field, "is not allowed"))

    for index, (invalid_payload, expected_message) in enumerate(cases):
        report_path = repo / f"invalid-{index}.json"
        report_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
        validation = validate_project_report(report_path)
        assert validation["success"] is False
        messages = "\n".join(
            diagnostic["message"] for diagnostic in validation["diagnostics"]
        )
        assert expected_message in messages


def test_project_config_requires_source_patterns_for_discovered_entry_artifacts(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            translate_discovered_entry_points = ["kernels/*.metal"]
            """).strip() + "\n",
        encoding="utf-8",
    )

    config = load_project_config(repo)

    assert config.translate_discovered_entry_points == ["kernels/*.metal"]
    assert "project.config.discovered-entry-point-pattern-unmatched" in {
        diagnostic.code for diagnostic in scan_project(config).diagnostics
    }

    outside_config = ProjectConfig(
        root=repo,
        translate_discovered_entry_points=("../*.metal",),
    )
    assert "project.config.discovered-entry-point-pattern-outside-project" in {
        diagnostic.code for diagnostic in scan_project(outside_config).diagnostics
    }

    (repo / "crosstl.toml").write_text(
        "[project]\ntranslate_discovered_entry_points = true\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=(
            "project.translate_discovered_entry_points must be a string or list "
            "of strings"
        ),
    ):
        load_project_config(repo)

    with pytest.raises(
        ValueError,
        match="translate_discovered_entry_points entries must be strings",
    ):
        ProjectConfig(
            root=repo,
            translate_discovered_entry_points=["kernels/*.metal", 1],
        )
    with pytest.raises(ValueError, match="must not contain duplicate patterns"):
        ProjectConfig(
            root=repo,
            translate_discovered_entry_points=["*.metal", "*.metal"],
        )


def test_translate_project_expands_selected_discovered_entries_with_checkpoint(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_multi_entry_metal(repo)
    checkpoint_path = repo / "progress.json"
    config = ProjectConfig(
        root=repo,
        targets=("opengl",),
        output_dir="translated",
    )

    report = translate_project(
        config,
        format_output=False,
        checkpoint_path=checkpoint_path,
        translate_discovered_entry_points=("multi.metal",),
    )
    payload = report.to_json()

    assert [artifact["entryPoint"]["source"] for artifact in payload["artifacts"]] == [
        "first",
        "second",
    ]
    assert [artifact["path"] for artifact in payload["artifacts"]] == [
        "translated/opengl/multi/first.glsl",
        "translated/opengl/multi/second.glsl",
    ]
    assert payload["project"]["entryPointSelections"] == {}
    assert payload["project"]["translateDiscoveredEntryPoints"] == ["multi.metal"]
    assert payload["project"]["translateDiscoveredEntryPointPatternCount"] == 1
    assert payload["project"]["resolvedEntryPointSelections"] == {
        "multi.metal": ["first", "second"]
    }
    assert payload["project"]["resolvedEntryPointSelectionCount"] == 1
    assert payload["artifactMatrix"]["expectedArtifactCount"] == 2
    assert payload["artifactMatrix"]["complete"] is True
    assert all(
        artifact["provenance"]["pipeline"] == "entry-scoped-translate"
        for artifact in payload["artifacts"]
    )
    assert all(
        "sourceMap" in artifact and "sourceRemap" in artifact
        for artifact in payload["artifacts"]
    )

    checkpoint = load_project_translation_checkpoint(checkpoint_path)
    assert checkpoint["state"] == "complete"
    assert checkpoint["plan"]["completedCount"] == 2
    assert [
        completion["coordinate"]["entryPoint"]
        for completion in checkpoint["plan"]["completed"]
    ] == ["first", "second"]
    report_path = repo / "translated" / "report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True

    with pytest.raises(ProjectTranslationCheckpointError):
        translate_project(
            config,
            format_output=False,
            checkpoint_path=checkpoint_path,
            resume=True,
        )


@pytest.mark.parametrize(
    "entry_points",
    (
        {"multi.metal": "second"},
        {"*.metal": "second"},
    ),
    ids=("exact", "glob"),
)
def test_explicit_entry_selectors_override_discovered_entry_artifacts(
    tmp_path,
    entry_points,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_multi_entry_metal(repo)
    config = ProjectConfig(
        root=repo,
        targets=("opengl",),
        output_dir="translated",
        entry_points=entry_points,
        translate_discovered_entry_points=("*.metal",),
    )

    payload = translate_project(config, format_output=False).to_json()

    assert len(payload["artifacts"]) == 1
    assert payload["artifacts"][0]["entryPoint"]["source"] == "second"
    assert payload["project"]["resolvedEntryPointSelections"] == entry_points


def test_discovered_entry_artifacts_keep_unavailable_frontends_explicit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(
        "shader Simple { void compute main() {} }\n",
        encoding="utf-8",
    )
    config = ProjectConfig(
        root=repo,
        targets=("opengl",),
        output_dir="translated",
        translate_discovered_entry_points=("simple.cgl",),
    )

    payload = translate_project(config, format_output=False).to_json()

    assert payload["project"]["resolvedEntryPointSelections"] == {}
    assert len(payload["artifacts"]) == 1
    assert "entryPoint" not in payload["artifacts"][0]
    diagnostic = next(
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "project.config.discovered-entry-points-unavailable"
    )
    assert diagnostic["severity"] == "error"
    assert diagnostic["details"]["status"] == ENTRY_DISCOVERY_UNAVAILABLE


def test_discovered_entry_artifact_patterns_do_not_expand_unselected_dense_source(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_multi_entry_metal(repo, "selected.metal")
    dense_entries = "\n".join(
        "kernel void dense_{0:03d}(device float* output [[buffer(0)]]) "
        "{{ output[0] = {0}.0f; }}".format(index)
        for index in range(128)
    )
    (repo / "dense.metal").write_text(dense_entries + "\n", encoding="utf-8")
    config = ProjectConfig(
        root=repo,
        targets=("opengl",),
        translate_discovered_entry_points=("selected.metal",),
    )

    payload = scan_project(config).to_report().to_json()

    assert payload["project"]["resolvedEntryPointSelections"] == {
        "selected.metal": ["first", "second"]
    }
    assert payload["artifactMatrix"]["unitCount"] == 2
    assert payload["artifactMatrix"]["expectedArtifactCount"] == 3


def test_project_report_rejects_incorrect_resolved_entry_selection(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_multi_entry_metal(repo)
    config = ProjectConfig(
        root=repo,
        targets=("opengl",),
        translate_discovered_entry_points=("multi.metal",),
    )
    payload = scan_project(config).to_report().to_json()
    payload["project"]["resolvedEntryPointSelections"] = {"multi.metal": ["second"]}
    report_path = repo / "invalid-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert "must match configured selectors and available discovered entries" in (
        validation["diagnostics"][0]["message"]
    )


def test_translate_project_cli_expands_discovered_entries_for_selected_source(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_multi_entry_metal(repo)
    report_path = repo / "report.json"

    exit_code = crosstl_cli.main(
        [
            "translate-project",
            str(repo),
            "--target",
            "opengl",
            "--output-dir",
            "translated",
            "--translate-discovered-entry-points",
            "multi.metal",
            "--workers",
            "2",
            "--report",
            str(report_path),
            "--no-format",
        ]
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["project"]["translateDiscoveredEntryPoints"] == ["multi.metal"]
    assert [artifact["entryPoint"]["source"] for artifact in payload["artifacts"]] == [
        "first",
        "second",
    ]
