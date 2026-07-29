import copy
import json
import textwrap
from dataclasses import replace

from crosstl.project import scan_project, translate_project, validate_project_report
from crosstl.translator.entry_discovery import (
    ENTRY_DISCOVERY_AVAILABLE,
    ENTRY_DISCOVERY_UNAVAILABLE,
)


def test_project_scan_reports_metal_entries_and_drives_artifact_jobs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "multi.metal"
    source.write_text(
        textwrap.dedent("""
            kernel void first(device float* output [[buffer(0)]]) {
              output[0] = 1.0f;
            }

            kernel void second(device float* output [[buffer(0)]]) {
              output[0] = 2.0f;
            }
            """),
        encoding="utf-8",
    )

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
