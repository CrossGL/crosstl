#!/usr/bin/env python3
"""Summarize a qualification report as the machine-readable port results.

``qualify.py`` writes ``qualification.json`` with every recomputed check, the
per-case image comparisons and the evidence details. This module reduces it to
``results.json``: for each profile the device and its hardware or software
execution, the case outcomes per contract category, the evidence counts, every
negative control, the intrinsic precision against WARP and the image metrics
of every shader against the original WARP output with their bounds. The
summary is a pure function of the report and the workflow provenance, so the
results of a workflow run can be regenerated from its qualification artifact:

    python results_summary.py --qualification qualification/qualification.json \\
      --workflow-run <run URL> --source-commit <commit> --output results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KIND = "computesharp-port-results"


def _section(profile: dict[str, Any], key: str) -> dict[str, Any]:
    value = profile.get(key)
    return value if isinstance(value, dict) else {}


def _checks(profile: dict[str, Any]) -> dict[str, str]:
    return {
        key: value["status"]
        for key, value in sorted(profile.items())
        if isinstance(value, dict) and "status" in value
    }


def _cases(profile: dict[str, Any]) -> dict[str, Any]:
    accounting = _section(profile, "accounting")
    summary: dict[str, Any] = {
        "categories": _section(profile, "categories").get("categories"),
        "expected": accounting.get("expected"),
        "requiredCasesPassed": accounting.get("requiredCasesPassed"),
        "requiredGpuCasesPassed": accounting.get("requiredGpuCasesPassed"),
        "capabilityGated": len(accounting.get("capabilityInconclusive") or []),
    }
    if "optionalHardwareSkipped" in accounting:
        summary["optionalHardwareSkipped"] = len(
            accounting.get("optionalHardwareSkipped") or []
        )
    return summary


def _evidence(section: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "records",
        "resources",
        "releasedResources",
        "pipelines",
        "translatedShaders",
        "dispatchedShaders",
        "casesWithEvidence",
        "casesWithDispatch",
        "requiredCaseNamesExecuted",
        "requiredCaseNamesWithEvidence",
        "hostOnlyCaseNames",
        "gatedCaseNamesWithCapabilityQuery",
        "policySha256",
    )
    return {"status": section.get("status")} | {
        key: section[key] for key in keys if key in section
    }


def _controls(section: dict[str, Any]) -> dict[str, Any]:
    controls = {}
    for control in section.get("controls") or []:
        controls[control.get("fault")] = {
            "status": control.get("status"),
            "methods": control.get("methods"),
            "cases": control.get("cases"),
            "failed": control.get("failed"),
        }
    return {"status": section.get("status"), "faults": controls}


def _precision(section: dict[str, Any]) -> dict[str, Any]:
    functions = {}
    for name, entry in sorted((section.get("functions") or {}).items()):
        item = {
            "expression": entry.get("expression"),
            "identical": entry.get("identical"),
            "ulpDistance": entry.get("ulpDistance"),
        }
        if "required" in entry:
            item["required"] = entry["required"]
            item["mismatches"] = entry.get("mismatches")
        functions[name] = item
    return {
        "status": section.get("status"),
        "inputs": section.get("inputs"),
        "device": section.get("device"),
        "referenceDevice": section.get("referenceDevice"),
        "functions": functions,
    }


def _shader_images(section: dict[str, Any]) -> dict[str, Any]:
    references = section.get("references") or {}
    shaders: dict[str, Any] = {}
    for name, reference in sorted(references.items()):
        warp = reference.get("directxVsReference") or {}
        shaders[name] = {
            "size": reference.get("size"),
            "upstreamThreshold": reference.get("upstreamThreshold"),
            "directxUpstreamDelta": reference.get("directxUpstreamDelta"),
            "directxMeanAbsoluteErrorVsReference": warp.get("meanAbsoluteError"),
            "outputs": [],
        }
    for output in section.get("outputs") or []:
        versus = output.get("versusDirectX") or {}
        bounds = output.get("directxBounds") or {}
        entry = {
            "sha256": output.get("sha256"),
            "upstreamDelta": output.get("upstreamDelta"),
            "meanAbsoluteErrorVsDirectX": versus.get("meanAbsoluteError"),
            "meanAbsoluteErrorBound": bounds.get("meanAbsoluteError"),
            "largePixelFractionVsDirectX": versus.get("largePixelFraction"),
            "largePixelFractionBound": bounds.get("largePixelFraction"),
            "maxAbsoluteErrorVsDirectX": versus.get("maxAbsoluteError"),
        }
        if output.get("errors"):
            entry["errors"] = output["errors"]
        shaders.setdefault(output.get("shader"), {"outputs": []})["outputs"].append(
            entry
        )
    return {
        "status": section.get("status"),
        "casesExecuted": section.get("casesExecuted"),
        "casesCompared": section.get("casesCompared"),
        "shaders": shaders,
    }


def _processing_images(section: dict[str, Any]) -> dict[str, Any]:
    methods = {}
    for name, entry in sorted((section.get("methods") or {}).items()):
        versus = entry.get("versusDirectX") or {}
        bounds = entry.get("directxBounds") or {}
        methods[name] = {
            "upstreamThreshold": entry.get("upstreamThreshold"),
            "upstreamDelta": entry.get("upstreamDelta"),
            "directxUpstreamDelta": entry.get("directxUpstreamDelta"),
            "meanAbsoluteErrorVsDirectX": versus.get("meanAbsoluteError"),
            "meanAbsoluteErrorBound": bounds.get("meanAbsoluteError"),
            "largePixelFractionVsDirectX": versus.get("largePixelFraction"),
            "largePixelFractionBound": bounds.get("largePixelFraction"),
        }
        if entry.get("errors"):
            methods[name]["errors"] = entry["errors"]
    return {"status": section.get("status"), "methods": methods}


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Reduce the qualification of one profile to its results."""

    summary: dict[str, Any] = {"status": profile.get("status")}
    if profile.get("errors"):
        summary["errors"] = profile["errors"]
    run = _section(profile, "run")
    accounting = _section(profile, "accounting")
    summary["execution"] = run.get("execution") or accounting.get("execution")
    device = run.get("preflightDevice") or profile.get("device")
    if device is not None:
        summary["device"] = device
    if run:
        summary["host"] = run.get("host")
        summary["translator"] = run.get("translator")
        summary["translation"] = run.get("translation")
    summary["checks"] = _checks(profile)
    summary["cases"] = _cases(profile)
    if "evidence" in profile:
        summary["evidence"] = _evidence(_section(profile, "evidence"))
    if "negativeControls" in profile:
        summary["negativeControls"] = _controls(_section(profile, "negativeControls"))
    if "intrinsicPrecision" in profile:
        summary["intrinsicPrecision"] = _precision(
            _section(profile, "intrinsicPrecision")
        )
    if "shaderImages" in profile:
        summary["shaderImages"] = _shader_images(_section(profile, "shaderImages"))
    if "processingImages" in profile:
        summary["processingImages"] = _processing_images(
            _section(profile, "processingImages")
        )
    if "outputs" in profile:
        summary["outputs"] = _section(profile, "outputs")
    return summary


def summarize(
    report: dict[str, Any], provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the results summary of a ``qualification.json`` report."""

    return {
        "schemaVersion": 1,
        "kind": KIND,
        "status": report.get("status"),
        "upstreamCommit": report.get("upstreamCommit"),
        "contractSHA256": report.get("contractSHA256"),
        "imageChecksSHA256": report.get("imageChecksSHA256"),
        "negativeControlsSHA256": report.get("negativeControlsSHA256"),
        "provenance": dict(provenance or report.get("provenance") or {}),
        "profiles": {
            name: summarize_profile(profile)
            for name, profile in sorted((report.get("profiles") or {}).items())
        },
    }


def write(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def provenance_from_arguments(
    workflow_run: str | None, source_commit: str | None
) -> dict[str, Any]:
    provenance = {}
    if workflow_run:
        provenance["workflowRun"] = workflow_run
    if source_commit:
        provenance["sourceCommit"] = source_commit
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workflow-run", help="URL of the qualifying workflow run")
    parser.add_argument("--source-commit", help="commit whose artifacts qualified")
    args = parser.parse_args(argv)
    report = json.loads(args.qualification.read_text(encoding="utf-8"))
    provenance = provenance_from_arguments(args.workflow_run, args.source_commit)
    summary = summarize(report, provenance or None)
    write(args.output, summary)
    print(f"{summary['status']}: {args.output}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
