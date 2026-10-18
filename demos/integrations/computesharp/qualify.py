#!/usr/bin/env python3
"""Qualify the ComputeSharp port from the uploaded artifacts of every profile.

``--directx`` is the evidence directory of the original Windows WARP job, and
``--metal`` and ``--opengl`` are the evidence directories of the translated
jobs. Every result is recomputed from the raw artifacts instead of being read
from the job reports:

- each discovery export reproduces the frozen coverage lock, and the translated
  builds expose the same cases and byte-identical generated HLSL as Windows;
- each TRX accounts for every discovered case of its profile, and outcomes are
  reported per contract category so CPU-only and Windows-specific cases never
  count as translated GPU execution;
- the dispatch evidence of each translated run verifies against its manifest,
  translated artifacts, discovery and TRX, and every negative control failed;
- the intrinsic probe of each translated target evaluated the same generated
  HLSL over the same inputs as WARP; its sin, cos and tan results are
  bit-identical to WARP, and its contraction-free arithmetic (dot products,
  multiply-adds, lerp and smoothstep) matches WARP and an independent CPU
  model except for subnormal flushing (see ``precision_report.py``);
- every ShadersTests case of each translated target has exactly one captured
  readback whose pixels hash to the recorded readback, reproduce the upstream
  assertion and stay within the per-case bounds derived from the WARP output;
- the ImageProcessingTests GPU outputs stay within the bounds derived from the
  WARP GPU output and the CPU reference of the Windows run.

The report is ``qualification.json``, with amplified difference images under
``images/``, and its summary ``results.json`` (see ``results_summary.py``). The
process exits non-zero unless every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402
import compare_discovery  # noqa: E402
import coverage_contract  # noqa: E402
import evidence as evidence_module  # noqa: E402
import precision_report  # noqa: E402
import results_summary  # noqa: E402

CONTROLS = HERE / "negative_controls.json"
IMAGE_CHECKS = HERE / "image_checks.json"
DIRECTX_PROFILE = "directx-warp"
TARGETS = {"metal": "metal-apple-gpu", "opengl": "opengl-mesa-llvmpipe"}
# DXGI_FORMAT_R8G8B8A8_UNORM and DXGI_FORMAT_B8G8R8A8_UNORM.
CHANNEL_ORDERS = {28: (0, 1, 2, 3), 87: (2, 1, 0, 3)}
DIFFERENCE_GAIN = 8
MAX_ERRORS = 100
SHADER_ARGUMENT = re.compile(r"\((?:[^()]*, )?([A-Za-z_][\w.]*)\)$")


def _failure(errors: list[str]) -> dict[str, Any]:
    return {"status": "failed", "errors": list(errors)[:MAX_ERRORS]}


def _check(report: dict[str, Any], key: str, function: Callable, *args, **kwargs):
    """Store the summary of one check, or its errors, under ``report[key]``."""

    try:
        value = function(*args, **kwargs)
    except (baseline.AccountingError, evidence_module.EvidenceError) as exc:
        value = _failure(exc.errors)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        value = _failure([f"{type(exc).__name__}: {exc}"])
    if isinstance(value, dict) and "status" not in value:
        value = {"status": "passed", **value}
    report[key] = value
    return value


def evidence_file(directory: Path) -> Path:
    """Return the compressed or plain evidence file of a run directory."""

    compressed = directory / "evidence.jsonl.gz"
    plain = directory / "evidence.jsonl"
    return plain if plain.is_file() and not compressed.is_file() else compressed


def category_outcomes(contract_path: Path, discovery_dir: Path, trx: Path) -> dict:
    """Count discovered, executed and unrun cases and TRX outcomes per category."""

    discovery = baseline.load_discovery(discovery_dir)
    categories = evidence_module.case_categories(contract_path, discovery)
    table: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"discovered": 0, "executed": 0, "outcomes": Counter()}
    )
    for case in discovery["tests"]:
        owner = case["method"].rsplit(".", 1)[0]
        table[categories[f"{owner}.{case['displayName']}"]]["discovered"] += 1
    for case in evidence_module.trx_cases(trx):
        entry = table[categories.get(case["name"], "unclassified")]
        entry["executed"] += 1
        entry["outcomes"][case["outcome"] or "unknown"] += 1
    result = {
        name: {
            "discovered": entry["discovered"],
            "executed": entry["executed"],
            "notRun": entry["discovered"] - entry["executed"],
            "outcomes": dict(sorted(entry["outcomes"].items())),
        }
        for name, entry in sorted(table.items())
    }
    return {"status": "passed", "categories": result}


def _coverage(discovery: Path, lock: Path, contract: Path) -> dict[str, Any]:
    checked = coverage_contract.check(discovery, lock, contract)
    return {"totals": checked["totals"], "caseCount": checked["caseCount"]}


def _run_report(results: Path, profile: str) -> dict[str, Any]:
    report = json.loads(results.read_text(encoding="utf-8"))
    errors = []
    if report.get("profile") != profile:
        errors.append(f"results.json is for {report.get('profile')!r}")
    if report.get("status") != "passed":
        errors.append(f"the job reported {report.get('status')!r}")
    if report.get("nativeTargetExecutionQualified") is not True:
        errors.append("the job did not qualify native target execution")
    preflight = report.get("preflight") or {}
    if preflight.get("status") != "passed":
        errors.append("the device preflight did not pass before the tests ran")
    summary = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "host": report.get("host"),
        "execution": report.get("execution"),
        "preflightDevice": preflight.get("device"),
        "translator": report.get("translator"),
        "translation": report.get("translation"),
    }
    return summary


def qualify_controls(
    spec: dict[str, Any],
    run: Path,
    *,
    manifest: Path,
    discovery: Path,
    profile: str,
    contract_path: Path,
    passed: Counter[str] | None,
) -> dict[str, Any]:
    """Re-verify every frozen negative control from its TRX and evidence."""

    results = []
    for control in spec["controls"]:
        directory = run / "controls" / control["fault"]
        results.append(
            evidence_module.verify_control(
                control,
                trx=directory / "trx" / "results.trx",
                evidence=evidence_file(directory),
                manifest_path=manifest,
                discovery_dir=discovery,
                profile=profile,
                contract_path=contract_path,
                passed=passed,
            )
        )
    ok = bool(results) and all(item["status"] == "passed" for item in results)
    return {"status": "passed" if ok else "failed", "controls": results}


# Image comparison. numpy and Pillow are imported lazily so that the inventory
# and accounting checks need only the standard library.


def load_image(path: Path):
    """Decode an image into an RGBA int16 array of shape (height, width, 4)."""

    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.int16)


def image_metrics(
    actual, expected, *, large: int, reference_large: int | None = None
) -> dict[str, Any]:
    """RGB differences in 0-255 units: mean, maximum and large-pixel fractions."""

    difference = abs(actual[..., :3] - expected[..., :3])
    worst = difference.max(axis=2)
    metrics = {
        "meanAbsoluteError": float(difference.mean()),
        "maxAbsoluteError": int(difference.max()),
        "largePixelFraction": float((worst > large).mean()),
    }
    if reference_large is not None:
        metrics["referenceLargePixelFraction"] = float((worst > reference_large).mean())
    return metrics


def upstream_delta(expected, actual) -> float:
    """TolerantImageComparer's normalized delta, for an actual image saved as RGB."""

    import numpy as np

    opaque = actual.astype(np.int64)
    opaque[..., 3] = 255
    total = abs(expected.astype(np.int64) - opaque).sum()
    height, width = expected.shape[:2]
    return float(total) / (width * height) / (4 * 65535)


def write_difference(path: Path, actual, expected) -> str:
    """Write |actual - expected| amplified by DIFFERENCE_GAIN as an RGB PNG."""

    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    difference = abs(actual[..., :3] - expected[..., :3]) * DIFFERENCE_GAIN
    pixels = np.clip(difference, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, "RGB").save(path, optimize=True)
    return path.name


def capture_digest(image, record: dict[str, Any]) -> str | None:
    """SHA-256 of a captured image in readback byte order, when rows are tight."""

    import numpy as np

    order = CHANNEL_ORDERS.get(record.get("format"))
    if order is None or record.get("rowPitch") != record.get("width", 0) * 4:
        return None
    pixels = image[..., list(order)].astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def shader_of(case_name: str) -> str | None:
    """Shader type name of a ShadersTests case, as upstream names its files."""

    match = SHADER_ARGUMENT.search(case_name)
    return match.group(1).rsplit(".", 1)[-1] if match else None


def _bounds(spec: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    bounds = spec["bounds"]
    return {
        "meanAbsoluteError": (
            bounds["meanAbsoluteErrorScale"] * reference["meanAbsoluteError"]
            + bounds["meanAbsoluteErrorMargin"]
        ),
        "largePixelFraction": (
            bounds["largePixelFractionScale"] * reference["referenceLargePixelFraction"]
            + bounds["largePixelFractionMargin"]
        ),
    }


def _within(metrics: dict[str, Any], bounds: dict[str, float]) -> list[str]:
    return [
        f"{name} {metrics[name]:.6g} exceeds the DirectX bound {limit:.6g}"
        for name, limit in bounds.items()
        if metrics[name] > limit
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shader_images(
    spec: dict[str, Any],
    *,
    run: Path,
    directx_outputs: Path,
    trx: Path,
    evidence: Path,
    images: Path,
) -> dict[str, Any]:
    """Compare the captured output of every ShadersTests case with WARP."""

    section = spec["shaderImages"]
    prefix = section["methodPrefix"]
    large = spec["metrics"]["largeDifference"]
    reference_large = spec["metrics"]["referenceLargeDifference"]
    target_outputs = run / "outputs"
    errors: list[str] = []
    executed = Counter(
        case["name"]
        for case in evidence_module.trx_cases(trx)
        if case["name"].startswith(prefix)
    )
    if not executed:
        errors.append("no ShadersTests case was executed")
    captured: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in evidence_module.read_records(evidence):
        case = record.get("case")
        if (
            record.get("record") == "readback"
            and isinstance(case, str)
            and case.startswith(prefix)
            and record.get("capture")
        ):
            captured[case].append(record)
    references: dict[str, dict[str, Any]] = {}
    compared: dict[tuple[str, str], dict[str, Any]] = {}
    cases = []
    for name, count in sorted(executed.items()):
        records = captured.pop(name, [])
        shader = shader_of(name)
        entry: dict[str, Any] = {"case": name, "shader": shader}
        cases.append(entry)
        problems: list[str] = []
        threshold = section["upstreamThresholds"].get(shader or "")
        if threshold is None:
            problems.append("no frozen upstream threshold for this shader")
        if len(records) != count:
            problems.append(
                f"{len(records)} captured readback(s) for {count} result(s)"
            )
        if problems:
            entry["errors"] = problems
            errors.extend(f"{name}: {text}" for text in problems)
            continue
        if shader not in references:
            references[shader] = _shader_reference(
                section,
                shader,
                directx_outputs,
                target_outputs,
                large=large,
                reference_large=reference_large,
                threshold=threshold,
            )
        reference = references[shader]
        if reference.get("errors"):
            problems.extend(reference["errors"])
        else:
            for record in records:
                key = (shader, record["sha256"])
                if key not in compared:
                    compared[key] = _compare_capture(
                        spec,
                        record,
                        run / "captures" / record["capture"],
                        reference,
                        images / f"{shader}-{record['sha256'][:12]}.png",
                        threshold=threshold,
                    )
                result = compared[key]
                entry.setdefault("captures", []).append(record["sha256"])
                problems.extend(result.get("errors", []))
        if problems:
            entry["errors"] = problems
            errors.extend(f"{name}: {text}" for text in problems)
    for name in sorted(captured):
        errors.append(f"{name}: captured output of a case without a result")
    return {
        "status": "failed" if errors else "passed",
        "errors": errors[:MAX_ERRORS],
        "casesCompared": sum(1 for entry in cases if "captures" in entry),
        "casesExecuted": sum(executed.values()),
        "references": {
            name: {key: value for key, value in item.items() if key != "_images"}
            for name, item in sorted(references.items())
        },
        "outputs": [
            {"shader": shader, "sha256": digest, **result}
            for (shader, digest), result in sorted(compared.items())
        ],
        "cases": cases,
    }


def _shader_reference(
    section: dict[str, Any],
    shader: str,
    directx_outputs: Path,
    target_outputs: Path,
    *,
    large: int,
    reference_large: int,
    threshold: float,
) -> dict[str, Any]:
    reference_path = directx_outputs / section["reference"].format(shader=shader)
    warp_path = directx_outputs / section["directx"].format(shader=shader)
    target_reference = target_outputs / section["reference"].format(shader=shader)
    errors = []
    for path in (reference_path, warp_path, target_reference):
        if not path.is_file():
            errors.append(f"missing {path}")
    if errors:
        return {"errors": errors}
    if _sha256(target_reference) != _sha256(reference_path):
        return {"errors": ["the reference image differs between the profiles"]}
    reference = load_image(reference_path)
    warp = load_image(warp_path)
    if warp.shape != reference.shape:
        return {"errors": [f"WARP output {warp.shape} and reference {reference.shape}"]}
    metrics = image_metrics(
        warp, reference, large=large, reference_large=reference_large
    )
    delta = upstream_delta(reference, warp)
    result = {
        "reference": reference_path.name,
        "referenceSha256": _sha256(reference_path),
        "directxSha256": _sha256(warp_path),
        "size": [int(reference.shape[1]), int(reference.shape[0])],
        "upstreamThreshold": threshold,
        "directxUpstreamDelta": delta,
        "directxVsReference": metrics,
        "_images": (reference, warp),
    }
    if delta > threshold:
        result["errors"] = [
            f"the WARP output does not reproduce its upstream assertion ({delta:.6g} > {threshold:.6g})"
        ]
    return result


def _compare_capture(
    spec: dict[str, Any],
    record: dict[str, Any],
    path: Path,
    reference: dict[str, Any],
    difference: Path,
    *,
    threshold: float,
) -> dict[str, Any]:
    reference_image, warp = reference["_images"]
    if not path.is_file():
        return {"errors": [f"missing capture {path.name}"]}
    image = load_image(path)
    errors = []
    size = [int(image.shape[1]), int(image.shape[0])]
    if size != [record.get("width"), record.get("height")]:
        errors.append(f"capture size {size} differs from the readback region")
    digest = capture_digest(image, record)
    if digest is None:
        errors.append("the capture cannot be checked against its readback hash")
    elif digest != record["sha256"]:
        errors.append("the captured pixels do not hash to the recorded readback")
    if image.shape != warp.shape:
        errors.append(f"output size {size} differs from the DirectX output")
        return {"errors": errors}
    large = spec["metrics"]["largeDifference"]
    versus_directx = image_metrics(image, warp, large=large)
    versus_reference = image_metrics(image, reference_image, large=large)
    delta = upstream_delta(reference_image, image)
    bounds = _bounds(spec, reference["directxVsReference"])
    if delta > threshold:
        errors.append(f"upstream delta {delta:.6g} exceeds {threshold:.6g}")
    errors.extend(_within(versus_directx, bounds))
    return {
        "upstreamDelta": delta,
        "versusDirectX": versus_directx,
        "versusReference": versus_reference,
        "directxBounds": bounds,
        "difference": write_difference(difference, image, warp),
        **({"errors": errors} if errors else {}),
    }


def processing_images(
    spec: dict[str, Any],
    *,
    run: Path,
    directx_outputs: Path,
    trx: Path,
    images: Path,
) -> dict[str, Any]:
    """Compare the ImageProcessingTests GPU outputs with WARP and the CPU references."""

    large = spec["metrics"]["largeDifference"]
    reference_large = spec["metrics"]["referenceLargeDifference"]
    target_outputs = run / "outputs"
    passed = Counter(
        case["name"].split(" (", 1)[0]
        for case in evidence_module.trx_cases(trx)
        if case["outcome"] == "Passed"
    )
    errors: list[str] = []
    results = {}
    for method, entry in sorted(spec["processingImages"]["cases"].items()):
        paths = {
            "gpu": target_outputs / entry["gpu"],
            "cpu": target_outputs / entry["cpu"],
            "directxGpu": directx_outputs / entry["gpu"],
            "directxCpu": directx_outputs / entry["cpu"],
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        problems = [f"missing {path}" for path in missing]
        if not passed[method]:
            problems.append("no case of the method passed")
        if problems:
            results[method] = {"errors": problems}
            errors.extend(f"{method}: {text}" for text in problems)
            continue
        loaded = {key: load_image(path) for key, path in paths.items()}
        shapes = {key: tuple(image.shape) for key, image in loaded.items()}
        if len(set(shapes.values())) != 1:
            problems.append(f"image sizes differ: {shapes}")
            results[method] = {"errors": problems}
            errors.extend(f"{method}: {text}" for text in problems)
            continue
        reference = image_metrics(
            loaded["directxGpu"],
            loaded["directxCpu"],
            large=large,
            reference_large=reference_large,
        )
        versus_directx = image_metrics(loaded["gpu"], loaded["directxGpu"], large=large)
        bounds = _bounds(spec, reference)
        problems.extend(_within(versus_directx, bounds))
        results[method] = {
            "sha256": {key: _sha256(path) for key, path in paths.items()},
            "upstreamThreshold": entry["upstreamThreshold"],
            "upstreamDelta": upstream_delta(loaded["cpu"], loaded["gpu"]),
            "directxUpstreamDelta": upstream_delta(
                loaded["directxCpu"], loaded["directxGpu"]
            ),
            "directxGpuVsCpu": reference,
            "versusDirectX": versus_directx,
            "versusOwnCpu": image_metrics(loaded["gpu"], loaded["cpu"], large=large),
            "cpuVersusDirectXCpu": image_metrics(
                loaded["cpu"], loaded["directxCpu"], large=large
            ),
            "directxBounds": bounds,
            "difference": write_difference(
                images / f"{method.rsplit('.', 1)[-1]}.png",
                loaded["gpu"],
                loaded["directxGpu"],
            ),
        }
        if problems:
            results[method]["errors"] = problems
            errors.extend(f"{method}: {text}" for text in problems)
    return {
        "status": "failed" if errors else "passed",
        "errors": errors[:MAX_ERRORS],
        "methods": results,
    }


def qualify_directx(
    directory: Path, *, contract_path: Path, lock_path: Path
) -> dict[str, Any]:
    discovery = directory / "discovery"
    trx = directory / "trx" / "baseline.trx"
    report: dict[str, Any] = {"profile": DIRECTX_PROFILE}
    _check(report, "coverage", _coverage, discovery, lock_path, contract_path)
    _check(
        report,
        "accounting",
        baseline.verify_results,
        discovery,
        trx,
        DIRECTX_PROFILE,
        contract_path,
    )
    _check(report, "categories", category_outcomes, contract_path, discovery, trx)
    outputs = directory / "outputs" / "outputs.json"
    if outputs.is_file():
        collected = json.loads(outputs.read_text(encoding="utf-8"))
        report["outputs"] = {
            "status": "passed" if collected.get("outputs") else "failed",
            "outputs": len(collected.get("outputs", [])),
            "references": len(collected.get("references", [])),
        }
    else:
        report["outputs"] = _failure([f"missing {outputs}"])
    probe = directory / "precision" / "directx" / "probe.json"
    if probe.is_file():
        report["device"] = json.loads(probe.read_text(encoding="utf-8")).get("device")
    return report


def qualify_target(
    name: str,
    directory: Path,
    *,
    directx: Path,
    contract_path: Path,
    lock_path: Path,
    controls: dict[str, Any],
    image_checks: dict[str, Any],
    images: Path,
) -> dict[str, Any]:
    profile = TARGETS[name]
    discovery = directory / "discovery"
    run = directory / "run"
    trx = run / "trx" / "results.trx"
    manifest = run / "artifacts" / name / "manifest.json"
    evidence = evidence_file(run)
    report: dict[str, Any] = {"profile": profile}
    _check(report, "coverage", _coverage, discovery, lock_path, contract_path)
    _check(
        report,
        "discoveryParity",
        compare_discovery.compare,
        directx / "discovery",
        discovery,
    )
    _check(
        report,
        "accounting",
        baseline.verify_results,
        discovery,
        trx,
        profile,
        contract_path,
    )
    _check(report, "categories", category_outcomes, contract_path, discovery, trx)
    _check(
        report,
        "evidence",
        evidence_module.verify,
        evidence=evidence,
        manifest_path=manifest,
        discovery_dir=discovery,
        trx=trx,
        profile=profile,
        contract_path=contract_path,
    )
    passed = None
    if trx.is_file():
        passed = Counter(
            case["name"]
            for case in evidence_module.trx_cases(trx)
            if case["outcome"] == "Passed"
        )
    _check(
        report,
        "negativeControls",
        qualify_controls,
        controls,
        run,
        manifest=manifest,
        discovery=discovery,
        profile=profile,
        contract_path=contract_path,
        passed=passed,
    )
    _check(report, "run", _run_report, run / "results.json", profile)
    _check(
        report,
        "intrinsicPrecision",
        precision_report.compare,
        directx / "precision" / "directx",
        directory / "precision" / name,
    )
    directx_outputs = directx / "outputs"
    _check(
        report,
        "shaderImages",
        shader_images,
        image_checks,
        run=run,
        directx_outputs=directx_outputs,
        trx=trx,
        evidence=evidence,
        images=images / name / "shaders",
    )
    _check(
        report,
        "processingImages",
        processing_images,
        image_checks,
        run=run,
        directx_outputs=directx_outputs,
        trx=trx,
        images=images / name / "processing",
    )
    return report


def _status(report: dict[str, Any]) -> str:
    sections = [value for value in report.values() if isinstance(value, dict)]
    failed = any(section.get("status") == "failed" for section in sections)
    return "failed" if failed else "passed"


def qualify(
    *,
    directx: Path,
    targets: dict[str, Path],
    output: Path,
    contract_path: Path = coverage_contract.CONTRACT,
    lock_path: Path = coverage_contract.LOCK,
    controls_path: Path = CONTROLS,
    image_checks_path: Path = IMAGE_CHECKS,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    image_checks = json.loads(image_checks_path.read_text(encoding="utf-8"))
    profiles = {
        DIRECTX_PROFILE: qualify_directx(
            directx, contract_path=contract_path, lock_path=lock_path
        )
    }
    for name in sorted(TARGETS):
        if name not in targets:
            profiles[TARGETS[name]] = _failure([f"no {name} artifacts were supplied"])
            continue
        profiles[TARGETS[name]] = qualify_target(
            name,
            targets[name],
            directx=directx,
            contract_path=contract_path,
            lock_path=lock_path,
            controls=controls,
            image_checks=image_checks,
            images=output / "images",
        )
    for value in profiles.values():
        value["status"] = (
            value["status"] if value.get("status") == "failed" else _status(value)
        )
    report = {
        "schemaVersion": 1,
        "kind": "computesharp-port-qualification",
        "upstreamCommit": coverage_contract.PIN,
        "contractSHA256": (
            hashlib.sha256(
                contract_path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
        ),
        "imageChecksSHA256": _sha256(image_checks_path),
        "negativeControlsSHA256": _sha256(controls_path),
        "provenance": dict(provenance or {}),
        "status": (
            "passed"
            if all(value["status"] == "passed" for value in profiles.values())
            else "failed"
        ),
        "profiles": profiles,
    }
    (output / "qualification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results_summary.write(output / "results.json", results_summary.summarize(report))
    return report


def summary_lines(report: dict[str, Any]) -> list[str]:
    lines = [f"qualification: {report['status']}"]
    for name, profile in sorted(report["profiles"].items()):
        failed = sorted(
            key
            for key, value in profile.items()
            if isinstance(value, dict) and value.get("status") == "failed"
        )
        accounting = profile.get("accounting", {})
        detail = ""
        if isinstance(accounting, dict) and "requiredGpuCasesPassed" in accounting:
            detail = (
                f" required GPU passed {accounting['requiredGpuCasesPassed']},"
                f" capability-gated {len(accounting['capabilityInconclusive'])}"
            )
        lines.append(
            f"  {name}: {profile['status']}{detail}"
            + (f" (failed: {', '.join(failed)})" if failed else "")
        )
        for key in failed:
            for error in profile[key].get("errors", [])[:5]:
                lines.append(f"    {key}: {error}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directx", required=True, type=Path)
    parser.add_argument("--metal", type=Path)
    parser.add_argument("--opengl", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=coverage_contract.CONTRACT)
    parser.add_argument("--lock", type=Path, default=coverage_contract.LOCK)
    parser.add_argument("--controls", type=Path, default=CONTROLS)
    parser.add_argument("--image-checks", type=Path, default=IMAGE_CHECKS)
    parser.add_argument("--workflow-run", help="URL of the workflow run qualified")
    parser.add_argument(
        "--source-commit", help="commit whose workflow produced the artifacts"
    )
    args = parser.parse_args(argv)
    targets = {
        name: path
        for name, path in (("metal", args.metal), ("opengl", args.opengl))
        if path is not None
    }
    report = qualify(
        directx=args.directx,
        targets=targets,
        output=args.output,
        contract_path=args.contract,
        lock_path=args.lock,
        controls_path=args.controls,
        image_checks_path=args.image_checks,
        provenance=results_summary.provenance_from_arguments(
            args.workflow_run, args.source_commit
        ),
    )
    print("\n".join(summary_lines(report)))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
