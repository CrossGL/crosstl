#!/usr/bin/env python3
"""Check that the target compiler keeps the translated arithmetic as written.

WARP executes DXIL, in which every floating-point product and sum is rounded
separately, so ``translate.py`` requests ``floating_point_contraction="off"``.
This check inspects what each target compiler produces from every translated
artifact of a run, using the same compile options as the runtime:

- Metal: the LLVM IR of the offline compiler (``-fmetal-math-mode=safe
  -fmetal-math-fp32-functions=fast``) may contain no fused multiply-add
  (``llvm.fmuladd``, ``llvm.fma``, ``air.fma``), no ``air.mix`` and no
  ``air.dot`` over values computed at run time, and no floating-point
  instruction with a contraction or reassociation flag.
- OpenGL: the driver compiles each artifact through the native backend
  library with ``MESA_GLSL=dump``. Mesa prints the NIR it compiles, marking
  exact instructions with ``!``; every floating-point add, multiply,
  multiply-add, interpolation and dot product must be exact, which keeps the
  driver from fusing or reassociating it.

The report ``exactness.json`` lists every manifest shader with its artifact
hash and counts. :func:`verify` checks a report against its manifest; the
qualification recomputes it from the uploaded report and manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "computesharp-arithmetic-exactness"
METAL_FLAGS = (
    "-fmetal-math-mode=safe",
    "-fmetal-math-fp32-functions=fast",
    "-O2",
)
# Calls that evaluate a product and a sum with one rounding.
METAL_FUSED_CALLS = re.compile(
    r"@(?:llvm\.fmuladd|llvm\.fma|air\.(?:fast_)?fma|air\.(?:fast_)?mix)\."
)
METAL_DOT_CALL = re.compile(r"@air\.dot\.v\d+f(?:16|32)\(([^)]*)\)")
METAL_FLOAT_INSTRUCTION = re.compile(
    r"= (fadd|fsub|fmul|fdiv|fneg)((?: [a-z]+)*) (?:half|float|double|<)"
)
METAL_CONTRACTING_FLAGS = {"contract", "fast", "reassoc"}
NIR_INSTRUCTION = re.compile(r"^\s*\d+(?:x\d+)?\s+%\d+ = (\w+)(!?) ")
NIR_FLOAT_ARITHMETIC = {
    "fadd",
    "fsub",
    "fmul",
    "ffma",
    "flrp",
    "fdot2",
    "fdot3",
    "fdot4",
    "fdot5",
    "fdot8",
    "fdot16",
    "fdph",
}
NIR_SHADER_HEADER = "shader: MESA_SHADER_COMPUTE"


class ExactnessError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_metal_ir(text: str) -> dict[str, int]:
    """Count float instructions and contracting operations in LLVM IR."""
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        if line.startswith("declare ") or "call " not in line and "= f" not in line:
            continue
        if "call " in line:
            counts["fusedCalls"] += len(METAL_FUSED_CALLS.findall(line))
            counts["fusedCalls"] += sum(
                "%" in operands for operands in METAL_DOT_CALL.findall(line)
            )
        match = METAL_FLOAT_INSTRUCTION.search(line)
        if match is None:
            continue
        counts["floatOperations"] += 1
        if set(match.group(2).split()) & METAL_CONTRACTING_FLAGS:
            counts["contractingFlags"] += 1
    return {
        "floatOperations": counts["floatOperations"],
        "contractible": counts["contractingFlags"] + counts["fusedCalls"],
        "fusedCalls": counts["fusedCalls"],
    }


def analyze_nir(text: str) -> dict[str, int]:
    """Count exact and inexact float arithmetic in a Mesa NIR dump."""
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        if line.startswith(NIR_SHADER_HEADER):
            counts["nirShaders"] += 1
            continue
        match = NIR_INSTRUCTION.match(line)
        if match is None or match.group(1) not in NIR_FLOAT_ARITHMETIC:
            continue
        counts["floatOperations"] += 1
        if not match.group(2):
            counts["contractible"] += 1
    return {
        "floatOperations": counts["floatOperations"],
        "contractible": counts["contractible"],
        "nirShaders": counts["nirShaders"],
    }


def _manifest_shaders(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(manifest["shaders"].items())


def metal_compiler_version() -> str:
    completed = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "--version"],
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip().splitlines()[0]


def _metal_entry(artifacts: Path, shader: dict[str, Any]) -> dict[str, Any]:
    source = artifacts / shader["artifact"]
    entry: dict[str, Any] = {"artifactSha256": sha256_file(source)}
    with tempfile.TemporaryDirectory() as scratch:
        ir = Path(scratch) / "shader.ll"
        completed = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", *METAL_FLAGS, "-S", "-emit-llvm"]
            + [str(source), "-o", str(ir)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            entry.update(
                status="compile-failed",
                log=(completed.stderr or completed.stdout)[-2000:],
            )
            return entry
        entry.update(analyze_metal_ir(ir.read_text(encoding="utf-8")))
    entry["status"] = "exact" if entry["contractible"] == 0 else "contractible"
    return entry


def _opengl_entry(
    artifacts: Path, shader: dict[str, Any], library: Path
) -> dict[str, Any]:
    source = artifacts / shader["artifact"]
    entry: dict[str, Any] = {"artifactSha256": sha256_file(source)}
    threads = shader.get("threads") or [1, 1, 1]
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--compile-opengl",
            str(library),
            str(source),
            *(str(value) for value in threads),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(os.environ, MESA_GLSL="dump", MESA_SHADER_CACHE_DISABLE="true"),
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        entry.update(status="compile-failed", log=output[-2000:])
        return entry
    entry.update(analyze_nir(output))
    if entry["nirShaders"] == 0:
        entry["status"] = "no-nir"
        entry["log"] = output[-2000:]
    else:
        entry["status"] = "exact" if entry["contractible"] == 0 else "contractible"
    return entry


def compile_opengl(library_path: str, source_path: str, threads: list[str]) -> int:
    """Compile one artifact with the runtime's native OpenGL backend."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_tests import _DeviceInfo  # noqa: PLC0415

    library = ctypes.CDLL(library_path)
    library.cs_device_create.restype = ctypes.c_int
    library.cs_device_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_DeviceInfo),
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    library.cs_pipeline_create.restype = ctypes.c_int
    library.cs_pipeline_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    library.cs_pipeline_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.cs_device_destroy.argtypes = [ctypes.c_void_p]
    device = ctypes.c_void_p()
    info = _DeviceInfo()
    message = ctypes.create_string_buffer(4096)
    if library.cs_device_create(
        ctypes.byref(device), ctypes.byref(info), message, len(message)
    ):
        print(message.value.decode("utf-8", "replace"), file=sys.stderr)
        return 2
    source = Path(source_path).read_bytes()
    pipeline = ctypes.c_void_p()
    status = library.cs_pipeline_create(
        device,
        source,
        b"main",
        *(int(value) for value in threads),
        ctypes.byref(pipeline),
        message,
        len(message),
    )
    if status:
        print(message.value.decode("utf-8", "replace"), file=sys.stderr)
    else:
        library.cs_pipeline_destroy(device, pipeline)
    library.cs_device_destroy(device)
    return 3 if status else 0


def describe_device(device: dict[str, Any] | None) -> str | None:
    if not device:
        return None
    return " ".join(part for part in (device.get("name"), device.get("driver")) if part)


def opengl_device(library: Path) -> dict[str, Any] | None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_tests  # noqa: PLC0415

    check = run_tests.preflight(library, {"target": "opengl", "capabilities": {}})
    return check.get("device")


def analyze(
    target: str,
    artifacts: Path,
    *,
    library: Path | None = None,
    device: dict[str, Any] | None = None,
    workers: int = 0,
    require_float_operations: bool = True,
) -> dict[str, Any]:
    """Compile every translated artifact of a manifest and count its arithmetic.

    ``device`` is the preflight description of the device the OpenGL backend
    creates; without it the backend is queried once more.
    ``require_float_operations`` fails a report that found no floating-point
    operation at all, which would mean the analysis saw nothing; a suite
    whose shaders do no float arithmetic states why it turns this off.
    """
    manifest_path = artifacts / target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shaders = _manifest_shaders(manifest)
    entries: dict[str, dict[str, Any]] = {}
    pending = {}
    workers = workers or min(8, os.cpu_count() or 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for shader_id, shader in shaders:
            if shader.get("status") != "translated":
                entries[shader_id] = {
                    "status": "not-compiled",
                    "manifestStatus": shader.get("status"),
                }
                continue
            if target == "metal":
                pending[shader_id] = pool.submit(_metal_entry, artifacts, shader)
            else:
                assert library is not None, "the OpenGL check needs the backend"
                pending[shader_id] = pool.submit(
                    _opengl_entry, artifacts, shader, library
                )
        for shader_id, future in pending.items():
            entries[shader_id] = future.result()
    if target == "metal":
        tool = {"compiler": metal_compiler_version(), "flags": list(METAL_FLAGS)}
    else:
        tool = {
            "driver": describe_device(device or opengl_device(library)),
            "dump": "MESA_GLSL=dump",
        }
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": KIND,
        "target": target,
        "tool": tool,
        "manifestSha256": sha256_file(manifest_path),
        "floatOperationsRequired": require_float_operations,
        "artifacts": dict(sorted(entries.items())),
    }
    report["totals"] = totals(report)
    try:
        verify(
            report,
            manifest,
            manifest_sha256=report["manifestSha256"],
            require_float_operations=require_float_operations,
        )
    except ExactnessError as exc:
        report["status"] = "failed"
        report["errors"] = exc.errors[:100]
    else:
        report["status"] = "passed"
    return report


def totals(report: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in report["artifacts"].values():
        counts[entry["status"]] += 1
        counts["floatOperations"] += entry.get("floatOperations", 0)
        counts["contractible"] += entry.get("contractible", 0)
    return dict(sorted(counts.items()))


def verify(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    manifest_sha256: str | None = None,
    require_float_operations: bool = True,
) -> dict[str, Any]:
    """Check a report against the manifest whose artifacts it analyzed."""
    errors: list[str] = []
    if report.get("kind") != KIND or report.get("schemaVersion") != SCHEMA_VERSION:
        raise ExactnessError(["not an arithmetic exactness report"])
    if manifest_sha256 is not None and report.get("manifestSha256") != manifest_sha256:
        errors.append("the report analyzed a different manifest")
    entries = report.get("artifacts", {})
    shaders = dict(_manifest_shaders(manifest))
    for shader_id in sorted(set(entries) - set(shaders)):
        errors.append(f"{shader_id}: not in the manifest")
    analyzed = 0
    for shader_id, shader in sorted(shaders.items()):
        entry = entries.get(shader_id)
        if entry is None:
            errors.append(f"{shader_id}: not analyzed")
            continue
        if shader.get("status") != "translated":
            if entry.get("status") != "not-compiled":
                errors.append(
                    f"{shader_id}: the manifest records {shader.get('status')} "
                    f"but the report records {entry.get('status')}"
                )
            continue
        if entry.get("artifactSha256") != shader.get("artifactSha256"):
            errors.append(f"{shader_id}: the analyzed artifact differs")
        if entry.get("status") != "exact":
            errors.append(f"{shader_id}: {entry.get('status')}")
            continue
        if entry.get("contractible") != 0:
            errors.append(f"{shader_id}: {entry.get('contractible')} contractible")
        analyzed += 1
    counted = totals({"artifacts": entries})
    if require_float_operations and analyzed and counted.get("floatOperations", 0) == 0:
        errors.append("no floating-point operation was found in any artifact")
    if errors:
        raise ExactnessError(errors)
    return {
        "status": "passed",
        "target": report.get("target"),
        "tool": report.get("tool"),
        "exactArtifacts": analyzed,
        "notCompiled": counted.get("not-compiled", 0),
        "floatOperations": counted.get("floatOperations", 0),
        "contractible": counted.get("contractible", 0),
    }


def verify_file(
    report_path: Path,
    manifest_path: Path,
    *,
    target: str,
    require_float_operations: bool = True,
) -> dict[str, Any]:
    """Check the ``exactness.json`` of a run against the run's manifest."""
    if not report_path.is_file():
        raise ExactnessError([f"missing {report_path}"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("target") != target:
        raise ExactnessError(
            [f"the report is for {report.get('target')!r}, not {target!r}"]
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return verify(
        report,
        manifest,
        manifest_sha256=sha256_file(manifest_path),
        require_float_operations=require_float_operations,
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--compile-opengl":
        return compile_opengl(argv[1], argv[2], argv[3:6])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("metal", "opengl"))
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--library", type=Path, help="native OpenGL backend")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.target == "opengl" and args.library is None:
        parser.error("--library is required for OpenGL")
    report = analyze(args.target, args.artifacts, library=args.library)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], **report["totals"]}, indent=2))
    for error in report.get("errors", [])[:20]:
        print(f"  {error}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
