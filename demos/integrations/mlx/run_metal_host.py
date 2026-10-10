#!/usr/bin/env python3
"""Run pinned MLX host operations with independently translated Metal entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

MLX_COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
HERE = Path(__file__).resolve().parent
SOURCE = "mlx/backend/metal/kernels/binary.metal"
SHAPES = (
    "ss",
    "sv",
    "vs",
    "vv",
    "sv2",
    "vs2",
    "vv2",
    "g1",
    "g1large",
    "g2",
    "g2large",
    "g3",
    "g3large",
    "gn2",
    "gn4large",
)
ENTRIES = {f"{shape}_Powercomplex64" for shape in SHAPES}
HOST_DATASETS = ("ordinary", "positive-zero", "negative-zero")
HOST_LAYOUTS = {
    "ss": [],
    "sv": [7],
    "vs": [7],
    "vv": [7],
    "g1": [7],
    "g2": [3, 5],
    "g3": [2, 3, 5],
    "gn2": [2, 2, 3, 5],
}
SOURCE_HASHES = {
    "mlx/backend/metal/device.cpp": (
        "d2254ea5cb2ac282f65c2869c6ef56decc97669e2a5fc26fba9d831b60569c9f",
        "25981c452fc28eb4a8f3273b87966578df03f2a891ec901630dfe52beaca8af7",
    ),
    "mlx/backend/metal/device.h": (
        "ab1e07495b916a86eb762cddc68c6f1ae5e4c61020baacdea3c0dadd59150603",
        "d70264da6a7c4ace820037c65af0fec75acb595e4a0cfa297a9e258d8515d402",
    ),
}
HEADER = "mlx/backend/metal/metal_library_overrides.h"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(command, output, name, *, cwd=None, env=None, timeout=120, check=True):
    """Retain every command result, including bounded failures."""
    output.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in command]
    record = {"command": command, "timeoutSeconds": timeout, "cwd": str(cwd)}
    if env is not None:
        record["runtimeEnvironment"] = {
            name: env[name]
            for name in (
                "DEVICE",
                "CROSTL_METAL_LIBRARY_OVERRIDES",
                "CROSTL_METAL_LIBRARY_TRACE",
            )
            if name in env
        }
    save_json(output / f"{name}.json", record)
    with (output / f"{name}.stdout").open("w") as stdout, (
        output / f"{name}.stderr"
    ).open("w") as stderr:
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        ) as process:
            try:
                record["returncode"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                record["returncode"] = process.wait()
                record["timedOut"] = True
    save_json(output / f"{name}.json", record)
    if record.get("timedOut") or (check and record["returncode"]):
        raise RuntimeError(f"{name} failed; inspect {output / (name + '.stderr')}")
    return record


def verify_checkout(root, *, patched):
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if head != MLX_COMMIT:
        raise ValueError(f"Expected MLX {MLX_COMMIT}, got {head}")
    modified = subprocess.check_output(
        ["git", "-C", str(root), "diff", "HEAD", "--name-only"],
        text=True,
        timeout=30,
    ).splitlines()
    if set(modified) != (set(SOURCE_HASHES) if patched else set()):
        raise ValueError("MLX has unexpected tracked source changes")
    for path, hashes in SOURCE_HASHES.items():
        if digest(root / path) != hashes[int(patched)]:
            raise ValueError(f"Unexpected MLX runtime source: {path}")
    if patched and digest(root / HEADER) != digest(HERE / "metal_library_overrides.h"):
        raise ValueError("MLX override header does not match this harness")


def prepare(root, output):
    verify_checkout(root, patched=False)
    header = root / HEADER
    if header.exists():
        raise ValueError(f"Refusing to replace existing header: {header}")
    patch = HERE / "metal-library-overrides.patch"
    run(["git", "apply", "--check", patch], output, "patch-check", cwd=root)
    run(["git", "apply", patch], output, "patch-apply", cwd=root)
    shutil.copyfile(HERE / "metal_library_overrides.h", header)
    verify_checkout(root, patched=True)
    save_json(
        output / "source-adaptation.json",
        {
            "commit": MLX_COMMIT,
            "patchSha256": digest(patch),
            "headerSha256": digest(header),
            "changes": {path: digest(root / path) for path in SOURCE_HASHES},
        },
    )


def compile_libraries(root, output):
    from crosstl.project import load_project_config, translate_project

    libraries = output / "libraries"
    libraries.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=".host-metal-", dir=root) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(
            f"""[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["metal"]
output_dir = "{work.name}/out"
[project.entry_points]
"{SOURCE}" = {json.dumps(sorted(ENTRIES))}
[project.entry_workgroup_size_rules."{SOURCE}"]
"*_Powercomplex64" = [1, 1, 1]
[project.source_options.metal]
max_template_specializations = 64
max_template_materialization_work = 4096
""",
            encoding="utf-8",
        )
        try:
            report = translate_project(
                load_project_config(root, config),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            artifacts = payload["artifacts"]
            if (
                payload["summary"]["failedCount"]
                or {artifact["entryPoint"]["source"] for artifact in artifacts}
                != ENTRIES
                or len(artifacts) != len(ENTRIES)
            ):
                raise RuntimeError(
                    "Translation did not produce the exact required entries"
                )
            records = []
            for artifact in artifacts:
                entry = artifact["entryPoint"]["source"]
                source = root / artifact["path"]
                air = libraries / f"{entry}.air"
                library = libraries / f"{entry}.metallib"
                run(
                    [
                        "xcrun",
                        "--sdk",
                        "macosx",
                        "metal",
                        "-Werror",
                        "-fno-fast-math",
                        "-c",
                        source,
                        "-o",
                        air,
                    ],
                    output / "logs",
                    f"compile-{entry}",
                )
                run(
                    ["xcrun", "--sdk", "macosx", "metallib", air, "-o", library],
                    output / "logs",
                    f"link-{entry}",
                )
                records.append(
                    {
                        "entry": entry,
                        "sourceSha256": digest(source),
                        "librarySha256": digest(library),
                    }
                )
        finally:
            shutil.copytree(work, output / "translation")
    (libraries / "libraries.txt").write_text(
        "".join(entry + "\n" for entry in sorted(ENTRIES)), encoding="utf-8"
    )
    save_json(output / "libraries.json", records)
    combined = output / "combined.metallib"
    run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "metallib",
            *[libraries / f"{entry}.air" for entry in sorted(ENTRIES)],
            "-o",
            combined,
        ],
        output / "logs",
        "link-combined",
    )
    combined_libraries = output / "combined-libraries"
    combined_libraries.mkdir()
    shutil.copyfile(libraries / "libraries.txt", combined_libraries / "libraries.txt")
    combined_hash = digest(combined)
    combined_records = []
    for record in records:
        shutil.copyfile(combined, combined_libraries / f"{record['entry']}.metallib")
        combined_records.append({**record, "librarySha256": combined_hash})
    save_json(output / "combined-libraries.json", combined_records)
    return combined_libraries, combined_records


def parse_trace(path):
    libraries, dispatches = set(), []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[0] == "library" and fields[1] in ENTRIES:
            libraries.add(fields[1])
        elif (
            len(fields) == 5
            and fields[0] == "dispatch"
            and fields[1] in libraries
            and fields[2] in {"threads", "threadgroups"}
        ):
            dimensions = [list(map(int, value.split(","))) for value in fields[3:]]
            if any(
                len(size) != 3 or any(value <= 0 for value in size)
                for size in dimensions
            ):
                raise ValueError("Invalid native dispatch dimensions")
            dispatches.append(
                {
                    "entry": fields[1],
                    "kind": fields[2],
                    "grid": dimensions[0],
                    "group": dimensions[1],
                }
            )
        else:
            raise ValueError("Invalid or unproven Metal override trace event")
    if not dispatches:
        raise ValueError("No translated Metal dispatch was recorded")
    return dispatches


def unittest_counts(path):
    text = Path(path).read_text(encoding="utf-8")
    totals = re.findall(r"^Ran (\d+) tests? in ", text, re.MULTILINE)
    successful = re.findall(r"^OK(?: \(skipped=(\d+)\))?$", text, re.MULTILINE)
    if len(totals) != 1 or len(successful) != 1 or int(totals[0]) <= 0:
        raise ValueError("Missing successful upstream unittest accounting")
    return {"total": int(totals[0]), "skipped": int(successful[0] or 0)}


def verify_host_workloads(results, dispatches, dataset):
    """Check retained values independently of the workload's pass flags."""
    if dataset not in HOST_DATASETS:
        raise ValueError("Unknown host workload dataset")
    if Counter(item["entry"] for item in dispatches) != Counter(
        {f"{shape}_Powercomplex64": 1 for shape in HOST_LAYOUTS}
    ):
        raise ValueError("Host workloads did not dispatch each expected entry once")
    if Counter(item["shape"] for item in results) != Counter(
        {shape: 1 for shape in HOST_LAYOUTS}
    ):
        raise ValueError("Host workload numerical evidence is incomplete")

    def complex_value(pair):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)
        ):
            raise ValueError("Host workload requires finite complex readbacks")
        return complex(*pair)

    for item in results:
        layout = HOST_LAYOUTS[item["shape"]]
        count = math.prod(layout)
        if (
            item["dataset"] != dataset
            or item["arrayShape"] != layout
            or item["dtype"] != "complex64"
            or type(item["outputs"]) is not int
            or item["outputs"] != count
            or item["matched"] is not True
        ):
            raise ValueError("Host workload layout or numerical evidence is invalid")
        values = [item[key] for key in ("base", "exponent", "expected", "actual")]
        if any(not isinstance(v, list) or len(v) != count for v in values):
            raise ValueError("Host workload readback count is incomplete")
        for pairs in zip(*values):
            base, exponent, expected, actual = map(complex_value, pairs)
            if dataset != "ordinary" and (
                base.real >= 0
                or base.imag != 0
                or math.copysign(1, base.imag)
                != (-1 if dataset == "negative-zero" else 1)
                or not 0 < exponent.real < 1
                or exponent.imag != 0
            ):
                raise ValueError("Host workload lost its signed-zero inputs")
            reference = base**exponent
            bound = 5e-5 * max(1.0, abs(reference))
            if abs(expected - reference) > bound or abs(actual - reference) > bound:
                raise ValueError("Host workload readback does not match the reference")


def verify(root, python, output):
    verify_checkout(root, patched=True)
    output.mkdir(parents=True, exist_ok=True)
    libraries, artifacts = compile_libraries(root, output)
    env = dict(os.environ, DEVICE="gpu")
    for name in ("CROSTL_METAL_LIBRARY_OVERRIDES", "CROSTL_METAL_LIBRARY_TRACE"):
        env.pop(name, None)
    identity_code = (
        "import json; import mlx.core as mx; "
        "print(json.dumps({'module':mx.__file__, 'device':mx.default_device().type.name, "
        "'metalAvailable':mx.metal.is_available(), 'deviceInfo':mx.device_info()}))"
    )
    run([python, "-c", identity_code], output, "runtime-identity", env=env)
    identity = json.loads((output / "runtime-identity.stdout").read_text())
    if (
        identity["device"] != "gpu"
        or identity["metalAvailable"] is not True
        or root not in Path(identity["module"]).resolve().parents
    ):
        raise ValueError("MLX runtime is not the adapted checkout on a Metal device")
    tests = [str(python), str(HERE / "unittest_evidence.py"), "--evidence-dir"]
    cwd = root / "python/tests"
    run(
        [*tests, output / "upstream-original-failures", "test_ops"],
        output,
        "upstream-original",
        cwd=cwd,
        env=env,
        timeout=1800,
    )
    baseline = unittest_counts(output / "upstream-original.stderr")
    trace = output / "translated-dispatch.tsv"
    trace.touch(exist_ok=False)
    env.update(
        CROSTL_METAL_LIBRARY_OVERRIDES=str(libraries),
        CROSTL_METAL_LIBRARY_TRACE=str(trace),
    )
    run(
        [*tests, output / "upstream-translated-failures", "test_ops"],
        output,
        "upstream-translated",
        cwd=cwd,
        env=env,
        timeout=1800,
    )
    translated = unittest_counts(output / "upstream-translated.stderr")
    if baseline != translated:
        raise ValueError("Upstream test or skip accounting changed with translation")
    dispatches = parse_trace(trace)
    if "ss_Powercomplex64" not in {item["entry"] for item in dispatches}:
        raise ValueError("Upstream complex-power test did not use translated code")
    host_results, host_dispatches = [], []
    for dataset in HOST_DATASETS:
        workload_trace = output / f"host-workloads-{dataset}.tsv"
        workload_trace.touch(exist_ok=False)
        env["CROSTL_METAL_LIBRARY_TRACE"] = str(workload_trace)
        workload_values = output / f"host-workloads-{dataset}.json"
        run(
            [
                python,
                HERE / "metal_host_workloads.py",
                "--dataset",
                dataset,
                workload_values,
            ],
            output,
            f"execute-host-workloads-{dataset}",
            env=env,
        )
        records = json.loads(workload_values.read_text())
        dispatch_records = parse_trace(workload_trace)
        verify_host_workloads(records, dispatch_records, dataset)
        host_results.extend(records)
        host_dispatches.extend(dict(item, dataset=dataset) for item in dispatch_records)
    missing = output / "missing-library"
    missing.mkdir()
    (missing / "libraries.txt").write_text("ss_Powercomplex64\n", encoding="utf-8")
    env["CROSTL_METAL_LIBRARY_OVERRIDES"] = str(missing)
    negative = run(
        [str(python), "-m", "unittest", "test_ops.TestOps.test_complex_power"],
        output,
        "missing-required-library",
        cwd=cwd,
        env=env,
        check=False,
    )
    error = (output / "missing-required-library.stderr").read_text(encoding="utf-8")
    if (
        not negative["returncode"]
        or "Cannot load required translated Metal library" not in error
    ):
        raise ValueError("A missing required library did not fail closed")
    for artifact in artifacts:
        if (
            digest(libraries / (artifact["entry"] + ".metallib"))
            != artifact["librarySha256"]
        ):
            raise ValueError("Compiled library changed during upstream execution")
    verify_checkout(root, patched=True)
    save_json(
        output / "evidence.json",
        {
            "kind": "mlx-selected-metal-host-redirection",
            "schemaVersion": 2,
            "commit": MLX_COMMIT,
            "upstreamModule": "test_ops",
            "runtime": identity,
            "upstreamTestSourceSha256": digest(cwd / "test_ops.py"),
            "original": baseline,
            "translated": translated,
            "dispatches": dispatches,
            "hostWorkloads": host_results,
            "hostWorkloadDispatches": host_dispatches,
            "artifacts": artifacts,
            "missingRequiredLibraryRejected": True,
            "fullUpstreamSuite": False,
            "fullTranslatedBackend": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.mlx_root.resolve(), args.output_dir.resolve())
    else:
        if args.python is None:
            parser.error("verify requires --python for the isolated MLX installation")
        verify(
            args.mlx_root.resolve(), args.python.absolute(), args.output_dir.resolve()
        )


if __name__ == "__main__":
    main()
