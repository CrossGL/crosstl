#!/usr/bin/env python3
"""Compare pinned affine kernels with original Metal and exact reference bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from demos.integrations.mlx.run_metal_host import run
from tools.compile_artifact_bundle import _bundle_records, _contract

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
SOURCE = "mlx/backend/metal/kernels/quantized.metal"
SOURCE_SHA256 = "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd"
DEFAULT_CONTRACT = Path(__file__).parent / "contracts/quantized.metal-roundtrip.json"
GUARD = bytes.fromhex("1937a5c3e28b4d6f019284765abcdeff")
OPERATIONS = ("affine_quantize", "affine_dequantize")
DATA_TYPES = ("float", "float16_t", "bfloat16_t")
GROUP_SIZES = (32, 64, 128)
BIT_WIDTHS = (2, 3, 4, 5, 6, 8)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _encode(values, dtype):
    if dtype == "float":
        return struct.pack("<" + "f" * len(values), *values)
    if dtype == "float16_t":
        return struct.pack("<" + "e" * len(values), *values)
    _require(dtype == "bfloat16_t", "Unsupported affine scalar type")
    words = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
    rounded = [(word + 0x7FFF + ((word >> 16) & 1)) >> 16 for word in words]
    return struct.pack("<" + "H" * len(rounded), *rounded)


def _pack_codes(codes, bits):
    _require(bits in BIT_WIDTHS, "Unsupported affine bit width")
    _require(len(codes) * bits % 8 == 0, "Packed values must fill complete bytes")
    result = 0
    for index, value in enumerate(codes):
        _require(type(value) is int and 0 <= value < (1 << bits), "Invalid packed code")
        result |= value << (index * bits)
    return result.to_bytes(len(codes) * bits // 8, "little")


def _selected_entries(contract):
    entries = [entry for entry in contract["entries"] if entry["variant"] in OPERATIONS]
    expected = {
        (operation, dtype, group, bits)
        for operation in OPERATIONS
        for dtype in DATA_TYPES
        for group in GROUP_SIZES
        for bits in BIT_WIDTHS
    }
    actual = {
        (entry["variant"], entry["dataType"], entry["groupSize"], entry["bitWidth"])
        for entry in entries
    }
    _require(
        len(entries) == len(expected) and actual == expected, "Affine coverage differs"
    )
    return entries


def _dataset(entry):
    dtype, group, bits = entry["dataType"], entry["groupSize"], entry["bitWidth"]
    _require(
        dtype in DATA_TYPES
        and type(group) is int
        and group in GROUP_SIZES
        and type(bits) is int
        and bits in BIT_WIDTHS,
        "Invalid affine configuration",
    )
    bins = (1 << bits) - 1
    codes = [(index * 17) % (bins + 1) for index in range(group)]
    _require(
        min(codes) == 0 and max(codes) == bins, "Dataset omits a quantization endpoint"
    )
    if entry["variant"] == "affine_quantize":
        # Exact dyadic inputs exercise both scale signs and a zero-width range.
        values = [
            factor * value for factor in (1.0, -1.0, 0.25, -0.5, 0.0) for value in codes
        ]
        expected_codes = [bins - value for _ in range(4) for value in codes] + [
            0
        ] * group
        eps = struct.unpack("<f", struct.pack("<f", 1e-7))[0]
        scales = [-1.0, 1.0, -0.25, 0.5, -eps]
        biases = [float(bins), float(-bins), bins * 0.25, bins * -0.5, 0.0]
        expected = {
            1: _pack_codes(expected_codes, bits),
            2: _encode(scales, dtype),
            3: _encode(biases, dtype),
        }
        inputs = [_encode(values, dtype)] + [
            b"\xa5" * len(expected[slot]) for slot in range(1, 4)
        ]
        geometry = {"workgroupCount": [1, 5, 1], "workgroupSize": [32, 1, 1]}
    else:
        _require(
            entry["variant"] == "affine_dequantize", "Unsupported affine operation"
        )
        scales = [-1.0, 1.0, 0.25, -0.5, 0.0]
        biases = [float(bins), float(-bins), 0.0, bins * 0.5, 3.0]
        values = [
            scale * value + bias
            for scale, bias in zip(scales, biases)
            for value in codes
        ]
        expected = {3: _encode(values, dtype)}
        inputs = [
            _pack_codes(codes * 5, bits),
            _encode(scales, dtype),
            _encode(biases, dtype),
            b"\xa5" * len(expected[3]),
        ]
        pack_factor = {2: 4, 3: 8, 4: 2, 5: 8, 6: 4, 8: 1}[bits]
        geometry = {
            "workgroupCount": [group // pack_factor, 5, 1],
            "workgroupSize": [1, 1, 1],
        }
    return (
        [blob + GUARD for blob in inputs],
        {slot: blob + GUARD for slot, blob in expected.items()},
        geometry,
    )


def _verify_checkout(mlx_root):
    revision = subprocess.check_output(
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    _require(revision == MLX_COMMIT, "MLX revision differs")
    changes = subprocess.check_output(
        [
            "git",
            "-C",
            str(mlx_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "mlx/backend/metal/kernels",
        ],
        text=True,
        timeout=30,
    )
    _require(not changes, "MLX kernel tree is modified")
    _require(_sha256(mlx_root / SOURCE) == SOURCE_SHA256, "MLX source hash differs")


def _compile_source(source, output, mlx_root, *, original=False):
    air, library = output / "kernel.air", output / "kernel.metallib"
    run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "metal",
            "-std=metal3.1",
            "-fno-fast-math",
            *([] if original else ["-Werror"]),
            "-I",
            mlx_root,
            "-c",
            source,
            "-o",
            air,
        ],
        output,
        "compile",
        timeout=600 if original else 180,
    )
    if not original:
        _require(
            not (output / "compile.stdout").read_bytes()
            and not (output / "compile.stderr").read_bytes(),
            "Generated Metal compiler emitted diagnostics",
        )
    _require(
        air.is_file() and air.stat().st_size > 0, "Metal compiler produced no object"
    )
    run(["xcrun", "--sdk", "macosx", "metallib", air, "-o", library], output, "link")
    _require(
        library.is_file() and library.stat().st_size > 0,
        "Metal linker produced no library",
    )
    return library


def _check_readbacks(readback, inputs, expected):
    errors, identities = [], []
    for slot, input_data in enumerate(inputs):
        path = readback / f"buffer-{slot}.bin"
        data = path.read_bytes()
        wanted = expected.get(slot, input_data)
        identities.append(_sha256(path))
        if data != wanted:
            errors.append(
                {
                    "slot": slot,
                    "size": len(data),
                    "expectedSize": len(wanted),
                    "guardIntact": data[-len(GUARD) :] == GUARD,
                    "firstDifferences": [
                        index
                        for index, pair in enumerate(zip(data, wanted))
                        if pair[0] != pair[1]
                    ][:16],
                }
            )
    return {"errors": errors, "readbacks": identities}


def prove(mlx_root, bundle_root, contract_path, output):
    mlx_root, bundle_root, contract_path, output = (
        Path(path).resolve() for path in (mlx_root, bundle_root, contract_path, output)
    )
    for source in (mlx_root, bundle_root, contract_path):
        _require(
            output != source
            and source not in output.parents
            and output not in source.parents,
            "Proof output must be separate from inputs",
        )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schemaVersion": 1,
        "kind": "mlx-affine-metal-roundtrip-proof",
        "status": "running",
        "commit": MLX_COMMIT,
        "records": [],
        "failures": [],
        "numericalExecution": False,
        "fullQuantizedCorpus": False,
        "fullUpstreamSuite": False,
    }
    report_path = output / "report.json"
    _write_json(report_path, report)
    try:
        _verify_checkout(mlx_root)
        contract, expected, contract_hash = _contract(contract_path)
        _require(
            contract["commit"] == MLX_COMMIT
            and contract["source"] == SOURCE
            and contract["sourceSha256"] == SOURCE_SHA256
            and contract["target"] == "metal",
            "Quantized source contract differs",
        )
        entries = _selected_entries(contract)
        artifacts = {
            record["entryPoint"]: record
            for record in _bundle_records(bundle_root, expected, contract_hash)
        }
        report.update(
            contractSha256=contract_hash,
            sourceSha256=SOURCE_SHA256,
            bundleArtifactCount=len(artifacts),
            expectedEntryCount=len(entries),
        )
        runner = output / "metal-readback"
        run(
            [
                "swiftc",
                ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
                "-o",
                runner,
            ],
            output,
            "runner",
        )
        original = _compile_source(
            mlx_root / SOURCE, output / "original", mlx_root, original=True
        )
        for entry in entries:
            name = entry["entryPoint"]
            artifact = artifacts[name]
            source = Path(artifact["path"])
            _require(
                _sha256(source) == entry["sha256"]
                and source.stat().st_size == entry["sizeBytes"],
                "Generated source changed",
            )
            directory = output / hashlib.sha256(name.encode()).hexdigest()
            generated = _compile_source(source, directory / "generated", mlx_root)
            inputs, expected_outputs, geometry = _dataset(entry)
            for slot, data in enumerate(inputs):
                (directory / f"input-{slot}.bin").write_bytes(data)
            for slot, data in expected_outputs.items():
                (directory / f"expected-{slot}.bin").write_bytes(data)
            request = {
                "buffers": [str(directory / f"input-{slot}.bin") for slot in range(4)],
                **geometry,
                "simdWidth": 32,
            }
            request_path = directory / "request.json"
            _write_json(request_path, request)
            results = {}
            for label, library in (("original", original), ("generated", generated)):
                readback = directory / label / "readback"
                run(
                    [runner, library, name, request_path, readback],
                    directory,
                    f"execute-{label}",
                )
                report["numericalExecution"] = True
                results[label] = {
                    **_check_readbacks(readback, inputs, expected_outputs),
                    "moduleSha256": _sha256(library),
                }
            record = {
                "entryPoint": name,
                "generatedSha256": _sha256(source),
                "caseGroups": 5,
                "scalarValues": entry["groupSize"] * 5,
                "geometry": geometry,
                "outputSlots": sorted(expected_outputs),
                "results": results,
                "nativeByteParity": (
                    results["original"]["readbacks"]
                    == results["generated"]["readbacks"]
                ),
            }
            report["records"].append(record)
            if not record["nativeByteParity"] or any(
                result["errors"] for result in results.values()
            ):
                report["failures"].append(name)
            _write_json(report_path, report)
        _verify_checkout(mlx_root)
        _require(
            _sha256(contract_path) == contract_hash, "Contract changed during execution"
        )
        _bundle_records(bundle_root, expected, contract_hash)
        _require(not report["failures"], "Native affine results differ")
        report["status"] = "passed"
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        report["status"] = "failed"
        report["failures"].append(str(error))
    _write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prove(args.mlx_root, args.bundle_root, args.contract, args.output_dir)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"}, indent=2
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
