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
MATRIX_VARIANTS = {
    "wide": ("affine_qmv_wide_nv_2_kl_8_batch_1", 64, 2),
    "transposed": ("affine_qmm_t_alN_true_batch_0", 128, 3),
    "nontransposed": ("affine_qmm_n_batch_0", 32, 2),
}


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


def _matrix_entries(contract):
    entries = []
    for family, (variant, group, bits) in MATRIX_VARIANTS.items():
        selected = [
            entry
            for entry in contract["entries"]
            if entry["variant"] == variant
            and entry["groupSize"] == group
            and entry["bitWidth"] == bits
        ]
        _require(
            len(selected) == len(DATA_TYPES)
            and {entry["dataType"] for entry in selected} == set(DATA_TYPES),
            "Matrix control coverage differs",
        )
        entries.extend((family, entry) for entry in selected)
    return entries


def _matrix_dataset(dtype, family, layout, fill):
    _require(dtype in DATA_TYPES and family in MATRIX_VARIANTS, "Invalid matrix type")
    _require(layout in ("flat", "broadcast"), "Invalid batch layout")
    _require(
        type(fill) is int and fill in (0xA5, 0x5A), "Invalid output initialization"
    )
    wide = family == "wide"
    _require(wide or layout == "flat", "Only vector controls use broadcast batches")
    batch, m, n, k = (
        (4, 2, 9, 64) if wide else (1, 33, 32, 128 if family == "transposed" else 64)
    )
    _, group, bits = MATRIX_VARIANTS[family]
    rows, columns = (k, n) if family == "nontransposed" else (n, k)
    bins = (1 << bits) - 1
    codes = [
        [
            (row * 3 + column + b) % (bins + 1)
            for row in range(rows)
            for column in range(columns)
        ]
        for b in range(batch)
    ]
    scales = [
        [
            (b + 1) * (0.125 if row % 2 else -0.125)
            for row in range(rows)
            for _ in range(columns // group)
        ]
        for b in range(batch)
    ]
    biases = [
        [
            ((row + b) % 3 - 1) * 0.25
            for row in range(rows)
            for _ in range(columns // group)
        ]
        for b in range(batch)
    ]
    vectors = [
        [
            ((row * 3 + column + b) % 7 - 3) * 0.25
            for row in range(m)
            for column in range(k)
        ]
        for b in range(batch)
    ]
    values = []
    for b in range(batch):
        x_batch, w_batch = (b // 2 * 2, b % 2) if layout == "broadcast" else (b, b)
        for row in range(m):
            for column in range(n):
                total = 0.0
                for inner in range(k):
                    index = (
                        inner * n + column
                        if family == "nontransposed"
                        else column * k + inner
                    )
                    scale_index = index // group
                    weight = (
                        codes[w_batch][index] * scales[w_batch][scale_index]
                        + biases[w_batch][scale_index]
                    )
                    total += vectors[x_batch][row * k + inner] * weight
                values.append(total)
    expected = _encode(values, dtype)
    inputs = [
        _pack_codes([value for part in codes for value in part], bits),
        *[
            _encode([value for part in data for value in part], dtype)
            for data in (scales, biases, vectors)
        ],
        bytes([fill]) * len(expected),
        *[struct.pack("<i", value) for value in (k, n, m)],
    ]
    x_stride = m * k
    w_stride = rows * columns * bits // 32
    scale_stride = rows * columns // group
    if layout == "broadcast":
        shape = [2, 2]
        x_strides, w_strides = [2 * x_stride, 0], [0, w_stride]
        scale_strides = [0, scale_stride]
    else:
        shape = [batch]
        x_strides, w_strides, scale_strides = [x_stride], [w_stride], [scale_stride]
    for code, data in (
        ("i", [len(shape)]),
        ("i", shape),
        ("q", x_strides),
        ("i", [len(shape)]),
        ("i", shape),
        ("q", w_strides),
        ("q", scale_strides),
        ("q", scale_strides),
    ):
        inputs.append(struct.pack("<" + code * len(data), *data))
    geometry = {
        "workgroupCount": [1, 2, batch],
        "workgroupSize": [32, 2, 1] if wide else [32, 2, 2],
    }
    parameters = {
        "batch": batch,
        "M": m,
        "N": n,
        "K": k,
        "bits": bits,
        "groupSize": group,
        "layout": layout,
    }
    return (
        [data + GUARD for data in inputs],
        {4: expected + GUARD},
        geometry,
        parameters,
    )


def _run_matrix_controls(
    contract, artifacts, mlx_root, output, runner, original, report
):
    entries = _matrix_entries(contract)
    report["matrixEntryCount"] = len(entries)
    for family, entry in entries:
        name = entry["entryPoint"]
        # The bundle record is the pinned identity, or in report mode the
        # drifted one _bundle_records accepted.
        source = Path(artifacts[name]["path"])
        _require(_sha256(source) == artifacts[name]["sha256"], "Matrix source changed")
        directory = output / ("matrix-" + hashlib.sha256(name.encode()).hexdigest())
        generated = _compile_source(source, directory / "generated", mlx_root)
        layouts = ("flat", "broadcast") if family == "wide" else ("flat",)
        for layout in layouts:
            for fill in (0xA5, 0x5A):
                work = directory / f"{layout}-{fill}"
                work.mkdir(parents=True)
                inputs, expected, geometry, parameters = _matrix_dataset(
                    entry["dataType"], family, layout, fill
                )
                for slot, data in enumerate(inputs):
                    (work / f"input-{slot}.bin").write_bytes(data)
                (work / "expected-4.bin").write_bytes(expected[4])
                request = {
                    "buffers": [
                        str(work / f"input-{slot}.bin") for slot in range(len(inputs))
                    ],
                    **geometry,
                    "simdWidth": 32,
                }
                _write_json(work / "request.json", request)
                results = {}
                for label, library in (
                    ("original", original),
                    ("generated", generated),
                ):
                    readback = work / label / "readback"
                    run(
                        [runner, library, name, work / "request.json", readback],
                        work,
                        f"execute-{label}",
                    )
                    results[label] = {
                        **_check_readbacks(readback, inputs, expected),
                        "moduleSha256": _sha256(library),
                    }
                record = {
                    "entryPoint": name,
                    "generatedSha256": _sha256(source),
                    "parameters": parameters,
                    "geometry": geometry,
                    "initialization": fill,
                    "results": results,
                    "nativeByteParity": (
                        results["original"]["readbacks"]
                        == results["generated"]["readbacks"]
                    ),
                }
                report["matrixRecords"].append(record)
                if not record["nativeByteParity"] or any(
                    result["errors"] for result in results.values()
                ):
                    report["failures"].append(f"{name}:{layout}:{fill}")
                _write_json(output / "report.json", report)
    _require(len(report["matrixRecords"]) == 24, "Matrix control count differs")


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


def prove(mlx_root, bundle_root, contract_path, output, *, allow_drift=False):
    """Run the bundle's affine kernels against MLX's originals on Metal.

    With ``allow_drift`` (report pin mode) an entry whose bytes differ from its
    pin is executed as exported and named in ``driftedEntries``.
    """
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
        "matrixRecords": [],
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
            for record in _bundle_records(
                bundle_root, expected, contract_hash, allow_drift=allow_drift
            )
        }
        report.update(
            contractSha256=contract_hash,
            sourceSha256=SOURCE_SHA256,
            bundleArtifactCount=len(artifacts),
            driftedEntries=sorted(
                name for name, record in artifacts.items() if "pinnedSha256" in record
            ),
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
                _sha256(source) == artifact["sha256"]
                and source.stat().st_size == artifact["sizeBytes"],
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
        _run_matrix_controls(
            contract, artifacts, mlx_root, output, runner, original, report
        )
        _verify_checkout(mlx_root)
        _require(
            _sha256(contract_path) == contract_hash, "Contract changed during execution"
        )
        _bundle_records(bundle_root, expected, contract_hash, allow_drift=allow_drift)
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
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="execute bundle entries whose bytes differ from their pins (report mode)",
    )
    args = parser.parse_args(argv)
    report = prove(
        args.mlx_root,
        args.bundle_root,
        args.contract,
        args.output_dir,
        allow_drift=args.allow_drift,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("records", "matrixRecords")
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
