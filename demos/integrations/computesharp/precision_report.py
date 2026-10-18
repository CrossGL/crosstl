#!/usr/bin/env python3
"""Compare intrinsic probe results of a translated backend with WARP.

Both directories are written by ``precision/Program.cs`` (``run`` mode): an
``inputs.bin`` of little-endian float32 values, one result file per probe and
``probe.json`` with the device, the generated HLSL hash of every probe and the
content hashes. The two runs must evaluate the same generated HLSL over the
same inputs. ``EXACT_FUNCTIONS`` are translated with the WARP evaluation
(``trigonometric_reduction="warp"``) and must match WARP bit for bit for every
operand whose single-precision revolution ``x / (2 * pi)`` is a normal number
(WARP flushes subnormal intermediates to zero, the targets need not). Every
other probe reports how far the target's native intrinsic is from WARP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EXACT_FUNCTIONS = ("sin", "cos", "tan")
# Smallest operand whose single-precision revolution is normal: 2^-126 * 2 * pi.
SUBNORMAL_REVOLUTION_LIMIT = float(np.finfo(np.float32).tiny) * 2.0 * np.pi
MAX_EXAMPLES = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory: Path) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    meta = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
    inputs_path = directory / meta["inputs"]["file"]
    if _sha256(inputs_path) != meta["inputs"]["sha256"]:
        raise ValueError(f"{inputs_path} does not match probe.json")
    inputs = np.fromfile(inputs_path, dtype="<f4")
    values = {}
    for function in meta["functions"]:
        path = directory / function["file"]
        if _sha256(path) != function["sha256"]:
            raise ValueError(f"{path} does not match probe.json")
        values[function["name"]] = np.fromfile(path, dtype="<f4")
    return meta, inputs, values


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in float32 ulps; NaNs compare equal to NaNs and far from numbers."""

    def ordered(values: np.ndarray) -> np.ndarray:
        bits = values.view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

    distance = np.abs(ordered(a) - ordered(b))
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    distance = np.where(nan_a & nan_b, 0, distance)
    return np.where(nan_a ^ nan_b, np.int64(1) << 40, distance)


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    errors: list[str] = []
    ref_meta, ref_inputs, ref_values = load(reference)
    meta, inputs, values = load(candidate)
    if meta["inputs"]["sha256"] != ref_meta["inputs"]["sha256"]:
        errors.append("the probe inputs differ from the WARP run")
    reference_functions = {f["name"]: f for f in ref_meta["functions"]}
    functions = {f["name"]: f for f in meta["functions"]}
    if sorted(functions) != sorted(reference_functions):
        errors.append(
            f"probe functions differ: {sorted(functions)} vs {sorted(reference_functions)}"
        )
    report: dict[str, Any] = {
        "device": meta["device"],
        "referenceDevice": ref_meta["device"],
        "inputs": meta["inputs"]["count"],
        "functions": {},
    }
    if errors:
        return {"status": "failed", "errors": errors, **report}

    # Operands whose revolution is subnormal are the only ones exempt.
    exempt = (np.abs(inputs) < SUBNORMAL_REVOLUTION_LIMIT) & (inputs != 0)
    for name in sorted(functions):
        if functions[name]["hlslSha256"] != reference_functions[name]["hlslSha256"]:
            errors.append(f"{name}: the generated HLSL differs from the WARP run")
            continue
        distance = ulp_distance(values[name], ref_values[name])
        finite = distance < (np.int64(1) << 40)
        entry: dict[str, Any] = {
            "expression": functions[name]["expression"],
            "identical": int(np.count_nonzero(distance == 0)),
            "nanMismatches": int(np.count_nonzero(~finite)),
        }
        if finite.any():
            p50, p99, p999, largest = np.percentile(
                distance[finite], [50, 99, 99.9, 100]
            )
            entry["ulpDistance"] = {
                "p50": int(p50),
                "p99": int(p99),
                "p99.9": int(p999),
                "max": int(largest),
            }
        if name in EXACT_FUNCTIONS:
            required = ~exempt
            mismatched = np.nonzero(required & (distance != 0))[0]
            entry["required"] = "bit-identical to WARP"
            entry["requiredInputs"] = int(np.count_nonzero(required))
            entry["mismatches"] = int(mismatched.size)
            entry["examples"] = [
                {
                    "input": float(inputs[i]),
                    "value": float(values[name][i]),
                    "warp": float(ref_values[name][i]),
                }
                for i in mismatched[:MAX_EXAMPLES]
            ]
            if mismatched.size:
                errors.append(
                    f"{name}: {mismatched.size} of {int(np.count_nonzero(required))} "
                    "results differ from WARP"
                )
        report["functions"][name] = entry
    report["status"] = "failed" if errors else "passed"
    if errors:
        report["errors"] = errors
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.reference, args.candidate)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
