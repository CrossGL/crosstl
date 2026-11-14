#!/usr/bin/env python3
"""Compare intrinsic probe results of a translated backend with WARP.

Both directories are written by ``precision/Program.cs`` (``run`` mode): an
``inputs.bin`` of little-endian float32 values, one result file per probe and
``probe.json`` with the device, the generated HLSL hash of every probe and the
content hashes. The two runs must evaluate the same generated HLSL over the
same inputs. ``EXACT_FUNCTIONS`` are translated with the WARP evaluation
(``trigonometric_reduction="warp"``) and must match WARP bit for bit for every
operand whose single-precision revolution ``x / (2 * pi)`` is a normal number
(WARP flushes subnormal intermediates to zero, the targets need not).

``ARITHMETIC_MODELS`` are probes that only use additions, subtractions,
multiplications and ``floor``, which the translation evaluates in DXC's
operation order without contraction (``floating_point_contraction="off"``).
An independent CPU model evaluates each one in single precision twice: with
every operand and result flushed to zero when subnormal, as WARP does, and
with IEEE subnormals. WARP must equal the flushing evaluation for every
input, and the target must equal one of the two evaluations for every input,
so the target can only differ from WARP where subnormals are involved. Every
other probe reports how far the target's native intrinsic is from WARP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

EXACT_FUNCTIONS = ("sin", "cos", "tan")
# Smallest operand whose single-precision revolution is normal: 2^-126 * 2 * pi.
SUBNORMAL_REVOLUTION_LIMIT = float(np.finfo(np.float32).tiny) * 2.0 * np.pi
MAX_EXAMPLES = 8
_TINY = np.float32(np.finfo(np.float32).tiny)


class _SinglePrecision:
    """Single-precision operations, optionally flushing subnormals to zero."""

    def __init__(self, flush: bool):
        self.flush = flush

    def value(self, operand: Any) -> np.ndarray:
        result = np.asarray(operand, dtype=np.float32)
        if self.flush:
            result = np.where(
                np.abs(result) < _TINY, np.copysign(np.float32(0), result), result
            ).astype(np.float32)
        return result

    def add(self, a: Any, b: Any) -> np.ndarray:
        return self.value(self.value(a) + self.value(b))

    def sub(self, a: Any, b: Any) -> np.ndarray:
        return self.value(self.value(a) - self.value(b))

    def mul(self, a: Any, b: Any) -> np.ndarray:
        return self.value(self.value(a) * self.value(b))

    def frac(self, a: Any) -> np.ndarray:
        # DXIL Frc: x - floor(x).
        return self.sub(a, self.value(np.floor(self.value(a))))

    def saturate(self, a: Any) -> np.ndarray:
        # DXIL FMax/FMin return the non-NaN operand, so NaN saturates to 0.
        return self.value(np.fmin(np.fmax(self.value(a), np.float32(0)), np.float32(1)))


def _f32(value: float) -> np.float32:
    return np.float32(value)


def _dot2(o: _SinglePrecision, x, y, z) -> np.ndarray:
    return o.add(o.mul(x, _f32(27.619)), o.mul(y, _f32(57.583)))


def _dot3(o: _SinglePrecision, x, y, z) -> np.ndarray:
    return o.add(
        o.add(o.mul(x, _f32(0.2126)), o.mul(y, _f32(0.7152))), o.mul(z, _f32(0.0722))
    )


def _multiply_add(o: _SinglePrecision, x, y, z) -> np.ndarray:
    return o.add(o.mul(x, _f32(27.619)), y)


def _lerp(o: _SinglePrecision, x, y, z) -> np.ndarray:
    # lerp(a, b, s) = a + s * (b - a)
    return o.add(x, o.mul(o.frac(z), o.sub(y, x)))


def _smoothstep(o: _SinglePrecision, x, y, z) -> np.ndarray:
    # smoothstep(0, 1, s): t = saturate(s), t * t * (3 - 2 * t)
    t = o.saturate(o.frac(x))
    return o.mul(o.mul(t, t), o.sub(_f32(3), o.mul(_f32(2), t)))


# name -> (probe expression, model). The two- and three-operand probes read
# x = input[i], y = input[i + 1] and z = input[i + 2], wrapping around.
ARITHMETIC_MODELS: dict[str, tuple[str, Callable[..., np.ndarray]]] = {
    "dot2": ("dot(float2(x, y), float2(27.619, 57.583))", _dot2),
    "dot3": ("dot(float3(x, y, z), float3(0.2126, 0.7152, 0.0722))", _dot3),
    "multiply-add": ("x * 27.619 + y", _multiply_add),
    "mad": ("mad(x, 27.619, y)", _multiply_add),
    "lerp": ("lerp(x, y, frac(z))", _lerp),
    "smoothstep": ("smoothstep(0.0, 1.0, frac(x))", _smoothstep),
}


def arithmetic_model(name: str, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(flushing, ieee)`` single-precision evaluations of an arithmetic probe."""
    _, model = ARITHMETIC_MODELS[name]
    x = inputs.astype(np.float32)
    y = np.roll(x, -1)
    z = np.roll(x, -2)
    with np.errstate(all="ignore"):
        return (
            model(_SinglePrecision(flush=True), x, y, z),
            model(_SinglePrecision(flush=False), x, y, z),
        )


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


def _check_arithmetic(
    name: str,
    entry: dict[str, Any],
    inputs: np.ndarray,
    value: np.ndarray,
    warp: np.ndarray,
    functions: dict[str, Any],
) -> list[str]:
    """Require an arithmetic probe to match WARP up to subnormal flushing."""
    expression, _ = ARITHMETIC_MODELS[name]
    if functions[name]["expression"] != expression:
        return [
            f"{name}: the probe evaluates {functions[name]['expression']!r}, "
            f"the model {expression!r}"
        ]
    flushing, ieee = arithmetic_model(name, inputs)
    errors = []
    model_mismatches = np.nonzero(ulp_distance(warp, flushing) != 0)[0]
    if model_mismatches.size:
        errors.append(
            f"{name}: {model_mismatches.size} WARP results differ from the "
            "single-precision model"
        )
    subnormal = ulp_distance(flushing, ieee) != 0
    allowed = (ulp_distance(value, flushing) == 0) | (ulp_distance(value, ieee) == 0)
    mismatched = np.nonzero(~allowed)[0]
    entry["required"] = "bit-identical to WARP; subnormal flushing may differ"
    entry["requiredInputs"] = int(inputs.size)
    entry["subnormalInputs"] = int(np.count_nonzero(subnormal))
    entry["modelMismatches"] = int(model_mismatches.size)
    entry["mismatches"] = int(mismatched.size)
    entry["examples"] = [
        {
            "input": float(inputs[i]),
            "value": float(value[i]),
            "warp": float(warp[i]),
            "ieee": float(ieee[i]),
        }
        for i in mismatched[:MAX_EXAMPLES]
    ]
    if mismatched.size:
        errors.append(
            f"{name}: {mismatched.size} of {inputs.size} results differ from WARP "
            "beyond subnormal flushing"
        )
    return errors


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
    missing = [
        name
        for name in (*EXACT_FUNCTIONS, *ARITHMETIC_MODELS)
        if name not in functions or name not in reference_functions
    ]
    if missing:
        errors.append(f"required probes are missing: {missing}")
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
        if name in ARITHMETIC_MODELS:
            errors.extend(
                _check_arithmetic(
                    name, entry, inputs, values[name], ref_values[name], functions
                )
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
