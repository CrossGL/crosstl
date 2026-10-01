"""Deterministic unary host workloads and independent numerical references."""

from __future__ import annotations

import json
import math
import struct

from demos.integrations.mlx.portable_host.packages import UNARY_OPERATIONS
from demos.integrations.mlx.portable_host.runtime import wire_value

NAMES = {
    "ArcCos": "arccos",
    "ArcCosh": "arccosh",
    "ArcSin": "arcsin",
    "ArcSinh": "arcsinh",
    "ArcTan": "arctan",
    "ArcTanh": "arctanh",
    "ErfInv": "erfinv",
}


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def cases():
    records = []
    for operation in UNARY_OPERATIONS:
        values = [0.125, 0.25, 0.5, 0.75]
        if operation == "ArcCosh":
            values = [1.0, 1.5, 2.0, 3.0]
        elif operation in {
            "Abs",
            "Negative",
            "Sign",
            "Floor",
            "Ceil",
            "Round",
            "Erf",
            "ErfInv",
        }:
            values = [-0.75, -0.5, -0.0, 0.0, 0.5, 0.75]
        for count in (0, 1, 7, 257):
            records.append(
                {
                    "operation": operation,
                    "count": count,
                    "inputs": [values[index % len(values)] for index in range(count)],
                }
            )
    for operation in ("Abs", "Negative", "Floor", "Ceil", "Round", "ErfInv"):
        values = [-math.inf, -1.0, -0.0, 0.0, 1.0, math.inf, math.nan]
        records.append(
            {
                "operation": operation,
                "case": "nonfinite",
                "count": len(values),
                "inputs": [wire_value(value) for value in values],
            }
        )
    for operation in ("Sin", "Cos"):
        records.append(
            {
                "operation": operation,
                "case": "large-angles",
                "count": 5,
                "inputs": [_f32(value) for value in (1e8, 1e9, 1e10, 1e20, 1e30)],
            }
        )
    values = [
        struct.unpack("<f", struct.pack("<I", word))[0]
        for word in range(0x3F800000, 0x3F802001)
    ]
    records.append(
        {
            "operation": "ArcCosh",
            "case": "near-one",
            "count": len(values),
            "inputs": values,
        }
    )
    return records


def _inverse_erf(value):
    if math.isnan(value) or abs(value) > 1:
        return math.nan
    if abs(value) == 1:
        return math.copysign(math.inf, value)
    if value == 0:
        return value
    lower, upper = -8.0, 8.0
    for _ in range(80):
        middle = (lower + upper) / 2
        if math.erf(middle) < value:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def _reference(operation, value):
    if operation == "ErfInv":
        return _inverse_erf(value)
    if operation in {"Floor", "Ceil", "Round"}:
        if not math.isfinite(value):
            return value
        function = {"Floor": math.floor, "Ceil": math.ceil, "Round": round}[operation]
        result = float(function(value))
        return math.copysign(0.0, value) if result == 0 else result
    functions = {
        "Abs": abs,
        "Negative": lambda x: -x,
        "Sign": lambda x: float((x > 0) - (x < 0)),
        "Square": lambda x: x * x,
        "Sqrt": math.sqrt,
        "Rsqrt": lambda x: 1 / math.sqrt(x),
        "Sigmoid": lambda x: 1 / (1 + math.exp(-x)),
        "ArcCos": math.acos,
        "ArcCosh": math.acosh,
        "ArcSin": math.asin,
        "ArcSinh": math.asinh,
        "ArcTan": math.atan,
        "ArcTanh": math.atanh,
        "Cos": math.cos,
        "Cosh": math.cosh,
        "Exp": math.exp,
        "Expm1": math.expm1,
        "Log": math.log,
        "Log2": math.log2,
        "Log10": math.log10,
        "Log1p": math.log1p,
        "Sin": math.sin,
        "Sinh": math.sinh,
        "Tan": math.tan,
        "Tanh": math.tanh,
        "Erf": math.erf,
    }
    return functions[operation](value)


def expected_records(*, cpu=False):
    records = []
    for case in cases():
        values = [
            _reference(case["operation"], float(value)) for value in case["inputs"]
        ]
        if cpu and case["operation"] == "Erf":
            # The pinned CPU approximation returns negative zero for both
            # input zero signs; the source Metal operation preserves the sign.
            values = [-0.0 if value == 0 else value for value in values]
        records.append(
            {**case, "values": [wire_value(_f32(value)) for value in values]}
        )
    records.append(
        {
            "operation": "chain",
            "count": 7,
            "values": [9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0],
        }
    )
    return records


def run(mx, np):
    records = []
    for case in cases():
        operation = case["operation"]
        data = np.array([float(value) for value in case["inputs"]], dtype=np.float32)
        result = np.array(
            getattr(mx, NAMES.get(operation, operation.lower()))(mx.array(data))
        )
        if result.shape != data.shape or result.dtype != np.float32:
            raise RuntimeError(f"{operation} changed the array layout")
        records.append(
            {**case, "values": [wire_value(float(value)) for value in result]}
        )
    result = np.array(
        mx.square(mx.negative(mx.abs(mx.arange(-3, 4, dtype=mx.float32))))
    )
    np.testing.assert_array_equal(result, np.arange(-3, 4, dtype=np.float32) ** 2)
    records.append({"operation": "chain", "count": 7, "values": result.tolist()})
    return records


def validate(records, *, cpu=False):
    import numpy as np

    expected = expected_records(cpu=cpu)
    if not isinstance(records, list) or len(records) != len(expected):
        raise RuntimeError("Incomplete unary workload coverage")
    for reference, actual in zip(expected, records):
        if not isinstance(actual, dict) or type(actual.get("count")) is not int:
            raise RuntimeError("Invalid unary workload identity")
        identity = {key: value for key, value in actual.items() if key != "values"}
        required = {key: value for key, value in reference.items() if key != "values"}
        if json.dumps(identity, sort_keys=True, allow_nan=False) != json.dumps(
            required, sort_keys=True, allow_nan=False
        ):
            raise RuntimeError("Unary workload identity changed")
        values = actual.get("values")
        if (
            not isinstance(values, list)
            or len(values) != reference["count"]
            or any(
                not (
                    type(value) in {int, float}
                    and math.isfinite(value)
                    or isinstance(value, str)
                    and value in {"nan", "+infinity", "-infinity"}
                )
                for value in values
            )
        ):
            raise RuntimeError("Incomplete or invalid unary readbacks")
        got = np.array([float(value) for value in values])
        want = np.array([float(value) for value in reference["values"]])
        np.testing.assert_allclose(
            got,
            want,
            rtol=1e-5 if reference.get("case") == "near-one" else 2e-5,
            atol=1e-6,
            equal_nan=True,
            err_msg=str(required),
        )
        zeros = want == 0
        np.testing.assert_array_equal(
            got[zeros], want[zeros], err_msg="Unary zero changed"
        )
        np.testing.assert_array_equal(
            np.signbit(got[zeros]),
            np.signbit(want[zeros]),
            err_msg="Unary zero sign changed",
        )


def compare(original, translated):
    validate(original, cpu=True)
    validate(translated)


def dispatches():
    result = [
        (f"v_{case['operation']}float32float32", case["count"])
        for case in cases()
        if case["count"]
    ]
    return result + [
        ("arangefloat32", 7),
        ("v_Absfloat32float32", 7),
        ("v_Negativefloat32float32", 7),
        ("v_Squarefloat32float32", 7),
    ]
