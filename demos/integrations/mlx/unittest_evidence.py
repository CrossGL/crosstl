"""Run unchanged upstream unittest suites with bounded failure-array evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
import unittest
from pathlib import Path

MAX_ARRAY_ELEMENTS = 16384
MAX_CAPTURE_ELEMENTS = 65536
MAX_FRAMES = 8
MAX_LOCALS = 64
MAX_TEXT = 2048


def scalar(value):
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value if value.bit_length() <= 256 else {"omitted": "integer-size"}
    if type(value) is float:
        return value if math.isfinite(value) else {"float": str(value)}
    if type(value) is complex:
        return {"real": scalar(value.real), "imag": scalar(value.imag)}
    raise TypeError("Unsupported scalar evidence type")


def array_values(value):
    if isinstance(value, list):
        return [array_values(item) for item in value]
    return scalar(value)


def capture_value(value, budget):
    kind = type(value)
    if kind in (type(None), bool, int, float, complex):
        return scalar(value)
    if kind is str:
        if len(value) <= MAX_TEXT:
            return value
        return {"text": value[:MAX_TEXT], "truncated": True}
    type_name = f"{kind.__module__}.{kind.__name__}"
    if type_name not in {"numpy.ndarray", "mlx.core.array"}:
        return {"type": type_name, "omitted": "unsupported-type"}
    shape = list(value.shape)
    record = {"type": type_name, "dtype": str(value.dtype), "shape": shape}
    count = math.prod(shape)
    if len(shape) > 8 or count > MAX_ARRAY_ELEMENTS or count > budget[0]:
        return dict(record, omitted="array-size", elements=count)
    if count == 0:
        return dict(record, values=[])
    try:
        record["values"] = array_values(value.tolist())
        budget[0] -= count
    except Exception as error:
        record["captureError"] = f"{type(error).__name__}: {error}"[:MAX_TEXT]
    return record


def capture_failure(test, error, source_root):
    frames = []
    for frame, line in traceback.walk_tb(error[2]):
        path = Path(frame.f_code.co_filename).resolve()
        if source_root in path.parents and path.is_file():
            frames.append((frame, line, path))
    budget = [MAX_CAPTURE_ELEMENTS]
    records = []
    for frame, line, path in frames[-MAX_FRAMES:]:
        names = sorted(frame.f_locals)
        records.append(
            {
                "source": path.relative_to(source_root).as_posix(),
                "sourceSHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "line": line,
                "function": frame.f_code.co_name,
                "omittedLocals": max(0, len(names) - MAX_LOCALS),
                "locals": {
                    name: capture_value(frame.f_locals[name], budget)
                    for name in names[:MAX_LOCALS]
                },
            }
        )
    return {
        "schemaVersion": 1,
        "test": test.id()[:MAX_TEXT],
        "exception": error[0].__name__,
        "message": str(error[1])[:MAX_TEXT],
        "omittedFrames": max(0, len(frames) - MAX_FRAMES),
        "capturedElements": MAX_CAPTURE_ELEMENTS - budget[0],
        "frames": records,
    }


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, output, source_root, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output
        self.source_root = source_root
        self.receipt_count = 0

    def record_failure(self, test, error):
        self.receipt_count += 1
        try:
            payload = capture_failure(test, error, self.source_root)
            key = hashlib.sha256(test.id().encode()).hexdigest()[:16]
            path = self.output / f"{self.receipt_count:04d}-{key}.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except Exception as failure:
            self.stream.writeln(f"Failure evidence unavailable: {failure}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record_failure(test, err)

    def addError(self, test, err):
        super().addError(test, err)
        self.record_failure(test, err)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.record_failure(subtest, err)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args, tests = parser.parse_known_args(argv)
    if not tests:
        parser.error("an upstream unittest suite is required")
    output = args.evidence_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path.cwd().resolve()
    sys.path.insert(0, str(source_root))

    class Runner(unittest.TextTestRunner):
        def _makeResult(self):
            return EvidenceResult(
                self.stream,
                self.descriptions,
                self.verbosity,
                output=output,
                source_root=source_root,
            )

    program = unittest.main(
        module=None, argv=[sys.argv[0], *tests], testRunner=Runner, exit=False
    )
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
