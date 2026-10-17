"""Compare two discovery exports of the upstream test assembly.

The original Windows build and the patched non-Windows build must expose the same
expanded test cases and byte-identical generated HLSL with the same descriptor
metadata. Assembly hashes are expected to differ and are reported separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from baseline import load_discovery  # noqa: E402

CASE_FIELDS = (
    "id",
    "method",
    "signature",
    "displayName",
    "occurrence",
    "arguments",
    "ignored",
)
SHADER_FIELDS = (
    "type",
    "sha256",
    "threads",
    "constantBufferSize",
    "staticSamplerRequired",
    "resourceDescriptorRanges",
)


def _index(items: list, key: str, fields: tuple) -> dict:
    return {item[key]: {field: item.get(field) for field in fields} for item in items}


def compare(reference_dir: Path, candidate_dir: Path) -> dict:
    reference = load_discovery(reference_dir)
    candidate = load_discovery(candidate_dir)
    problems = []
    for name in ("testMethodCount", "caseCount", "shaderCount"):
        if reference[name] != candidate[name]:
            problems.append(f"{name}: {reference[name]} != {candidate[name]}")
    for label, key, fields in (
        ("tests", "id", CASE_FIELDS),
        ("shaders", "type", SHADER_FIELDS),
    ):
        left = _index(reference[label], key, fields)
        right = _index(candidate[label], key, fields)
        for missing in sorted(set(left) - set(right)):
            problems.append(f"{label}: missing {missing}")
        for extra in sorted(set(right) - set(left)):
            problems.append(f"{label}: unexpected {extra}")
        for shared in sorted(set(left) & set(right)):
            if left[shared] != right[shared]:
                changed = [f for f in fields if left[shared][f] != right[shared][f]]
                problems.append(f"{label}: {shared} differs in {', '.join(changed)}")
    if problems:
        raise ValueError("Discovery exports differ:\n" + "\n".join(problems[:50]))
    return {
        "schemaVersion": 1,
        "kind": "computesharp.discovery-parity",
        "caseCount": reference["caseCount"],
        "shaderCount": reference["shaderCount"],
        "referenceAssembly": reference["assembly"],
        "candidateAssembly": candidate["assembly"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.candidate)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + "\n")
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
