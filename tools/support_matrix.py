#!/usr/bin/env python3
"""Maintain CrossGL backend support metadata and generated reports.

The source of truth lives in ``support/backends.json`` and
``support/features.json``. This script validates that data and renders the
checked-in machine-readable and Sphinx documentation artifacts.
"""

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIR = ROOT / "support"
BACKENDS_PATH = SUPPORT_DIR / "backends.json"
FEATURES_PATH = SUPPORT_DIR / "features.json"
GENERATED_DIR = SUPPORT_DIR / "generated"
MATRIX_JSON_PATH = GENERATED_DIR / "support-matrix.json"
GRAPHICS_ROADMAP_JSON_PATH = GENERATED_DIR / "graphics-backend-roadmap.json"
PROJECT_PORTING_ROADMAP_JSON_PATH = GENERATED_DIR / "project-porting-roadmap.json"
DOCS_RST_PATH = ROOT / "docs" / "source" / "support-matrix.rst"
DEFAULT_DOC_REPORT_PATH = GENERATED_DIR / "backend-docs-report.json"


def load_support_signals_module():
    spec = importlib.util.spec_from_file_location(
        "support_signals",
        ROOT / "tools" / "support_signals.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STATUS_CODES = {
    "supported": "Y",
    "partial": "P",
    "diagnostic": "D",
    "validated_rejection": "R",
    "unsupported": "U",
    "unknown": "?",
}

STATUS_ORDER = [
    "supported",
    "partial",
    "diagnostic",
    "validated_rejection",
    "unsupported",
    "unknown",
]

BACKLOG_STATUSES = {
    "partial",
    "unsupported",
    "unknown",
}

GRAPHICS_BACKEND_IDS = ("directx", "opengl", "metal")
PROJECT_PORTING_CATEGORY = "project"
GENERATED_DIFF_PREVIEW_LIMIT = 120

TEST_PATTERN = re.compile(r"^\s*def\s+test_", re.MULTILINE)
UNSUPPORTED_PATTERN = re.compile(
    r"unsupported|not support|does not support|notimplemented",
    re.IGNORECASE,
)
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
SUPPORT_PLAN_KEY = "support_plan"
SUPPORT_PLAN_KEYS = ("current_gap", "next_scope", "completion_criteria")
SUPPORT_ENTRY_KEYS = {"status", "notes", "evidence", *SUPPORT_PLAN_KEYS}


class SupportMatrixError(Exception):
    """Raised when the support metadata is invalid."""


def require_mapping(value, description):
    if not isinstance(value, dict):
        raise SupportMatrixError(f"{description} must be an object")


def require_list(value, description):
    if not isinstance(value, list):
        raise SupportMatrixError(f"{description} must be a list")


def validate_id(value, description):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise SupportMatrixError(f"{description} must match {ID_PATTERN.pattern}")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def relpath(path):
    return os.path.relpath(str(path), str(ROOT)).replace(os.sep, "/")


def display_path(path):
    try:
        return str(path.relative_to(ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def iter_python_files(path):
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return
    if not path.is_dir():
        return
    for child in sorted(path.rglob("*.py")):
        if "__pycache__" in child.parts:
            continue
        yield child


def count_tests(paths):
    count = 0
    for entry in paths:
        path = ROOT / entry
        for file_path in iter_python_files(path):
            count += len(TEST_PATTERN.findall(read_text(file_path)))
    return count


def count_unsupported_markers(paths):
    count = 0
    for entry in paths:
        path = ROOT / entry
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(
                child
                for child in path.rglob("*.py")
                if "__pycache__" not in child.parts
            )
        else:
            files = []
        for file_path in files:
            count += len(UNSUPPORTED_PATTERN.findall(read_text(file_path)))
    return count


def unsupported_marker_samples(paths, limit=20):
    """Return stable sample markers without line numbers.

    These samples are intended to identify unsupported-code hotspots in the
    generated matrix. Omitting line numbers keeps generated artifacts stable
    when backend agents add unrelated lines above an existing marker.
    """
    samples = []
    for entry in paths:
        path = ROOT / entry
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(
                child
                for child in path.rglob("*.py")
                if "__pycache__" not in child.parts
            )
        else:
            files = []
        for file_path in files:
            for line in read_text(file_path).splitlines():
                if not UNSUPPORTED_PATTERN.search(line):
                    continue
                samples.append(
                    {
                        "path": relpath(file_path),
                        "text": line.strip()[:180],
                    }
                )
                if len(samples) >= limit:
                    return samples
    return samples


def path_exists(entry):
    return (ROOT / entry).exists()


def backend_signal_paths(backend):
    paths = []
    for key in ("translator_codegen", "native_backend"):
        value = backend.get(key)
        if value:
            paths.append(value)
    return paths


def validate_backend_catalog(backends_data):
    backends = backends_data.get("backends")
    if not isinstance(backends, list) or not backends:
        raise SupportMatrixError(
            "support/backends.json must contain a non-empty 'backends' list"
        )

    ids = set()
    aliases = set()
    extensions = set()
    for backend in backends:
        backend_id = backend.get("id")
        if not backend_id:
            raise SupportMatrixError("Every backend requires an 'id'")
        validate_id(backend_id, f"Backend id '{backend_id}'")
        if backend_id in ids:
            raise SupportMatrixError(f"Duplicate backend id: {backend_id}")
        ids.add(backend_id)

        for key in ("name", "translator_codegen", "native_backend", "tests", "docs"):
            if key not in backend:
                raise SupportMatrixError(f"Backend '{backend_id}' is missing '{key}'")

        if not isinstance(backend["name"], str) or not backend["name"].strip():
            raise SupportMatrixError(
                f"Backend '{backend_id}' name must be a non-empty string"
            )
        target_extension = backend.get("target_extension")
        if not isinstance(target_extension, str) or not target_extension.startswith(
            "."
        ):
            raise SupportMatrixError(
                f"Backend '{backend_id}' target_extension must start with '.'"
            )
        if target_extension in extensions:
            raise SupportMatrixError(
                f"Duplicate backend target extension: {target_extension}"
            )
        extensions.add(target_extension)

        require_list(backend.get("aliases", []), f"Backend '{backend_id}' aliases")
        for alias in backend.get("aliases", []):
            validate_id(alias, f"Backend '{backend_id}' alias '{alias}'")
            if alias in aliases:
                raise SupportMatrixError(f"Duplicate backend alias: {alias}")
            aliases.add(alias)

        if not path_exists(backend["translator_codegen"]):
            raise SupportMatrixError(
                "Backend '{}' codegen path does not exist: {}".format(
                    backend_id, backend["translator_codegen"]
                )
            )
        if not path_exists(backend["native_backend"]):
            raise SupportMatrixError(
                "Backend '{}' native backend path does not exist: {}".format(
                    backend_id, backend["native_backend"]
                )
            )
        require_list(backend.get("tests", []), f"Backend '{backend_id}' tests")
        for test_path in backend.get("tests", []):
            if not isinstance(test_path, str):
                raise SupportMatrixError(
                    f"Backend '{backend_id}' test paths must be strings"
                )
            if not path_exists(test_path):
                raise SupportMatrixError(
                    "Backend '{}' test path does not exist: {}".format(
                        backend_id, test_path
                    )
                )
        require_list(backend.get("docs", []), f"Backend '{backend_id}' docs")
        for doc in backend.get("docs", []):
            require_mapping(doc, f"Backend '{backend_id}' docs entry")
            if not doc.get("name") or not doc.get("url"):
                raise SupportMatrixError(
                    f"Backend '{backend_id}' has a docs entry missing name/url"
                )
            if not isinstance(doc["name"], str) or not isinstance(doc["url"], str):
                raise SupportMatrixError(
                    f"Backend '{backend_id}' docs name/url must be strings"
                )
            if urllib.parse.urlparse(doc["url"]).scheme not in {"http", "https"}:
                raise SupportMatrixError(
                    "Backend '{}' docs URL must be http(s): {}".format(
                        backend_id, doc["url"]
                    )
                )
    return ids


def validate_evidence(feature_id, backend_id, evidence):
    if not isinstance(evidence, list):
        raise SupportMatrixError(
            "Feature '{}' backend '{}' evidence must be a list".format(
                feature_id, backend_id
            )
        )

    for entry in evidence:
        if not isinstance(entry, str):
            raise SupportMatrixError(
                "Feature '{}' backend '{}' evidence entries must be strings".format(
                    feature_id, backend_id
                )
            )
        path_text, separator, pattern = entry.partition("::")
        path = ROOT / path_text
        if not path.exists():
            raise SupportMatrixError(
                "Feature '{}' backend '{}' evidence path does not exist: {}".format(
                    feature_id, backend_id, path_text
                )
            )
        if separator:
            if path.is_dir():
                haystack = "\n".join(
                    read_text(child) for child in iter_python_files(path)
                )
            else:
                haystack = read_text(path)
            try:
                matches = re.search(pattern, haystack)
            except re.error as exc:
                raise SupportMatrixError(
                    "Feature '{}' backend '{}' evidence regex is invalid '{}': {}".format(
                        feature_id, backend_id, pattern, exc
                    )
                )
            if not matches:
                raise SupportMatrixError(
                    "Feature '{}' backend '{}' evidence pattern not found: {}".format(
                        feature_id, backend_id, entry
                    )
                )


def validate_support_plan(plan, description):
    if not isinstance(plan, dict):
        raise SupportMatrixError(f"{description} must be an object")
    unknown_keys = set(plan) - set(SUPPORT_PLAN_KEYS)
    if unknown_keys:
        raise SupportMatrixError(
            "{} has unsupported support plan key(s): {}".format(
                description, ", ".join(sorted(unknown_keys))
            )
        )
    for key in SUPPORT_PLAN_KEYS:
        if key in plan and not isinstance(plan[key], str):
            raise SupportMatrixError(f"{description} '{key}' must be a string")


def validate_feature_catalog(features_data, backend_ids):
    statuses = features_data.get("statuses")
    if not isinstance(statuses, dict) or not statuses:
        raise SupportMatrixError("support/features.json must define 'statuses'")

    missing_statuses = set(STATUS_CODES) - set(statuses)
    if missing_statuses:
        raise SupportMatrixError(
            "features.json is missing status definition(s): {}".format(
                ", ".join(sorted(missing_statuses))
            )
        )
    missing_status_codes = set(statuses) - set(STATUS_CODES)
    if missing_status_codes:
        raise SupportMatrixError(
            "Unsupported status value(s) in features.json: {}".format(
                ", ".join(sorted(missing_status_codes))
            )
        )

    features = features_data.get("features")
    if not isinstance(features, list) or not features:
        raise SupportMatrixError(
            "support/features.json must contain a non-empty 'features' list"
        )

    feature_ids = set()
    for feature in features:
        feature_id = feature.get("id")
        if not feature_id:
            raise SupportMatrixError("Every feature requires an 'id'")
        validate_id(feature_id, f"Feature id '{feature_id}'")
        if feature_id in feature_ids:
            raise SupportMatrixError(f"Duplicate feature id: {feature_id}")
        feature_ids.add(feature_id)

        for key in ("category", "name", "description"):
            if key not in feature:
                raise SupportMatrixError(f"Feature '{feature_id}' is missing '{key}'")
            if not isinstance(feature[key], str) or not feature[key].strip():
                raise SupportMatrixError(
                    "Feature '{}' '{}' must be a non-empty string".format(
                        feature_id, key
                    )
                )

        if SUPPORT_PLAN_KEY in feature:
            validate_support_plan(
                feature[SUPPORT_PLAN_KEY], f"Feature '{feature_id}' support_plan"
            )

        support = feature.get("support", {})
        if not isinstance(support, dict):
            raise SupportMatrixError(
                f"Feature '{feature_id}' support must be an object"
            )
        unknown_backend_ids = set(support) - backend_ids
        if unknown_backend_ids:
            raise SupportMatrixError(
                "Feature '{}' references unknown backend(s): {}".format(
                    feature_id, ", ".join(sorted(unknown_backend_ids))
                )
            )

        for backend_id, entry in support.items():
            if not isinstance(entry, dict):
                raise SupportMatrixError(
                    "Feature '{}' backend '{}' support must be an object".format(
                        feature_id, backend_id
                    )
                )
            unknown_keys = set(entry) - SUPPORT_ENTRY_KEYS
            if unknown_keys:
                raise SupportMatrixError(
                    "Feature '{}' backend '{}' has unsupported support key(s): {}".format(
                        feature_id, backend_id, ", ".join(sorted(unknown_keys))
                    )
                )
            status = entry.get("status")
            if status not in statuses:
                raise SupportMatrixError(
                    "Feature '{}' backend '{}' has invalid status '{}'".format(
                        feature_id, backend_id, status
                    )
                )
            for key in ("notes", *SUPPORT_PLAN_KEYS):
                if key in entry and not isinstance(entry[key], str):
                    raise SupportMatrixError(
                        "Feature '{}' backend '{}' {} must be a string".format(
                            feature_id, backend_id, key
                        )
                    )
            if "evidence" in entry:
                validate_evidence(feature_id, backend_id, entry["evidence"])


def validate_catalogs(backends_data, features_data):
    backend_ids = validate_backend_catalog(backends_data)
    validate_feature_catalog(features_data, backend_ids)


def backend_inventory(backend):
    tests = backend.get("tests", [])
    scanned_paths = backend_signal_paths(backend)
    return {
        "id": backend["id"],
        "name": backend["name"],
        "aliases": backend.get("aliases", []),
        "target_extension": backend.get("target_extension"),
        "translator_codegen": {
            "path": backend["translator_codegen"],
            "exists": path_exists(backend["translator_codegen"]),
        },
        "native_backend": {
            "path": backend["native_backend"],
            "exists": path_exists(backend["native_backend"]),
        },
        "tests": {
            "paths": tests,
            "test_count": count_tests(tests),
        },
        "signals": {
            "unsupported_marker_count": count_unsupported_markers(scanned_paths),
            "unsupported_marker_samples": unsupported_marker_samples(scanned_paths),
        },
        "docs": backend.get("docs", []),
    }


def normalized_support_entry(feature, backend_id):
    support = feature.get("support", {}).get(backend_id)
    if support is None:
        return {
            "status": "unknown",
            "notes": "No support status has been audited for this backend.",
            "evidence": [],
        }
    entry = {
        "status": support["status"],
        "notes": support.get("notes", ""),
        "evidence": support.get("evidence", []),
    }
    if entry["status"] in BACKLOG_STATUSES:
        support_plan = feature.get(SUPPORT_PLAN_KEY, {})
        for key in SUPPORT_PLAN_KEYS:
            value = support.get(key) or support_plan.get(key)
            if value:
                entry[key] = value
    return entry


def backlog_row_for_feature(feature, backend_id, backend_name, entry):
    row = {
        "feature_id": feature["id"],
        "feature": feature["name"],
        "category": feature["category"],
        "backend_id": backend_id,
        "backend": backend_name,
        "status": entry["status"],
        "notes": entry.get("notes", ""),
    }
    for key in SUPPORT_PLAN_KEYS:
        if entry.get(key):
            row[key] = entry[key]
    return row


def build_matrix(backends_data, features_data):
    backends = backends_data["backends"]
    backend_ids = [backend["id"] for backend in backends]
    backend_names = {backend["id"]: backend["name"] for backend in backends}

    features = []
    counts = {
        backend_id: {status: 0 for status in STATUS_ORDER} for backend_id in backend_ids
    }
    backlog = []

    for source_feature in features_data["features"]:
        support = {}
        for backend_id in backend_ids:
            entry = normalized_support_entry(source_feature, backend_id)
            support[backend_id] = entry
            counts[backend_id][entry["status"]] += 1
            if entry["status"] in BACKLOG_STATUSES:
                backlog.append(
                    backlog_row_for_feature(
                        source_feature, backend_id, backend_names[backend_id], entry
                    )
                )

        features.append(
            {
                "id": source_feature["id"],
                "category": source_feature["category"],
                "name": source_feature["name"],
                "description": source_feature["description"],
                "support": support,
            }
        )

    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py",
        "status_codes": STATUS_CODES,
        "status_descriptions": features_data["statuses"],
        "backends": [backend_inventory(backend) for backend in backends],
        "features": features,
        "summary": {
            "feature_count": len(features),
            "backend_count": len(backends),
            "status_counts": counts,
            "backlog_count": len(backlog),
        },
        "backlog": backlog,
    }


def validate_matrix(matrix):
    backends = matrix.get("backends", [])
    features = matrix.get("features", [])
    backend_ids = [backend.get("id") for backend in backends]
    if len(backend_ids) != len(set(backend_ids)):
        raise SupportMatrixError("Generated matrix contains duplicate backend ids")
    missing_graphics = set(GRAPHICS_BACKEND_IDS) - set(backend_ids)
    if missing_graphics:
        raise SupportMatrixError(
            "Generated matrix is missing graphics backend(s): {}".format(
                ", ".join(sorted(missing_graphics))
            )
        )

    if matrix.get("summary", {}).get("feature_count") != len(features):
        raise SupportMatrixError("Generated matrix feature_count is inconsistent")
    if matrix.get("summary", {}).get("backend_count") != len(backends):
        raise SupportMatrixError("Generated matrix backend_count is inconsistent")

    counts = {
        backend_id: {status: 0 for status in STATUS_ORDER} for backend_id in backend_ids
    }
    expected_backlog = []
    seen_features = set()
    backend_id_set = set(backend_ids)
    for feature in features:
        feature_id = feature.get("id")
        if feature_id in seen_features:
            raise SupportMatrixError(
                f"Generated matrix contains duplicate feature id: {feature_id}"
            )
        seen_features.add(feature_id)
        support = feature.get("support", {})
        support_backend_ids = set(support)
        if support_backend_ids != backend_id_set:
            missing = backend_id_set - support_backend_ids
            extra = support_backend_ids - backend_id_set
            details = []
            if missing:
                details.append("missing {}".format(", ".join(sorted(missing))))
            if extra:
                details.append("extra {}".format(", ".join(sorted(extra))))
            raise SupportMatrixError(
                "Generated matrix feature '{}' has inconsistent backend support: {}".format(
                    feature_id, "; ".join(details)
                )
            )
        for backend_id, entry in support.items():
            status = entry.get("status")
            if status not in STATUS_CODES:
                raise SupportMatrixError(
                    "Generated matrix feature '{}' backend '{}' has invalid status '{}'".format(
                        feature_id, backend_id, status
                    )
                )
            counts[backend_id][status] += 1
            if status in BACKLOG_STATUSES:
                expected_backlog.append(
                    backlog_row_for_feature(
                        feature,
                        backend_id,
                        next(
                            backend["name"]
                            for backend in backends
                            if backend["id"] == backend_id
                        ),
                        entry,
                    )
                )

    summary_counts = matrix.get("summary", {}).get("status_counts")
    if summary_counts != counts:
        raise SupportMatrixError("Generated matrix status_counts are inconsistent")
    backlog = matrix.get("backlog", [])
    if matrix.get("summary", {}).get("backlog_count") != len(backlog):
        raise SupportMatrixError("Generated matrix backlog_count is inconsistent")
    if sorted(backlog, key=stable_json) != sorted(expected_backlog, key=stable_json):
        raise SupportMatrixError("Generated matrix backlog rows are inconsistent")


def filtered_backlog(matrix, backend_ids=None, categories=None, statuses=None):
    backend_filter = set(backend_ids or [])
    category_filter = set(categories or [])
    status_filter = set(statuses or [])

    rows = []
    for item in matrix["backlog"]:
        if backend_filter and item["backend_id"] not in backend_filter:
            continue
        if category_filter and item["category"] not in category_filter:
            continue
        if status_filter and item["status"] not in status_filter:
            continue
        rows.append(item)
    return rows


def backlog_summary(rows):
    summary = {
        "backlog_count": len(rows),
        "by_backend": {},
        "by_category": {},
        "by_status": {status: 0 for status in STATUS_ORDER},
    }
    for item in rows:
        backend_id = item["backend_id"]
        category = item["category"]
        status = item["status"]
        summary["by_backend"][backend_id] = summary["by_backend"].get(backend_id, 0) + 1
        summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
        summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
    return summary


def build_audit_report(matrix, rows, filters):
    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py",
        "filters": filters,
        "summary": backlog_summary(rows),
        "backlog": rows,
    }


def filtered_support_rows(
    matrix, backend_ids=None, categories=None, statuses=None, evidence="any"
):
    backend_filter = set(backend_ids or [])
    category_filter = set(categories or [])
    status_filter = set(statuses or [])
    backend_names = {backend["id"]: backend["name"] for backend in matrix["backends"]}

    rows = []
    for feature in matrix["features"]:
        if category_filter and feature["category"] not in category_filter:
            continue
        for backend_id, support in feature["support"].items():
            if backend_filter and backend_id not in backend_filter:
                continue
            status = support["status"]
            if status_filter and status not in status_filter:
                continue

            evidence_entries = support.get("evidence", [])
            has_evidence = bool(evidence_entries)
            if evidence == "missing" and has_evidence:
                continue
            if evidence == "present" and not has_evidence:
                continue

            rows.append(
                {
                    "backend": backend_names[backend_id],
                    "backend_id": backend_id,
                    "category": feature["category"],
                    "feature": feature["name"],
                    "feature_id": feature["id"],
                    "status": status,
                    "notes": support.get("notes", ""),
                    "evidence_count": len(evidence_entries),
                    "evidence": evidence_entries,
                }
            )
    return rows


def evidence_summary(rows):
    summary = {
        "row_count": len(rows),
        "missing_evidence_count": 0,
        "present_evidence_count": 0,
        "by_backend": {},
        "by_status": {status: 0 for status in STATUS_ORDER},
    }
    for item in rows:
        backend_id = item["backend_id"]
        status = item["status"]
        backend_summary = summary["by_backend"].setdefault(
            backend_id, {"rows": 0, "present": 0, "missing": 0}
        )
        backend_summary["rows"] += 1
        summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
        if item["evidence_count"]:
            summary["present_evidence_count"] += 1
            backend_summary["present"] += 1
        else:
            summary["missing_evidence_count"] += 1
            backend_summary["missing"] += 1
    return summary


def build_evidence_report(matrix, rows, filters):
    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py evidence",
        "filters": filters,
        "summary": evidence_summary(rows),
        "rows": rows,
    }


def build_backend_view(matrix, view_id, title, backend_ids):
    backend_id_set = set(backend_ids)
    backend_names = {
        backend["id"]: backend["name"]
        for backend in matrix["backends"]
        if backend["id"] in backend_id_set
    }
    features = []
    for feature in matrix["features"]:
        support = {
            backend_id: feature["support"][backend_id] for backend_id in backend_ids
        }
        features.append(
            {
                "id": feature["id"],
                "category": feature["category"],
                "name": feature["name"],
                "description": feature["description"],
                "support": support,
            }
        )

    rows = filtered_backlog(matrix, backend_ids=backend_ids)
    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py",
        "view": {
            "id": view_id,
            "title": title,
            "backend_ids": list(backend_ids),
            "backends": backend_names,
        },
        "summary": {
            "feature_count": len(features),
            "backend_count": len(backend_ids),
            "status_counts": {
                backend_id: matrix["summary"]["status_counts"][backend_id]
                for backend_id in backend_ids
            },
            "backlog": backlog_summary(rows),
        },
        "features": features,
        "backlog": rows,
    }


def build_graphics_backend_roadmap(matrix):
    return build_backend_view(
        matrix,
        "graphics_backends",
        "DirectX, OpenGL, and Metal support roadmap",
        GRAPHICS_BACKEND_IDS,
    )


def build_category_view(matrix, view_id, title, categories):
    category_set = set(categories)
    backend_ids = [backend["id"] for backend in matrix["backends"]]
    backend_names = {backend["id"]: backend["name"] for backend in matrix["backends"]}
    status_counts = {
        backend_id: {status: 0 for status in STATUS_ORDER} for backend_id in backend_ids
    }

    features = []
    for feature in matrix["features"]:
        if feature["category"] not in category_set:
            continue
        support = {}
        for backend_id in backend_ids:
            entry = feature["support"][backend_id]
            support[backend_id] = entry
            status_counts[backend_id][entry["status"]] += 1
        features.append(
            {
                "id": feature["id"],
                "category": feature["category"],
                "name": feature["name"],
                "description": feature["description"],
                "support": support,
            }
        )

    rows = filtered_backlog(matrix, categories=categories)
    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py",
        "view": {
            "id": view_id,
            "title": title,
            "categories": list(categories),
            "backend_ids": backend_ids,
            "backends": backend_names,
        },
        "summary": {
            "feature_count": len(features),
            "backend_count": len(backend_ids),
            "status_counts": status_counts,
            "backlog": backlog_summary(rows),
        },
        "features": features,
        "backlog": rows,
    }


def build_project_porting_roadmap(matrix):
    return build_category_view(
        matrix,
        "project_porting",
        "Project-porting support roadmap",
        (PROJECT_PORTING_CATEGORY,),
    )


def rst_escape(text):
    if text is None:
        return ""
    return str(text).replace("\n", " ").strip()


def csv_cell(text):
    escaped = rst_escape(text).replace('"', '""')
    return f'"{escaped}"'


def csv_row(cells):
    return ", ".join(csv_cell(cell) for cell in cells)


def render_csv_table(title, headers, rows, widths=None):
    lines = []
    lines.append(f".. csv-table:: {title}")
    lines.append(f"   :header: {csv_row(headers)}")
    if widths:
        lines.append("   :widths: {}".format(", ".join(str(width) for width in widths)))
    lines.append("")
    for row in rows:
        lines.append(f"   {csv_row(row)}")
    lines.append("")
    return lines


def grouped_features(features):
    groups = []
    by_category = {}
    for feature in features:
        category = feature["category"]
        if category not in by_category:
            by_category[category] = []
            groups.append(category)
        by_category[category].append(feature)
    return [(category, by_category[category]) for category in groups]


def render_docs(matrix):
    backend_ids = [backend["id"] for backend in matrix["backends"]]
    backend_name = {backend["id"]: backend["name"] for backend in matrix["backends"]}
    graphics_roadmap = build_graphics_backend_roadmap(matrix)
    graphics_backend_names = graphics_roadmap["view"]["backends"]
    project_roadmap = build_project_porting_roadmap(matrix)
    lines = [
        "Support Matrix",
        "==============",
        "",
        ".. This file is generated by tools/support_matrix.py. Do not edit by hand.",
        "",
        "This matrix tracks CrossGL backend coverage from the checked-in support",
        "catalog. A feature is marked supported only when the repo has implementation",
        "and test evidence for the CrossGL contract. Unknown means unaudited, not",
        "implicitly supported.",
        "",
    ]

    legend_rows = [
        [STATUS_CODES[status], status, matrix["status_descriptions"][status]]
        for status in STATUS_ORDER
        if status in matrix["status_descriptions"]
    ]
    lines.extend(
        render_csv_table(
            "Status legend",
            ["Code", "Status", "Meaning"],
            legend_rows,
            widths=[8, 20, 72],
        )
    )

    backend_rows = []
    for backend in matrix["backends"]:
        docs = "; ".join(doc["name"] for doc in backend.get("docs", []))
        backend_rows.append(
            [
                backend["name"],
                backend.get("target_extension") or "",
                backend["translator_codegen"]["path"],
                backend["native_backend"]["path"],
                ", ".join(backend["tests"]["paths"]),
                backend["tests"]["test_count"],
                backend["signals"]["unsupported_marker_count"],
                docs,
            ]
        )
    lines.extend(
        render_csv_table(
            "Backend inventory",
            [
                "Backend",
                "Ext",
                "Target generator",
                "Native frontend",
                "Tests",
                "Test count",
                "Unsupported markers",
                "Docs source",
            ],
            backend_rows,
        )
    )

    summary_rows = []
    for backend in matrix["backends"]:
        counts = matrix["summary"]["status_counts"][backend["id"]]
        summary_rows.append(
            [backend["name"]] + [counts.get(status, 0) for status in STATUS_ORDER]
        )
    lines.extend(
        render_csv_table(
            "Summary by backend",
            ["Backend"] + [status for status in STATUS_ORDER],
            summary_rows,
        )
    )

    lines.extend(
        [
            "Graphics Backend Focus",
            "----------------------",
            "",
            "This view isolates the DirectX, OpenGL, and Metal rows that are in",
            "scope for graphics backend completion work.",
            "",
        ]
    )
    graphics_summary_rows = []
    for backend_id in GRAPHICS_BACKEND_IDS:
        counts = graphics_roadmap["summary"]["status_counts"][backend_id]
        graphics_summary_rows.append(
            [graphics_backend_names[backend_id]]
            + [counts.get(status, 0) for status in STATUS_ORDER]
        )
    lines.extend(
        render_csv_table(
            "Graphics backend status summary",
            ["Backend"] + [status for status in STATUS_ORDER],
            graphics_summary_rows,
        )
    )
    graphics_backlog_rows = [
        [
            item["backend"],
            item["category"],
            item["feature"],
            item["status"],
            item.get("notes", ""),
        ]
        for item in graphics_roadmap["backlog"]
    ]
    lines.extend(
        render_csv_table(
            "DirectX/OpenGL/Metal actionable backlog",
            ["Backend", "Category", "Feature", "Status", "Notes"],
            graphics_backlog_rows,
        )
    )

    lines.extend(
        [
            "Project Porting Focus",
            "---------------------",
            "",
            "This view isolates repository-scale scanning, translation, report",
            "inspection, diagnostics, validation, and corpus-coverage rows.",
            "",
        ]
    )
    project_summary_rows = []
    for backend_id in backend_ids:
        counts = project_roadmap["summary"]["status_counts"][backend_id]
        project_summary_rows.append(
            [backend_name[backend_id]]
            + [counts.get(status, 0) for status in STATUS_ORDER]
        )
    lines.extend(
        render_csv_table(
            "Project-porting status summary",
            ["Backend"] + [status for status in STATUS_ORDER],
            project_summary_rows,
        )
    )
    project_backlog_rows = [
        [
            item["backend"],
            item["feature"],
            item["status"],
            item.get("current_gap") or item.get("notes", ""),
            item.get("next_scope", ""),
        ]
        for item in project_roadmap["backlog"]
    ]
    lines.extend(
        render_csv_table(
            "Project-porting actionable backlog",
            ["Backend", "Feature", "Status", "Current gap", "Next scope"],
            project_backlog_rows,
        )
    )

    lines.extend(
        [
            "Feature Matrix",
            "--------------",
            "",
            "Each category below uses the status codes from the legend.",
            "",
        ]
    )

    for category, features in grouped_features(matrix["features"]):
        headers = ["Feature"] + [backend_name[backend_id] for backend_id in backend_ids]
        rows = []
        for feature in features:
            rows.append(
                [feature["name"]]
                + [
                    STATUS_CODES[feature["support"][backend_id]["status"]]
                    for backend_id in backend_ids
                ]
            )
        lines.extend(render_csv_table(category, headers, rows))

    backlog_rows = []
    for item in matrix["backlog"]:
        backlog_rows.append(
            [
                item["backend"],
                item["category"],
                item["feature"],
                item["status"],
                item.get("current_gap") or item.get("notes", ""),
                item.get("next_scope", ""),
                item.get("notes", ""),
            ]
        )
    lines.extend(
        [
            "Actionable Backlog",
            "------------------",
            "",
            "These rows are actionable coverage gaps with ``partial``,",
            "``unsupported``, or ``unknown`` status. Evidence-backed",
            "``diagnostic`` and ``validated_rejection`` rows remain visible in",
            "the matrix counts, but are closed-loop behavior rather than synced",
            "implementation backlog work. Unknown rows need an audit before",
            "implementation work can be scoped accurately.",
            "",
        ]
    )
    lines.extend(
        render_csv_table(
            "Actionable backlog rows",
            [
                "Backend",
                "Category",
                "Feature",
                "Status",
                "Current gap",
                "Next scope",
                "Notes",
            ],
            backlog_rows,
        )
    )

    lines.extend(
        [
            "Documentation Sources",
            "---------------------",
            "",
            "The nightly documentation probe checks these URLs for reachability and",
            "content hash changes. It does not overwrite feature statuses; status",
            "changes must be reviewed and committed in the source catalog.",
            "",
        ]
    )
    doc_rows = []
    for backend in matrix["backends"]:
        for doc in backend.get("docs", []):
            doc_rows.append([backend["name"], doc["name"], doc["url"]])
    lines.extend(
        render_csv_table(
            "Official documentation URLs",
            ["Backend", "Source", "URL"],
            doc_rows,
        )
    )

    return "\n".join(lines).rstrip() + "\n"


def write_generated(matrix):
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_JSON_PATH.write_text(stable_json(matrix), encoding="utf-8")
    GRAPHICS_ROADMAP_JSON_PATH.write_text(
        stable_json(build_graphics_backend_roadmap(matrix)), encoding="utf-8"
    )
    PROJECT_PORTING_ROADMAP_JSON_PATH.write_text(
        stable_json(build_project_porting_roadmap(matrix)), encoding="utf-8"
    )
    DOCS_RST_PATH.write_text(render_docs(matrix), encoding="utf-8")


def generated_artifact_specs(matrix):
    return (
        (MATRIX_JSON_PATH, stable_json(matrix)),
        (
            GRAPHICS_ROADMAP_JSON_PATH,
            stable_json(build_graphics_backend_roadmap(matrix)),
        ),
        (
            PROJECT_PORTING_ROADMAP_JSON_PATH,
            stable_json(build_project_porting_roadmap(matrix)),
        ),
        (DOCS_RST_PATH, render_docs(matrix)),
    )


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare_file(path, expected):
    if path.exists():
        actual = path.read_text(encoding="utf-8")
    else:
        actual = ""
    if actual == expected:
        return []
    return list(
        difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=str(path),
            tofile=str(path) + " (expected)",
            lineterm="",
        )
    )


def check_generated(matrix):
    failures = []
    for path, expected in generated_artifact_specs(matrix):
        diff = compare_file(path, expected)
        if diff:
            failures.append((path, diff))
    return failures


def build_generated_check_report(matrix):
    artifacts = []
    for path, expected in generated_artifact_specs(matrix):
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        diff = compare_file(path, expected)
        artifacts.append(
            {
                "path": relpath(path),
                "exists": path.exists(),
                "stale": bool(diff),
                "actual_sha256": text_sha256(actual),
                "expected_sha256": text_sha256(expected),
                "diff_line_count": len(diff),
                "diff": diff,
            }
        )

    stale_count = sum(1 for artifact in artifacts if artifact["stale"])
    stale_artifacts = [artifact["path"] for artifact in artifacts if artifact["stale"]]
    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py check",
        "ok": stale_count == 0,
        "summary": {
            "artifact_count": len(artifacts),
            "stale_count": stale_count,
            "stale_artifacts": stale_artifacts,
            "total_diff_line_count": sum(
                artifact["diff_line_count"] for artifact in artifacts
            ),
        },
        "artifacts": artifacts,
    }


def write_json_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(report), encoding="utf-8")
    print(f"Wrote {display_path(path)}")


def print_generated_failure_summary(report):
    stale_artifacts = [
        artifact for artifact in report["artifacts"] if artifact["stale"]
    ]
    if not stale_artifacts:
        return
    print("", file=sys.stderr)
    print("Stale artifact summary:", file=sys.stderr)
    for artifact in stale_artifacts:
        existence = "exists" if artifact["exists"] else "missing"
        print(
            "- {path}: {diff_line_count} diff lines ({existence}); "
            "actual={actual_sha256}, expected={expected_sha256}".format(
                **artifact,
                existence=existence,
            ),
            file=sys.stderr,
        )


def print_generated_failures(report):
    print("Generated support matrix artifacts are stale.", file=sys.stderr)
    print("Run: python tools/support_matrix.py update", file=sys.stderr)
    print_generated_failure_summary(report)
    for artifact in report["artifacts"]:
        if not artifact["stale"]:
            continue
        diff = artifact["diff"]
        print("\nDiff for {}:".format(artifact["path"]), file=sys.stderr)
        preview = diff[:GENERATED_DIFF_PREVIEW_LIMIT]
        for line in preview:
            print(line, file=sys.stderr)
        if len(diff) > len(preview):
            print("... diff truncated ...", file=sys.stderr)


def split_filter_values(values):
    result = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result


def validate_audit_filters(matrix, backend_ids, categories, statuses):
    known_backend_ids = {backend["id"] for backend in matrix["backends"]}
    unknown_backend_ids = set(backend_ids) - known_backend_ids
    if unknown_backend_ids:
        raise SupportMatrixError(
            "Unknown backend filter(s): {}".format(
                ", ".join(sorted(unknown_backend_ids))
            )
        )

    known_categories = {feature["category"] for feature in matrix["features"]}
    unknown_categories = set(categories) - known_categories
    if unknown_categories:
        raise SupportMatrixError(
            "Unknown category filter(s): {}".format(
                ", ".join(sorted(unknown_categories))
            )
        )

    unknown_statuses = set(statuses) - set(STATUS_ORDER)
    if unknown_statuses:
        raise SupportMatrixError(
            "Unknown status filter(s): {}".format(", ".join(sorted(unknown_statuses)))
        )


def audit(
    matrix, fail_on, backend_ids=None, categories=None, statuses=None, output=None
):
    fail_on = set(fail_on or [])
    backend_ids = split_filter_values(backend_ids)
    categories = split_filter_values(categories)
    statuses = split_filter_values(statuses)
    validate_audit_filters(matrix, backend_ids, categories, statuses)

    rows = filtered_backlog(
        matrix, backend_ids=backend_ids, categories=categories, statuses=statuses
    )
    filters = {
        "backend_ids": backend_ids,
        "categories": categories,
        "statuses": statuses,
    }
    report = build_audit_report(matrix, rows, filters)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(stable_json(report), encoding="utf-8")
        print(f"Wrote {display_path(output)}")

    print(
        "Support matrix: {} features across {} backends".format(
            matrix["summary"]["feature_count"], matrix["summary"]["backend_count"]
        )
    )
    print("Backlog rows: {}".format(report["summary"]["backlog_count"]))
    if backend_ids or categories or statuses:
        print(
            "Filters: backend_ids={}, categories={}, statuses={}".format(
                ",".join(backend_ids) or "*",
                ",".join(categories) or "*",
                ",".join(statuses) or "*",
            )
        )
    print("")

    for backend in matrix["backends"]:
        if backend_ids and backend["id"] not in backend_ids:
            continue
        counts = matrix["summary"]["status_counts"][backend["id"]]
        count_text = ", ".join(
            f"{status}={counts.get(status, 0)}" for status in STATUS_ORDER
        )
        print("{}: {}".format(backend["name"], count_text))

    if not rows:
        return 0

    print("")
    print("Backlog:")
    for item in rows:
        print(
            "- {backend}: {feature} [{status}]".format(
                backend=item["backend"], feature=item["feature"], status=item["status"]
            )
        )

    if any(item["status"] in fail_on for item in rows):
        return 1
    return 0


def evidence_audit(
    matrix,
    backend_ids=None,
    categories=None,
    statuses=None,
    evidence="any",
    output=None,
    fail_on_missing=False,
):
    backend_ids = split_filter_values(backend_ids)
    categories = split_filter_values(categories)
    statuses = split_filter_values(statuses)
    if evidence not in {"any", "present", "missing"}:
        raise SupportMatrixError(f"Unknown evidence filter: {evidence}")
    validate_audit_filters(matrix, backend_ids, categories, statuses)

    rows = filtered_support_rows(
        matrix,
        backend_ids=backend_ids,
        categories=categories,
        statuses=statuses,
        evidence=evidence,
    )
    filters = {
        "backend_ids": backend_ids,
        "categories": categories,
        "statuses": statuses,
        "evidence": evidence,
    }
    report = build_evidence_report(matrix, rows, filters)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(stable_json(report), encoding="utf-8")
        print(f"Wrote {display_path(output)}")

    print(
        "Support matrix evidence: {} features across {} backends".format(
            matrix["summary"]["feature_count"], matrix["summary"]["backend_count"]
        )
    )
    print("Evidence rows: {}".format(report["summary"]["row_count"]))
    print(
        "Evidence coverage: present={}, missing={}".format(
            report["summary"]["present_evidence_count"],
            report["summary"]["missing_evidence_count"],
        )
    )
    if backend_ids or categories or statuses or evidence != "any":
        print(
            "Filters: backend_ids={}, categories={}, statuses={}, evidence={}".format(
                ",".join(backend_ids) or "*",
                ",".join(categories) or "*",
                ",".join(statuses) or "*",
                evidence,
            )
        )
    print("")

    for backend in matrix["backends"]:
        if backend_ids and backend["id"] not in backend_ids:
            continue
        matching_rows = [row for row in rows if row["backend_id"] == backend["id"]]
        if not matching_rows:
            continue
        missing = sum(1 for row in matching_rows if row["evidence_count"] == 0)
        present = len(matching_rows) - missing
        print("{}: present={}, missing={}".format(backend["name"], present, missing))

    if rows:
        print("")
        print("Evidence rows:")
        for item in rows:
            print(
                "- {backend}: {feature} [{status}] evidence={evidence_count}".format(
                    backend=item["backend"],
                    feature=item["feature"],
                    status=item["status"],
                    evidence_count=item["evidence_count"],
                )
            )

    if fail_on_missing and report["summary"]["missing_evidence_count"]:
        return 1
    return 0


def docs_report(
    backends_data,
    features_data,
    output_path,
    timeout,
    strict,
    max_linked_pages,
):
    support_signals = load_support_signals_module()

    report = support_signals.build_docs_report(
        backends_data,
        features_data,
        timeout=timeout,
        max_linked_pages=max_linked_pages,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stable_json(report), encoding="utf-8")

    for row in report["documents"]:
        status = row.get("status", "error")
        marker = "OK" if row.get("ok") else "FAIL"
        print("{} {} {} - {}".format(marker, status, row["backend"], row["source"]))
        if not row.get("ok"):
            print("  {}".format(row.get("error", row.get("url"))))

    if strict and report["summary"]["failed"]:
        return 1
    return 0


def load_and_validate():
    backends_data = load_json(BACKENDS_PATH)
    features_data = load_json(FEATURES_PATH)
    validate_catalogs(backends_data, features_data)
    matrix = build_matrix(backends_data, features_data)
    validate_matrix(matrix)
    return backends_data, features_data, matrix


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("update", help="Regenerate checked-in support artifacts")
    check_parser = subparsers.add_parser(
        "check", help="Validate catalogs and fail if generated artifacts are stale"
    )
    check_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path for generated artifact freshness.",
    )

    audit_parser = subparsers.add_parser("audit", help="Print current support backlog")
    audit_parser.add_argument(
        "--fail-on",
        action="append",
        choices=STATUS_ORDER,
        default=[],
        help="Exit non-zero if the backlog contains this status. Can be repeated.",
    )
    audit_parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Filter backlog by backend id. Comma-separated values are accepted.",
    )
    audit_parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Filter backlog by feature category. Comma-separated values are accepted.",
    )
    audit_parser.add_argument(
        "--status",
        action="append",
        default=[],
        help=(
            "Filter backlog by support status. Can be repeated; comma-separated "
            "values are accepted."
        ),
    )
    audit_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report output path for the filtered backlog.",
    )

    evidence_parser = subparsers.add_parser(
        "evidence", help="Audit support rows by evidence coverage"
    )
    evidence_parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Filter support rows by backend id. Comma-separated values are accepted.",
    )
    evidence_parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Filter support rows by feature category. Comma-separated values are accepted.",
    )
    evidence_parser.add_argument(
        "--status",
        action="append",
        default=[],
        help=(
            "Filter support rows by support status. Can be repeated; "
            "comma-separated values are accepted."
        ),
    )
    evidence_parser.add_argument(
        "--evidence",
        choices=("any", "present", "missing"),
        default="any",
        help="Filter rows by whether evidence is present.",
    )
    evidence_parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero when the filtered rows include missing evidence.",
    )
    evidence_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report output path for the filtered evidence rows.",
    )

    docs_parser = subparsers.add_parser(
        "docs", help="Fetch official backend documentation URLs and write a report"
    )
    docs_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DOC_REPORT_PATH,
        help="Report JSON path",
    )
    docs_parser.add_argument("--timeout", type=float, default=20.0)
    docs_parser.add_argument(
        "--max-linked-pages",
        type=int,
        default=3,
        help="Fetch up to this many relevant same-site links per configured docs URL",
    )
    docs_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any documentation URL cannot be fetched",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    try:
        backends_data, features_data, matrix = load_and_validate()
    except SupportMatrixError as exc:
        print(f"support matrix error: {exc}", file=sys.stderr)
        return 2

    if args.command == "update":
        write_generated(matrix)
        print(f"Updated {relpath(MATRIX_JSON_PATH)}")
        print(f"Updated {relpath(GRAPHICS_ROADMAP_JSON_PATH)}")
        print(f"Updated {relpath(PROJECT_PORTING_ROADMAP_JSON_PATH)}")
        print(f"Updated {relpath(DOCS_RST_PATH)}")
        return 0

    if args.command == "check":
        report = build_generated_check_report(matrix)
        if args.output is not None:
            output = args.output if args.output.is_absolute() else ROOT / args.output
            write_json_report(output, report)
        if not report["ok"]:
            print_generated_failures(report)
            return 1
        print("Support matrix catalogs and generated artifacts are current.")
        return 0

    if args.command == "audit":
        output = args.output
        if output is not None and not output.is_absolute():
            output = ROOT / output
        try:
            return audit(
                matrix,
                args.fail_on,
                backend_ids=args.backend,
                categories=args.category,
                statuses=args.status,
                output=output,
            )
        except SupportMatrixError as exc:
            print(f"support matrix error: {exc}", file=sys.stderr)
            return 2

    if args.command == "evidence":
        output = args.output
        if output is not None and not output.is_absolute():
            output = ROOT / output
        try:
            return evidence_audit(
                matrix,
                backend_ids=args.backend,
                categories=args.category,
                statuses=args.status,
                evidence=args.evidence,
                output=output,
                fail_on_missing=args.fail_on_missing,
            )
        except SupportMatrixError as exc:
            print(f"support matrix error: {exc}", file=sys.stderr)
            return 2

    if args.command == "docs":
        output_path = args.output
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        return docs_report(
            backends_data,
            features_data,
            output_path,
            args.timeout,
            args.strict,
            args.max_linked_pages,
        )

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
