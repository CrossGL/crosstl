#!/usr/bin/env python3
"""Extract automated support signals from docs metadata, code, and tests.

This report is intentionally generated data. The checked-in support catalog
still defines the reviewed feature taxonomy, while this tool independently
scans implementation paths, tests, unsupported markers, and optional
documentation probe metadata to find evidence gaps and possible drift.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
from urllib import error, parse, request

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIR = ROOT / "support"
BACKENDS_PATH = SUPPORT_DIR / "backends.json"
FEATURES_PATH = SUPPORT_DIR / "features.json"
DEFAULT_OUTPUT_PATH = SUPPORT_DIR / "generated" / "support-signals.json"
DEFAULT_DOCS_REPORT_PATH = SUPPORT_DIR / "generated" / "backend-docs-report.json"
PYTEST_FAILURE_SUMMARY_GENERATOR = "tools/pytest_failure_summary.py"
DOCS_REPORT_GENERATOR = "tools/support_signals.py docs"
DOCS_REPORT_SCHEMA_VERSION = 1
DOCS_REPORT_REQUIRED_FIELDS = (
    "schema_version",
    "generator",
    "summary",
    "documents",
)
FRONTEND_ID = "frontend"
FRONTEND_NAME = "Frontend / IR / Parser"
PROJECT_FEATURE_PREFIX = "project."
PROJECT_IMPLEMENTATION_PATHS = (
    "crosstl/project",
    "crosstl/_crosstl.py",
)
PROJECT_TEST_PATHS = ("tests/test_translator/test_project_translation.py",)

# Support-signal extraction treats every reviewed non-success catalog row as
# already accounted for. Issue sync keeps the narrower actionable backlog
# policy in sync_support_issues.py.
CATALOG_NON_SUCCESS_STATUSES = {
    "partial",
    "diagnostic",
    "validated_rejection",
    "unsupported",
    "unknown",
}
CATALOG_REVIEW_STATUSES = {"unsupported"}

STOP_WORDS = {
    "and",
    "are",
    "backend",
    "backends",
    "builtin",
    "builtins",
    "code",
    "crossgl",
    "current",
    "deterministic",
    "emit",
    "emits",
    "feature",
    "forms",
    "full",
    "function",
    "functions",
    "helper",
    "helpers",
    "implemented",
    "including",
    "invalid",
    "lower",
    "lowering",
    "native",
    "operations",
    "preserve",
    "reject",
    "resource",
    "resources",
    "shader",
    "source",
    "support",
    "target",
    "test",
    "tests",
    "the",
    "this",
    "type",
    "types",
    "validation",
    "where",
    "with",
}

GENERIC_FEATURE_TOKENS = {
    "stage",
    "stages",
    "language",
    "resources",
    "texture",
    "textures",
    "image",
    "images",
    "source",
    "target",
    "validation",
}
CATALOG_REVIEW_NOISE_TERMS = GENERIC_FEATURE_TOKENS | {
    "acces",
    "access",
    "array",
    "arrays",
    "attribute",
    "attributes",
    "basic",
    "block",
    "blocks",
    "buffer",
    "buffers",
    "coordinate",
    "count",
    "diagnostic",
    "diagnostics",
    "direct",
    "docs",
    "entry",
    "fixed",
    "for",
    "from",
    "handle",
    "handling",
    "input",
    "level",
    "levels",
    "member",
    "members",
    "object",
    "output",
    "parameter",
    "parameters",
    "point",
    "points",
    "resource",
    "resources",
    "sample",
    "sampler",
    "semantic",
    "semantics",
    "shadow",
    "size",
    "style",
    "texture",
    "textures",
    "unsupported",
    "variant",
    "variants",
    "without",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TEST_DEF_RE = re.compile(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
TEST_CLASS_RE = re.compile(r"^\s*class\s+(Test[A-Za-z0-9_]*)\s*[:(]", re.MULTILINE)
UNSUPPORTED_RE = re.compile(
    r"unsupported|not support|does not support|notimplemented", re.IGNORECASE
)
DOC_CANDIDATE_RE = re.compile(
    r"^(?:Op[A-Z]|gl[A-Z_]|vk[A-Z]|SV_|RW|ByteAddress|Structured|Texture|Sampler|"
    r"Image|Buffer|Ray|Mesh|Wave|Atomic|Barrier|Descriptor|Pipeline)"
)
LOWER_API_CANDIDATE_RE = re.compile(
    r"^(?:cuda|hip|atomic|barrier_|texture[A-Z0-9]|sampler[A-Z0-9]|"
    r"buffer[A-Z0-9]|image[A-Z0-9]|ray[A-Z0-9]|mesh[A-Z0-9]|"
    r"wave[A-Z0-9]|descriptor[A-Z0-9]|pipeline[A-Z0-9])"
)
SKIPPED_LINK_EXTENSIONS = {
    ".7z",
    ".css",
    ".dmg",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".mp4",
    ".png",
    ".svg",
    ".tar",
    ".tgz",
    ".webp",
    ".zip",
}
LINK_KEYWORDS = {
    "api",
    "atomic",
    "barrier",
    "buffer",
    "builtin",
    "descriptor",
    "function",
    "guide",
    "image",
    "intrinsic",
    "language",
    "manual",
    "mesh",
    "pipeline",
    "ray",
    "reference",
    "sampler",
    "shader",
    "spec",
    "texture",
    "wave",
}
DOC_CANDIDATE_NOISE = {
    "atomically",
    "descriptorhandle",
    "fontdescriptor",
    "getimagetag",
    "gltf",
    "hipdeviceptrt",
    "largeimage",
    "opcapability",
    "opextension",
    "opline",
    "opmemorymodel",
    "opname",
    "opnop",
    "opnoline",
    "opsourcecontinued",
    "opsourceextension",
    "opstring",
    "optypedeviceevent",
    "optypeopaque",
    "optypepipe",
    "optypequeue",
    "optypereserveid",
    "optypexxx",
    "pipelineenablealtera",
    "pipelineenableintel",
    "pipelines",
    "summarylargeimage",
}
GENERIC_DOC_CANDIDATES = {
    "atomic",
    "barrier",
    "buffer",
    "descriptor",
    "image",
    "mesh",
    "pipeline",
    "ray",
    "sampler",
    "texture",
    "wave",
}
DOC_CANDIDATE_FEATURE_ALIASES = {
    "descriptors": "resources.bindings",
    "descriptorset": "resources.bindings",
    "raytracing": "stage.ray_tracing",
    "svinstanceid": "io.stage_parameters",
    "svvertexid": "io.stage_parameters",
}
DOC_CANDIDATE_FEATURE_ALIAS_PATTERNS = (
    (re.compile(r"^descriptorheap"), "resources.bindings"),
    (re.compile(r"^gl[a-z0-9]+"), "io.stage_parameters"),
    (re.compile(r"^op(?:arraylength|typearray|typeruntimearray)$"), "language.arrays"),
    (
        re.compile(
            r"^op(?:branch|branchconditional|label|loopmerge|phi|"
            r"selectionmerge|switch)$"
        ),
        "language.control_flow",
    ),
    (
        re.compile(r"^op(?:functioncall|functionparameter|typevoid)$"),
        "language.functions",
    ),
    (
        re.compile(
            r"^op(?:constant|constantcomposite|constantnull|copyobject|"
            r"specconstantop|typebool|typefloat|typeint|typematrix|typevector|"
            r"undef)$"
        ),
        "language.vector_matrix",
    ),
    (
        re.compile(
            r"^op(?:decorate|decorationgroup|executionmode|executionmodeid|"
            r"memberdecorate|variable)$"
        ),
        "resources.bindings",
    ),
    (
        re.compile(
            r"^op(?:load|ptr(?:not)?equal|store|typeforwardpointer|typepointer)$"
        ),
        "resources.memory_qualifiers",
    ),
    (re.compile(r"^op(?:extinst|extinstimport)$"), "target.codegen"),
    (re.compile(r"^op(?:membername|typestruct)$"), "language.structs"),
    (re.compile(r"^structured$"), "resources.structured_buffers"),
    (re.compile(r"^sv[a-z0-9]+"), "io.stage_parameters"),
    (
        re.compile(r"^rw(?:byteaddress|structured)?buffer"),
        "resources.structured_buffers",
    ),
    (re.compile(r"^sampler[a-z0-9]*"), "resources.texture_sampler_split"),
    (re.compile(r"^texture[a-z0-9]*"), "texture.sampling"),
    (re.compile(r"^image[a-z0-9]*"), "image.storage"),
    (
        re.compile(
            r"^hip(?:addressmode|bindtexture|channelformatdesc|filtermode|"
            r"getchanneldesc|resource(?:view)?desc|tex(?:object|ref))"
        ),
        "resources.texture_sampler_split",
    ),
    (
        re.compile(
            r"^hip(?:array|extent|free)?mipmappedarray|"
            r"^hip(?:array|arrayconst|arrayformat|extent|getmipmappedarraylevel|"
            r"mallocmipmappedarray)"
        ),
        "resources.resource_arrays",
    ),
)
DOC_CANDIDATE_NOISE_PATTERNS = (re.compile(r"^(?:cuda|hip)(?:error|success)"),)


class DocumentHTMLParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
            return
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self.links.append(parse.urljoin(self.base_url, href))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.text_parts.append(text)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def relpath(path: Path) -> str:
    return os.path.relpath(str(path), str(ROOT)).replace(os.sep, "/")


def json_load_error(path: Path | None, error_type: str, message: str) -> dict[str, Any]:
    return {
        "load_error": {
            "path": relpath(path) if path is not None else None,
            "type": error_type,
            "message": message,
        }
    }


def value_type_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def type_label(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(type_label(item) for item in expected_type)
    if expected_type is dict:
        return "object"
    if expected_type is list:
        return "list"
    if expected_type is bool:
        return "bool"
    if expected_type is int:
        return "int"
    if expected_type is str:
        return "str"
    return expected_type.__name__


def value_matches_type(value: Any, expected_type: type | tuple[type, ...]) -> bool:
    if isinstance(expected_type, tuple):
        return any(value_matches_type(value, item) for item in expected_type)
    if expected_type is bool:
        return type(value) is bool
    if expected_type is int:
        return type(value) is int
    return isinstance(value, expected_type)


def invalid_report_field(
    path: Path | None,
    field: str,
    expected_type: type | tuple[type, ...],
    value: Any,
) -> dict[str, Any]:
    return json_load_error(
        path,
        "InvalidReportField",
        "{} must be {}, got {}".format(
            field,
            type_label(expected_type),
            value_type_label(value),
        ),
    )["load_error"]


def missing_report_fields(
    path: Path | None,
    field: str,
    fields: list[str],
) -> dict[str, Any]:
    return json_load_error(
        path,
        "MissingReportFields",
        "{} missing required fields: {}".format(field, ", ".join(fields)),
    )["load_error"]


def validate_required_fields(
    value: dict[str, Any],
    path: Path | None,
    field: str,
    required_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    missing = [required for required in required_fields if required not in value]
    if missing:
        return missing_report_fields(path, field, missing)
    return None


def validate_nested_field_types(
    value: dict[str, Any],
    path: Path | None,
    field: str,
    fields: dict[str, type | tuple[type, ...]],
) -> dict[str, Any] | None:
    for key, expected_type in fields.items():
        if key not in value:
            continue
        if not value_matches_type(value[key], expected_type):
            return invalid_report_field(
                path,
                f"{field}.{key}",
                expected_type,
                value[key],
            )
    return None


def validate_string_list(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return invalid_report_field(path, field, list, value)
    for index, item in enumerate(value):
        if not value_matches_type(item, str):
            return invalid_report_field(path, f"{field}[{index}]", str, item)
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def candidate_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return key[:80] or "unknown"


def looks_binary(content: bytes) -> bool:
    if not content:
        return False
    sample = content[:4096]
    if b"\0" in sample:
        return True
    control_count = sum(1 for byte in sample if byte < 9 or 13 < byte < 32)
    return control_count / max(len(sample), 1) > 0.08


def extract_pdf_text(content: bytes) -> tuple[str, dict[str, Any]]:
    reader_class = None
    parser_name = None
    try:
        from pypdf import PdfReader  # type: ignore

        reader_class = PdfReader
        parser_name = "pypdf"
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader_class = PdfReader
            parser_name = "PyPDF2"
        except Exception as exc:
            return (
                "",
                {
                    "kind": "pdf",
                    "parser": None,
                    "error": f"No PDF text extractor is available: {exc}",
                    "links": [],
                    "text_length": 0,
                },
            )

    text_parts = []
    errors = 0
    try:
        reader = reader_class(BytesIO(content))
        page_count = len(reader.pages)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                errors += 1
    except Exception as exc:
        return (
            "",
            {
                "kind": "pdf",
                "parser": parser_name,
                "error": str(exc),
                "links": [],
                "text_length": 0,
            },
        )

    text = "\n".join(text_parts)
    return (
        text,
        {
            "kind": "pdf",
            "parser": parser_name,
            "page_count": page_count,
            "page_errors": errors,
            "links": [],
            "text_length": len(text),
        },
    )


def extract_html_text(text: str, base_url: str) -> tuple[str, list[str]]:
    parser = DocumentHTMLParser(base_url)
    parser.feed(text)
    return "\n".join(parser.text_parts), parser.links


def extract_document_text(
    content: bytes, content_type: str, url: str
) -> tuple[str, dict[str, Any]]:
    content_type_lower = content_type.lower()
    parsed_url = parse.urlparse(url)
    path_lower = parsed_url.path.lower()
    if "application/pdf" in content_type_lower or path_lower.endswith(".pdf"):
        return extract_pdf_text(content)

    if looks_binary(content):
        return (
            "",
            {
                "kind": "binary",
                "links": [],
                "text_length": 0,
            },
        )

    decoded = content.decode("utf-8", errors="replace")
    if "html" in content_type_lower or "<html" in decoded[:4096].lower():
        html_text, links = extract_html_text(decoded, url)
        return (
            html_text,
            {
                "kind": "html",
                "links": links,
                "text_length": len(html_text),
            },
        )

    return (
        decoded,
        {
            "kind": "text",
            "links": [],
            "text_length": len(decoded),
        },
    )


def iter_python_files(entries: list[str]) -> list[Path]:
    files: list[Path] = []
    for entry in entries:
        path = ROOT / entry
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(
                child
                for child in path.rglob("*.py")
                if "__pycache__" not in child.parts
            )
    return sorted(set(files))


@lru_cache(maxsize=None)
def cached_file_tokens(path_text: str) -> frozenset[str]:
    return frozenset(tokenize(read_text(ROOT / path_text)))


@lru_cache(maxsize=None)
def cached_file_identifiers(path_text: str) -> frozenset[str]:
    return frozenset(
        normalize_identifier(token)
        for token in TOKEN_RE.findall(read_text(ROOT / path_text))
        if normalize_identifier(token)
    )


@lru_cache(maxsize=None)
def cached_test_symbols(path_text: str) -> tuple[str, ...]:
    path = ROOT / path_text
    text = read_text(path)
    names: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        names.update(TEST_DEF_RE.findall(text))
        names.update(TEST_CLASS_RE.findall(text))
        return tuple(sorted(names))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.add(node.name)
    return tuple(sorted(names))


@lru_cache(maxsize=None)
def cached_unsupported_lines(path_text: str) -> tuple[tuple[str, frozenset[str]], ...]:
    rows = []
    for line in read_text(ROOT / path_text).splitlines():
        if UNSUPPORTED_RE.search(line):
            rows.append((line.strip()[:180], frozenset(tokenize(line))))
    return tuple(rows)


def backend_implementation_paths(backend: dict[str, Any]) -> list[str]:
    paths = []
    for key in ("translator_codegen", "native_backend"):
        value = backend.get(key)
        if value:
            paths.append(value)
    return paths


def feature_implementation_paths(
    feature: dict[str, Any],
    backend: dict[str, Any],
) -> list[str]:
    if feature.get("id", "").startswith(PROJECT_FEATURE_PREFIX):
        return list(PROJECT_IMPLEMENTATION_PATHS)
    return backend_implementation_paths(backend)


def feature_test_paths(feature: dict[str, Any], backend: dict[str, Any]) -> list[str]:
    if feature.get("id", "").startswith(PROJECT_FEATURE_PREFIX):
        return list(PROJECT_TEST_PATHS)
    return list(backend.get("tests", []))


def backend_source_kind(backend: dict[str, Any]) -> str:
    return backend.get("source_kind", "native")


def split_identifier(token: str) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"[_\W]+", token):
        if not chunk:
            continue
        parts.extend(CAMEL_RE.sub(" ", chunk).split())
    return [part.lower() for part in parts if part]


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in TOKEN_RE.findall(text):
        for part in split_identifier(token):
            if len(part) < 3:
                continue
            if part in STOP_WORDS:
                continue
            tokens.add(part)
            if part.endswith("s") and len(part) > 4:
                tokens.add(part[:-1])
    return tokens


def feature_terms(feature: dict[str, Any]) -> list[str]:
    fields = [
        feature.get("id", "").replace(".", " "),
        feature.get("category", ""),
        feature.get("name", ""),
        feature.get("description", ""),
    ]
    terms = tokenize(" ".join(fields))
    id_terms = tokenize(feature.get("id", "").replace(".", " "))
    terms.update(term for term in id_terms if term not in GENERIC_FEATURE_TOKENS)
    return sorted(terms)


def score_tokens(candidate_tokens: set[str], terms: set[str]) -> tuple[int, list[str]]:
    matched = sorted(candidate_tokens & terms)
    return len(matched), matched


def test_symbols(paths: list[str]) -> list[dict[str, str]]:
    symbols: list[dict[str, str]] = []
    for path in iter_python_files(paths):
        path_text = relpath(path)
        for name in cached_test_symbols(path_text):
            symbols.append({"path": path_text, "symbol": name})
    return sorted(symbols, key=lambda item: (item["path"], item["symbol"]))


def collect_test_hits(
    paths: list[str], terms: list[str], *, limit: int = 8
) -> list[dict[str, Any]]:
    term_set = set(terms)
    ranked = []
    for symbol in test_symbols(paths):
        token_set = tokenize(symbol["symbol"])
        score, matched = score_tokens(token_set, term_set)
        if score <= 0:
            continue
        ranked.append(
            {
                "path": symbol["path"],
                "symbol": symbol["symbol"],
                "matched_terms": matched,
                "score": score,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["path"], item["symbol"]))
    return ranked[:limit]


def collect_file_hits(
    paths: list[str], terms: list[str], *, limit: int = 8
) -> list[dict[str, Any]]:
    term_set = set(terms)
    ranked = []
    for path in iter_python_files(paths):
        token_set = set(cached_file_tokens(relpath(path)))
        score, matched = score_tokens(token_set, term_set)
        if score <= 0:
            continue
        ranked.append(
            {
                "path": relpath(path),
                "matched_terms": matched,
                "score": score,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["path"]))
    return ranked[:limit]


def collect_identifier_hits(
    paths: list[str], candidate: str, *, limit: int = 8
) -> list[dict[str, Any]]:
    normalized = normalize_identifier(candidate)
    if not normalized:
        return []
    hits = []
    for path in iter_python_files(paths):
        path_text = relpath(path)
        identifiers = cached_file_identifiers(path_text)
        if normalized not in identifiers:
            continue
        hits.append(
            {
                "path": path_text,
                "matched_terms": [candidate],
                "score": 1,
            }
        )
        if len(hits) >= limit:
            break
    return hits


def collect_unsupported_hits(
    paths: list[str], terms: list[str], *, limit: int = 8
) -> list[dict[str, Any]]:
    term_set = set(terms)
    hits = []
    for path in iter_python_files(paths):
        path_text = relpath(path)
        for line, token_set in cached_unsupported_lines(path_text):
            score, matched = score_tokens(token_set, term_set)
            if score <= 0:
                continue
            hits.append(
                {
                    "path": path_text,
                    "matched_terms": matched,
                    "score": score,
                    "text": line.strip()[:180],
                }
            )
            if len(hits) >= limit:
                return hits
    return hits


def docs_feature_hits(
    docs_report: dict[str, Any] | None, backend_id: str, feature_id: str
) -> list[dict[str, Any]]:
    if not docs_report:
        return []
    rows = []
    for document in docs_report.get("documents", []):
        if document.get("backend_id") != backend_id:
            continue
        for hit in document.get("feature_hits", []):
            if hit.get("feature_id") == feature_id:
                rows.append(
                    {
                        "source": document.get("source", ""),
                        "url": document.get("url", ""),
                        "matched_terms": hit.get("matched_terms", []),
                        "score": hit.get("score", 0),
                    }
                )
    rows.sort(key=lambda item: (-item["score"], item["source"]))
    return rows[:8]


def document_feature_hits(
    text: str, features: list[dict[str, Any]], *, limit: int = 80
) -> list[dict[str, Any]]:
    document_terms = tokenize(text)
    hits = []
    for feature in features:
        terms = set(feature_terms(feature))
        score, matched = score_tokens(document_terms, terms)
        strong_terms = [term for term in matched if term not in GENERIC_FEATURE_TOKENS]
        if score < 2 and not strong_terms:
            continue
        hits.append(
            {
                "feature_id": feature["id"],
                "category": feature["category"],
                "name": feature["name"],
                "matched_terms": matched,
                "score": score,
            }
        )
    hits.sort(key=lambda item: (-item["score"], item["feature_id"]))
    return hits[:limit]


def document_candidate_terms(text: str, *, limit: int = 80) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for token in TOKEN_RE.findall(text):
        if len(token) < 4:
            continue
        parts = split_identifier(token)
        if not parts:
            continue
        lowered = token.lower()
        if lowered in STOP_WORDS:
            continue
        if DOC_CANDIDATE_RE.search(token) or LOWER_API_CANDIDATE_RE.search(token):
            counter[token] += 1
    return [
        {"term": term, "count": count} for term, count in counter.most_common(limit)
    ]


def canonical_doc_url(url: str) -> str:
    url, _fragment = parse.urldefrag(url)
    parsed = parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    return parsed._replace(query="").geturl()


def link_score(url: str, base_url: str, backend: dict[str, Any]) -> int:
    parsed = parse.urlparse(url)
    base = parse.urlparse(base_url)
    if parsed.netloc and parsed.netloc != base.netloc:
        return 0
    extension = Path(parsed.path).suffix.lower()
    if extension in SKIPPED_LINK_EXTENSIONS:
        return 0
    haystack = f"{parsed.path} {parsed.query}".lower()
    backend_terms = set(backend.get("aliases", [])) | set(
        split_identifier(backend["id"])
    )
    keywords = LINK_KEYWORDS | backend_terms
    score = sum(1 for keyword in keywords if keyword and keyword in haystack)
    return score


def select_doc_links(
    links: list[str], base_url: str, backend: dict[str, Any], *, limit: int
) -> list[str]:
    if limit <= 0:
        return []
    seen = {canonical_doc_url(base_url)}
    ranked = []
    for link in links:
        canonical = canonical_doc_url(link)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        score = link_score(canonical, base_url, backend)
        if score <= 0:
            continue
        ranked.append((score, canonical))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [url for _score, url in ranked[:limit]]


def candidate_parts(term: str) -> set[str]:
    parts = set()
    for part in split_identifier(term):
        if len(part) < 3 or part in STOP_WORDS:
            continue
        parts.add(part)
        if part.endswith("s") and len(part) > 4:
            parts.add(part[:-1])
    return parts


def actionable_doc_candidate(term: str, backend_id: str | None = None) -> bool:
    normalized = normalize_identifier(term)
    if len(normalized) < 4:
        return False
    if normalized in DOC_CANDIDATE_NOISE:
        return False
    if any(pattern.search(normalized) for pattern in DOC_CANDIDATE_NOISE_PATTERNS):
        return False
    if backend_id == "cuda" and normalized.startswith("cuda"):
        return False
    if backend_id != "vulkan" and term.startswith(("Op", "vk")):
        return False
    parts = candidate_parts(term)
    if not parts:
        return False
    if all(part in GENERIC_DOC_CANDIDATES for part in parts):
        return False
    if (
        term.isupper()
        and not term.startswith(("SV_", "GL_", "SPV_", "VK_"))
        and not term.startswith("Op")
    ):
        return False
    if re.search(r"_[0-9]+$", term):
        return False
    has_specific_shape = (
        DOC_CANDIDATE_RE.search(term) is not None
        or "_" in term
        or CAMEL_RE.search(term) is not None
    )
    if has_specific_shape:
        return True
    if len(parts) == 1 and next(iter(parts)) in GENERIC_DOC_CANDIDATES:
        return False
    return len(parts & GENERIC_DOC_CANDIDATES) > 0 and len(parts) >= 2


def candidate_feature_matches(
    term: str, features: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    normalized = normalize_identifier(term)
    alias_feature_id = DOC_CANDIDATE_FEATURE_ALIASES.get(normalized)
    if not alias_feature_id:
        alias_feature_id = next(
            (
                feature_id
                for pattern, feature_id in DOC_CANDIDATE_FEATURE_ALIAS_PATTERNS
                if pattern.search(normalized)
            ),
            None,
        )
    if alias_feature_id:
        for feature in features:
            if feature["id"] == alias_feature_id:
                return [
                    {
                        "feature_id": feature["id"],
                        "name": feature["name"],
                        "matched_terms": [term],
                        "score": 1,
                    }
                ]

    parts = candidate_parts(term)
    if not parts:
        return []
    matches = []
    for feature in features:
        terms = set(feature_terms(feature))
        matched = sorted(parts & terms)
        family_match = bool(set(matched) & GENERIC_DOC_CANDIDATES)
        if len(matched) < 2 and not family_match:
            continue
        matches.append(
            {
                "feature_id": feature["id"],
                "name": feature["name"],
                "matched_terms": matched,
                "score": len(matched),
            }
        )
    matches.sort(key=lambda item: (-item["score"], item["feature_id"]))
    return matches[:5]


def fetch_url(url: str, timeout: float) -> dict[str, Any]:
    started = time.time()
    req = request.Request(
        url,
        headers={
            "User-Agent": "CrossGL-support-signals/1.0 (+https://github.com/CrossGL)"
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            content = response.read()
            status = getattr(response, "status", response.getcode())
            content_type = response.headers.get("content-type", "")
            text, extraction = extract_document_text(
                content, content_type, response.geturl()
            )
            return {
                "ok": 200 <= int(status) < 400,
                "status": int(status),
                "url": url,
                "final_url": response.geturl(),
                "content_type": content_type,
                "content_length": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "elapsed_ms": int((time.time() - started) * 1000),
                "text": text,
                "text_extraction": extraction,
            }
    except error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "url": url,
            "final_url": exc.geturl(),
            "error": str(exc),
            "elapsed_ms": int((time.time() - started) * 1000),
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "error": f"{exc.__class__.__name__}: {exc}",
            "elapsed_ms": int((time.time() - started) * 1000),
        }


def build_docs_report(
    backends_data: dict[str, Any],
    features_data: dict[str, Any],
    *,
    timeout: float,
    max_linked_pages: int = 3,
) -> dict[str, Any]:
    documents = []
    seen_urls: set[str] = set()

    def append_document(
        backend: dict[str, Any],
        source: str,
        fetched: dict[str, Any],
        *,
        linked_from: str | None = None,
    ) -> None:
        text = fetched.get("text", "")
        row = {key: value for key, value in fetched.items() if key != "text"}
        row.update(
            {
                "backend_id": backend["id"],
                "backend": backend["name"],
                "source": source,
                "feature_hits": (
                    document_feature_hits(text, features_data["features"])
                    if fetched.get("ok")
                    else []
                ),
                "candidate_terms": (
                    document_candidate_terms(text) if fetched.get("ok") else []
                ),
            }
        )
        if linked_from:
            row["linked_from"] = linked_from
        documents.append(row)

    for backend in backends_data["backends"]:
        for doc in backend.get("docs", []):
            canonical = canonical_doc_url(doc["url"])
            if canonical in seen_urls:
                continue
            seen_urls.add(canonical)
            fetched = fetch_url(doc["url"], timeout)
            append_document(backend, doc["name"], fetched)

            links = fetched.get("text_extraction", {}).get("links", [])
            for linked_url in select_doc_links(
                links,
                fetched.get("final_url") or doc["url"],
                backend,
                limit=max_linked_pages,
            ):
                if linked_url in seen_urls:
                    continue
                seen_urls.add(linked_url)
                linked_fetched = fetch_url(linked_url, timeout)
                linked_name = "{} :: {}".format(
                    doc["name"],
                    Path(parse.urlparse(linked_url).path).name or linked_url,
                )
                append_document(
                    backend,
                    linked_name,
                    linked_fetched,
                    linked_from=doc["url"],
                )
    return {
        "schema_version": 1,
        "generator": "tools/support_signals.py docs",
        "source": {
            "backends": "support/backends.json",
            "features": "support/features.json",
        },
        "documents": documents,
        "summary": {
            "total": len(documents),
            "ok": sum(1 for row in documents if row.get("ok")),
            "failed": sum(1 for row in documents if not row.get("ok")),
            "feature_hits": sum(len(row.get("feature_hits", [])) for row in documents),
            "candidate_terms": sum(
                len(row.get("candidate_terms", [])) for row in documents
            ),
            "linked_documents": sum(1 for row in documents if row.get("linked_from")),
        },
    }


def infer_state(
    support: dict[str, Any],
    implementation_hits: list[dict[str, Any]],
    test_hits: list[dict[str, Any]],
    unsupported_hits: list[dict[str, Any]],
) -> str:
    if test_hits and implementation_hits:
        return "tested"
    if test_hits:
        return "tests_only"
    if unsupported_hits:
        return "unsupported_signal"
    if implementation_hits:
        return "implementation_only"
    if support.get("status") in CATALOG_NON_SUCCESS_STATUSES:
        return "catalog_backlog"
    return "not_detected"


def extraction_issue(
    backend: dict[str, Any],
    feature: dict[str, Any],
    support: dict[str, Any],
    state: str,
    test_hits: list[dict[str, Any]],
    unsupported_hits: list[dict[str, Any]],
) -> dict[str, Any] | None:
    status = support.get("status")
    if status in CATALOG_NON_SUCCESS_STATUSES:
        return None
    if status != "supported":
        return None
    if state in {"tested", "tests_only"}:
        return None

    if unsupported_hits:
        kind = "supported_with_unsupported_signal"
        title = "Review unsupported-code signal for supported row"
    elif not support.get("evidence"):
        kind = "supported_without_catalog_evidence"
        title = "Record or add tests for supported row"
    else:
        kind = "supported_without_detected_tests"
        title = "Extractor did not find tests for supported row"

    return {
        "key": "extracted:{}:{}:{}".format(backend["id"], feature["id"], kind),
        "kind": kind,
        "title": title,
        "backend_id": backend["id"],
        "backend": backend["name"],
        "feature_id": feature["id"],
        "feature": feature["name"],
        "category": feature["category"],
        "status": status,
        "state": state,
    }


def strong_catalog_review_terms(hits: list[dict[str, Any]]) -> set[str]:
    return {
        term
        for hit in hits
        for term in hit.get("matched_terms", [])
        if term not in CATALOG_REVIEW_NOISE_TERMS
    }


def catalog_review_issue(
    backend: dict[str, Any],
    feature: dict[str, Any],
    support: dict[str, Any],
    state: str,
    implementation_hits: list[dict[str, Any]],
    test_hits: list[dict[str, Any]],
    unsupported_hits: list[dict[str, Any]],
) -> dict[str, Any] | None:
    status = support.get("status")
    if status not in CATALOG_REVIEW_STATUSES:
        return None
    if state != "tested":
        return None
    if support.get("evidence") and unsupported_hits:
        return None

    matched_terms = sorted(
        strong_catalog_review_terms(implementation_hits)
        & strong_catalog_review_terms(test_hits)
    )
    if len(matched_terms) < 2:
        return None

    kind = "catalog_unsupported_with_detected_tests"
    return {
        "key": "extracted:{}:{}:{}".format(backend["id"], feature["id"], kind),
        "kind": kind,
        "title": "Review unsupported catalog row with detected implementation/tests",
        "backend_id": backend["id"],
        "backend": backend["name"],
        "feature_id": feature["id"],
        "feature": feature["name"],
        "category": feature["category"],
        "status": status,
        "state": state,
        "matched_terms": matched_terms[:12],
    }


def docs_candidates_by_backend(
    docs_report: dict[str, Any] | None, backend_id: str
) -> list[dict[str, Any]]:
    if not docs_report:
        return []
    candidates: dict[str, dict[str, Any]] = {}
    for document in docs_report.get("documents", []):
        if document.get("backend_id") != backend_id or not document.get("ok"):
            continue
        for candidate in document.get("candidate_terms", []):
            term = candidate.get("term", "")
            if not actionable_doc_candidate(term, backend_id):
                continue
            key = normalize_identifier(term)
            row = candidates.setdefault(
                key,
                {
                    "term": term,
                    "normalized": key,
                    "count": 0,
                    "sources": [],
                },
            )
            row["count"] += int(candidate.get("count", 0))
            row["sources"].append(
                {
                    "source": document.get("source", ""),
                    "url": document.get("url", ""),
                    "term": term,
                    "count": candidate.get("count", 0),
                }
            )
    rows = list(candidates.values())
    rows.sort(key=lambda item: (-item["count"], item["term"].lower()))
    return rows


def documented_candidate_issues(
    backends: list[dict[str, Any]],
    features: list[dict[str, Any]],
    docs_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not docs_report:
        return []
    issues = []
    for backend in backends:
        backend_id = backend["id"]
        implementation_paths = backend_implementation_paths(backend)
        test_paths = backend.get("tests", [])
        for candidate in docs_candidates_by_backend(docs_report, backend_id):
            mapped_features = candidate_feature_matches(candidate["term"], features)
            if mapped_features:
                continue
            implementation_hits = collect_identifier_hits(
                implementation_paths, candidate["term"]
            )
            test_hits = collect_identifier_hits(test_paths, candidate["term"])
            if implementation_hits and test_hits:
                kind = "documented_candidate_not_cataloged"
                title = "Classify documented API candidate in the support catalog"
                state = "detected_but_uncataloged"
            elif implementation_hits:
                kind = "documented_candidate_without_detected_tests"
                title = "Add tests or status for documented API candidate"
                state = "implementation_only"
            else:
                kind = "documented_candidate_not_detected"
                title = "Triage missing documented API candidate"
                state = "docs_only"

            feature_id = "docs.{}".format(candidate_key(candidate["term"]))
            issues.append(
                {
                    "key": f"extracted:{backend_id}:{feature_id}:{kind}",
                    "kind": kind,
                    "title": title,
                    "backend_id": backend_id,
                    "backend": backend["name"],
                    "feature_id": feature_id,
                    "feature": candidate["term"],
                    "category": "docs",
                    "status": "untracked",
                    "state": state,
                    "signal": {
                        "state": state,
                        "catalog_evidence_count": 0,
                        "docs": candidate["sources"][:8],
                        "implementation": implementation_hits,
                        "tests": test_hits,
                        "unsupported": [],
                    },
                }
            )
    issues.sort(
        key=lambda item: (
            item["backend_id"],
            item["kind"],
            item["feature"].lower(),
        )
    )
    return issues


PYTEST_FAILURE_SUMMARY_COUNTER_FIELDS = (
    "report_count",
    "load_error_count",
    "testcase_count",
    "failure_count",
    "error_count",
    "skipped_count",
    "failed_testcase_count",
)
PYTEST_FAILURE_REQUIRED_FAILURE_FIELDS = (
    "nodeid",
    "file",
    "kind",
    "category",
    "backend",
    "message",
)
PYTEST_FAILURE_CLEAN_WORKFLOW_FIELDS = (
    "workflow",
    "run_id",
    "conclusion",
    "head_sha",
)


def valid_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def pytest_failure_summary_load_error(
    path: Path, error_type: str, message: str
) -> dict[str, Any]:
    return {
        "path": relpath(path),
        "load_error": {
            "type": error_type,
            "message": message,
        },
    }


def pytest_failure_summary_mapping_errors(
    prefix: str, value: Any, required_fields: tuple[str, ...]
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]
    errors = []
    for field_name in required_fields:
        field_value = value.get(field_name)
        if not isinstance(field_value, str):
            errors.append(f"{prefix}.{field_name} must be a string")
    return errors


def pytest_failure_summary_count_map_errors(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix} must be an object"]
    errors = []
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{prefix} keys must be non-empty strings")
            break
        if not valid_non_negative_int(count):
            errors.append(f"{prefix}.{key} must be a non-negative integer")
    return errors


def pytest_failure_summary_report_errors(index: int, report: Any) -> list[str]:
    prefix = f"reports[{index}]"
    if not isinstance(report, Mapping):
        return [f"{prefix} must be an object"]
    errors = []
    path = report.get("path")
    if not isinstance(path, str) or not path:
        errors.append(f"{prefix}.path must be a non-empty string")
    if "load_error" in report:
        errors.extend(
            pytest_failure_summary_mapping_errors(
                f"{prefix}.load_error", report.get("load_error"), ("type", "message")
            )
        )
        return errors
    for field_name in ("tests", "failures", "errors", "skipped"):
        if not valid_non_negative_int(report.get(field_name)):
            errors.append(f"{prefix}.{field_name} must be a non-negative integer")
    failed_testcases = report.get("failed_testcases")
    if not isinstance(failed_testcases, list):
        errors.append(f"{prefix}.failed_testcases must be a list")
    else:
        for failure_index, failure in enumerate(failed_testcases):
            errors.extend(
                pytest_failure_summary_mapping_errors(
                    f"{prefix}.failed_testcases[{failure_index}]",
                    failure,
                    PYTEST_FAILURE_REQUIRED_FAILURE_FIELDS,
                )
            )
    return errors


def pytest_failure_summary_loaded_reports(
    reports: list[Any],
) -> list[Mapping[str, Any]]:
    return [
        report
        for report in reports
        if isinstance(report, Mapping) and "load_error" not in report
    ]


def pytest_failure_summary_nested_failures(
    reports: list[Any],
) -> list[Any]:
    nested_failures = []
    for report in pytest_failure_summary_loaded_reports(reports):
        failed_testcases = report.get("failed_testcases")
        if isinstance(failed_testcases, list):
            nested_failures.extend(failed_testcases)
    return nested_failures


def pytest_failure_summary_counter_mismatch_errors(
    summary: Mapping[str, Any],
    reports: list[Any],
    failures: list[Any],
) -> list[str]:
    loaded_reports = pytest_failure_summary_loaded_reports(reports)
    nested_failures = pytest_failure_summary_nested_failures(reports)
    valid_nested_failures = [
        failure for failure in nested_failures if isinstance(failure, Mapping)
    ]
    expected_counts = {
        "report_count": len(reports),
        "load_error_count": sum(
            1
            for report in reports
            if isinstance(report, Mapping) and report.get("load_error")
        ),
        "testcase_count": sum(
            report.get("tests", 0)
            for report in loaded_reports
            if valid_non_negative_int(report.get("tests"))
        ),
        "failure_count": sum(
            report.get("failures", 0)
            for report in loaded_reports
            if valid_non_negative_int(report.get("failures"))
        ),
        "error_count": sum(
            report.get("errors", 0)
            for report in loaded_reports
            if valid_non_negative_int(report.get("errors"))
        ),
        "skipped_count": sum(
            report.get("skipped", 0)
            for report in loaded_reports
            if valid_non_negative_int(report.get("skipped"))
        ),
        "failed_testcase_count": len(valid_nested_failures),
    }
    errors = []
    for field_name, expected_count in expected_counts.items():
        if (
            valid_non_negative_int(summary.get(field_name))
            and summary.get(field_name) != expected_count
        ):
            errors.append(f"summary.{field_name} must match reports")

    if failures != nested_failures:
        errors.append("failures must match reports[].failed_testcases")

    categories = Counter(
        failure.get("category")
        for failure in valid_nested_failures
        if isinstance(failure.get("category"), str)
    )
    backends = Counter(
        failure.get("backend")
        for failure in valid_nested_failures
        if isinstance(failure.get("backend"), str)
    )
    if isinstance(summary.get("categories"), Mapping) and dict(
        summary["categories"]
    ) != dict(sorted(categories.items())):
        errors.append("summary.categories must match failures")
    if isinstance(summary.get("backends"), Mapping) and dict(
        summary["backends"]
    ) != dict(sorted(backends.items())):
        errors.append("summary.backends must match failures")
    return errors


def pytest_failure_summary_contract_errors(report: Mapping[str, Any]) -> list[str]:
    errors = []
    for field_name in (
        "schema_version",
        "generator",
        "summary",
        "reports",
        "clean_workflow_runs",
        "failures",
    ):
        if field_name not in report:
            errors.append(f"{field_name} is required")

    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        errors.append("summary must be an object")
    else:
        for field_name in PYTEST_FAILURE_SUMMARY_COUNTER_FIELDS:
            if not valid_non_negative_int(summary.get(field_name)):
                errors.append(f"summary.{field_name} must be a non-negative integer")
        errors.extend(
            pytest_failure_summary_count_map_errors(
                "summary.categories", summary.get("categories")
            )
        )
        errors.extend(
            pytest_failure_summary_count_map_errors(
                "summary.backends", summary.get("backends")
            )
        )

    reports = report.get("reports")
    if not isinstance(reports, list):
        errors.append("reports must be a list")
        reports = []
    else:
        for index, parsed_report in enumerate(reports):
            errors.extend(pytest_failure_summary_report_errors(index, parsed_report))

    clean_workflow_runs = report.get("clean_workflow_runs")
    if not isinstance(clean_workflow_runs, list):
        errors.append("clean_workflow_runs must be a list")
        clean_workflow_runs = []
    else:
        for index, run in enumerate(clean_workflow_runs):
            errors.extend(
                pytest_failure_summary_mapping_errors(
                    f"clean_workflow_runs[{index}]",
                    run,
                    PYTEST_FAILURE_CLEAN_WORKFLOW_FIELDS,
                )
            )

    failures = report.get("failures")
    if not isinstance(failures, list):
        errors.append("failures must be a list")
        failures = []
    else:
        for index, failure in enumerate(failures):
            errors.extend(
                pytest_failure_summary_mapping_errors(
                    f"failures[{index}]",
                    failure,
                    PYTEST_FAILURE_REQUIRED_FAILURE_FIELDS,
                )
            )

    if not reports and not clean_workflow_runs and not failures:
        errors.append(
            "report must include at least one parsed report, clean workflow run, "
            "or failure"
        )
    if isinstance(summary, Mapping):
        errors.extend(
            pytest_failure_summary_counter_mismatch_errors(summary, reports, failures)
        )

    return errors


def load_pytest_failure_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return pytest_failure_summary_load_error(
            path,
            "FileNotFoundError",
            "pytest failure summary does not exist",
        )
    try:
        report = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return pytest_failure_summary_load_error(path, type(exc).__name__, str(exc))
    if not isinstance(report, Mapping):
        return pytest_failure_summary_load_error(
            path,
            "InvalidPytestFailureSummary",
            "expected a JSON object",
        )
    if report.get("generator") != PYTEST_FAILURE_SUMMARY_GENERATOR:
        return pytest_failure_summary_load_error(
            path,
            "UnexpectedGenerator",
            "expected {}, got {}".format(
                PYTEST_FAILURE_SUMMARY_GENERATOR,
                report.get("generator"),
            ),
        )
    errors = pytest_failure_summary_contract_errors(report)
    if errors:
        return pytest_failure_summary_load_error(
            path,
            "InvalidPytestFailureSummary",
            "; ".join(errors),
        )
    report.setdefault("path", relpath(path))
    return dict(report)


def summarize_pytest_failure_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = [
        failure
        for report in reports
        if not report.get("load_error")
        for failure in report.get("failures", [])
    ]
    nested_load_error_count = 0
    for report in reports:
        if report.get("load_error"):
            continue
        summary = report.get("summary")
        if not isinstance(summary, Mapping):
            continue
        load_error_count = summary.get("load_error_count", 0)
        if valid_non_negative_int(load_error_count):
            nested_load_error_count += load_error_count
    category_counts = Counter(
        failure.get("category", "unknown") for failure in failures
    )
    backend_counts = Counter(failure.get("backend", "unknown") for failure in failures)
    return {
        "provided": bool(reports),
        "report_count": len(reports),
        "load_error_count": (
            sum(1 for report in reports if report.get("load_error"))
            + nested_load_error_count
        ),
        "failed_testcase_count": len(failures),
        "categories": dict(sorted(category_counts.items())),
        "backends": dict(sorted(backend_counts.items())),
    }


def pytest_failure_backend_id(
    failure: dict[str, Any],
    backend_by_id: dict[str, dict[str, Any]],
) -> str:
    backend = failure.get("backend", "unknown")
    if backend in backend_by_id:
        return backend
    return FRONTEND_ID


def pytest_failure_sample(failure: dict[str, Any]) -> dict[str, Any]:
    message = str(failure.get("message", "")).splitlines()[0][:240]
    category = failure.get("category", "unknown")
    return {
        "nodeid": failure.get("nodeid", ""),
        "path": failure.get("file", ""),
        "kind": failure.get("kind", "failure"),
        "category": category,
        "backend": failure.get("backend", "unknown"),
        "message": message,
        "matched_terms": [category],
    }


def pytest_failure_issues(
    backends: list[dict[str, Any]],
    pytest_failure_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    backend_by_id = {backend["id"]: backend for backend in backends}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for report in pytest_failure_reports:
        if report.get("load_error"):
            continue
        for failure in report.get("failures", []):
            backend_id = pytest_failure_backend_id(failure, backend_by_id)
            category = failure.get("category", "unknown")
            grouped.setdefault((backend_id, category), []).append(failure)

    issues = []
    for (backend_id, category), failures in sorted(grouped.items()):
        backend = backend_by_id.get(
            backend_id,
            {"id": FRONTEND_ID, "name": FRONTEND_NAME},
        )
        category_key = candidate_key(category)
        feature_id = f"ci.pytest.{category_key}"
        samples = [pytest_failure_sample(failure) for failure in failures[:12]]
        issues.append(
            {
                "key": "extracted:{}:{}:pytest_failure_summary".format(
                    backend_id,
                    feature_id,
                ),
                "kind": "pytest_failure_summary",
                "title": "Investigate CI pytest failures for {}".format(
                    category.replace("_", " ")
                ),
                "backend_id": backend_id,
                "backend": backend["name"],
                "feature_id": feature_id,
                "feature": "CI pytest {} failures".format(category.replace("_", " ")),
                "category": "ci",
                "status": "failing",
                "state": "ci_failure",
                "signal": {
                    "state": "ci_failure",
                    "catalog_evidence_count": 0,
                    "docs": [],
                    "implementation": [],
                    "tests": [],
                    "unsupported": [],
                    "failures": samples,
                    "failure_count": len(failures),
                    "category": category,
                },
            }
        )
    return issues


def build_report(
    backends_data: dict[str, Any],
    features_data: dict[str, Any],
    docs_report: dict[str, Any] | None = None,
    pytest_failure_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    backends = backends_data["backends"]
    backend_by_id = {backend["id"]: backend for backend in backends}
    pytest_failure_reports = pytest_failure_reports or []
    features = []
    issues = []
    state_counts: dict[str, int] = {}

    for feature in features_data["features"]:
        terms = feature_terms(feature)
        support_by_backend = {}
        for backend_id, support in feature.get("support", {}).items():
            backend = backend_by_id[backend_id]
            implementation_paths = feature_implementation_paths(feature, backend)
            test_paths = feature_test_paths(feature, backend)
            implementation_hits = collect_file_hits(implementation_paths, terms)
            test_hits = collect_test_hits(test_paths, terms)
            unsupported_hits = collect_unsupported_hits(implementation_paths, terms)
            doc_hits = docs_feature_hits(docs_report, backend_id, feature["id"])
            state = infer_state(
                support, implementation_hits, test_hits, unsupported_hits
            )
            state_counts[state] = state_counts.get(state, 0) + 1
            issue = extraction_issue(
                backend, feature, support, state, test_hits, unsupported_hits
            )
            if issue:
                issues.append(issue)
            issue = catalog_review_issue(
                backend,
                feature,
                support,
                state,
                implementation_hits,
                test_hits,
                unsupported_hits,
            )
            if issue:
                issues.append(issue)
            support_by_backend[backend_id] = {
                "catalog_status": support.get("status"),
                "catalog_evidence_count": len(support.get("evidence", [])),
                "state": state,
                "docs": doc_hits,
                "implementation": implementation_hits,
                "tests": test_hits,
                "unsupported": unsupported_hits,
            }
        features.append(
            {
                "id": feature["id"],
                "category": feature["category"],
                "name": feature["name"],
                "terms": terms,
                "support": support_by_backend,
            }
        )

    issues.extend(
        documented_candidate_issues(
            backends,
            features_data["features"],
            docs_report,
        )
    )
    issues.extend(pytest_failure_issues(backends, pytest_failure_reports))
    issues.sort(
        key=lambda item: (item["backend_id"], item["category"], item["feature_id"])
    )
    pytest_failure_summary = summarize_pytest_failure_reports(pytest_failure_reports)
    return {
        "schema_version": 1,
        "generator": "tools/support_signals.py",
        "source": {
            "backends": "support/backends.json",
            "features": "support/features.json",
            "docs_report": docs_report_source(docs_report),
            "pytest_failure_summaries": [
                report.get("path")
                for report in pytest_failure_reports
                if report.get("path")
            ],
        },
        "summary": {
            "backend_count": len(backends),
            "feature_count": len(features),
            "state_counts": dict(sorted(state_counts.items())),
            "issue_count": len(issues),
            "pytest_failures": pytest_failure_summary,
            "docs_probe": docs_probe_summary(docs_report),
        },
        "backends": [
            {
                "id": backend["id"],
                "name": backend["name"],
                "source_kind": backend_source_kind(backend),
                "implementation_paths": backend_implementation_paths(backend),
                "test_paths": backend.get("tests", []),
            }
            for backend in backends
        ],
        "features": features,
        "issues": issues,
    }


def compare_file(path: Path, expected: str) -> list[str]:
    actual = path.read_text(encoding="utf-8") if path.exists() else ""
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


def load_optional_json_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        report = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return json_load_error(path, type(exc).__name__, str(exc))
    if not isinstance(report, dict):
        return json_load_error(
            path,
            "InvalidReportType",
            f"expected JSON object, got {type(report).__name__}",
        )
    return report


def json_report_schema_error(
    report: dict[str, Any],
    path: Path | None,
    *,
    expected_generator: str,
    required_fields: tuple[str, ...],
    schema_version: int,
) -> dict[str, Any] | None:
    missing_fields = [field for field in required_fields if field not in report]
    if missing_fields:
        return json_load_error(
            path,
            "MissingReportFields",
            "missing required fields: {}".format(", ".join(missing_fields)),
        )["load_error"]
    if report.get("schema_version") != schema_version:
        return json_load_error(
            path,
            "UnsupportedSchemaVersion",
            "expected schema_version {}, got {}".format(
                schema_version,
                report.get("schema_version"),
            ),
        )["load_error"]
    if report.get("generator") != expected_generator:
        return json_load_error(
            path,
            "UnexpectedReportGenerator",
            "expected generator {}, got {}".format(
                expected_generator,
                report.get("generator"),
            ),
        )["load_error"]
    return None


def validate_docs_counter_summary(
    summary: Any,
    path: Path | None,
) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return invalid_report_field(path, "summary", dict, summary)
    counters = (
        "total",
        "ok",
        "failed",
        "feature_hits",
        "candidate_terms",
        "linked_documents",
    )
    error = validate_required_fields(summary, path, "summary", counters)
    if error is not None:
        return error
    error = validate_nested_field_types(
        summary,
        path,
        "summary",
        {counter: int for counter in counters},
    )
    if error is not None:
        return error
    if summary["ok"] + summary["failed"] != summary["total"]:
        return json_load_error(
            path,
            "InvalidReportField",
            "summary.total must match ok + failed: {} != {}".format(
                summary["total"],
                summary["ok"] + summary["failed"],
            ),
        )["load_error"]
    return None


def validate_docs_feature_hits(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return invalid_report_field(path, field, list, value)
    for index, hit in enumerate(value):
        hit_field = f"{field}[{index}]"
        if not isinstance(hit, dict):
            return invalid_report_field(path, hit_field, dict, hit)
        error = validate_required_fields(
            hit,
            path,
            hit_field,
            ("feature_id", "category", "name", "matched_terms", "score"),
        )
        if error is not None:
            return error
        error = validate_nested_field_types(
            hit,
            path,
            hit_field,
            {
                "feature_id": str,
                "category": str,
                "name": str,
                "matched_terms": list,
                "score": int,
            },
        )
        if error is not None:
            return error
        error = validate_string_list(
            hit["matched_terms"],
            path,
            f"{hit_field}.matched_terms",
        )
        if error is not None:
            return error
    return None


def validate_docs_candidate_terms(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return invalid_report_field(path, field, list, value)
    for index, candidate in enumerate(value):
        candidate_field = f"{field}[{index}]"
        if not isinstance(candidate, dict):
            return invalid_report_field(path, candidate_field, dict, candidate)
        error = validate_required_fields(
            candidate,
            path,
            candidate_field,
            ("term", "count"),
        )
        if error is not None:
            return error
        error = validate_nested_field_types(
            candidate,
            path,
            candidate_field,
            {
                "term": str,
                "count": int,
            },
        )
        if error is not None:
            return error
    return None


def validate_docs_report_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    error = validate_docs_counter_summary(report.get("summary"), path)
    if error is not None:
        return error
    documents = report.get("documents")
    if not isinstance(documents, list):
        return invalid_report_field(path, "documents", list, documents)

    actual_ok = 0
    actual_failed = 0
    actual_feature_hits = 0
    actual_candidate_terms = 0
    actual_linked = 0
    for index, document in enumerate(documents):
        document_field = f"documents[{index}]"
        if not isinstance(document, dict):
            return invalid_report_field(path, document_field, dict, document)
        error = validate_required_fields(
            document,
            path,
            document_field,
            (
                "backend_id",
                "backend",
                "source",
                "url",
                "ok",
                "feature_hits",
                "candidate_terms",
            ),
        )
        if error is not None:
            return error
        error = validate_nested_field_types(
            document,
            path,
            document_field,
            {
                "backend_id": str,
                "backend": str,
                "source": str,
                "url": str,
                "final_url": str,
                "content_type": str,
                "content_length": int,
                "sha256": str,
                "elapsed_ms": int,
                "ok": bool,
                "status": int,
                "error": str,
                "text_extraction": dict,
                "linked_from": str,
                "feature_hits": list,
                "candidate_terms": list,
            },
        )
        if error is not None:
            return error
        text_extraction = document.get("text_extraction")
        if text_extraction is not None:
            error = validate_nested_field_types(
                text_extraction,
                path,
                f"{document_field}.text_extraction",
                {
                    "kind": str,
                    "parser": (str, type(None)),
                    "error": str,
                    "links": list,
                    "text_length": int,
                    "page_count": int,
                },
            )
            if error is not None:
                return error
            if "links" in text_extraction:
                error = validate_string_list(
                    text_extraction["links"],
                    path,
                    f"{document_field}.text_extraction.links",
                )
                if error is not None:
                    return error
        error = validate_docs_feature_hits(
            document["feature_hits"],
            path,
            f"{document_field}.feature_hits",
        )
        if error is not None:
            return error
        error = validate_docs_candidate_terms(
            document["candidate_terms"],
            path,
            f"{document_field}.candidate_terms",
        )
        if error is not None:
            return error
        if document["ok"]:
            actual_ok += 1
        else:
            actual_failed += 1
        actual_feature_hits += len(document["feature_hits"])
        actual_candidate_terms += len(document["candidate_terms"])
        if document.get("linked_from"):
            actual_linked += 1

    summary = report["summary"]
    expected = {
        "total": len(documents),
        "ok": actual_ok,
        "failed": actual_failed,
        "feature_hits": actual_feature_hits,
        "candidate_terms": actual_candidate_terms,
        "linked_documents": actual_linked,
    }
    for counter, expected_value in expected.items():
        if summary[counter] != expected_value:
            return json_load_error(
                path,
                "InvalidReportField",
                "summary.{} must match documents: {} != {}".format(
                    counter,
                    summary[counter],
                    expected_value,
                ),
            )["load_error"]
    return None


def load_docs_report(path: Path | None) -> dict[str, Any] | None:
    report = load_optional_json_report(path)
    if report is None or report.get("load_error"):
        return report
    schema_error = json_report_schema_error(
        report,
        path,
        expected_generator=DOCS_REPORT_GENERATOR,
        required_fields=DOCS_REPORT_REQUIRED_FIELDS,
        schema_version=DOCS_REPORT_SCHEMA_VERSION,
    )
    if schema_error is not None:
        return {"load_error": schema_error}
    contract_error = validate_docs_report_contract(report, path)
    if contract_error is not None:
        return {"load_error": contract_error}
    if path is not None:
        report.setdefault("path", relpath(path))
    return report


def docs_probe_summary(docs_report: dict[str, Any] | None) -> dict[str, Any]:
    if docs_report is None:
        return {
            "provided": False,
            "total": 0,
            "ok": 0,
            "failed": 0,
            "linked_documents": 0,
        }
    load_error = docs_report.get("load_error")
    if load_error:
        return {
            "provided": True,
            "total": 1,
            "ok": 0,
            "failed": 1,
            "linked_documents": 0,
            "load_error": load_error,
        }
    summary = docs_report.get("summary", {})
    return {
        "provided": True,
        "total": int(summary.get("total", 0)),
        "ok": int(summary.get("ok", 0)),
        "failed": int(summary.get("failed", 0)),
        "linked_documents": int(summary.get("linked_documents", 0)),
    }


def docs_report_source(docs_report: dict[str, Any] | None) -> str | None:
    if docs_report is None:
        return None
    return docs_report.get("path") or relpath(DEFAULT_DOCS_REPORT_PATH)


def resolve_docs_report_path(
    command: str, docs_report_path: Path | None
) -> Path | None:
    if docs_report_path is not None:
        return (
            docs_report_path
            if docs_report_path.is_absolute()
            else ROOT / docs_report_path
        )
    if command in {"extract", "update", "check"} and DEFAULT_DOCS_REPORT_PATH.exists():
        return DEFAULT_DOCS_REPORT_PATH
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("extract", "update", "check"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--output",
            type=Path,
            default=DEFAULT_OUTPUT_PATH,
            help="Generated support signals JSON path",
        )
        subparser.add_argument(
            "--docs-report",
            type=Path,
            default=None,
            help="Optional backend docs probe report with feature hits",
        )
        subparser.add_argument(
            "--pytest-failure-summary",
            type=Path,
            action="append",
            default=[],
            help="Optional pytest failure summary JSON from tools/pytest_failure_summary.py",
        )
    docs_parser = subparsers.add_parser(
        "docs", help="Fetch backend docs and extract feature/candidate terms"
    )
    docs_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DOCS_REPORT_PATH,
        help="Generated backend docs report JSON path",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    output = args.output if args.output.is_absolute() else ROOT / args.output
    backends_data = load_json(BACKENDS_PATH)
    features_data = load_json(FEATURES_PATH)

    if args.command == "docs":
        report = build_docs_report(
            backends_data,
            features_data,
            timeout=args.timeout,
            max_linked_pages=args.max_linked_pages,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(stable_json(report), encoding="utf-8")
        print(f"Wrote {relpath(output)}")
        print(
            "Docs: total={total}, ok={ok}, failed={failed}, linked={linked_documents}, feature_hits={feature_hits}, candidate_terms={candidate_terms}".format(
                **report["summary"]
            )
        )
        return 1 if args.strict and report["summary"]["failed"] else 0

    docs_report_path = resolve_docs_report_path(args.command, args.docs_report)
    pytest_failure_paths = []
    for path in args.pytest_failure_summary:
        pytest_failure_paths.append(path if path.is_absolute() else ROOT / path)

    report = build_report(
        backends_data,
        features_data,
        docs_report=load_docs_report(docs_report_path),
        pytest_failure_reports=[
            load_pytest_failure_report(path) for path in pytest_failure_paths
        ],
    )
    expected = stable_json(report)

    if args.command in {"extract", "update"}:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(expected, encoding="utf-8")
        print(f"Wrote {relpath(output)}")
        return 0

    diff = compare_file(output, expected)
    if diff:
        print("Generated support signals artifact is stale.", file=sys.stderr)
        print("Run: python tools/support_signals.py update", file=sys.stderr)
        for line in diff[:120]:
            print(line, file=sys.stderr)
        if len(diff) > 120:
            print("... diff truncated ...", file=sys.stderr)
        return 1
    print("Support signals artifact is current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
