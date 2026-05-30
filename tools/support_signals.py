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
from collections import Counter
import difflib
from functools import lru_cache
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib import error
from urllib import parse
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIR = ROOT / "support"
BACKENDS_PATH = SUPPORT_DIR / "backends.json"
FEATURES_PATH = SUPPORT_DIR / "features.json"
DEFAULT_OUTPUT_PATH = SUPPORT_DIR / "generated" / "support-signals.json"
DEFAULT_DOCS_REPORT_PATH = SUPPORT_DIR / "generated" / "backend-docs-report.json"
PYTEST_FAILURE_SUMMARY_GENERATOR = "tools/pytest_failure_summary.py"
FRONTEND_ID = "frontend"
FRONTEND_NAME = "Frontend / IR / Parser"

BACKLOG_STATUSES = {
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
    "descriptorset": "resources.bindings",
    "raytracing": "stage.ray_tracing",
    "svinstanceid": "io.stage_parameters",
    "svvertexid": "io.stage_parameters",
}
DOC_CANDIDATE_FEATURE_ALIAS_PATTERNS = (
    (re.compile(r"^gl[a-z0-9]+"), "io.stage_parameters"),
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
                    "error": "No PDF text extractor is available: {}".format(exc),
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
    haystack = "{} {}".format(parsed.path, parsed.query).lower()
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
            "error": "{}: {}".format(exc.__class__.__name__, exc),
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
    if support.get("status") in BACKLOG_STATUSES:
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
    if status in BACKLOG_STATUSES:
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
                    "key": "extracted:{}:{}:{}".format(backend_id, feature_id, kind),
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


def load_pytest_failure_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": relpath(path),
            "load_error": {
                "type": "FileNotFoundError",
                "message": "pytest failure summary does not exist",
            },
        }
    try:
        report = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": relpath(path),
            "load_error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    if report.get("generator") != PYTEST_FAILURE_SUMMARY_GENERATOR:
        return {
            "path": relpath(path),
            "load_error": {
                "type": "UnexpectedGenerator",
                "message": "expected {}, got {}".format(
                    PYTEST_FAILURE_SUMMARY_GENERATOR,
                    report.get("generator"),
                ),
            },
        }
    report.setdefault("path", relpath(path))
    return report


def summarize_pytest_failure_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = [
        failure
        for report in reports
        if not report.get("load_error")
        for failure in report.get("failures", [])
    ]
    category_counts = Counter(
        failure.get("category", "unknown") for failure in failures
    )
    backend_counts = Counter(failure.get("backend", "unknown") for failure in failures)
    return {
        "provided": bool(reports),
        "report_count": len(reports),
        "load_error_count": sum(1 for report in reports if report.get("load_error")),
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
        feature_id = "ci.pytest.{}".format(category_key)
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
            implementation_hits = collect_file_hits(
                backend_implementation_paths(backend), terms
            )
            test_hits = collect_test_hits(backend.get("tests", []), terms)
            unsupported_hits = collect_unsupported_hits(
                backend_implementation_paths(backend), terms
            )
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
    docs_summary = (docs_report or {}).get("summary", {})
    pytest_failure_summary = summarize_pytest_failure_reports(pytest_failure_reports)
    return {
        "schema_version": 1,
        "generator": "tools/support_signals.py",
        "source": {
            "backends": "support/backends.json",
            "features": "support/features.json",
            "docs_report": (
                relpath(DEFAULT_DOCS_REPORT_PATH) if docs_report is not None else None
            ),
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
            "docs_probe": {
                "provided": docs_report is not None,
                "total": int(docs_summary.get("total", 0)),
                "ok": int(docs_summary.get("ok", 0)),
                "failed": int(docs_summary.get("failed", 0)),
                "linked_documents": int(docs_summary.get("linked_documents", 0)),
            },
        },
        "backends": [
            {
                "id": backend["id"],
                "name": backend["name"],
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


def load_docs_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return load_json(path)


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
        print("Wrote {}".format(relpath(output)))
        print(
            "Docs: total={total}, ok={ok}, failed={failed}, linked={linked_documents}, feature_hits={feature_hits}, candidate_terms={candidate_terms}".format(
                **report["summary"]
            )
        )
        return 1 if args.strict and report["summary"]["failed"] else 0

    docs_report_path = args.docs_report
    if docs_report_path is not None and not docs_report_path.is_absolute():
        docs_report_path = ROOT / docs_report_path
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
        print("Wrote {}".format(relpath(output)))
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
