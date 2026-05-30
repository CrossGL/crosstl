import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "support_signals.py"


def load_signals_module():
    spec = importlib.util.spec_from_file_location("support_signals", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_document_feature_hits_extracts_feature_matches_without_storing_text():
    module = load_signals_module()
    feature = {
        "id": "texture.gather",
        "category": "textures",
        "name": "Texture gather operations",
        "description": (
            "Lower gather, gather-offset, and gather-offsets texture operations."
        ),
    }

    hits = module.document_feature_hits(
        "Texture2D supports gather and gather offset sampling.", [feature]
    )

    assert hits == [
        {
            "feature_id": "texture.gather",
            "category": "textures",
            "name": "Texture gather operations",
            "matched_terms": ["gather", "offset"],
            "score": 2,
        }
    ]


def test_document_candidate_terms_extracts_backend_api_identifiers():
    module = load_signals_module()

    candidates = module.document_candidate_terms(
        "Texture2D RWStructuredBuffer OpImageFetch gl_WorkGroupID SV_Position"
    )
    terms = {item["term"] for item in candidates}

    assert {
        "Texture2D",
        "RWStructuredBuffer",
        "OpImageFetch",
        "gl_WorkGroupID",
        "SV_Position",
    }.issubset(terms)


def test_extract_document_text_uses_html_body_text_and_collects_links():
    module = load_signals_module()

    text, metadata = module.extract_document_text(
        b"""
        <html>
          <head><meta property="og:image" content="summary_large_image"></head>
          <body>
            <a href="/hlsl/sv-position">SV docs</a>
            <script>Texture2DShouldNotAppear</script>
            Texture2D SV_Position
          </body>
        </html>
        """,
        "text/html",
        "https://example.com/hlsl/index.html",
    )

    assert metadata["kind"] == "html"
    assert "Texture2D SV_Position" in text
    assert "summary_large_image" not in text
    assert "Texture2DShouldNotAppear" not in text
    assert metadata["links"] == ["https://example.com/hlsl/sv-position"]


def test_build_docs_report_crawls_relevant_same_site_links(monkeypatch):
    module = load_signals_module()

    def fake_fetch_url(url, timeout):
        if url.endswith("index.html"):
            return {
                "ok": True,
                "status": 200,
                "url": url,
                "final_url": url,
                "content_type": "text/html",
                "content_length": 1,
                "sha256": "root",
                "elapsed_ms": 1,
                "text": "Texture2D",
                "text_extraction": {
                    "kind": "html",
                    "links": [
                        "https://example.com/hlsl/sv-position.html",
                        "https://example.com/assets/site.css",
                    ],
                    "text_length": 9,
                },
            }
        return {
            "ok": True,
            "status": 200,
            "url": url,
            "final_url": url,
            "content_type": "text/html",
            "content_length": 1,
            "sha256": "child",
            "elapsed_ms": 1,
            "text": "SV_Position",
            "text_extraction": {"kind": "html", "links": [], "text_length": 11},
        }

    monkeypatch.setattr(module, "fetch_url", fake_fetch_url)
    backends = {
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "aliases": ["hlsl"],
                "docs": [
                    {
                        "name": "HLSL reference",
                        "url": "https://example.com/hlsl/index.html",
                    }
                ],
            }
        ]
    }
    features = {"features": []}

    report = module.build_docs_report(
        backends,
        features,
        timeout=1,
        max_linked_pages=2,
    )

    assert report["summary"]["total"] == 2
    assert report["summary"]["linked_documents"] == 1
    assert (
        report["documents"][1]["linked_from"] == "https://example.com/hlsl/index.html"
    )


def test_build_report_scans_repo_implementation_and_tests():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "support.signals",
                "category": "validation",
                "name": "Support signals",
                "description": "Extract generated support signal tests.",
                "support": {"directx": {"status": "partial"}},
            }
        ]
    }

    report = module.build_report(backends, features)
    support = report["features"][0]["support"]["directx"]

    assert support["state"] == "tested"
    assert support["implementation"]
    assert support["tests"]
    assert report["issues"] == []


def test_build_report_flags_unsupported_catalog_rows_with_detected_tests():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "docs.candidate_terms",
                "category": "validation",
                "name": "Document candidate terms",
                "description": (
                    "Extract document candidate terms from backend API identifiers."
                ),
                "support": {"directx": {"status": "unsupported"}},
            }
        ]
    }

    report = module.build_report(backends, features)
    support = report["features"][0]["support"]["directx"]

    assert support["state"] == "tested"
    assert report["issues"] == [
        {
            "key": (
                "extracted:directx:docs.candidate_terms:"
                "catalog_unsupported_with_detected_tests"
            ),
            "kind": "catalog_unsupported_with_detected_tests",
            "title": (
                "Review unsupported catalog row with detected implementation/tests"
            ),
            "backend_id": "directx",
            "backend": "DirectX / HLSL",
            "feature_id": "docs.candidate_terms",
            "feature": "Document candidate terms",
            "category": "validation",
            "status": "unsupported",
            "state": "tested",
            "matched_terms": [
                "api",
                "candidate",
                "document",
                "extract",
                "identifier",
                "identifiers",
                "term",
                "terms",
            ],
        }
    ]


def test_build_report_does_not_flag_weak_unsupported_catalog_matches():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "support.signals",
                "category": "validation",
                "name": "Support signals",
                "description": "Extract generated support signal tests.",
                "support": {"directx": {"status": "unsupported"}},
            }
        ]
    }

    report = module.build_report(backends, features)

    assert report["features"][0]["support"]["directx"]["state"] == "tested"
    assert report["issues"] == []


def test_build_report_creates_issues_for_unmapped_documented_candidates():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "texture.gather",
                "category": "textures",
                "name": "Texture gather",
                "description": "Gather texture samples.",
                "support": {"directx": {"status": "supported"}},
            }
        ]
    }
    docs_report = {
        "summary": {
            "total": 1,
            "ok": 1,
            "failed": 0,
            "linked_documents": 0,
        },
        "documents": [
            {
                "backend_id": "directx",
                "backend": "DirectX / HLSL",
                "source": "HLSL reference",
                "url": "https://example.com/hlsl",
                "ok": True,
                "candidate_terms": [{"term": "SV_Position", "count": 3}],
            }
        ],
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert report["summary"]["docs_probe"] == {
        "provided": True,
        "total": 1,
        "ok": 1,
        "failed": 0,
        "linked_documents": 0,
    }
    assert any(
        issue["kind"] == "documented_candidate_not_detected"
        and issue["feature"] == "SV_Position"
        and issue["category"] == "docs"
        for issue in report["issues"]
    )


def test_build_report_skips_documented_candidates_mapped_to_existing_features():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "resources.byte_address_buffers",
                "category": "resources",
                "name": "Byte address buffers",
                "description": "Lower ByteAddressBuffer operations.",
                "support": {"directx": {"status": "supported"}},
            }
        ]
    }
    docs_report = {
        "documents": [
            {
                "backend_id": "directx",
                "backend": "DirectX / HLSL",
                "source": "HLSL reference",
                "url": "https://example.com/hlsl",
                "ok": True,
                "candidate_terms": [{"term": "ByteAddressBuffer", "count": 5}],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert not any(
        issue["feature"] == "ByteAddressBuffer" for issue in report["issues"]
    )


def test_build_report_records_missing_docs_probe_health():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {"features": []}

    report = module.build_report(backends, features)

    assert report["summary"]["docs_probe"] == {
        "provided": False,
        "total": 0,
        "ok": 0,
        "failed": 0,
        "linked_documents": 0,
    }
