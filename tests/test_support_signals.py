import importlib.util
import json
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


def test_build_report_counts_test_class_names_as_evidence():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "hip",
                "name": "HIP",
                "translator_codegen": "crosstl/backend/HIP/HipLexer.py",
                "tests": ["tests/test_backend/test_HIP/test_lexer.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "source.lexing",
                "category": "source",
                "name": "Native lexer coverage",
                "description": (
                    "Tokenize backend source language constructs used by the "
                    "native parser."
                ),
                "support": {"hip": {"status": "supported", "evidence": ["test"]}},
            }
        ]
    }

    report = module.build_report(backends, features)
    support = report["features"][0]["support"]["hip"]
    test_symbols = {hit["symbol"] for hit in support["tests"]}

    assert support["state"] == "tested"
    assert "TestHipLexer" in test_symbols
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


def test_build_report_skips_reviewed_unsupported_rows_with_unsupported_markers():
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
                "id": "catalog.unsupported",
                "category": "validation",
                "name": "Unsupported catalog issue",
                "description": (
                    "Review unsupported catalog rows with detected "
                    "implementation tests."
                ),
                "support": {
                    "directx": {
                        "status": "unsupported",
                        "evidence": ["tools/support_signals.py::unsupported marker"],
                    }
                },
            }
        ]
    }

    report = module.build_report(backends, features)
    support = report["features"][0]["support"]["directx"]

    assert support["state"] == "tested"
    assert support["unsupported"]
    assert report["issues"] == []


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


def test_build_report_maps_documented_semantic_candidates_to_stage_io_feature():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "slang",
                "name": "Slang",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "io.stage_parameters",
                "category": "stage I/O",
                "name": "Stage parameter semantics",
                "description": (
                    "Input parameter semantics and target builtin attributes."
                ),
                "support": {"slang": {"status": "partial"}},
            }
        ]
    }
    docs_report = {
        "documents": [
            {
                "backend_id": "slang",
                "backend": "Slang",
                "source": "Slang user guide",
                "url": "https://example.com/slang",
                "ok": True,
                "candidate_terms": [
                    {"term": "SV_VertexID", "count": 1},
                    {"term": "SV_InstanceID", "count": 1},
                ],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert not any(issue["feature"] == "SV_VertexID" for issue in report["issues"])
    assert not any(issue["feature"] == "SV_InstanceID" for issue in report["issues"])


def test_build_report_ignores_non_surface_documented_noise_candidates():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "slang",
                "name": "Slang",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {"features": []}
    docs_report = {
        "documents": [
            {
                "backend_id": "slang",
                "backend": "Slang",
                "source": "Slang user guide",
                "url": "https://example.com/slang",
                "ok": True,
                "candidate_terms": [{"term": "DescriptorHandle", "count": 1}],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert report["issues"] == []


def test_build_report_creates_ci_failure_issues_from_pytest_summaries():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "opengl",
                "name": "OpenGL / GLSL",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {"features": []}
    failure_report = {
        "schema_version": 1,
        "generator": "tools/pytest_failure_summary.py",
        "path": "support/generated/full-tests-failure-summary.json",
        "failures": [
            {
                "nodeid": (
                    "tests.test_translator.test_codegen."
                    "test_external_shader_validators::test_generated_glsl_validates"
                ),
                "file": (
                    "tests/test_translator/test_codegen/"
                    "test_external_shader_validators.py"
                ),
                "kind": "failure",
                "category": "backend_compiler_validation",
                "backend": "opengl",
                "message": "glslangValidator rejected generated GLSL",
            },
            {
                "nodeid": "tests.test_support_matrix::test_matrix_check",
                "file": "tests/test_support_matrix.py",
                "kind": "failure",
                "category": "support_automation",
                "backend": "unknown",
                "message": "support matrix check failed",
            },
        ],
    }

    report = module.build_report(
        backends,
        features,
        pytest_failure_reports=[failure_report],
    )

    assert report["summary"]["pytest_failures"] == {
        "provided": True,
        "report_count": 1,
        "load_error_count": 0,
        "failed_testcase_count": 2,
        "categories": {
            "backend_compiler_validation": 1,
            "support_automation": 1,
        },
        "backends": {"opengl": 1, "unknown": 1},
    }
    assert {
        issue["key"]
        for issue in report["issues"]
        if issue["kind"] == "pytest_failure_summary"
    } == {
        (
            "extracted:opengl:ci.pytest.backend-compiler-validation:"
            "pytest_failure_summary"
        ),
        "extracted:frontend:ci.pytest.support-automation:pytest_failure_summary",
    }
    opengl_issue = next(
        issue for issue in report["issues"] if issue["backend_id"] == "opengl"
    )
    assert opengl_issue["signal"]["failure_count"] == 1
    assert (
        "glslangValidator rejected generated GLSL"
        in opengl_issue["signal"]["failures"][0]["message"]
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


def test_build_report_creates_issues_from_pytest_failure_summary():
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
    failure_report = {
        "path": "support/generated/full-tests-failure-summary.json",
        "generator": "tools/pytest_failure_summary.py",
        "failures": [
            {
                "nodeid": (
                    "tests.test_translator.test_codegen.test_directx_codegen::test_wave"
                ),
                "file": "tests/test_translator/test_codegen/test_directx_codegen.py",
                "kind": "failure",
                "category": "backend_codegen",
                "backend": "directx",
                "message": "assert generated HLSL contains WaveActiveSum",
            },
            {
                "nodeid": "tests.test_translator.test_parser::test_parse",
                "file": "tests/test_translator/test_parser.py",
                "kind": "failure",
                "category": "frontend_ir",
                "backend": "unknown",
                "message": "parser failed",
            },
        ],
    }

    report = module.build_report(
        backends,
        features,
        pytest_failure_reports=[failure_report],
    )

    assert report["summary"]["pytest_failures"] == {
        "provided": True,
        "report_count": 1,
        "load_error_count": 0,
        "failed_testcase_count": 2,
        "categories": {"backend_codegen": 1, "frontend_ir": 1},
        "backends": {"directx": 1, "unknown": 1},
    }
    assert report["source"]["pytest_failure_summaries"] == [
        "support/generated/full-tests-failure-summary.json"
    ]
    failure_issues = [
        issue for issue in report["issues"] if issue["kind"] == "pytest_failure_summary"
    ]
    assert [issue["backend_id"] for issue in failure_issues] == [
        "directx",
        "frontend",
    ]
    assert failure_issues[0]["feature_id"] == "ci.pytest.backend-codegen"
    assert failure_issues[0]["signal"]["failure_count"] == 1
    assert failure_issues[1]["backend"] == "Frontend / IR / Parser"
    assert failure_issues[1]["signal"]["failures"][0]["message"] == "parser failed"


def test_load_pytest_failure_report_validates_generator_and_missing_files(tmp_path):
    module = load_signals_module()
    missing = tmp_path / "missing.json"
    wrong_generator = tmp_path / "wrong-generator.json"
    wrong_generator.write_text(
        json.dumps({"generator": "tools/other.py", "failures": []}),
        encoding="utf-8",
    )

    missing_report = module.load_pytest_failure_report(missing)
    wrong_report = module.load_pytest_failure_report(wrong_generator)

    assert missing_report["load_error"]["type"] == "FileNotFoundError"
    assert wrong_report["load_error"] == {
        "type": "UnexpectedGenerator",
        "message": "expected tools/pytest_failure_summary.py, got tools/other.py",
    }
