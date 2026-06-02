import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "support_signals.py"


def load_signals_module():
    spec = importlib.util.spec_from_file_location("support_signals", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_docs_report():
    return {
        "schema_version": 1,
        "generator": "tools/support_signals.py docs",
        "source": {
            "backends": "support/backends.json",
            "features": "support/features.json",
        },
        "summary": {
            "total": 1,
            "ok": 1,
            "failed": 0,
            "feature_hits": 1,
            "candidate_terms": 1,
            "linked_documents": 0,
        },
        "documents": [
            {
                "backend_id": "directx",
                "backend": "DirectX / HLSL",
                "source": "HLSL reference",
                "url": "https://example.com/hlsl",
                "final_url": "https://example.com/hlsl",
                "content_type": "text/html",
                "content_length": 64,
                "sha256": "abc123",
                "elapsed_ms": 12,
                "ok": True,
                "status": 200,
                "text_extraction": {
                    "kind": "html",
                    "links": [],
                    "text_length": 32,
                },
                "feature_hits": [
                    {
                        "feature_id": "texture.gather",
                        "category": "textures",
                        "name": "Texture gather",
                        "matched_terms": ["gather"],
                        "score": 1,
                    }
                ],
                "candidate_terms": [
                    {
                        "term": "Texture2D",
                        "count": 1,
                    }
                ],
            }
        ],
    }


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


def test_load_docs_report_accepts_valid_contract(tmp_path):
    module = load_signals_module()
    docs_path = tmp_path / "backend-docs-report.json"
    docs_path.write_text(json.dumps(valid_docs_report()), encoding="utf-8")

    report = module.load_docs_report(docs_path)

    assert "load_error" not in report
    assert report["path"] == module.relpath(docs_path)


def test_load_docs_report_reports_malformed_json(tmp_path):
    module = load_signals_module()
    docs_path = tmp_path / "backend-docs-report.json"
    docs_path.write_text("{not json", encoding="utf-8")

    report = module.load_docs_report(docs_path)

    assert report["load_error"]["path"] == module.relpath(docs_path)
    assert report["load_error"]["type"] == "JSONDecodeError"
    assert "Expecting property name" in report["load_error"]["message"]


def test_load_docs_report_reports_missing_required_fields(tmp_path):
    module = load_signals_module()
    docs_path = tmp_path / "backend-docs-report.json"
    docs_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "tools/support_signals.py docs",
            }
        ),
        encoding="utf-8",
    )

    report = module.load_docs_report(docs_path)

    assert report["load_error"] == {
        "path": module.relpath(docs_path),
        "type": "MissingReportFields",
        "message": "missing required fields: summary, documents",
    }


def test_load_docs_report_reports_invalid_document_candidate_type(tmp_path):
    module = load_signals_module()
    docs_path = tmp_path / "backend-docs-report.json"
    docs_report = valid_docs_report()
    docs_report["documents"][0]["candidate_terms"][0]["count"] = "one"
    docs_path.write_text(json.dumps(docs_report), encoding="utf-8")

    report = module.load_docs_report(docs_path)

    assert report["load_error"] == {
        "path": module.relpath(docs_path),
        "type": "InvalidReportField",
        "message": "documents[0].candidate_terms[0].count must be int, got str",
    }


def test_load_docs_report_reports_summary_mismatch(tmp_path):
    module = load_signals_module()
    docs_path = tmp_path / "backend-docs-report.json"
    docs_report = valid_docs_report()
    docs_report["summary"]["candidate_terms"] = 2
    docs_path.write_text(json.dumps(docs_report), encoding="utf-8")

    report = module.load_docs_report(docs_path)

    assert report["load_error"] == {
        "path": module.relpath(docs_path),
        "type": "InvalidReportField",
        "message": "summary.candidate_terms must match documents: 2 != 1",
    }


def test_build_report_summarizes_docs_probe_load_error():
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
    docs_report = {
        "path": "support/generated/backend-docs-report.json",
        "load_error": {
            "path": "support/generated/backend-docs-report.json",
            "type": "JSONDecodeError",
            "message": "Expecting property name",
        },
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert (
        report["source"]["docs_report"] == "support/generated/backend-docs-report.json"
    )
    assert report["summary"]["docs_probe"] == {
        "provided": True,
        "total": 1,
        "ok": 0,
        "failed": 1,
        "linked_documents": 0,
        "load_error": {
            "path": "support/generated/backend-docs-report.json",
            "type": "JSONDecodeError",
            "message": "Expecting property name",
        },
    }
    assert report["issues"] == []


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
                "candidate_terms": [{"term": "PipelineMagicState", "count": 3}],
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
        and issue["feature"] == "PipelineMagicState"
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


def test_build_report_maps_directx_surface_candidates_and_skips_spirv_noise():
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
                "id": "io.stage_parameters",
                "category": "stage I/O",
                "name": "Stage parameter semantics",
                "description": "Input parameter semantics.",
                "support": {"directx": {"status": "partial"}},
            },
            {
                "id": "resources.structured_buffers",
                "category": "resources",
                "name": "Structured/storage buffers",
                "description": "Structured and RW buffer resources.",
                "support": {"directx": {"status": "partial"}},
            },
            {
                "id": "resources.texture_sampler_split",
                "category": "resources",
                "name": "Texture and sampler object model",
                "description": "Texture and sampler resources.",
                "support": {"directx": {"status": "partial"}},
            },
            {
                "id": "texture.sampling",
                "category": "textures",
                "name": "Texture sampling",
                "description": "Sample texture resources.",
                "support": {"directx": {"status": "partial"}},
            },
            {
                "id": "stage.ray_tracing",
                "category": "stages",
                "name": "Ray tracing stages",
                "description": "Ray tracing shader stages.",
                "support": {"directx": {"status": "partial"}},
            },
        ]
    }
    docs_report = {
        "documents": [
            {
                "backend_id": "directx",
                "backend": "DirectX / HLSL",
                "source": "HLSL docs",
                "url": "https://example.com/hlsl",
                "ok": True,
                "candidate_terms": [
                    {"term": "SV_Target0", "count": 1},
                    {"term": "RWBuffer", "count": 1},
                    {"term": "sampler2D", "count": 1},
                    {"term": "Texture2D", "count": 1},
                    {"term": "Raytracing", "count": 1},
                    {"term": "Atomically", "count": 1},
                    {"term": "OpDecorate", "count": 1},
                ],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert report["issues"] == []


def test_build_report_maps_opengl_builtin_candidates_to_catalog_features():
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
    features = {
        "features": [
            {
                "id": "io.stage_parameters",
                "category": "stage I/O",
                "name": "Stage parameter semantics",
                "description": "Input parameter semantics.",
                "support": {"opengl": {"status": "partial"}},
            },
            {
                "id": "resources.bindings",
                "category": "resources",
                "name": "Explicit and automatic resource bindings",
                "description": "Descriptor set and binding metadata.",
                "support": {"opengl": {"status": "partial"}},
            },
        ]
    }
    docs_report = {
        "documents": [
            {
                "backend_id": "opengl",
                "backend": "OpenGL / GLSL",
                "source": "GLSL docs",
                "url": "https://example.com/glsl",
                "ok": True,
                "candidate_terms": [
                    {"term": "gl_Position", "count": 1},
                    {"term": "gl_WorkGroupSize", "count": 1},
                    {"term": "gl_FragCoord", "count": 1},
                    {"term": "Descriptors", "count": 1},
                    {"term": "DescriptorSet", "count": 1},
                ],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    descriptor_match = module.candidate_feature_matches(
        "Descriptors", features["features"]
    )
    assert descriptor_match[0]["feature_id"] == "resources.bindings"
    assert report["issues"] == []


def test_build_report_keeps_spirv_op_candidates_for_vulkan():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "vulkan",
                "name": "Vulkan SPIR-V",
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
                "backend_id": "vulkan",
                "backend": "Vulkan SPIR-V",
                "source": "SPIR-V spec",
                "url": "https://example.com/spirv",
                "ok": True,
                "candidate_terms": [{"term": "OpDecorate", "count": 1}],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert any(issue["feature"] == "OpDecorate" for issue in report["issues"])


def test_build_report_maps_vulkan_spirv_opcode_candidates_to_catalog_features():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "vulkan",
                "name": "Vulkan SPIR-V",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "language.control_flow",
                "category": "language",
                "name": "Control flow",
                "description": "Branch, loop, and selection control flow.",
                "support": {"vulkan": {"status": "partial"}},
            },
            {
                "id": "language.arrays",
                "category": "language",
                "name": "Array declarations and access",
                "description": "Array declarations and access.",
                "support": {"vulkan": {"status": "partial"}},
            },
            {
                "id": "language.functions",
                "category": "language",
                "name": "Function declarations and calls",
                "description": "Function calls and parameters.",
                "support": {"vulkan": {"status": "partial"}},
            },
            {
                "id": "language.vector_matrix",
                "category": "language",
                "name": "Vector and matrix expressions",
                "description": "Scalar, vector, and matrix expressions.",
                "support": {"vulkan": {"status": "partial"}},
            },
            {
                "id": "language.structs",
                "category": "language",
                "name": "Struct declarations and construction",
                "description": "Struct types and members.",
                "support": {"vulkan": {"status": "partial"}},
            },
            {
                "id": "resources.bindings",
                "category": "resources",
                "name": "Explicit and automatic resource bindings",
                "description": "SPIR-V decorations and bindings.",
                "support": {"vulkan": {"status": "partial"}},
            },
            {
                "id": "resources.memory_qualifiers",
                "category": "resources",
                "name": "Resource memory qualifiers",
                "description": "Pointer, load, and store memory semantics.",
                "support": {"vulkan": {"status": "partial"}},
            },
            {
                "id": "resources.structured_buffers",
                "category": "resources",
                "name": "Structured/storage buffers",
                "description": "Structured buffer resources.",
                "support": {"vulkan": {"status": "partial"}},
            },
            {
                "id": "target.codegen",
                "category": "target",
                "name": "CrossGL to target code generation",
                "description": "Target code generation helpers.",
                "support": {"vulkan": {"status": "partial"}},
            },
        ]
    }
    docs_report = {
        "documents": [
            {
                "backend_id": "vulkan",
                "backend": "Vulkan SPIR-V",
                "source": "SPIR-V spec",
                "url": "https://example.com/spirv",
                "ok": True,
                "candidate_terms": [
                    {"term": "OpBranchConditional", "count": 1},
                    {"term": "OpTypeArray", "count": 1},
                    {"term": "OpFunctionCall", "count": 1},
                    {"term": "OpTypeFloat", "count": 1},
                    {"term": "OpTypeStruct", "count": 1},
                    {"term": "OpDecorate", "count": 1},
                    {"term": "OpTypePointer", "count": 1},
                    {"term": "OpExtInstImport", "count": 1},
                    {"term": "DescriptorHeapEXT", "count": 1},
                    {"term": "Structured", "count": 1},
                    {"term": "OpCapability", "count": 1},
                    {"term": "OpExtension", "count": 1},
                    {"term": "OpMemoryModel", "count": 1},
                    {"term": "OpNop", "count": 1},
                    {"term": "OpSourceContinued", "count": 1},
                    {"term": "OpSourceExtension", "count": 1},
                    {"term": "OpTypeOpaque", "count": 1},
                    {"term": "OpLine", "count": 1},
                    {"term": "OpName", "count": 1},
                    {"term": "OpTypePipe", "count": 1},
                    {"term": "PipelineEnableALTERA", "count": 1},
                    {"term": "PipelineEnableINTEL", "count": 1},
                ],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert report["issues"] == []


def test_build_report_ignores_cuda_runtime_doc_candidates():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "cuda",
                "name": "CUDA",
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
                "backend_id": "cuda",
                "backend": "CUDA",
                "source": "CUDA docs",
                "url": "https://example.com/cuda",
                "ok": True,
                "candidate_terms": [
                    {"term": "cudaMalloc", "count": 1},
                    {"term": "cudaMemcpyAsync", "count": 1},
                    {"term": "cuda_runtime_api", "count": 1},
                    {"term": "Pipelines", "count": 1},
                    {"term": "vkExt", "count": 1},
                ],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert report["issues"] == []


def test_build_report_maps_hip_texture_resource_candidates_to_catalog_features():
    module = load_signals_module()
    backends = {
        "backends": [
            {
                "id": "hip",
                "name": "HIP",
                "translator_codegen": "tools/support_signals.py",
                "native_backend": "tools",
                "tests": ["tests/test_support_signals.py"],
            }
        ]
    }
    features = {
        "features": [
            {
                "id": "resources.texture_sampler_split",
                "category": "resources",
                "name": "Texture and sampler object model",
                "description": (
                    "Represent combined and separate texture/sampler models."
                ),
                "support": {"hip": {"status": "partial"}},
            },
            {
                "id": "resources.resource_arrays",
                "category": "resources",
                "name": "Resource arrays",
                "description": "Fixed and unsized texture and sampler arrays.",
                "support": {"hip": {"status": "partial"}},
            },
        ]
    }
    docs_report = {
        "documents": [
            {
                "backend_id": "hip",
                "backend": "HIP",
                "source": "HIP docs",
                "url": "https://example.com/hip",
                "ok": True,
                "candidate_terms": [
                    {"term": "hipTexRefSetMipmappedArray", "count": 1},
                    {"term": "hipTexObjectGetResourceDesc", "count": 1},
                    {"term": "hipBindTexture2D", "count": 1},
                    {"term": "hipGetChannelDesc", "count": 1},
                    {"term": "hipMipmappedArrayGetLevel", "count": 1},
                    {"term": "hipArray_t", "count": 1},
                ],
            }
        ]
    }

    report = module.build_report(backends, features, docs_report=docs_report)

    assert report["issues"] == []


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
                "candidate_terms": [
                    {"term": "DescriptorHandle", "count": 1},
                    {"term": "hipDeviceptr_t", "count": 1},
                    {"term": "hipSuccess", "count": 1},
                    {"term": "cudaErrorNoDevice", "count": 1},
                ],
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


def test_build_report_treats_clean_pytest_summary_as_authoritative_input():
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
    clean_report = {
        "path": (
            "support/generated/pytest-failures/clean-workflow/clean-workflow-failure-summary.json"
        ),
        "generator": "tools/pytest_failure_summary.py",
        "clean_workflow_runs": [
            {
                "workflow": "Complete Test Suite",
                "run_id": "123",
                "conclusion": "success",
                "head_sha": "abc",
            }
        ],
        "failures": [],
    }

    report = module.build_report(
        backends,
        features,
        pytest_failure_reports=[clean_report],
    )

    assert report["summary"]["pytest_failures"] == {
        "provided": True,
        "report_count": 1,
        "load_error_count": 0,
        "failed_testcase_count": 0,
        "categories": {},
        "backends": {},
    }
    assert report["source"]["pytest_failure_summaries"] == [
        (
            "support/generated/pytest-failures/clean-workflow/"
            "clean-workflow-failure-summary.json"
        )
    ]
    assert [
        issue for issue in report["issues"] if issue["kind"] == "pytest_failure_summary"
    ] == []


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
