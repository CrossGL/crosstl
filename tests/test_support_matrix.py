import json
import subprocess
import sys
from pathlib import Path

import crosstl.translator.codegen as codegen
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

ROOT = Path(__file__).resolve().parents[1]


def test_support_matrix_generated_artifacts_are_current():
    result = subprocess.run(
        [sys.executable, "tools/support_matrix.py", "check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "generated artifacts are current" in result.stdout


def test_support_matrix_covers_all_cataloged_backends():
    matrix_path = ROOT / "support" / "generated" / "support-matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    backend_ids = {backend["id"] for backend in matrix["backends"]}
    assert backend_ids == {
        "directx",
        "opengl",
        "metal",
        "vulkan",
        "cuda",
        "hip",
        "mojo",
        "rust",
        "slang",
    }

    assert matrix["summary"]["feature_count"] == len(matrix["features"])
    assert matrix["summary"]["backend_count"] == len(backend_ids)

    statuses = set(matrix["status_codes"])
    for backend_id in backend_ids:
        counts = matrix["summary"]["status_counts"][backend_id]
        assert set(counts) == statuses
        assert sum(counts.values()) == matrix["summary"]["feature_count"]


def test_support_backend_catalog_matches_codegen_registry():
    backends_path = ROOT / "support" / "backends.json"
    catalog = json.loads(backends_path.read_text(encoding="utf-8"))
    register_default_sources()

    backend_ids = {backend["id"] for backend in catalog["backends"]}
    assert backend_ids == set(codegen.backend_names())
    assert backend_ids.issubset(set(SOURCE_REGISTRY.names()))

    for backend in catalog["backends"]:
        backend_id = backend["id"]
        spec = codegen.get_backend(backend_id)
        assert spec is not None, f"{backend_id} is not registered"
        assert codegen.get_backend_extension(backend_id) == backend["target_extension"]
        assert (
            SOURCE_REGISTRY.get_by_extension(backend["target_extension"]).name
            == backend_id
        )

        for alias in backend.get("aliases", []):
            assert (
                codegen.normalize_backend_name(alias) == backend_id
            ), f"{backend_id} support alias is not accepted by codegen: {alias}"
            assert (
                SOURCE_REGISTRY.get(alias).name == backend_id
            ), f"{backend_id} support alias is not accepted by source registry: {alias}"


def test_graphics_backend_roadmap_is_focused_on_primary_graphics_targets():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    assert roadmap["view"]["backend_ids"] == ["directx", "opengl", "metal"]
    assert roadmap["summary"]["feature_count"] == len(roadmap["features"])
    assert roadmap["summary"]["backend_count"] == 3
    assert roadmap["summary"]["backlog"]["backlog_count"] == len(roadmap["backlog"])

    for item in roadmap["backlog"]:
        assert item["backend_id"] in {"directx", "opengl", "metal"}


def test_graphics_texture_query_row_is_supported_with_evidence():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    texture_query = next(
        feature for feature in roadmap["features"] if feature["id"] == "texture.query"
    )

    for backend_id in ("directx", "opengl", "metal"):
        support = texture_query["support"][backend_id]
        assert support["status"] == "supported"
        assert support["evidence"]

    assert all(item["feature_id"] != "texture.query" for item in roadmap["backlog"])


def test_graphics_texel_fetch_row_is_supported_with_evidence():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    texel_fetch = next(
        feature
        for feature in roadmap["features"]
        if feature["id"] == "texture.texel_fetch"
    )

    for backend_id in ("directx", "opengl", "metal"):
        support = texel_fetch["support"][backend_id]
        assert support["status"] == "supported"
        assert support["evidence"]

    assert all(
        item["feature_id"] != "texture.texel_fetch" for item in roadmap["backlog"]
    )


def test_graphics_match_row_is_supported_with_evidence():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    match_lowering = next(
        feature for feature in roadmap["features"] if feature["id"] == "language.match"
    )

    for backend_id in ("directx", "opengl", "metal"):
        support = match_lowering["support"][backend_id]
        assert support["status"] == "supported"
        assert support["evidence"]

    assert all(item["feature_id"] != "language.match" for item in roadmap["backlog"])


def test_support_matrix_audit_writes_filtered_json(tmp_path):
    output_path = tmp_path / "graphics-partial.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/support_matrix.py",
            "audit",
            "--backend",
            "directx,opengl,metal",
            "--status",
            "partial",
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Wrote" in result.stdout
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["filters"] == {
        "backend_ids": ["directx", "opengl", "metal"],
        "categories": [],
        "statuses": ["partial"],
    }
    assert report["summary"]["backlog_count"] == len(report["backlog"])
    assert report["backlog"]
    for item in report["backlog"]:
        assert item["backend_id"] in {"directx", "opengl", "metal"}
        assert item["status"] == "partial"
