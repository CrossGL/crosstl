import json
import subprocess
import sys
from pathlib import Path

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


def test_graphics_backend_roadmap_is_focused_on_primary_graphics_targets():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    assert roadmap["view"]["backend_ids"] == ["directx", "opengl", "metal"]
    assert roadmap["summary"]["feature_count"] == len(roadmap["features"])
    assert roadmap["summary"]["backend_count"] == 3
    assert roadmap["summary"]["backlog"]["backlog_count"] == len(roadmap["backlog"])

    for item in roadmap["backlog"]:
        assert item["backend_id"] in {"directx", "opengl", "metal"}


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
