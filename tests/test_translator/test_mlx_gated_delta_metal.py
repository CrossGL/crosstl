"""Native compilation of one unchanged pinned MLX backward entry."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from crosstl.project import ProjectConfig, translate_project
from demos.integrations.mlx.run_mlx_metal_host import MLX_COMMIT, run

SOURCE = "mlx/backend/metal/kernels/gated_delta_update_vjp.metal"
SOURCE_SHA256 = "59590c0bc3dbeb6ce1c808660b43abdf462a3f43f16abcb484220605b8aa86e7"
ENTRY = "seq_gated_delta_vjp_float_128_128_24_24_1"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_GATED_DELTA_METAL"


def _translate_pinned(
    tmp_path,
    target,
    *,
    entry=ENTRY,
    workgroup_size=(32, 1, 1),
    target_options=None,
    source_path=SOURCE,
    source_sha256=SOURCE_SHA256,
    index_range_assertions=(),
):
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    assert root_value, "Set CROSTL_MLX_CURRENT_ROOT to the pinned checkout"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    source = root / source_path
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-gated-delta-", dir=root
    ) as output:
        report = translate_project(
            ProjectConfig(
                root=root,
                include_patterns=(source_path,),
                include_dirs=(".",),
                targets=(target,),
                entry_points={source_path: (entry,)},
                entry_workgroup_size_rules={source_path: {entry: workgroup_size}},
                index_range_assertions=index_range_assertions,
                source_options={
                    "metal": {
                        "max_template_specializations": 128,
                        "max_template_materialization_work": 8192,
                        "target_options": {target: target_options or {}},
                    }
                },
                output_dir=Path(output).relative_to(root),
            ),
            format_output=False,
        )
        report.write_json(tmp_path / "report.json")
        payload = report.to_json()
        shutil.copytree(output, tmp_path / "translated")
        assert payload["diagnostics"] == []
        assert payload["summary"]["translatedCount"] == 1
        if index_range_assertions:
            assert payload["project"]["indexRangeAssertions"] == list(
                index_range_assertions
            )
        record = payload["artifacts"][0]
        assert (
            record["entryPoint"]["target"]
            == {
                "metal": entry,
                "directx": "CSMain",
                "opengl": "main",
            }[target]
        )
        generated = (
            tmp_path / "translated" / (root / record["path"]).relative_to(output)
        )
    return revision, source, generated


def test_pinned_gated_delta_backward_compiles_with_float_storage(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require the pinned Metal compiler gate")
    assert sys.platform == "darwin", "Native Metal compilation requires macOS"
    revision, source, generated = _translate_pinned(tmp_path, "metal")
    root = source.parents[4]
    text = generated.read_text(encoding="utf-8")
    assert "reinterpret_cast<device atomic_float*>" in text
    assert "atomic_fetch_add_explicit(" in text
    evidence = {"commit": revision, "entry": ENTRY, "execution": "not-tested"}
    for label, artifact in (("original", source), ("generated", generated)):
        directory = tmp_path / label
        air = directory / "kernel.air"
        module = directory / "kernel.metallib"
        run(
            [
                "xcrun",
                "--sdk",
                "macosx",
                "metal",
                "-Werror",
                "-I",
                root,
                "-c",
                artifact,
                "-o",
                air,
            ],
            directory,
            "compile",
        )
        run(
            ["xcrun", "--sdk", "macosx", "metallib", air, "-o", module],
            directory,
            "link",
        )
        assert module.stat().st_size > 0
        evidence[label] = {
            "sourceSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "moduleSizeBytes": module.stat().st_size,
        }
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
