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


def test_pinned_gated_delta_backward_compiles_with_float_storage(tmp_path):
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require the pinned Metal compiler gate")
    assert sys.platform == "darwin", "Native Metal compilation requires macOS"
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
    source = root / SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-gated-delta-", dir=root
    ) as output:
        report = translate_project(
            ProjectConfig(
                root=root,
                include_patterns=(SOURCE,),
                include_dirs=(".",),
                targets=("metal",),
                entry_points={SOURCE: (ENTRY,)},
                entry_workgroup_size_rules={SOURCE: {ENTRY: (32, 1, 1)}},
                source_options={
                    "metal": {
                        "max_template_specializations": 128,
                        "max_template_materialization_work": 8192,
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
        record = payload["artifacts"][0]
        assert record["entryPoint"]["target"] == ENTRY
        generated = (
            tmp_path / "translated" / (root / record["path"]).relative_to(output)
        )
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
