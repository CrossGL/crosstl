from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

from crosstl.project import (
    build_runtime_artifact_manifest,
    load_project_config,
    translate_project,
    validate_project_report,
)

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
SOURCE = "mlx/backend/metal/kernels/quantized.metal"
SOURCE_SHA256 = "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd"
HOST_SOURCE = "mlx/backend/metal/quantized.cpp"
HOST_SOURCE_SHA256 = "c8e05c172b37887c713db0a62343f73710fe1fde14306a995f52149473cfe291"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_QUANTIZED_WIDE_OPENGL"


@pytest.fixture(scope="module")
def quantized_opengl_tools():
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    tools = {name: shutil.which(name) for name in ("glslangValidator", "spirv-val")}
    missing = [name for name, path in tools.items() if path is None]
    if not root_value:
        missing.append("CROSTL_MLX_ROOT")
    if missing:
        message = "MLX quantized OpenGL proof requires " + ", ".join(missing)
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(message)
        pytest.skip(message)

    root = Path(root_value).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert commit == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
            HOST_SOURCE,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    host_source = (root / HOST_SOURCE).read_text(encoding="utf-8")
    assert hashlib.sha256(host_source.encode("utf-8")).hexdigest() == (
        HOST_SOURCE_SHA256
    )
    assert re.search(
        r"constexpr int num_simdgroups = 2;.*?"
        r"MTL::Size group_dims\(32, num_simdgroups, 1\);",
        host_source,
        re.DOTALL,
    )
    return root, tools


@pytest.mark.parametrize("dtype", ["float", "float16_t", "bfloat16_t"])
@pytest.mark.parametrize("bits", [3, 5, 6])
@pytest.mark.parametrize("vectors", [2, 4])
def test_current_mlx_quantized_wide_compiles_to_opengl(
    tmp_path, quantized_opengl_tools, dtype, bits, vectors
):
    root, tools = quantized_opengl_tools
    entry = f"affine_qmv_wide_{dtype}_gs_128_b_{bits}_nv_{vectors}_kl_8_batch_0"
    with tempfile.TemporaryDirectory(prefix=".qwide-", dir=root) as temporary:
        work = Path(temporary)
        config = work / "crosstl.toml"
        config.write_text(
            textwrap.dedent(f"""
                [project]
                source_roots = ["mlx/backend/metal/kernels"]
                include = ["{SOURCE}"]
                include_dirs = ["."]
                targets = ["opengl"]
                output_dir = "{work.name}/out"

                [project.entry_points]
                "{SOURCE}" = "{entry}"

                [project.entry_workgroup_size_rules."{SOURCE}"]
                "{entry}" = [32, 2, 1]

                [project.source_options.metal]
                max_template_specializations = 4096
                max_template_materialization_work = 1048576

                [project.source_options.metal.target_options.opengl]
                software_subgroup_width = 32
                """).strip() + "\n",
            encoding="utf-8",
        )
        report = translate_project(
            load_project_config(root, config),
            format_output=False,
            validate=True,
            run_toolchains=False,
        ).to_json()
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert validate_project_report(report_path)["success"] is True
        assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
        assert report["summary"]["failedCount"] == 0
        assert report["diagnostics"] == []
        assert report["project"]["sourceOptions"]["metal"]["target_options"] == {
            "opengl": {"software_subgroup_width": 32}
        }
        assert len(report["artifacts"]) == 1
        artifact = report["artifacts"][0]
        generated = (root / artifact["path"]).read_bytes()
        output = tmp_path / "quantized.comp"
        output.write_bytes(generated)
        assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
        assert artifact["entryPoint"] == {
            "source": entry,
            "target": "main",
            "stage": "compute",
        }
        execution = artifact["execution"]
        assert execution["sourceEntryPoints"] == [entry]
        assert execution["provenance"] == {
            "kind": "materialized-template-entry-rules",
            "path": f'project.entry_workgroup_size_rules["{SOURCE}"]',
        }
        assert len(execution["entryPoints"]) == 1
        entry_execution = execution["entryPoints"][0]
        assert entry_execution["sourceEntryPoint"] == entry
        assert entry_execution["materializedEntryPoint"] == entry
        assert entry_execution["targetEntryPoint"] == "main"
        assert entry_execution["workgroupSize"] == [32, 2, 1]
        assert entry_execution["rule"] == {
            "components": ["32", "2", "1"],
            "sourcePattern": SOURCE,
            "path": f'project.entry_workgroup_size_rules["{SOURCE}"].{entry}',
            "entryPattern": entry,
        }
        assert entry_execution["parameters"] == {
            "T": dtype,
            "batched": "0",
            "bits": str(bits),
            "group_size": "128",
            "k_lanes": "8",
            "vecs_per_tg": str(vectors),
        }
        assert "subgroupWidth" not in execution
        assert "subgroupWidthEnforcement" not in execution
        assert "subgroupWidth" not in entry_execution
        assert "subgroupWidthEnforcement" not in entry_execution
        assert (
            artifact["generatedHash"]["value"] == hashlib.sha256(generated).hexdigest()
        )

        runtime_manifest = build_runtime_artifact_manifest(report_path)
        assert runtime_manifest["success"] is True, json.dumps(
            runtime_manifest, indent=2
        )
        execution_config = runtime_manifest["artifacts"][0]["hostInterface"][
            "entryPoints"
        ][0]["executionConfig"]
        assert execution_config == {
            "local_size": [32, 2, 1],
            "local_size_x": 32,
            "local_size_y": 2,
            "local_size_z": 1,
        }
        assert "subgroupWidth" not in execution_config

        shader = generated.decode("utf-8")
        assert "inout float w_local[8]" in shader
        assert not re.search(r"\bw_local\s*(?:[+\-*/]?=)", shader)
        assert "CROSSTL_PRIVATE_POINTER_OOB_READ_ZERO" not in shader
        assert "dequantize_float_8_" in shader
        assert "layout(local_size_x = 32, local_size_y = 2, local_size_z = 1)" in shader
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in shader
        assert "shared float crossglSoftwareSubgroupScratchFloat[64];" in shader
        assert "crossglSoftwareSubgroupShuffleDownFloat" in shader
        assert "gl_LocalInvocationIndex / CROSSTL_SOFTWARE_SUBGROUP_WIDTH" in shader
        assert "gl_LocalInvocationIndex % CROSSTL_SOFTWARE_SUBGROUP_WIDTH" in shader
        assert "CROSSTL_REQUIRED_SUBGROUP_WIDTH" not in shader
        assert "GL_KHR_shader_subgroup" not in shader
        assert "gl_Subgroup" not in shader
        if bits == 6:
            assert "w_local_base += int((4 * i));" in shader

        spirv = tmp_path / "quantized.spv"
        for command in (
            [
                tools["glslangValidator"],
                "-G",
                "--target-env",
                "spirv1.3",
                "-S",
                "comp",
                str(output),
                "-o",
                str(spirv),
            ],
            [tools["spirv-val"], "--target-env", "spv1.3", str(spirv)],
        ):
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
        assert spirv.stat().st_size > 20
