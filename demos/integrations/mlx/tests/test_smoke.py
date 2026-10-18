"""Translate two pinned entries across targets before the short numerical gate."""

import os
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    load_project_config,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels import test_current_arange as arange
from tests.runtime_helpers import _validate_generated_artifact

ENTRIES = ("arangeint32", "arangefloat32")


@pytest.fixture(scope="module")
def smoke_root():
    if os.environ.get("CROSTL_REQUIRE_MLX_SMOKE") != "1":
        pytest.skip("set CROSTL_REQUIRE_MLX_SMOKE=1 for pinned translation smoke")
    root, _target = arange.current_mlx.__wrapped__()
    return root


@pytest.mark.parametrize("target", arange.TARGETS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_pinned_translation_smoke(smoke_root, tmp_path, target, entry):
    # Project reports must keep their output inside the project root.
    with tempfile.TemporaryDirectory(prefix=".mlx-smoke-", dir=smoke_root) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(
            arange._config(f"{work.name}/out", target, entry), encoding="utf-8"
        )
        translation = translate_project(
            load_project_config(smoke_root, config),
            format_output=False,
            validate=True,
            run_toolchains=False,
        )
        report_path = work / "report.json"
        translation.write_json(report_path)
        report = translation.to_json()
        validation = validate_project_report(report_path)
        assert validation["success"] is True, validation["diagnostics"]
        assert report["diagnostics"] == []
        assert report["summary"]["translatedCount"] == 1
        assert report["summary"]["failedCount"] == 0
        (artifact,) = report["artifacts"]
        expected = arange._contract()["artifacts"][target][entry]
        assert artifact["sourceHash"]["value"] == arange.SOURCE_SHA256
        assert artifact["generatedHash"]["value"] == expected["sha256"]
        assert artifact["generatedSizeBytes"] == expected["sizeBytes"]
        (execution,) = artifact["execution"]["entryPoints"]
        assert execution["workgroupSize"] == list(arange.WORKGROUP_SIZE)
        if target != "metal":
            _validate_generated_artifact(
                smoke_root / artifact["path"], tmp_path, target
            )
