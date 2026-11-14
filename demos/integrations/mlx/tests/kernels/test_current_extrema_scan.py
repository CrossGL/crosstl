"""Pinned extrema scans retain native boundaries, identities and NaN handling."""

import hashlib
import json

import pytest

from demos.integrations.mlx.tests.kernels.test_current_scan import (
    MLX_COMMIT,
)
from demos.integrations.mlx.tests.kernels.test_current_scan import (
    REFERENCE as SCAN_REFERENCE,
)
from demos.integrations.mlx.tests.kernels.test_current_scan import (
    REQUIRE_ENV,
    ROOT,
    SOURCE,
    SOURCE_SHA256,
    _entry,
    _execute_scan_case,
    _guard,
    _input_words,
    _packet,
)
from demos.integrations.mlx.tests.kernels.test_current_scan import (
    scan_package as scan_package,
)
from tests import workflow_text

EXTREMA = tuple(
    (operation, dtype) for operation in ("min", "max") for dtype in ("float32", "int32")
)
REFERENCE = SCAN_REFERENCE.with_name("scan_extrema_reference.json")
REFERENCE_SHA256 = "8c4bdf8b6b6fa459faf8bbcf1fbaf885ca8dc927e95f607541cfeb2b023a0304"
LAUNCHES = tuple(
    pytest.param(
        (width, tuple(_entry(dtype, operation) for operation, dtype in EXTREMA)),
        id=str(width),
        marks=pytest.mark.xdist_group(f"extrema-width-{width}"),
    )
    for width in (128, 256)
)


def _axes(width):
    return 1, 33, width * 4 - 1, width * 4, width * 4 + 1


def _references():
    raw = REFERENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA256
    data = json.loads(raw)
    assert data["upstreamCommit"] == MLX_COMMIT
    assert data["source"] == SOURCE and data["sourceSha256"] == SOURCE_SHA256
    assert data["reference"] == "original-metal"
    assert data["compileFlags"] == ["-std=metal3.2", "-fno-fast-math"]
    assert data["workgroupCount"] == [1, 3, 1] and data["guardElements"] == 32
    cases = {
        (
            case["workgroupWidth"],
            case["operation"],
            case["dtype"],
            case["axisSize"],
        ): case
        for case in data["cases"]
    }
    assert len(cases) == len(data["cases"]) == 40
    assert set(cases) == {
        (width, operation, dtype, axis)
        for width in (128, 256)
        for operation, dtype in EXTREMA
        for axis in _axes(width)
    }
    for (_, _, dtype, axis), case in cases.items():
        words = case["outputWords"]
        assert len(words) == 3 * axis + 32
        assert words[-32:] == [_guard(dtype)] * 32
        assert all(type(word) is int and 0 <= word < 1 << 32 for word in words)
    return cases


def test_extrema_references_cover_native_launch_boundaries():
    cases = _references()
    assert sum(len(case["outputWords"]) - 32 for case in cases.values()) == 56112


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("dtype", ("float32", "int32"))
def test_extrema_transport_retains_nan_and_integer_boundary_bits(target, dtype):
    words = _input_words(dtype, 33, "min")
    packet = _packet(dtype, target, words)
    assert [word & 0xFFFFFFFF for word in packet["values"]] == words
    assert 0x80000000 in words
    assert (0x7FA12345 if dtype == "float32" else 0x7FFFFFFF) in words


@pytest.mark.parametrize("scan_package", LAUNCHES, indirect=True)
@pytest.mark.parametrize("operation,dtype", EXTREMA)
@pytest.mark.parametrize("axis_index", range(5))
def test_complete_extrema_scan_executes_natively(
    scan_package, tmp_path, operation, dtype, axis_index
):
    width = scan_package[-1]
    axis = _axes(width)[axis_index]
    expected = _references()[width, operation, dtype, axis]["outputWords"]
    _execute_scan_case(scan_package, tmp_path, dtype, axis, expected, operation)


def test_ci_requires_extrema_scan_execution_on_each_native_platform():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    job = workflow_text.workflow_job_text(workflow, "mlx-metal-porting")
    step = workflow_text.workflow_step_section(job, "Validate pinned extrema scans")
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    for platform, target in (
        ("Linux", "opengl"),
        ("Windows", "directx"),
        ("macOS", "metal"),
    ):
        assert f"{platform}) export CROSTL_MLX_CURRENT_TARGET={target}" in step
    assert "test_current_extrema_scan.py" in step
    assert "--timeout-seconds 240" in step and "pytest -q -n auto" in step
    assert "--dist=loadgroup" in step
    for parameter in LAUNCHES:
        ((width, _entries),) = parameter.values
        assert [(mark.name, mark.args) for mark in parameter.marks] == [
            ("xdist_group", (f"extrema-width-{width}",))
        ]
    assert "--basetemp=" in step and "--junitxml=" in step
    upload = workflow_text.workflow_step_section(job, "Upload extrema scan evidence")
    assert "if: always()" in upload and "if-no-files-found: error" in upload
    assert "path: mlx-extrema-scan-results" in upload
    assert "include-hidden-files: true" in upload
