"""Complete complex scans preserve both scalar components and launch order."""

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
    _guard_words,
    _input_words,
    _packet,
)
from demos.integrations.mlx.tests.kernels.test_current_scan import (
    scan_package as scan_package,
)
from tests import workflow_text

REFERENCE = SCAN_REFERENCE.with_name("scan_complex_reference.json")
REFERENCE_SHA256 = "4ecef33032d9d589055010a285a73e49fda79ac462bf020be5b91a8b2c413d8e"
ENTRY = _entry("complex64")
LAUNCHES = tuple(
    pytest.param(
        (width, (ENTRY,)),
        id=str(width),
        marks=pytest.mark.xdist_group(f"complex-width-{width}"),
    )
    for width in (64, 128, 256)
)


def _axes(width):
    return 1, 31, 33, width * 2 - 1, width * 2, width * 2 + 1, width * 4 + 1


def _references():
    raw = REFERENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA256
    data = json.loads(raw)
    assert data["upstreamCommit"] == MLX_COMMIT
    assert data["source"] == SOURCE and data["sourceSha256"] == SOURCE_SHA256
    assert data["reference"] == "original-metal"
    assert data["compileFlags"] == ["-std=metal3.2", "-fno-fast-math"]
    assert data["workgroupCount"] == [1, 3, 1] and data["guardElements"] == 32
    assert data["components"] == ["real", "imag"]
    cases = {(case["workgroupWidth"], case["axisSize"]): case for case in data["cases"]}
    assert len(cases) == len(data["cases"]) == 21
    assert set(cases) == {
        (width, axis) for width in (64, 128, 256) for axis in _axes(width)
    }
    for (_, axis), case in cases.items():
        words = case["outputWords"]
        assert len(words) == 3 * axis * 2 + 64
        assert words[-64:] == _guard_words("complex64")
        assert all(type(word) is int and 0 <= word < 1 << 32 for word in words)
    return cases


def test_complex_references_cover_components_and_native_boundaries():
    cases = _references()
    assert sum(len(case["outputWords"]) - 64 for case in cases.values()) == 28068


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_complex_transport_retains_interleaved_float_bits(target):
    words = _input_words("complex64", 33)
    packet = _packet("complex64", target, words)
    assert packet == _packet("float32", target, words)
    assert len(words) == 198 and 0x80000000 in words
    assert len(_guard_words("complex64")) == 64
    with pytest.raises(AssertionError):
        _packet("complex64", target, words[:-1])


@pytest.mark.parametrize("scan_package", LAUNCHES, indirect=True)
def test_complete_complex_scan_retains_reflected_component_layout(scan_package):
    _target, _package, descriptors, _modules, _original, _width = scan_package
    descriptor = descriptors[ENTRY]
    layouts = {
        binding["provenance"]["sourceResource"]["parameter"]: binding["scalarLayout"]
        for binding in descriptor["bindings"]
        if binding.get("provenance", {}).get("sourceResource", {}).get("parameter")
        in {"in", "out"}
    }
    assert set(layouts) == {"in", "out"}
    for layout in layouts.values():
        assert layout["elementType"] == "float32" and layout["componentCount"] == 2
        assert layout["elementSizeBytes"] == layout["elementStrideBytes"] == 8
        assert layout["structMembers"] == [
            {"name": "real", "physicalType": "float", "offsetBytes": 0},
            {"name": "imag", "physicalType": "float", "offsetBytes": 4},
        ]


@pytest.mark.parametrize("scan_package", LAUNCHES, indirect=True)
@pytest.mark.parametrize("axis_index", range(7))
def test_complete_complex_scan_executes_natively(scan_package, tmp_path, axis_index):
    width = scan_package[-1]
    axis = _axes(width)[axis_index]
    expected = _references()[width, axis]["outputWords"]
    _execute_scan_case(scan_package, tmp_path, "complex64", axis, expected)


def test_ci_requires_complex_scan_execution_on_each_native_platform():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    job = workflow_text.workflow_job_text(workflow, "mlx-metal-porting")
    step = workflow_text.workflow_step_section(job, "Validate pinned complex scans")
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    for platform, target in (
        ("Linux", "opengl"),
        ("Windows", "directx"),
        ("macOS", "metal"),
    ):
        assert f"{platform}) export CROSTL_MLX_CURRENT_TARGET={target}" in step
    assert "test_current_complex_scan.py" in step
    assert "--timeout-seconds 240" in step and "pytest -q -n auto" in step
    assert "--dist=loadgroup" in step
    for parameter in LAUNCHES:
        ((width, _entries),) = parameter.values
        assert [(mark.name, mark.args) for mark in parameter.marks] == [
            ("xdist_group", (f"complex-width-{width}",))
        ]
    assert "--basetemp=" in step and "--junitxml=" in step
    upload = workflow_text.workflow_step_section(job, "Upload complex scan evidence")
    assert "if: always()" in upload and "if-no-files-found: error" in upload
    assert (
        "path: mlx-complex-scan-results" in upload
        and "include-hidden-files: true" in upload
    )
