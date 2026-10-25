"""Compare complex inverse trigonometry with the pinned original Metal source."""

import ast
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels import test_unary_native_loader as unary
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_current_unary_logarithms import (
    ATOL,
    ROOT,
    RTOL,
    SHAPES,
    _build_complex_reference,
    _compare_complex_source,
    _workload,
)
from tests import workflow_text
from tests.test_translator.test_metal_precise_trig import _bits

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_COMPLEX_INVERSE_TRIG"


def _check_upstream_tolerance(source):
    module = ast.parse(source)
    (function,) = (
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "test_complex_ops"
    )
    (operations,) = (
        node.value
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ops"
            for target in node.targets
        )
    )
    assert {"arcsin", "arccos"} <= set(ast.literal_eval(operations))
    (loop,) = (
        node
        for node in function.body
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "ops"
    )
    (comparison,) = (
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "allclose"
    )
    assert len(comparison.args) == 2 and not comparison.keywords
    assert (RTOL, ATOL) == (1e-5, 1e-8)


@pytest.mark.parametrize("keywords", ("atol=1e-3", "rtol=1e-2", "equal_nan=True"))
def test_complex_inverse_trig_rejects_changed_upstream_comparison(keywords):
    source = """
def test_complex_ops():
    ops = ["arcsin", "arccos"]
    for op in ops:
        assert np.allclose(mx_op(x), np_op(x))
"""
    _check_upstream_tolerance(source)
    with pytest.raises(AssertionError):
        _check_upstream_tolerance(source.replace("np_op(x))", f"np_op(x), {keywords})"))


@pytest.mark.parametrize("shape", SHAPES)
def test_complex_inverse_trig_inputs_cover_branch_boundaries(shape):
    values, indices, _, _ = _workload(shape)
    assert len(indices) == (105 if shape.startswith("gn") else 25200)
    if not shape.startswith("gn"):
        for sign in (0, 0x80000000):
            for real in range(_bits(1) - 4, _bits(1) + 5):
                for imaginary in (0, 0x80000000, 1, 0x80000001):
                    assert (real | sign, imaginary) in values


@pytest.fixture(scope="module")
def inverse_trig_source():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for original-Metal inverse trigonometry")
    assert (
        os.environ.get("CROSTL_MLX_CURRENT_TARGET") == "metal"
    ), "Native Metal required"
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
            "python/tests/test_ops.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert (
        hashlib.sha256((root / unary.MLX_UNARY_SOURCE).read_bytes()).hexdigest()
        == unary.MLX_UNARY_SHA256
    )
    _check_upstream_tolerance((root / "python/tests/test_ops.py").read_text())
    return root


@pytest.mark.parametrize("target", (None, "directx", "opengl"))
def test_required_complex_inverse_trig_rejects_non_metal(monkeypatch, target):
    monkeypatch.setenv(REQUIRE_ENV, "1")
    if target is None:
        monkeypatch.delenv("CROSTL_MLX_CURRENT_TARGET", raising=False)
    else:
        monkeypatch.setenv("CROSTL_MLX_CURRENT_TARGET", target)
    with pytest.raises(AssertionError, match="Native Metal required"):
        inverse_trig_source.__wrapped__()


@pytest.fixture(scope="module")
def inverse_trig_reference(inverse_trig_source, tmp_path_factory):
    return _build_complex_reference(
        inverse_trig_source, tmp_path_factory.mktemp("complex-inverse-trig-source")
    )


@pytest.mark.parametrize("operation", ("ArcSin", "ArcCos"))
def test_current_complex_inverse_trig_native_comparison(
    inverse_trig_source, inverse_trig_reference, tmp_path, operation
):
    _compare_complex_source(
        inverse_trig_source, inverse_trig_reference, tmp_path, operation
    )


def test_ci_requires_guarded_complex_inverse_trig():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    job = workflow_text.workflow_job_text(workflow, "mlx-metal-porting")
    step = job.split("      - name: Validate pinned complex inverse trigonometry\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if: runner.os == 'macOS'" in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step and "CROSTL_MLX_CURRENT_TARGET: metal" in step
    assert "test_current_complex_inverse_trig.py" in step
    assert "--timeout-seconds 180" in step and "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    upload = job.split(
        "      - name: Upload complex inverse trigonometry evidence\n", 1
    )[1].split("      - name:", 1)[0]
    assert "if: always() && runner.os == 'macOS'" in upload
    assert "path: mlx-complex-inverse-trig-results" in upload
    assert "if-no-files-found: error" in upload
