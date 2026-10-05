"""Pinned bfloat Sigmoid boundaries with precise exponential and division."""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_unary_complete_directx import (
    UNARY_DIRECTX_WORKLOADS,
    _project_config,
)
from tests.test_translator.test_bfloat_buffer_runtime import _storage
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_exp import _narrow, _oracle
from tests.test_translator.test_metal_precise_trig import _float, _round_decimal
from tools import ci_coverage

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_DIVISION"
ENTRY = "v_Sigmoidbfloat16bfloat16"
SOURCE = "mlx/backend/metal/kernels/unary.metal"
# Original Metal 3.1, -fno-fast-math readbacks at MLX_COMMIT. These cover the
# division underflow boundary and the exponential's bfloat rounding midpoint.
CASES = (
    (0xC0DB, 0x3A8B),
    (0xC2AC, 0x0173),
    (0xC2AD, 0x0114),
    (0xC2AE, 0x00B3),
    (0xC2AF, 0x0000),
    (0xC2B0, 0x0000),
    (0xC2B1, 0x0000),
    (0xC2B2, 0x0000),
    (0xC2B3, 0x0000),
    (0xC2B4, 0x0000),
    (0x0000, 0x3F00),
    (0x8000, 0x3F00),
    (0x3F80, 0x3F3B),
    (0xBF80, 0x3E8A),
    (0x7F80, 0x3F80),
    (0xFF80, 0x0000),
    (0x42B4, 0x3F80),
    (0x0001, 0x3F00),
    (0x8001, 0x3F00),
)


@lru_cache(maxsize=None)
def _sigmoid_reference(word):
    """Evaluate the source's bfloat boundaries with binary32 result flushing."""
    magnitude = word & 0x7FFF
    assert magnitude <= 0x7F80, "NaN inputs require a separate payload policy"
    exponential = _narrow(_oracle(magnitude << 16))
    with localcontext() as context:
        context.prec = 100
        if exponential == 0x7F800000:
            quotient = 0
        else:
            denominator = _narrow(
                _round_decimal(Decimal(1) + Decimal.from_float(_float(exponential)))
            )
            quotient = _round_decimal(
                Decimal(1) / Decimal.from_float(_float(denominator))
            )
            if quotient < 0x800000:
                quotient = 0
        narrowed = _narrow(quotient)
        if word & 0x8000 and magnitude:
            return narrowed >> 16
        return (
            _narrow(_round_decimal(Decimal(1) - Decimal.from_float(_float(narrowed))))
            >> 16
        )


def _sigmoid_cases():
    return tuple(
        (word, _sigmoid_reference(word))
        for word in range(65536)
        if word & 0x7FFF <= 0x7F80
    )


def test_sigmoid_reference_matches_original_boundary_readbacks():
    cases = _sigmoid_cases()
    assert len(cases) == 65282
    assert all(_sigmoid_reference(word) == result for word, result in CASES)
    assert (0x0000, 0x3F00) in cases and (0x8000, 0x3F00) in cases
    assert (0x7F80, 0x3F80) in cases and (0xFF80, 0x0000) in cases


def test_ci_requires_pinned_division_once_per_platform():
    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned unary arithmetic"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'CROSTL_REQUIRE_METAL_PRECISE_EXP: "1"' in step
    assert "CROSTL_MLX_CURRENT_ROOT:" in step
    assert "CROSTL_MLX_CURRENT_TARGET" in step
    assert (
        workflow.count(
            "demos/integrations/mlx/tests/kernels/test_current_unary_division.py"
        )
        == 1
    )
    assert "test_current_unary_division.py" in step
    assert "tests/test_translator/test_metal_precise_exp.py" in step
    assert workflow.count("tests/test_translator/test_metal_precise_exp.py") == 1
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


def test_current_bfloat_sigmoid_non_nan_domain(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned Sigmoid execution")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    assert target in {"directx", "opengl", "metal"}
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
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
        timeout=30,
    )
    workload = next(w for w in UNARY_DIRECTX_WORKLOADS if w.entry_point == ENTRY)
    cases = _sigmoid_cases()
    with tempfile.TemporaryDirectory(prefix=".unary-division-", dir=root) as directory:
        work = Path(directory)
        try:
            config_path = work / "crosstl.toml"
            config_path.write_text(_project_config(workload), encoding="utf-8")
            base = load_project_config(root, config_path)
            options = {
                "metal": {
                    **base.source_options["metal"],
                    "binary32_division_profile": "rne-flush",
                }
            }
            config = replace(base, source_options=options)
            (work / "source-options.json").write_text(
                json.dumps(options, indent=2), encoding="utf-8"
            )
            report = translate_project(
                config,
                targets=(target,),
                output_dir=work.name + "/out",
                format_output=False,
            )
            report.write_json(work / "report.json")
            data = report.to_json()
            assert (
                data["summary"]["translatedCount"] == 1
                and data["summary"]["failedCount"] == 0
            ), data["diagnostics"]
            artifact = data["artifacts"][0]
            assert artifact["entryPoint"]["source"] == ENTRY
            assert artifact["provenance"]["binary32DivisionProfile"] == "rne-flush"
            manifest = build_runtime_artifact_manifest(work / "report.json")
            assert manifest["success"], manifest
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = work / "package"
            assert build_runtime_package(work / "artifacts.json", package)["success"]
            loader = build_runtime_loader_manifest(package / "runtime-package.json")
            assert loader["success"] and len(loader["loadUnits"]) == 1
            descriptor = build_native_loader_abi_descriptor(
                loader, load_unit_id=loader["loadUnits"][0]["id"]
            )
            (work / "descriptor.json").write_text(
                json.dumps(descriptor, indent=2), encoding="utf-8"
            )
            guard = [0x422A] * 8
            inputs = {
                "in_": _storage(target, [word for word, _ in cases] + guard),
                "out_": _storage(target, [0x422A] * (len(cases) + len(guard))),
                "size": {"dtype": "uint32", "shape": [1], "values": [len(cases)]},
            }
            outputs = {"out_": _storage(target, [word for _, word in cases] + guard)}
            request = _request(descriptor, package, inputs, outputs, len(cases))
            assert not request.execution_plan.diagnostics
            (work / "reference.json").write_text(
                json.dumps(
                    {
                        "commit": revision,
                        "entry": ENTRY,
                        "cases": cases,
                        "guardCount": len(guard),
                        "coverage": "all 65282 non-NaN bfloat inputs",
                        "oracle": (
                            "Decimal exponential and arithmetic, binary32/bfloat rounding at source boundaries"
                        ),
                        "originalBoundaryReadbacks": CASES,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            def validate(source, directory, target):
                return _compile(
                    source.read_text(),
                    target,
                    directory,
                    directx_compile_flags=("-enable-16bit-types",),
                    metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
                )[1]

            _execute(
                request, _bound_values(descriptor, outputs), work, validate=validate
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
