"""Pinned bfloat Sigmoid boundaries with precise exponential and division."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_current_gather import _bound_inputs
from demos.integrations.mlx.tests.kernels.test_unary_complete_directx import (
    UNARY_DIRECTX_WORKLOADS,
    _project_config,
)
from tests.test_translator.test_bfloat_buffer_runtime import _storage
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_half_buffer_runtime import _directx_buffers
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_exp import _narrow, _oracle
from tests.test_translator.test_metal_precise_trig import _float, _round_decimal
from tests.test_translator.test_software_subgroup_product import _package
from tools import ci_coverage

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_DIVISION"
ENTRY = "v_Sigmoidbfloat16bfloat16"
SOURCE = "mlx/backend/metal/kernels/unary.metal"
SOURCE_SHA256 = "51af04126d68e1f5baee5f467268408650d24a68db66e8c044f7f0be3f15368b"
PROFILE = "rne-flush"
ARTIFACTS = {
    "metal": {
        "sha256": "058518d8f88cf954951e1eb3ec95cc12f6e3faac9d0ccfcd2fb2b47cf104b7dc",
        "sizeBytes": 6502,
    },
    "opengl": {
        "sha256": "c875953cd46217256c9228247220ebec524bb8b938b03b6733799e53bddfeeaa",
        "sizeBytes": 10670,
    },
    "directx": {
        "sha256": "26d709850bef007fc27f68de6f624120c4cb93b711ee9b649401d854cc272a0d",
        "sizeBytes": 9914,
    },
}
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


def _dispatch_request(descriptor, package, cases):
    target = descriptor["target"]
    guard = [0x422A] * 8
    inputs = {
        "in_": _storage(target, [word for word, _ in cases] + guard),
        "out_": _storage(target, [0x422A] * (len(cases) + len(guard))),
        "size": {"dtype": "uint32", "shape": [1], "values": [len(cases)]},
    }
    outputs = _bound_values(
        descriptor, {"out_": _storage(target, [word for _, word in cases] + guard)}
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_inputs(descriptor, ENTRY, inputs),
        outputs,
        {"workgroupCount": [len(cases), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, outputs


@pytest.mark.parametrize("target", ("directx", "opengl", "metal"))
def test_sigmoid_dispatch_binds_reflected_constant_buffer(tmp_path, target):
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void {ENTRY}(device const bfloat* in_ [[buffer(0)]],
                    device bfloat* out_ [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {{
    if (i < size) {{ out_[i] = in_[i]; }}
}}
"""
    _, descriptor, package = _package(
        tmp_path, target, "bfloat", (1, 1, 1), source=source, software_subgroups=False
    )
    request, outputs = _dispatch_request(descriptor, package, CASES)
    assert request.execution_plan.dispatch.global_size == (len(CASES), 1, 1)
    output_name = "out_Buffer" if target == "opengl" else "out_"
    assert outputs == {
        output_name: _storage(target, [word for _, word in CASES] + [0x422A] * 8)
    }
    if target == "directx":
        binding = next(
            b for b in descriptor["bindings"] if b["kind"] == "constant-buffer"
        )
        assert binding["name"] == f"{ENTRY}_size_Constants"
        assert binding["scalarLayout"]["memberName"] == f"{ENTRY}_size"
        buffers = {buffer.name: buffer for buffer in _directx_buffers(request)}
        constant = buffers[binding["name"]]
        assert constant.payload == struct.pack("<I", len(CASES))
        assert constant.byte_length == 16 and constant.allocation_size == 256
        words = [word for word, _ in CASES] + [0x422A] * 8
        assert buffers["in_"].payload == struct.pack(f"<{len(words)}H", *words)
        assert buffers["out_"].payload == struct.pack(
            f"<{len(words)}H", *([0x422A] * len(words))
        )
        assert buffers["in_"].stride == buffers["out_"].stride == 2


@pytest.mark.parametrize("profile", (None, "rne-gradual", "rne-flush"))
def test_unary_project_configuration_retains_arithmetic_profile(tmp_path, profile):
    workload = next(w for w in UNARY_DIRECTX_WORKLOADS if w.entry_point == ENTRY)
    config_path = tmp_path / "crosstl.toml"
    config_path.write_text(
        _project_config(workload, binary32_division_profile=profile), encoding="utf-8"
    )
    config = load_project_config(tmp_path, config_path)
    expected = {
        "max_template_specializations": 64,
        "max_template_materialization_work": 4096,
    }
    if profile is not None:
        expected["binary32_division_profile"] = profile
    assert config.source_options == {"metal": expected}
    assert config.entry_points == {SOURCE: ENTRY}


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
            config_path.write_text(
                _project_config(workload, binary32_division_profile=PROFILE),
                encoding="utf-8",
            )
            config = load_project_config(root, config_path)
            options = config.source_options
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
            assert artifact["sourceHash"] == {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
            assert artifact["provenance"]["binary32DivisionProfile"] == PROFILE
            assert data["project"]["sourceOptions"] == options
            identity = ARTIFACTS[target]
            generated = (root / artifact["path"]).read_bytes()
            assert artifact["generatedHash"] == {
                "algorithm": "sha256",
                "value": identity["sha256"],
            }
            assert hashlib.sha256(generated).hexdigest() == identity["sha256"]
            assert (
                len(generated)
                == artifact["generatedSizeBytes"]
                == identity["sizeBytes"]
            )
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
            request, outputs = _dispatch_request(descriptor, package, cases)
            (work / "reference.json").write_text(
                json.dumps(
                    {
                        "commit": revision,
                        "entry": ENTRY,
                        "cases": cases,
                        "guardCount": 8,
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

            _execute(request, outputs, work, validate=validate)
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
