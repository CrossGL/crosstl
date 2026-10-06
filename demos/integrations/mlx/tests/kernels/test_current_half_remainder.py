"""Native half remainder from the pinned, unmodified binary kernel tree."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
)
from demos.integrations.mlx.tests.kernels.test_binary_complete_opengl import (
    BINARY_OPENGL_WORKLOADS,
    MLX_BINARY_SHA256,
    MLX_BINARY_SOURCE,
    _project_config,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_current_copy import _half_payload
from demos.integrations.mlx.tests.kernels.test_current_gather import _bound_inputs
from tests.runtime_helpers import _prepare_native_package, _validate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_half_remainder import _oracle, _pairs
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_HALF_REMAINDER"
ENTRY = "vv_Remainderfloat16"
GUARD = 0x3555


def _value(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


def _remainder(a, b):
    word = _oracle(a, b)
    r, y = _value(word), _value(b)
    if r != 0 and (r < 0) != (y < 0):
        word = struct.unpack("<H", struct.pack("<e", r + y))[0]
    return word


def _check_words(actual, expected, width):
    assert width in (16, 32)
    assert len(actual) == len(expected)
    magnitude, infinity = (0x7FFFFFFF, 0x7F800000) if width == 32 else (0x7FFF, 0x7C00)
    mismatches = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert type(got) is int and 0 <= got < 1 << width
        if got != want and not (
            got & magnitude > infinity and want & magnitude > infinity
        ):
            mismatches.append({"index": index, "expected": want, "actual": got})
    assert not mismatches, mismatches[:20]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0xC580, 0x4000, 0x3800),
        (0x4580, 0xC000, 0xB800),
        (0x8000, 0x4000, 0),
        (3, 2, 1),
    ],
)
def test_remainder_reference_retains_source_sign_adjustment(a, b, expected):
    assert _remainder(a, b) == expected


def test_remainder_comparison_rejects_finite_and_zero_sign_errors():
    for width, positive, negative in ((16, 0, 0x8000), (32, 0, 0x80000000)):
        with pytest.raises(AssertionError):
            _check_words([negative], [positive], width)
        with pytest.raises(AssertionError):
            _check_words([positive + 1], [positive], width)
        with pytest.raises(AssertionError):
            _check_words([1 << width], [positive], width)


def test_current_half_remainder_native_parity(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned native half remainder")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    assert target in {"directx", "opengl", "metal"}
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
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
    assert (
        hashlib.sha256((root / MLX_BINARY_SOURCE).read_bytes()).hexdigest()
        == MLX_BINARY_SHA256
    )
    pairs = _pairs()
    expected = [_remainder(a, b) for a, b in pairs] + [GUARD] * 8
    workload = next(w for w in BINARY_OPENGL_WORKLOADS if w.entry_point == ENTRY)
    with tempfile.TemporaryDirectory(
        prefix=".current-half-remainder-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            config_path = work / "crosstl.toml"
            config_path.write_text(_project_config(workload), encoding="utf-8")
            report = translate_project(
                load_project_config(root, config_path),
                targets=(target,),
                output_dir=work.name + "/out",
                format_output=False,
            )
            report.write_json(work / "report.json")
            data = report.to_json()
            assert (
                data["summary"]["translatedCount"] == 1
                and not data["summary"]["failedCount"]
            ), data["diagnostics"]
            (artifact,) = data["artifacts"]
            assert (
                artifact["provenance"]["binary16RemainderProfile"]
                == "binary32-quotient"
            )
            descriptor, package = _prepare_native_package(report, work)
            source = package / descriptor["artifact"]["packagePath"]
            _validate(source, work, target)
            inputs = {
                "a": _half_payload(target, [a for a, _ in pairs]),
                "b": _half_payload(target, [b for _, b in pairs]),
                "c": _half_payload(target, [GUARD] * len(expected)),
                "size": {"dtype": "uint32", "shape": [1], "values": [len(pairs)]},
            }
            outputs = _bound_values(descriptor, {"c": _half_payload(target, expected)})
            request = build_native_loader_dispatch_request(
                descriptor,
                package,
                _bound_inputs(descriptor, ENTRY, inputs),
                outputs,
                {"workgroupCount": [len(pairs), 1, 1], "workgroupSize": [1, 1, 1]},
                expected_target=target,
            )
            assert not request.execution_plan.diagnostics
            (work / "values.json").write_text(
                json.dumps(
                    {"pairs": pairs, "inputs": inputs, "outputs": outputs}, indent=2
                ),
                encoding="utf-8",
            )
            executor = _executor(target)
            try:
                availability = executor.is_available(request)
                assert availability.available, availability.reason
                result = executor.run(request)
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if close:
                    close()
            (work / "result.json").write_text(
                json.dumps(
                    {
                        "status": result.status,
                        "outputs": result.outputs,
                        "details": result.details,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            assert result.status == "ok"
            (actual,) = result.outputs.values()
            (wanted,) = outputs.values()
            assert {k: v for k, v in actual.items() if k != "values"} == {
                k: v for k, v in wanted.items() if k != "values"
            }
            _check_words(
                actual["values"], wanted["values"], 32 if target == "opengl" else 16
            )
            assert actual["values"][-8:] == wanted["values"][-8:]
            if target == "metal":
                _original_metal(root, work, pairs, expected)
            (work / "evidence.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entryPoint": ENTRY,
                        "target": target,
                        "artifactSha256": (
                            hashlib.sha256(source.read_bytes()).hexdigest()
                        ),
                        "pairCount": len(pairs),
                        "guardCount": 8,
                        "comparison": "exact words; NaN classification",
                        "wholeFamilyParity": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def _original_metal(root, work, pairs, expected):
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-original-runner",
    )
    library = _metal_library(
        root / MLX_BINARY_SOURCE, work / "original.metallib", root, upstream=True
    )
    values = [
        [a for a, _ in pairs],
        [b for _, b in pairs],
        [GUARD] * len(expected),
        [len(pairs)],
    ]
    buffers = []
    for index, words in enumerate(values):
        path = work / f"original-input-{index}.bin"
        path.write_bytes(
            struct.pack("<" + ("I" if index == 3 else "H") * len(words), *words)
        )
        buffers.append(str(path))
    request = work / "original-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": buffers,
                "workgroupCount": [len(pairs), 1, 1],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = work / "original-readback"
    _run([runner, library, ENTRY, request, output], work, "original-execute")
    actual = list(
        struct.unpack(f"<{len(expected)}H", (output / "buffer-2.bin").read_bytes())
    )
    _check_words(actual, expected, 16)
    assert actual[-8:] == expected[-8:]
    for index in (0, 1, 3):
        assert (output / f"buffer-{index}.bin").read_bytes() == Path(
            buffers[index]
        ).read_bytes()


def test_ci_requires_pinned_half_remainder_in_existing_native_jobs():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    for job, name in (
        ("portable-host", "Validate pinned native binary math"),
        ("mlx-metal-porting", "Prove current MLX binary shape parity"),
    ):
        step = ci_coverage.workflow_job_step_section(workflow, job, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert (
            "demos/integrations/mlx/tests/kernels/test_current_half_remainder.py"
            in step
        )
        assert "pytest -q -n auto" in step
