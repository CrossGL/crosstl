"""Current MLX FP8 conversion through native project packages."""

import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
)
from demos.integrations.mlx.tests.corpus_evidence import (
    record_native_request,
    record_native_result,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_binary_shapes import (
    _validate_generated_artifact,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_unary_complete_opengl import (
    MLX_UNARY_SOURCE,
    UNARY_OPENGL_WORKLOADS,
    _project_config,
)
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_half_copy_identity import _widen
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FP8"
ENTRY = "v2_ToFP8float16uint8"
PROFILE = "canonical-nan"
GUARDS = [57, 186] * 16
GRID = [13, 1261, 1]


def _expected(word):
    # This source profile canonicalizes half NaNs during numeric float widening.
    if word & 0x7FFF > 0x7C00:
        return 126
    sign = (word >> 8) & 128
    value = abs(struct.unpack("<e", struct.pack("<H", word))[0])
    if not math.isfinite(value) or value >= 448:
        code = 126
    elif value < 2**-6:
        code = round(value * 512)
    else:
        exponent = math.frexp(value)[1] - 1
        mantissa = round(math.ldexp(value, 3 - exponent))
        if mantissa == 16:
            exponent += 1
            mantissa = 8
        code = min(126, (exponent + 7) * 8 + mantissa - 8)
    return sign | code


@pytest.mark.parametrize(
    "word,expected",
    [
        (0, 0),
        (0x8000, 128),
        (0x3C00, 56),
        (0xBC00, 184),
        (0x7C00, 126),
        (0xFC00, 254),
        (0x7C01, 126),
        (0xFC01, 126),
        (0x7E00, 126),
        (0xFE00, 126),
        (1, 0),
        (0x8001, 128),
        (0x6400, 126),
        (0xE400, 254),
    ],
)
def test_current_half_fp8_oracle(word, expected):
    assert _expected(word) == expected


def _bindings(descriptor, words, expected, fill):
    inputs, outputs, names = {}, {}, set()
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        member = layout.get("memberName", binding["name"]).removeprefix(ENTRY + "_")
        member = {"in_": "in", "out_": "out"}.get(member, member)
        assert member in {"in", "out", "size"} and member not in names
        names.add(member)
        dtype, encoding = layout["elementType"], None
        if member == "in":
            dtype = layout.get("storageEncoding", {}).get("logicalElementType", dtype)
            assert dtype in {"float32", "float16"}
            data = [_widen(word) for word in words] if dtype == "float32" else words
            encoding = {
                "float32": "ieee754-binary32",
                "float16": "ieee754-binary16",
            }[dtype]
        elif member == "size":
            assert dtype == "int64"
            data = [len(words)]
        else:
            assert dtype in {"uint8", "uint32"}
            data = [fill] * len(words) + GUARDS
        value = {"dtype": dtype, "shape": [len(data)], "values": data}
        if encoding:
            value["encoding"] = encoding
        inputs[binding["name"]] = value
        if member == "out":
            outputs[binding["name"]] = {**value, "values": expected + GUARDS}
    assert names == {"in", "out", "size"} and len(outputs) == 1
    return inputs, outputs


def _check_output(actual, expected):
    assert len(actual) == len(expected) + len(GUARDS)
    assert actual[-len(GUARDS) :] == GUARDS
    mismatches = [
        (index, got, want)
        for index, (got, want) in enumerate(zip(actual[: -len(GUARDS)], expected))
        if got != want
    ]
    assert not mismatches, mismatches[:16]


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_current_half_fp8_bindings_retain_logical_input_type(tmp_path, target):
    from tests.test_translator.test_half_buffer_runtime import WORDS, _directx_buffers
    from tests.test_translator.test_software_subgroup_product import _package

    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void {ENTRY}(const device half* in [[buffer(0)]],
                    device uchar* out [[buffer(1)]],
                    constant long& size [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {{
    if (long(i) < size) {{ out[i] = uchar(as_type<ushort>(in[i])); }}
}}
"""
    _, descriptor, package = _package(
        tmp_path, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    words = list(WORDS)
    inputs, outputs = _bindings(descriptor, words, [word & 255 for word in words], 85)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [len(words), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    input_name = next(name for name, value in inputs.items() if value.get("encoding"))
    assert inputs[input_name] == {
        "dtype": "float32" if target == "opengl" else "float16",
        "encoding": "ieee754-binary32" if target == "opengl" else "ieee754-binary16",
        "shape": [len(words)],
        "values": [_widen(word) for word in words] if target == "opengl" else words,
    }
    if target == "directx":
        buffer = next(
            buffer for buffer in _directx_buffers(request) if buffer.name == input_name
        )
        assert buffer.stride == 2 and buffer.byte_length == len(words) * 2
        assert buffer.payload == struct.pack("<" + "H" * len(words), *words)


@pytest.mark.parametrize("defect", ["nan-sign", "guard", "truncated", "empty"])
def test_current_half_fp8_rejects_bad_readbacks(defect):
    expected = [_expected(word) for word in range(65536)]
    actual = expected + GUARDS
    if defect == "nan-sign":
        actual[0xFC01] = 254
    elif defect == "guard":
        actual[-1] ^= 1
    elif defect == "truncated":
        actual.pop()
    else:
        actual = []
    with pytest.raises(AssertionError):
        _check_output(actual, expected)


def test_current_half_fp8_native_loader(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for current FP8 native execution")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    assert target in {"metal", "directx", "opengl"}
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
    words = list(range(65536))
    expected = [_expected(word) for word in words]
    assert GRID[0] * GRID[1] * 4 >= len(words)
    assert GRID[0] * GRID[1] * 4 < 2**31
    workload = next(row for row in UNARY_OPENGL_WORKLOADS if row.entry_point == ENTRY)
    with tempfile.TemporaryDirectory(
        prefix=".current-half-fp8-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            config_path = work / "crosstl.toml"
            config_path.write_text(_project_config(workload), encoding="utf-8")
            config = load_project_config(root, config_path)
            options = {
                **config.source_options,
                "metal": {
                    **config.source_options["metal"],
                    "binary16_widening_profile": PROFILE,
                },
            }
            config = replace(
                config,
                targets=(target,),
                output_dir=work / "out",
                source_options=options,
            )
            report = translate_project(config, format_output=False)
            report.write_json(work / "report.json")
            payload = report.to_json()
            assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
            assert len(payload["artifacts"]) == 1
            artifact = payload["artifacts"][0]
            assert artifact["entryPoint"]["source"] == ENTRY
            assert artifact["provenance"]["binary16WideningProfile"] == PROFILE
            descriptor, package = _prepare_native_package(report, work)
            packaged = json.loads((package / "runtime-package.json").read_text())
            assert (
                packaged["artifacts"][0]["provenance"]["binary16WideningProfile"]
                == PROFILE
            )
            source = package / descriptor["artifact"]["packagePath"]
            if target == "metal":
                _metal_library(source, work / "strict.metallib", root)
                runner = work / "readback"
                fixture = (
                    Path(__file__).resolve().parents[5]
                    / "tests/fixtures/runtime_verification/metal_raw_buffers.swift"
                )
                _run(
                    ["xcrun", "swiftc", fixture, "-o", runner],
                    work,
                    "build-reference-runner",
                )
                original = _metal_library(
                    root / MLX_UNARY_SOURCE,
                    work / "original.metallib",
                    root,
                    upstream=True,
                )
            else:
                _validate_generated_artifact(source, work, target)
            readbacks = []
            for fill in (85, 170):
                trial = work / f"fill-{fill}"
                trial.mkdir()
                inputs, outputs = _bindings(descriptor, words, expected, fill)
                request = build_native_loader_dispatch_request(
                    descriptor,
                    package,
                    inputs,
                    outputs,
                    {"workgroupCount": GRID, "workgroupSize": [1, 1, 1]},
                    expected_target=target,
                )
                record_native_request(
                    trial, request, commit=MLX_COMMIT, workload_id=ENTRY
                )
                assert not request.execution_plan.diagnostics
                executor = _executor(target)
                availability = executor.is_available(request)
                assert availability.available, availability.reason
                result = executor.run(request)
                record_native_result(trial, result)
                assert result.status == "ok"
                actual = result.outputs[next(iter(outputs))]["values"]
                _check_output(actual, expected)
                readbacks.append(actual)
                if target == "metal":
                    raw = [
                        struct.pack("<65536H", *words),
                        bytes([fill] * len(words) + GUARDS),
                        struct.pack("<q", len(words)),
                    ]
                    paths = []
                    for index, contents in enumerate(raw):
                        path = trial / f"original-input-{index}.bin"
                        path.write_bytes(contents)
                        paths.append(str(path))
                    original_request = trial / "original-request.json"
                    original_request.write_text(
                        json.dumps(
                            {
                                "buffers": paths,
                                "workgroupCount": GRID,
                                "workgroupSize": [1, 1, 1],
                                "simdWidth": 32,
                            }
                        ),
                        encoding="utf-8",
                    )
                    output = trial / "original-readback"
                    _run(
                        [runner, original, ENTRY, original_request, output],
                        trial,
                        "original-execute",
                    )
                    original_values = list((output / "buffer-1.bin").read_bytes())
                    _check_output(original_values, expected)
                    assert original_values == actual
                    for index in (0, 2):
                        assert (output / f"buffer-{index}.bin").read_bytes() == raw[
                            index
                        ]
            assert readbacks[0] == readbacks[1]
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_current_half_fp8():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Validate pinned unary arithmetic"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "demos/integrations/mlx/tests/kernels/test_current_unary_fp8.py" in step
    assert "pytest -q -n auto" in step
    assert 'PYTHONUNBUFFERED: "1"' in step
    assert "-vv -s --durations=20" in step
    assert "-o faulthandler_timeout=0" in step
    assert "--timeout-seconds 300" in step
