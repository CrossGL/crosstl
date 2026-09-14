from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    DirectXComputeRuntime,
    DirectXRuntimeParityAdapter,
    OpenGLComputeRuntime,
    OpenGLRuntimeParityAdapter,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
    validate_project_report,
)

ROOT = Path(__file__).resolve().parents[2]
MLX_COMMIT = "d9add9d11f3154111a4c85f267ec2fd307ecd18e"
SOURCE = "mlx/backend/metal/kernels/arg_reduce.metal"
SOURCE_SHA256 = "036a586af92b869a94f0b67e54590abf935e87f99290db46daed86cd8941e429"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ARG_REDUCE"
REQUIRE_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ARG_REDUCE_RUNTIME"
ARTIFACTS = {
    "directx": {
        "argmin_float32": {
            "sha256": "0768cdc9658dd81ab76c02b6d59baf9a7b1103dae09ef56e77e09e181c0ed825",
            "sizeBytes": 6813,
        },
        "argmax_float32": {
            "sha256": "35be05ff5f8485644cb3cddad86c1e17b19eaa05599611baf95997e9ff646d88",
            "sizeBytes": 6815,
        },
    },
    "opengl": {
        "argmin_float32": {
            "sha256": "14fcee01e9c7f5b5bb05a32262e6c3932cf80ea8bdd50582f26ae77f8976d320",
            "sizeBytes": 7750,
        },
        "argmax_float32": {
            "sha256": "4b0595da819da2efaa81672604182e87f870b6fde30cb255bbcf434d2ddc7e21",
            "sizeBytes": 7756,
        },
    },
    "metal": {
        "argmin_float32": {
            "sha256": "955ededf34bbf85d26c24b75da7d9a8f2d394eb359af0898a81c379e5ae2205e",
            "sizeBytes": 4000,
        },
        "argmax_float32": {
            "sha256": "1a9f08f4a607e9f1f45ee95da2d08fd753fac967073f081f6c3bb2d2d3c39d4c",
            "sizeBytes": 4002,
        },
    },
}


CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "arg_reduce.current-tree.translation.json"
)


def test_current_mlx_arg_reduce_contract_is_exact():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-arg-reduce-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 42,
        "entryCount": 17478,
        "discoveryDiagnosticCount": 0,
    }
    assert contract["entries"] == ["argmin_float32", "argmax_float32"]
    assert contract["artifacts"] == ARTIFACTS
    validation = contract["validationContract"]
    assert validation["workgroupSize"] == [32, 1, 1]
    assert validation["axisSizes"] == [32, 129]
    assert validation["axisStrides"] == [1, 2]
    assert validation["rowsPerCase"] == 2
    assert validation["ordinaryValues"] is True
    assert validation["nanAndInfinityValues"] is True
    assert all(
        target["runtimeRequiredInCi"] is True
        for target in validation["targets"].values()
    )
    assert contract["scope"] == {
        "coveredEntryCount": 2,
        "completeCorpusEntryCount": 17478,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(command, directory, name):
    result = subprocess.run(
        [str(arg) for arg in command],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=300,
    )
    _write_json(
        directory / f"{name}.json",
        {
            "command": result.args,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if not root_value or not target:
        message = "Set CROSTL_MLX_CURRENT_ROOT and CROSTL_MLX_CURRENT_TARGET"
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert target in {"directx", "opengl", "metal"}
    tool_names = {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]
    for name in tool_names:
        assert shutil.which(name), f"Required tool is missing: {name}"
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
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    return root, target


def _config(output, target, entry):
    source_options = """
[project.source_options.metal]
max_template_specializations = 128
max_template_materialization_work = 8192
"""
    if target == "directx":
        source_options += f"""
[project.subgroup_width_rules]
"{SOURCE}" = 32
[project.source_options.metal.target_options.directx]
relative_wave_shuffle_out_of_range = "self"
"""
    elif target == "opengl":
        source_options += """
[project.source_options.metal.target_options.opengl]
software_subgroup_width = 32
"""
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [32, 1, 1]
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "in_idx + current_index * axis_stride"
minimum = 0
maximum = 515
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "out_idx"
minimum = 0
maximum = 1
{source_options}
"""


def _metal_library(source, output, root):
    air = output.with_suffix(".air")
    _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.1",
            "-fno-fast-math",
            "-I",
            root,
            "-c",
            source,
            "-o",
            air,
        ],
        output.parent,
        output.stem + "-compile",
    )
    _run(["xcrun", "metallib", air, "-o", output], output.parent, output.stem + "-link")
    assert output.stat().st_size > 0
    return output


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("metal-reference")
    runner = work / "arg-reduce"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/mlx_arg_reduce_metal.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    library = _metal_library(root / SOURCE, work / "upstream.metallib", root)
    return runner, library


def _cases():
    for size in (32, 129):
        for stride in (1, 2):
            for special in (False, True):
                rows = []
                for row in range(2):
                    values = [
                        float((index * 17 + row * 3) % 31) for index in range(size)
                    ]
                    values[3] = values[-1] = 100.0
                    values[5] = values[-2] = -100.0
                    if special:
                        values[6] = values[14] = math.nan
                        values[2] = math.inf
                        values[1] = -math.inf
                    rows.append(values)
                # Padding must not become a candidate when axis_stride is greater than one.
                storage = [-1e20] * (2 * size * stride)
                for row, values in enumerate(rows):
                    for index, value in enumerate(values):
                        storage[row * size * stride + index * stride] = value
                request = {
                    "inputBits": [
                        struct.unpack("<I", struct.pack("<f", value))[0]
                        for value in storage
                    ],
                    "rows": 2,
                    "axisSize": size,
                    "axisStride": stride,
                    "rowStride": size * stride,
                }
                yield (
                    f"size-{size}-stride-{stride}-special-{int(special)}",
                    request,
                    rows,
                    storage,
                )


def _expected_indices(entry, rows):
    expected = []
    for row in rows:
        nan_indices = [index for index, value in enumerate(row) if math.isnan(value)]
        reduce = min if entry.startswith("argmin") else max
        expected.append(
            nan_indices[0]
            if nan_indices
            else reduce(range(len(row)), key=row.__getitem__)
        )
    return expected


def _runtime_binding_names(target, entry):
    if target == "directx":
        return {
            "input": "in_",
            "output": "out_",
            "shape": "shape",
            "inStrides": "in_strides",
            "outStrides": "out_strides",
            "ndim": f"{entry}_ndim_Constants",
            "axisStride": f"{entry}_axis_stride_Constants",
            "axisSize": f"{entry}_axis_size_Constants",
        }
    return {
        "input": "in_Buffer",
        "output": "out_Buffer",
        "shape": "shapeBuffer",
        "inStrides": "in_stridesBuffer",
        "outStrides": "out_stridesBuffer",
        "ndim": f"{entry}_ndim_Args",
        "axisStride": f"{entry}_axis_stride_Args",
        "axisSize": f"{entry}_axis_size_Args",
    }


def _runtime_descriptor(translation, work, target, entry):
    report_path = work / "portability-report.json"
    translation.write_json(report_path)
    artifacts = build_runtime_artifact_manifest(report_path)
    assert artifacts["success"] is True, json.dumps(artifacts, indent=2)
    assert artifacts["summary"]["artifactCount"] == 1
    artifacts_path = work / "runtime-artifacts.json"
    _write_json(artifacts_path, artifacts)
    package_dir = work / "runtime-package"
    package = build_runtime_package(artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)
    loader = build_runtime_loader_manifest(package_dir / "runtime-package.json")
    assert loader["success"] is True, json.dumps(loader, indent=2)
    assert loader["summary"]["loadUnitCount"] == 1
    assert loader["summary"]["readyLoadUnitCount"] == 1
    assert loader["summary"]["blockedLoadUnitCount"] == 0
    descriptor = build_native_loader_abi_descriptor(
        loader,
        load_unit_id=loader["loadUnits"][0]["id"],
    )
    assert descriptor["target"] == target
    assert descriptor["entryPoint"]["name"] == (
        "CSMain" if target == "directx" else "main"
    )
    names = _runtime_binding_names(target, entry)
    expected_types = {
        names["input"]: "float32",
        names["output"]: "uint32",
        names["shape"]: "int32",
        names["inStrides"]: "int64",
        names["outStrides"]: "int64",
        names["ndim"]: "uint64",
        names["axisStride"]: "int64",
        names["axisSize"]: "uint64",
    }
    reflected_types = {
        binding["name"]: binding["scalarLayout"]["elementType"]
        for binding in descriptor["bindings"]
        if binding["name"] in expected_types
    }
    assert reflected_types == expected_types
    return descriptor, package_dir


def _runtime_executor(target):
    if target == "directx":
        adapter = DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
    else:
        adapter = OpenGLRuntimeParityAdapter(
            runtime=OpenGLComputeRuntime(context_backends=("egl",))
        )
    return RuntimeParityExecutor(
        RuntimeTestAdapterSpec(
            adapter_id=f"mlx-current-arg-reduce-{target}",
            target=target,
            executor=target,
            adapter_kind=f"{target}-native-runtime",
        ),
        runtime_adapter=adapter,
    )


def _runtime_dispatch_request(
    descriptor,
    package_dir,
    target,
    entry,
    request,
    storage,
    expected,
):
    names = _runtime_binding_names(target, entry)
    encoded_storage = []
    for value in storage:
        if math.isnan(value):
            encoded_storage.append("nan")
        elif value == math.inf:
            encoded_storage.append("+infinity")
        elif value == -math.inf:
            encoded_storage.append("-infinity")
        else:
            encoded_storage.append(value)
    dispatch = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        {
            names["input"]: {
                "dtype": "float32",
                "shape": [len(storage)],
                "values": encoded_storage,
            },
            names["shape"]: {
                "dtype": "int32",
                "shape": [1],
                "values": [request["rows"]],
            },
            names["inStrides"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [request["rowStride"]],
            },
            names["outStrides"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [1],
            },
            names["ndim"]: {
                "dtype": "uint64",
                "shape": [1],
                "values": [1],
            },
            names["axisStride"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [request["axisStride"]],
            },
            names["axisSize"]: {
                "dtype": "uint64",
                "shape": [1],
                "values": [request["axisSize"]],
            },
        },
        {
            names["output"]: {
                "dtype": "uint32",
                "shape": [request["rows"]],
                "values": expected,
            }
        },
        [1, request["rows"], 1],
        expected_target=target,
    )
    assert dispatch.execution_plan is not None
    assert dispatch.execution_plan.diagnostics == ()
    assert dispatch.execution_plan.dispatch.workgroup_size == (32, 1, 1)
    assert dispatch.execution_plan.dispatch.workgroup_count == (
        1,
        request["rows"],
        1,
    )
    return dispatch, names["output"]


def _run_runtime_parity(translation, work, target, entry):
    descriptor, package_dir = _runtime_descriptor(
        translation,
        work,
        target,
        entry,
    )
    executor = _runtime_executor(target)
    evidence = []
    for name, request, rows, storage in _cases():
        expected = _expected_indices(entry, rows)
        dispatch, output_name = _runtime_dispatch_request(
            descriptor,
            package_dir,
            target,
            entry,
            request,
            storage,
            expected,
        )
        availability = executor.is_available(dispatch)
        if not availability.available:
            if os.environ.get(REQUIRE_RUNTIME_ENV) == "1":
                pytest.fail(
                    availability.reason or f"The native {target} runtime is unavailable"
                )
            return []
        result = executor.run(dispatch)
        assert result.status == "ok"
        assert result.outputs[output_name]["dtype"] == "uint32"
        assert result.outputs[output_name]["shape"] == [request["rows"]]
        assert result.outputs[output_name]["values"] == expected
        evidence.append(
            {
                "case": name,
                "expected": expected,
                "translated": result.outputs[output_name]["values"],
            }
        )
    return evidence


@pytest.mark.parametrize("entry", ["argmin_float32", "argmax_float32"])
def test_current_mlx_arg_reduce_native_validation(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    with tempfile.TemporaryDirectory(
        prefix=".current-arg-reduce-", dir=root
    ) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(_config(f"{work.name}/out", target, entry), encoding="utf-8")
        translation = translate_project(
            load_project_config(root, config),
            format_output=False,
            validate=True,
            run_toolchains=False,
        )
        report = translation.to_json()
        report_path = tmp_path / "report.json"
        translation.write_json(report_path)
        assert validate_project_report(report_path)["success"] is True
        assert report["diagnostics"] == []
        assert report["summary"]["translatedCount"] == 1
        assert report["summary"]["failedCount"] == 0
        (artifact,) = report["artifacts"]
        assert artifact["sourceHash"] == {
            "algorithm": "sha256",
            "value": SOURCE_SHA256,
        }
        assert artifact["generatedHash"] == {
            "algorithm": "sha256",
            "value": ARTIFACTS[target][entry]["sha256"],
        }
        assert artifact["generatedSizeBytes"] == ARTIFACTS[target][entry]["sizeBytes"]
        assert artifact["templateMaterialization"]["status"] == "materialized"
        assert artifact["templateMaterialization"]["specializationCount"] == 5
        generated = root / artifact["path"]
        output = tmp_path / generated.name
        shutil.copyfile(generated, output)
        assert (
            hashlib.sha256(output.read_bytes()).hexdigest()
            == ARTIFACTS[target][entry]["sha256"]
        )
        generated_text = output.read_text(encoding="utf-8")
        assert "unsupported" not in generated_text
        (execution,) = artifact["execution"]["entryPoints"]
        assert execution["workgroupSize"] == [32, 1, 1]
        if target == "directx":
            _run(
                [
                    "dxc",
                    "-T",
                    "cs_6_6",
                    "-E",
                    "CSMain",
                    output,
                    "-Fo",
                    tmp_path / "kernel.dxil",
                ],
                tmp_path,
                "dxc",
            )
            evidence = _run_runtime_parity(translation, work, target, entry)
            if evidence:
                _write_json(
                    tmp_path / "parity.json",
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "cases": evidence,
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                )
        elif target == "opengl":
            binary = tmp_path / "kernel.spv"
            _run(
                [
                    "glslangValidator",
                    "--target-env",
                    "opengl",
                    "--target-env",
                    "spirv1.3",
                    "-S",
                    "comp",
                    output,
                    "-o",
                    binary,
                ],
                tmp_path,
                "glslang",
            )
            _run(
                ["spirv-val", "--target-env", "spv1.3", binary],
                tmp_path,
                "spirv-val",
            )
            evidence = _run_runtime_parity(translation, work, target, entry)
            if evidence:
                _write_json(
                    tmp_path / "parity.json",
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "cases": evidence,
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                )
        else:
            assert "current_in += axis_stride;" in generated_text
            assert "isnan_float" not in generated_text
            library = _metal_library(output, tmp_path / "translated.metallib", root)
            runner, original = metal_reference
            evidence = []
            for name, request, rows, _storage in _cases():
                request_path = tmp_path / f"{name}.json"
                _write_json(request_path, request)
                expected = _expected_indices(entry, rows)
                observed = {}
                for label, compiled in (
                    ("upstream", original),
                    ("translated", library),
                ):
                    result = json.loads(
                        _run(
                            [runner, compiled, entry, request_path],
                            tmp_path,
                            f"{name}-{label}",
                        )
                    )
                    observed[label] = result
                    assert result["threadExecutionWidth"] == 32
                    assert result["indices"] == expected, (
                        name,
                        label,
                        result,
                        expected,
                    )
                evidence.append({"case": name, "expected": expected, **observed})
            _write_json(
                tmp_path / "parity.json",
                {
                    "commit": MLX_COMMIT,
                    "entry": entry,
                    "cases": evidence,
                    "upstreamMlxTestSuiteExecuted": False,
                },
            )
