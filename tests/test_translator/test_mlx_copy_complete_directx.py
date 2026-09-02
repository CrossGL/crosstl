from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from crosstl.project import (
    build_runtime_artifact_manifest,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from tests.test_translator.test_mlx_copy_complete_metal_roundtrip import (
    COPY_METAL_CONTRACT,
    EXPECTED_COPY_CLASSIFICATIONS,
    MLX_COMMIT,
    MLX_COPY_SHA256,
    MLX_COPY_SOURCE,
    CopyMetalWorkload,
    _expected_materializations,
    _expected_specialization_count,
)

REQUIRE_COPY_DIRECTX_ENV = "CROSTL_REQUIRE_MLX_COPY_DIRECTX_TRANSLATION"
COPY_DIRECTX_SHARD_INDEX_ENV = "CROSTL_MLX_COPY_DIRECTX_SHARD_INDEX"
COPY_DIRECTX_SHARD_COUNT_ENV = "CROSTL_MLX_COPY_DIRECTX_SHARD_COUNT"
COPY_DIRECTX_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
COPY_DIRECTX_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "copy.directx-translation.json"
)
COPY_DIRECTX_CONTRACT_SHA256 = (
    "d0a2d946437fbb08826ebd2a6c981ec45e524ad2f85292c304ae9db383674473"
)
COPY_DIRECTX_CONTRACT_SIZE_BYTES = 876824
COPY_DIRECTX_GENERATED_SIZE_BYTES_TOTAL = 4462390
COPY_DIRECTX_GENERATED_SIZE_MINIMUM = ("s_copybool_bool_", 1201)
COPY_DIRECTX_GENERATED_SIZE_MAXIMUM = (
    "ggn4large_dynamic_copybfloat16bfloat16",
    3798,
)
CONTRACT_RESOURCE_ABI_FIELDS = (
    "name",
    "kind",
    "set",
    "binding",
    "access",
    "type",
)
HLSL_STORAGE_TYPES = {
    "bfloat16_t": "uint16_t",
    "bool": "bool",
    "complex64_t": "complex_t_float",
    "float": "float",
    "half": "float16_t",
    "int16_t": "int16_t",
    "int32_t": "int",
    "int64_t": "int64_t",
    "int8_t": "int",
    "uint16_t": "uint16_t",
    "uint32_t": "uint",
    "uint64_t": "uint64_t",
    "uint8_t": "uint",
}
ARRAY_RESOURCE_TYPES = {
    "src_shape": "StructuredBuffer<int>",
    "src_strides": "StructuredBuffer<int64_t>",
    "dst_strides": "StructuredBuffer<int64_t>",
}
TARGET_RESOURCE_NAMES = {
    "src": "src",
    "dst": "dst",
    "size": "{sanitizedEntryPoint}_size_Constants",
    "src_stride": "{sanitizedEntryPoint}_src_stride_Constants",
    "dst_stride": "{sanitizedEntryPoint}_dst_stride_Constants",
    "src_shape": "src_shape",
    "src_strides": "src_strides",
    "dst_strides": "dst_strides",
    "ndim": "{sanitizedEntryPoint}_ndim_Constants",
    "src_offset": "{sanitizedEntryPoint}_src_offset_Constants",
    "dst_offset": "{sanitizedEntryPoint}_dst_offset_Constants",
}
DISPATCH_BINDINGS = {
    "g2": 0,
    "g2large": 0,
    "g3": 0,
    "g3large": 0,
    "gn2": 6,
    "gn4large": 6,
    "s2": 3,
    "v2": 3,
}


def _load_contract(path: Path, expected_sha256: str) -> dict:
    contract_bytes = path.read_bytes()
    assert len(contract_bytes) == COPY_DIRECTX_CONTRACT_SIZE_BYTES
    assert hashlib.sha256(contract_bytes).hexdigest() == expected_sha256
    return json.loads(contract_bytes)


COPY_DIRECTX_CONTRACT = _load_contract(
    COPY_DIRECTX_CONTRACT_PATH,
    COPY_DIRECTX_CONTRACT_SHA256,
)
COPY_DIRECTX_ENTRIES = tuple(COPY_DIRECTX_CONTRACT["entries"])
COPY_DIRECTX_WORKLOADS = tuple(
    CopyMetalWorkload(
        entry_point=entry["entryPoint"],
        shape=entry["shape"],
        template_name=entry["templateName"],
        input_type=entry["inputType"],
        output_type=entry["outputType"],
        family=entry["family"],
        sha256=entry["sha256"],
        size_bytes=entry["sizeBytes"],
    )
    for entry in COPY_DIRECTX_ENTRIES
)


def _partition_workloads(
    workloads: tuple[CopyMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[CopyMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX copy DirectX shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX copy DirectX shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX copy DirectX shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_workloads() -> tuple[CopyMetalWorkload, ...]:
    raw_index = os.environ.get(COPY_DIRECTX_SHARD_INDEX_ENV)
    raw_count = os.environ.get(COPY_DIRECTX_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return COPY_DIRECTX_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{COPY_DIRECTX_SHARD_INDEX_ENV} and {COPY_DIRECTX_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError("MLX copy DirectX shard values must be integers") from error
    try:
        return _partition_workloads(
            COPY_DIRECTX_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_COPY_DIRECTX_WORKLOADS = _current_workloads()


def _target_shape_contracts() -> dict[str, object]:
    contracts = json.loads(json.dumps(COPY_METAL_CONTRACT["shapeContracts"]))
    for shape, shape_contract in contracts.items():
        for resource in shape_contract["hostResources"]:
            source_name = resource["name"]
            target_name = TARGET_RESOURCE_NAMES[source_name]
            resource["sourceName"] = source_name
            resource["name"] = target_name
            if source_name == "src":
                resource["type"] = "StructuredBuffer<{inputHlslType}>"
            elif source_name == "dst":
                resource["type"] = "RWStructuredBuffer<{outputHlslType}>"
            elif source_name in ARRAY_RESOURCE_TYPES:
                resource["kind"] = "buffer"
                resource["type"] = ARRAY_RESOURCE_TYPES[source_name]
            else:
                resource["type"] = target_name
        if shape in DISPATCH_BINDINGS:
            shape_contract["hostResourceCountPerArtifact"] += 1
            shape_contract["hostResources"].append(
                {
                    "name": "CrossGLDispatchInfo",
                    "kind": "constant-buffer",
                    "set": 0,
                    "binding": DISPATCH_BINDINGS[shape],
                    "access": "read",
                    "type": "CrossGLDispatchInfo",
                    "sourceName": "generated-dispatch-workgroup-count",
                }
            )
    return contracts


def test_current_mlx_copy_directx_contract_is_complete_and_classified() -> None:
    contract = COPY_DIRECTX_CONTRACT
    assert list(contract) == [
        "schemaVersion",
        "kind",
        "commit",
        "source",
        "sourceSha256",
        "target",
        "selection",
        "shapeContracts",
        "classifications",
        "artifactContract",
        "entries",
    ]
    assert contract["schemaVersion"] == 2
    assert contract["kind"] == "crosstl-mlx-copy-directx-translation-contract"
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_COPY_SOURCE
    assert contract["sourceSha256"] == MLX_COPY_SHA256
    assert contract["target"] == "directx"
    assert contract["selection"] == COPY_METAL_CONTRACT["selection"]
    assert contract["classifications"] == EXPECTED_COPY_CLASSIFICATIONS
    assert contract["classifications"] == COPY_METAL_CONTRACT["classifications"]
    expected_shapes = _target_shape_contracts()
    assert contract["shapeContracts"] == expected_shapes
    resource_counts = {
        shape: value["entryCount"] * value["hostResourceCountPerArtifact"]
        for shape, value in expected_shapes.items()
    }
    assert contract["artifactContract"] == {
        "artifactCount": 2496,
        "artifactCountPerEntry": 1,
        "specializationCount": 6566,
        "specializationCountsByShape": COPY_METAL_CONTRACT["artifactContract"][
            "specializationCountsByShape"
        ],
        "explicitCastSpecializationCount": 2496,
        "nestedComplexBoolSpecializationCount": 14,
        "registeredComplexScalarProjectionCount": 150,
        "unsupportedSpecializationCount": 0,
        "reachableKernelCountPerArtifact": 1,
        "provenance": "entry-scoped-translate",
        "intermediate": "crossgl",
        "hostInterfaceStatus": "ready",
        "targetEntryPoint": "CSMain",
        "reflectedResourceCount": 10036,
        "reflectedResourceCountsByShape": resource_counts,
        "resourceAbiFields": list(CONTRACT_RESOURCE_ABI_FIELDS),
        "exactTargetResourceTypesIncluded": True,
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "dispatchMetadataShapeCount": 8,
        "generatedSizeBytesTotal": COPY_DIRECTX_GENERATED_SIZE_BYTES_TOTAL,
        "generatedSizeRange": {
            "minimum": {
                "entryPoint": COPY_DIRECTX_GENERATED_SIZE_MINIMUM[0],
                "sizeBytes": COPY_DIRECTX_GENERATED_SIZE_MINIMUM[1],
            },
            "maximum": {
                "entryPoint": COPY_DIRECTX_GENERATED_SIZE_MAXIMUM[0],
                "sizeBytes": COPY_DIRECTX_GENERATED_SIZE_MAXIMUM[1],
            },
        },
        "nativeCompiler": "dxc -enable-16bit-types -WX -T cs_6_2 -E CSMain",
        "compilerArguments": ["-enable-16bit-types"],
        "requiresNonemptyDxilArtifact": True,
    }

    entries = COPY_DIRECTX_ENTRIES
    assert len(entries) == 2496
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert len({entry["entryPoint"] for entry in entries}) == 2496
    assert Counter(entry["shape"] for entry in entries) == (
        EXPECTED_COPY_CLASSIFICATIONS["shapes"]
    )
    assert Counter(entry["templateName"] for entry in entries) == (
        EXPECTED_COPY_CLASSIFICATIONS["templates"]
    )
    assert Counter(entry["inputType"] for entry in entries) == (
        EXPECTED_COPY_CLASSIFICATIONS["inputTypes"]
    )
    assert Counter(entry["outputType"] for entry in entries) == (
        EXPECTED_COPY_CLASSIFICATIONS["outputTypes"]
    )
    assert (
        Counter(f'{entry["inputType"]}->{entry["outputType"]}' for entry in entries)
        == EXPECTED_COPY_CLASSIFICATIONS["typePairs"]
    )
    assert Counter(entry["family"] for entry in entries) == (
        EXPECTED_COPY_CLASSIFICATIONS["families"]
    )
    assert sum(entry["sizeBytes"] for entry in entries) == (
        COPY_DIRECTX_GENERATED_SIZE_BYTES_TOTAL
    )
    assert min((entry["sizeBytes"], entry["entryPoint"]) for entry in entries) == (
        COPY_DIRECTX_GENERATED_SIZE_MINIMUM[1],
        COPY_DIRECTX_GENERATED_SIZE_MINIMUM[0],
    )
    assert max((entry["sizeBytes"], entry["entryPoint"]) for entry in entries) == (
        COPY_DIRECTX_GENERATED_SIZE_MAXIMUM[1],
        COPY_DIRECTX_GENERATED_SIZE_MAXIMUM[0],
    )

    metal_entries = {
        entry["entryPoint"]: entry for entry in COPY_METAL_CONTRACT["entries"]
    }
    identity_fields = (
        "entryPoint",
        "shape",
        "templateName",
        "inputType",
        "outputType",
        "family",
    )
    artifact_hashes = set()
    for entry in entries:
        assert list(entry) == [
            "entryPoint",
            "shape",
            "templateName",
            "inputType",
            "outputType",
            "family",
            "sha256",
            "sizeBytes",
        ]
        assert {key: entry[key] for key in identity_fields} == {
            key: metal_entries[entry["entryPoint"]][key] for key in identity_fields
        }
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
        assert entry["sha256"] not in artifact_hashes
        artifact_hashes.add(entry["sha256"])
        assert entry["sizeBytes"] > 0


def test_current_mlx_copy_directx_ci_shards_are_complete_and_disjoint() -> None:
    shards = tuple(
        _partition_workloads(
            COPY_DIRECTX_WORKLOADS,
            shard_index,
            COPY_DIRECTX_CI_SHARD_COUNT,
        )
        for shard_index in range(COPY_DIRECTX_CI_SHARD_COUNT)
    )
    assert [len(shard) for shard in shards] == [104] * 24
    for shard_index, shard in enumerate(shards):
        assert shard == COPY_DIRECTX_WORKLOADS[shard_index::COPY_DIRECTX_CI_SHARD_COUNT]
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 2496
    assert len(set(entry_points)) == 2496
    assert set(entry_points) == {
        workload.entry_point for workload in COPY_DIRECTX_WORKLOADS
    }


def _project_config(workload: CopyMetalWorkload) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_COPY_SOURCE}"]
        include_dirs = ["."]
        targets = ["directx"]
        output_dir = "out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_COPY_SOURCE}" = "{workload.entry_point}"

        [project.entry_workgroup_size_rules."{MLX_COPY_SOURCE}"]
        "{workload.entry_point}" = [1, 1, 1]

        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 131072
        """).strip()


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_COPY_DIRECTX_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")
    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_COPY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX copy source is missing: {source_path}")
    checkout = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checkout.returncode == 0, checkout.stderr
    assert checkout.stdout.strip() == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_COPY_SHA256
    return mlx_root


def _required_tool(name: str) -> str:
    path = shutil.which(name)
    if path is not None:
        return path
    message = f"{name} is required for the complete MLX copy DirectX proof"
    if os.environ.get(REQUIRE_COPY_DIRECTX_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _expected_resources(workload: CopyMetalWorkload) -> dict[str, tuple[object, ...]]:
    sanitized_entry = workload.entry_point.rstrip("_")
    substitutions = {
        "sanitizedEntryPoint": sanitized_entry,
        "inputHlslType": HLSL_STORAGE_TYPES[workload.input_type],
        "outputHlslType": HLSL_STORAGE_TYPES[workload.output_type],
    }
    resources = {}
    for resource in COPY_DIRECTX_CONTRACT["shapeContracts"][workload.shape][
        "hostResources"
    ]:
        name = str(resource["name"]).format(**substitutions)
        type_name = str(resource["type"]).format(**substitutions)
        resources[name] = (
            resource["kind"],
            resource["set"],
            resource["binding"],
            resource["access"],
            type_name,
        )
    return resources


def _selected_cast_helper(generated: str, workload: CopyMetalWorkload) -> str:
    helper_name = f"cast_to_{workload.output_type}_{workload.input_type}"
    match = re.search(
        rf"(?ms)^[^\n;]*\b{re.escape(helper_name)}\([^;\n]*\)\s*\{{\n"
        r"(?P<body>.*?)^\}",
        generated,
    )
    assert match is not None, helper_name
    return match.group("body")


def _translate_and_validate(
    mlx_root: Path,
    work_dir: Path,
    workload: CopyMetalWorkload,
) -> None:
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(_project_config(workload) + "\n", encoding="utf-8")
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=("directx",),
        output_dir=(work_dir / "out").relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    payload = report.to_json()
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["artifactCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    if shutil.which("dxc") is None:
        assert len(payload["diagnostics"]) == 1
        diagnostic = payload["diagnostics"][0]
        assert diagnostic["code"] == "project.validate.toolchain-unavailable"
        assert diagnostic["severity"] == "warning"
        assert diagnostic["target"] == "directx"
        assert diagnostic["missingCapabilities"] == ["toolchain.validation"]
    else:
        assert payload["diagnostics"] == []
    assert payload["project"]["indexRangeAssertions"] == []

    artifact = payload["artifacts"][0]
    assert artifact["source"] == MLX_COPY_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_COPY_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": workload.sha256,
    }
    assert artifact["generatedSizeBytes"] == workload.size_bytes
    assert artifact["entryPoint"] == {
        "source": workload.entry_point,
        "target": "CSMain",
        "stage": "compute",
    }
    assert artifact["provenance"] == {
        "pipeline": "entry-scoped-translate",
        "intermediate": "crossgl",
    }
    execution_entries = artifact["execution"]["entryPoints"]
    assert len(execution_entries) == 1
    assert execution_entries[0]["sourceEntryPoint"] == workload.entry_point
    assert execution_entries[0]["materializedEntryPoint"] == workload.entry_point
    assert execution_entries[0]["targetEntryPoint"] == "CSMain"
    assert execution_entries[0]["workgroupSize"] == [1, 1, 1]
    materialization = artifact["templateMaterialization"]
    expected_materializations = _expected_materializations(workload)
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == _expected_specialization_count(
        workload
    )
    assert materialization["specializations"] == expected_materializations
    assert materialization["unsupported"] == []
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "[numthreads(1, 1, 1)]" in generated
    assert generated.count("void CSMain(") == 1
    cast_helper = _selected_cast_helper(generated, workload)
    if workload.input_type == "complex64_t" and workload.output_type == "bool":
        assert ".real" in cast_helper
        assert ".imag" in cast_helper
        assert "cast_to_bool_float" in cast_helper
    elif workload.input_type == "complex64_t" and workload.output_type != "complex64_t":
        assert cast_helper.count(".real") == 1
        assert not re.search(r"return\s+[A-Za-z0-9_]+\(\s*val\s*\)", cast_helper)
    for residue in (
        "template <",
        "decltype(",
        "operator()",
        "unsupported Metal",
        "fallback for unmatched generated control flow",
    ):
        assert residue not in generated

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(
        runtime_artifacts,
        indent=2,
    )
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    reflected = runtime_artifacts["artifacts"][0]["hostInterface"]
    assert reflected["status"] == "ready"
    assert reflected["entryPoints"] == [
        {
            "name": "CSMain",
            "stage": "compute",
            "executionConfig": {"numthreads": [1, 1, 1]},
        }
    ]
    assert {
        resource["name"]: (
            resource["kind"],
            resource["set"],
            resource["binding"],
            resource["access"],
            resource["type"],
        )
        for resource in reflected["resources"]
    } == _expected_resources(workload)

    dxc = _required_tool("dxc")
    compiler_arguments = dxc_compiler_arguments_for_source(generated)
    assert compiler_arguments == ("-enable-16bit-types",)
    dxil_path = work_dir / f"{workload.entry_point}.dxil"
    compilation = subprocess.run(
        [
            dxc,
            *compiler_arguments,
            "-WX",
            "-T",
            "cs_6_2",
            "-E",
            "CSMain",
            str(generated_path),
            "-Fo",
            str(dxil_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    assert dxil_path.is_file()
    dxil_bytes = dxil_path.read_bytes()
    assert dxil_bytes.startswith(b"DXBC")
    assert len(dxil_bytes) > 4


@pytest.mark.parametrize(
    "workload",
    CURRENT_COPY_DIRECTX_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_copy_family_translates_to_directx(
    workload: CopyMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-copy-directx-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_and_validate(
            mlx_root,
            Path(temporary_directory),
            workload,
        )
