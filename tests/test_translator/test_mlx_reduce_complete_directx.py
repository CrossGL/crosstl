from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.project import (
    build_runtime_artifact_manifest,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from tests.test_translator.test_mlx_reduce_complete_metal_roundtrip import (
    ENTRY_CLASSIFICATION_FIELDS,
    EXPECTED_INPUT_TYPE_COUNTS,
    EXPECTED_OPERATOR_COUNTS,
    EXPECTED_OPERATOR_TYPE_COUNTS,
    EXPECTED_OUTPUT_TYPE_COUNTS,
    EXPECTED_SHAPE_COUNTS,
    EXPECTED_TEMPLATE_COUNTS,
    MLX_COMMIT,
    MLX_REDUCE_SHA256,
    MLX_REDUCE_SOURCE,
    REDUCE_METAL_CONTRACT,
    REDUCE_METAL_ENTRY_CLASSIFICATION_SHA256,
    ReduceMetalWorkload,
    _canonical_json_sha256,
    _normalized_materialization,
)

REQUIRE_REDUCE_DIRECTX_ENV = "CROSTL_REQUIRE_MLX_REDUCE_DIRECTX_TRANSLATION"
REDUCE_DIRECTX_SHARD_INDEX_ENV = "CROSTL_MLX_REDUCE_DIRECTX_SHARD_INDEX"
REDUCE_DIRECTX_SHARD_COUNT_ENV = "CROSTL_MLX_REDUCE_DIRECTX_SHARD_COUNT"
REDUCE_DIRECTX_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
REDUCE_DIRECTX_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "reduce.directx-translation.json"
)
REDUCE_DIRECTX_CONTRACT_SHA256 = (
    "aea3e2639d5f46f1cc4af679c79b83eb23c6184e2687b1998b23bb24a3f184ad"
)
REDUCE_DIRECTX_CONTRACT_SIZE_BYTES = 1795438
REDUCE_DIRECTX_SPECIALIZATION_COUNT = 9216
REDUCE_DIRECTX_REFLECTED_RESOURCE_COUNT = 27382
REDUCE_DIRECTX_GENERATED_SIZE_BYTES_TOTAL = 36431375
REDUCE_DIRECTX_DXIL_SIZE_BYTES_TOTAL = 24322760
REDUCE_DIRECTX_GENERATED_SIZE_MINIMUM = ("init_reduce_minbool_", 1616)
REDUCE_DIRECTX_GENERATED_SIZE_MAXIMUM = (
    "row_reduce_looped_large_5_reduce_mincomplex64",
    30312,
)
REDUCE_DIRECTX_DXIL_SIZE_MINIMUM = ("init_reduce_andbool_", 2888)
REDUCE_DIRECTX_DXIL_SIZE_MAXIMUM = (
    "row_reduce_small_large_5_reduce_mincomplex64",
    18780,
)
DXC_SHA256 = "766ebfe2bd172074aa82c2e48f9f9ecffe57abd16f689b26f632337e67fa33a4"
RESOURCE_ABI_FIELDS = ("name", "kind", "type", "set", "binding", "access")
ENTRY_FIELDS = (
    "entryPoint",
    "shape",
    "inputType",
    "outputType",
    "operator",
    "sha256",
    "sizeBytes",
    "dxilSha256",
    "dxilSizeBytes",
    "specializationCount",
    "materializationSha256",
    "resourceCount",
    "resourcesSha256",
)
HLSL_STORAGE_TYPES = {
    "bfloat16_t": "uint16_t",
    "bool": "bool",
    "complex64_t": "complex_t_float",
    "float": "float",
    "float16_t": "float16_t",
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
    "shape": "StructuredBuffer<int>",
    "strides": "StructuredBuffer<int64_t>",
    "reduce_shape": "StructuredBuffer<int>",
    "reduce_strides": "StructuredBuffer<int64_t>",
}


@dataclass(frozen=True)
class ReduceDirectXWorkload(ReduceMetalWorkload):
    dxil_sha256: str
    dxil_size_bytes: int


def _load_contract(path: Path) -> dict:
    payload = path.read_bytes()
    assert len(payload) == REDUCE_DIRECTX_CONTRACT_SIZE_BYTES
    assert hashlib.sha256(payload).hexdigest() == REDUCE_DIRECTX_CONTRACT_SHA256
    return json.loads(payload)


REDUCE_DIRECTX_CONTRACT = _load_contract(REDUCE_DIRECTX_CONTRACT_PATH)
REDUCE_DIRECTX_ENTRIES = tuple(REDUCE_DIRECTX_CONTRACT["entries"])
REDUCE_DIRECTX_WORKLOADS = tuple(
    ReduceDirectXWorkload(
        entry_point=entry["entryPoint"],
        shape=entry["shape"],
        input_type=entry["inputType"],
        output_type=entry["outputType"],
        operator=entry["operator"],
        sha256=entry["sha256"],
        size_bytes=entry["sizeBytes"],
        specialization_count=entry["specializationCount"],
        materialization_sha256=entry["materializationSha256"],
        resource_count=entry["resourceCount"],
        resources_sha256=entry["resourcesSha256"],
        dxil_sha256=entry["dxilSha256"],
        dxil_size_bytes=entry["dxilSizeBytes"],
    )
    for entry in REDUCE_DIRECTX_ENTRIES
)


def _target_shape_contracts() -> dict:
    contracts = json.loads(json.dumps(REDUCE_METAL_CONTRACT["shapeContracts"]))
    for shape, contract in contracts.items():
        resources = contract["hostResources"]
        for resource in resources:
            source_name = str(resource["name"])
            resource["sourceName"] = source_name
            if source_name == "in_":
                resource["type"] = "StructuredBuffer<{inputHlslType}>"
            elif source_name == "out_":
                resource["type"] = "RWStructuredBuffer<{outputHlslType}>"
            elif source_name in ARRAY_RESOURCE_TYPES:
                resource["kind"] = "buffer"
                resource["type"] = ARRAY_RESOURCE_TYPES[source_name]
            else:
                target_name = f"{{sanitizedEntryPoint}}_{source_name}_Constants"
                resource["name"] = target_name
                resource["type"] = target_name
        if shape not in {"init", "all"}:
            resources.append(
                {
                    "name": "CrossGLDispatchInfo",
                    "kind": "constant-buffer",
                    "set": 0,
                    "binding": max(int(item["binding"]) for item in resources) + 1,
                    "access": "read",
                    "type": "CrossGLDispatchInfo",
                    "sourceName": "generated-dispatch-workgroup-count",
                }
            )
            contract["hostResourceCountPerArtifact"] += 1
    return contracts


def _classification_sha256(entries: tuple[dict, ...]) -> str:
    classifications = [
        {field: entry[field] for field in ENTRY_CLASSIFICATION_FIELDS}
        for entry in entries
    ]
    return _canonical_json_sha256(classifications)


def test_current_mlx_reduce_directx_contract_is_complete_and_classified() -> None:
    contract = REDUCE_DIRECTX_CONTRACT
    assert set(contract) == {
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
    }
    assert contract["schemaVersion"] == 2
    assert contract["kind"] == "crosstl-mlx-reduce-directx-translation-contract"
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_REDUCE_SOURCE
    assert contract["sourceSha256"] == MLX_REDUCE_SHA256
    assert contract["target"] == "directx"
    assert contract["selection"] == REDUCE_METAL_CONTRACT["selection"]
    assert contract["shapeContracts"] == _target_shape_contracts()
    assert contract["classifications"] == REDUCE_METAL_CONTRACT["classifications"]
    assert contract["classifications"]["templates"] == EXPECTED_TEMPLATE_COUNTS
    assert contract["classifications"]["shapes"] == EXPECTED_SHAPE_COUNTS
    assert contract["classifications"]["operators"] == EXPECTED_OPERATOR_COUNTS
    assert contract["classifications"]["operatorTypes"] == EXPECTED_OPERATOR_TYPE_COUNTS
    assert contract["classifications"]["inputTypes"] == EXPECTED_INPUT_TYPE_COUNTS
    assert contract["classifications"]["outputTypes"] == EXPECTED_OUTPUT_TYPE_COUNTS

    entries = REDUCE_DIRECTX_ENTRIES
    assert len(entries) == 2396
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert len({entry["entryPoint"] for entry in entries}) == 2396
    assert len({entry["sha256"] for entry in entries}) == 2396
    assert all(list(entry) == list(ENTRY_FIELDS) for entry in entries)
    assert _classification_sha256(entries) == REDUCE_METAL_ENTRY_CLASSIFICATION_SHA256
    assert Counter(entry["shape"] for entry in entries) == EXPECTED_SHAPE_COUNTS
    assert (
        Counter(entry["inputType"] for entry in entries) == EXPECTED_INPUT_TYPE_COUNTS
    )
    assert (
        Counter(entry["outputType"] for entry in entries) == EXPECTED_OUTPUT_TYPE_COUNTS
    )
    assert (
        Counter(entry["operator"] for entry in entries) == EXPECTED_OPERATOR_TYPE_COUNTS
    )
    assert all(len(entry["dxilSha256"]) == 64 for entry in entries)
    assert all(entry["dxilSizeBytes"] > 4 for entry in entries)
    assert sum(entry["specializationCount"] for entry in entries) == (
        REDUCE_DIRECTX_SPECIALIZATION_COUNT
    )
    assert sum(entry["resourceCount"] for entry in entries) == (
        REDUCE_DIRECTX_REFLECTED_RESOURCE_COUNT
    )

    artifact = contract["artifactContract"]
    assert artifact["artifactCount"] == 2396
    assert artifact["artifactCountPerEntry"] == 1
    assert artifact["nativeArtifactCount"] == 2396
    assert artifact["nativeArtifactCountPerEntry"] == 1
    assert artifact["preservedProofArtifactCount"] == 4792
    assert artifact["specializationCount"] == REDUCE_DIRECTX_SPECIALIZATION_COUNT
    assert artifact["unsupportedSpecializationCount"] == 0
    assert artifact["reachableKernelCountPerArtifact"] == 1
    assert artifact["provenance"] == "entry-scoped-translate"
    assert artifact["intermediate"] == "crossgl"
    assert artifact["hostInterfaceStatus"] == "ready"
    assert artifact["targetEntryPoint"] == "CSMain"
    assert artifact["reflectedResourceCount"] == REDUCE_DIRECTX_REFLECTED_RESOURCE_COUNT
    assert artifact["resourceAbiFields"] == list(RESOURCE_ABI_FIELDS)
    assert artifact["exactMaterializationDigestsIncluded"] is True
    assert artifact["exactResourceDigestsIncluded"] is True
    assert artifact["exactTargetResourceTypesIncluded"] is True
    assert artifact["exactHlslIdentitiesIncluded"] is True
    assert artifact["exactDxilIdentitiesIncluded"] is True
    assert artifact["hostDispatchWorkgroupSize"] == [1, 1, 1]
    assert artifact["dispatchMetadataShapeCount"] == 37
    assert artifact["generatedSizeBytesTotal"] == (
        REDUCE_DIRECTX_GENERATED_SIZE_BYTES_TOTAL
    )
    assert artifact["generatedSizeRange"] == {
        "minimum": {
            "entryPoint": REDUCE_DIRECTX_GENERATED_SIZE_MINIMUM[0],
            "sizeBytes": REDUCE_DIRECTX_GENERATED_SIZE_MINIMUM[1],
        },
        "maximum": {
            "entryPoint": REDUCE_DIRECTX_GENERATED_SIZE_MAXIMUM[0],
            "sizeBytes": REDUCE_DIRECTX_GENERATED_SIZE_MAXIMUM[1],
        },
    }
    assert artifact["dxilSizeBytesTotal"] == REDUCE_DIRECTX_DXIL_SIZE_BYTES_TOTAL
    assert artifact["dxilSizeRange"] == {
        "minimum": {
            "entryPoint": REDUCE_DIRECTX_DXIL_SIZE_MINIMUM[0],
            "sizeBytes": REDUCE_DIRECTX_DXIL_SIZE_MINIMUM[1],
        },
        "maximum": {
            "entryPoint": REDUCE_DIRECTX_DXIL_SIZE_MAXIMUM[0],
            "sizeBytes": REDUCE_DIRECTX_DXIL_SIZE_MAXIMUM[1],
        },
    }
    assert artifact["nativeCompiler"] == (
        "dxc -enable-16bit-types -WX -T cs_6_2 -E CSMain"
    )
    assert artifact["nativeCompilerSha256"] == DXC_SHA256
    assert artifact["compilerArguments"] == ["-enable-16bit-types"]
    assert artifact["requiresNonemptyDxilArtifact"] is True
    assert artifact["independentNativeRecompileCount"] == 2396
    assert artifact["independentDxilByteMatchCount"] == 2396


def _shard_parameters():
    raw_index = os.environ.get(REDUCE_DIRECTX_SHARD_INDEX_ENV)
    raw_count = os.environ.get(REDUCE_DIRECTX_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return None
    if raw_index is None or raw_count is None:
        pytest.fail(
            f"{REDUCE_DIRECTX_SHARD_INDEX_ENV} and "
            f"{REDUCE_DIRECTX_SHARD_COUNT_ENV} must be configured together"
        )
    try:
        index = int(raw_index)
        count = int(raw_count)
    except ValueError:
        pytest.fail("reduce DirectX shard values must be integers")
    if count != REDUCE_DIRECTX_CI_SHARD_COUNT or not 0 <= index < count:
        pytest.fail("reduce DirectX shard configuration is invalid")
    return index, count


def _current_workloads() -> tuple[ReduceDirectXWorkload, ...]:
    shard = _shard_parameters()
    if shard is None:
        return REDUCE_DIRECTX_WORKLOADS
    index, count = shard
    return REDUCE_DIRECTX_WORKLOADS[index::count]


CURRENT_REDUCE_DIRECTX_WORKLOADS = _current_workloads()


def test_current_mlx_reduce_directx_ci_shards_are_disjoint_and_complete() -> None:
    shards = [
        REDUCE_DIRECTX_WORKLOADS[index::REDUCE_DIRECTX_CI_SHARD_COUNT]
        for index in range(REDUCE_DIRECTX_CI_SHARD_COUNT)
    ]
    assert [len(shard) for shard in shards] == [100] * 20 + [99] * 4
    assert sum((list(shard) for shard in shards), []) != list(REDUCE_DIRECTX_WORKLOADS)
    assert {workload.entry_point for shard in shards for workload in shard} == {
        workload.entry_point for workload in REDUCE_DIRECTX_WORKLOADS
    }


def _project_config(workload: ReduceDirectXWorkload) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_REDUCE_SOURCE}"]
        include_dirs = ["."]
        targets = ["directx"]
        output_dir = "out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_REDUCE_SOURCE}" = "{workload.entry_point}"

        [project.entry_workgroup_size_rules."{MLX_REDUCE_SOURCE}"]
        "{workload.entry_point}" = [1, 1, 1]

        [project.source_options.metal]
        max_template_specializations = 1024
        max_template_materialization_work = 1048576
        """).strip()


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_REDUCE_DIRECTX_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")
    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_REDUCE_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX reduce source is missing: {source_path}")
    checkout = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checkout.returncode == 0, checkout.stderr
    assert checkout.stdout.strip() == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_REDUCE_SHA256
    return mlx_root


def _required_tool(name: str) -> str:
    path = shutil.which(name)
    if path is not None:
        return path
    message = f"{name} is required for the complete MLX reduce DirectX proof"
    if os.environ.get(REQUIRE_REDUCE_DIRECTX_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _normalized_resources(resources: list[dict]) -> list[dict]:
    return [
        {field: resource[field] for field in RESOURCE_ABI_FIELDS}
        for resource in resources
    ]


def _expected_resources(
    workload: ReduceDirectXWorkload,
) -> dict[str, tuple[object, ...]]:
    substitutions = {
        "sanitizedEntryPoint": workload.entry_point.rstrip("_"),
        "inputHlslType": HLSL_STORAGE_TYPES[workload.input_type],
        "outputHlslType": HLSL_STORAGE_TYPES[workload.output_type],
    }
    resources = {}
    shape = REDUCE_DIRECTX_CONTRACT["shapeContracts"][workload.shape]
    for resource in shape["hostResources"]:
        name = str(resource["name"]).format(**substitutions)
        type_name = str(resource["type"]).format(**substitutions)
        resources[name] = (
            resource["kind"],
            type_name,
            resource["set"],
            resource["binding"],
            resource["access"],
        )
    return resources


def _translate_and_validate(
    mlx_root: Path,
    work_dir: Path,
    workload: ReduceDirectXWorkload,
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
    summary = payload["summary"]
    assert summary["unitCount"] == 1
    assert summary["artifactCount"] == 1
    assert summary["translatedCount"] == 1
    assert summary["failedCount"] == 0
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
    assert artifact["source"] == MLX_REDUCE_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_REDUCE_SHA256,
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
    execution = artifact["execution"]["entryPoints"]
    assert len(execution) == 1
    assert execution[0]["sourceEntryPoint"] == workload.entry_point
    assert execution[0]["materializedEntryPoint"] == workload.entry_point
    assert execution[0]["targetEntryPoint"] == "CSMain"
    assert execution[0]["workgroupSize"] == [1, 1, 1]
    materialization = _normalized_materialization(artifact["templateMaterialization"])
    assert materialization["specializationCount"] == workload.specialization_count
    assert _canonical_json_sha256(materialization) == workload.materialization_sha256
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "[numthreads(1, 1, 1)]" in generated
    assert generated.count("void CSMain(") == 1
    for residue in (
        "template <",
        "decltype(",
        "operator()",
        "unsupported Metal",
        "fallback for unmatched generated control flow",
    ):
        assert residue not in generated
    HLSLParser(HLSLLexer(generated).tokenize()).parse()

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    runtime = build_runtime_artifact_manifest(report_path)
    assert runtime["success"] is True, json.dumps(runtime, indent=2)
    host = runtime["artifacts"][0]["hostInterface"]
    assert host["status"] == "ready"
    assert host["entryPoints"] == [
        {
            "name": "CSMain",
            "stage": "compute",
            "executionConfig": {"numthreads": [1, 1, 1]},
        }
    ]
    resources = _normalized_resources(host["resources"])
    assert len(resources) == workload.resource_count
    assert _canonical_json_sha256(resources) == workload.resources_sha256
    assert {
        resource["name"]: (
            resource["kind"],
            resource["type"],
            resource["set"],
            resource["binding"],
            resource["access"],
        )
        for resource in resources
    } == _expected_resources(workload)

    compiler_arguments = dxc_compiler_arguments_for_source(generated)
    assert compiler_arguments == ("-enable-16bit-types",)
    dxil_path = work_dir / f"{workload.entry_point}.dxil"
    compilation = subprocess.run(
        [
            _required_tool("dxc"),
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
        timeout=180,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    dxil = dxil_path.read_bytes()
    assert dxil.startswith(b"DXBC")
    assert len(dxil) > 4


@pytest.mark.parametrize(
    "workload",
    CURRENT_REDUCE_DIRECTX_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_reduce_family_translates_to_directx(
    workload: ReduceDirectXWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-reduce-directx-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_and_validate(mlx_root, Path(temporary_directory), workload)
