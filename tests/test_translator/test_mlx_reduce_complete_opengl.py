from __future__ import annotations

import hashlib
import json
import os
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
from tests.test_translator.test_mlx_reduce_complete_metal_roundtrip import (
    ENTRY_CLASSIFICATION_FIELDS,
    EXPECTED_DIMENSION_COUNTS,
    EXPECTED_INDEX_TYPE_COUNTS,
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

REQUIRE_REDUCE_OPENGL_ENV = "CROSTL_REQUIRE_MLX_REDUCE_OPENGL_TRANSLATION"
REDUCE_OPENGL_SHARD_INDEX_ENV = "CROSTL_MLX_REDUCE_OPENGL_SHARD_INDEX"
REDUCE_OPENGL_SHARD_COUNT_ENV = "CROSTL_MLX_REDUCE_OPENGL_SHARD_COUNT"
REDUCE_OPENGL_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
REDUCE_OPENGL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "reduce.opengl-translation.json"
)
REDUCE_OPENGL_CONTRACT_SHA256 = (
    "56688f3037f1898b5e8dcced5e6e29cb0c1225d9deef4d31ceea29734bca1b9c"
)
REDUCE_OPENGL_CONTRACT_SIZE_BYTES = 1497997
REDUCE_OPENGL_SPECIALIZATION_COUNT = 9216
REDUCE_OPENGL_REFLECTED_RESOURCE_COUNT = 25088
REDUCE_OPENGL_GENERATED_SIZE_BYTES_TOTAL = 31155122
REDUCE_OPENGL_GENERATED_SIZE_MINIMUM = ("init_reduce_minbool_", 1326)
REDUCE_OPENGL_GENERATED_SIZE_MAXIMUM = (
    "col_reduce_2pass_large_5_32_32_reduce_mincomplex64",
    27130,
)
INDEX_RANGE_ASSERTIONS = (
    ("in_ + LoopedElemToLoc_1_int64_t_false__location(loop)", 0, 2**31 - 1),
    ("in_ + LoopedElemToLoc_2_int64_t_false__location(loop)", 0, 2**31 - 1),
    ("in_ + LoopedElemToLoc_5_int64_t_true__location(loop)", 0, 2**31 - 1),
    ("inputs[i - 1] + reduction_size", 0, 2**31 - 1),
    (
        "gid.z * int64(out_size) + out_idx * int64(reduction_stride) + lid.x",
        0,
        2**31 - 1,
    ),
    ("out_idx", 0, 2**31 - 1),
)
RESOURCE_ABI_FIELDS = (
    "name",
    "kind",
    "type",
    "set",
    "binding",
    "access",
    "scalarLayout",
)
ENTRY_FIELDS = (
    "entryPoint",
    "shape",
    "inputType",
    "outputType",
    "operator",
    "sha256",
    "sizeBytes",
    "specializationCount",
    "materializationSha256",
    "resourceCount",
    "resourcesSha256",
)


def _load_contract(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    assert len(payload) == REDUCE_OPENGL_CONTRACT_SIZE_BYTES
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    return json.loads(payload)


REDUCE_OPENGL_CONTRACT = _load_contract(
    REDUCE_OPENGL_CONTRACT_PATH,
    REDUCE_OPENGL_CONTRACT_SHA256,
)
REDUCE_OPENGL_ENTRIES = tuple(REDUCE_OPENGL_CONTRACT["entries"])
REDUCE_OPENGL_WORKLOADS = tuple(
    ReduceMetalWorkload(
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
    )
    for entry in REDUCE_OPENGL_ENTRIES
)


def _target_resource_name(source_name: str, source_type: str) -> tuple[str, str]:
    if source_name in {"in_", "out_"}:
        return f"{source_name.rstrip('_')}_Buffer", "buffer"
    if "*" in source_type:
        return f"{source_name}Buffer", "buffer"
    return f"{{sanitizedEntryPoint}}_{source_name}_Args", "constant-buffer"


def _target_shape_contracts() -> dict[str, object]:
    contracts = json.loads(json.dumps(REDUCE_METAL_CONTRACT["shapeContracts"]))
    for contract in contracts.values():
        for resource in contract["hostResources"]:
            source_name = resource["name"]
            target_name, target_kind = _target_resource_name(
                source_name,
                resource["type"],
            )
            resource["sourceName"] = source_name
            resource["name"] = target_name
            resource["kind"] = target_kind
    return contracts


def _expected_assertions() -> list[dict[str, object]]:
    return [
        {
            "source": MLX_REDUCE_SOURCE,
            "expression": expression,
            "minimum": minimum,
            "maximum": maximum,
        }
        for expression, minimum, maximum in INDEX_RANGE_ASSERTIONS
    ]


def _normalized_resource_abi(resource: dict[str, object]) -> dict[str, object]:
    required_fields = set(RESOURCE_ABI_FIELDS) - {"scalarLayout"}
    missing_fields = required_fields - set(resource)
    assert not missing_fields
    return {field: resource.get(field) for field in RESOURCE_ABI_FIELDS}


def test_current_mlx_reduce_opengl_contract_is_complete_and_classified() -> None:
    contract = REDUCE_OPENGL_CONTRACT
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
        "portabilityPreconditions",
        "artifactContract",
        "entries",
    }
    assert contract["schemaVersion"] == 2
    assert contract["kind"] == "mlx-reduce-opengl-translation-contract"
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_REDUCE_SOURCE
    assert contract["sourceSha256"] == MLX_REDUCE_SHA256
    assert contract["target"] == "opengl"
    assert contract["selection"] == REDUCE_METAL_CONTRACT["selection"]
    assert contract["shapeContracts"] == _target_shape_contracts()
    assert contract["classifications"] == REDUCE_METAL_CONTRACT["classifications"]
    assert contract["classifications"]["templates"] == EXPECTED_TEMPLATE_COUNTS
    assert contract["classifications"]["shapes"] == EXPECTED_SHAPE_COUNTS
    assert contract["classifications"]["operators"] == EXPECTED_OPERATOR_COUNTS
    assert contract["classifications"]["operatorTypes"] == EXPECTED_OPERATOR_TYPE_COUNTS
    assert contract["classifications"]["inputTypes"] == EXPECTED_INPUT_TYPE_COUNTS
    assert contract["classifications"]["outputTypes"] == EXPECTED_OUTPUT_TYPE_COUNTS
    assert contract["classifications"]["indexTypes"] == EXPECTED_INDEX_TYPE_COUNTS
    assert contract["classifications"]["dimensions"] == EXPECTED_DIMENSION_COUNTS
    assert contract["portabilityPreconditions"] == {
        "indexRangeAssertions": _expected_assertions(),
        "contractKind": "explicit-host-runtime-portability-preconditions",
        "inferred": False,
        "runtimeEnforced": False,
    }

    entries = REDUCE_OPENGL_ENTRIES
    assert len(entries) == 2396
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert len({entry["entryPoint"] for entry in entries}) == 2396
    assert len({entry["sha256"] for entry in entries}) == 2396
    assert Counter(entry["shape"] for entry in entries) == EXPECTED_SHAPE_COUNTS
    assert Counter(entry["inputType"] for entry in entries) == (
        EXPECTED_INPUT_TYPE_COUNTS
    )
    assert Counter(entry["outputType"] for entry in entries) == (
        EXPECTED_OUTPUT_TYPE_COUNTS
    )
    assert Counter(entry["operator"] for entry in entries) == (
        EXPECTED_OPERATOR_TYPE_COUNTS
    )
    assert (
        _canonical_json_sha256(
            [
                {field: entry[field] for field in ENTRY_CLASSIFICATION_FIELDS}
                for entry in entries
            ]
        )
        == REDUCE_METAL_ENTRY_CLASSIFICATION_SHA256
    )
    for entry in entries:
        assert tuple(entry) == ENTRY_FIELDS
        assert entry["sizeBytes"] > 0
        assert entry["specializationCount"] > 0
        assert entry["resourceCount"] > 0
        for field in ("sha256", "materializationSha256", "resourcesSha256"):
            assert len(entry[field]) == 64
            int(entry[field], 16)

    artifact = contract["artifactContract"]
    assert set(artifact) == {
        "artifactCount",
        "artifactCountPerEntry",
        "specializationCount",
        "specializationCountsByShape",
        "unsupportedSpecializationCount",
        "reachableKernelCountPerArtifact",
        "provenance",
        "intermediate",
        "hostInterfaceStatus",
        "reflectedResourceCount",
        "reflectedResourceCountsByShape",
        "resourceAbiFields",
        "exactMaterializationDigestsIncluded",
        "exactResourceDigestsIncluded",
        "hostDispatchWorkgroupSize",
        "generatedSizeBytesTotal",
        "generatedSizeRange",
        "nativeCompiler",
        "targetEntryPoint",
        "nativeValidator",
        "requiresNonemptySpirvArtifact",
    }
    assert artifact["artifactCount"] == 2396
    assert artifact["artifactCountPerEntry"] == 1
    assert artifact["specializationCount"] == REDUCE_OPENGL_SPECIALIZATION_COUNT
    assert artifact["specializationCount"] == sum(
        entry["specializationCount"] for entry in entries
    )
    assert artifact["specializationCountsByShape"] == {
        shape: sum(
            entry["specializationCount"] for entry in entries if entry["shape"] == shape
        )
        for shape in sorted(EXPECTED_SHAPE_COUNTS)
    }
    assert artifact["unsupportedSpecializationCount"] == 0
    assert artifact["reachableKernelCountPerArtifact"] == 1
    assert artifact["provenance"] == "entry-scoped-translate"
    assert artifact["intermediate"] == "crossgl"
    assert artifact["hostInterfaceStatus"] == "ready"
    assert artifact["reflectedResourceCount"] == REDUCE_OPENGL_REFLECTED_RESOURCE_COUNT
    assert artifact["reflectedResourceCount"] == sum(
        entry["resourceCount"] for entry in entries
    )
    assert artifact["reflectedResourceCountsByShape"] == {
        shape: sum(
            entry["resourceCount"] for entry in entries if entry["shape"] == shape
        )
        for shape in sorted(EXPECTED_SHAPE_COUNTS)
    }
    assert artifact["resourceAbiFields"] == list(RESOURCE_ABI_FIELDS)
    assert artifact["exactMaterializationDigestsIncluded"] is True
    assert artifact["exactResourceDigestsIncluded"] is True
    assert artifact["hostDispatchWorkgroupSize"] == [1, 1, 1]
    assert (
        artifact["generatedSizeBytesTotal"] == REDUCE_OPENGL_GENERATED_SIZE_BYTES_TOTAL
    )
    assert artifact["generatedSizeBytesTotal"] == sum(
        entry["sizeBytes"] for entry in entries
    )
    minimum = min(entries, key=lambda row: (row["sizeBytes"], row["entryPoint"]))
    maximum = max(entries, key=lambda row: (row["sizeBytes"], row["entryPoint"]))
    assert (minimum["entryPoint"], minimum["sizeBytes"]) == (
        REDUCE_OPENGL_GENERATED_SIZE_MINIMUM
    )
    assert (maximum["entryPoint"], maximum["sizeBytes"]) == (
        REDUCE_OPENGL_GENERATED_SIZE_MAXIMUM
    )
    assert artifact["generatedSizeRange"] == {
        "minimum": {
            "entryPoint": minimum["entryPoint"],
            "sizeBytes": minimum["sizeBytes"],
        },
        "maximum": {
            "entryPoint": maximum["entryPoint"],
            "sizeBytes": maximum["sizeBytes"],
        },
    }
    assert artifact["nativeCompiler"] == (
        "glslangValidator --target-env opengl --target-env spirv1.3 -S comp"
    )
    assert artifact["targetEntryPoint"] == "main"
    assert artifact["nativeValidator"] == "spirv-val --target-env spv1.3"
    assert artifact["requiresNonemptySpirvArtifact"] is True


def test_reduce_opengl_resource_digest_detects_scalar_layout_drift() -> None:
    resource = {
        "name": "in_Buffer",
        "kind": "buffer",
        "type": "in_Buffer",
        "set": 0,
        "binding": 0,
        "access": "read",
        "scalarLayout": {
            "physicalType": "float",
            "elementType": "float32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 4,
            "memberOffsetBytes": 0,
            "storageLayout": "std430",
            "runtimeSized": True,
            "memberName": "in_",
        },
    }
    expected = _canonical_json_sha256([resource])
    wrong = json.loads(json.dumps(resource))
    wrong["scalarLayout"]["elementStrideBytes"] = 8
    assert _canonical_json_sha256([wrong]) != expected

    aggregate = dict(resource)
    del aggregate["scalarLayout"]
    normalized_aggregate = _normalized_resource_abi(aggregate)
    assert normalized_aggregate["scalarLayout"] is None
    assert _canonical_json_sha256([normalized_aggregate]) != expected

    missing_type = dict(resource)
    del missing_type["type"]
    with pytest.raises(AssertionError):
        _normalized_resource_abi(missing_type)


def _partition_workloads(
    workloads: tuple[ReduceMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[ReduceMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX reduce OpenGL shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX reduce OpenGL shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX reduce OpenGL shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_workloads() -> tuple[ReduceMetalWorkload, ...]:
    raw_index = os.environ.get(REDUCE_OPENGL_SHARD_INDEX_ENV)
    raw_count = os.environ.get(REDUCE_OPENGL_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return REDUCE_OPENGL_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{REDUCE_OPENGL_SHARD_INDEX_ENV} and {REDUCE_OPENGL_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError("MLX reduce OpenGL shard values must be integers") from error
    try:
        return _partition_workloads(
            REDUCE_OPENGL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_REDUCE_OPENGL_WORKLOADS = _current_workloads()


def test_current_mlx_reduce_opengl_ci_shards_are_complete_and_disjoint() -> None:
    shards = tuple(
        _partition_workloads(
            REDUCE_OPENGL_WORKLOADS,
            shard_index,
            REDUCE_OPENGL_CI_SHARD_COUNT,
        )
        for shard_index in range(REDUCE_OPENGL_CI_SHARD_COUNT)
    )
    assert [len(shard) for shard in shards] == [100] * 20 + [99] * 4
    for shard_index, shard in enumerate(shards):
        assert (
            shard == REDUCE_OPENGL_WORKLOADS[shard_index::REDUCE_OPENGL_CI_SHARD_COUNT]
        )
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 2396
    assert len(set(entry_points)) == 2396
    assert set(entry_points) == {
        workload.entry_point for workload in REDUCE_OPENGL_WORKLOADS
    }


def _project_config(workload: ReduceMetalWorkload) -> str:
    assertions = "\n\n".join(textwrap.dedent(f"""
            [[project.index_range_assertions]]
            source = "{MLX_REDUCE_SOURCE}"
            expression = "{expression}"
            minimum = {minimum}
            maximum = {maximum}
            """).strip() for expression, minimum, maximum in INDEX_RANGE_ASSERTIONS)
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_REDUCE_SOURCE}"]
        include_dirs = ["."]
        targets = ["opengl"]
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

        {assertions}
        """).strip()


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_REDUCE_OPENGL_ENV) == "1":
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
    message = f"{name} is required for the complete MLX reduce OpenGL proof"
    if os.environ.get(REQUIRE_REDUCE_OPENGL_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _expected_resources(workload: ReduceMetalWorkload) -> dict[str, tuple]:
    sanitized_entry = workload.entry_point.rstrip("_")
    resources = {}
    shape_contract = REDUCE_OPENGL_CONTRACT["shapeContracts"][workload.shape]
    for resource in shape_contract["hostResources"]:
        name = resource["name"].replace("{sanitizedEntryPoint}", sanitized_entry)
        resources[name] = (
            resource["kind"],
            resource["set"],
            resource["binding"],
            resource["access"],
        )
    return resources


def _translate_and_validate(
    mlx_root: Path,
    work_dir: Path,
    workload: ReduceMetalWorkload,
) -> None:
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(_project_config(workload) + "\n", encoding="utf-8")
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=("opengl",),
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
    assert payload["diagnostics"] == []
    assert payload["project"]["indexRangeAssertions"] == _expected_assertions()
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
        "target": "main",
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
    assert execution[0]["targetEntryPoint"] == "main"
    assert execution[0]["workgroupSize"] == [1, 1, 1]
    materialization = _normalized_materialization(artifact["templateMaterialization"])
    assert materialization["specializationCount"] == workload.specialization_count
    assert _canonical_json_sha256(materialization) == workload.materialization_sha256
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert generated.count("void main()") == 1
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
    runtime = build_runtime_artifact_manifest(report_path)
    assert runtime["success"] is True, json.dumps(runtime, indent=2)
    host = runtime["artifacts"][0]["hostInterface"]
    assert host["status"] == "ready"
    assert host["entryPoints"] == [
        {
            "name": "main",
            "stage": "compute",
            "executionConfig": {
                "local_size_x": 1,
                "local_size_y": 1,
                "local_size_z": 1,
                "local_size": [1, 1, 1],
            },
        }
    ]
    resources = [_normalized_resource_abi(resource) for resource in host["resources"]]
    assert len(resources) == workload.resource_count
    assert _canonical_json_sha256(resources) == workload.resources_sha256
    assert {
        resource["name"]: (
            resource["kind"],
            resource["set"],
            resource["binding"],
            resource["access"],
        )
        for resource in resources
    } == _expected_resources(workload)

    spirv_path = work_dir / f"{workload.entry_point}.spv"
    compilation = subprocess.run(
        [
            _required_tool("glslangValidator"),
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
            str(generated_path),
            "-o",
            str(spirv_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    assert spirv_path.is_file()
    assert spirv_path.stat().st_size > 0
    validation = subprocess.run(
        [
            _required_tool("spirv-val"),
            "--target-env",
            "spv1.3",
            str(spirv_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr


@pytest.mark.parametrize(
    "workload",
    CURRENT_REDUCE_OPENGL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_reduce_family_translates_to_opengl(
    workload: ReduceMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-reduce-opengl-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_and_validate(
            mlx_root,
            Path(temporary_directory),
            workload,
        )
