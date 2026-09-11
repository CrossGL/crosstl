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

from crosstl.project import (
    build_runtime_artifact_manifest,
    load_project_config,
    translate_project,
    validate_project_report,
)

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_REDUCE_SOURCE = "mlx/backend/metal/kernels/reduce.metal"
MLX_REDUCE_SHA256 = "f1e410ab635eaa940ec195461069cbb013ab88ab0d1f8d5dd8790d30b32c454a"
REQUIRE_REDUCE_METAL_ENV = "CROSTL_REQUIRE_MLX_REDUCE_METAL_ROUNDTRIP"
REDUCE_METAL_SHARD_INDEX_ENV = "CROSTL_MLX_REDUCE_METAL_SHARD_INDEX"
REDUCE_METAL_SHARD_COUNT_ENV = "CROSTL_MLX_REDUCE_METAL_SHARD_COUNT"
REDUCE_METAL_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
REDUCE_METAL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "reduce.metal-roundtrip.json"
)
REDUCE_METAL_CONTRACT_SHA256 = (
    # Replaced by the guarded proof integrator.
    "0b00b118c0fc2137a60d28400abe3b48de1ccb7a27193273b7b83d3119330abc"
)
RESOURCE_ABI_FIELDS = ("name", "kind", "set", "binding", "access", "type")
REDUCE_METAL_TYPE_NAMES = {
    "bfloat16_t": "bfloat",
    "bool": "bool",
    "complex64_t": "complex_t_float",
    "float": "float",
    "float16_t": "half",
    "int16_t": "int",
    "int32_t": "int",
    "int64_t": "int64_t",
    "int8_t": "int",
    "uint16_t": "uint",
    "uint32_t": "uint",
    "uint64_t": "uint64_t",
    "uint8_t": "uint",
}
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
ENTRY_CLASSIFICATION_FIELDS = (
    "entryPoint",
    "shape",
    "inputType",
    "outputType",
    "operator",
)
REDUCE_METAL_ENTRY_CLASSIFICATION_SHA256 = (
    "b3aef57d816410e35d3016546de3e847042c9d6dac793000ada80b94a63dbd24"
)


@dataclass(frozen=True)
class ReduceMetalWorkload:
    entry_point: str
    shape: str
    input_type: str
    output_type: str
    operator: str
    sha256: str
    size_bytes: int
    specialization_count: int
    materialization_sha256: str
    resource_count: int
    resources_sha256: str


EXPECTED_TEMPLATE_COUNTS = {
    "all_reduce": 62,
    "col_reduce_2pass": 372,
    "col_reduce_longcolumn": 372,
    "col_reduce_looped": 372,
    "col_reduce_small": 372,
    "init_reduce": 40,
    "row_reduce_looped": 372,
    "row_reduce_simple": 62,
    "row_reduce_small": 372,
}
EXPECTED_OPERATOR_COUNTS = {
    "And": 267,
    "Max": 469,
    "Min": 469,
    "Or": 267,
    "Prod": 462,
    "Sum": 462,
}
EXPECTED_INPUT_TYPE_COUNTS = {
    "bfloat16_t": 232,
    "bool": 80,
    "complex64_t": 156,
    "float": 232,
    "float16_t": 232,
    "int16_t": 230,
    "int32_t": 232,
    "int64_t": 232,
    "int8_t": 154,
    "uint16_t": 154,
    "uint32_t": 154,
    "uint64_t": 154,
    "uint8_t": 154,
}
EXPECTED_OUTPUT_TYPE_COUNTS = {
    "bfloat16_t": 156,
    "bool": 536,
    "complex64_t": 156,
    "float": 156,
    "float16_t": 156,
    "int16_t": 78,
    "int32_t": 384,
    "int64_t": 156,
    "int8_t": 78,
    "uint16_t": 78,
    "uint32_t": 230,
    "uint64_t": 154,
    "uint8_t": 78,
}
EXPECTED_INDEX_TYPE_COUNTS = {"int": 1116, "int64_t": 1240, "none": 40}
EXPECTED_DIMENSION_COUNTS = {"1": 744, "2": 744, "5": 744, "none": 164}
EXPECTED_OPERATOR_TYPE_COUNTS = {
    "And<bool>": 267,
    "Max<bfloat16_t>": 39,
    "Max<bool>": 1,
    "Max<complex64_t>": 39,
    "Max<float16_t>": 39,
    "Max<float>": 39,
    "Max<int16_t>": 39,
    "Max<int32_t>": 39,
    "Max<int64_t>": 39,
    "Max<int8_t>": 39,
    "Max<uint16_t>": 39,
    "Max<uint32_t>": 39,
    "Max<uint64_t>": 39,
    "Max<uint8_t>": 39,
    "Min<bfloat16_t>": 39,
    "Min<bool>": 1,
    "Min<complex64_t>": 39,
    "Min<float16_t>": 39,
    "Min<float>": 39,
    "Min<int16_t>": 39,
    "Min<int32_t>": 39,
    "Min<int64_t>": 39,
    "Min<int8_t>": 39,
    "Min<uint16_t>": 39,
    "Min<uint32_t>": 39,
    "Min<uint64_t>": 39,
    "Min<uint8_t>": 39,
    "Or<bool>": 267,
    "Prod<bfloat16_t>": 39,
    "Prod<complex64_t>": 39,
    "Prod<float16_t>": 39,
    "Prod<float>": 39,
    "Prod<int32_t>": 153,
    "Prod<int64_t>": 39,
    "Prod<uint32_t>": 76,
    "Prod<uint64_t>": 38,
    "Sum<bfloat16_t>": 39,
    "Sum<complex64_t>": 39,
    "Sum<float16_t>": 39,
    "Sum<float>": 39,
    "Sum<int32_t>": 153,
    "Sum<int64_t>": 39,
    "Sum<uint32_t>": 76,
    "Sum<uint64_t>": 38,
}


def _resource_templates(template_name: str) -> list[dict[str, object]]:
    output = {
        "name": "out_",
        "kind": "buffer",
        "set": 0,
        "binding": 0 if template_name == "init_reduce" else 1,
        "access": "read_write",
        "type": "device {outputMetalType}*",
    }
    if template_name == "init_reduce":
        return [output]

    resources: list[dict[str, object]] = [
        {
            "name": "in_",
            "kind": "buffer",
            "set": 0,
            "binding": 0,
            "access": "read",
            "type": "const device {inputMetalType}*",
        },
        output,
    ]
    constant_parameters = {
        "all_reduce": (
            ("in_size", "constant uint64_t&"),
            ("row_size", "constant uint64_t&"),
        ),
        "row_reduce_simple": (
            ("reduction_size", "constant uint64_t&"),
            ("out_size", "constant int64_t&"),
        ),
        "row_reduce_small": (
            ("row_size", "constant int64_t&"),
            ("non_row_reductions", "constant int64_t&"),
            ("shape", "constant int*"),
            ("strides", "constant int64_t*"),
            ("ndim", "constant int&"),
            ("reduce_shape", "constant int*"),
            ("reduce_strides", "constant int64_t*"),
            ("reduce_ndim", "constant int&"),
        ),
        "row_reduce_looped": (
            ("row_size", "constant int64_t&"),
            ("non_row_reductions", "constant int64_t&"),
            ("shape", "constant int*"),
            ("strides", "constant int64_t*"),
            ("ndim", "constant int&"),
            ("reduce_shape", "constant int*"),
            ("reduce_strides", "constant int64_t*"),
            ("reduce_ndim", "constant int&"),
        ),
        "col_reduce_small": (
            ("reduction_size", "constant uint64_t&"),
            ("reduction_stride", "constant int64_t&"),
            ("shape", "constant int*"),
            ("strides", "constant int64_t*"),
            ("ndim", "constant int&"),
            ("reduce_shape", "constant int*"),
            ("reduce_strides", "constant int64_t*"),
            ("reduce_ndim", "constant int&"),
            ("non_col_reductions", "constant uint64_t&"),
        ),
        "col_reduce_longcolumn": (
            ("reduction_size", "constant uint64_t&"),
            ("reduction_stride", "constant uint64_t&"),
            ("shape", "constant int*"),
            ("strides", "constant int64_t*"),
            ("ndim", "constant int&"),
            ("reduce_shape", "constant int*"),
            ("reduce_strides", "constant int64_t*"),
            ("reduce_ndim", "constant int&"),
            ("non_col_reductions", "constant uint64_t&"),
            ("out_size", "constant uint64_t&"),
        ),
        "col_reduce_looped": (
            ("reduction_size", "constant uint64_t&"),
            ("reduction_stride", "constant int64_t&"),
            ("shape", "constant int*"),
            ("strides", "constant int64_t*"),
            ("ndim", "constant int&"),
            ("reduce_shape", "constant int*"),
            ("reduce_strides", "constant int64_t*"),
            ("reduce_ndim", "constant int&"),
            ("non_col_reductions", "constant uint64_t&"),
        ),
        "col_reduce_2pass": (
            ("reduction_size", "constant uint64_t&"),
            ("reduction_stride", "constant int64_t&"),
            ("shape", "constant int*"),
            ("strides", "constant int64_t*"),
            ("ndim", "constant int&"),
            ("reduce_shape", "constant int*"),
            ("reduce_strides", "constant int64_t*"),
            ("reduce_ndim", "constant int&"),
            ("non_col_reductions", "constant uint64_t&"),
            ("out_size", "constant uint64_t&"),
        ),
    }
    resources.extend(
        {
            "name": name,
            "kind": "constant-buffer",
            "set": 0,
            "binding": binding,
            "access": "read",
            "type": type_name,
        }
        for binding, (name, type_name) in enumerate(
            constant_parameters[template_name],
            start=2,
        )
    )
    return resources


def _resources(
    template_name: str,
    input_type: str,
    output_type: str,
) -> list[dict[str, object]]:
    substitutions = {
        "inputMetalType": REDUCE_METAL_TYPE_NAMES[input_type],
        "outputMetalType": REDUCE_METAL_TYPE_NAMES[output_type],
    }
    return [
        {**resource, "type": str(resource["type"]).format(**substitutions)}
        for resource in _resource_templates(template_name)
    ]


def _assert_exact_resources(
    actual: object,
    expected: list[dict[str, object]],
    entry_point: str,
) -> None:
    assert actual == expected, (
        f"exact source/reflected resource ABI mismatch for {entry_point}: "
        f"expected {expected!r}, got {actual!r}"
    )


def _expected_shape_contracts() -> dict[str, dict[str, object]]:
    contracts: dict[str, dict[str, object]] = {
        "init": {
            "entryCount": 40,
            "templateName": "init_reduce",
            "sourceTemplateArgumentCount": 2,
            "indexType": None,
            "dimensions": None,
            "tile": None,
        },
        "all": {
            "entryCount": 62,
            "templateName": "all_reduce",
            "sourceTemplateArgumentCount": 3,
            "indexType": "int64_t",
            "dimensions": None,
            "tile": None,
        },
        "row-simple": {
            "entryCount": 62,
            "templateName": "row_reduce_simple",
            "sourceTemplateArgumentCount": 3,
            "indexType": "int64_t",
            "dimensions": None,
            "tile": None,
        },
    }
    family_shapes = {
        "row-reduce-small": ("row_reduce_small", 5, None),
        "row-reduce-looped": ("row_reduce_looped", 5, None),
        "col-reduce-small": ("col_reduce_small", 5, None),
        "col-reduce-longcolumn": ("col_reduce_longcolumn", 5, None),
        "col-reduce-looped": ("col_reduce_looped", 7, [32, 32]),
        "col-reduce-2pass": ("col_reduce_2pass", 7, [32, 32]),
    }
    for prefix, (template_name, argument_count, tile) in family_shapes.items():
        for index_suffix, index_type in (("i32", "int"), ("i64", "int64_t")):
            for dimensions in (1, 2, 5):
                contracts[f"{prefix}-{index_suffix}-d{dimensions}"] = {
                    "entryCount": 62,
                    "templateName": template_name,
                    "sourceTemplateArgumentCount": argument_count,
                    "indexType": index_type,
                    "dimensions": dimensions,
                    "tile": tile,
                }
    for contract in contracts.values():
        resources = _resource_templates(str(contract["templateName"]))
        contract["hostResourceCountPerArtifact"] = len(resources)
        contract["hostResources"] = resources
    return {key: contracts[key] for key in sorted(contracts)}


EXPECTED_SHAPE_CONTRACTS = _expected_shape_contracts()
EXPECTED_SHAPE_COUNTS = {
    shape: contract["entryCount"]
    for shape, contract in EXPECTED_SHAPE_CONTRACTS.items()
}


def _contract(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    return json.loads(payload)


REDUCE_METAL_CONTRACT = _contract(
    REDUCE_METAL_CONTRACT_PATH,
    REDUCE_METAL_CONTRACT_SHA256,
)
REDUCE_METAL_ENTRIES = tuple(REDUCE_METAL_CONTRACT["entries"])
REDUCE_METAL_WORKLOADS = tuple(
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
    for entry in REDUCE_METAL_ENTRIES
)


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_current_mlx_reduce_metal_contract_is_complete_and_classified():
    contract = REDUCE_METAL_CONTRACT
    assert set(contract) == {
        "schemaVersion",
        "kind",
        "commit",
        "source",
        "sourceSha256",
        "target",
        "metalTypeNames",
        "selection",
        "shapeContracts",
        "classifications",
        "artifactContract",
        "entries",
    }
    assert contract["schemaVersion"] == 2
    assert contract["kind"] == "mlx-reduce-metal-roundtrip-contract"
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_REDUCE_SOURCE
    assert contract["sourceSha256"] == MLX_REDUCE_SHA256
    assert contract["target"] == "metal"
    assert contract["metalTypeNames"] == REDUCE_METAL_TYPE_NAMES
    assert contract["selection"] == {
        "entryCount": 2396,
        "shapeCount": 39,
        "templateCount": 9,
        "operatorCount": 6,
        "operatorTypeCount": 44,
        "inputTypeCount": 13,
        "outputTypeCount": 13,
        "allDiscoveredSourceInstantiationsIncluded": True,
    }
    shape_contracts = contract["shapeContracts"]
    assert shape_contracts == EXPECTED_SHAPE_CONTRACTS

    classifications = contract["classifications"]
    assert set(classifications) == {
        "templates",
        "shapes",
        "operators",
        "operatorTypes",
        "inputTypes",
        "outputTypes",
        "indexTypes",
        "dimensions",
    }
    assert classifications["templates"] == EXPECTED_TEMPLATE_COUNTS
    assert classifications["shapes"] == EXPECTED_SHAPE_COUNTS
    assert classifications["operators"] == EXPECTED_OPERATOR_COUNTS
    assert classifications["operatorTypes"] == EXPECTED_OPERATOR_TYPE_COUNTS
    assert classifications["inputTypes"] == EXPECTED_INPUT_TYPE_COUNTS
    assert classifications["outputTypes"] == EXPECTED_OUTPUT_TYPE_COUNTS
    assert classifications["indexTypes"] == EXPECTED_INDEX_TYPE_COUNTS
    assert classifications["dimensions"] == EXPECTED_DIMENSION_COUNTS

    entries = REDUCE_METAL_ENTRIES
    assert len(entries) == 2396
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert len({entry["entryPoint"] for entry in entries}) == 2396
    assert len({entry["sha256"] for entry in entries}) == 2396
    assert (
        _canonical_json_sha256(
            [
                {field: entry[field] for field in ENTRY_CLASSIFICATION_FIELDS}
                for entry in entries
            ]
        )
        == REDUCE_METAL_ENTRY_CLASSIFICATION_SHA256
    )
    assert Counter(entry["shape"] for entry in entries) == EXPECTED_SHAPE_COUNTS
    assert Counter(entry["inputType"] for entry in entries) == (
        EXPECTED_INPUT_TYPE_COUNTS
    )
    assert Counter(entry["outputType"] for entry in entries) == (
        EXPECTED_OUTPUT_TYPE_COUNTS
    )
    assert Counter(entry["operator"].split("<", 1)[0] for entry in entries) == (
        EXPECTED_OPERATOR_COUNTS
    )
    assert Counter(entry["operator"] for entry in entries) == (
        EXPECTED_OPERATOR_TYPE_COUNTS
    )
    for entry in entries:
        assert tuple(entry) == ENTRY_FIELDS
        assert len(entry["sha256"]) == 64
        assert len(entry["materializationSha256"]) == 64
        assert len(entry["resourcesSha256"]) == 64
        int(entry["sha256"], 16)
        int(entry["materializationSha256"], 16)
        int(entry["resourcesSha256"], 16)
        assert entry["sizeBytes"] > 0
        assert entry["specializationCount"] > 0
        shape_contract = shape_contracts[entry["shape"]]
        expected_resources = _resources(
            str(shape_contract["templateName"]),
            entry["inputType"],
            entry["outputType"],
        )
        assert entry["resourceCount"] == len(expected_resources)
        assert entry["resourceCount"] == shape_contract["hostResourceCountPerArtifact"]
        assert _canonical_json_sha256(expected_resources) == entry["resourcesSha256"]

    artifact_contract = contract["artifactContract"]
    assert set(artifact_contract) == {
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
        "reflectedResourceTypeCounts",
        "resourceAbiFields",
        "exactResourceDigestsIncluded",
        "exactSourceResourceReconstruction",
        "hostDispatchWorkgroupSize",
        "generatedSizeBytesTotal",
        "generatedSizeRange",
        "nativeCompiler",
        "requiresNonemptyAirArtifact",
    }
    assert artifact_contract["artifactCount"] == 2396
    assert artifact_contract["artifactCountPerEntry"] == 1
    assert artifact_contract["specializationCount"] == sum(
        entry["specializationCount"] for entry in entries
    )
    assert artifact_contract["specializationCountsByShape"] == {
        shape: sum(
            entry["specializationCount"] for entry in entries if entry["shape"] == shape
        )
        for shape in sorted(EXPECTED_SHAPE_CONTRACTS)
    }
    assert artifact_contract["unsupportedSpecializationCount"] == 0
    assert artifact_contract["reachableKernelCountPerArtifact"] == 1
    assert artifact_contract["provenance"] == "entry-scoped-translate"
    assert artifact_contract["intermediate"] == "crossgl"
    assert artifact_contract["hostInterfaceStatus"] == "ready"
    assert artifact_contract["reflectedResourceCount"] == sum(
        entry["resourceCount"] for entry in entries
    )
    assert artifact_contract["reflectedResourceCountsByShape"] == {
        shape: sum(
            entry["resourceCount"] for entry in entries if entry["shape"] == shape
        )
        for shape in sorted(EXPECTED_SHAPE_CONTRACTS)
    }
    assert artifact_contract["resourceAbiFields"] == list(RESOURCE_ABI_FIELDS)
    assert artifact_contract["exactResourceDigestsIncluded"] is True
    assert artifact_contract["exactSourceResourceReconstruction"] is True
    assert artifact_contract["hostDispatchWorkgroupSize"] == [1, 1, 1]
    assert artifact_contract["generatedSizeBytesTotal"] == sum(
        entry["sizeBytes"] for entry in entries
    )
    minimum = min(entries, key=lambda entry: (entry["sizeBytes"], entry["entryPoint"]))
    maximum = max(entries, key=lambda entry: (entry["sizeBytes"], entry["entryPoint"]))
    assert artifact_contract["generatedSizeRange"] == {
        "minimum": {
            "entryPoint": minimum["entryPoint"],
            "sizeBytes": minimum["sizeBytes"],
        },
        "maximum": {
            "entryPoint": maximum["entryPoint"],
            "sizeBytes": maximum["sizeBytes"],
        },
    }
    assert artifact_contract["nativeCompiler"] == ("xcrun -sdk macosx metal -Werror -c")
    assert artifact_contract["requiresNonemptyAirArtifact"] is True


@pytest.mark.parametrize(
    ("resource_name", "wrong_type"),
    (
        ("in_", "const device definitely_wrong*"),
        ("out_", "device also_wrong*"),
    ),
)
def test_reduce_metal_exact_resource_contract_rejects_wrong_pointee_types(
    resource_name,
    wrong_type,
):
    expected = _resources("all_reduce", "bfloat16_t", "bool")
    wrong = [dict(resource) for resource in expected]
    next(resource for resource in wrong if resource["name"] == resource_name)[
        "type"
    ] = wrong_type
    with pytest.raises(AssertionError, match="exact source/reflected resource ABI"):
        _assert_exact_resources(wrong, expected, "adversarial_reduce_entry")


def _partition_reduce_metal_workloads(
    workloads: tuple[ReduceMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[ReduceMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX reduce Metal shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX reduce Metal shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX reduce Metal shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_reduce_metal_workloads() -> tuple[ReduceMetalWorkload, ...]:
    raw_index = os.environ.get(REDUCE_METAL_SHARD_INDEX_ENV)
    raw_count = os.environ.get(REDUCE_METAL_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return REDUCE_METAL_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{REDUCE_METAL_SHARD_INDEX_ENV} and {REDUCE_METAL_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError("MLX reduce Metal shard values must be integers") from error
    try:
        return _partition_reduce_metal_workloads(
            REDUCE_METAL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_REDUCE_METAL_WORKLOADS = _current_reduce_metal_workloads()


def test_current_mlx_reduce_metal_ci_shards_are_complete_and_disjoint():
    shards = tuple(
        _partition_reduce_metal_workloads(
            REDUCE_METAL_WORKLOADS,
            shard_index,
            REDUCE_METAL_CI_SHARD_COUNT,
        )
        for shard_index in range(REDUCE_METAL_CI_SHARD_COUNT)
    )
    assert [len(shard) for shard in shards] == [100] * 20 + [99] * 4
    for shard_index, shard in enumerate(shards):
        assert shard == REDUCE_METAL_WORKLOADS[shard_index::REDUCE_METAL_CI_SHARD_COUNT]
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 2396
    assert len(set(entry_points)) == 2396
    assert set(entry_points) == {
        workload.entry_point for workload in REDUCE_METAL_WORKLOADS
    }


def _project_config(workload: ReduceMetalWorkload) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_REDUCE_SOURCE}"]
        include_dirs = ["."]
        targets = ["metal"]
        output_dir = ".crosstl-mlx-reduce-metal-roundtrip/out"

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
        if os.environ.get(REQUIRE_REDUCE_METAL_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")
    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_REDUCE_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX reduce source is missing: {source_path}")
    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_REDUCE_SHA256
    return mlx_root


def _normalized_materialization(materialization: dict) -> dict:
    result = {
        "status": materialization.get("status"),
        "specializationCount": materialization.get("specializationCount"),
        "specializations": materialization.get("specializations"),
        "unsupported": materialization.get("unsupported"),
        "configuredParameterCount": materialization.get("configuredParameterCount"),
        "configuredParameters": materialization.get("configuredParameters"),
        "configuredParameterSources": materialization.get("configuredParameterSources"),
    }
    assert result["status"] == "materialized"
    assert result["specializationCount"] == len(result["specializations"])
    assert result["unsupported"] == []
    assert result["configuredParameterCount"] == 0
    assert result["configuredParameters"] == {}
    assert result["configuredParameterSources"] == {}
    return result


def _translate_reduce_metal_artifact(
    mlx_root: Path,
    work_dir: Path,
    workload: ReduceMetalWorkload,
) -> tuple[Path, Path]:
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(_project_config(workload) + "\n", encoding="utf-8")
    output_dir = work_dir / "out"
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=("metal",),
        output_dir=output_dir.relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    payload = report.to_json()
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["artifactCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["summary"]["diagnosticCounts"] == {
        "note": 0,
        "warning": 0,
        "error": 0,
    }
    assert payload["diagnostics"] == []
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
        "target": workload.entry_point,
        "stage": "compute",
    }
    assert artifact["provenance"] == {
        "pipeline": "entry-scoped-translate",
        "intermediate": "crossgl",
    }
    assert artifact["execution"]["entryPoints"][0]["workgroupSize"] == [1, 1, 1]
    materialization = _normalized_materialization(artifact["templateMaterialization"])
    assert materialization["specializationCount"] == workload.specialization_count
    assert _canonical_json_sha256(materialization) == (workload.materialization_sha256)
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert generated.count("kernel void ") == 1
    assert f"kernel void {workload.entry_point}" in generated
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
    return report_path, generated_path


def _roundtrip_pinned_mlx_reduce_through_metal(
    workload: ReduceMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-reduce-{workload.entry_point}-metal-roundtrip-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        report_path, generated_path = _translate_reduce_metal_artifact(
            mlx_root,
            work_dir,
            workload,
        )
        runtime_artifacts = build_runtime_artifact_manifest(report_path)
        assert runtime_artifacts["success"] is True, json.dumps(
            runtime_artifacts,
            indent=2,
        )
        host_interface = runtime_artifacts["artifacts"][0]["hostInterface"]
        assert host_interface["status"] == "ready"
        assert host_interface["entryPoints"] == [
            {
                "name": workload.entry_point,
                "stage": "compute",
                "executionConfig": {},
            }
        ]
        resources = [
            {field: resource[field] for field in RESOURCE_ABI_FIELDS}
            for resource in host_interface["resources"]
        ]
        shape_contract = EXPECTED_SHAPE_CONTRACTS[workload.shape]
        expected_resources = _resources(
            str(shape_contract["templateName"]),
            workload.input_type,
            workload.output_type,
        )
        _assert_exact_resources(resources, expected_resources, workload.entry_point)
        assert len(resources) == workload.resource_count
        assert _canonical_json_sha256(expected_resources) == workload.resources_sha256
        assert _canonical_json_sha256(resources) == workload.resources_sha256
        assert [resource["metadata"] for resource in host_interface["resources"]] == [
            {"entryPoint": workload.entry_point}
        ] * workload.resource_count

        xcrun = shutil.which("xcrun")
        if xcrun is None:
            message = "xcrun is required for the MLX reduce Metal proof"
            if os.environ.get(REQUIRE_REDUCE_METAL_ENV) == "1":
                pytest.fail(message)
            pytest.skip(message)
        air_path = work_dir / f"{workload.entry_point}.air"
        compiled = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-Werror",
                "-c",
                str(generated_path),
                "-o",
                str(air_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        assert air_path.is_file()
        assert air_path.stat().st_size > 0


@pytest.mark.parametrize(
    "workload",
    CURRENT_REDUCE_METAL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_reduce_family_roundtrips_through_metal(workload):
    _roundtrip_pinned_mlx_reduce_through_metal(workload)
