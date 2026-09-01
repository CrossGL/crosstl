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
MLX_COPY_SOURCE = "mlx/backend/metal/kernels/copy.metal"
MLX_COPY_SHA256 = "ed8a579eb6fe6a14c36560d2c8b548baf99e66fa77d300fb4ad7554883820eba"
REQUIRE_COPY_METAL_ENV = "CROSTL_REQUIRE_MLX_COPY_METAL_ROUNDTRIP"
COPY_METAL_SHARD_INDEX_ENV = "CROSTL_MLX_COPY_METAL_SHARD_INDEX"
COPY_METAL_SHARD_COUNT_ENV = "CROSTL_MLX_COPY_METAL_SHARD_COUNT"
COPY_METAL_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
COPY_METAL_CONTRACT_PATH = (
    ROOT / "demos" / "integrations" / "mlx" / "contracts" / "copy.metal-roundtrip.json"
)
# Installed from the terminal exhaustive proof before this harness is committed.
COPY_METAL_CONTRACT_SHA256 = (
    "5efd3b2b238ba4022e75fcf6848aedb6623af9d50ed1a9bf4e94cbd48aa3ef4f"
)


@dataclass(frozen=True)
class CopyMetalWorkload:
    entry_point: str
    shape: str
    template_name: str
    input_type: str
    output_type: str
    family: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CopyShapeSpec:
    template_name: str
    source_template_argument_count: int
    parameter_values: tuple[tuple[str, str, str], ...]
    helper_name: str | None
    helper_index_type: str | None
    resource_kind: str


BASE_PARAMETERS = (
    ("T", "entry.inputType", "source-instantiation"),
    ("U", "entry.outputType", "source-instantiation"),
)
N_ONE = (("N", "1", "source-instantiation"),)
N_DEFAULT = (("N", "WorkPerThread<U>::n", "source-default"),)
IDX_INT = (("IdxT", "int", "source-instantiation"),)
IDX_INT64_DEFAULT = (("IdxT", "int64_t", "source-default"),)
N_TWO = (("N", "2", "source-instantiation"),)
N_FOUR = (("N", "4", "source-instantiation"),)
COPY_SHAPE_SPECS = {
    "s": CopyShapeSpec("copy_s", 3, BASE_PARAMETERS + N_ONE, None, None, "size-u32"),
    "sn": CopyShapeSpec(
        "copy_s", 2, BASE_PARAMETERS + N_DEFAULT, None, None, "size-u32"
    ),
    "v": CopyShapeSpec("copy_v", 3, BASE_PARAMETERS + N_ONE, None, None, "size-u32"),
    "vn": CopyShapeSpec(
        "copy_v", 2, BASE_PARAMETERS + N_DEFAULT, None, None, "size-u32"
    ),
    "s2": CopyShapeSpec(
        "copy_s2", 2, BASE_PARAMETERS + N_DEFAULT, None, None, "size-i64"
    ),
    "v2": CopyShapeSpec(
        "copy_v2", 2, BASE_PARAMETERS + N_DEFAULT, None, None, "size-i64"
    ),
    "g1": CopyShapeSpec(
        "copy_g_nd1", 3, BASE_PARAMETERS + IDX_INT, "elem_to_loc_1", "int", "g-fixed1"
    ),
    "g1large": CopyShapeSpec(
        "copy_g_nd1",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_1",
        "int64_t",
        "g-fixed1",
    ),
    "g2": CopyShapeSpec(
        "copy_g_nd2", 3, BASE_PARAMETERS + IDX_INT, "elem_to_loc_2", "int", "g-fixed"
    ),
    "g2large": CopyShapeSpec(
        "copy_g_nd2",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_2",
        "int64_t",
        "g-fixed",
    ),
    "g3": CopyShapeSpec(
        "copy_g_nd3", 3, BASE_PARAMETERS + IDX_INT, "elem_to_loc_3", "int", "g-fixed"
    ),
    "g3large": CopyShapeSpec(
        "copy_g_nd3",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_3",
        "int64_t",
        "g-fixed",
    ),
    "gn2": CopyShapeSpec(
        "copy_g",
        4,
        BASE_PARAMETERS + N_TWO + IDX_INT,
        "elem_to_loc",
        "int",
        "g-general",
    ),
    "gn4large": CopyShapeSpec(
        "copy_g",
        3,
        BASE_PARAMETERS + N_FOUR + IDX_INT64_DEFAULT,
        "elem_to_loc",
        "int64_t",
        "g-general",
    ),
    "gg1": CopyShapeSpec(
        "copy_gg_nd1",
        3,
        BASE_PARAMETERS + IDX_INT,
        "elem_to_loc_1",
        "int",
        "gg-fixed1",
    ),
    "gg1large": CopyShapeSpec(
        "copy_gg_nd1",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_1",
        "int64_t",
        "gg-fixed1",
    ),
    "gg2": CopyShapeSpec(
        "copy_gg_nd2",
        3,
        BASE_PARAMETERS + IDX_INT,
        "elem_to_loc_2",
        "int",
        "gg-fixed",
    ),
    "gg2large": CopyShapeSpec(
        "copy_gg_nd2",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_2",
        "int64_t",
        "gg-fixed",
    ),
    "gg3": CopyShapeSpec(
        "copy_gg_nd3",
        3,
        BASE_PARAMETERS + IDX_INT,
        "elem_to_loc_3",
        "int",
        "gg-fixed",
    ),
    "gg3large": CopyShapeSpec(
        "copy_gg_nd3",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_3",
        "int64_t",
        "gg-fixed",
    ),
    "ggn2": CopyShapeSpec(
        "copy_gg",
        4,
        BASE_PARAMETERS + N_TWO + IDX_INT,
        "elem_to_loc_2_nd",
        "int",
        "gg-general",
    ),
    "ggn4large": CopyShapeSpec(
        "copy_gg",
        3,
        BASE_PARAMETERS + N_FOUR + IDX_INT64_DEFAULT,
        "elem_to_loc_2_nd",
        "int64_t",
        "gg-general",
    ),
    "gg1_dynamic": CopyShapeSpec(
        "copy_gg_dynamic_nd1",
        3,
        BASE_PARAMETERS + IDX_INT,
        "elem_to_loc_1",
        "int",
        "gg-fixed1-dynamic",
    ),
    "gg1large_dynamic": CopyShapeSpec(
        "copy_gg_dynamic_nd1",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_1",
        "int64_t",
        "gg-fixed1-dynamic",
    ),
    "gg2_dynamic": CopyShapeSpec(
        "copy_gg_dynamic_nd2",
        3,
        BASE_PARAMETERS + IDX_INT,
        "elem_to_loc_2",
        "int",
        "gg-fixed-dynamic",
    ),
    "gg2large_dynamic": CopyShapeSpec(
        "copy_gg_dynamic_nd2",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_2",
        "int64_t",
        "gg-fixed-dynamic",
    ),
    "gg3_dynamic": CopyShapeSpec(
        "copy_gg_dynamic_nd3",
        3,
        BASE_PARAMETERS + IDX_INT,
        "elem_to_loc_3",
        "int",
        "gg-fixed-dynamic",
    ),
    "gg3large_dynamic": CopyShapeSpec(
        "copy_gg_dynamic_nd3",
        2,
        BASE_PARAMETERS + IDX_INT64_DEFAULT,
        "elem_to_loc_3",
        "int64_t",
        "gg-fixed-dynamic",
    ),
    "ggn2_dynamic": CopyShapeSpec(
        "copy_gg_dynamic",
        4,
        BASE_PARAMETERS + N_TWO + IDX_INT,
        "elem_to_loc_2_nd",
        "int",
        "gg-general-dynamic",
    ),
    "ggn4large_dynamic": CopyShapeSpec(
        "copy_gg_dynamic",
        3,
        BASE_PARAMETERS + N_FOUR + IDX_INT64_DEFAULT,
        "elem_to_loc_2_nd",
        "int64_t",
        "gg-general-dynamic",
    ),
}

COPY_TYPES = (
    "bfloat16_t",
    "bool",
    "complex64_t",
    "float",
    "half",
    "int16_t",
    "int32_t",
    "int64_t",
    "int8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "uint8_t",
)
COPY_METAL_TYPE_NAMES = {
    "bfloat16_t": "bfloat",
    "bool": "bool",
    "complex64_t": "complex_t_float",
    "float": "float",
    "half": "half",
    "int16_t": "int",
    "int32_t": "int",
    "int64_t": "int64_t",
    "int8_t": "int",
    "uint16_t": "uint",
    "uint32_t": "uint",
    "uint64_t": "uint64_t",
    "uint8_t": "uint",
}
CONTRACT_RESOURCE_ABI_FIELDS = (
    "name",
    "kind",
    "set",
    "binding",
    "access",
    "type",
)
WORK_PER_THREAD_OUTPUT_TYPES = frozenset(COPY_TYPES) - {
    "complex64_t",
    "int64_t",
    "uint64_t",
}
EXPECTED_COPY_TYPE_PAIRS = {
    f"{input_type}->{output_type}": (
        12
        + (2 if output_type in WORK_PER_THREAD_OUTPUT_TYPES else 0)
        + (16 if input_type == output_type else 0)
    )
    for input_type in COPY_TYPES
    for output_type in COPY_TYPES
}
EXPECTED_COPY_CLASSIFICATIONS = {
    "shapes": {
        "g1": 169,
        "g1large": 169,
        "g2": 169,
        "g2large": 169,
        "g3": 169,
        "g3large": 169,
        "gg1": 13,
        "gg1_dynamic": 13,
        "gg1large": 13,
        "gg1large_dynamic": 13,
        "gg2": 13,
        "gg2_dynamic": 13,
        "gg2large": 13,
        "gg2large_dynamic": 13,
        "gg3": 13,
        "gg3_dynamic": 13,
        "gg3large": 13,
        "gg3large_dynamic": 13,
        "ggn2": 13,
        "ggn2_dynamic": 13,
        "ggn4large": 13,
        "ggn4large_dynamic": 13,
        "gn2": 169,
        "gn4large": 169,
        "s": 169,
        "s2": 169,
        "sn": 130,
        "v": 169,
        "v2": 169,
        "vn": 130,
    },
    "templates": {
        "copy_g": 338,
        "copy_g_nd1": 338,
        "copy_g_nd2": 338,
        "copy_g_nd3": 338,
        "copy_gg": 26,
        "copy_gg_dynamic": 26,
        "copy_gg_dynamic_nd1": 26,
        "copy_gg_dynamic_nd2": 26,
        "copy_gg_dynamic_nd3": 26,
        "copy_gg_nd1": 26,
        "copy_gg_nd2": 26,
        "copy_gg_nd3": 26,
        "copy_s": 299,
        "copy_s2": 169,
        "copy_v": 299,
        "copy_v2": 169,
    },
    "inputTypes": {copy_type: 192 for copy_type in COPY_TYPES},
    "outputTypes": {
        copy_type: 198 if copy_type in WORK_PER_THREAD_OUTPUT_TYPES else 172
        for copy_type in COPY_TYPES
    },
    "typePairs": EXPECTED_COPY_TYPE_PAIRS,
    "families": {
        "complex64-to-bool": 14,
        "complex64-to-scalar": 150,
        "same-type": 384,
        "scalar-conversion": 1650,
        "scalar-to-bool": 154,
        "scalar-to-complex64": 144,
    },
}


def _contract(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    return json.loads(payload)


COPY_METAL_CONTRACT = _contract(
    COPY_METAL_CONTRACT_PATH,
    COPY_METAL_CONTRACT_SHA256,
)
COPY_METAL_ENTRIES = tuple(COPY_METAL_CONTRACT["entries"])
COPY_METAL_WORKLOADS = tuple(
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
    for entry in COPY_METAL_ENTRIES
)


COPY_METAL_RESOURCES_BY_ENTRY = {
    entry["entryPoint"]: entry["resources"] for entry in COPY_METAL_ENTRIES
}


def _resource_templates(kind: str) -> list[dict[str, object]]:
    resources: list[dict[str, object]] = [
        {
            "name": "src",
            "kind": "buffer",
            "set": 0,
            "binding": 0,
            "access": "read",
            "type": "const device {inputMetalType}*",
        },
        {
            "name": "dst",
            "kind": "buffer",
            "set": 0,
            "binding": 1,
            "access": "read_write",
            "type": "device {outputMetalType}*",
        },
    ]
    resources_by_kind = {
        "size-u32": (("size", 2, "constant uint&"),),
        "size-i64": (("size", 2, "constant int64_t&"),),
        "g-fixed1": (("src_stride", 3, "constant int64_t&"),),
        "g-fixed": (("src_strides", 3, "constant int64_t*"),),
        "g-general": (
            ("src_shape", 2, "constant int*"),
            ("src_strides", 3, "constant int64_t*"),
            ("ndim", 5, "constant int&"),
        ),
        "gg-fixed1": (
            ("src_stride", 3, "constant int64_t&"),
            ("dst_stride", 4, "constant int64_t&"),
        ),
        "gg-fixed": (
            ("src_strides", 3, "constant int64_t*"),
            ("dst_strides", 4, "constant int64_t*"),
        ),
        "gg-general": (
            ("src_shape", 2, "constant int*"),
            ("src_strides", 3, "constant int64_t*"),
            ("dst_strides", 4, "constant int64_t*"),
            ("ndim", 5, "constant int&"),
        ),
        "gg-fixed1-dynamic": (
            ("src_stride", 3, "constant int64_t&"),
            ("dst_stride", 4, "constant int64_t&"),
            ("src_offset", 6, "constant int64_t&"),
            ("dst_offset", 7, "constant int64_t&"),
        ),
        "gg-fixed-dynamic": (
            ("src_strides", 3, "constant int64_t*"),
            ("dst_strides", 4, "constant int64_t*"),
            ("src_offset", 6, "constant int64_t&"),
            ("dst_offset", 7, "constant int64_t&"),
        ),
        "gg-general-dynamic": (
            ("src_shape", 2, "constant int*"),
            ("src_strides", 3, "constant int64_t*"),
            ("dst_strides", 4, "constant int64_t*"),
            ("ndim", 5, "constant int&"),
            ("src_offset", 6, "constant int64_t&"),
            ("dst_offset", 7, "constant int64_t&"),
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
        for name, binding, type_name in resources_by_kind[kind]
    )
    return resources


def _resources(kind: str, input_type: str, output_type: str) -> list[dict[str, object]]:
    substitutions = {
        "inputMetalType": COPY_METAL_TYPE_NAMES[input_type],
        "outputMetalType": COPY_METAL_TYPE_NAMES[output_type],
    }
    return [
        {**resource, "type": str(resource["type"]).format(**substitutions)}
        for resource in _resource_templates(kind)
    ]


def _shape_has_complex_bool(shape: str) -> bool:
    return EXPECTED_COPY_CLASSIFICATIONS["shapes"][shape] > 13


def _expected_shape_contracts() -> dict[str, dict[str, object]]:
    contracts = {}
    for shape, spec in COPY_SHAPE_SPECS.items():
        resources = _resource_templates(spec.resource_kind)
        reachable = [spec.template_name]
        if spec.helper_name is not None:
            reachable.append(f"{spec.helper_name}<{spec.helper_index_type}>")
        reachable.append("cast_to<entry.outputType,entry.inputType>")
        conditional = []
        if _shape_has_complex_bool(shape):
            conditional.append(
                {
                    "when": {
                        "inputType": "complex64_t",
                        "outputType": "bool",
                    },
                    "additionalCount": 1,
                    "reachableSpecializations": ["cast_to<bool,float>"],
                }
            )
        contracts[shape] = {
            "entryPrefix": f"{shape}_",
            "entryCount": EXPECTED_COPY_CLASSIFICATIONS["shapes"][shape],
            "templateName": spec.template_name,
            "sourceTemplateArgumentCount": spec.source_template_argument_count,
            "resourceKind": spec.resource_kind,
            "templateParameters": {
                name: {"value": value, "source": source}
                for name, value, source in spec.parameter_values
            },
            "specializationContract": {
                "baseCountPerArtifact": len(reachable),
                "baseReachableSpecializations": reachable,
                "conditionalAdditions": conditional,
            },
            "hostResourceCountPerArtifact": len(resources),
            "hostResources": resources,
        }
    return {shape: contracts[shape] for shape in sorted(contracts)}


def _expected_specialization_count(workload: CopyMetalWorkload) -> int:
    spec = COPY_SHAPE_SPECS[workload.shape]
    return (
        2
        + (spec.helper_name is not None)
        + (workload.input_type == "complex64_t" and workload.output_type == "bool")
    )


def test_current_mlx_copy_metal_contract_is_complete_and_classified():
    contract = COPY_METAL_CONTRACT
    assert contract["schemaVersion"] == 2
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_COPY_SOURCE
    assert contract["sourceSha256"] == MLX_COPY_SHA256
    assert contract["target"] == "metal"
    assert contract["metalTypeNames"] == COPY_METAL_TYPE_NAMES
    assert contract["selection"] == {
        "entryCount": 2496,
        "shapeCount": 30,
        "templateCount": 16,
        "inputTypeCount": 13,
        "outputTypeCount": 13,
        "typePairCount": 169,
        "familyCount": 6,
        "allDiscoveredSourceInstantiationsIncluded": True,
    }
    expected_shapes = _expected_shape_contracts()
    assert contract["shapeContracts"] == expected_shapes
    assert contract["classifications"] == EXPECTED_COPY_CLASSIFICATIONS

    specialization_counts = {
        shape: (
            shape_contract["entryCount"]
            * shape_contract["specializationContract"]["baseCountPerArtifact"]
            + sum(
                addition["additionalCount"]
                for addition in shape_contract["specializationContract"][
                    "conditionalAdditions"
                ]
            )
        )
        for shape, shape_contract in expected_shapes.items()
    }
    resource_counts = {
        shape: (
            shape_contract["entryCount"]
            * shape_contract["hostResourceCountPerArtifact"]
        )
        for shape, shape_contract in expected_shapes.items()
    }
    artifact_contract = contract["artifactContract"]
    assert artifact_contract["artifactCount"] == 2496
    assert artifact_contract["artifactCountPerEntry"] == 1
    assert artifact_contract["specializationCount"] == 6566
    assert artifact_contract["specializationCountsByShape"] == specialization_counts
    assert artifact_contract["explicitCastSpecializationCount"] == 2496
    assert artifact_contract["nestedComplexBoolSpecializationCount"] == 14
    assert artifact_contract["unsupportedSpecializationCount"] == 0
    assert artifact_contract["reachableKernelCountPerArtifact"] == 1
    assert artifact_contract["provenance"] == "entry-scoped-translate"
    assert artifact_contract["intermediate"] == "crossgl"
    assert artifact_contract["hostInterfaceStatus"] == "ready"
    assert artifact_contract["reflectedResourceCount"] == 8684
    assert artifact_contract["reflectedResourceCountsByShape"] == resource_counts
    reflected_resource_type_counts = dict(
        sorted(
            Counter(
                resource["type"]
                for entry in COPY_METAL_ENTRIES
                for resource in entry["resources"]
            ).items()
        )
    )
    assert artifact_contract["reflectedResourceTypeCounts"] == (
        reflected_resource_type_counts
    )
    assert artifact_contract["resourceAbiFields"] == list(CONTRACT_RESOURCE_ABI_FIELDS)
    assert artifact_contract["resourceTypesIncluded"] is True
    assert artifact_contract["hostDispatchWorkgroupSize"] == [1, 1, 1]
    assert artifact_contract["generatedSizeBytesTotal"] > 0
    assert artifact_contract["generatedSizeRange"]["minimum"]["sizeBytes"] > 0
    assert (
        artifact_contract["generatedSizeRange"]["maximum"]["sizeBytes"]
        >= artifact_contract["generatedSizeRange"]["minimum"]["sizeBytes"]
    )
    assert artifact_contract["nativeCompiler"] == ("xcrun -sdk macosx metal -Werror -c")
    assert artifact_contract["requiresNonemptyAirArtifact"] is True

    entries = COPY_METAL_ENTRIES
    assert len(entries) == 2496
    assert len({entry["entryPoint"] for entry in entries}) == 2496
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
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
            "resources",
        ]
        assert entry["resources"] == _resources(
            COPY_SHAPE_SPECS[entry["shape"]].resource_kind,
            entry["inputType"],
            entry["outputType"],
        )
        assert all(
            list(resource) == list(CONTRACT_RESOURCE_ABI_FIELDS)
            for resource in entry["resources"]
        )
        assert entry["entryPoint"].startswith(f'{entry["shape"]}_')
        assert entry["templateName"] == COPY_SHAPE_SPECS[entry["shape"]].template_name
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
        assert entry["sizeBytes"] > 0


def _partition_copy_metal_workloads(
    workloads: tuple[CopyMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[CopyMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX copy Metal shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX copy Metal shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX copy Metal shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_copy_metal_workloads() -> tuple[CopyMetalWorkload, ...]:
    raw_index = os.environ.get(COPY_METAL_SHARD_INDEX_ENV)
    raw_count = os.environ.get(COPY_METAL_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return COPY_METAL_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{COPY_METAL_SHARD_INDEX_ENV} and {COPY_METAL_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError("MLX copy Metal shard values must be integers") from error
    try:
        return _partition_copy_metal_workloads(
            COPY_METAL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_COPY_METAL_WORKLOADS = _current_copy_metal_workloads()


def test_current_mlx_copy_metal_ci_shards_are_complete_and_disjoint():
    shards = tuple(
        _partition_copy_metal_workloads(
            COPY_METAL_WORKLOADS,
            shard_index,
            COPY_METAL_CI_SHARD_COUNT,
        )
        for shard_index in range(COPY_METAL_CI_SHARD_COUNT)
    )

    assert [len(shard) for shard in shards] == [104] * 24
    for shard_index, shard in enumerate(shards):
        assert shard == COPY_METAL_WORKLOADS[shard_index::COPY_METAL_CI_SHARD_COUNT]
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 2496
    assert len(set(entry_points)) == 2496
    assert set(entry_points) == {
        workload.entry_point for workload in COPY_METAL_WORKLOADS
    }


def _project_config(workload: CopyMetalWorkload) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_COPY_SOURCE}"]
        include_dirs = ["."]
        targets = ["metal"]
        output_dir = ".crosstl-mlx-copy-metal-roundtrip/out"

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
        if os.environ.get(REQUIRE_COPY_METAL_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_COPY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX copy source is missing: {source_path}")
    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_COPY_SHA256
    return mlx_root


def _materialization_parameters(workload: CopyMetalWorkload) -> tuple[dict, dict]:
    spec = COPY_SHAPE_SPECS[workload.shape]
    substitutions = {
        "entry.inputType": workload.input_type,
        "entry.outputType": workload.output_type,
        "WorkPerThread<U>::n": f"WorkPerThread<{workload.output_type}>::n",
    }
    parameters = {
        name: substitutions.get(value, value)
        for name, value, _source in spec.parameter_values
    }
    parameter_sources = {name: source for name, _value, source in spec.parameter_values}
    return parameters, parameter_sources


def _expected_materializations(workload: CopyMetalWorkload) -> list[dict]:
    spec = COPY_SHAPE_SPECS[workload.shape]
    parameters, parameter_sources = _materialization_parameters(workload)
    records = [
        {
            "name": workload.template_name,
            "materializedName": workload.entry_point,
            "parameters": parameters,
            "parameterSources": parameter_sources,
            "source": "source-instantiation",
            "hostName": workload.entry_point,
        }
    ]
    if spec.helper_name is not None:
        records.append(
            {
                "name": spec.helper_name,
                "materializedName": f"{spec.helper_name}_{spec.helper_index_type}",
                "parameters": {"IdxT": spec.helper_index_type},
                "parameterSources": {"IdxT": "call-site"},
                "source": "call-site",
            }
        )
    records.append(
        {
            "name": "cast_to",
            "materializedName": f"cast_to_{workload.output_type}_{workload.input_type}",
            "parameters": {
                "T": workload.input_type,
                "U": workload.output_type,
            },
            "parameterSources": {"T": "call-site", "U": "call-site"},
            "source": "call-site",
        }
    )
    if workload.input_type == "complex64_t" and workload.output_type == "bool":
        records.append(
            {
                "name": "cast_to",
                "materializedName": "cast_to_bool_float",
                "parameters": {"T": "float", "U": "bool"},
                "parameterSources": {"T": "call-site", "U": "call-site"},
                "source": "call-site",
            }
        )
    return records


def _translate_copy_metal_artifact(
    mlx_root: Path,
    work_dir: Path,
    workload: CopyMetalWorkload,
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
        "target": workload.entry_point,
        "stage": "compute",
    }
    assert artifact["provenance"] == {
        "pipeline": "entry-scoped-translate",
        "intermediate": "crossgl",
    }
    assert artifact["execution"]["entryPoints"][0]["workgroupSize"] == [1, 1, 1]
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
    assert generated.count("kernel void ") == 1
    assert f"kernel void {workload.entry_point}" in generated
    assert f"cast_to_{workload.output_type}_{workload.input_type}(" in generated
    if workload.input_type == "float" and workload.output_type == "bool":
        assert "(as_type<uint>(val) & 2147483647) != 0" in generated
    if workload.input_type == "bfloat16_t" and workload.output_type == "bool":
        assert "(as_type<ushort>(val) & 32767) != 0" in generated
    if workload.input_type == "complex64_t" and workload.output_type == "bool":
        assert "cast_to_bool_float((val).real)" in generated or (
            "cast_to_bool_float(val.real)" in generated
        )
        assert "cast_to_bool_float((val).imag)" in generated or (
            "cast_to_bool_float(val.imag)" in generated
        )
    elif workload.input_type == "complex64_t" and workload.output_type != "complex64_t":
        assert "((val).real)" in generated
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


def _roundtrip_pinned_mlx_copy_through_metal(workload: CopyMetalWorkload) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-copy-{workload.entry_point}-metal-roundtrip-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        report_path, generated_path = _translate_copy_metal_artifact(
            mlx_root,
            work_dir,
            workload,
        )
        runtime_artifacts = build_runtime_artifact_manifest(report_path)
        assert runtime_artifacts["success"] is True, json.dumps(
            runtime_artifacts,
            indent=2,
        )
        reflected = runtime_artifacts["artifacts"][0]["hostInterface"]
        assert reflected["status"] == "ready"
        assert reflected["entryPoints"] == [
            {
                "name": workload.entry_point,
                "stage": "compute",
                "executionConfig": {},
            }
        ]
        expected_resources = COPY_METAL_RESOURCES_BY_ENTRY[workload.entry_point]
        assert [
            {field: resource[field] for field in CONTRACT_RESOURCE_ABI_FIELDS}
            for resource in reflected["resources"]
        ] == expected_resources
        assert [resource["metadata"] for resource in reflected["resources"]] == [
            {"entryPoint": workload.entry_point}
        ] * len(expected_resources)

        xcrun = shutil.which("xcrun")
        if xcrun is None:
            message = "xcrun is required for the MLX copy Metal proof"
            if os.environ.get(REQUIRE_COPY_METAL_ENV) == "1":
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
    CURRENT_COPY_METAL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_copy_family_roundtrips_through_metal(workload):
    _roundtrip_pinned_mlx_copy_through_metal(workload)
