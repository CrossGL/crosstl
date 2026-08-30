import hashlib
import json
import os
import re
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
MLX_BINARY_SOURCE = "mlx/backend/metal/kernels/binary.metal"
MLX_BINARY_SHA256 = "4dadb612a9b768f9d51b3b394b32fc0129d361a55b35d545b3c014c87e00897e"
REQUIRE_BINARY_METAL_ENV = "CROSTL_REQUIRE_MLX_BINARY_METAL_ROUNDTRIP"
BINARY_METAL_SHARD_INDEX_ENV = "CROSTL_MLX_BINARY_METAL_SHARD_INDEX"
BINARY_METAL_SHARD_COUNT_ENV = "CROSTL_MLX_BINARY_METAL_SHARD_COUNT"
BINARY_METAL_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
BINARY_SCALAR_METAL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "binary.scalar-metal-roundtrip.json"
)
BINARY_SCALAR_METAL_CONTRACT_SHA256 = (
    "afb4d74c25d07203ded397df7f740d9721f9611b0c9bae64ee629a9043296e83"
)
BINARY_METAL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "binary.metal-roundtrip.json"
)
BINARY_METAL_CONTRACT_SHA256 = (
    "01fb47137f7a330cad3404a5c81abd677b455ae010aceb3d88e92f01d257ca5e"
)


@dataclass(frozen=True)
class BinaryMetalWorkload:
    entry_point: str
    shape: str
    template_name: str
    operator_type: str
    input_type: str
    output_type: str
    family: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class BinaryShapeSpec:
    template_name: str
    source_template_argument_count: int
    parameter_values: tuple[tuple[str, str, str], ...]
    helper_name: str | None
    helper_index_type: str | None
    resource_kind: str


BASE_PARAMETER_VALUES = (
    ("T", "entry.inputType", "source-instantiation"),
    ("U", "entry.outputType", "source-instantiation"),
    ("Op", "entry.operator", "source-instantiation"),
)
N_ONE = (("N", "1", "source-instantiation"),)
N_DEFAULT = (("N", "WorkPerThread<T>::n", "source-default"),)
BINARY_SHAPE_SPECS = {
    "g1": BinaryShapeSpec(
        "binary_g_nd1",
        4,
        BASE_PARAMETER_VALUES + (("IdxT", "int", "source-instantiation"),),
        "elem_to_loc_1",
        "int",
        "fixed1",
    ),
    "g1large": BinaryShapeSpec(
        "binary_g_nd1",
        3,
        BASE_PARAMETER_VALUES + (("IdxT", "int64_t", "source-default"),),
        "elem_to_loc_1",
        "int64_t",
        "fixed1",
    ),
    "g2": BinaryShapeSpec(
        "binary_g_nd2",
        4,
        BASE_PARAMETER_VALUES + (("IdxT", "int", "source-instantiation"),),
        "elem_to_loc_2",
        "int",
        "fixed",
    ),
    "g2large": BinaryShapeSpec(
        "binary_g_nd2",
        3,
        BASE_PARAMETER_VALUES + (("IdxT", "int64_t", "source-default"),),
        "elem_to_loc_2",
        "int64_t",
        "fixed",
    ),
    "g3": BinaryShapeSpec(
        "binary_g_nd3",
        4,
        BASE_PARAMETER_VALUES + (("IdxT", "int", "source-instantiation"),),
        "elem_to_loc_3",
        "int",
        "fixed",
    ),
    "g3large": BinaryShapeSpec(
        "binary_g_nd3",
        3,
        BASE_PARAMETER_VALUES + (("IdxT", "int64_t", "source-default"),),
        "elem_to_loc_3",
        "int64_t",
        "fixed",
    ),
    "gn2": BinaryShapeSpec(
        "binary_g",
        5,
        BASE_PARAMETER_VALUES
        + (
            ("N", "2", "source-instantiation"),
            ("IdxT", "int", "source-instantiation"),
        ),
        "elem_to_loc_2_nd",
        "int",
        "general",
    ),
    "gn4large": BinaryShapeSpec(
        "binary_g",
        4,
        BASE_PARAMETER_VALUES
        + (
            ("N", "4", "source-instantiation"),
            ("IdxT", "int64_t", "source-default"),
        ),
        "elem_to_loc_2_nd",
        "int64_t",
        "general",
    ),
    "ss": BinaryShapeSpec("binary_ss", 3, BASE_PARAMETER_VALUES, None, None, "scalar"),
    "sv": BinaryShapeSpec(
        "binary_sv", 4, BASE_PARAMETER_VALUES + N_ONE, None, None, "size"
    ),
    "sv2": BinaryShapeSpec(
        "binary_sv2", 3, BASE_PARAMETER_VALUES + N_DEFAULT, None, None, "size"
    ),
    "svn": BinaryShapeSpec(
        "binary_sv", 3, BASE_PARAMETER_VALUES + N_DEFAULT, None, None, "size"
    ),
    "vs": BinaryShapeSpec(
        "binary_vs", 4, BASE_PARAMETER_VALUES + N_ONE, None, None, "size"
    ),
    "vs2": BinaryShapeSpec(
        "binary_vs2", 3, BASE_PARAMETER_VALUES + N_DEFAULT, None, None, "size"
    ),
    "vsn": BinaryShapeSpec(
        "binary_vs", 3, BASE_PARAMETER_VALUES + N_DEFAULT, None, None, "size"
    ),
    "vv": BinaryShapeSpec(
        "binary_vv", 4, BASE_PARAMETER_VALUES + N_ONE, None, None, "size"
    ),
    "vv2": BinaryShapeSpec(
        "binary_vv2", 3, BASE_PARAMETER_VALUES + N_DEFAULT, None, None, "size"
    ),
    "vvn": BinaryShapeSpec(
        "binary_vv", 3, BASE_PARAMETER_VALUES + N_DEFAULT, None, None, "size"
    ),
}
EXPECTED_BINARY_CLASSIFICATIONS = {
    "shapes": {
        "g1": 238,
        "g1large": 238,
        "g2": 238,
        "g2large": 238,
        "g3": 238,
        "g3large": 238,
        "gn2": 238,
        "gn4large": 238,
        "ss": 238,
        "sv": 238,
        "sv2": 238,
        "svn": 184,
        "vs": 238,
        "vs2": 238,
        "vsn": 184,
        "vv": 238,
        "vv2": 238,
        "vvn": 184,
    },
    "templates": {
        "binary_g": 476,
        "binary_g_nd1": 476,
        "binary_g_nd2": 476,
        "binary_g_nd3": 476,
        "binary_ss": 238,
        "binary_sv": 422,
        "binary_sv2": 238,
        "binary_vs": 422,
        "binary_vs2": 238,
        "binary_vv": 422,
        "binary_vv2": 238,
    },
    "operators": {
        "Add": 225,
        "ArcTan2": 54,
        "BitwiseAnd": 156,
        "BitwiseOr": 156,
        "BitwiseXor": 156,
        "Divide": 225,
        "Equal": 225,
        "Greater": 225,
        "GreaterEqual": 225,
        "LeftShift": 138,
        "Less": 225,
        "LessEqual": 225,
        "LogAddExp": 69,
        "LogicalAnd": 18,
        "LogicalOr": 18,
        "Maximum": 225,
        "Minimum": 225,
        "Multiply": 225,
        "NaNEqual": 69,
        "NotEqual": 225,
        "Power": 225,
        "Remainder": 225,
        "RightShift": 138,
        "Subtract": 225,
    },
    "typePairs": {
        "bfloat16_t->bfloat16_t": 180,
        "bfloat16_t->bool": 126,
        "bool->bool": 342,
        "complex64_t->bool": 105,
        "complex64_t->complex64_t": 135,
        "float->bool": 126,
        "float->float": 180,
        "half->bool": 126,
        "half->half": 180,
        "int16_t->bool": 108,
        "int16_t->int16_t": 234,
        "int32_t->bool": 108,
        "int32_t->int32_t": 234,
        "int64_t->bool": 90,
        "int64_t->int64_t": 195,
        "int8_t->bool": 108,
        "int8_t->int8_t": 234,
        "uint16_t->bool": 108,
        "uint16_t->uint16_t": 234,
        "uint32_t->bool": 108,
        "uint32_t->uint32_t": 234,
        "uint64_t->bool": 90,
        "uint64_t->uint64_t": 195,
        "uint8_t->bool": 108,
        "uint8_t->uint8_t": 234,
    },
    "families": {
        "bfloat16-comparison": 126,
        "bfloat16-same-type": 180,
        "boolean-same-type": 342,
        "complex64-comparison": 105,
        "complex64-same-type": 135,
        "float16-comparison": 126,
        "float16-same-type": 180,
        "float32-comparison": 126,
        "float32-same-type": 180,
        "int16-comparison": 108,
        "int16-same-type": 234,
        "int32-comparison": 108,
        "int32-same-type": 234,
        "int64-comparison": 90,
        "int64-same-type": 195,
        "int8-comparison": 108,
        "int8-same-type": 234,
        "uint16-comparison": 108,
        "uint16-same-type": 234,
        "uint32-comparison": 108,
        "uint32-same-type": 234,
        "uint64-comparison": 90,
        "uint64-same-type": 195,
        "uint8-comparison": 108,
        "uint8-same-type": 234,
    },
}


def _contract(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    return json.loads(payload)


BINARY_SCALAR_METAL_CONTRACT = _contract(
    BINARY_SCALAR_METAL_CONTRACT_PATH,
    BINARY_SCALAR_METAL_CONTRACT_SHA256,
)
BINARY_METAL_CONTRACT = _contract(
    BINARY_METAL_CONTRACT_PATH,
    BINARY_METAL_CONTRACT_SHA256,
)
BINARY_METAL_ENTRIES = tuple(BINARY_METAL_CONTRACT["entries"])
BINARY_METAL_OPERATOR_TYPES = frozenset(
    entry["operator"] for entry in BINARY_METAL_ENTRIES
)
BINARY_METAL_WORKLOADS = tuple(
    BinaryMetalWorkload(
        entry_point=entry["entryPoint"],
        shape=entry["shape"],
        template_name=entry["templateName"],
        operator_type=entry["operator"],
        input_type=entry["inputType"],
        output_type=entry["outputType"],
        family=entry["family"],
        sha256=entry["sha256"],
        size_bytes=entry["sizeBytes"],
    )
    for entry in BINARY_METAL_ENTRIES
)


def _resources(kind: str) -> list[dict[str, object]]:
    resources: list[dict[str, object]] = [
        {"name": "a", "kind": "buffer", "binding": 0, "access": "read"},
        {"name": "b", "kind": "buffer", "binding": 1, "access": "read"},
        {"name": "c", "kind": "buffer", "binding": 2, "access": "read_write"},
    ]
    if kind == "scalar":
        return resources
    if kind == "size":
        return resources + [
            {
                "name": "size",
                "kind": "constant-buffer",
                "binding": 3,
                "access": "read",
            }
        ]
    if kind == "fixed1":
        names = ("a_stride", "b_stride")
    elif kind == "fixed":
        names = ("a_strides", "b_strides")
    elif kind == "general":
        names = ("shape", "a_strides", "b_strides", "ndim")
    else:
        raise AssertionError(f"unknown resource contract: {kind}")
    return resources + [
        {
            "name": name,
            "kind": "constant-buffer",
            "binding": binding,
            "access": "read",
        }
        for binding, name in enumerate(names, 3)
    ]


def _expected_shape_contracts() -> dict[str, dict[str, object]]:
    contracts = {}
    for shape, spec in BINARY_SHAPE_SPECS.items():
        resources = _resources(spec.resource_kind)
        reachable = [spec.template_name]
        if spec.helper_name is not None:
            reachable.append(f"{spec.helper_name}<{spec.helper_index_type}>")
        contracts[shape] = {
            "entryPrefix": f"{shape}_",
            "entryCount": EXPECTED_BINARY_CLASSIFICATIONS["shapes"][shape],
            "templateName": spec.template_name,
            "sourceTemplateArgumentCount": spec.source_template_argument_count,
            "templateParameters": {
                name: {"value": value, "source": source}
                for name, value, source in spec.parameter_values
            },
            "specializationCountPerArtifact": len(reachable),
            "reachableSpecializations": reachable,
            "hostResourceCountPerArtifact": len(resources),
            "hostResources": resources,
        }
    return {shape: contracts[shape] for shape in sorted(contracts)}


def test_current_mlx_binary_metal_contract_is_complete_and_classified():
    contract = BINARY_METAL_CONTRACT
    assert contract["schemaVersion"] == 2
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_BINARY_SOURCE
    assert contract["sourceSha256"] == MLX_BINARY_SHA256
    assert contract["target"] == "metal"
    assert contract["selection"] == {
        "entryCount": 4122,
        "shapeCount": 18,
        "templateCount": 11,
        "operatorCount": 24,
        "typePairCount": 25,
        "familyCount": 25,
        "allDiscoveredSourceInstantiationsIncluded": True,
    }
    expected_shape_contracts = _expected_shape_contracts()
    assert contract["shapeContracts"] == expected_shape_contracts
    assert contract["classifications"] == EXPECTED_BINARY_CLASSIFICATIONS

    specialization_counts = {
        shape: (
            shape_contract["entryCount"]
            * shape_contract["specializationCountPerArtifact"]
        )
        for shape, shape_contract in expected_shape_contracts.items()
    }
    resource_counts = {
        shape: (
            shape_contract["entryCount"]
            * shape_contract["hostResourceCountPerArtifact"]
        )
        for shape, shape_contract in expected_shape_contracts.items()
    }
    assert contract["artifactContract"] == {
        "artifactCount": 4122,
        "artifactCountPerEntry": 1,
        "specializationCount": 6026,
        "specializationCountsByShape": specialization_counts,
        "unsupportedSpecializationCount": 0,
        "selectedOperatorImplementationCountPerArtifact": 1,
        "unselectedOperatorBodiesPruned": True,
        "reachableKernelCountPerArtifact": 1,
        "provenance": "entry-scoped-translate",
        "intermediate": "crossgl",
        "hostInterfaceStatus": "ready",
        "reflectedResourceCount": 19106,
        "reflectedResourceCountsByShape": resource_counts,
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "generatedSizeBytesTotal": 5140983,
        "generatedSizeRange": {
            "minimum": {
                "entryPoint": "ss_Addint8",
                "sizeBytes": 668,
            },
            "maximum": {
                "entryPoint": "gn4large_LogAddExpcomplex64",
                "sizeBytes": 4717,
            },
        },
        "nativeCompiler": "xcrun -sdk macosx metal -Werror -c",
        "requiresNonemptyAirArtifact": True,
    }

    entries = BINARY_METAL_ENTRIES
    assert len(entries) == 4122
    assert len({entry["entryPoint"] for entry in entries}) == 4122
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert Counter(entry["shape"] for entry in entries) == (
        EXPECTED_BINARY_CLASSIFICATIONS["shapes"]
    )
    assert Counter(entry["templateName"] for entry in entries) == (
        EXPECTED_BINARY_CLASSIFICATIONS["templates"]
    )
    assert Counter(entry["operator"] for entry in entries) == (
        EXPECTED_BINARY_CLASSIFICATIONS["operators"]
    )
    assert (
        Counter(f'{entry["inputType"]}->{entry["outputType"]}' for entry in entries)
        == EXPECTED_BINARY_CLASSIFICATIONS["typePairs"]
    )
    assert Counter(entry["family"] for entry in entries) == (
        EXPECTED_BINARY_CLASSIFICATIONS["families"]
    )
    for entry in entries:
        assert list(entry) == [
            "entryPoint",
            "shape",
            "templateName",
            "operator",
            "inputType",
            "outputType",
            "family",
            "sha256",
            "sizeBytes",
        ]
        assert entry["entryPoint"].startswith(f'{entry["shape"]}_')
        assert entry["templateName"] == BINARY_SHAPE_SPECS[entry["shape"]].template_name
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
        assert entry["sizeBytes"] > 0

    scalar_entries = {
        entry["entryPoint"]: entry for entry in entries if entry["shape"] == "ss"
    }
    assert len(scalar_entries) == 238
    for scalar_entry in BINARY_SCALAR_METAL_CONTRACT["entries"]:
        complete_entry = scalar_entries[scalar_entry["entryPoint"]]
        assert {
            key: complete_entry[key]
            for key in (
                "entryPoint",
                "operator",
                "inputType",
                "outputType",
                "family",
                "sha256",
                "sizeBytes",
            )
        } == scalar_entry


def _partition_binary_metal_workloads(
    workloads: tuple[BinaryMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[BinaryMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX binary Metal shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX binary Metal shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX binary Metal shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_binary_metal_workloads() -> tuple[BinaryMetalWorkload, ...]:
    raw_index = os.environ.get(BINARY_METAL_SHARD_INDEX_ENV)
    raw_count = os.environ.get(BINARY_METAL_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return BINARY_METAL_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{BINARY_METAL_SHARD_INDEX_ENV} and {BINARY_METAL_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError("MLX binary Metal shard values must be integers") from error
    try:
        return _partition_binary_metal_workloads(
            BINARY_METAL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_BINARY_METAL_WORKLOADS = _current_binary_metal_workloads()


def test_current_mlx_binary_metal_ci_shards_are_complete_and_disjoint():
    shards = tuple(
        _partition_binary_metal_workloads(
            BINARY_METAL_WORKLOADS,
            shard_index,
            BINARY_METAL_CI_SHARD_COUNT,
        )
        for shard_index in range(BINARY_METAL_CI_SHARD_COUNT)
    )

    assert [len(shard) for shard in shards] == [172] * 18 + [171] * 6
    for shard_index, shard in enumerate(shards):
        assert shard == BINARY_METAL_WORKLOADS[shard_index::BINARY_METAL_CI_SHARD_COUNT]
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 4122
    assert len(set(entry_points)) == 4122
    assert set(entry_points) == {
        workload.entry_point for workload in BINARY_METAL_WORKLOADS
    }


def _project_config(workload: BinaryMetalWorkload) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_BINARY_SOURCE}"]
        include_dirs = ["."]
        targets = ["metal"]
        output_dir = ".crosstl-mlx-binary-metal-roundtrip/out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_BINARY_SOURCE}" = "{workload.entry_point}"

        [project.entry_workgroup_size_rules."{MLX_BINARY_SOURCE}"]
        "{workload.entry_point}" = [1, 1, 1]

        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 4096
        """).strip()


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_BINARY_METAL_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_BINARY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX binary source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_BINARY_SHA256
    return mlx_root


def _materialization_parameters(workload: BinaryMetalWorkload) -> tuple[dict, dict]:
    spec = BINARY_SHAPE_SPECS[workload.shape]
    substitutions = {
        "entry.inputType": workload.input_type,
        "entry.outputType": workload.output_type,
        "entry.operator": workload.operator_type,
        "WorkPerThread<T>::n": f"WorkPerThread<{workload.input_type}>::n",
    }
    parameters = {
        name: substitutions.get(value, value)
        for name, value, _source in spec.parameter_values
    }
    parameter_sources = {name: source for name, _value, source in spec.parameter_values}
    return parameters, parameter_sources


def _expected_materializations(workload: BinaryMetalWorkload) -> list[dict]:
    spec = BINARY_SHAPE_SPECS[workload.shape]
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
    return records


def _translate_binary_metal_artifact(
    mlx_root: Path,
    work_dir: Path,
    workload: BinaryMetalWorkload,
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
    artifact = payload["artifacts"][0]
    assert artifact["source"] == MLX_BINARY_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_BINARY_SHA256,
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
    assert materialization["status"] == "materialized"
    expected_materializations = _expected_materializations(workload)
    assert materialization["specializationCount"] == len(expected_materializations)
    assert materialization["specializations"] == expected_materializations
    assert materialization["unsupported"] == []
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert generated.count("kernel void ") == 1
    assert f"kernel void {workload.entry_point}" in generated
    assert generated.count(f"struct {workload.operator_type} {{") == 1
    selected_implementation = re.compile(
        rf"(?m)^[A-Za-z_][A-Za-z0-9_]*\s+"
        rf"{re.escape(workload.operator_type)}__operator_call"
        rf"(?:__[A-Za-z0-9_]+)*(?<!__temporary)\("
    )
    assert len(selected_implementation.findall(generated)) == 1
    for pruned_operator in BINARY_METAL_OPERATOR_TYPES - {workload.operator_type}:
        assert (
            re.search(
                rf"(?m)^[A-Za-z_][A-Za-z0-9_]*\s+"
                rf"{re.escape(pruned_operator)}__operator_call"
                rf"(?:__[A-Za-z0-9_]+)*\(",
                generated,
            )
            is None
        )
        marker = f"struct {pruned_operator} {{"
        cursor = 0
        while (start := generated.find(marker, cursor)) != -1:
            end = generated.find("};", start + len(marker))
            assert end != -1
            assert generated[start + len(marker) : end].strip() == ""
            cursor = end + 2
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


def _roundtrip_pinned_mlx_binary_through_metal(
    workload: BinaryMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-binary-{workload.entry_point}-metal-roundtrip-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        report_path, generated_path = _translate_binary_metal_artifact(
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
        expected_resources = {
            resource["name"]: (
                resource["kind"],
                resource["binding"],
                resource["access"],
                workload.entry_point,
            )
            for resource in _resources(BINARY_SHAPE_SPECS[workload.shape].resource_kind)
        }
        assert {
            resource["name"]: (
                resource["kind"],
                resource["binding"],
                resource["access"],
                resource["metadata"]["entryPoint"],
            )
            for resource in reflected["resources"]
        } == expected_resources

        xcrun = shutil.which("xcrun")
        if xcrun is None:
            message = "xcrun is required for the MLX binary Metal proof"
            if os.environ.get(REQUIRE_BINARY_METAL_ENV) == "1":
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
    CURRENT_BINARY_METAL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_binary_family_roundtrips_through_metal(workload):
    _roundtrip_pinned_mlx_binary_through_metal(workload)
