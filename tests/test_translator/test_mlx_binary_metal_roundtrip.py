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
BINARY_METAL_CI_SHARD_COUNT = 3
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


@dataclass(frozen=True)
class BinaryMetalWorkload:
    entry_point: str
    operator_type: str
    input_type: str
    output_type: str
    family: str
    sha256: str
    size_bytes: int


def _load_binary_scalar_metal_contract() -> dict:
    contract_bytes = BINARY_SCALAR_METAL_CONTRACT_PATH.read_bytes()
    assert hashlib.sha256(contract_bytes).hexdigest() == (
        BINARY_SCALAR_METAL_CONTRACT_SHA256
    )
    return json.loads(contract_bytes)


BINARY_SCALAR_METAL_CONTRACT = _load_binary_scalar_metal_contract()
BINARY_SCALAR_METAL_ENTRIES = tuple(BINARY_SCALAR_METAL_CONTRACT["entries"])
BINARY_SCALAR_METAL_OPERATOR_TYPES = frozenset(
    entry["operator"] for entry in BINARY_SCALAR_METAL_ENTRIES
)
BINARY_SCALAR_METAL_WORKLOADS = tuple(
    BinaryMetalWorkload(
        entry_point=entry["entryPoint"],
        operator_type=entry["operator"],
        input_type=entry["inputType"],
        output_type=entry["outputType"],
        family=entry["family"],
        sha256=entry["sha256"],
        size_bytes=entry["sizeBytes"],
    )
    for entry in BINARY_SCALAR_METAL_ENTRIES
)


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
        return BINARY_SCALAR_METAL_WORKLOADS
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
            BINARY_SCALAR_METAL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_BINARY_METAL_WORKLOADS = _current_binary_metal_workloads()


def test_current_mlx_binary_scalar_metal_contract_is_complete_and_classified():
    expected_classifications = {
        "operators": {
            "Add": 13,
            "ArcTan2": 3,
            "BitwiseAnd": 9,
            "BitwiseOr": 9,
            "BitwiseXor": 9,
            "Divide": 13,
            "Equal": 13,
            "Greater": 13,
            "GreaterEqual": 13,
            "LeftShift": 8,
            "Less": 13,
            "LessEqual": 13,
            "LogAddExp": 4,
            "LogicalAnd": 1,
            "LogicalOr": 1,
            "Maximum": 13,
            "Minimum": 13,
            "Multiply": 13,
            "NaNEqual": 4,
            "NotEqual": 13,
            "Power": 13,
            "Remainder": 13,
            "RightShift": 8,
            "Subtract": 13,
        },
        "typePairs": {
            "bfloat16_t->bfloat16_t": 10,
            "bfloat16_t->bool": 7,
            "bool->bool": 19,
            "complex64_t->bool": 7,
            "complex64_t->complex64_t": 9,
            "float->bool": 7,
            "float->float": 10,
            "half->bool": 7,
            "half->half": 10,
            "int16_t->bool": 6,
            "int16_t->int16_t": 13,
            "int32_t->bool": 6,
            "int32_t->int32_t": 13,
            "int64_t->bool": 6,
            "int64_t->int64_t": 13,
            "int8_t->bool": 6,
            "int8_t->int8_t": 13,
            "uint16_t->bool": 6,
            "uint16_t->uint16_t": 13,
            "uint32_t->bool": 6,
            "uint32_t->uint32_t": 13,
            "uint64_t->bool": 6,
            "uint64_t->uint64_t": 13,
            "uint8_t->bool": 6,
            "uint8_t->uint8_t": 13,
        },
        "families": {
            "bfloat16-comparison": 7,
            "bfloat16-same-type": 10,
            "boolean-same-type": 19,
            "complex64-comparison": 7,
            "complex64-same-type": 9,
            "float16-comparison": 7,
            "float16-same-type": 10,
            "float32-comparison": 7,
            "float32-same-type": 10,
            "int16-comparison": 6,
            "int16-same-type": 13,
            "int32-comparison": 6,
            "int32-same-type": 13,
            "int64-comparison": 6,
            "int64-same-type": 13,
            "int8-comparison": 6,
            "int8-same-type": 13,
            "uint16-comparison": 6,
            "uint16-same-type": 13,
            "uint32-comparison": 6,
            "uint32-same-type": 13,
            "uint64-comparison": 6,
            "uint64-same-type": 13,
            "uint8-comparison": 6,
            "uint8-same-type": 13,
        },
    }
    expected_shape_contract = {
        "sourceTemplateArgumentCount": 3,
        "templateParameters": {
            "T": {"value": "entry.inputType", "source": "source-instantiation"},
            "U": {"value": "entry.outputType", "source": "source-instantiation"},
            "Op": {"value": "entry.operator", "source": "source-instantiation"},
        },
        "specializationCountPerArtifact": 1,
        "reachableSpecializations": ["binary_ss"],
        "hostResourceCountPerArtifact": 3,
        "hostResources": [
            {"name": "a", "kind": "buffer", "binding": 0, "access": "read"},
            {"name": "b", "kind": "buffer", "binding": 1, "access": "read"},
            {
                "name": "c",
                "kind": "buffer",
                "binding": 2,
                "access": "read_write",
            },
        ],
    }
    expected_artifact_contract = {
        "artifactCount": 238,
        "artifactCountPerEntry": 1,
        "specializationCount": 238,
        "unsupportedSpecializationCount": 0,
        "selectedOperatorImplementationCountPerArtifact": 1,
        "unselectedOperatorBodiesPruned": True,
        "reachableKernelCountPerArtifact": 1,
        "provenance": "entry-scoped-translate",
        "intermediate": "crossgl",
        "hostInterfaceStatus": "ready",
        "hostResourceCountPerArtifact": 3,
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "generatedSizeBytesTotal": 189047,
        "generatedSizeRange": {
            "minimum": {"entryPoint": "ss_Addint8", "sizeBytes": 668},
            "maximum": {
                "entryPoint": "ss_LogAddExpcomplex64",
                "sizeBytes": 3490,
            },
        },
        "nativeCompiler": "xcrun -sdk macosx metal -Werror -c",
        "requiresNonemptyAirArtifact": True,
    }

    contract = BINARY_SCALAR_METAL_CONTRACT
    assert contract["schemaVersion"] == 1
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_BINARY_SOURCE
    assert contract["sourceSha256"] == MLX_BINARY_SHA256
    assert contract["target"] == "metal"
    assert contract["selection"] == {
        "entryPrefix": "ss_",
        "templateName": "binary_ss",
        "shape": "scalar-scalar",
        "entryCount": 238,
        "discoveredBinaryEntryCount": 4122,
        "operatorCount": 24,
        "typePairCount": 25,
        "familyCount": 25,
    }
    assert contract["shapeContract"] == expected_shape_contract
    assert contract["classifications"] == expected_classifications
    assert contract["artifactContract"] == expected_artifact_contract

    entries = BINARY_SCALAR_METAL_ENTRIES
    assert len(entries) == 238
    assert len({entry["entryPoint"] for entry in entries}) == 238
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert Counter(entry["operator"] for entry in entries) == (
        expected_classifications["operators"]
    )
    assert Counter(
        f'{entry["inputType"]}->{entry["outputType"]}' for entry in entries
    ) == expected_classifications["typePairs"]
    assert Counter(entry["family"] for entry in entries) == (
        expected_classifications["families"]
    )
    for entry in entries:
        assert list(entry) == [
            "entryPoint",
            "operator",
            "inputType",
            "outputType",
            "family",
            "sha256",
            "sizeBytes",
        ]
        assert entry["entryPoint"].startswith("ss_")
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
        assert entry["sizeBytes"] > 0


def test_current_mlx_binary_scalar_metal_ci_shards_are_complete_and_disjoint():
    shards = tuple(
        _partition_binary_metal_workloads(
            BINARY_SCALAR_METAL_WORKLOADS,
            shard_index,
            BINARY_METAL_CI_SHARD_COUNT,
        )
        for shard_index in range(BINARY_METAL_CI_SHARD_COUNT)
    )

    assert [len(shard) for shard in shards] == [80, 79, 79]
    for shard_index, shard in enumerate(shards):
        assert shard == BINARY_SCALAR_METAL_WORKLOADS[
            shard_index::BINARY_METAL_CI_SHARD_COUNT
        ]
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 238
    assert len(set(entry_points)) == 238
    assert set(entry_points) == {
        workload.entry_point for workload in BINARY_SCALAR_METAL_WORKLOADS
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


def _expected_materialization(workload: BinaryMetalWorkload) -> dict:
    return {
        "name": "binary_ss",
        "materializedName": workload.entry_point,
        "parameters": {
            "Op": workload.operator_type,
            "T": workload.input_type,
            "U": workload.output_type,
        },
        "parameterSources": {
            "Op": "source-instantiation",
            "T": "source-instantiation",
            "U": "source-instantiation",
        },
        "source": "source-instantiation",
        "hostName": workload.entry_point,
    }


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
    assert materialization["specializationCount"] == 1
    assert materialization["specializations"] == [
        _expected_materialization(workload)
    ]
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
    for pruned_operator in BINARY_SCALAR_METAL_OPERATOR_TYPES - {
        workload.operator_type
    }:
        assert re.search(
            rf"(?m)^[A-Za-z_][A-Za-z0-9_]*\s+"
            rf"{re.escape(pruned_operator)}__operator_call"
            rf"(?:__[A-Za-z0-9_]+)*\(",
            generated,
        ) is None
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
        assert {
            resource["name"]: (
                resource["kind"],
                resource["binding"],
                resource["access"],
                resource["metadata"]["entryPoint"],
            )
            for resource in reflected["resources"]
        } == {
            "a": ("buffer", 0, "read", workload.entry_point),
            "b": ("buffer", 1, "read", workload.entry_point),
            "c": ("buffer", 2, "read_write", workload.entry_point),
        }

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
def test_current_mlx_binary_scalar_family_roundtrips_through_metal(workload):
    _roundtrip_pinned_mlx_binary_through_metal(workload)
