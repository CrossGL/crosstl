from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

from crosstl.project import (
    build_runtime_artifact_manifest,
    load_project_config,
    translate_project,
    validate_project_report,
)
from tests.test_translator.test_mlx_binary_complete_metal_roundtrip import (
    BINARY_METAL_CONTRACT,
    BINARY_METAL_OPERATOR_TYPES,
    EXPECTED_BINARY_CLASSIFICATIONS,
    MLX_BINARY_SHA256,
    MLX_BINARY_SOURCE,
    MLX_COMMIT,
    BinaryMetalWorkload,
    _expected_materializations,
)

REQUIRE_BINARY_OPENGL_ENV = "CROSTL_REQUIRE_MLX_BINARY_OPENGL_TRANSLATION"
BINARY_OPENGL_SHARD_INDEX_ENV = "CROSTL_MLX_BINARY_OPENGL_SHARD_INDEX"
BINARY_OPENGL_SHARD_COUNT_ENV = "CROSTL_MLX_BINARY_OPENGL_SHARD_COUNT"
BINARY_OPENGL_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
BINARY_OPENGL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "binary.opengl-translation.json"
)
BINARY_OPENGL_CONTRACT_SHA256 = (
    "041504d65e8a73f2da0fcf4d89330065fa9fd0d0260613067c06202e8ab20f67"
)
BINARY_OPENGL_CONTRACT_SIZE_BYTES = 1467711
BINARY_OPENGL_GENERATED_SIZE_BYTES_TOTAL = 16276504
BINARY_OPENGL_GENERATED_SIZE_MINIMUM = ("ss_Addint32", 2875)
BINARY_OPENGL_GENERATED_SIZE_MAXIMUM = ("gn4large_LogAddExpcomplex64", 8526)
INDEX_RANGE_ASSERTIONS = (
    ("offset + i", 0, 2147483647),
    ("a_idx", 0, 2147483647),
    ("b_idx", 0, 2147483647),
    ("out_idx", 0, 2147483647),
    ("out_idx++", 0, 2147483647),
    ("idx.x", 0, 2147483647),
    ("idx.y", 0, 2147483647),
)


def _load_contract(path: Path, expected_sha256: str) -> dict:
    contract_bytes = path.read_bytes()
    assert len(contract_bytes) == BINARY_OPENGL_CONTRACT_SIZE_BYTES
    assert hashlib.sha256(contract_bytes).hexdigest() == expected_sha256
    return json.loads(contract_bytes)


BINARY_OPENGL_CONTRACT = _load_contract(
    BINARY_OPENGL_CONTRACT_PATH,
    BINARY_OPENGL_CONTRACT_SHA256,
)
BINARY_OPENGL_ENTRIES = tuple(BINARY_OPENGL_CONTRACT["entries"])
BINARY_OPENGL_WORKLOADS = tuple(
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
    for entry in BINARY_OPENGL_ENTRIES
)


def _partition_workloads(
    workloads: tuple[BinaryMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[BinaryMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX binary OpenGL shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX binary OpenGL shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX binary OpenGL shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_workloads() -> tuple[BinaryMetalWorkload, ...]:
    raw_index = os.environ.get(BINARY_OPENGL_SHARD_INDEX_ENV)
    raw_count = os.environ.get(BINARY_OPENGL_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return BINARY_OPENGL_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{BINARY_OPENGL_SHARD_INDEX_ENV} and {BINARY_OPENGL_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError("MLX binary OpenGL shard values must be integers") from error
    try:
        return _partition_workloads(
            BINARY_OPENGL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_BINARY_OPENGL_WORKLOADS = _current_workloads()


def _target_shape_contracts() -> dict[str, object]:
    contracts = json.loads(json.dumps(BINARY_METAL_CONTRACT["shapeContracts"]))
    target_names = {
        "a": "aBuffer",
        "b": "bBuffer",
        "c": "cBuffer",
        "size": "{sanitizedEntryPoint}_size_Args",
        "a_stride": "{sanitizedEntryPoint}_a_stride_Args",
        "b_stride": "{sanitizedEntryPoint}_b_stride_Args",
        "a_strides": "a_stridesBuffer",
        "b_strides": "b_stridesBuffer",
        "shape": "shapeBuffer",
        "ndim": "{sanitizedEntryPoint}_ndim_Args",
    }
    array_resources = {"a_strides", "b_strides", "shape"}
    for shape_contract in contracts.values():
        for resource in shape_contract["hostResources"]:
            source_name = resource["name"]
            resource["sourceName"] = source_name
            resource["name"] = target_names[source_name]
            if source_name in array_resources:
                resource["kind"] = "buffer"
    return contracts


def test_current_mlx_binary_opengl_contract_is_complete_and_classified() -> None:
    contract = BINARY_OPENGL_CONTRACT
    assert contract["schemaVersion"] == 2
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_BINARY_SOURCE
    assert contract["sourceSha256"] == MLX_BINARY_SHA256
    assert contract["target"] == "opengl"
    assert contract["selection"] == {
        "entryCount": 4122,
        "shapeCount": 18,
        "templateCount": 11,
        "operatorCount": 24,
        "typePairCount": 25,
        "familyCount": 25,
        "allDiscoveredSourceInstantiationsIncluded": True,
    }
    assert contract["selection"] == BINARY_METAL_CONTRACT["selection"]
    assert contract["classifications"] == EXPECTED_BINARY_CLASSIFICATIONS
    assert contract["classifications"] == BINARY_METAL_CONTRACT["classifications"]
    assert contract["shapeContracts"] == _target_shape_contracts()
    assert contract["portabilityPreconditions"] == {
        "indexRangeAssertions": [
            {
                "source": MLX_BINARY_SOURCE,
                "expression": expression,
                "minimum": minimum,
                "maximum": maximum,
            }
            for expression, minimum, maximum in INDEX_RANGE_ASSERTIONS
        ],
        "contractKind": "explicit-host-runtime-portability-preconditions",
        "inferred": False,
        "runtimeEnforced": False,
    }
    assert contract["artifactContract"] == {
        "artifactCount": 4122,
        "artifactCountPerEntry": 1,
        "specializationCount": 6026,
        "specializationCountsByShape": BINARY_METAL_CONTRACT["artifactContract"][
            "specializationCountsByShape"
        ],
        "unsupportedSpecializationCount": 0,
        "selectedOperatorImplementationCountPerArtifact": 1,
        "unselectedOperatorBodiesPruned": True,
        "reachableKernelCountPerArtifact": 1,
        "provenance": "entry-scoped-translate",
        "intermediate": "crossgl",
        "hostInterfaceStatus": "ready",
        "reflectedResourceCount": 19106,
        "reflectedResourceCountsByShape": BINARY_METAL_CONTRACT["artifactContract"][
            "reflectedResourceCountsByShape"
        ],
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "generatedSizeBytesTotal": BINARY_OPENGL_GENERATED_SIZE_BYTES_TOTAL,
        "generatedSizeRange": {
            "minimum": {
                "entryPoint": BINARY_OPENGL_GENERATED_SIZE_MINIMUM[0],
                "sizeBytes": BINARY_OPENGL_GENERATED_SIZE_MINIMUM[1],
            },
            "maximum": {
                "entryPoint": BINARY_OPENGL_GENERATED_SIZE_MAXIMUM[0],
                "sizeBytes": BINARY_OPENGL_GENERATED_SIZE_MAXIMUM[1],
            },
        },
        "nativeCompiler": (
            "glslangValidator --target-env opengl --target-env spirv1.3 -S comp"
        ),
        "targetEntryPoint": "main",
        "nativeValidator": "spirv-val --target-env spv1.3",
        "requiresNonemptySpirvArtifact": True,
    }
    assert len(BINARY_OPENGL_ENTRIES) == 4122
    assert len({entry["entryPoint"] for entry in BINARY_OPENGL_ENTRIES}) == 4122
    assert [entry["entryPoint"] for entry in BINARY_OPENGL_ENTRIES] == sorted(
        entry["entryPoint"] for entry in BINARY_OPENGL_ENTRIES
    )
    assert all(
        list(entry)
        == [
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
        for entry in BINARY_OPENGL_ENTRIES
    )
    for classification_name, expected in EXPECTED_BINARY_CLASSIFICATIONS.items():
        observed: dict[str, int] = {}
        entry_key = {
            "shapes": "shape",
            "templates": "templateName",
            "operators": "operator",
            "typePairs": None,
            "families": "family",
        }[classification_name]
        for entry in BINARY_OPENGL_ENTRIES:
            value = (
                f"{entry['inputType']}->{entry['outputType']}"
                if entry_key is None
                else entry[entry_key]
            )
            observed[value] = observed.get(value, 0) + 1
        assert observed == expected


def test_current_mlx_binary_opengl_ci_shards_are_complete_and_disjoint() -> None:
    shards = tuple(
        _partition_workloads(
            BINARY_OPENGL_WORKLOADS,
            shard_index,
            BINARY_OPENGL_CI_SHARD_COUNT,
        )
        for shard_index in range(BINARY_OPENGL_CI_SHARD_COUNT)
    )
    assert [len(shard) for shard in shards] == [172] * 18 + [171] * 6
    for shard_index, shard in enumerate(shards):
        assert (
            shard == BINARY_OPENGL_WORKLOADS[shard_index::BINARY_OPENGL_CI_SHARD_COUNT]
        )
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 4122
    assert len(set(entry_points)) == 4122
    assert set(entry_points) == {
        workload.entry_point for workload in BINARY_OPENGL_WORKLOADS
    }


def _project_config(workload: BinaryMetalWorkload) -> str:
    assertions = "\n\n".join(textwrap.dedent(f"""
            [[project.index_range_assertions]]
            source = "{MLX_BINARY_SOURCE}"
            expression = "{expression}"
            minimum = {minimum}
            maximum = {maximum}
            """).strip() for expression, minimum, maximum in INDEX_RANGE_ASSERTIONS)
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_BINARY_SOURCE}"]
        include_dirs = ["."]
        targets = ["opengl"]
        output_dir = "out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_BINARY_SOURCE}" = "{workload.entry_point}"

        [project.entry_workgroup_size_rules."{MLX_BINARY_SOURCE}"]
        "{workload.entry_point}" = [1, 1, 1]

        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 4096

        {assertions}
        """).strip()


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_BINARY_OPENGL_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")
    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_BINARY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX binary source is missing: {source_path}")
    checkout = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checkout.returncode == 0, checkout.stderr
    assert checkout.stdout.strip() == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_BINARY_SHA256
    return mlx_root


def _required_tool(name: str) -> str:
    path = shutil.which(name)
    if path is not None:
        return path
    message = f"{name} is required for the complete MLX binary OpenGL proof"
    if os.environ.get(REQUIRE_BINARY_OPENGL_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _expected_resources(workload: BinaryMetalWorkload) -> dict[str, tuple]:
    sanitized_entry = workload.entry_point.rstrip("_")
    resources = {}
    shape_contract = BINARY_OPENGL_CONTRACT["shapeContracts"][workload.shape]
    for resource in shape_contract["hostResources"]:
        name = resource["name"].replace("{sanitizedEntryPoint}", sanitized_entry)
        resources[name] = (
            resource["kind"],
            resource["binding"],
            resource["access"],
        )
    return resources


def _translate_and_validate(
    mlx_root: Path,
    work_dir: Path,
    workload: BinaryMetalWorkload,
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
    assert payload["project"]["indexRangeAssertions"] == [
        {
            "source": MLX_BINARY_SOURCE,
            "expression": expression,
            "minimum": minimum,
            "maximum": maximum,
        }
        for expression, minimum, maximum in INDEX_RANGE_ASSERTIONS
    ]
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
        "target": "main",
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
    assert execution_entries[0]["targetEntryPoint"] == "main"
    assert execution_entries[0]["workgroupSize"] == [1, 1, 1]
    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    expected_materializations = _expected_materializations(workload)
    assert materialization["specializations"] == expected_materializations
    assert materialization["specializationCount"] == len(expected_materializations)
    assert materialization["unsupported"] == []
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert (
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in generated
    )
    assert generated.count("void main()") == 1
    selected_implementation = re.compile(
        rf"(?m)^[A-Za-z_][A-Za-z0-9_]*\s+"
        rf"{re.escape(workload.operator_type)}_operator_call"
        rf"(?:_[A-Za-z0-9_]+)*(?<!_temporary)"
        r"\([^;\n]*\)\s*\{$"
    )
    assert len(selected_implementation.findall(generated)) == 1
    defined_operator_bodies = {
        operator
        for operator in BINARY_METAL_OPERATOR_TYPES
        if any(
            f" {operator}_operator_call" in line
            and "_temporary(" not in line
            and line.rstrip().endswith("{")
            for line in generated.splitlines()
        )
    }
    expected_operator_bodies = {workload.operator_type}
    assert defined_operator_bodies == expected_operator_bodies
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
    assert {
        resource["name"]: (
            resource["kind"],
            resource["binding"],
            resource["access"],
        )
        for resource in reflected["resources"]
    } == _expected_resources(workload)

    glslang = _required_tool("glslangValidator")
    spirv_val = _required_tool("spirv-val")
    spirv_path = work_dir / f"{workload.entry_point}.spv"
    compilation = subprocess.run(
        [
            glslang,
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
        timeout=120,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    assert spirv_path.is_file()
    assert spirv_path.stat().st_size > 0
    validation = subprocess.run(
        [spirv_val, "--target-env", "spv1.3", str(spirv_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr


@pytest.mark.parametrize(
    "workload",
    CURRENT_BINARY_OPENGL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_binary_family_translates_to_opengl(
    workload: BinaryMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-binary-opengl-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_and_validate(
            mlx_root,
            Path(temporary_directory),
            workload,
        )
