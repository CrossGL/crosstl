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
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
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

REQUIRE_BINARY_DIRECTX_ENV = "CROSTL_REQUIRE_MLX_BINARY_DIRECTX_TRANSLATION"
BINARY_DIRECTX_SHARD_INDEX_ENV = "CROSTL_MLX_BINARY_DIRECTX_SHARD_INDEX"
BINARY_DIRECTX_SHARD_COUNT_ENV = "CROSTL_MLX_BINARY_DIRECTX_SHARD_COUNT"
BINARY_DIRECTX_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
BINARY_DIRECTX_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "binary.directx-translation.json"
)
BINARY_DIRECTX_CONTRACT_SHA256 = (
    "2645dbde8b1ed36875c941b9ea7ba8bb6152fb0d1956d1de42fc0ee161731cbf"
)
BINARY_DIRECTX_CONTRACT_SIZE_BYTES = 1469277
BINARY_DIRECTX_GENERATED_SIZE_BYTES_TOTAL = 11321166
BINARY_DIRECTX_GENERATED_SIZE_MINIMUM = ("ss_Addint8", 1842)
BINARY_DIRECTX_GENERATED_SIZE_MAXIMUM = ("gn4large_LogAddExpcomplex64", 7570)
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
    assert len(contract_bytes) == BINARY_DIRECTX_CONTRACT_SIZE_BYTES
    assert hashlib.sha256(contract_bytes).hexdigest() == expected_sha256
    return json.loads(contract_bytes)


BINARY_DIRECTX_CONTRACT = _load_contract(
    BINARY_DIRECTX_CONTRACT_PATH,
    BINARY_DIRECTX_CONTRACT_SHA256,
)
BINARY_DIRECTX_ENTRIES = tuple(BINARY_DIRECTX_CONTRACT["entries"])
BINARY_DIRECTX_WORKLOADS = tuple(
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
    for entry in BINARY_DIRECTX_ENTRIES
)


def _partition_workloads(
    workloads: tuple[BinaryMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[BinaryMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX binary DirectX shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX binary DirectX shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX binary DirectX shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_workloads() -> tuple[BinaryMetalWorkload, ...]:
    raw_index = os.environ.get(BINARY_DIRECTX_SHARD_INDEX_ENV)
    raw_count = os.environ.get(BINARY_DIRECTX_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return BINARY_DIRECTX_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{BINARY_DIRECTX_SHARD_INDEX_ENV} and {BINARY_DIRECTX_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError(
            "MLX binary DirectX shard values must be integers"
        ) from error
    try:
        return _partition_workloads(
            BINARY_DIRECTX_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_BINARY_DIRECTX_WORKLOADS = _current_workloads()


def _target_shape_contracts() -> dict[str, object]:
    contracts = json.loads(json.dumps(BINARY_METAL_CONTRACT["shapeContracts"]))
    target_names = {
        "a": "a",
        "b": "b",
        "c": "c",
        "size": "{sanitizedEntryPoint}_size_Constants",
        "a_stride": "{sanitizedEntryPoint}_a_stride_Constants",
        "b_stride": "{sanitizedEntryPoint}_b_stride_Constants",
        "a_strides": "a_strides",
        "b_strides": "b_strides",
        "shape": "shape",
        "ndim": "{sanitizedEntryPoint}_ndim_Constants",
    }
    array_resources = {"a_strides", "b_strides", "shape"}
    dispatch_bindings = {
        "sv2": 4,
        "vs2": 4,
        "vv2": 4,
        "g2": 0,
        "g2large": 0,
        "g3": 0,
        "g3large": 0,
        "gn2": 7,
        "gn4large": 7,
    }
    for shape, shape_contract in contracts.items():
        for resource in shape_contract["hostResources"]:
            source_name = resource["name"]
            resource["sourceName"] = source_name
            resource["name"] = target_names[source_name]
            if source_name in array_resources:
                resource["kind"] = "buffer"
        if shape in dispatch_bindings:
            shape_contract["hostResourceCountPerArtifact"] += 1
            shape_contract["hostResources"].append(
                {
                    "name": "CrossGLDispatchInfo",
                    "kind": "constant-buffer",
                    "binding": dispatch_bindings[shape],
                    "access": "read",
                    "sourceName": "generated-dispatch-workgroup-count",
                }
            )
    return contracts


def test_current_mlx_binary_directx_contract_is_complete_and_classified() -> None:
    contract = BINARY_DIRECTX_CONTRACT
    assert list(contract) == [
        "schemaVersion",
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
    ]
    assert contract["schemaVersion"] == 2
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_BINARY_SOURCE
    assert contract["sourceSha256"] == MLX_BINARY_SHA256
    assert contract["target"] == "directx"
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
        "reflectedResourceCount": 21248,
        "reflectedResourceCountsByShape": {
            "g1": 1190,
            "g1large": 1190,
            "g2": 1428,
            "g2large": 1428,
            "g3": 1428,
            "g3large": 1428,
            "gn2": 1904,
            "gn4large": 1904,
            "ss": 714,
            "sv": 952,
            "sv2": 1190,
            "svn": 736,
            "vs": 952,
            "vs2": 1190,
            "vsn": 736,
            "vv": 952,
            "vv2": 1190,
            "vvn": 736,
        },
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "generatedSizeBytesTotal": BINARY_DIRECTX_GENERATED_SIZE_BYTES_TOTAL,
        "generatedSizeRange": {
            "minimum": {
                "entryPoint": BINARY_DIRECTX_GENERATED_SIZE_MINIMUM[0],
                "sizeBytes": BINARY_DIRECTX_GENERATED_SIZE_MINIMUM[1],
            },
            "maximum": {
                "entryPoint": BINARY_DIRECTX_GENERATED_SIZE_MAXIMUM[0],
                "sizeBytes": BINARY_DIRECTX_GENERATED_SIZE_MAXIMUM[1],
            },
        },
        "nativeCompiler": "dxc -enable-16bit-types -WX -T cs_6_2 -E CSMain",
        "targetEntryPoint": "CSMain",
        "compilerArguments": ["-enable-16bit-types"],
        "requiresNonemptyDxilArtifact": True,
    }
    assert len(BINARY_DIRECTX_ENTRIES) == 4122
    assert len({entry["entryPoint"] for entry in BINARY_DIRECTX_ENTRIES}) == 4122
    assert [entry["entryPoint"] for entry in BINARY_DIRECTX_ENTRIES] == sorted(
        entry["entryPoint"] for entry in BINARY_DIRECTX_ENTRIES
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
        for entry in BINARY_DIRECTX_ENTRIES
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
        for entry in BINARY_DIRECTX_ENTRIES:
            value = (
                f"{entry['inputType']}->{entry['outputType']}"
                if entry_key is None
                else entry[entry_key]
            )
            observed[value] = observed.get(value, 0) + 1
        assert observed == expected


def test_current_mlx_binary_directx_ci_shards_are_complete_and_disjoint() -> None:
    shards = tuple(
        _partition_workloads(
            BINARY_DIRECTX_WORKLOADS,
            shard_index,
            BINARY_DIRECTX_CI_SHARD_COUNT,
        )
        for shard_index in range(BINARY_DIRECTX_CI_SHARD_COUNT)
    )
    assert [len(shard) for shard in shards] == [172] * 18 + [171] * 6
    for shard_index, shard in enumerate(shards):
        assert (
            shard
            == BINARY_DIRECTX_WORKLOADS[shard_index::BINARY_DIRECTX_CI_SHARD_COUNT]
        )
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 4122
    assert len(set(entry_points)) == 4122
    assert set(entry_points) == {
        workload.entry_point for workload in BINARY_DIRECTX_WORKLOADS
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
        targets = ["directx"]
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
        if os.environ.get(REQUIRE_BINARY_DIRECTX_ENV) == "1":
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
    message = f"{name} is required for the complete MLX binary DirectX proof"
    if os.environ.get(REQUIRE_BINARY_DIRECTX_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _expected_resources(workload: BinaryMetalWorkload) -> dict[str, tuple]:
    sanitized_entry = workload.entry_point.rstrip("_")
    resources = {}
    shape_contract = BINARY_DIRECTX_CONTRACT["shapeContracts"][workload.shape]
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
    assert materialization["status"] == "materialized"
    expected_materializations = _expected_materializations(workload)
    assert materialization["specializations"] == expected_materializations
    assert materialization["specializationCount"] == len(expected_materializations)
    assert materialization["unsupported"] == []
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "[numthreads(1, 1, 1)]" in generated
    assert generated.count("void CSMain(") == 1
    selected_implementation = re.compile(
        rf"(?m)^[A-Za-z_][A-Za-z0-9_]*\s+"
        rf"{re.escape(workload.operator_type)}__operator_call"
        rf"(?:__[A-Za-z0-9_]+)*(?<!__temporary)"
        r"\([^;\n]*\)\s*\{$"
    )
    assert len(selected_implementation.findall(generated)) == 1
    defined_operator_bodies = {
        operator
        for operator in BINARY_METAL_OPERATOR_TYPES
        if any(
            f" {operator}__operator_call" in line
            and "__temporary(" not in line
            and line.rstrip().endswith("{")
            for line in generated.splitlines()
        )
    }
    assert defined_operator_bodies == {workload.operator_type}
    for residue in (
        "template <",
        "decltype(",
        "operator()",
        "unsupported Metal",
        "fallback for unmatched generated control flow",
    ):
        assert residue not in generated
    if workload.input_type == "bfloat16_t" and workload.operator_type in {
        "ArcTan2",
        "LogAddExp",
        "Maximum",
        "Minimum",
        "Power",
    }:
        assert "__crossgl_bfloat16_to_float" in generated
    if (
        workload.input_type == "bfloat16_t"
        and workload.output_type == "bfloat16_t"
        and workload.operator_type in {"ArcTan2", "LogAddExp", "Power"}
    ):
        assert "__crossgl_bfloat16_from_float" in generated
    if (
        workload.input_type == "bfloat16_t"
        and workload.output_type == "bfloat16_t"
        and workload.operator_type in {"Maximum", "Minimum"}
    ):
        assert "__crossgl_bfloat16_from_float" not in generated
    if workload.shape in {"gn2", "gn4large"}:
        sanitized_entry = workload.entry_point.rstrip("_")
        assert f"cbuffer {sanitized_entry}_ndim_Constants : register(b6)" in generated

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
            resource["binding"],
            resource["access"],
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
    assert len(dxil_bytes) > 0


@pytest.mark.parametrize(
    "workload",
    CURRENT_BINARY_DIRECTX_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_binary_family_translates_to_directx(
    workload: BinaryMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-binary-directx-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_and_validate(
            mlx_root,
            Path(temporary_directory),
            workload,
        )
