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

REQUIRE_COPY_OPENGL_ENV = "CROSTL_REQUIRE_MLX_COPY_OPENGL_TRANSLATION"
COPY_OPENGL_SHARD_INDEX_ENV = "CROSTL_MLX_COPY_OPENGL_SHARD_INDEX"
COPY_OPENGL_SHARD_COUNT_ENV = "CROSTL_MLX_COPY_OPENGL_SHARD_COUNT"
COPY_OPENGL_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
COPY_OPENGL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "copy.opengl-translation.json"
)
COPY_OPENGL_CONTRACT_SHA256 = (
    "64a709bc0bbf59598cbbcd94104ee918830fd58e9a62d950b76e76dd5fc06650"
)
COPY_OPENGL_CONTRACT_SIZE_BYTES = 874507
COPY_OPENGL_GENERATED_SIZE_BYTES_TOTAL = 5702794
COPY_OPENGL_GENERATED_SIZE_MINIMUM = ("s_copybool_bool_", 1585)
COPY_OPENGL_GENERATED_SIZE_MAXIMUM = (
    "ggn4large_dynamic_copycomplex64complex64",
    3971,
)
INDEX_RANGE_ASSERTIONS = (
    ("offset + i", 0, 2147483647),
    ("src_idx", 0, 2147483647),
    ("dst_idx", 0, 2147483647),
    ("dst_idx + i", 0, 2147483647),
    ("src_idx + src_offset", 0, 2147483647),
    ("dst_idx + dst_offset", 0, 2147483647),
    ("idx.x", 0, 2147483647),
    ("idx.y", 0, 2147483647),
)
CONTRACT_RESOURCE_ABI_FIELDS = (
    "name",
    "kind",
    "set",
    "binding",
    "access",
    "type",
)
ARRAY_RESOURCES = {"src_shape", "src_strides", "dst_strides"}
TARGET_RESOURCE_NAMES = {
    "src": "srcBuffer",
    "dst": "dstBuffer",
    "size": "{sanitizedEntryPoint}_size_Args",
    "src_stride": "{sanitizedEntryPoint}_src_stride_Args",
    "dst_stride": "{sanitizedEntryPoint}_dst_stride_Args",
    "src_shape": "src_shapeBuffer",
    "src_strides": "src_stridesBuffer",
    "dst_strides": "dst_stridesBuffer",
    "ndim": "{sanitizedEntryPoint}_ndim_Args",
    "src_offset": "{sanitizedEntryPoint}_src_offset_Args",
    "dst_offset": "{sanitizedEntryPoint}_dst_offset_Args",
}


def _load_contract(path: Path, expected_sha256: str) -> dict:
    contract_bytes = path.read_bytes()
    assert len(contract_bytes) == COPY_OPENGL_CONTRACT_SIZE_BYTES
    assert hashlib.sha256(contract_bytes).hexdigest() == expected_sha256
    return json.loads(contract_bytes)


COPY_OPENGL_CONTRACT = _load_contract(
    COPY_OPENGL_CONTRACT_PATH,
    COPY_OPENGL_CONTRACT_SHA256,
)
COPY_OPENGL_ENTRIES = tuple(COPY_OPENGL_CONTRACT["entries"])
COPY_OPENGL_WORKLOADS = tuple(
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
    for entry in COPY_OPENGL_ENTRIES
)


def _partition_workloads(
    workloads: tuple[CopyMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[CopyMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX copy OpenGL shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX copy OpenGL shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX copy OpenGL shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_workloads() -> tuple[CopyMetalWorkload, ...]:
    raw_index = os.environ.get(COPY_OPENGL_SHARD_INDEX_ENV)
    raw_count = os.environ.get(COPY_OPENGL_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return COPY_OPENGL_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{COPY_OPENGL_SHARD_INDEX_ENV} and {COPY_OPENGL_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError("MLX copy OpenGL shard values must be integers") from error
    try:
        return _partition_workloads(
            COPY_OPENGL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_COPY_OPENGL_WORKLOADS = _current_workloads()


def _target_shape_contracts() -> dict[str, object]:
    contracts = json.loads(json.dumps(COPY_METAL_CONTRACT["shapeContracts"]))
    for shape_contract in contracts.values():
        for resource in shape_contract["hostResources"]:
            source_name = resource["name"]
            target_name = TARGET_RESOURCE_NAMES[source_name]
            resource["sourceName"] = source_name
            resource["name"] = target_name
            resource["type"] = target_name
            if source_name in ARRAY_RESOURCES:
                resource["kind"] = "buffer"
    return contracts


def test_current_mlx_copy_opengl_contract_is_complete_and_classified() -> None:
    contract = COPY_OPENGL_CONTRACT
    assert contract["schemaVersion"] == 2
    assert contract["kind"] == "crosstl-mlx-copy-opengl-translation-contract"
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_COPY_SOURCE
    assert contract["sourceSha256"] == MLX_COPY_SHA256
    assert contract["target"] == "opengl"
    assert contract["selection"] == COPY_METAL_CONTRACT["selection"]
    assert contract["classifications"] == EXPECTED_COPY_CLASSIFICATIONS
    assert contract["classifications"] == COPY_METAL_CONTRACT["classifications"]
    assert contract["shapeContracts"] == _target_shape_contracts()
    assert contract["portabilityPreconditions"] == {
        "indexRangeAssertions": [
            {
                "source": MLX_COPY_SOURCE,
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
        "targetEntryPoint": "main",
        "reflectedResourceCount": 8684,
        "reflectedResourceCountsByShape": COPY_METAL_CONTRACT["artifactContract"][
            "reflectedResourceCountsByShape"
        ],
        "resourceAbiFields": list(CONTRACT_RESOURCE_ABI_FIELDS),
        "exactTargetResourceTypesIncluded": True,
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "generatedSizeBytesTotal": COPY_OPENGL_GENERATED_SIZE_BYTES_TOTAL,
        "generatedSizeRange": {
            "minimum": {
                "entryPoint": COPY_OPENGL_GENERATED_SIZE_MINIMUM[0],
                "sizeBytes": COPY_OPENGL_GENERATED_SIZE_MINIMUM[1],
            },
            "maximum": {
                "entryPoint": COPY_OPENGL_GENERATED_SIZE_MAXIMUM[0],
                "sizeBytes": COPY_OPENGL_GENERATED_SIZE_MAXIMUM[1],
            },
        },
        "nativeCompiler": (
            "glslangValidator --target-env opengl --target-env spirv1.3 -S comp"
        ),
        "nativeValidator": "spirv-val --target-env spv1.3",
        "requiresNonemptySpirvArtifact": True,
    }

    entries = COPY_OPENGL_ENTRIES
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
        COPY_OPENGL_GENERATED_SIZE_BYTES_TOTAL
    )
    assert min((entry["sizeBytes"], entry["entryPoint"]) for entry in entries) == (
        COPY_OPENGL_GENERATED_SIZE_MINIMUM[1],
        COPY_OPENGL_GENERATED_SIZE_MINIMUM[0],
    )
    assert max((entry["sizeBytes"], entry["entryPoint"]) for entry in entries) == (
        COPY_OPENGL_GENERATED_SIZE_MAXIMUM[1],
        COPY_OPENGL_GENERATED_SIZE_MAXIMUM[0],
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
        assert entry["sizeBytes"] > 0


def test_current_mlx_copy_opengl_ci_shards_are_complete_and_disjoint() -> None:
    shards = tuple(
        _partition_workloads(
            COPY_OPENGL_WORKLOADS,
            shard_index,
            COPY_OPENGL_CI_SHARD_COUNT,
        )
        for shard_index in range(COPY_OPENGL_CI_SHARD_COUNT)
    )
    assert [len(shard) for shard in shards] == [104] * 24
    for shard_index, shard in enumerate(shards):
        assert shard == COPY_OPENGL_WORKLOADS[shard_index::COPY_OPENGL_CI_SHARD_COUNT]
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 2496
    assert len(set(entry_points)) == 2496
    assert set(entry_points) == {
        workload.entry_point for workload in COPY_OPENGL_WORKLOADS
    }


def _project_config(workload: CopyMetalWorkload) -> str:
    assertions = "\n\n".join(textwrap.dedent(f"""
            [[project.index_range_assertions]]
            source = "{MLX_COPY_SOURCE}"
            expression = "{expression}"
            minimum = {minimum}
            maximum = {maximum}
            """).strip() for expression, minimum, maximum in INDEX_RANGE_ASSERTIONS)
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_COPY_SOURCE}"]
        include_dirs = ["."]
        targets = ["opengl"]
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

        {assertions}
        """).strip()


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_COPY_OPENGL_ENV) == "1":
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
    message = f"{name} is required for the complete MLX copy OpenGL proof"
    if os.environ.get(REQUIRE_COPY_OPENGL_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _expected_resources(workload: CopyMetalWorkload) -> list[dict[str, object]]:
    sanitized_entry = workload.entry_point.rstrip("_")
    shape_contract = COPY_OPENGL_CONTRACT["shapeContracts"][workload.shape]
    return [
        {
            field: (
                str(resource[field]).replace(
                    "{sanitizedEntryPoint}",
                    sanitized_entry,
                )
                if field in {"name", "type"}
                else resource[field]
            )
            for field in CONTRACT_RESOURCE_ABI_FIELDS
        }
        for resource in shape_contract["hostResources"]
    ]


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
    assert payload["summary"]["diagnosticCounts"] == {
        "note": 0,
        "warning": 0,
        "error": 0,
    }
    assert payload["diagnostics"] == []
    assert payload["project"]["indexRangeAssertions"] == [
        {
            "source": MLX_COPY_SOURCE,
            "expression": expression,
            "minimum": minimum,
            "maximum": maximum,
        }
        for expression, minimum, maximum in INDEX_RANGE_ASSERTIONS
    ]

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
    assert (
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in generated
    )
    assert generated.count("void main()") == 1
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
    assert [
        {field: resource[field] for field in CONTRACT_RESOURCE_ABI_FIELDS}
        for resource in reflected["resources"]
    ] == _expected_resources(workload)

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
    CURRENT_COPY_OPENGL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_copy_family_translates_to_opengl(
    workload: CopyMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-copy-opengl-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_and_validate(
            mlx_root,
            Path(temporary_directory),
            workload,
        )
