from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from crosstl.project import (
    build_runtime_artifact_manifest,
    load_project_config,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.corpus_evidence import (
    compile_opengl_artifact,
    corpus_workspace,
)
from demos.integrations.mlx.tests.kernels.test_binary_complete_metal_roundtrip import (
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
ROOT = Path(__file__).resolve().parents[5]
BINARY_OPENGL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "binary.opengl-translation.json"
)
BINARY_OPENGL_CONTRACT_SHA256 = (
    "5e964bb82aae5c59d0a21c71eab72f3e1a6ff4d53fe3491d09ae47f77b6e1100"
)
BINARY_OPENGL_CONTRACT_SIZE_BYTES = 1467836
BINARY_OPENGL_GENERATED_SIZE_BYTES_TOTAL = 18004588
BINARY_OPENGL_GENERATED_SIZE_MINIMUM = ("ss_Addint32", 2875)
BINARY_OPENGL_GENERATED_SIZE_MAXIMUM = ("gn4large_Remainderfloat16", 19553)
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


def _project_config(workload: BinaryMetalWorkload, *, entry_points=None) -> str:
    entries = workload.entry_point if entry_points is None else list(entry_points)
    workgroup_entries = [workload.entry_point] if entry_points is None else ["*"]
    workgroup_rules = "\n        ".join(
        f"{json.dumps(entry)} = [1, 1, 1]" for entry in workgroup_entries
    )
    profiles = (
        ['binary16_remainder_profile = "binary32-quotient"']
        if workload.operator_type == "Remainder" and workload.input_type == "half"
        else []
    )
    if workload.input_type in {"float", "bfloat16_t"}:
        if workload.operator_type in {"Remainder", "Minimum", "Maximum"}:
            profiles.append('binary32_comparison_profile = "flush-subnormals"')
        if workload.operator_type == "Remainder":
            profiles.append(
                'binary32_remainder_profile = "flush-arithmetic-subnormals"'
            )
        if workload.operator_type in {"Remainder", "Add", "Subtract"}:
            profiles.append('binary32_additive_profile = "rne-flush"')
        if workload.operator_type == "Divide":
            profiles.append('binary32_division_profile = "rne-flush"')
    profile_options = "\n        ".join(profiles)
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
        "{MLX_BINARY_SOURCE}" = {json.dumps(entries)}

        [project.entry_workgroup_size_rules."{MLX_BINARY_SOURCE}"]
        {workgroup_rules}

        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 4096
        {profile_options}

        {assertions}
        """).strip()


def test_binary_half_remainder_profile_is_scoped_to_all_eighteen_shapes(tmp_path):
    selected = []
    checked_options = set()
    for workload in BINARY_OPENGL_WORKLOADS:
        text = _project_config(workload)
        selected_profile = workload.entry_point.endswith("_Remainderfloat16")
        assert (
            'binary16_remainder_profile = "binary32-quotient"' in text
        ) == selected_profile
        option_key = (workload.operator_type, workload.input_type)
        if selected_profile or option_key not in checked_options:
            config_path = tmp_path / "crosstl.toml"
            config_path.write_text(text, encoding="utf-8")
            options = load_project_config(tmp_path, config_path).source_options["metal"]
            assert options.get("binary16_remainder_profile") == (
                "binary32-quotient" if selected_profile else None
            )
            checked_options.add(option_key)
        if selected_profile:
            selected.append(workload.shape)
    assert len(selected) == 18
    assert set(selected) == set(BINARY_OPENGL_CONTRACT["shapeContracts"])


def test_binary_float_remainder_profiles_are_scoped_to_all_shapes(tmp_path):
    profiles = {
        "binary32_comparison_profile": "flush-subnormals",
        "binary32_remainder_profile": "flush-arithmetic-subnormals",
        "binary32_additive_profile": "rne-flush",
    }
    selected = []
    checked_options = set()
    for workload in BINARY_OPENGL_WORKLOADS:
        text = _project_config(workload)
        enabled = workload.entry_point.endswith(
            ("_Remainderfloat32", "_Remainderbfloat16")
        )
        floating = workload.input_type in {"float", "bfloat16_t"}
        selected_profiles = {
            "binary32_comparison_profile": (
                enabled
                or (floating and workload.operator_type in {"Minimum", "Maximum"})
            ),
            "binary32_remainder_profile": enabled,
            "binary32_additive_profile": (
                enabled or (floating and workload.operator_type in {"Add", "Subtract"})
            ),
        }
        for option, value in profiles.items():
            assert (f'{option} = "{value}"' in text) == selected_profiles[option]
        option_key = (workload.operator_type, workload.input_type)
        if enabled or option_key not in checked_options:
            path = tmp_path / "crosstl.toml"
            path.write_text(text, encoding="utf-8")
            options = load_project_config(tmp_path, path).source_options["metal"]
            for option, value in profiles.items():
                assert options.get(option) == (
                    value if selected_profiles[option] else None
                )
            checked_options.add(option_key)
        if enabled:
            selected.append((workload.shape, workload.input_type))
    assert len(selected) == 36
    assert set(selected) == {
        (shape, dtype)
        for shape in BINARY_OPENGL_CONTRACT["shapeContracts"]
        for dtype in ("float", "bfloat16_t")
    }


def test_binary_extrema_comparison_profile_is_scoped_to_all_shapes(tmp_path):
    selected = []
    for workload in BINARY_OPENGL_WORKLOADS:
        if workload.operator_type not in {"Minimum", "Maximum"}:
            continue
        path = tmp_path / "crosstl.toml"
        path.write_text(_project_config(workload), encoding="utf-8")
        options = load_project_config(tmp_path, path).source_options["metal"]
        enabled = workload.input_type in {"float", "bfloat16_t"}
        assert options.get("binary32_comparison_profile") == (
            "flush-subnormals" if enabled else None
        )
        assert "binary32_remainder_profile" not in options
        assert "binary16_remainder_profile" not in options
        assert "binary32_additive_profile" not in options
        if enabled:
            selected.append(
                (workload.operator_type, workload.shape, workload.input_type)
            )
    assert len(selected) == 72
    assert set(selected) == {
        (operation, shape, dtype)
        for operation in ("Minimum", "Maximum")
        for shape in BINARY_OPENGL_CONTRACT["shapeContracts"]
        for dtype in ("float", "bfloat16_t")
    }


def test_binary_additive_profile_is_scoped_to_all_shapes(tmp_path):
    selected = []
    for workload in BINARY_OPENGL_WORKLOADS:
        if workload.operator_type not in {"Add", "Subtract"}:
            continue
        path = tmp_path / "crosstl.toml"
        path.write_text(_project_config(workload), encoding="utf-8")
        options = load_project_config(tmp_path, path).source_options["metal"]
        enabled = workload.input_type in {"float", "bfloat16_t"}
        assert options.get("binary32_additive_profile") == (
            "rne-flush" if enabled else None
        )
        assert "binary32_comparison_profile" not in options
        assert "binary32_remainder_profile" not in options
        assert "binary16_remainder_profile" not in options
        if enabled:
            selected.append(
                (workload.operator_type, workload.shape, workload.input_type)
            )
    assert len(selected) == 72
    assert set(selected) == {
        (operation, shape, dtype)
        for operation in ("Add", "Subtract")
        for shape in BINARY_OPENGL_CONTRACT["shapeContracts"]
        for dtype in ("float", "bfloat16_t")
    }


def test_binary_division_profile_is_scoped_to_all_shapes(tmp_path):
    selected = []
    checked_options = set()
    for workload in BINARY_OPENGL_WORKLOADS:
        enabled = workload.operator_type == "Divide" and workload.input_type in {
            "float",
            "bfloat16_t",
        }
        text = _project_config(workload)
        assert ('binary32_division_profile = "rne-flush"' in text) == enabled
        key = (workload.operator_type, workload.input_type)
        if enabled or key not in checked_options:
            path = tmp_path / "crosstl.toml"
            path.write_text(text, encoding="utf-8")
            options = load_project_config(tmp_path, path).source_options["metal"]
            assert options.get("binary32_division_profile") == (
                "rne-flush" if enabled else None
            )
            if enabled:
                assert not any(
                    name in options
                    for name in (
                        "binary32_additive_profile",
                        "binary32_comparison_profile",
                        "binary32_remainder_profile",
                        "binary16_remainder_profile",
                    )
                )
            checked_options.add(key)
        if enabled:
            selected.append((workload.shape, workload.input_type))
    assert len(selected) == 36
    assert set(selected) == {
        (shape, dtype)
        for shape in BINARY_OPENGL_CONTRACT["shapeContracts"]
        for dtype in ("float", "bfloat16_t")
    }


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
    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
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
    expected_provenance = {
        "pipeline": "entry-scoped-translate",
        "intermediate": "crossgl",
    }
    if workload.operator_type == "Remainder" and workload.input_type == "half":
        expected_provenance["binary16RemainderProfile"] = "binary32-quotient"
        assert (
            payload["project"]["sourceOptions"]["metal"]["binary16_remainder_profile"]
            == "binary32-quotient"
        )
    if workload.operator_type in {"Add", "Subtract"} and workload.input_type in {
        "float",
        "bfloat16_t",
    }:
        expected_provenance["binary32AdditiveProfile"] = "rne-flush"
        assert (
            payload["project"]["sourceOptions"]["metal"]["binary32_additive_profile"]
            == "rne-flush"
        )
    if workload.operator_type in {"Minimum", "Maximum"} and workload.input_type in {
        "float",
        "bfloat16_t",
    }:
        expected_provenance["binary32ComparisonProfile"] = "flush-subnormals"
        assert (
            payload["project"]["sourceOptions"]["metal"]["binary32_comparison_profile"]
            == "flush-subnormals"
        )
    if workload.operator_type == "Divide" and workload.input_type in {
        "float",
        "bfloat16_t",
    }:
        expected_provenance["binary32DivisionProfile"] = "rne-flush"
        assert (
            payload["project"]["sourceOptions"]["metal"]["binary32_division_profile"]
            == "rne-flush"
        )
    if workload.operator_type == "Remainder" and workload.input_type in {
        "float",
        "bfloat16_t",
    }:
        for option, field, value in (
            (
                "binary32_comparison_profile",
                "binary32ComparisonProfile",
                "flush-subnormals",
            ),
            (
                "binary32_remainder_profile",
                "binary32RemainderProfile",
                "flush-arithmetic-subnormals",
            ),
            ("binary32_additive_profile", "binary32AdditiveProfile", "rne-flush"),
        ):
            expected_provenance[field] = value
            assert payload["project"]["sourceOptions"]["metal"][option] == value
    assert artifact["provenance"] == expected_provenance
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
    compile_opengl_artifact(
        generated_path,
        spirv_path,
        compiler=glslang,
        validator=spirv_val,
        work_dir=work_dir,
    )


@pytest.mark.parametrize(
    "workload",
    CURRENT_BINARY_OPENGL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_binary_family_translates_to_opengl(
    workload: BinaryMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with corpus_workspace(
        mlx_root, family="binary", target="opengl", entry_point=workload.entry_point
    ) as work_dir:
        _translate_and_validate(
            mlx_root,
            work_dir,
            workload,
        )
