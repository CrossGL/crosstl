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
MLX_UNARY_SOURCE = "mlx/backend/metal/kernels/unary.metal"
MLX_UNARY_SHA256 = "51af04126d68e1f5baee5f467268408650d24a68db66e8c044f7f0be3f15368b"
REQUIRE_UNARY_OPENGL_ENV = "CROSTL_REQUIRE_MLX_UNARY_OPENGL_TRANSLATION"
UNARY_OPENGL_SHARD_INDEX_ENV = "CROSTL_MLX_UNARY_OPENGL_SHARD_INDEX"
UNARY_OPENGL_SHARD_COUNT_ENV = "CROSTL_MLX_UNARY_OPENGL_SHARD_COUNT"
UNARY_OPENGL_CI_SHARD_COUNT = 5
ROOT = Path(__file__).resolve().parents[2]
UNARY_OPENGL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "unary.opengl-translation.json"
)
UNARY_OPENGL_CONTRACT_SHA256 = (
    "9708dd97ea36d14cb6bcf167a0153fc89b8e61d01698130722eac3733669758f"
)
UNARY_METAL_CONTRACT_PATH = (
    ROOT / "demos" / "integrations" / "mlx" / "contracts" / "unary.metal-roundtrip.json"
)
UNARY_METAL_CONTRACT_SHA256 = (
    "35a7f1de77b178cc459651431336b987a336e11827ff46b2c8ac38221ab0e741"
)
INDEX_RANGE_ASSERTIONS = (
    ("offset + i", 0, 2147483647),
    ("out_idx++", 0, 2147483647),
    ("idx", 0, 2147483647),
)


@dataclass(frozen=True)
class UnaryOpenGLWorkload:
    entry_point: str
    shape: str
    template_name: str
    operator_type: str
    input_type: str
    output_type: str
    family: str
    sha256: str
    size_bytes: int


COMPLEX_OPERATOR_DEPENDENCIES = {
    "ArcCos": frozenset({"Abs", "Log", "Sqrt"}),
    "ArcSin": frozenset({"Abs", "Log", "Sqrt"}),
    "ArcTan": frozenset({"Abs", "Log"}),
    "Log": frozenset({"Abs"}),
    "Log10": frozenset({"Abs", "Log"}),
    "Log2": frozenset({"Abs", "Log"}),
    "Rsqrt": frozenset({"Abs", "Sqrt"}),
    "Sqrt": frozenset({"Abs"}),
}


def _load_contract(path: Path, expected_sha256: str) -> dict:
    contract_bytes = path.read_bytes()
    assert hashlib.sha256(contract_bytes).hexdigest() == expected_sha256
    return json.loads(contract_bytes)


UNARY_OPENGL_CONTRACT = _load_contract(
    UNARY_OPENGL_CONTRACT_PATH,
    UNARY_OPENGL_CONTRACT_SHA256,
)
UNARY_METAL_CONTRACT = _load_contract(
    UNARY_METAL_CONTRACT_PATH,
    UNARY_METAL_CONTRACT_SHA256,
)
UNARY_OPENGL_ENTRIES = tuple(UNARY_OPENGL_CONTRACT["entries"])
UNARY_OPENGL_OPERATOR_TYPES = frozenset(
    entry["operator"] for entry in UNARY_OPENGL_ENTRIES
)
UNARY_OPENGL_WORKLOADS = tuple(
    UnaryOpenGLWorkload(
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
    for entry in UNARY_OPENGL_ENTRIES
)


def _partition_workloads(
    workloads: tuple[UnaryOpenGLWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[UnaryOpenGLWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX unary OpenGL shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX unary OpenGL shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX unary OpenGL shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_workloads() -> tuple[UnaryOpenGLWorkload, ...]:
    raw_index = os.environ.get(UNARY_OPENGL_SHARD_INDEX_ENV)
    raw_count = os.environ.get(UNARY_OPENGL_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return UNARY_OPENGL_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{UNARY_OPENGL_SHARD_INDEX_ENV} and {UNARY_OPENGL_SHARD_COUNT_ENV} "
            "must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError("MLX unary OpenGL shard values must be integers") from error
    try:
        return _partition_workloads(
            UNARY_OPENGL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_UNARY_OPENGL_WORKLOADS = _current_workloads()


def _target_shape_contracts() -> dict[str, object]:
    contracts = json.loads(json.dumps(UNARY_METAL_CONTRACT["shapeContracts"]))
    target_names = {
        "in_": "in_Buffer",
        "out_": "out_Buffer",
        "size": "{sanitizedEntryPoint}_size_Args",
        "in_shape": "in_shapeBuffer",
        "in_strides": "in_stridesBuffer",
        "ndim": "ndimBuffer",
    }
    for shape_contract in contracts.values():
        for resource in shape_contract["hostResources"]:
            source_name = resource["name"]
            resource["sourceName"] = source_name
            resource["name"] = target_names[source_name]
            if source_name in {"in_shape", "in_strides"}:
                resource["kind"] = "buffer"
    return contracts


def test_current_mlx_unary_opengl_contract_is_complete_and_classified() -> None:
    contract = UNARY_OPENGL_CONTRACT
    assert contract["schemaVersion"] == 2
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_UNARY_SOURCE
    assert contract["sourceSha256"] == MLX_UNARY_SHA256
    assert contract["target"] == "opengl"
    assert contract["selection"] == {
        "entryCount": 877,
        "shapeCount": 5,
        "templateCount": 3,
        "operatorCount": 37,
        "typePairCount": 20,
        "familyCount": 16,
        "allDiscoveredSourceInstantiationsIncluded": True,
    }
    assert contract["selection"] == UNARY_METAL_CONTRACT["selection"]
    assert contract["classifications"] == UNARY_METAL_CONTRACT["classifications"]
    assert contract["shapeContracts"] == _target_shape_contracts()
    assert contract["portabilityPreconditions"] == {
        "indexRangeAssertions": [
            {
                "source": MLX_UNARY_SOURCE,
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
    artifact_contract = contract["artifactContract"]
    assert artifact_contract == {
        "artifactCount": 877,
        "artifactCountPerEntry": 1,
        "specializationCount": 1243,
        "specializationCountsByShape": {
            "v": 183,
            "v2": 183,
            "vn": 145,
            "gn1": 366,
            "gn4large": 366,
        },
        "unsupportedSpecializationCount": 0,
        "selectedOperatorImplementationCountPerArtifact": 1,
        "unselectedOperatorBodiesPruned": True,
        "reachableKernelCountPerArtifact": 1,
        "provenance": "entry-scoped-translate",
        "intermediate": "crossgl",
        "hostInterfaceStatus": "ready",
        "targetEntryPoint": "main",
        "reflectedResourceCount": 3363,
        "reflectedResourceCountsByShape": {
            "v": 549,
            "v2": 549,
            "vn": 435,
            "gn1": 915,
            "gn4large": 915,
        },
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "generatedSizeBytesTotal": 4060696,
        "generatedSizeRange": {
            "minimum": {
                "entryPoint": "v_Absint32int32",
                "sizeBytes": 3568,
            },
            "maximum": {
                "entryPoint": "gn4large_ArcTancomplex64complex64",
                "sizeBytes": 8746,
            },
        },
        "nativeCompiler": (
            "glslangValidator --target-env opengl --target-env spirv1.3 -S comp"
        ),
        "nativeValidator": "spirv-val --target-env spv1.3",
        "requiresNonemptySpirvArtifact": True,
    }

    entries = UNARY_OPENGL_ENTRIES
    assert len(entries) == 877
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert len({entry["entryPoint"] for entry in entries}) == 877
    assert (
        Counter(entry["shape"] for entry in entries)
        == contract["classifications"]["shapes"]
    )
    assert (
        Counter(entry["templateName"] for entry in entries)
        == contract["classifications"]["templates"]
    )
    assert (
        Counter(entry["operator"] for entry in entries)
        == contract["classifications"]["operators"]
    )
    assert (
        Counter(f'{entry["inputType"]}->{entry["outputType"]}' for entry in entries)
        == contract["classifications"]["typePairs"]
    )
    assert (
        Counter(entry["family"] for entry in entries)
        == contract["classifications"]["families"]
    )
    assert sum(entry["sizeBytes"] for entry in entries) == 4060696
    assert min((entry["sizeBytes"], entry["entryPoint"]) for entry in entries) == (
        3568,
        "v_Absint32int32",
    )
    assert max((entry["sizeBytes"], entry["entryPoint"]) for entry in entries) == (
        8746,
        "gn4large_ArcTancomplex64complex64",
    )

    metal_entries = {
        entry["entryPoint"]: entry for entry in UNARY_METAL_CONTRACT["entries"]
    }
    for entry in entries:
        assert set(entry) == {
            "entryPoint",
            "shape",
            "templateName",
            "operator",
            "inputType",
            "outputType",
            "family",
            "sha256",
            "sizeBytes",
        }
        assert {
            key: entry[key]
            for key in (
                "entryPoint",
                "shape",
                "templateName",
                "operator",
                "inputType",
                "outputType",
                "family",
            )
        } == {
            key: metal_entries[entry["entryPoint"]][key]
            for key in (
                "entryPoint",
                "shape",
                "templateName",
                "operator",
                "inputType",
                "outputType",
                "family",
            )
        }
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
        assert entry["sizeBytes"] > 0


def test_current_mlx_unary_opengl_ci_shards_are_complete_and_disjoint() -> None:
    shards = tuple(
        _partition_workloads(
            UNARY_OPENGL_WORKLOADS,
            shard_index,
            UNARY_OPENGL_CI_SHARD_COUNT,
        )
        for shard_index in range(UNARY_OPENGL_CI_SHARD_COUNT)
    )
    assert [len(shard) for shard in shards] == [176, 176, 175, 175, 175]
    for shard_index, shard in enumerate(shards):
        assert shard == UNARY_OPENGL_WORKLOADS[shard_index::UNARY_OPENGL_CI_SHARD_COUNT]
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 877
    assert len(set(entry_points)) == 877
    assert set(entry_points) == {
        workload.entry_point for workload in UNARY_OPENGL_WORKLOADS
    }


def _expected_materializations(workload: UnaryOpenGLWorkload) -> list[dict]:
    if workload.shape == "v":
        n_value = "1"
        n_source = "source-instantiation"
    elif workload.shape in {"v2", "vn"}:
        n_value = f"WorkPerThread<{workload.input_type}>::n"
        n_source = "source-default"
    elif workload.shape == "gn1":
        n_value = "1"
        n_source = "source-instantiation"
    else:
        assert workload.shape == "gn4large"
        n_value = "4"
        n_source = "source-instantiation"

    parameters = {
        "N": n_value,
        "Op": workload.operator_type,
        "T": workload.input_type,
        "U": workload.output_type,
    }
    parameter_sources = {
        "N": n_source,
        "Op": "source-instantiation",
        "T": "source-instantiation",
        "U": "source-instantiation",
    }
    if workload.shape.startswith("gn"):
        index_type = "int" if workload.shape == "gn1" else "int64_t"
        parameters["IdxT"] = index_type
        parameter_sources["IdxT"] = (
            "source-instantiation" if workload.shape == "gn1" else "source-default"
        )

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
    if workload.shape.startswith("gn"):
        records.append(
            {
                "name": "elem_to_loc",
                "materializedName": f"elem_to_loc_{index_type}",
                "parameters": {"IdxT": index_type},
                "parameterSources": {"IdxT": "call-site"},
                "source": "call-site",
            }
        )
    return records


def _project_config(workload: UnaryOpenGLWorkload) -> str:
    assertions = "\n\n".join(textwrap.dedent(f"""
            [[project.index_range_assertions]]
            source = "{MLX_UNARY_SOURCE}"
            expression = "{expression}"
            minimum = {minimum}
            maximum = {maximum}
            """).strip() for expression, minimum, maximum in INDEX_RANGE_ASSERTIONS)
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_UNARY_SOURCE}"]
        include_dirs = ["."]
        targets = ["opengl"]
        output_dir = "out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_UNARY_SOURCE}" = "{workload.entry_point}"

        [project.entry_workgroup_size_rules."{MLX_UNARY_SOURCE}"]
        "{workload.entry_point}" = [1, 1, 1]

        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 4096

        {assertions}
        """).strip()


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_UNARY_OPENGL_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")
    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_UNARY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX unary source is missing: {source_path}")
    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checkout_commit.returncode == 0, checkout_commit.stderr
    assert checkout_commit.stdout.strip() == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_UNARY_SHA256
    return mlx_root


def _required_tool(name: str) -> str:
    path = shutil.which(name)
    if path is not None:
        return path
    message = f"{name} is required for the complete MLX unary OpenGL proof"
    if os.environ.get(REQUIRE_UNARY_OPENGL_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _expected_resources(workload: UnaryOpenGLWorkload) -> dict[str, tuple]:
    resources = {
        "in_Buffer": ("buffer", 0, "read"),
        "out_Buffer": ("buffer", 1, "read_write"),
    }
    if workload.shape.startswith("gn"):
        resources.update(
            {
                "in_shapeBuffer": ("buffer", 2, "read"),
                "in_stridesBuffer": ("buffer", 3, "read"),
                "ndimBuffer": ("buffer", 4, "read"),
            }
        )
    else:
        resources[f"{workload.entry_point.rstrip('_')}_size_Args"] = (
            "constant-buffer",
            2,
            "read",
        )
    return resources


def _translate_and_validate(
    mlx_root: Path,
    work_dir: Path,
    workload: UnaryOpenGLWorkload,
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
    artifact = payload["artifacts"][0]
    assert artifact["source"] == MLX_UNARY_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_UNARY_SHA256,
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
    assert materialization["specializations"] == _expected_materializations(workload)
    assert materialization["specializationCount"] == (
        2 if workload.shape.startswith("gn") else 1
    )
    assert materialization["unsupported"] == []
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert (
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in generated
    )
    assert generated.count("void main()") == 1
    selected_signatures = [
        line
        for line in generated.splitlines()
        if f" {workload.operator_type}_operator_call" in line
        and "_temporary(" not in line
        and line.rstrip().endswith("{")
    ]
    assert selected_signatures
    defined_operator_bodies = {
        operator
        for operator in UNARY_OPENGL_OPERATOR_TYPES
        if any(
            f" {operator}_operator_call" in line
            and "_temporary(" not in line
            and line.rstrip().endswith("{")
            for line in generated.splitlines()
        )
    }
    expected_operator_bodies = {workload.operator_type}
    if workload.input_type == "complex64_t":
        expected_operator_bodies.update(
            COMPLEX_OPERATOR_DEPENDENCIES.get(workload.operator_type, ())
        )
    assert defined_operator_bodies == expected_operator_bodies
    for residue in (
        "template <",
        "decltype(",
        "operator()",
        "unsupported Metal",
        "fallback for unmatched generated control flow",
    ):
        assert residue not in generated
    if workload.operator_type == "Log10":
        assert "log10(" not in generated
        assert "metal_u3a_u3alog10(" not in generated
        if workload.input_type == "complex64_t":
            assert "2.30258509299404568401799145468436421" in generated
        else:
            assert "0.3010299956639812" in generated
    if workload.shape.startswith("gn"):
        assert "ndim[0]" in generated
        assert "readonly buffer ndimBuffer" in generated

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
    CURRENT_UNARY_OPENGL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_unary_family_translates_to_opengl(
    workload: UnaryOpenGLWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-unary-opengl-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_and_validate(
            mlx_root,
            Path(temporary_directory),
            workload,
        )
