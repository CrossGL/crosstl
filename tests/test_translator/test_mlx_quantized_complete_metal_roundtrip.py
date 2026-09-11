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
from crosstl.translator.entry_discovery import ENTRY_DISCOVERY_AVAILABLE
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_QUANTIZED_SOURCE = "mlx/backend/metal/kernels/quantized.metal"
MLX_QUANTIZED_SHA256 = (
    "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd"
)
REQUIRE_QUANTIZED_METAL_ENV = "CROSTL_REQUIRE_MLX_QUANTIZED_METAL_ROUNDTRIP"
QUANTIZED_METAL_SHARD_INDEX_ENV = "CROSTL_MLX_QUANTIZED_METAL_SHARD_INDEX"
QUANTIZED_METAL_SHARD_COUNT_ENV = "CROSTL_MLX_QUANTIZED_METAL_SHARD_COUNT"
QUANTIZED_METAL_CI_SHARD_COUNT = 24
ROOT = Path(__file__).resolve().parents[2]
QUANTIZED_METAL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "quantized.metal-roundtrip.json"
)
QUANTIZED_METAL_CONTRACT_SHA256 = (
    "bdf470101f52a18d7b131542be091697cec1d667bc56e1282229720398adaa27"
)
QUANTIZED_METAL_ENTRY_CLASSIFICATION_SHA256 = (
    "5c7001d6e8eaa0135da1228044f22fd7cad8ef70def36d7ea15db94ef7fb24c6"
)
RESOURCE_ABI_FIELDS = ("name", "kind", "set", "binding", "access", "type")
MATERIALIZATION_FIELDS = (
    "status",
    "specializationCount",
    "specializations",
    "unsupported",
    "configuredParameterCount",
    "configuredParameters",
    "configuredParameterSources",
)
ENTRY_FIELDS = (
    "entryPoint",
    "templateName",
    "templateArguments",
    "variant",
    "dataType",
    "groupSize",
    "bitWidth",
    "sha256",
    "sizeBytes",
    "specializationCount",
    "materializationSha256",
    "resourceCount",
    "resourcesSha256",
)
ENTRY_CLASSIFICATION_FIELDS = (
    "entryPoint",
    "templateName",
    "templateArguments",
    "variant",
    "dataType",
    "groupSize",
    "bitWidth",
)
ENTRY_NAME_RE = re.compile(
    r"^(?P<stem>.+)_(?P<data_type>bfloat16_t|float16_t|float)"
    r"_gs_(?P<group_size>32|64|128)_b_(?P<bit_width>2|3|4|5|6|8)"
    r"(?P<suffix>.*)$"
)
RESIDUE_MARKERS = (
    "Int<",
    "template <",
    "get_pack_factor<",
    "get_bytes_per_pack<",
    "dequantize<",
    "decltype(",
    "operator()",
    "unsupported Metal",
    "fallback for unmatched generated control flow",
)


@dataclass(frozen=True)
class QuantizedMetalWorkload:
    entry_point: str
    template_name: str
    template_arguments: tuple[str, ...]
    variant: str
    data_type: str
    group_size: int
    bit_width: int
    sha256: str
    size_bytes: int
    specialization_count: int
    materialization_sha256: str
    resource_count: int
    resources_sha256: str


# Each normalized variant reconstructs its source declaration and every template
# argument after the common <data type, group size, bit width> prefix.
VARIANT_CONTRACTS = {
    "affine_dequantize": ("affine_dequantize", ()),
    "affine_gather_qmm_n": ("affine_gather_qmm_n", ()),
    "affine_gather_qmm_rhs_nn_bm_16_bn_32_bk_32_wm_1_wn_2": (
        "affine_gather_qmm_rhs",
        ("16", "32", "32", "1", "2", "false"),
    ),
    "affine_gather_qmm_rhs_nt_bm_16_bn_32_bk_32_wm_1_wn_2": (
        "affine_gather_qmm_rhs",
        ("16", "32", "32", "1", "2", "true"),
    ),
    "affine_gather_qmm_t_alN_false": ("affine_gather_qmm_t", ("false",)),
    "affine_gather_qmm_t_alN_true": ("affine_gather_qmm_t", ("true",)),
    "affine_gather_qmv": ("affine_gather_qmv", ()),
    "affine_gather_qmv_fast": ("affine_gather_qmv_fast", ()),
    "affine_gather_qvm": ("affine_gather_qvm", ()),
    "affine_qmm_n_batch_0": ("affine_qmm_n", ("0",)),
    "affine_qmm_n_batch_1": ("affine_qmm_n", ("1",)),
    "affine_qmm_t_alN_false_batch_0": ("affine_qmm_t", ("false", "0")),
    "affine_qmm_t_alN_false_batch_1": ("affine_qmm_t", ("false", "1")),
    "affine_qmm_t_alN_true_batch_0": ("affine_qmm_t", ("true", "0")),
    "affine_qmm_t_alN_true_batch_1": ("affine_qmm_t", ("true", "1")),
    "affine_qmm_t_splitk_alN_false": ("affine_qmm_t_splitk", ("false",)),
    "affine_qmm_t_splitk_alN_true": ("affine_qmm_t_splitk", ("true",)),
    "affine_qmv_batch_0": ("affine_qmv", ("0",)),
    "affine_qmv_batch_1": ("affine_qmv", ("1",)),
    "affine_qmv_fast_batch_0": ("affine_qmv_fast", ("0",)),
    "affine_qmv_fast_batch_1": ("affine_qmv_fast", ("1",)),
    "affine_qmv_quad_d_128_batch_0": ("affine_qmv_quad", ("128", "0")),
    "affine_qmv_quad_d_128_batch_1": ("affine_qmv_quad", ("128", "1")),
    "affine_qmv_quad_d_64_batch_0": ("affine_qmv_quad", ("64", "0")),
    "affine_qmv_quad_d_64_batch_1": ("affine_qmv_quad", ("64", "1")),
    "affine_qmv_wide_nv_2_kl_8_batch_0": (
        "affine_qmv_wide",
        ("2", "8", "0"),
    ),
    "affine_qmv_wide_nv_2_kl_8_batch_1": (
        "affine_qmv_wide",
        ("2", "8", "1"),
    ),
    "affine_qmv_wide_nv_3_kl_8_batch_0": (
        "affine_qmv_wide",
        ("3", "8", "0"),
    ),
    "affine_qmv_wide_nv_3_kl_8_batch_1": (
        "affine_qmv_wide",
        ("3", "8", "1"),
    ),
    "affine_qmv_wide_nv_4_kl_8_batch_0": (
        "affine_qmv_wide",
        ("4", "8", "0"),
    ),
    "affine_qmv_wide_nv_4_kl_8_batch_1": (
        "affine_qmv_wide",
        ("4", "8", "1"),
    ),
    "affine_qmv_wide_nv_5_kl_8_batch_0": (
        "affine_qmv_wide",
        ("5", "8", "0"),
    ),
    "affine_qmv_wide_nv_5_kl_8_batch_1": (
        "affine_qmv_wide",
        ("5", "8", "1"),
    ),
    "affine_quantize": ("affine_quantize", ()),
    "affine_qvm_batch_0": ("affine_qvm", ("0",)),
    "affine_qvm_batch_1": ("affine_qvm", ("1",)),
    "affine_qvm_split_k_spk_32": ("affine_qvm_split_k", ("32",)),
    "affine_qvm_split_k_spk_8": ("affine_qvm_split_k", ("8",)),
}
EXPECTED_VARIANT_COUNTS = {variant: 54 for variant in VARIANT_CONTRACTS}
EXPECTED_TEMPLATE_COUNTS = dict(
    sorted(
        Counter(
            template_name
            for template_name, _tail in VARIANT_CONTRACTS.values()
            for _index in range(54)
        ).items()
    )
)
EXPECTED_DATA_TYPE_COUNTS = {
    "bfloat16_t": 684,
    "float": 684,
    "float16_t": 684,
}
EXPECTED_GROUP_SIZE_COUNTS = {"32": 684, "64": 684, "128": 684}
EXPECTED_BIT_WIDTH_COUNTS = {
    "2": 342,
    "3": 342,
    "4": 342,
    "5": 342,
    "6": 342,
    "8": 342,
}
EXPECTED_SPECIALIZATION_COUNTS = {
    "3": 108,
    "5": 432,
    "6": 216,
    "7": 378,
    "8": 162,
    "9": 486,
    "10": 162,
    "11": 108,
}
EXPECTED_RESOURCE_COUNTS = {
    "4": 108,
    "9": 108,
    "10": 108,
    "15": 540,
    "16": 864,
    "21": 162,
    "22": 162,
}
EXPECTED_RESOURCE_TYPE_COUNTS = {
    "const device bfloat*": 1998,
    "const device float*": 1998,
    "const device half*": 1998,
    "const device uchar*": 54,
    "const device uint*": 2700,
    "constant int&": 9126,
    "constant int*": 3780,
    "constant int64_t*": 7560,
    "device bfloat*": 702,
    "device float*": 702,
    "device half*": 702,
    "device uchar*": 54,
}
EXPECTED_PROOF = {
    "candidateBaseCommit": "6f68e09951ee3eee768eb638960b4baf9bad9259",
    "candidateManifestSha256": (
        "ecf5316aba8aecfdec90fb2d9e3efe9c95d75ab32d6d6b940c312ad95e41c4c3"
    ),
    "candidatePatchSha256": (
        "414a3ac6075666bb6464fb1e04cc191aa496201a088911cdb1f736e89193db25"
    ),
    "discoverySha256": (
        "cdb4d881ef0d39bc22e28dbc6ffb876b06ae871de676ee251badefd52ea3f9fc"
    ),
    "proofPlanSha256": (
        "ce3c9c84ddfb48e98715232eeca9addfa8145fc5d5b24d1cb8d7c84a1677dd66"
    ),
    "rowIdentitySha256": (
        "50b74eebe51c3fd368743f4d5448a5c9a88c60c918da153aa1550def00c7d766"
    ),
    "terminalCrosscheckSha256": (
        "b8eacc95af8fb2bc162eb7f4c2d9381850aa58139b3d69d7c404806b39445e62"
    ),
    "independentAuditSha256": (
        "e5b1d0b6e082808e8d9c715f130279527062ca27f2812098b3ac4ab098ffb7da"
    ),
    "adversarialSummarySha256": (
        "118e7720c530a917a53d85fd9cd708fe839fe6292532644fc84ea35b1d127492"
    ),
    "nativeCompileCount": 2052,
    "independentNativeRecompileCount": 2052,
    "generatedSizeBytesTotal": 39638916,
    "airSizeBytesTotal": 36953008,
    "nativeCompilerDiagnosticsEmpty": True,
    "nativeCompilerStreamsEmpty": True,
    "nonemptyAirArtifacts": True,
    "independentAirByteIdentity": True,
    "numericalExecutionClaimed": False,
}


def _canonical_json_sha256(value: object) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    return json.loads(payload)


QUANTIZED_METAL_CONTRACT = _contract(
    QUANTIZED_METAL_CONTRACT_PATH,
    QUANTIZED_METAL_CONTRACT_SHA256,
)
QUANTIZED_METAL_ENTRIES = tuple(QUANTIZED_METAL_CONTRACT["entries"])
QUANTIZED_METAL_RESOURCE_CONTRACTS = QUANTIZED_METAL_CONTRACT["resourceContracts"]
QUANTIZED_METAL_WORKLOADS = tuple(
    QuantizedMetalWorkload(
        entry_point=entry["entryPoint"],
        template_name=entry["templateName"],
        template_arguments=tuple(entry["templateArguments"]),
        variant=entry["variant"],
        data_type=entry["dataType"],
        group_size=entry["groupSize"],
        bit_width=entry["bitWidth"],
        sha256=entry["sha256"],
        size_bytes=entry["sizeBytes"],
        specialization_count=entry["specializationCount"],
        materialization_sha256=entry["materializationSha256"],
        resource_count=entry["resourceCount"],
        resources_sha256=entry["resourcesSha256"],
    )
    for entry in QUANTIZED_METAL_ENTRIES
)


def _classify_entry_point(entry_point: str) -> tuple[str, str, int, int]:
    match = ENTRY_NAME_RE.fullmatch(entry_point)
    assert match is not None, f"unclassified quantized entry: {entry_point}"
    variant = match.group("stem") + match.group("suffix")
    return (
        variant,
        match.group("data_type"),
        int(match.group("group_size")),
        int(match.group("bit_width")),
    )


def _assert_exact_resources(
    actual: object,
    expected: list[dict[str, object]],
    entry_point: str,
) -> None:
    assert actual == expected, (
        f"exact reflected resource ABI mismatch for {entry_point}: "
        f"expected {expected!r}, got {actual!r}"
    )


def test_current_mlx_quantized_metal_contract_is_complete_and_classified():
    contract = QUANTIZED_METAL_CONTRACT
    assert set(contract) == {
        "schemaVersion",
        "kind",
        "commit",
        "source",
        "sourceSha256",
        "target",
        "selection",
        "classifications",
        "resourceContracts",
        "artifactContract",
        "proof",
        "entries",
    }
    assert contract["schemaVersion"] == 2
    assert contract["kind"] == "crosstl-mlx-quantized-metal-roundtrip-contract"
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_QUANTIZED_SOURCE
    assert contract["sourceSha256"] == MLX_QUANTIZED_SHA256
    assert contract["target"] == "metal"
    assert contract["selection"] == {
        "entryCount": 2052,
        "templateCount": 17,
        "variantCount": 38,
        "dataTypeCount": 3,
        "groupSizeCount": 3,
        "bitWidthCount": 6,
        "variantsPerDataTypeGroupSizeBitWidth": 38,
        "allDiscoveredSourceInstantiationsIncluded": True,
    }

    classifications = contract["classifications"]
    assert classifications == {
        "templates": EXPECTED_TEMPLATE_COUNTS,
        "variants": EXPECTED_VARIANT_COUNTS,
        "dataTypes": EXPECTED_DATA_TYPE_COUNTS,
        "groupSizes": EXPECTED_GROUP_SIZE_COUNTS,
        "bitWidths": EXPECTED_BIT_WIDTH_COUNTS,
        "specializationCounts": EXPECTED_SPECIALIZATION_COUNTS,
        "resourceCounts": EXPECTED_RESOURCE_COUNTS,
        "entryClassificationSha256": QUANTIZED_METAL_ENTRY_CLASSIFICATION_SHA256,
    }

    entries = QUANTIZED_METAL_ENTRIES
    assert len(entries) == 2052
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert len({entry["entryPoint"] for entry in entries}) == 2052
    assert len({entry["sha256"] for entry in entries}) == 2052
    assert len({entry["materializationSha256"] for entry in entries}) == 2052
    assert (
        _canonical_json_sha256(
            [
                {field: entry[field] for field in ENTRY_CLASSIFICATION_FIELDS}
                for entry in entries
            ]
        )
        == QUANTIZED_METAL_ENTRY_CLASSIFICATION_SHA256
    )

    for entry in entries:
        assert tuple(entry) == ENTRY_FIELDS
        for digest_field in (
            "sha256",
            "materializationSha256",
            "resourcesSha256",
        ):
            assert len(entry[digest_field]) == 64
            int(entry[digest_field], 16)
        assert entry["sizeBytes"] > 0
        assert entry["specializationCount"] > 0
        assert entry["resourceCount"] > 0

        variant, data_type, group_size, bit_width = _classify_entry_point(
            entry["entryPoint"]
        )
        assert entry["variant"] == variant
        assert entry["dataType"] == data_type
        assert entry["groupSize"] == group_size
        assert entry["bitWidth"] == bit_width
        template_name, trailing_arguments = VARIANT_CONTRACTS[variant]
        assert entry["templateName"] == template_name
        assert entry["templateArguments"] == [
            data_type,
            str(group_size),
            str(bit_width),
            *trailing_arguments,
        ]

        expected_resources = QUANTIZED_METAL_RESOURCE_CONTRACTS[
            entry["resourcesSha256"]
        ]
        assert entry["resourceCount"] == len(expected_resources)
        assert _canonical_json_sha256(expected_resources) == entry["resourcesSha256"]

    assert Counter(entry["templateName"] for entry in entries) == (
        EXPECTED_TEMPLATE_COUNTS
    )
    assert Counter(entry["variant"] for entry in entries) == (EXPECTED_VARIANT_COUNTS)
    assert Counter(entry["dataType"] for entry in entries) == (
        EXPECTED_DATA_TYPE_COUNTS
    )
    assert Counter(str(entry["groupSize"]) for entry in entries) == (
        EXPECTED_GROUP_SIZE_COUNTS
    )
    assert Counter(str(entry["bitWidth"]) for entry in entries) == (
        EXPECTED_BIT_WIDTH_COUNTS
    )

    expected_cube = set(VARIANT_CONTRACTS)
    for data_type in EXPECTED_DATA_TYPE_COUNTS:
        for group_size in (32, 64, 128):
            for bit_width in (2, 3, 4, 5, 6, 8):
                cube = [
                    entry
                    for entry in entries
                    if entry["dataType"] == data_type
                    and entry["groupSize"] == group_size
                    and entry["bitWidth"] == bit_width
                ]
                assert len(cube) == 38
                assert {entry["variant"] for entry in cube} == expected_cube

    resource_contracts = contract["resourceContracts"]
    assert len(resource_contracts) == 30
    assert set(resource_contracts) == {entry["resourcesSha256"] for entry in entries}
    for digest, resources in resource_contracts.items():
        assert len(digest) == 64
        int(digest, 16)
        assert _canonical_json_sha256(resources) == digest
        assert all(set(resource) == set(RESOURCE_ABI_FIELDS) for resource in resources)

    artifact = contract["artifactContract"]
    assert artifact == {
        "artifactCount": 2052,
        "artifactCountPerEntry": 1,
        "specializationCount": 14904,
        "unsupportedSpecializationCount": 0,
        "reachableKernelCountPerArtifact": 1,
        "provenance": "entry-scoped-translate",
        "intermediate": "crossgl",
        "hostInterfaceStatus": "ready",
        "reflectedResourceCount": 31374,
        "reflectedResourceTypeCounts": EXPECTED_RESOURCE_TYPE_COUNTS,
        "resourceAbiFields": list(RESOURCE_ABI_FIELDS),
        "resourceContractCount": 30,
        "exactResourceDigestsIncluded": True,
        "exactResourceContractsIncluded": True,
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "generatedSizeBytesTotal": 39638916,
        "generatedSizeRange": {
            "minimum": {
                "entryPoint": "affine_dequantize_float16_t_gs_32_b_2",
                "sizeBytes": 2860,
            },
            "maximum": {
                "entryPoint": (
                    "affine_gather_qmm_rhs_nt_bfloat16_t_gs_128_b_8_"
                    "bm_16_bn_32_bk_32_wm_1_wn_2"
                ),
                "sizeBytes": 40290,
            },
        },
        "nativeCompiler": "xcrun -sdk macosx metal -std=metal3.1 -Werror -c",
        "requiresEmptyCompilerStreams": True,
        "requiresNonemptyAirArtifact": True,
    }
    assert artifact["specializationCount"] == sum(
        entry["specializationCount"] for entry in entries
    )
    assert artifact["reflectedResourceCount"] == sum(
        entry["resourceCount"] for entry in entries
    )
    assert artifact["generatedSizeBytesTotal"] == sum(
        entry["sizeBytes"] for entry in entries
    )
    assert contract["proof"] == EXPECTED_PROOF


def test_quantized_metal_exact_resource_contract_rejects_wrong_types():
    workload = QUANTIZED_METAL_WORKLOADS[0]
    expected = QUANTIZED_METAL_RESOURCE_CONTRACTS[workload.resources_sha256]
    wrong = [dict(resource) for resource in expected]
    wrong[0]["type"] = "const device definitely_wrong*"
    with pytest.raises(AssertionError, match="exact reflected resource ABI"):
        _assert_exact_resources(wrong, expected, workload.entry_point)


def _partition_quantized_metal_workloads(
    workloads: tuple[QuantizedMetalWorkload, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[QuantizedMetalWorkload, ...]:
    if shard_count <= 0:
        raise ValueError("MLX quantized Metal shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "MLX quantized Metal shard index must be in "
            f"[0, {shard_count}), got {shard_index}"
        )
    selected = workloads[shard_index::shard_count]
    if not selected:
        raise ValueError(
            f"MLX quantized Metal shard {shard_index} of {shard_count} is empty"
        )
    return selected


def _current_quantized_metal_workloads() -> tuple[QuantizedMetalWorkload, ...]:
    raw_index = os.environ.get(QUANTIZED_METAL_SHARD_INDEX_ENV)
    raw_count = os.environ.get(QUANTIZED_METAL_SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return QUANTIZED_METAL_WORKLOADS
    if raw_index is None or raw_count is None:
        raise RuntimeError(
            f"{QUANTIZED_METAL_SHARD_INDEX_ENV} and "
            f"{QUANTIZED_METAL_SHARD_COUNT_ENV} must be configured together"
        )
    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as error:
        raise RuntimeError(
            "MLX quantized Metal shard values must be integers"
        ) from error
    try:
        return _partition_quantized_metal_workloads(
            QUANTIZED_METAL_WORKLOADS,
            shard_index,
            shard_count,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error


CURRENT_QUANTIZED_METAL_WORKLOADS = _current_quantized_metal_workloads()


def test_current_mlx_quantized_metal_ci_shards_are_complete_and_disjoint():
    shards = tuple(
        _partition_quantized_metal_workloads(
            QUANTIZED_METAL_WORKLOADS,
            shard_index,
            QUANTIZED_METAL_CI_SHARD_COUNT,
        )
        for shard_index in range(QUANTIZED_METAL_CI_SHARD_COUNT)
    )
    assert [len(shard) for shard in shards] == [86] * 12 + [85] * 12
    for shard_index, shard in enumerate(shards):
        assert (
            shard
            == QUANTIZED_METAL_WORKLOADS[shard_index::QUANTIZED_METAL_CI_SHARD_COUNT]
        )
    entry_points = [workload.entry_point for shard in shards for workload in shard]
    assert len(entry_points) == 2052
    assert len(set(entry_points)) == 2052
    assert set(entry_points) == {
        workload.entry_point for workload in QUANTIZED_METAL_WORKLOADS
    }


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_QUANTIZED_METAL_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")
    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_QUANTIZED_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX quantized source is missing: {source_path}")
    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        MLX_QUANTIZED_SHA256
    )
    return mlx_root


def test_current_mlx_quantized_metal_discovery_matches_contract():
    mlx_root = _pinned_mlx_root()
    source_path = mlx_root / MLX_QUANTIZED_SOURCE
    register_default_sources()
    source_spec = SOURCE_REGISTRY.get("metal")
    assert source_spec is not None
    discovery = source_spec.discover_entry_points(
        source_path.read_text(encoding="utf-8"),
        file_path=str(source_path),
        include_paths=[str(mlx_root)],
    )
    assert discovery.status == ENTRY_DISCOVERY_AVAILABLE
    assert discovery.diagnostics == ()
    assert len(discovery.entries) == 2052
    discovered = {entry.name: entry for entry in discovery.entries}
    assert len(discovered) == 2052
    assert set(discovered) == {
        workload.entry_point for workload in QUANTIZED_METAL_WORKLOADS
    }
    for workload in QUANTIZED_METAL_WORKLOADS:
        entry = discovered[workload.entry_point]
        assert entry.stage == "compute"
        assert entry.provenance.kind == "host-named-materialization"
        assert entry.provenance.declared_name == workload.template_name
        assert tuple(entry.provenance.template_arguments) == (
            workload.template_arguments
        )


def _project_config(workload: QuantizedMetalWorkload, output_dir: str) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_QUANTIZED_SOURCE}"]
        include_dirs = ["."]
        targets = ["metal"]
        output_dir = "{output_dir}"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_QUANTIZED_SOURCE}" = "{workload.entry_point}"

        [project.entry_workgroup_size_rules."{MLX_QUANTIZED_SOURCE}"]
        "{workload.entry_point}" = [1, 1, 1]

        [project.source_options.metal]
        max_template_specializations = 4096
        max_template_materialization_work = 1048576
        """).strip()


def _normalized_materialization(materialization: object) -> dict:
    assert isinstance(materialization, dict)
    result = {field: materialization.get(field) for field in MATERIALIZATION_FIELDS}
    assert isinstance(result["specializations"], list)
    assert result["status"] == "materialized"
    assert result["specializationCount"] == len(result["specializations"])
    assert result["unsupported"] == []
    assert result["configuredParameterCount"] == 0
    assert result["configuredParameters"] == {}
    assert result["configuredParameterSources"] == {}
    return result


def _translate_quantized_metal_artifact(
    mlx_root: Path,
    work_dir: Path,
    workload: QuantizedMetalWorkload,
) -> tuple[Path, Path]:
    relative_work = work_dir.relative_to(mlx_root).as_posix()
    output_dir = f"{relative_work}/out"
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(workload, output_dir) + "\n",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=("metal",),
        output_dir=output_dir,
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    payload = report.to_json()
    expected_summary = {
        "unitCount": 1,
        "artifactCount": 1,
        "translatedCount": 1,
        "failedCount": 0,
        "diagnosticCounts": {"note": 0, "warning": 0, "error": 0},
    }
    for field, expected in expected_summary.items():
        assert payload["summary"][field] == expected
    assert payload["diagnostics"] == []
    assert len(payload["artifacts"]) == 1
    artifact = payload["artifacts"][0]
    assert artifact["status"] == "translated"
    assert artifact["source"] == MLX_QUANTIZED_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_QUANTIZED_SHA256,
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
    execution_entries = artifact["execution"]["entryPoints"]
    assert len(execution_entries) == 1
    assert execution_entries[0]["workgroupSize"] == [1, 1, 1]
    materialization = _normalized_materialization(artifact["templateMaterialization"])
    assert materialization["specializationCount"] == (workload.specialization_count)
    assert _canonical_json_sha256(materialization) == (workload.materialization_sha256)
    assert payload["validation"].get("toolchainRuns", []) == []

    generated_path = mlx_root / artifact["path"]
    generated_bytes = generated_path.read_bytes()
    assert len(generated_bytes) == workload.size_bytes
    assert hashlib.sha256(generated_bytes).hexdigest() == workload.sha256
    generated = generated_bytes.decode("utf-8")
    assert generated.count("kernel void ") == 1
    assert f"kernel void {workload.entry_point}" in generated
    for residue in RESIDUE_MARKERS:
        assert residue not in generated
    assert re.search(r"(?<![A-Za-z0-9_])type\s*\(", generated) is None

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    return report_path, generated_path


def _roundtrip_pinned_mlx_quantized_through_metal(
    workload: QuantizedMetalWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-quantized-{workload.entry_point}-metal-roundtrip-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        report_path, generated_path = _translate_quantized_metal_artifact(
            mlx_root,
            work_dir,
            workload,
        )
        runtime_artifacts = build_runtime_artifact_manifest(report_path)
        assert runtime_artifacts["success"] is True, json.dumps(
            runtime_artifacts,
            indent=2,
        )
        assert len(runtime_artifacts["artifacts"]) == 1
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
        expected_resources = QUANTIZED_METAL_RESOURCE_CONTRACTS[
            workload.resources_sha256
        ]
        _assert_exact_resources(
            resources,
            expected_resources,
            workload.entry_point,
        )
        assert len(resources) == workload.resource_count
        assert _canonical_json_sha256(resources) == workload.resources_sha256
        assert [resource["metadata"] for resource in host_interface["resources"]] == [
            {"entryPoint": workload.entry_point}
        ] * workload.resource_count

        xcrun = shutil.which("xcrun")
        if xcrun is None:
            message = "xcrun is required for the MLX quantized Metal proof"
            if os.environ.get(REQUIRE_QUANTIZED_METAL_ENV) == "1":
                pytest.fail(message)
            pytest.skip(message)
        air_path = work_dir / f"{workload.entry_point}.air"
        compiled = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-std=metal3.1",
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
        assert compiled.stdout == ""
        assert compiled.stderr == ""
        assert air_path.is_file()
        assert air_path.stat().st_size > 0


@pytest.mark.parametrize(
    "workload",
    CURRENT_QUANTIZED_METAL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_quantized_family_roundtrips_through_metal(workload):
    _roundtrip_pinned_mlx_quantized_through_metal(workload)
