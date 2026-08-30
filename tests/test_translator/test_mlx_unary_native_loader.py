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
    DirectXComputeRuntime,
    DirectXRuntimeParityAdapter,
    OpenGLComputeRuntime,
    OpenGLRuntimeParityAdapter,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
    validate_project_report,
)

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_UNARY_SOURCE = "mlx/backend/metal/kernels/unary.metal"
MLX_UNARY_SHA256 = "51af04126d68e1f5baee5f467268408650d24a68db66e8c044f7f0be3f15368b"
REQUIRE_PROOF_ENVS = {
    "directx": "CROSTL_REQUIRE_MLX_UNARY_DIRECTX_NATIVE_LOADER",
    "metal": "CROSTL_REQUIRE_MLX_UNARY_METAL_ROUNDTRIP",
    "opengl": "CROSTL_REQUIRE_MLX_UNARY_OPENGL_NATIVE_LOADER",
}
ROOT = Path(__file__).resolve().parents[2]
SCALAR_UNARY_METAL_CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "unary.scalar-metal-roundtrip.json"
)
SCALAR_UNARY_METAL_CONTRACT_SHA256 = (
    "3385bafa20d3d14b195fe0ce80d0dcacc56ce7fdadd32eb8ca7fe39aefd16a16"
)
UNARY_METAL_CONTRACT_PATH = (
    ROOT / "demos" / "integrations" / "mlx" / "contracts" / "unary.metal-roundtrip.json"
)
UNARY_METAL_CONTRACT_SHA256 = (
    "315890720b94701cfe1049a7aaad9bfd513c4933eeca5b0a504cb88572819d94"
)


@dataclass(frozen=True)
class UnaryWorkload:
    name: str
    entry_point: str
    operator_type: str
    input_type: str
    output_type: str
    family: str
    generated_operation: dict[str, str]
    generated_artifacts: dict[str, dict[str, str | int]]
    input_values: tuple[float, ...]
    expected_values: tuple[float, ...]
    absolute_tolerance: float
    relative_tolerance: float


SQUARE_WORKLOAD = UnaryWorkload(
    name="square",
    entry_point="v_Squarefloat32float32",
    operator_type="Square",
    input_type="float",
    output_type="float",
    family="float32-same-type",
    generated_operation={
        "directx": "return (x * x);",
        "metal": "return x * x;",
        "opengl": "return (x * x);",
    },
    generated_artifacts={
        "directx": {
            "sha256": (
                "64540a89c95e39914a4d616aff9bec98b939a5209fa4caef5cc1425511abb4e5"
            ),
            "sizeBytes": 2314,
        },
        "metal": {
            "sha256": (
                "244e34b7aa58b7abe7c3ff09f3f51f3aa283a42bf7585bf88200590767032495"
            ),
            "sizeBytes": 1015,
        },
        "opengl": {
            "sha256": (
                "2bb46a3bb0858eb849e533bfe46eff1d59b9192436e15b2639c7998698db6a48"
            ),
            "sizeBytes": 3613,
        },
    },
    input_values=(-3.0, -1.5, 0.0, 2.0, 4.25),
    expected_values=(9.0, 2.25, 0.0, 4.0, 18.0625),
    absolute_tolerance=1e-6,
    relative_tolerance=1e-6,
)

ARCCOS_WORKLOAD = UnaryWorkload(
    name="arccos",
    entry_point="v_ArcCosfloat32float32",
    operator_type="ArcCos",
    input_type="float",
    output_type="float",
    family="float32-same-type",
    generated_operation={
        "directx": "return __crossgl_metal_precise_acos_float(x);",
        "metal": "return __crossgl_metal_precise_acos_float(x);",
        "opengl": "return crossgl_metal_precise_acos_float(x);",
    },
    generated_artifacts={
        "directx": {
            "sha256": (
                "4562332ad4fb951478ca419180ccdf3589f74b9e0956226badc6de877d343239"
            ),
            "sizeBytes": 4175,
        },
        "metal": {
            "sha256": (
                "1247739bc0c48d11692aee81953d8a6a4071de488bfe7ea8d7b2083aa48d9b2b"
            ),
            "sizeBytes": 2742,
        },
        "opengl": {
            "sha256": (
                "280864c39e88198cd5e660127db453877349fadb090cb37f022bcc46300660b3"
            ),
            "sizeBytes": 5965,
        },
    },
    input_values=(-1.0, -0.5, 0.0, 0.5, 1.0),
    expected_values=(
        3.141592653589793,
        2.0943951023931957,
        1.5707963267948966,
        1.0471975511965979,
        0.0,
    ),
    absolute_tolerance=1e-6,
    relative_tolerance=1e-5,
)


def _load_unary_metal_contract(path: Path, expected_sha256: str) -> dict:
    contract_bytes = path.read_bytes()
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    assert contract_sha256 == expected_sha256
    return json.loads(contract_bytes)


SCALAR_UNARY_METAL_CONTRACT = _load_unary_metal_contract(
    SCALAR_UNARY_METAL_CONTRACT_PATH,
    SCALAR_UNARY_METAL_CONTRACT_SHA256,
)
SCALAR_UNARY_METAL_ENTRIES = tuple(SCALAR_UNARY_METAL_CONTRACT["entries"])
SCALAR_UNARY_METAL_ARTIFACT_IDENTITIES = {
    entry["entryPoint"]: {
        "operator": entry["operator"],
        "inputType": entry["inputType"],
        "outputType": entry["outputType"],
        "family": entry["family"],
        "sha256": entry["sha256"],
        "sizeBytes": entry["sizeBytes"],
    }
    for entry in SCALAR_UNARY_METAL_ENTRIES
}

UNARY_METAL_CONTRACT = _load_unary_metal_contract(
    UNARY_METAL_CONTRACT_PATH,
    UNARY_METAL_CONTRACT_SHA256,
)
UNARY_METAL_ENTRIES = tuple(UNARY_METAL_CONTRACT["entries"])
UNARY_METAL_OPERATOR_TYPES = frozenset(
    entry["operator"] for entry in UNARY_METAL_ENTRIES
)
UNARY_METAL_ARTIFACT_IDENTITIES = {
    entry["entryPoint"]: entry for entry in UNARY_METAL_ENTRIES
}


def _metal_roundtrip_workload(entry: dict) -> UnaryWorkload:
    entry_point = entry["entryPoint"]
    if entry_point == SQUARE_WORKLOAD.entry_point:
        return SQUARE_WORKLOAD
    if entry_point == ARCCOS_WORKLOAD.entry_point:
        return ARCCOS_WORKLOAD
    return UnaryWorkload(
        name=entry_point.lower().replace("_", "-"),
        entry_point=entry_point,
        operator_type=entry["operator"],
        input_type=entry["inputType"],
        output_type=entry["outputType"],
        family=entry["family"],
        generated_operation={},
        generated_artifacts={
            "metal": {
                "sha256": entry["sha256"],
                "sizeBytes": entry["sizeBytes"],
            }
        },
        input_values=(),
        expected_values=(),
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )


UNARY_METAL_WORKLOADS = tuple(
    _metal_roundtrip_workload(entry) for entry in UNARY_METAL_ENTRIES
)


def test_current_mlx_scalar_unary_metal_contract_is_complete_and_classified():
    contract = SCALAR_UNARY_METAL_CONTRACT
    assert contract["schemaVersion"] == 1
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_UNARY_SOURCE
    assert contract["sourceSha256"] == MLX_UNARY_SHA256
    assert contract["target"] == "metal"
    assert contract["selection"] == {
        "entryPrefix": "v_",
        "templateName": "unary_v",
        "elementCount": 1,
        "shape": "scalar",
        "entryCount": 183,
        "operatorCount": 37,
        "typePairCount": 20,
        "familyCount": 16,
    }
    expected_operator_counts = {
        "Abs": 13,
        "ArcCos": 4,
        "ArcCosh": 3,
        "ArcSin": 4,
        "ArcSinh": 3,
        "ArcTan": 4,
        "ArcTanh": 3,
        "BitwiseInvert": 8,
        "Ceil": 12,
        "Conjugate": 1,
        "Cos": 4,
        "Cosh": 4,
        "Erf": 3,
        "ErfInv": 3,
        "Exp": 4,
        "Expm1": 3,
        "Floor": 12,
        "FromFP8": 3,
        "Imag": 1,
        "Log": 4,
        "Log10": 4,
        "Log1p": 4,
        "Log2": 4,
        "LogicalNot": 1,
        "Negative": 13,
        "Real": 1,
        "Round": 4,
        "Rsqrt": 4,
        "Sigmoid": 3,
        "Sign": 13,
        "Sin": 4,
        "Sinh": 4,
        "Sqrt": 4,
        "Square": 13,
        "Tan": 4,
        "Tanh": 4,
        "ToFP8": 3,
    }
    expected_type_pair_counts = {
        "bfloat16_t->bfloat16_t": 30,
        "bfloat16_t->uint8_t": 1,
        "bool->bool": 7,
        "complex64_t->complex64_t": 22,
        "complex64_t->float": 2,
        "float->float": 30,
        "float->uint8_t": 1,
        "float16_t->uint8_t": 1,
        "half->half": 30,
        "int16_t->int16_t": 7,
        "int32_t->int32_t": 7,
        "int64_t->int64_t": 7,
        "int8_t->int8_t": 7,
        "uint16_t->uint16_t": 7,
        "uint32_t->uint32_t": 7,
        "uint64_t->uint64_t": 7,
        "uint8_t->bfloat16_t": 1,
        "uint8_t->float": 1,
        "uint8_t->float16_t": 1,
        "uint8_t->uint8_t": 7,
    }
    expected_family_counts = {
        "bfloat16-same-type": 30,
        "boolean-same-type": 7,
        "complex64-projection": 2,
        "complex64-same-type": 22,
        "float16-same-type": 30,
        "float32-same-type": 30,
        "fp8-decode": 3,
        "fp8-encode": 3,
        "int16-same-type": 7,
        "int32-same-type": 7,
        "int64-same-type": 7,
        "int8-same-type": 7,
        "uint16-same-type": 7,
        "uint32-same-type": 7,
        "uint64-same-type": 7,
        "uint8-same-type": 7,
    }
    entries = SCALAR_UNARY_METAL_ENTRIES
    assert len(entries) == 183
    assert len({entry["entryPoint"] for entry in entries}) == 183
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert Counter(entry["operator"] for entry in entries) == expected_operator_counts
    assert (
        Counter(f'{entry["inputType"]}->{entry["outputType"]}' for entry in entries)
        == expected_type_pair_counts
    )
    assert Counter(entry["family"] for entry in entries) == expected_family_counts
    assert contract["classifications"] == {
        "operators": expected_operator_counts,
        "typePairs": expected_type_pair_counts,
        "families": expected_family_counts,
    }
    assert contract["artifactContract"] == {
        "artifactCountPerEntry": 1,
        "specializationCountPerArtifact": 1,
        "unsupportedSpecializationCount": 0,
        "selectedOperatorImplementationCountPerArtifact": 1,
        "unselectedOperatorBodiesPruned": True,
        "reachableKernelCountPerArtifact": 1,
        "provenance": "entry-scoped-translate",
        "intermediate": "crossgl",
        "hostInterfaceStatus": "ready",
        "hostResourceCountPerArtifact": 3,
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "nativeCompiler": "xcrun -sdk macosx metal -c",
        "requiresNonemptyAirArtifact": True,
    }
    for entry in entries:
        assert set(entry) == {
            "entryPoint",
            "operator",
            "inputType",
            "outputType",
            "family",
            "sha256",
            "sizeBytes",
        }
        assert entry["entryPoint"].startswith("v_")
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
        assert entry["sizeBytes"] > 0
    assert SCALAR_UNARY_METAL_ARTIFACT_IDENTITIES[SQUARE_WORKLOAD.entry_point] == {
        "operator": SQUARE_WORKLOAD.operator_type,
        "inputType": SQUARE_WORKLOAD.input_type,
        "outputType": SQUARE_WORKLOAD.output_type,
        "family": SQUARE_WORKLOAD.family,
        **SQUARE_WORKLOAD.generated_artifacts["metal"],
    }
    assert SCALAR_UNARY_METAL_ARTIFACT_IDENTITIES[ARCCOS_WORKLOAD.entry_point] == {
        "operator": ARCCOS_WORKLOAD.operator_type,
        "inputType": ARCCOS_WORKLOAD.input_type,
        "outputType": ARCCOS_WORKLOAD.output_type,
        "family": ARCCOS_WORKLOAD.family,
        **ARCCOS_WORKLOAD.generated_artifacts["metal"],
    }


def test_current_mlx_unary_metal_contract_is_complete_and_classified():
    contract = UNARY_METAL_CONTRACT
    assert contract["schemaVersion"] == 2
    assert contract["commit"] == MLX_COMMIT
    assert contract["source"] == MLX_UNARY_SOURCE
    assert contract["sourceSha256"] == MLX_UNARY_SHA256
    assert contract["target"] == "metal"
    assert contract["selection"] == {
        "entryCount": 877,
        "shapeCount": 5,
        "templateCount": 3,
        "operatorCount": 37,
        "typePairCount": 20,
        "familyCount": 16,
        "allDiscoveredSourceInstantiationsIncluded": True,
    }
    expected_classifications = {
        "shapes": {"gn1": 183, "gn4large": 183, "v": 183, "v2": 183, "vn": 145},
        "templates": {"unary_g": 366, "unary_v": 328, "unary_v2": 183},
        "operators": {
            "Abs": 62,
            "ArcCos": 19,
            "ArcCosh": 15,
            "ArcSin": 19,
            "ArcSinh": 15,
            "ArcTan": 19,
            "ArcTanh": 15,
            "BitwiseInvert": 38,
            "Ceil": 58,
            "Conjugate": 4,
            "Cos": 19,
            "Cosh": 19,
            "Erf": 15,
            "ErfInv": 15,
            "Exp": 19,
            "Expm1": 15,
            "Floor": 58,
            "FromFP8": 15,
            "Imag": 4,
            "Log": 19,
            "Log10": 19,
            "Log1p": 19,
            "Log2": 19,
            "LogicalNot": 5,
            "Negative": 62,
            "Real": 4,
            "Round": 19,
            "Rsqrt": 19,
            "Sigmoid": 15,
            "Sign": 62,
            "Sin": 19,
            "Sinh": 19,
            "Sqrt": 19,
            "Square": 62,
            "Tan": 19,
            "Tanh": 19,
            "ToFP8": 15,
        },
        "typePairs": {
            "bfloat16_t->bfloat16_t": 150,
            "bfloat16_t->uint8_t": 5,
            "bool->bool": 35,
            "complex64_t->complex64_t": 88,
            "complex64_t->float": 8,
            "float->float": 150,
            "float->uint8_t": 5,
            "float16_t->uint8_t": 5,
            "half->half": 150,
            "int16_t->int16_t": 35,
            "int32_t->int32_t": 35,
            "int64_t->int64_t": 28,
            "int8_t->int8_t": 35,
            "uint16_t->uint16_t": 35,
            "uint32_t->uint32_t": 35,
            "uint64_t->uint64_t": 28,
            "uint8_t->bfloat16_t": 5,
            "uint8_t->float": 5,
            "uint8_t->float16_t": 5,
            "uint8_t->uint8_t": 35,
        },
        "families": {
            "bfloat16-same-type": 150,
            "boolean-same-type": 35,
            "complex64-projection": 8,
            "complex64-same-type": 88,
            "float16-same-type": 150,
            "float32-same-type": 150,
            "fp8-decode": 15,
            "fp8-encode": 15,
            "int16-same-type": 35,
            "int32-same-type": 35,
            "int64-same-type": 28,
            "int8-same-type": 35,
            "uint16-same-type": 35,
            "uint32-same-type": 35,
            "uint64-same-type": 28,
            "uint8-same-type": 35,
        },
    }
    expected_shape_contracts = {
        "v": {
            "entryPrefix": "v_",
            "entryCount": 183,
            "templateName": "unary_v",
            "sourceTemplateArgumentCount": 4,
            "templateParameters": {
                "T": {"value": "entry.inputType", "source": "source-instantiation"},
                "U": {"value": "entry.outputType", "source": "source-instantiation"},
                "Op": {"value": "entry.operator", "source": "source-instantiation"},
                "N": {"value": "1", "source": "source-instantiation"},
            },
            "specializationCountPerArtifact": 1,
            "reachableSpecializations": ["unary_v"],
            "hostResourceCountPerArtifact": 3,
            "hostResources": [
                {"name": "in_", "kind": "buffer", "binding": 0, "access": "read"},
                {
                    "name": "out_",
                    "kind": "buffer",
                    "binding": 1,
                    "access": "read_write",
                },
                {
                    "name": "size",
                    "kind": "constant-buffer",
                    "binding": 2,
                    "access": "read",
                },
            ],
        },
        "v2": {
            "entryPrefix": "v2_",
            "entryCount": 183,
            "templateName": "unary_v2",
            "sourceTemplateArgumentCount": 3,
            "templateParameters": {
                "T": {"value": "entry.inputType", "source": "source-instantiation"},
                "U": {"value": "entry.outputType", "source": "source-instantiation"},
                "Op": {"value": "entry.operator", "source": "source-instantiation"},
                "N": {"value": "WorkPerThread<T>::n", "source": "source-default"},
            },
            "specializationCountPerArtifact": 1,
            "reachableSpecializations": ["unary_v2"],
            "hostResourceCountPerArtifact": 3,
            "hostResources": [
                {"name": "in_", "kind": "buffer", "binding": 0, "access": "read"},
                {
                    "name": "out_",
                    "kind": "buffer",
                    "binding": 1,
                    "access": "read_write",
                },
                {
                    "name": "size",
                    "kind": "constant-buffer",
                    "binding": 2,
                    "access": "read",
                },
            ],
        },
        "vn": {
            "entryPrefix": "vn_",
            "entryCount": 145,
            "templateName": "unary_v",
            "sourceTemplateArgumentCount": 3,
            "templateParameters": {
                "T": {"value": "entry.inputType", "source": "source-instantiation"},
                "U": {"value": "entry.outputType", "source": "source-instantiation"},
                "Op": {"value": "entry.operator", "source": "source-instantiation"},
                "N": {"value": "WorkPerThread<T>::n", "source": "source-default"},
            },
            "specializationCountPerArtifact": 1,
            "reachableSpecializations": ["unary_v"],
            "hostResourceCountPerArtifact": 3,
            "hostResources": [
                {"name": "in_", "kind": "buffer", "binding": 0, "access": "read"},
                {
                    "name": "out_",
                    "kind": "buffer",
                    "binding": 1,
                    "access": "read_write",
                },
                {
                    "name": "size",
                    "kind": "constant-buffer",
                    "binding": 2,
                    "access": "read",
                },
            ],
        },
        "gn1": {
            "entryPrefix": "gn1_",
            "entryCount": 183,
            "templateName": "unary_g",
            "sourceTemplateArgumentCount": 5,
            "templateParameters": {
                "T": {"value": "entry.inputType", "source": "source-instantiation"},
                "U": {"value": "entry.outputType", "source": "source-instantiation"},
                "Op": {"value": "entry.operator", "source": "source-instantiation"},
                "N": {"value": "1", "source": "source-instantiation"},
                "IdxT": {"value": "int", "source": "source-instantiation"},
            },
            "specializationCountPerArtifact": 2,
            "reachableSpecializations": ["unary_g", "elem_to_loc<int>"],
            "hostResourceCountPerArtifact": 5,
            "hostResources": [
                {"name": "in_", "kind": "buffer", "binding": 0, "access": "read"},
                {
                    "name": "out_",
                    "kind": "buffer",
                    "binding": 1,
                    "access": "read_write",
                },
                {
                    "name": "in_shape",
                    "kind": "constant-buffer",
                    "binding": 2,
                    "access": "read",
                },
                {
                    "name": "in_strides",
                    "kind": "constant-buffer",
                    "binding": 3,
                    "access": "read",
                },
                {"name": "ndim", "kind": "buffer", "binding": 4, "access": "read"},
            ],
        },
        "gn4large": {
            "entryPrefix": "gn4large_",
            "entryCount": 183,
            "templateName": "unary_g",
            "sourceTemplateArgumentCount": 4,
            "templateParameters": {
                "T": {"value": "entry.inputType", "source": "source-instantiation"},
                "U": {"value": "entry.outputType", "source": "source-instantiation"},
                "Op": {"value": "entry.operator", "source": "source-instantiation"},
                "N": {"value": "4", "source": "source-instantiation"},
                "IdxT": {"value": "int64_t", "source": "source-default"},
            },
            "specializationCountPerArtifact": 2,
            "reachableSpecializations": ["unary_g", "elem_to_loc<int64_t>"],
            "hostResourceCountPerArtifact": 5,
            "hostResources": [
                {"name": "in_", "kind": "buffer", "binding": 0, "access": "read"},
                {
                    "name": "out_",
                    "kind": "buffer",
                    "binding": 1,
                    "access": "read_write",
                },
                {
                    "name": "in_shape",
                    "kind": "constant-buffer",
                    "binding": 2,
                    "access": "read",
                },
                {
                    "name": "in_strides",
                    "kind": "constant-buffer",
                    "binding": 3,
                    "access": "read",
                },
                {"name": "ndim", "kind": "buffer", "binding": 4, "access": "read"},
            ],
        },
    }
    expected_artifact_contract = {
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
        "hostDispatchWorkgroupSize": [1, 1, 1],
        "generatedSizeBytesTotal": 1321446,
        "generatedSizeRange": {
            "minimum": {"entryPoint": "v_Absint8int8", "sizeBytes": 972},
            "maximum": {
                "entryPoint": "gn4large_ArcTancomplex64complex64",
                "sizeBytes": 4468,
            },
        },
        "nativeCompiler": (
            "xcrun -sdk macosx metal -Werror -Wno-tautological-constant-compare " "-c"
        ),
        "requiresNonemptyAirArtifact": True,
    }
    assert contract["classifications"] == expected_classifications
    assert contract["shapeContracts"] == expected_shape_contracts
    assert contract["artifactContract"] == expected_artifact_contract

    entries = UNARY_METAL_ENTRIES
    assert len(entries) == 877
    assert len({entry["entryPoint"] for entry in entries}) == 877
    assert [entry["entryPoint"] for entry in entries] == sorted(
        entry["entryPoint"] for entry in entries
    )
    assert Counter(entry["shape"] for entry in entries) == {
        "v": 183,
        "v2": 183,
        "vn": 145,
        "gn1": 183,
        "gn4large": 183,
    }
    assert Counter(entry["templateName"] for entry in entries) == {
        "unary_v": 328,
        "unary_v2": 183,
        "unary_g": 366,
    }
    assert (
        Counter(entry["operator"] for entry in entries)
        == expected_classifications["operators"]
    )
    assert (
        Counter(f'{entry["inputType"]}->{entry["outputType"]}' for entry in entries)
        == expected_classifications["typePairs"]
    )
    assert (
        Counter(entry["family"] for entry in entries)
        == expected_classifications["families"]
    )
    prefixes = {
        "v": "v_",
        "v2": "v2_",
        "vn": "vn_",
        "gn1": "gn1_",
        "gn4large": "gn4large_",
    }
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
        assert entry["entryPoint"].startswith(prefixes[entry["shape"]])
        assert (
            entry["templateName"]
            == expected_shape_contracts[entry["shape"]]["templateName"]
        )
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
        assert entry["sizeBytes"] > 0

    scalar_entries = {
        entry["entryPoint"]: entry for entry in entries if entry["shape"] == "v"
    }
    assert len(scalar_entries) == 183
    for scalar_entry in SCALAR_UNARY_METAL_ENTRIES:
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


def _unary_shape(entry_point: str) -> str:
    for prefix, shape in (
        ("gn4large_", "gn4large"),
        ("gn1_", "gn1"),
        ("v2_", "v2"),
        ("vn_", "vn"),
        ("v_", "v"),
    ):
        if entry_point.startswith(prefix):
            return shape
    raise AssertionError(f"Unknown unary entry shape: {entry_point}")


def _expected_unary_materializations(workload: UnaryWorkload) -> list[dict]:
    shape = _unary_shape(workload.entry_point)
    if shape == "v":
        template_name = "unary_v"
        n_value = "1"
        n_source = "source-instantiation"
    elif shape == "v2":
        template_name = "unary_v2"
        n_value = f"WorkPerThread<{workload.input_type}>::n"
        n_source = "source-default"
    elif shape == "vn":
        template_name = "unary_v"
        n_value = f"WorkPerThread<{workload.input_type}>::n"
        n_source = "source-default"
    elif shape == "gn1":
        template_name = "unary_g"
        n_value = "1"
        n_source = "source-instantiation"
    else:
        assert shape == "gn4large"
        template_name = "unary_g"
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
    if shape.startswith("gn"):
        index_type = "int" if shape == "gn1" else "int64_t"
        parameters["IdxT"] = index_type
        parameter_sources["IdxT"] = (
            "source-instantiation" if shape == "gn1" else "source-default"
        )

    records = [
        {
            "name": template_name,
            "materializedName": workload.entry_point,
            "parameters": parameters,
            "parameterSources": parameter_sources,
            "source": "source-instantiation",
            "hostName": workload.entry_point,
        }
    ]
    if shape.startswith("gn"):
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


def _project_config(target: str, workload: UnaryWorkload) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_UNARY_SOURCE}"]
        include_dirs = ["."]
        targets = ["{target}"]
        output_dir = ".crosstl-mlx-unary-native-loader/out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_UNARY_SOURCE}" = "{workload.entry_point}"

        [project.entry_workgroup_size_rules."{MLX_UNARY_SOURCE}"]
        "{workload.entry_point}" = [1, 1, 1]

        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 4096
        """).strip()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _skip_or_fail(target: str, message: str) -> None:
    if os.environ.get(REQUIRE_PROOF_ENVS[target]) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if any(os.environ.get(name) == "1" for name in REQUIRE_PROOF_ENVS.values()):
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_UNARY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX unary source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_UNARY_SHA256
    return mlx_root


def _translate_unary_artifact(
    mlx_root: Path,
    work_dir: Path,
    target: str,
    workload: UnaryWorkload,
):
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(target, workload) + "\n",
        encoding="utf-8",
    )
    output_dir = work_dir / "out"
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=(target,),
        output_dir=output_dir.relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=True,
        run_toolchains=target != "metal",
    )
    payload = report.to_json()

    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    expected_identity = (
        UNARY_METAL_ARTIFACT_IDENTITIES[workload.entry_point]
        if target == "metal"
        else workload.generated_artifacts[target]
    )
    assert artifact["source"] == MLX_UNARY_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_UNARY_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected_identity["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected_identity["sizeBytes"]
    expected_target_entry = {
        "directx": "CSMain",
        "metal": workload.entry_point,
        "opengl": "main",
    }[target]
    assert artifact["entryPoint"] == {
        "source": workload.entry_point,
        "target": expected_target_entry,
        "stage": "compute",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"
    assert artifact["execution"]["entryPoints"][0]["workgroupSize"] == [1, 1, 1]
    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    expected_materializations = _expected_unary_materializations(workload)
    assert materialization["specializationCount"] == len(expected_materializations)
    assert materialization["specializations"] == expected_materializations
    assert materialization["unsupported"] == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    expected_operation = workload.generated_operation.get(target)
    if expected_operation is not None:
        assert expected_operation in generated
    if workload is ARCCOS_WORKLOAD:
        assert "return acos(x);" not in generated
        if target == "opengl":
            assert "precise float crossgl_metal_precise_acos" not in generated
            assert "precise float crossglPreciseReturn" in generated
        elif target == "directx":
            assert "precise float __crossgl_metal_precise_acos" in generated
        else:
            assert target == "metal"
            assert "float __crossgl_metal_precise_acos_ratio(float value)" in generated
            assert "float __crossgl_metal_precise_acos_float(float value)" in generated
            assert generated.count("#pragma clang fp contract(off)") == 2
            assert generated.count("#pragma clang fp contract(fast)") == 2
    assert "Log{}(x + i * Sqrt{}(1.0 - x * x))" not in generated
    assert "template <" not in generated
    assert "decltype(" not in generated
    assert "operator()" not in generated
    assert "union_member_layout" not in generated
    assert "unsupported Metal" not in generated
    assert "fallback for unmatched generated control flow" not in generated
    if target == "directx":
        assert "[numthreads(1, 1, 1)]" in generated
        validator = "dxc"
    elif target == "opengl":
        assert (
            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1)" in generated
        )
        validator = "glslangValidator"
    else:
        assert target == "metal"
        assert f"kernel void {workload.entry_point}" in generated
        assert generated.count("kernel void ") == 1
        assert "[[static]]" not in generated
        assert f"struct {workload.operator_type} {{" in generated
        assert generated.count(f"struct {workload.operator_type} {{") == 1
        for pruned_operator in UNARY_METAL_OPERATOR_TYPES - {workload.operator_type}:
            marker = f"struct {pruned_operator} {{"
            cursor = 0
            while (start := generated.find(marker, cursor)) != -1:
                end = generated.find("};", start + len(marker))
                assert end != -1
                assert generated[start + len(marker) : end].strip() == ""
                cursor = end + 2
        validator = "xcrun"
    if target == "metal":
        assert payload["validation"].get("toolchainRuns", []) == []
    elif shutil.which(validator) is not None:
        toolchain_runs = payload["validation"]["toolchainRuns"]
        assert len(toolchain_runs) == 1
        assert toolchain_runs[0]["status"] == "ok"
    elif os.environ.get(REQUIRE_PROOF_ENVS[target]) == "1":
        pytest.fail(f"{validator} is required for the MLX unary {target} proof")

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    return report_path


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_pinned_mlx_unary_square_translates_to_selected_target(target):
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-unary-{target}-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_unary_artifact(
            mlx_root,
            Path(temporary_directory),
            target,
            SQUARE_WORKLOAD,
        )


def _roundtrip_pinned_mlx_unary_through_metal(workload: UnaryWorkload) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-unary-{workload.name}-metal-roundtrip-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        report_path = _translate_unary_artifact(
            mlx_root,
            work_dir,
            "metal",
            workload,
        )
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        artifact = report_payload["artifacts"][0]
        generated_path = mlx_root / artifact["path"]

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
            "in_": ("buffer", 0, "read", workload.entry_point),
            "out_": ("buffer", 1, "read_write", workload.entry_point),
        }
        if _unary_shape(workload.entry_point).startswith("gn"):
            expected_resources.update(
                {
                    "in_shape": (
                        "constant-buffer",
                        2,
                        "read",
                        workload.entry_point,
                    ),
                    "in_strides": (
                        "constant-buffer",
                        3,
                        "read",
                        workload.entry_point,
                    ),
                    "ndim": ("buffer", 4, "read", workload.entry_point),
                }
            )
        else:
            expected_resources["size"] = (
                "constant-buffer",
                2,
                "read",
                workload.entry_point,
            )
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
            _skip_or_fail("metal", "xcrun is required for the MLX unary Metal proof")
        air_path = work_dir / f"{workload.entry_point}.air"
        compiled = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-Werror",
                "-Wno-tautological-constant-compare",
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


def test_pinned_mlx_unary_square_roundtrips_through_metal():
    _roundtrip_pinned_mlx_unary_through_metal(SQUARE_WORKLOAD)


def test_pinned_mlx_unary_arccos_roundtrips_through_metal():
    _roundtrip_pinned_mlx_unary_through_metal(ARCCOS_WORKLOAD)


@pytest.mark.parametrize(
    "workload",
    UNARY_METAL_WORKLOADS,
    ids=lambda workload: workload.entry_point,
)
def test_current_mlx_unary_family_roundtrips_through_metal(workload):
    _roundtrip_pinned_mlx_unary_through_metal(workload)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_pinned_mlx_unary_arccos_translates_to_selected_target(target):
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-unary-arccos-{target}-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_unary_artifact(
            mlx_root,
            Path(temporary_directory),
            target,
            ARCCOS_WORKLOAD,
        )


def _build_runtime_package(
    mlx_root: Path,
    work_dir: Path,
    target: str,
    workload: UnaryWorkload,
) -> tuple[dict, Path]:
    report_path = _translate_unary_artifact(
        mlx_root,
        work_dir,
        target,
        workload,
    )
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(
        runtime_artifacts,
        indent=2,
    )
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"

    expected_resources = {
        "directx": {
            "in_": (0, "read"),
            "out_": (1, "read_write"),
            f"{workload.entry_point}_size_Constants": (2, "read"),
        },
        "opengl": {
            "in_Buffer": (0, "read"),
            "out_Buffer": (1, "read_write"),
            f"{workload.entry_point}_size_Args": (2, "read"),
        },
    }[target]
    resources = reflected["hostInterface"]["resources"]
    assert {
        resource["name"]: (resource["binding"], resource["access"])
        for resource in resources
    } == expected_resources

    runtime_artifacts_path = work_dir / "runtime-artifacts.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "runtime-package"
    package = build_runtime_package(runtime_artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)
    loader_manifest = build_runtime_loader_manifest(
        package_dir / "runtime-package.json"
    )
    assert loader_manifest["success"] is True, json.dumps(
        loader_manifest,
        indent=2,
    )
    assert loader_manifest["summary"]["readyLoadUnitCount"] == 1
    assert loader_manifest["summary"]["blockedLoadUnitCount"] == 0
    descriptor = build_native_loader_abi_descriptor(
        loader_manifest,
        load_unit_id=loader_manifest["loadUnits"][0]["id"],
    )
    execution_config = descriptor["entryPoint"]["executionConfig"]
    if target == "directx":
        assert execution_config == {"numthreads": [1, 1, 1]}
    else:
        assert execution_config == {
            "local_size": [1, 1, 1],
            "local_size_x": 1,
            "local_size_y": 1,
            "local_size_z": 1,
        }
    assert {
        binding["name"]: binding["access"] for binding in descriptor["bindings"]
    } == {name: access for name, (_binding, access) in expected_resources.items()}
    return descriptor, package_dir


def _execute_pinned_mlx_unary(
    target: str,
    workload: UnaryWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-unary-{workload.name}-{target}-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            target,
            workload,
        )
        if target == "directx":
            input_binding = "in_"
            output_binding = "out_"
            size_binding = f"{workload.entry_point}_size_Constants"
        else:
            input_binding = "in_Buffer"
            output_binding = "out_Buffer"
            size_binding = f"{workload.entry_point}_size_Args"
        input_values = list(workload.input_values)
        expected_values = list(workload.expected_values)
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                input_binding: {
                    "dtype": "float32",
                    "shape": [len(input_values)],
                    "values": input_values,
                },
                size_binding: {
                    "dtype": "uint32",
                    "shape": [1],
                    "values": [len(input_values)],
                },
            },
            {
                output_binding: {
                    "dtype": "float32",
                    "shape": [len(expected_values)],
                    "values": expected_values,
                    "tolerance": {
                        "absolute": workload.absolute_tolerance,
                        "relative": workload.relative_tolerance,
                    },
                }
            },
            (len(input_values), 1, 1),
            expected_target=target,
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.workgroup_size == (1, 1, 1)
        assert request.execution_plan.dispatch.global_size == (
            len(input_values),
            1,
            1,
        )

        runtime_adapter = (
            DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
            if target == "directx"
            else OpenGLRuntimeParityAdapter(
                runtime=OpenGLComputeRuntime(context_backends=("egl",))
            )
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-unary-{workload.name}-{target}-native-loader",
                target=target,
                executor=target,
                adapter_kind=f"{target}-native-runtime",
            ),
            runtime_adapter=runtime_adapter,
        )
        availability = executor.is_available(request)
        if not availability.available:
            _skip_or_fail(
                target,
                availability.reason or f"The native {target} runtime is unavailable",
            )

        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs[output_binding]["dtype"] == "float32"
    assert result.outputs[output_binding]["shape"] == [len(expected_values)]
    assert result.outputs[output_binding]["values"] == pytest.approx(
        expected_values,
        abs=workload.absolute_tolerance,
        rel=workload.relative_tolerance,
    )


def test_pinned_mlx_unary_square_executes_through_directx_native_loader():
    _execute_pinned_mlx_unary("directx", SQUARE_WORKLOAD)


def test_pinned_mlx_unary_square_executes_through_opengl_native_loader():
    _execute_pinned_mlx_unary("opengl", SQUARE_WORKLOAD)


def test_pinned_mlx_unary_arccos_executes_through_directx_native_loader():
    _execute_pinned_mlx_unary("directx", ARCCOS_WORKLOAD)


def test_pinned_mlx_unary_arccos_executes_through_opengl_native_loader():
    _execute_pinned_mlx_unary("opengl", ARCCOS_WORKLOAD)
