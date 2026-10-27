from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    DirectXComputeRuntime,
    DirectXRuntimeParityAdapter,
    MetalRuntimeParityAdapter,
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
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from crosstl.project.native_runtime_drivers import (
    _complete_directx_register_layout,
    _prepare_directx_buffers,
    _prepare_directx_constants,
    _validate_directx_register_layout,
)
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.tests.pins import check_generated_artifact

ROOT = Path(__file__).resolve().parents[5]
MLX_COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
SOURCE = "mlx/backend/metal/kernels/arg_reduce.metal"
SOURCE_SHA256 = "036a586af92b869a94f0b67e54590abf935e87f99290db46daed86cd8941e429"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ARG_REDUCE"
REQUIRE_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ARG_REDUCE_RUNTIME"
ENTRIES = [
    "argmin_bool_",
    "argmax_bool_",
    "argmin_uint8",
    "argmax_uint8",
    "argmin_uint16",
    "argmax_uint16",
    "argmin_uint32",
    "argmax_uint32",
    "argmin_uint64",
    "argmax_uint64",
    "argmin_int8",
    "argmax_int8",
    "argmin_int16",
    "argmax_int16",
    "argmin_int32",
    "argmax_int32",
    "argmin_int64",
    "argmax_int64",
    "argmin_float16",
    "argmax_float16",
    "argmin_float32",
    "argmax_float32",
    "argmin_bfloat16",
    "argmax_bfloat16",
]
RUNTIME_ENTRIES = ["argmin_float32", "argmax_float32"]
INTEGER_RUNTIME_ENTRIES = [
    f"{operation}_{dtype}{width}"
    for width in (16, 32, 64)
    for dtype in ("uint", "int")
    for operation in ("argmin", "argmax")
]
RUNTIME_ENTRIES_BY_TARGET = {
    target: RUNTIME_ENTRIES + INTEGER_RUNTIME_ENTRIES
    for target in ("metal", "opengl", "directx")
}
ARTIFACTS = {
    "directx": {
        "argmin_bool_": {
            "sha256": (
                "f3505cdf87bb4de46d143b50fb0cac5f1ef84d0b9dd1186607e7b3f4ceba0df7"
            ),
            "sizeBytes": 12005,
        },
        "argmax_bool_": {
            "sha256": (
                "98a8422d3eeedaed7ff586c5847bf1d4829d33959d55efa42e61a868906fc644"
            ),
            "sizeBytes": 12007,
        },
        "argmin_uint8": {
            "sha256": (
                "cac0b1b8857e4c40372a83b7b7ceacc4e031c134e37543f9c324278e0d30d193"
            ),
            "sizeBytes": 10043,
        },
        "argmax_uint8": {
            "sha256": (
                "8808a95c1e47f9b0d5c60b836be60c70b7ea546fe4c8a68bcd45f85797976442"
            ),
            "sizeBytes": 10039,
        },
        "argmin_uint16": {
            "sha256": (
                "f32df27016c2dab2187a8e0dbb529adc9c3053ea355286de85d0dad42d34b794"
            ),
            "sizeBytes": 11128,
        },
        "argmax_uint16": {
            "sha256": (
                "655c0d0e6ffa96fe1b4eb32edc2b8901cc32a824755ce679fa67ea8c5d2d81aa"
            ),
            "sizeBytes": 11120,
        },
        "argmin_uint32": {
            "sha256": (
                "4be7a03ad62cf8d6d10fcb1b76a6ddcc678c0b29066d6f60d1837fff0c2d8582"
            ),
            "sizeBytes": 10409,
        },
        "argmax_uint32": {
            "sha256": (
                "95147763ae2c6e4ae665abf1cc10dc3ef0c5ff706bfb3c9dceaaa8a3b4299e38"
            ),
            "sizeBytes": 10387,
        },
        "argmin_uint64": {
            "sha256": (
                "f7881eb41cd8f1cd350c7cb8e97991bcf12e62e55444903cde693a4d7fdb1a4b"
            ),
            "sizeBytes": 13600,
        },
        "argmax_uint64": {
            "sha256": (
                "de380ced1954e96508c2c0d908e12cb259c8b93fb39f87aa43a8c9ed090e53fb"
            ),
            "sizeBytes": 13562,
        },
        "argmin_int8": {
            "sha256": (
                "d24eacd8dfb92b1f16ef2998ccc366c257696c18eb6c7d4ff773500e3e8dd02a"
            ),
            "sizeBytes": 10445,
        },
        "argmax_int8": {
            "sha256": (
                "f60f6c47cbf68ad523f6f6af24d66804a7b13039d725428f4c8bf7b2cd8e6663"
            ),
            "sizeBytes": 10447,
        },
        "argmin_int16": {
            "sha256": (
                "71757209b10e0f98f53614e926ea838410bd91f16b218efea0bfb9f789815457"
            ),
            "sizeBytes": 11033,
        },
        "argmax_int16": {
            "sha256": (
                "42d050b2d53bfd431ab08bdc2c56bd9495b5c8b23992419d744f5877ed5ec260"
            ),
            "sizeBytes": 11035,
        },
        "argmin_int32": {
            "sha256": (
                "749b8d8a52b94878aa3f7325283499322c41684c1ae9ce9e08940bc6b47415c6"
            ),
            "sizeBytes": 10937,
        },
        "argmax_int32": {
            "sha256": (
                "d6fd4859c5088daae2a4f961b706afbd1862ed517cecc8036c659900fa5bfda8"
            ),
            "sizeBytes": 10943,
        },
        "argmin_int64": {
            "sha256": (
                "f16edad6a65fac97c76d476529d14c68d81e3a4ced6d6f979a7fb29112e0168d"
            ),
            "sizeBytes": 13510,
        },
        "argmax_int64": {
            "sha256": (
                "f93bacd4eaefb4946082f55a50cd8fcf2b53991c1f11c77756f280849efd76c8"
            ),
            "sizeBytes": 13528,
        },
        "argmin_float16": {
            "sha256": (
                "2c10826d82749b2f21faacab5ce21947c423c62b2dd3507052e88b8835a92a03"
            ),
            "sizeBytes": 11322,
        },
        "argmax_float16": {
            "sha256": (
                "27600fbfbdc699b9d479cff30cc2c7e263df5c4c91c6effd5dffd8abdab740c0"
            ),
            "sizeBytes": 11324,
        },
        "argmin_float32": {
            "sha256": (
                "9a23e557a9d9c179c545e1f8bba40e09950e5f730269ef9f7035e33793362339"
            ),
            "sizeBytes": 10820,
        },
        "argmax_float32": {
            "sha256": (
                "37a90d0e9c00dfbfc8ec8c531c80fbf1d1c28a98acd27dd3800f9d44e798082d"
            ),
            "sizeBytes": 10882,
        },
        "argmin_bfloat16": {
            "sha256": (
                "00dbc073da2edaee393801fb61e014156af4053cc10c6c651bfdb876315d395f"
            ),
            "sizeBytes": 11394,
        },
        "argmax_bfloat16": {
            "sha256": (
                "0acc69a3d57f3056afc0924623f924eb4d426a63cfdd261d91929c1539d7982b"
            ),
            "sizeBytes": 11454,
        },
    },
    "opengl": {
        "argmin_bool_": {
            "sha256": (
                "7088fc8e44db7b0f6f953a12d62fdca8651323215559b288d8dd0fc6c46c56b0"
            ),
            "sizeBytes": 11274,
        },
        "argmax_bool_": {
            "sha256": (
                "2eb5ed4c0495d8e3c89ffbd4a728a4335936e8194684a0fc322b7fdec64b95e4"
            ),
            "sizeBytes": 11276,
        },
        "argmin_uint8": {
            "sha256": (
                "e0ec30c02cba37284bba5a3f7578066b734e91f8ed6ce7cbe4fb09cb96b8c2e8"
            ),
            "sizeBytes": 11068,
        },
        "argmax_uint8": {
            "sha256": (
                "bb578d14c33fb5afcbc36d18372241c2bdff6c45d75f70f084ae2412e3567c33"
            ),
            "sizeBytes": 11064,
        },
        "argmin_uint16": {
            "sha256": (
                "1c65e97966f0c728de0f1d9e50921b8c6ba3dbcfdb2a520065cd5ac342cb35b6"
            ),
            "sizeBytes": 11144,
        },
        "argmax_uint16": {
            "sha256": (
                "620b59906deaf0119f228a970ebdfa2bb9599a48aedbad8f2b4b9d692acc8571"
            ),
            "sizeBytes": 11136,
        },
        "argmin_uint32": {
            "sha256": (
                "f5cbc5e9c0d4ec75808d89b788345e8c2d10d0955e61ee46835f0fc9c7cb9094"
            ),
            "sizeBytes": 11030,
        },
        "argmax_uint32": {
            "sha256": (
                "4335a4309d8537f9352a9076fc81f5c9a86c04e3f5dc15bb92a294724c3c0a17"
            ),
            "sizeBytes": 11010,
        },
        "argmin_uint64": {
            "sha256": (
                "57a7408157b9cfd1525242a7e77b7f837b168417d8e4eddb280d4a9922615eb9"
            ),
            "sizeBytes": 11793,
        },
        "argmax_uint64": {
            "sha256": (
                "312305c09b9285c9121cfabfb4d18f47ab359976bfa5f1b1b1e56378558bac9c"
            ),
            "sizeBytes": 11755,
        },
        "argmin_int8": {
            "sha256": (
                "c87e2c9aac22e151b6ca457c9d4706dc81604b0502237f4ad20a722fded30b61"
            ),
            "sizeBytes": 11464,
        },
        "argmax_int8": {
            "sha256": (
                "fce920bb3914facf1e284c58303bf32840633e5b27e500729d57524bfd860fdc"
            ),
            "sizeBytes": 11470,
        },
        "argmin_int16": {
            "sha256": (
                "7953ec3c435223a8f685a4d4b39290164ab248f1416fe90541fe4d3efb550f4e"
            ),
            "sizeBytes": 11540,
        },
        "argmax_int16": {
            "sha256": (
                "c9841eaaa4beba89ef344f0ba0b24b8b0a0812366c8ce4d54eea0c16e7bbea66"
            ),
            "sizeBytes": 11546,
        },
        "argmin_int32": {
            "sha256": (
                "1a719e8d3d8260d73170e030b72ce737e5135a882cb53b47e0626540db8dc739"
            ),
            "sizeBytes": 11454,
        },
        "argmax_int32": {
            "sha256": (
                "f4ffbe52138cf2bae3367f9ac8bde4fee26a7e1ceebc8353793b39ffb43aad20"
            ),
            "sizeBytes": 11462,
        },
        "argmin_int64": {
            "sha256": (
                "d4fb0622380afdca8a99f397ae36891cb9e70e27a2f71669275641855c167e93"
            ),
            "sizeBytes": 11719,
        },
        "argmax_int64": {
            "sha256": (
                "b5be3a6e122267c7c1c3fbac809b76684a4d671e6041f0a29e51ba3b5fc24797"
            ),
            "sizeBytes": 11739,
        },
        "argmin_float16": {
            "sha256": (
                "08221bb0847b28c607cdc87f54f30a6faa0b9abafa025a3ca9b0f14e701112f2"
            ),
            "sizeBytes": 12680,
        },
        "argmax_float16": {
            "sha256": (
                "d1ddd908a66308e69fa0a8a7906f23eb00b16b7e74f25d862e5f183bc6c59136"
            ),
            "sizeBytes": 12686,
        },
        "argmin_float32": {
            "sha256": (
                "a301401508473b846cd0b90993cb03631c270f590ea4020f246d96a5f1c62c9d"
            ),
            "sizeBytes": 11333,
        },
        "argmax_float32": {
            "sha256": (
                "055ab02c5efac3d28728d0d39fc8f2262bd4d5dfb8fba36240d182a6c4ff2219"
            ),
            "sizeBytes": 11339,
        },
        "argmin_bfloat16": {
            "sha256": (
                "9f0a6282b1d382a7eb2d863168c5b00ed3bf53a027c7e47d1cb60b99d1bbd1a1"
            ),
            "sizeBytes": 11954,
        },
        "argmax_bfloat16": {
            "sha256": (
                "ebc5fd1520a1e8dd10c05706de821314da39f15f88dd1cc7555bb0e494b7bba7"
            ),
            "sizeBytes": 11960,
        },
    },
    "metal": {
        "argmin_bool_": {
            "sha256": (
                "f7d76d9cde4a9fd6bc950a329cf426240c579a8982e1c79a13222510618346f7"
            ),
            "sizeBytes": 7995,
        },
        "argmax_bool_": {
            "sha256": (
                "eae2860b807a42eec7e8bf1b57cc607ef3f49f28cb648a90409bcb8e8083cab8"
            ),
            "sizeBytes": 7997,
        },
        "argmin_uint8": {
            "sha256": (
                "4d1eab2d6defe71ab16546e8744721123dfed8d80b83deb276a180a236b4b8b6"
            ),
            "sizeBytes": 7252,
        },
        "argmax_uint8": {
            "sha256": (
                "2b3ebe6678a53c6d9fc67c34aab5def4cb042e18b668898b6860923d96c3317b"
            ),
            "sizeBytes": 7248,
        },
        "argmin_uint16": {
            "sha256": (
                "4995825797d17ae9b02b990cc53da4510db030affec32b1ea54b1e261b5fed8a"
            ),
            "sizeBytes": 7329,
        },
        "argmax_uint16": {
            "sha256": (
                "11f63c1b00365a8a9ac7f8d099092e395507eeafd41035950d1f72631e134097"
            ),
            "sizeBytes": 7321,
        },
        "argmin_uint32": {
            "sha256": (
                "b40dbbca4906a7b1005ad8a03965756450087b911c40585aadb64281827c775c"
            ),
            "sizeBytes": 7161,
        },
        "argmax_uint32": {
            "sha256": (
                "c3abc144d3333bb7a1d97d4c45ca93e1419a37b31fad62f65f1f3ab14851ad26"
            ),
            "sizeBytes": 7141,
        },
        "argmin_uint64": {
            "sha256": (
                "49fe5ad571c3832cdb4d09cd55b5bfd2d7e4cec0ef39d27a8cfad43c4b1683f0"
            ),
            "sizeBytes": 8210,
        },
        "argmax_uint64": {
            "sha256": (
                "e8f6ebe9156f4ec94057b5ec134f8c1b73a3e275ae120591aa8c51181eebf981"
            ),
            "sizeBytes": 8172,
        },
        "argmin_int8": {
            "sha256": (
                "eb111e149c5d802d938e1d8c8673c78db3d78f6dea5b1ba9f7cec95cd7d5195b"
            ),
            "sizeBytes": 7164,
        },
        "argmax_int8": {
            "sha256": (
                "56be6b039abfd0bb3244bc48d3a590a643cf10da399cbd6b6f1193176978bb5e"
            ),
            "sizeBytes": 7166,
        },
        "argmin_int16": {
            "sha256": (
                "68b31e6abb152ee1ce28edb4d6472a73d2019b4674b9da7d08a9d9b7cfb5eb2a"
            ),
            "sizeBytes": 7241,
        },
        "argmax_int16": {
            "sha256": (
                "84a7d11fa37a4b2991f6a88f2937704e0656a354e2e8237adb532e390878aa7d"
            ),
            "sizeBytes": 7243,
        },
        "argmin_int32": {
            "sha256": (
                "3bff201a301e8b5111d36ee97f0ef325c477a6a4fe8878c930389c1d29ef10f7"
            ),
            "sizeBytes": 7095,
        },
        "argmax_int32": {
            "sha256": (
                "c790cb1262fd1c93d4e08b6dfb7f5d08b7e23566dbbaf5ebf09ab9ee0a57f4c7"
            ),
            "sizeBytes": 7099,
        },
        "argmin_int64": {
            "sha256": (
                "0406a653f17d159a5152d0205b434180e52d1512d163c0ed976918e1cc8cb53e"
            ),
            "sizeBytes": 8142,
        },
        "argmax_int64": {
            "sha256": (
                "9bf0ea15ace7f309d680ee67cb053fd662b53a6b113ad85329f94383fafa8e62"
            ),
            "sizeBytes": 8154,
        },
        "argmin_float16": {
            "sha256": (
                "607547feaa9cb6b8cd33257f4c2d66e404e5d9de593e94cc6b417c60b7469f22"
            ),
            "sizeBytes": 6970,
        },
        "argmax_float16": {
            "sha256": (
                "f1e1220dc661ec91759a9e876e51d2de21866fc852a3f2026a9684788c62e1c4"
            ),
            "sizeBytes": 6972,
        },
        "argmin_float32": {
            "sha256": (
                "daebed04961834537bc93089b49b64c64526b5f5d8588f21a8f5a04810a0dbf0"
            ),
            "sizeBytes": 6990,
        },
        "argmax_float32": {
            "sha256": (
                "159e1eaf36336752ef9efab54e9aa9cb2daaebdb1a5224f3c6242f02c8d84a46"
            ),
            "sizeBytes": 6992,
        },
        "argmin_bfloat16": {
            "sha256": (
                "6889bde8be463785e55b64806e6dd69e6d4a0831b1fa96914f3d4bb4f5568fe5"
            ),
            "sizeBytes": 7196,
        },
        "argmax_bfloat16": {
            "sha256": (
                "bc4d57221950a893761ab9fe47517fbcad344ea201009d70da5967517e56d86f"
            ),
            "sizeBytes": 7206,
        },
    },
}

CONTRACT_PATH = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "arg_reduce.current-tree.translation.json"
)


def test_current_mlx_arg_reduce_contract_is_exact():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-arg-reduce-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert contract["entries"] == ENTRIES
    assert contract["artifacts"] == ARTIFACTS
    validation = contract["validationContract"]
    assert validation["workgroupSize"] == [32, 1, 1]
    assert validation["axisSizes"] == [32, 129]
    assert validation["axisStrides"] == [1, 2]
    assert validation["rowsPerCase"] == 2
    assert validation["ordinaryValues"] is True
    assert validation["nanAndInfinityValues"] is True
    assert validation["indexRangeAssertions"] == []
    assert validation["outputGuardCount"] == OUTPUT_GUARD_COUNT
    assert validation["observedInputCount"] == 7
    assert validation["preserveResourceOrigins"] is True
    assert validation["metadataPreconditions"] == [
        {"parameter": "shape", "element": 0, "minimum": 2, "maximum": 2},
        {"parameter": "in_strides", "element": 0, "minimum": 31, "maximum": 258},
        {"parameter": "out_strides", "element": 0, "minimum": 1, "maximum": 1},
        {"parameter": "ndim", "minimum": 1, "maximum": 1},
        {"parameter": "axis_stride", "minimum": 1, "maximum": 2},
        {"parameter": "axis_size", "minimum": 31, "maximum": 129},
    ]
    assert all(
        target["compilerEntries"] == ENTRIES
        for target in validation["targets"].values()
    )
    for name, target in validation["targets"].items():
        assert target["runtimeEntries"] == RUNTIME_ENTRIES_BY_TARGET[name]
    assert all(
        target["compilerValidationRequiredInCi"] is True
        for target in validation["targets"].values()
    )
    runtime = contract["runtimeMatrix"]
    assert runtime["scope"] == "representative-float32-numerical-parity"
    assert runtime["entries"] == RUNTIME_ENTRIES
    assert runtime["totalCaseCountByTarget"] == {
        "metal": 112,
        "opengl": 112,
        "directx": 112,
    }
    assert runtime["integerCases"] == {
        "entries": INTEGER_RUNTIME_ENTRIES,
        "axisSizes": [31, 32, 33, 129],
        "axisStrides": [1, 2],
        "entriesByTarget": {
            target: [entry for entry in entries if entry in INTEGER_RUNTIME_ENTRIES]
            for target, entries in RUNTIME_ENTRIES_BY_TARGET.items()
        },
        "caseCountByTarget": {"metal": 96, "opengl": 96, "directx": 96},
        "extremaAndLowestIndexTies": True,
        "upstreamMetallibParityTargets": ["metal"],
    }
    assert all(
        target["runtimeRequiredInCi"] is True
        for target in validation["targets"].values()
    )
    assert contract["scope"] == {
        "coveredEntryCount": 24,
        "runtimeCoveredEntryCount": 14,
        "runtimeCoveredEntryCountByTarget": {"metal": 14, "opengl": 14, "directx": 14},
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


@contextmanager
def _runtime_stage(work, stage, *, entry, target, case=None):
    record = {"stage": stage, "entry": entry, "target": target, "case": case}
    path = work / "runtime-progress.jsonl"

    def emit(status):
        payload = {**record, "status": status}
        line = json.dumps(payload, sort_keys=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(f"[mlx-runtime] {line}", flush=True)

    emit("started")
    try:
        yield
    except BaseException:
        emit("failed")
        raise
    else:
        emit("completed")


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(command, directory, name):
    result = subprocess.run(
        [str(arg) for arg in command],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=300,
    )
    _write_json(
        directory / f"{name}.json",
        {
            "command": result.args,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if not root_value or not target:
        message = "Set CROSTL_MLX_CURRENT_ROOT and CROSTL_MLX_CURRENT_TARGET"
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert target in {"directx", "opengl", "metal"}
    tool_names = {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]
    for name in tool_names:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    return root, target


def _config(output, target, entry):
    source_options = """
[project.source_options.metal]
max_template_specializations = 128
max_template_materialization_work = 8192
"""
    if target == "directx":
        source_options += f"""
[project.subgroup_width_rules]
"{SOURCE}" = 32
[project.source_options.metal.target_options.directx]
relative_wave_shuffle_out_of_range = "self"
"""
        if entry in RUNTIME_ENTRIES_BY_TARGET[target]:
            source_options += "software_subgroup_width = 32\n"
    elif target == "opengl":
        source_options += """
[project.source_options.metal.target_options.opengl]
software_subgroup_width = 32
"""
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [32, 1, 1]
{source_options}
"""


def _metal_library(source, output, root, *, upstream=False):
    air = output.with_suffix(".air")
    _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            *([] if upstream else ["-Werror"]),
            "-std=metal3.1",
            "-fno-fast-math",
            "-I",
            root,
            "-c",
            source,
            "-o",
            air,
        ],
        output.parent,
        output.stem + "-compile",
    )
    _run(["xcrun", "metallib", air, "-o", output], output.parent, output.stem + "-link")
    assert output.stat().st_size > 0
    return output


@pytest.mark.parametrize("upstream", [False, True])
def test_metal_reference_warning_policy(monkeypatch, tmp_path, upstream):
    commands = []

    def record(command, directory, label):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"compiled")

    monkeypatch.setitem(_metal_library.__globals__, "_run", record)
    source = tmp_path / "input.metal"
    output = tmp_path / "output.metallib"
    assert _metal_library(source, output, tmp_path, upstream=upstream) == output
    assert ("-Werror" in commands[0]) is not upstream
    assert "-fno-fast-math" in commands[0]
    assert commands[0][commands[0].index("-c") + 1] == source
    assert commands[1][:2] == ["xcrun", "metallib"]


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("metal-reference")
    runner = work / "arg-reduce"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "demos/integrations/mlx/tests/fixtures/arg_reduce_metal.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    library = _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )
    return runner, library


def _cases():
    for size in (32, 129):
        for stride in (1, 2):
            for special in (False, True):
                rows = []
                for row in range(2):
                    values = [
                        float((index * 17 + row * 3) % 31) for index in range(size)
                    ]
                    values[3] = values[-1] = 100.0
                    values[5] = values[-2] = -100.0
                    if special:
                        values[6] = values[14] = math.nan
                        values[2] = math.inf
                        values[1] = -math.inf
                    rows.append(values)
                # Padding must not become a candidate when axis_stride is greater than one.
                storage = [-1e20] * (2 * size * stride)
                for row, values in enumerate(rows):
                    for index, value in enumerate(values):
                        storage[row * size * stride + index * stride] = value
                request = {
                    "inputBits": [
                        struct.unpack("<I", struct.pack("<f", value))[0]
                        for value in storage
                    ],
                    "rows": 2,
                    "axisSize": size,
                    "axisStride": stride,
                    "rowStride": size * stride,
                }
                yield (
                    f"size-{size}-stride-{stride}-special-{int(special)}",
                    request,
                    rows,
                    storage,
                )


def _integer_cases(entry):
    dtype = entry.split("_", 1)[1]
    signed = dtype.startswith("int")
    width = int(dtype[3:] if signed else dtype[4:])
    low, high = (
        (-(2 ** (width - 1)), 2 ** (width - 1) - 1) if signed else (0, 2**width - 1)
    )
    base = 2**40 if width == 64 else 0
    element_format = {
        "int16": "h",
        "uint16": "H",
        "int32": "i",
        "uint32": "I",
        "int64": "q",
        "uint64": "Q",
    }[dtype]
    for size in (31, 32, 33, 129):
        for stride in (1, 2):
            rows = [
                [base + (index * 1973 + row * 13) % 30000 + 1 for index in range(size)]
                for row in range(2)
            ]
            for values in rows:
                values[3] = values[-1] = high
                values[5] = values[-2] = low
            padding = high if entry.startswith("argmax") else low
            storage = [padding] * (2 * size * stride)
            for row, values in enumerate(rows):
                for index, value in enumerate(values):
                    storage[row * size * stride + index * stride] = value
            payload = struct.pack("<" + element_format * len(storage), *storage)
            # The runner uploads raw words without converting the integer payload.
            words = struct.unpack("<" + "I" * (len(payload) // 4), payload)
            yield (
                f"size-{size}-stride-{stride}",
                {
                    "inputBits": list(words),
                    "rows": 2,
                    "axisSize": size,
                    "axisStride": stride,
                    "rowStride": size * stride,
                },
                rows,
                storage,
            )


@pytest.mark.parametrize("entry", INTEGER_RUNTIME_ENTRIES)
def test_integer_arg_reduce_case_encoding(entry):
    cases = list(_integer_cases(entry))
    assert len(cases) == 8
    for _name, request, rows, storage in cases:
        payload = struct.pack(
            "<" + "I" * len(request["inputBits"]), *request["inputBits"]
        )
        element_format = {
            "int16": "h",
            "uint16": "H",
            "int32": "i",
            "uint32": "I",
            "int64": "q",
            "uint64": "Q",
        }[entry.split("_", 1)[1]]
        assert (
            list(struct.unpack("<" + element_format * len(storage), payload)) == storage
        )
        assert len(storage) == request["rows"] * request["rowStride"]
        for row, values in enumerate(rows):
            start = row * request["rowStride"]
            stop = start + request["rowStride"]
            assert storage[start : stop : request["axisStride"]] == values
        expected_index = 5 if entry.startswith("argmin") else 3
        assert _expected_indices(entry, rows) == [expected_index, expected_index]


def _expected_indices(entry, rows):
    expected = []
    for row in rows:
        nan_indices = [index for index, value in enumerate(row) if math.isnan(value)]
        reduce = min if entry.startswith("argmin") else max
        expected.append(
            nan_indices[0]
            if nan_indices
            else reduce(range(len(row)), key=row.__getitem__)
        )
    return expected


def _runtime_input_type(target, entry):
    dtype = entry.split("_", 1)[1]
    if target == "opengl" and dtype in {"int16", "uint16"}:
        return "int32" if dtype == "int16" else "uint32"
    return dtype


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "dtype", ["float32", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
)
def test_arg_reduce_runtime_input_matches_physical_storage(target, dtype):
    expected = dtype
    if target == "opengl" and dtype.endswith("16"):
        expected = "int32" if dtype == "int16" else "uint32"
    assert _runtime_input_type(target, f"argmin_{dtype}") == expected


def _package_descriptor(translation, work):
    report_path = work / "portability-report.json"
    translation.write_json(report_path)
    artifacts = build_runtime_artifact_manifest(report_path)
    assert artifacts["success"] is True, json.dumps(artifacts, indent=2)
    assert artifacts["summary"]["artifactCount"] == 1
    artifacts_path = work / "runtime-artifacts.json"
    _write_json(artifacts_path, artifacts)
    package_dir = work / "runtime-package"
    package = build_runtime_package(artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)
    loader = build_runtime_loader_manifest(package_dir / "runtime-package.json")
    assert loader["success"] is True, json.dumps(loader, indent=2)
    assert loader["summary"]["loadUnitCount"] == 1
    assert loader["summary"]["readyLoadUnitCount"] == 1
    assert loader["summary"]["blockedLoadUnitCount"] == 0
    descriptor = build_native_loader_abi_descriptor(
        loader,
        load_unit_id=loader["loadUnits"][0]["id"],
    )
    return descriptor, package_dir


def _source_bindings(descriptor):
    return [
        binding
        for binding in descriptor["bindings"]
        if binding.get("provenance", {}).get("kind") != "generated-execution-input"
    ]


def _runtime_descriptor(translation, work, target, entry):
    descriptor, package_dir = _package_descriptor(translation, work)
    assert descriptor["target"] == target
    assert descriptor["entryPoint"]["name"] == (
        "CSMain" if target == "directx" else entry if target == "metal" else "main"
    )
    expected_types = {
        "in": _runtime_input_type(target, entry),
        "out": "uint32",
        "shape": "int32",
        "in_strides": "int64",
        "out_strides": "int64",
        "ndim": "uint64",
        "axis_stride": "int64",
        "axis_size": "uint64",
    }
    source_bindings = _source_bindings(descriptor)
    reflected_types = {
        binding["provenance"]["sourceResource"]["parameter"]: binding["scalarLayout"][
            "elementType"
        ]
        for binding in source_bindings
    }
    assert len(source_bindings) == len(expected_types)
    assert reflected_types == expected_types
    generated = [
        binding for binding in descriptor["bindings"] if binding not in source_bindings
    ]
    assert len(generated) == (1 if target == "directx" else 0)
    for binding in generated:
        assert binding["provenance"]["executionInput"] == {
            "kind": "dispatch-workgroup-count",
            "valueSource": "dispatch.workgroupCount",
            "coordinateSpace": "physical",
            "dimensions": 3,
            "memberName": binding["scalarLayout"]["memberName"],
        }
    return descriptor, package_dir


def _checked_runtime_config(config, target, entry):
    bounds = {
        "shape": (2, 2, True),
        "in_strides": (31, 258, True),
        "out_strides": (1, 1, True),
        "ndim": (1, 1, False),
        "axis_stride": (1, 2, False),
        "axis_size": (31, 129, False),
    }
    conditions = [
        dict(
            source=SOURCE,
            entryPoint=entry,
            parameter=name,
            minimum=low,
            maximum=high,
            **({"element": 0} if array else {}),
        )
        for name, (low, high, array) in bounds.items()
    ]
    metal = dict(config.source_options["metal"])
    if target == "opengl":
        options = dict(metal["target_options"][target])
        (region,) = plan_dispatch_regions((32, 2, 1), (32, 1, 1))
        options["dispatch_region"] = region.to_json()
        metal["target_options"] = {target: options}
    metal["preserve_resource_origins"] = True
    return replace(
        config,
        index_range_assertions=(),
        metadata_preconditions=conditions,
        source_options={"metal": metal},
    )


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("entry", ENTRIES)
def test_current_arg_reduce_checked_metadata_config(tmp_path, target, entry):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry), encoding="utf-8")
    config = _checked_runtime_config(load_project_config(tmp_path, path), target, entry)
    assert config.index_range_assertions == ()
    conditions = {
        item.parameter: item.to_json() for item in config.metadata_preconditions
    }
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert list(conditions.values()) == [
        {"source": SOURCE, "entryPoint": entry, **condition}
        for condition in contract["validationContract"]["metadataPreconditions"]
    ]
    assert set(conditions) == {
        "shape",
        "in_strides",
        "out_strides",
        "ndim",
        "axis_stride",
        "axis_size",
    }
    assert all(item["entryPoint"] == entry for item in conditions.values())
    assert all(item["source"] == SOURCE for item in conditions.values())
    assert {name for name, item in conditions.items() if item.get("element") == 0} == {
        "shape",
        "in_strides",
        "out_strides",
    }
    assert config.source_options["metal"]["preserve_resource_origins"] is True
    if target == "opengl":
        (region,) = plan_dispatch_regions((32, 2, 1), (32, 1, 1))
        assert (
            config.source_options["metal"]["target_options"][target]["dispatch_region"]
            == region.to_json()
        )


class _RecordingDirectXRuntime(DirectXComputeRuntime):
    def __init__(self, work):
        super().__init__()
        self.work = work
        self.dispatch_index = 0

    def dispatch_sequence(self, adapter, state, requests):
        requests = tuple(requests)
        for request in requests:
            directory = self.work / "native-dispatches" / str(self.dispatch_index)
            self.dispatch_index += 1
            directory.mkdir(parents=True)
            shader = self._shader_code(request)
            (directory / "kernel.dxil").write_bytes(shader)
            prepared = _validate_directx_register_layout(
                _complete_directx_register_layout(
                    (
                        *_prepare_directx_buffers(request.buffers),
                        *_prepare_directx_constants(request.constants),
                    )
                )
            )
            _write_json(
                directory / "dispatch.json",
                {
                    "shaderSha256": hashlib.sha256(shader).hexdigest(),
                    "entryPoint": request.entry_point,
                    "dispatch": request.dispatch.to_json(),
                    "resources": [
                        {
                            "name": resource.name,
                            "namespace": resource.namespace,
                            "binding": resource.binding_index,
                            "dtype": resource.dtype,
                            "shape": list(resource.shape),
                            "stride": resource.stride,
                            "allocationSize": resource.allocation_size,
                            "upload": resource.upload,
                            "readback": resource.readback,
                            "payloadHex": resource.payload.hex(),
                        }
                        for resource in prepared
                    ],
                },
            )
        return super().dispatch_sequence(adapter, state, requests)


def _runtime_executor(target, work):
    if target == "directx":
        adapter = DirectXRuntimeParityAdapter(runtime=_RecordingDirectXRuntime(work))
    elif target == "metal":
        adapter = MetalRuntimeParityAdapter(language_version="3.1")
    else:
        adapter = OpenGLRuntimeParityAdapter(
            runtime=OpenGLComputeRuntime(context_backends=("egl",))
        )
    return RuntimeParityExecutor(
        RuntimeTestAdapterSpec(
            adapter_id=f"mlx-current-arg-reduce-{target}",
            target=target,
            executor=target,
            adapter_kind=f"{target}-native-runtime",
        ),
        runtime_adapter=adapter,
    )


OUTPUT_GUARD = 0xABCDEF01
OUTPUT_GUARD_COUNT = 32


def _runtime_storage(entry, request, storage):
    dtype = entry.split("_", 1)[1]
    maximum = entry.startswith("argmax")
    if dtype == "float32":
        padding = math.inf if maximum else -math.inf
    else:
        signed = dtype.startswith("int")
        width = int(dtype[3:] if signed else dtype[4:])
        padding = (
            (1 << (width - int(signed))) - 1
            if maximum
            else -(1 << (width - 1)) if signed else 0
        )
    # Include inactive-lane pointer formation and the final pointer update.
    limit = (request["rows"] - 1) * request["rowStride"] + (
        (request["axisSize"] + 127) // 128 * 128 * request["axisStride"]
    )
    count = max(len(storage), limit + 1)
    if dtype in {"int16", "uint16"}:
        count += count % 2
    return [*storage, *([padding] * (count - len(storage)))]


def _runtime_dispatch_request(
    descriptor,
    package_dir,
    target,
    entry,
    request,
    storage,
    expected,
):
    storage = _runtime_storage(entry, request, storage)
    encoded_storage = [
        (
            "nan"
            if isinstance(value, float) and math.isnan(value)
            else (
                "+infinity"
                if value == math.inf
                else "-infinity" if value == -math.inf else value
            )
        )
        for value in storage
    ]
    guarded = [*expected, *([OUTPUT_GUARD] * OUTPUT_GUARD_COUNT)]
    data = {
        "in": (_runtime_input_type(target, entry), encoded_storage),
        "out": ("uint32", [OUTPUT_GUARD] * len(guarded)),
        "shape": ("int32", [request["rows"]]),
        "in_strides": ("int64", [request["rowStride"]]),
        "out_strides": ("int64", [1]),
        "ndim": ("uint64", [1]),
        "axis_stride": ("int64", [request["axisStride"]]),
        "axis_size": ("uint64", [request["axisSize"]]),
    }
    inputs, outputs, observed = {}, {}, []
    for binding in _source_bindings(descriptor):
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        dtype, values = data[parameter]
        inputs[binding["name"]] = {
            "dtype": dtype,
            "shape": [len(values)],
            "values": values,
        }
        if parameter == "out":
            outputs[binding["name"]] = {
                "dtype": "uint32",
                "shape": [len(guarded)],
                "values": guarded,
            }
        else:
            observed.append(binding["name"])
    assert len(inputs) == 8 and len(outputs) == 1 and len(observed) == 7
    dispatch = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        inputs,
        outputs,
        {"workgroupCount": [1, request["rows"], 1], "workgroupSize": [32, 1, 1]},
        expected_target=target,
        observe_inputs=observed,
    )
    assert dispatch.execution_plan is not None
    assert dispatch.execution_plan.diagnostics == ()
    assert dispatch.execution_plan.dispatch.workgroup_size == (32, 1, 1)
    assert dispatch.execution_plan.dispatch.workgroup_count == (1, request["rows"], 1)
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            (value,) = (
                item for item in dispatch.fixture.inputs if item.name == binding["name"]
            )
            assert value.dtype == "uint32" and value.shape == (3,)
            assert value.values == [1, request["rows"], 1]
            assert value.metadata["source"] == "dispatch.workgroupCount"
    return dispatch, next(iter(outputs))


def _run_runtime_parity(translation, work, target, entry, *, metal_reference=None):
    with _runtime_stage(work, "package", entry=entry, target=target):
        descriptor, package_dir = _runtime_descriptor(translation, work, target, entry)
    _write_json(work / "native-loader-abi.json", descriptor)
    executor = _runtime_executor(target, work)
    evidence = []
    cases = _integer_cases(entry) if entry in INTEGER_RUNTIME_ENTRIES else _cases()
    try:
        for name, request, rows, storage in cases:
            expected = _expected_indices(entry, rows)
            _write_json(work / f"{name}-input.json", {**request, "expected": expected})
            dispatch, output_name = _runtime_dispatch_request(
                descriptor, package_dir, target, entry, request, storage, expected
            )
            with _runtime_stage(
                work, "availability", entry=entry, target=target, case=name
            ):
                availability = executor.is_available(dispatch)
            if not availability.available:
                if os.environ.get(REQUIRE_RUNTIME_ENV) == "1":
                    pytest.fail(
                        availability.reason
                        or f"The native {target} runtime is unavailable"
                    )
                return []
            with _runtime_stage(work, "execute", entry=entry, target=target, case=name):
                result = executor.run(dispatch)
            _write_json(
                work / f"{name}-native-result.json",
                {
                    "status": result.status,
                    "outputs": result.outputs,
                    "details": result.details,
                },
            )
            assert result.status == "ok"
            guarded = [*expected, *([OUTPUT_GUARD] * OUTPUT_GUARD_COUNT)]
            assert result.outputs[output_name] == {
                "dtype": "uint32",
                "shape": [len(guarded)],
                "values": guarded,
            }
            assert len(result.details["metadataPreconditions"]) == 6
            observations = result.details["inputObservations"]
            assert len(observations) == 7
            assert all(item["unchanged"] for item in observations.values())
            reference = None
            if target == "metal":
                assert metal_reference is not None
                assert result.details["metalRuntime"]["threadExecutionWidth"] == 32
                runner, original = metal_reference
                request_path = work / f"{name}-upstream-input.json"
                _write_json(request_path, request)
                reference = json.loads(
                    _run(
                        [runner, original, entry, request_path],
                        work,
                        f"{name}-upstream",
                    )
                )
                assert reference["threadExecutionWidth"] == 32
                assert reference["indices"] == expected
            evidence.append(
                {
                    "case": name,
                    "expected": expected,
                    "translated": result.outputs[output_name]["values"][
                        : len(expected)
                    ],
                    "outputGuardCount": OUTPUT_GUARD_COUNT,
                    "checkedMetadataCount": 6,
                    "observedInputCount": 7,
                    "upstream": reference,
                }
            )
            _write_json(work / "runtime-cases.json", evidence)
        return evidence
    finally:
        runtime = getattr(getattr(executor, "runtime_adapter", None), "runtime", None)
        close = getattr(runtime, "close", None)
        if callable(close):
            close()


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_arg_reduce_native_validation(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    with tempfile.TemporaryDirectory(
        prefix=".current-arg-reduce-", dir=root
    ) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(_config(f"{work.name}/out", target, entry), encoding="utf-8")
        with _runtime_stage(tmp_path, "translate", entry=entry, target=target):
            translation = translate_project(
                _checked_runtime_config(
                    load_project_config(root, config), target, entry
                ),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
        report = translation.to_json()
        report_path = tmp_path / "report.json"
        translation.write_json(report_path)
        validation = validate_project_report(report_path)
        _write_json(tmp_path / "validation.json", validation)
        assert validation["success"] is True, validation
        assert report["diagnostics"] == []
        assert report["summary"]["translatedCount"] == 1
        assert report["summary"]["failedCount"] == 0
        (artifact,) = report["artifacts"]
        assert artifact["sourceHash"] == {
            "algorithm": "sha256",
            "value": SOURCE_SHA256,
        }
        check_generated_artifact(
            artifact,
            sha256=ARTIFACTS[target][entry]["sha256"],
            size_bytes=ARTIFACTS[target][entry]["sizeBytes"],
        )
        assert artifact["templateMaterialization"]["status"] == "materialized"
        assert artifact["templateMaterialization"]["specializationCount"] == 5
        generated = root / artifact["path"]
        output = tmp_path / generated.name
        shutil.copyfile(generated, output)
        # The copy is the reported artifact; the pin above checks its bytes.
        assert (
            hashlib.sha256(output.read_bytes()).hexdigest()
            == artifact["generatedHash"]["value"]
        )
        generated_text = output.read_text(encoding="utf-8")
        assert "unsupported" not in generated_text
        (execution,) = artifact["execution"]["entryPoints"]
        assert execution["workgroupSize"] == [32, 1, 1]
        if target == "directx":
            if entry in {"argmin_uint8", "argmax_uint8", "argmin_int8", "argmax_int8"}:
                assert "in_[uint(current_in_offset)]" in generated_text
                assert "(current_in_offset) / 4" not in generated_text
            if entry in RUNTIME_ENTRIES_BY_TARGET[target]:
                assert "WaveReadLaneAt" not in generated_text
                assert "WaveGetLane" not in generated_text
                assert "__crossgl_software_subgroup_invocation =" in generated_text
            compiler_arguments = dxc_compiler_arguments_for_source(generated_text)
            assert compiler_arguments == ("-enable-16bit-types",)
            _run(
                [
                    "dxc",
                    *compiler_arguments,
                    "-WX",
                    "-T",
                    "cs_6_6",
                    "-E",
                    "CSMain",
                    output,
                    "-Fo",
                    tmp_path / "kernel.dxil",
                ],
                tmp_path,
                "dxc",
            )
        elif target == "opengl":
            binary = tmp_path / "kernel.spv"
            _run(
                [
                    "glslangValidator",
                    "--target-env",
                    "opengl",
                    "--target-env",
                    "spirv1.3",
                    "-S",
                    "comp",
                    output,
                    "-o",
                    binary,
                ],
                tmp_path,
                "glslang",
            )
            _run(
                ["spirv-val", "--target-env", "spv1.3", binary],
                tmp_path,
                "spirv-val",
            )
        else:
            assert "current_in += axis_stride;" in generated_text
            assert "isnan_float" not in generated_text
            _metal_library(output, tmp_path / "translated.metallib", root)
        if entry in RUNTIME_ENTRIES_BY_TARGET[target]:
            evidence = _run_runtime_parity(
                translation, tmp_path, target, entry, metal_reference=metal_reference
            )
            if evidence:
                assert len(evidence) == 8
                _write_json(
                    tmp_path / "parity.json",
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "cases": evidence,
                        "indexRangeAssertions": [],
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                )
