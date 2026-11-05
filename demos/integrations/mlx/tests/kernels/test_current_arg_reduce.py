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
                "d4edec64191f1415a2d143685c45be8a00abfe0aababd4eb7c301ec292feb1cf"
            ),
            "sizeBytes": 12011,
        },
        "argmax_bool_": {
            "sha256": (
                "8ec55b33bfebda9c40846ee3c8dcb71568a374dc084f380113247063020017a4"
            ),
            "sizeBytes": 12013,
        },
        "argmin_uint8": {
            "sha256": (
                "3a6229b3cfa45db649a3fc69724c335148ec9d79e645033cb7d94a9f17bd1d75"
            ),
            "sizeBytes": 10049,
        },
        "argmax_uint8": {
            "sha256": (
                "21aa23cae256f2515996de53d5bfdf9a667754db952130860fddd5c75f75805b"
            ),
            "sizeBytes": 10045,
        },
        "argmin_uint16": {
            "sha256": (
                "75c0d32341c5e1f51af2d7f92812e335995ddd87b553f7a1f7c1e21eb15e536d"
            ),
            "sizeBytes": 11134,
        },
        "argmax_uint16": {
            "sha256": (
                "1fbec6b3eece7771f0a929740310fc53f6e90efdbce2e9a52f2384888d3cb48d"
            ),
            "sizeBytes": 11126,
        },
        "argmin_uint32": {
            "sha256": (
                "4a6d707f373cef63cdefff93c9ea2fab4a4683f723b9c282a872127e9a4796df"
            ),
            "sizeBytes": 10415,
        },
        "argmax_uint32": {
            "sha256": (
                "ef7e2df1ce7216a3a4d0a142e939c4880fd04ba437edc18a073f47b2c2176646"
            ),
            "sizeBytes": 10393,
        },
        "argmin_uint64": {
            "sha256": (
                "9bfa0881ebb8e5594da77526169aaec115e48db8c4aae6e055a351942253f11d"
            ),
            "sizeBytes": 13606,
        },
        "argmax_uint64": {
            "sha256": (
                "bb2728dae409baedf318061753b71e04777bbbb8ff8762b4024b1e9f9a65cdd0"
            ),
            "sizeBytes": 13568,
        },
        "argmin_int8": {
            "sha256": (
                "10ecf14362d0abb7a03476e5f747680790d4d0fb20d0d07f36df3e4f50f75b35"
            ),
            "sizeBytes": 10451,
        },
        "argmax_int8": {
            "sha256": (
                "d687f4c74202163479581ed23f69758f47b0f24ae263c06a542e9e3cb810e224"
            ),
            "sizeBytes": 10453,
        },
        "argmin_int16": {
            "sha256": (
                "2fae5097035189b5d9e46d0a8316d359b694c3b4f9f7b904c9b25e53a550f1a6"
            ),
            "sizeBytes": 11039,
        },
        "argmax_int16": {
            "sha256": (
                "7e512407a7748f622ba9563491fd28a51410e7386604376cbc3d7cc6e3744a1c"
            ),
            "sizeBytes": 11041,
        },
        "argmin_int32": {
            "sha256": (
                "252e5e9cc0596efe7b39efcf7b0fa8c85a397b6cb006e83e798a0591626aea3f"
            ),
            "sizeBytes": 10943,
        },
        "argmax_int32": {
            "sha256": (
                "f0d2aa952e5d407efa57b4f0fe0decb794a2ad64f6e9c749fdfb8cecf8e0e4e0"
            ),
            "sizeBytes": 10949,
        },
        "argmin_int64": {
            "sha256": (
                "154c997cfa4a5c38d921f48bbfef3bc3794dd24b57bf956554c8b2c36a4ef0e9"
            ),
            "sizeBytes": 13516,
        },
        "argmax_int64": {
            "sha256": (
                "4c342f2623cab182164d6a66cfbf0e0cac62062d5e9f7daf9002d62493599045"
            ),
            "sizeBytes": 13534,
        },
        "argmin_float16": {
            "sha256": (
                "a61794aeeaba7ce5cb8135019c5d77ba148b01f7c2a206e9946aeb574b02ad8b"
            ),
            "sizeBytes": 11328,
        },
        "argmax_float16": {
            "sha256": (
                "3931345866af7f959443238e191cd62df3cf6f5756a803fe82a3e040d2fe6cd4"
            ),
            "sizeBytes": 11330,
        },
        "argmin_float32": {
            "sha256": (
                "fb106d12d101c71e6ef379d8a732561ba90d23475ad7e1ffd2f5136f582431b6"
            ),
            "sizeBytes": 10826,
        },
        "argmax_float32": {
            "sha256": (
                "80b9e0e72ee8a9805a0312b2064141aa201fd98d3b8324206d5da7f8f9fd37cf"
            ),
            "sizeBytes": 10888,
        },
        "argmin_bfloat16": {
            "sha256": (
                "421d67c2d3c014af53f8380a582d03bc8824d0fe637db3dbe4e68f3936ee7b66"
            ),
            "sizeBytes": 11400,
        },
        "argmax_bfloat16": {
            "sha256": (
                "4afa881a8f8b419caebce3e5adaf17d10f3c9e948863a212ba6e9ea2948ccbe0"
            ),
            "sizeBytes": 11460,
        },
    },
    "opengl": {
        "argmin_bool_": {
            "sha256": (
                "44a93d15b61f303355193b1de16d6eac9e405fcfacdee06a379df5bae5bd8481"
            ),
            "sizeBytes": 11280,
        },
        "argmax_bool_": {
            "sha256": (
                "a7439558ade60e09542a90f55c39f6b2128b4ea78ebd8a6a6c4d3592a769c863"
            ),
            "sizeBytes": 11282,
        },
        "argmin_uint8": {
            "sha256": (
                "0fe67d367d1e4eea605f036111f493de279c6e3ceef62a233da90d44c981e466"
            ),
            "sizeBytes": 11074,
        },
        "argmax_uint8": {
            "sha256": (
                "c0d9ce3df2c816a5c769135786c2640cd14e9f23e0d89606af82d1fc001f2d98"
            ),
            "sizeBytes": 11070,
        },
        "argmin_uint16": {
            "sha256": (
                "9a216e30e80cecef0ae51a947588a347fb3ee30951a303be09ad49d2ac070d39"
            ),
            "sizeBytes": 11150,
        },
        "argmax_uint16": {
            "sha256": (
                "11689e91d22a61806359dc4ecac3235ac33bc62adb35fd2c8eac953d36a0d897"
            ),
            "sizeBytes": 11142,
        },
        "argmin_uint32": {
            "sha256": (
                "db1cbefb16ab2e605d813754ec19e397471732cc76f930d841c49da829fcc830"
            ),
            "sizeBytes": 11036,
        },
        "argmax_uint32": {
            "sha256": (
                "3f72cbf83c5253cab9fcc9574c8fe902c46738c835272d5328cf4dcfc135657b"
            ),
            "sizeBytes": 11016,
        },
        "argmin_uint64": {
            "sha256": (
                "adedfacc47629f147278c6e428220e38d9f08aa284b83e3a3d9c653383950312"
            ),
            "sizeBytes": 11799,
        },
        "argmax_uint64": {
            "sha256": (
                "f49a6208d5eae7f2022ddd40b497f7fe585a9b04dbbef0e52f0901c2456e2c6a"
            ),
            "sizeBytes": 11761,
        },
        "argmin_int8": {
            "sha256": (
                "593bd638d8a9c1ad9a5407b7c894fe2e0412af85a774748f8c6c0a43e339d771"
            ),
            "sizeBytes": 11470,
        },
        "argmax_int8": {
            "sha256": (
                "219da523e1de0211e437b746bcb3e7031b51c1a77216e21410988bc6242e6f97"
            ),
            "sizeBytes": 11476,
        },
        "argmin_int16": {
            "sha256": (
                "896d8268db21c7e5ae3cc59372c381a6177d877891a018909a4e18f12095d345"
            ),
            "sizeBytes": 11546,
        },
        "argmax_int16": {
            "sha256": (
                "896852ee57db1f4e79c6e098ac3eb82ba2426ddb4416f7782490ac60da5ea615"
            ),
            "sizeBytes": 11552,
        },
        "argmin_int32": {
            "sha256": (
                "dc41d951cc040f3061d9e79225f5d1608b6b77f5dcb2e0584115f3e0e394dc0e"
            ),
            "sizeBytes": 11460,
        },
        "argmax_int32": {
            "sha256": (
                "c9a580ebfb6c079d48c8975beada655c61052dccf2fdef95601a5d9d84e1108d"
            ),
            "sizeBytes": 11468,
        },
        "argmin_int64": {
            "sha256": (
                "32e2ff3cc23398a49fd3da1850fe7570e6b9f559a6523c447238bc0629355a2b"
            ),
            "sizeBytes": 11725,
        },
        "argmax_int64": {
            "sha256": (
                "abe0a2dd85cdf1608ba6631e7750714b36b8a9f3c843b7270bbce98e8d59a783"
            ),
            "sizeBytes": 11745,
        },
        "argmin_float16": {
            "sha256": (
                "063e31739f5ff906366ebb1df8cb87cf6e9eca64b1e19ef8b3e6c3a698d08987"
            ),
            "sizeBytes": 12686,
        },
        "argmax_float16": {
            "sha256": (
                "f912339d06c2a0c1f47bb4ea84519c3ec16227a1ed15f1659eecc9f1cff35303"
            ),
            "sizeBytes": 12692,
        },
        "argmin_float32": {
            "sha256": (
                "9c4d418e572730a7ff0421abe0c791b0ec98f7168d7ad08df98ed82fb220b819"
            ),
            "sizeBytes": 11339,
        },
        "argmax_float32": {
            "sha256": (
                "266005e9ec4596bd1ab2c54ab64e39d3cd1035bbf0d9971a9a577e22c1612120"
            ),
            "sizeBytes": 11345,
        },
        "argmin_bfloat16": {
            "sha256": (
                "bad5cdceade7f939386b4d854ce563341548020ea8495666c934e298e51e1b52"
            ),
            "sizeBytes": 11960,
        },
        "argmax_bfloat16": {
            "sha256": (
                "81c89f90e888f15a3cbfaa82d9cfec467fb93ab0ceb6192beca9ba7157cf9ead"
            ),
            "sizeBytes": 11966,
        },
    },
    "metal": {
        "argmin_bool_": {
            "sha256": (
                "ed4dd455196b25b58805f40ac5306a32566ad9ff407dbcdebdb74185bdff0a42"
            ),
            "sizeBytes": 7983,
        },
        "argmax_bool_": {
            "sha256": (
                "c2e639b6329fa065113ec01658c8d4d02f2183edcadb3d64b8ef0b102c35948f"
            ),
            "sizeBytes": 7985,
        },
        "argmin_uint8": {
            "sha256": (
                "e71dcad08d57a3b21f5442ff036b3801fa1011a021d1e1031ef5855d2481c3db"
            ),
            "sizeBytes": 7267,
        },
        "argmax_uint8": {
            "sha256": (
                "83b2ae03164c14f353f26370d7d0107a3c65128394d0c2abd2c748a2b24873d1"
            ),
            "sizeBytes": 7263,
        },
        "argmin_uint16": {
            "sha256": (
                "5958073e7434c6a2d24aee4b21e6310fdbab9d1f469955fdc88ae8ad8fb303fb"
            ),
            "sizeBytes": 7344,
        },
        "argmax_uint16": {
            "sha256": (
                "c0be31b31df53adc1b7b9430f9df483abee6520a0309ad1ec2c7a4dde0624ae3"
            ),
            "sizeBytes": 7336,
        },
        "argmin_uint32": {
            "sha256": (
                "9061f1d65b869e6120a602764602838e02611caf1642f71c6119c3e2f2163ec7"
            ),
            "sizeBytes": 7176,
        },
        "argmax_uint32": {
            "sha256": (
                "91cb18c21c4d991d65f3f098645111a099fef95c0602a7d17f8cd77e8a9cd22c"
            ),
            "sizeBytes": 7156,
        },
        "argmin_uint64": {
            "sha256": (
                "54b8187c2c4900defb4921b742f1b3c821fe3500ada9c90472c502bfd7d10f2a"
            ),
            "sizeBytes": 8198,
        },
        "argmax_uint64": {
            "sha256": (
                "7b5b3e636435b6029cc5c397437023f048c348785629e13a7c25e87f9567c65d"
            ),
            "sizeBytes": 8160,
        },
        "argmin_int8": {
            "sha256": (
                "3488c4d272d366f0e9a735f25e1986fb3c138f6b2ef26edcc097b8ebd4762191"
            ),
            "sizeBytes": 7179,
        },
        "argmax_int8": {
            "sha256": (
                "5744e80057011d5860cf06c601611985d3c61868e8682f72d0f60f43706df0ac"
            ),
            "sizeBytes": 7181,
        },
        "argmin_int16": {
            "sha256": (
                "b99e50e35adf26ec23fb71483fc0af66c6ca70df5a8f63dca7bb7e2c4eaeae1a"
            ),
            "sizeBytes": 7256,
        },
        "argmax_int16": {
            "sha256": (
                "9787d7b6f1a3f13e402fc00221e31293c309f75a06d8c9a31bb53a151abc53f7"
            ),
            "sizeBytes": 7258,
        },
        "argmin_int32": {
            "sha256": (
                "24222fb0aa54b67b86ae16f813b72e63e4521993dd5b3b9ddd9b4ab8a51e5d2d"
            ),
            "sizeBytes": 7110,
        },
        "argmax_int32": {
            "sha256": (
                "94db8f09c64fb9bd6270e6d052cdafad2fbe7934582de4e269c277ae2cf8643a"
            ),
            "sizeBytes": 7114,
        },
        "argmin_int64": {
            "sha256": (
                "02c85a9058a88644ebb5dcdeae83bc267367981568bc46c6c93ec207283343d5"
            ),
            "sizeBytes": 8130,
        },
        "argmax_int64": {
            "sha256": (
                "903ecb00bed20f3d1da3586b4b791ca1ba738e7445b9431f072ce8943064be37"
            ),
            "sizeBytes": 8142,
        },
        "argmin_float16": {
            "sha256": (
                "aab9b27ebe95ff2dab2249a17cc1c776acbab7bb0db5aaaaad3630fa306cddfe"
            ),
            "sizeBytes": 6985,
        },
        "argmax_float16": {
            "sha256": (
                "fde795e004e62404532a1a7d99158b4c221355fe00ed85aad2e2c600e54cd396"
            ),
            "sizeBytes": 6987,
        },
        "argmin_float32": {
            "sha256": (
                "1b54d8454fb49a62befe235c83123c85299c4cf5e45d3810c1a3121a822d94e4"
            ),
            "sizeBytes": 7005,
        },
        "argmax_float32": {
            "sha256": (
                "6e3fd9d72fadc8c2081591b29f077d93c3798b2249f81b86ef566388fe5db8c2"
            ),
            "sizeBytes": 7007,
        },
        "argmin_bfloat16": {
            "sha256": (
                "d93b7257c03df2d202d4e2556f3a25f7a6405decfb2008065d4042d412f8c9e1"
            ),
            "sizeBytes": 7211,
        },
        "argmax_bfloat16": {
            "sha256": (
                "7df6b211532462c567a50da4aee0326d36a61515185cf4020acf2690104f18bc"
            ),
            "sizeBytes": 7221,
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
