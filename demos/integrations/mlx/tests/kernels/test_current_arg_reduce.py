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
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from crosstl.project.native_runtime_drivers import (
    _complete_directx_register_layout,
    _prepare_directx_buffers,
    _prepare_directx_constants,
    _validate_directx_register_layout,
)

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
METAL_INTEGER_RUNTIME_ENTRIES = [
    f"{operation}_{dtype}{width}"
    for width in (16, 32, 64)
    for dtype in ("uint", "int")
    for operation in ("argmin", "argmax")
]
ARTIFACTS = {
    "directx": {
        "argmin_bool_": {
            "sha256": (
                "6df0716e13739d023f72a1b4da4e9a3e64b4734d565cfc2e5acc78baa53e5ae4"
            ),
            "sizeBytes": 8947,
        },
        "argmax_bool_": {
            "sha256": (
                "a08dcaf5d8ed5be980cb394df0af21ee19a713ac80af33349034b5dc505f7a89"
            ),
            "sizeBytes": 8949,
        },
        "argmin_uint8": {
            "sha256": (
                "37be693aa42c3bc31c2f97a8d3d7b1cbb6ec4d07e4703ad2186da7451a7a1d8c"
            ),
            "sizeBytes": 6982,
        },
        "argmax_uint8": {
            "sha256": (
                "532099709a78d7909ae2b15c73b5ef1ceb36725c015fcc193bd163385563ee56"
            ),
            "sizeBytes": 6978,
        },
        "argmin_uint16": {
            "sha256": (
                "9fcfeea59e66bdab7dca9ddc702cefc8bd27d29d204871797c4477f147634a19"
            ),
            "sizeBytes": 7264,
        },
        "argmax_uint16": {
            "sha256": (
                "f9a8a23afc0a3fbbdcf41c6411f392100d0bd6db78120f2d518b6d1d4781fe94"
            ),
            "sizeBytes": 7256,
        },
        "argmin_uint32": {
            "sha256": (
                "4bf79acd3dc0af91e660813ec4cb7993c5b3e1375bc3ffe32a6c24e8849522ed"
            ),
            "sizeBytes": 6847,
        },
        "argmax_uint32": {
            "sha256": (
                "8158bcd57be461b5ac0c9ff50c4a8b4e48eb799e2fda8aa0e057cbc6a3196be0"
            ),
            "sizeBytes": 6827,
        },
        "argmin_uint64": {
            "sha256": (
                "33d0d3f48c58af51087960ce21affb1288e081b0fa6c3c67535ac23af9b5e859"
            ),
            "sizeBytes": 9240,
        },
        "argmax_uint64": {
            "sha256": (
                "1a34328a13989ed141ce2b22ef5a4f2729b1952a9523d9c79d49b17b44b92c70"
            ),
            "sizeBytes": 9196,
        },
        "argmin_int8": {
            "sha256": (
                "2c08e3a540b8bced73590416fe038ac8fc038ad77acebc4aa237d8d1e37453b5"
            ),
            "sizeBytes": 7401,
        },
        "argmax_int8": {
            "sha256": (
                "6daa7dfd9c2e57c82d9aac2aa948f5f355403172cbf2e6c9ad120d6a64b9cd47"
            ),
            "sizeBytes": 7403,
        },
        "argmin_int16": {
            "sha256": (
                "f95b8aa3d4278e3a4312798c1509f9cb139719041cd52ca2fdb5cde0d9d55375"
            ),
            "sizeBytes": 7190,
        },
        "argmax_int16": {
            "sha256": (
                "1bcd634cf0620d624d81ed79635704685d409627d78c7b3bf65d9c9d54c97777"
            ),
            "sizeBytes": 7192,
        },
        "argmin_int32": {
            "sha256": (
                "f7466ec7e236feb83d090833f92267cd4c257d71c0aa9aacd1b353d1a0b98689"
            ),
            "sizeBytes": 7110,
        },
        "argmax_int32": {
            "sha256": (
                "2ff2e564dba1f07732822bac00ed6258c4bf9982e411171683cb4e2210c3ef5e"
            ),
            "sizeBytes": 7114,
        },
        "argmin_int64": {
            "sha256": (
                "82c01862a14364dddc58e1046873d57e160915bb8724c7774844bdeee830874d"
            ),
            "sizeBytes": 9169,
        },
        "argmax_int64": {
            "sha256": (
                "75903394a8cd0c5a3f40b6cbb9179df4a0cd7192cd04231ed791700136d3c136"
            ),
            "sizeBytes": 9171,
        },
        "argmin_float16": {
            "sha256": (
                "7737e8f5556b4a131897b59c8239123f73ab66eb172c671cafec997e82ad4537"
            ),
            "sizeBytes": 8227,
        },
        "argmax_float16": {
            "sha256": (
                "fb37a9243920e9357b56450a53fdbc61227f1c2be3b397cb2c9c6d06083f5285"
            ),
            "sizeBytes": 8229,
        },
        "argmin_float32": {
            "sha256": (
                "c1b0a153d30da43f310a489390ee6d59846052116a9f4415fdf0b2a832f9640e"
            ),
            "sizeBytes": 7725,
        },
        "argmax_float32": {
            "sha256": (
                "90795a457c45e9574d5c297befb90470f0741b0e331b28d4425011a43954ca7a"
            ),
            "sizeBytes": 7787,
        },
        "argmin_bfloat16": {
            "sha256": (
                "aa81e68f71228d084bc1e141e0af085b4d769cfcfecf0cae4a1a709f4738eb94"
            ),
            "sizeBytes": 8282,
        },
        "argmax_bfloat16": {
            "sha256": (
                "299b998df17cf0103686a39f12bacc9ef01be986449e588283ad70962677fa27"
            ),
            "sizeBytes": 8342,
        },
    },
    "opengl": {
        "argmin_bool_": {
            "sha256": (
                "1fc50fdbdf7bf258d2889c2be92a846e0516ec3d8b519ab9a2ed9ae96f3cede2"
            ),
            "sizeBytes": 8171,
        },
        "argmax_bool_": {
            "sha256": (
                "c4e2c3b2dc5540f003dad72cd101a988040aa8e15aeb6ff146845546b074624a"
            ),
            "sizeBytes": 8173,
        },
        "argmin_uint8": {
            "sha256": (
                "9f7295f1acccab3fc7434abf21cc94e6630bd672a9ef0aff9e5a8c4f22e68a16"
            ),
            "sizeBytes": 7962,
        },
        "argmax_uint8": {
            "sha256": (
                "6ab4270cf1725e5c8a2eedf687a8503d7de88cff297135fcbb5709c933f20f9f"
            ),
            "sizeBytes": 7958,
        },
        "argmin_uint16": {
            "sha256": (
                "42ef076afc61d1a73f1e4374625b699269453bfef7ce01c69bc44760c018b5af"
            ),
            "sizeBytes": 8021,
        },
        "argmax_uint16": {
            "sha256": (
                "5c8bec2628ef05ca5a3d42d096d7236ef85e4b41d2b83484e5c9988d94ecc5ef"
            ),
            "sizeBytes": 8013,
        },
        "argmin_uint32": {
            "sha256": (
                "4aa11f68ed12396aab0409916fc6e1b8e11e26b32079e02624e0223fc94b557b"
            ),
            "sizeBytes": 7907,
        },
        "argmax_uint32": {
            "sha256": (
                "17c6d3e44199a234a3e512593614ea661609858b01a88fb665338eb157e04ea6"
            ),
            "sizeBytes": 7887,
        },
        "argmin_uint64": {
            "sha256": (
                "b1c052baa967b3c38a55714f5ce23011b19f15dc92213b0a923c9bf49fcd0153"
            ),
            "sizeBytes": 8670,
        },
        "argmax_uint64": {
            "sha256": (
                "6536c1e9d1894e958e445f24b0580597cc2ad57c05f7d58e079fad89d3458e98"
            ),
            "sizeBytes": 8628,
        },
        "argmin_int8": {
            "sha256": (
                "aeccecbf9f9b96c1fdcf8e2ed9038a7abb9e82d30f62ea045f2cd6e0ca5aa830"
            ),
            "sizeBytes": 8375,
        },
        "argmax_int8": {
            "sha256": (
                "a1b4ba5a3ffe6e7bfee2cfba40296ce2a689770b8eba44bab348d86c6a9777ca"
            ),
            "sizeBytes": 8381,
        },
        "argmin_int16": {
            "sha256": (
                "5b716e5603d58045d754ce58149463d52d4e820b574cc34c6fa9b8c9c4916e9d"
            ),
            "sizeBytes": 8434,
        },
        "argmax_int16": {
            "sha256": (
                "8ede7c37791da8c37570562e8a70e7c412ae9439f220b3ca153cc6bfde6d8e01"
            ),
            "sizeBytes": 8440,
        },
        "argmin_int32": {
            "sha256": (
                "508b7acd99dd5b888e3050bdaa3de31bba58e50555af1cfc7a7cc1e1aa1e2841"
            ),
            "sizeBytes": 8348,
        },
        "argmax_int32": {
            "sha256": (
                "28fdbc744e1677e1139ead434251fc94b774b4758bda88b14b7ada0cf4551735"
            ),
            "sizeBytes": 8356,
        },
        "argmin_int64": {
            "sha256": (
                "2e28adafe28d7a8362eb285f667ab34e279d70699c308897e303c47b8cea198a"
            ),
            "sizeBytes": 8615,
        },
        "argmax_int64": {
            "sha256": (
                "f8abfe97178a36e1483829b9815aa359272819d2adec49a8fd99b6a2367910b8"
            ),
            "sizeBytes": 8621,
        },
        "argmin_float16": {
            "sha256": (
                "9b55b1a5adf9bc45a4ab4deec55387ee96ab8067e2fca358dc71c97714e5e39b"
            ),
            "sizeBytes": 9540,
        },
        "argmax_float16": {
            "sha256": (
                "3f4b07ac5c5ff7589063a87a900fbeb09981ddf40302ab7f91111f3f83afa1f4"
            ),
            "sizeBytes": 9546,
        },
        "argmin_float32": {
            "sha256": (
                "ee802da16f965a7dda2da0ac37f8f7a05001248fe3df768eca06e86836db389e"
            ),
            "sizeBytes": 8193,
        },
        "argmax_float32": {
            "sha256": (
                "16b40c743d66925d15c411f0969bdaebdaeb2aa20770656664d710719219a327"
            ),
            "sizeBytes": 8199,
        },
        "argmin_bfloat16": {
            "sha256": (
                "5c7a083032c1489b5849bb1f967075c99e31cd9777e12a78fcd084bd28586cdc"
            ),
            "sizeBytes": 8797,
        },
        "argmax_bfloat16": {
            "sha256": (
                "0e714cc919987a25caf3de3ca35acd381449003f720c06d657b5a311be02f620"
            ),
            "sizeBytes": 8803,
        },
    },
    "metal": {
        "argmin_bool_": {
            "sha256": (
                "48b4214a0a1461067ed81ce71f5f29158a5e82e37f969c68e667871bf3a034c1"
            ),
            "sizeBytes": 5231,
        },
        "argmax_bool_": {
            "sha256": (
                "4e192b25ddf5da8ed952c8a314639a5c2877f3e6aaca2fe6def4993f437e58ff"
            ),
            "sizeBytes": 5233,
        },
        "argmin_uint8": {
            "sha256": (
                "905adc21c642fca01fd93879f3bc966f4749f77cd523cad5480819901fec9eb4"
            ),
            "sizeBytes": 4501,
        },
        "argmax_uint8": {
            "sha256": (
                "400c1fa82e466b8d1de318f3cb80f0e961cb51b1154b738d4e633bb5730f1b40"
            ),
            "sizeBytes": 4497,
        },
        "argmin_uint16": {
            "sha256": (
                "98f89ddbf72abea643e49af14976c61a29f447e933d573b0fe99a7604c2573f1"
            ),
            "sizeBytes": 4389,
        },
        "argmax_uint16": {
            "sha256": (
                "0a3f61769d847ec708b02636ca735ced1dc1917a3bcf1b55db3a8f014c08e643"
            ),
            "sizeBytes": 4381,
        },
        "argmin_uint32": {
            "sha256": (
                "a21fd3ef30a8dd605121620aa87c712f7c8b96e2589a05599ef4f862d8810025"
            ),
            "sizeBytes": 4375,
        },
        "argmax_uint32": {
            "sha256": (
                "d4a48982d8e56ace8e8c2a86ffe39d1b8770a47551a932b0ffb3d97e3fd6c589"
            ),
            "sizeBytes": 4355,
        },
        "argmin_uint64": {
            "sha256": (
                "dad0c1b529c7af9dac0cdafbb49699103ecc9c9e7ef77963b11f1f01a0d87426"
            ),
            "sizeBytes": 5424,
        },
        "argmax_uint64": {
            "sha256": (
                "d7cd2df32a90e555823da11443a0932348f3d87721b99e7de427172c3277840c"
            ),
            "sizeBytes": 5386,
        },
        "argmin_int8": {
            "sha256": (
                "2fb9fcb77481baaa8daa16f96230c62d031b0677d7373a55efbff3761532ac9c"
            ),
            "sizeBytes": 4433,
        },
        "argmax_int8": {
            "sha256": (
                "8622d7005c115ff49accd1c45cacfbab3f44ae0e4993577d841ca7761ef8731b"
            ),
            "sizeBytes": 4435,
        },
        "argmin_int16": {
            "sha256": (
                "d48ca4d9e54fc3ae297ab0753469e30e9c6361fd1815381316c76d03dc8a483b"
            ),
            "sizeBytes": 4345,
        },
        "argmax_int16": {
            "sha256": (
                "545c1d0bf96dad0b32e90caef805982e57a52161d07193f5354f9d41831ebeb8"
            ),
            "sizeBytes": 4347,
        },
        "argmin_int32": {
            "sha256": (
                "fe22c75c073af75f4a5ac49b2825f826258c46d68e32f050fd0e359878e58bd5"
            ),
            "sizeBytes": 4331,
        },
        "argmax_int32": {
            "sha256": (
                "b222dc6326e8860d075754094401b3e12d7ee7e9d4a11758593f5bb6fba0a6b7"
            ),
            "sizeBytes": 4335,
        },
        "argmin_int64": {
            "sha256": (
                "bb08aee274e4173b74b7e54b274f29b880fd0ee0aa26ccb6be16d0d538b02a84"
            ),
            "sizeBytes": 5378,
        },
        "argmax_int64": {
            "sha256": (
                "fb5e266fe010b8c25a2152203b874831cc37251001028f8f8b212f5087b51ccd"
            ),
            "sizeBytes": 5390,
        },
        "argmin_float16": {
            "sha256": (
                "f2d793f603a7c2fa3d43d331adf5c3177662e6093b1d90da534b6cfe605a01fc"
            ),
            "sizeBytes": 4162,
        },
        "argmax_float16": {
            "sha256": (
                "5a68cb65060ef3e4c02b64e798e5deb1a40f066a3db43d6d121a1d80d81a53d9"
            ),
            "sizeBytes": 4164,
        },
        "argmin_float32": {
            "sha256": (
                "2cff5b36b3a8437d0e5f6be68b982f7ca6b168e36f4a7b6b7d4b42d47cd0ac26"
            ),
            "sizeBytes": 4182,
        },
        "argmax_float32": {
            "sha256": (
                "7b3e5c739949232212d9c155863510d4fb8b1dc64cd720375747a679894f01e2"
            ),
            "sizeBytes": 4184,
        },
        "argmin_bfloat16": {
            "sha256": (
                "93285c3d4d4b0e9a0c9fd5334bdc62fcb67481276da40dc5c572108e8b4952f8"
            ),
            "sizeBytes": 4366,
        },
        "argmax_bfloat16": {
            "sha256": (
                "a8c10582cae5046fdd206170182f9ff58b2d91db6572ee6cef55938d8b624959"
            ),
            "sizeBytes": 4376,
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
    assert all(
        target["compilerEntries"] == ENTRIES
        for target in validation["targets"].values()
    )
    for name, target in validation["targets"].items():
        assert target["runtimeEntries"] == RUNTIME_ENTRIES + (
            METAL_INTEGER_RUNTIME_ENTRIES if name == "metal" else []
        )
    assert all(
        target["compilerValidationRequiredInCi"] is True
        for target in validation["targets"].values()
    )
    runtime = contract["runtimeMatrix"]
    assert runtime["scope"] == "representative-float32-numerical-parity"
    assert runtime["entries"] == RUNTIME_ENTRIES
    assert runtime["metalIntegerCases"] == {
        "entries": METAL_INTEGER_RUNTIME_ENTRIES,
        "axisSizes": [31, 32, 33, 129],
        "axisStrides": [1, 2],
        "caseCount": 96,
        "extremaAndLowestIndexTies": True,
        "upstreamMetallibParity": True,
    }
    assert all(
        target["runtimeRequiredInCi"] is True
        for target in validation["targets"].values()
    )
    assert contract["scope"] == {
        "coveredEntryCount": 24,
        "runtimeCoveredEntryCount": 2,
        "runtimeCoveredEntryCountByTarget": {"metal": 14, "opengl": 2, "directx": 2},
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
        if entry in RUNTIME_ENTRIES:
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
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "in_idx + current_index * axis_stride"
minimum = 0
maximum = 515
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "out_idx"
minimum = 0
maximum = 1
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


@pytest.mark.parametrize("entry", METAL_INTEGER_RUNTIME_ENTRIES)
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


def _runtime_binding_names(target, entry):
    if target == "directx":
        return {
            "input": "in_",
            "output": "out_",
            "shape": "shape",
            "inStrides": "in_strides",
            "outStrides": "out_strides",
            "ndim": f"{entry}_ndim_Constants",
            "axisStride": f"{entry}_axis_stride_Constants",
            "axisSize": f"{entry}_axis_size_Constants",
        }
    return {
        "input": "in_Buffer",
        "output": "out_Buffer",
        "shape": "shapeBuffer",
        "inStrides": "in_stridesBuffer",
        "outStrides": "out_stridesBuffer",
        "ndim": f"{entry}_ndim_Args",
        "axisStride": f"{entry}_axis_stride_Args",
        "axisSize": f"{entry}_axis_size_Args",
    }


def _runtime_descriptor(translation, work, target, entry):
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
    assert descriptor["target"] == target
    assert descriptor["entryPoint"]["name"] == (
        "CSMain" if target == "directx" else "main"
    )
    names = _runtime_binding_names(target, entry)
    expected_types = {
        names["input"]: "float32",
        names["output"]: "uint32",
        names["shape"]: "int32",
        names["inStrides"]: "int64",
        names["outStrides"]: "int64",
        names["ndim"]: "uint64",
        names["axisStride"]: "int64",
        names["axisSize"]: "uint64",
    }
    reflected_types = {
        binding["name"]: binding["scalarLayout"]["elementType"]
        for binding in descriptor["bindings"]
        if binding["name"] in expected_types
    }
    assert reflected_types == expected_types
    return descriptor, package_dir


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


def _runtime_dispatch_request(
    descriptor,
    package_dir,
    target,
    entry,
    request,
    storage,
    expected,
):
    names = _runtime_binding_names(target, entry)
    encoded_storage = []
    for value in storage:
        if math.isnan(value):
            encoded_storage.append("nan")
        elif value == math.inf:
            encoded_storage.append("+infinity")
        elif value == -math.inf:
            encoded_storage.append("-infinity")
        else:
            encoded_storage.append(value)
    dispatch = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        {
            names["input"]: {
                "dtype": "float32",
                "shape": [len(storage)],
                "values": encoded_storage,
            },
            names["shape"]: {
                "dtype": "int32",
                "shape": [1],
                "values": [request["rows"]],
            },
            names["inStrides"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [request["rowStride"]],
            },
            names["outStrides"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [1],
            },
            names["ndim"]: {
                "dtype": "uint64",
                "shape": [1],
                "values": [1],
            },
            names["axisStride"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [request["axisStride"]],
            },
            names["axisSize"]: {
                "dtype": "uint64",
                "shape": [1],
                "values": [request["axisSize"]],
            },
        },
        {
            names["output"]: {
                "dtype": "uint32",
                "shape": [request["rows"]],
                "values": expected,
            }
        },
        [1, request["rows"], 1],
        expected_target=target,
    )
    assert dispatch.execution_plan is not None
    assert dispatch.execution_plan.diagnostics == ()
    assert dispatch.execution_plan.dispatch.workgroup_size == (32, 1, 1)
    assert dispatch.execution_plan.dispatch.workgroup_count == (
        1,
        request["rows"],
        1,
    )
    return dispatch, names["output"]


def _run_runtime_parity(translation, work, target, entry):
    with _runtime_stage(work, "package", entry=entry, target=target):
        descriptor, package_dir = _runtime_descriptor(
            translation,
            work,
            target,
            entry,
        )
    _write_json(work / "native-loader-abi.json", descriptor)
    executor = _runtime_executor(target, work)
    evidence = []
    for name, request, rows, storage in _cases():
        expected = _expected_indices(entry, rows)
        _write_json(work / f"{name}-input.json", {**request, "expected": expected})
        dispatch, output_name = _runtime_dispatch_request(
            descriptor,
            package_dir,
            target,
            entry,
            request,
            storage,
            expected,
        )
        with _runtime_stage(
            work, "availability", entry=entry, target=target, case=name
        ):
            availability = executor.is_available(dispatch)
        if not availability.available:
            if os.environ.get(REQUIRE_RUNTIME_ENV) == "1":
                pytest.fail(
                    availability.reason or f"The native {target} runtime is unavailable"
                )
            return []
        with _runtime_stage(work, "execute", entry=entry, target=target, case=name):
            result = executor.run(dispatch)
        assert result.status == "ok"
        assert result.outputs[output_name]["dtype"] == "uint32"
        assert result.outputs[output_name]["shape"] == [request["rows"]]
        assert result.outputs[output_name]["values"] == expected
        evidence.append(
            {
                "case": name,
                "expected": expected,
                "translated": result.outputs[output_name]["values"],
            }
        )
        _write_json(work / "runtime-cases.json", evidence)
    return evidence


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
                load_project_config(root, config),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
        report = translation.to_json()
        report_path = tmp_path / "report.json"
        translation.write_json(report_path)
        assert validate_project_report(report_path)["success"] is True
        assert report["diagnostics"] == []
        assert report["summary"]["translatedCount"] == 1
        assert report["summary"]["failedCount"] == 0
        (artifact,) = report["artifacts"]
        assert artifact["sourceHash"] == {
            "algorithm": "sha256",
            "value": SOURCE_SHA256,
        }
        assert artifact["generatedHash"] == {
            "algorithm": "sha256",
            "value": ARTIFACTS[target][entry]["sha256"],
        }
        assert artifact["generatedSizeBytes"] == ARTIFACTS[target][entry]["sizeBytes"]
        assert artifact["templateMaterialization"]["status"] == "materialized"
        assert artifact["templateMaterialization"]["specializationCount"] == 5
        generated = root / artifact["path"]
        output = tmp_path / generated.name
        shutil.copyfile(generated, output)
        assert (
            hashlib.sha256(output.read_bytes()).hexdigest()
            == ARTIFACTS[target][entry]["sha256"]
        )
        generated_text = output.read_text(encoding="utf-8")
        assert "unsupported" not in generated_text
        (execution,) = artifact["execution"]["entryPoints"]
        assert execution["workgroupSize"] == [32, 1, 1]
        if target == "directx":
            if entry in {"argmin_uint8", "argmax_uint8", "argmin_int8", "argmax_int8"}:
                assert "in_[uint(current_in_offset)]" in generated_text
                assert "(current_in_offset) / 4" not in generated_text
            if entry in RUNTIME_ENTRIES:
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
            if entry in RUNTIME_ENTRIES:
                evidence = _run_runtime_parity(translation, tmp_path, target, entry)
                if evidence:
                    _write_json(
                        tmp_path / "parity.json",
                        {
                            "commit": MLX_COMMIT,
                            "entry": entry,
                            "cases": evidence,
                            "upstreamMlxTestSuiteExecuted": False,
                        },
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
            if entry in RUNTIME_ENTRIES:
                evidence = _run_runtime_parity(translation, tmp_path, target, entry)
                if evidence:
                    _write_json(
                        tmp_path / "parity.json",
                        {
                            "commit": MLX_COMMIT,
                            "entry": entry,
                            "cases": evidence,
                            "upstreamMlxTestSuiteExecuted": False,
                        },
                    )
        else:
            assert "current_in += axis_stride;" in generated_text
            assert "isnan_float" not in generated_text
            library = _metal_library(output, tmp_path / "translated.metallib", root)
            if entry not in RUNTIME_ENTRIES + METAL_INTEGER_RUNTIME_ENTRIES:
                return
            runner, original = metal_reference
            evidence = []
            cases = (
                _integer_cases(entry)
                if entry in METAL_INTEGER_RUNTIME_ENTRIES
                else _cases()
            )
            for name, request, rows, _storage in cases:
                request_path = tmp_path / f"{name}.json"
                _write_json(request_path, request)
                expected = _expected_indices(entry, rows)
                observed = {}
                for label, compiled in (
                    ("upstream", original),
                    ("translated", library),
                ):
                    result = json.loads(
                        _run(
                            [runner, compiled, entry, request_path],
                            tmp_path,
                            f"{name}-{label}",
                        )
                    )
                    observed[label] = result
                    assert result["threadExecutionWidth"] == 32
                    assert result["indices"] == expected, (
                        name,
                        label,
                        result,
                        expected,
                    )
                evidence.append({"case": name, "expected": expected, **observed})
            _write_json(
                tmp_path / "parity.json",
                {
                    "commit": MLX_COMMIT,
                    "entry": entry,
                    "cases": evidence,
                    "upstreamMlxTestSuiteExecuted": False,
                },
            )
