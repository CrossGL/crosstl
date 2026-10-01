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

ROOT = Path(__file__).resolve().parents[2]
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
ARTIFACTS = {
    "directx": {
        "argmin_bool_": {
            "sha256": (
                "c6932272720f21d524037b58c0c9936725befe9f95a49d55d7bd867c1d8d1209"
            ),
            "sizeBytes": 8843,
        },
        "argmax_bool_": {
            "sha256": (
                "89b2289c06d7b912ba94b672d02246960122847b3a5705e2a4b380265e99689c"
            ),
            "sizeBytes": 8845,
        },
        "argmin_uint8": {
            "sha256": (
                "0fbfff4754ee43a3e32c4fac51f1d0b4678ea4fe24d66d8597620017dd726727"
            ),
            "sizeBytes": 6724,
        },
        "argmax_uint8": {
            "sha256": (
                "ad7814b422225f031fd8e054682562ef5e95792c08b0dee8a7faa28b74f8ef13"
            ),
            "sizeBytes": 6720,
        },
        "argmin_uint16": {
            "sha256": (
                "6164192e3984e7f3c7e72a32ad6459e1add1204ce867ba543446bfb31296509f"
            ),
            "sizeBytes": 7160,
        },
        "argmax_uint16": {
            "sha256": (
                "9e17503d06077f5f024edf6f182a7876e9d273e7a114df51f389a678421f2753"
            ),
            "sizeBytes": 7152,
        },
        "argmin_uint32": {
            "sha256": (
                "d373ca56f3fe1b1db9acb5d45c8930990cdf2b8c92f31ae35268e6e8c8339ce3"
            ),
            "sizeBytes": 6743,
        },
        "argmax_uint32": {
            "sha256": (
                "73c46188d0dc9e85123cc9aba217f1715c5e3f256e7c597b84f5ffbde58b1cbd"
            ),
            "sizeBytes": 6723,
        },
        "argmin_uint64": {
            "sha256": (
                "12df94acb201919585920dc28443b9ce0af054322ca8408984432b7b57555084"
            ),
            "sizeBytes": 9136,
        },
        "argmax_uint64": {
            "sha256": (
                "4151f3b38978eda9333af8377fb6961ab22da5c3fe2a9862f0c593942dc82d75"
            ),
            "sizeBytes": 9092,
        },
        "argmin_int8": {
            "sha256": (
                "522b446519c207f1c22381974e45e6880441b029d205065002d2af26eae1f71b"
            ),
            "sizeBytes": 7012,
        },
        "argmax_int8": {
            "sha256": (
                "9605f1e40982e58476a27d3790394fdf24a8f93396ee95965131f4986c231be1"
            ),
            "sizeBytes": 7014,
        },
        "argmin_int16": {
            "sha256": (
                "7a6a9bfaf57c20c9998231bc05e9d4f61df220cf3bc4ec2c4e915dea50acd6f1"
            ),
            "sizeBytes": 7086,
        },
        "argmax_int16": {
            "sha256": (
                "a99be0c7060a989c59c6a5642a226cf2f5a8f23b9ad01d113de3ba2234c2812f"
            ),
            "sizeBytes": 7088,
        },
        "argmin_int32": {
            "sha256": (
                "7d24677eb8a526dc4ad3ee76cfdd25f443d98980f9dac3c91fe8f8fc760f3e23"
            ),
            "sizeBytes": 7006,
        },
        "argmax_int32": {
            "sha256": (
                "e78c42e0b34dabaf5a706789820adfbe5d66c325453dd2fd8d711538af87c38e"
            ),
            "sizeBytes": 7010,
        },
        "argmin_int64": {
            "sha256": (
                "4b806f4fe9641f86ff2977f917421788b721771d8e5a447c0f2a4d69e9963f0d"
            ),
            "sizeBytes": 9065,
        },
        "argmax_int64": {
            "sha256": (
                "14090b8956251750431e03d21b520950f3d763ab4eedc78c1852d5cda29b9986"
            ),
            "sizeBytes": 9067,
        },
        "argmin_float16": {
            "sha256": (
                "32694370fdfde0225f394ea8cc9cbda45b0f75326a69f128ba26f96df0458a84"
            ),
            "sizeBytes": 7981,
        },
        "argmax_float16": {
            "sha256": (
                "689d3136de6087e31e6b7c507a89856b91db82692d4eded9325d004e7fe7d50c"
            ),
            "sizeBytes": 7983,
        },
        "argmin_float32": {
            "sha256": (
                "7b7be4e0680d7ad3e94b39943655f7d5375b839b2b51ce187c5f8bbfaa0d2cd4"
            ),
            "sizeBytes": 7621,
        },
        "argmax_float32": {
            "sha256": (
                "3779b83c2d8f3b8a9cf6e25e95709277b4f099be51c3b1cb1059274ee4ce3eff"
            ),
            "sizeBytes": 7623,
        },
        "argmin_bfloat16": {
            "sha256": (
                "24e468f45cb9a02689692ec0cd58a47e87019c64033dfaf08a48327e26e54f9a"
            ),
            "sizeBytes": 8178,
        },
        "argmax_bfloat16": {
            "sha256": (
                "70c700b2ef81ccca01caff5a447fd200b866be3535c1e91a4565f395f0873a67"
            ),
            "sizeBytes": 8238,
        },
    },
    "opengl": {
        "argmin_bool_": {
            "sha256": (
                "2f85b68ce0ae0b8270cc881ef224fd9b064db6376b10608c66b4f1835141188a"
            ),
            "sizeBytes": 7992,
        },
        "argmax_bool_": {
            "sha256": (
                "471db471200b4c9e992d674d1755c49f4c53039173a131e90189bf65d4f1a10f"
            ),
            "sizeBytes": 7994,
        },
        "argmin_uint8": {
            "sha256": (
                "0940733a6a2c9b0e5410f3b11162251d5d7848fa9fe3c535aca1e6998c201096"
            ),
            "sizeBytes": 7783,
        },
        "argmax_uint8": {
            "sha256": (
                "6b3aab58f2d2e3cda99c60267094f97e861794ac81a95cad1a4589390e0ff057"
            ),
            "sizeBytes": 7779,
        },
        "argmin_uint16": {
            "sha256": (
                "bae374d8a713410eaafa1c22678a364787446cb3f726a9e281b82e117ff9881f"
            ),
            "sizeBytes": 7842,
        },
        "argmax_uint16": {
            "sha256": (
                "dd0d61d0986a86097bb65218cb42246f11f81ec00f7e10ab55cb461f515ccb36"
            ),
            "sizeBytes": 7834,
        },
        "argmin_uint32": {
            "sha256": (
                "51147ac6941209a204e1391653b9380e1fee2a98af9ba0fa37d35f6c1c67639d"
            ),
            "sizeBytes": 7728,
        },
        "argmax_uint32": {
            "sha256": (
                "0b6566b0566e2cd2362fed85342f285b5726404775aa0ba8b5e1123013f4a0ae"
            ),
            "sizeBytes": 7708,
        },
        "argmin_uint64": {
            "sha256": (
                "fa5ca7edb1c38009611fcea2543bf62a014e746a1a0c6831b7fad6e853189840"
            ),
            "sizeBytes": 8491,
        },
        "argmax_uint64": {
            "sha256": (
                "47b39dc66d59a9b582691863ab88449840f241c81f8c10166dee7f070f967328"
            ),
            "sizeBytes": 8449,
        },
        "argmin_int8": {
            "sha256": (
                "b2c600b8714722baf887c49879c629080be1af100333869fe53b3fcd48ace19b"
            ),
            "sizeBytes": 8196,
        },
        "argmax_int8": {
            "sha256": (
                "60de637a64abfe2dc65aa55ca9c096455ce6806bb69b33d45abe40e7a784ff23"
            ),
            "sizeBytes": 8202,
        },
        "argmin_int16": {
            "sha256": (
                "7f8c3a1e13e2b9fc62d906b54933e1f2454a6c5238f438750c1fa89c05e24315"
            ),
            "sizeBytes": 8255,
        },
        "argmax_int16": {
            "sha256": (
                "f0903658329658f4b9afdea5d7d76940f1681a675135b36c88932883ccaf204c"
            ),
            "sizeBytes": 8261,
        },
        "argmin_int32": {
            "sha256": (
                "a833f05973e6284cf2cddcf78352ce21dc20fe46de372fc5fcf9ae4ed8e43bfe"
            ),
            "sizeBytes": 8169,
        },
        "argmax_int32": {
            "sha256": (
                "632226571a93c10ff3ed00c057f4f9f470d3f971725836f19a34923ec0274407"
            ),
            "sizeBytes": 8177,
        },
        "argmin_int64": {
            "sha256": (
                "2e4da4dbbb358132f5fd284c82f589d6ae1fa17199623a1fd784cdd1a020d005"
            ),
            "sizeBytes": 8436,
        },
        "argmax_int64": {
            "sha256": (
                "6dba1e2e63ba2686dff16c91519ed4f7ac1fa88f9d6c481d97419e32a3f1907b"
            ),
            "sizeBytes": 8442,
        },
        "argmin_float16": {
            "sha256": (
                "a790706f99b8f55159a6f83e2544df1cf85db1a4beef325f47f7895707a74967"
            ),
            "sizeBytes": 9403,
        },
        "argmax_float16": {
            "sha256": (
                "5eff78a5d24194fdcdd9a8f518fe09196e303b99d72a1fe021154adba26c0130"
            ),
            "sizeBytes": 9409,
        },
        "argmin_float32": {
            "sha256": (
                "9eea8e0f829c42e0c9923e11d54c56f641a7092eece1f21972296a32c7519f7b"
            ),
            "sizeBytes": 8014,
        },
        "argmax_float32": {
            "sha256": (
                "89dfa0d662fea2f4f4ea4b8c5d1d1671c88c51b2959fec97de31b9af83db51a5"
            ),
            "sizeBytes": 8020,
        },
        "argmin_bfloat16": {
            "sha256": (
                "667b5a117917153ec33513ee711292314d3ecff497d9d868cfb4e1636eab9b43"
            ),
            "sizeBytes": 8238,
        },
        "argmax_bfloat16": {
            "sha256": (
                "d072606bad3ac289298de751f7c1bfc722aae590ed81a9532b0b1b6db9dce1bc"
            ),
            "sizeBytes": 8244,
        },
    },
    "metal": {
        "argmin_bool_": {
            "sha256": (
                "2e3a402aaf2e524edb4574e18bd04f5b13fa3f9d13e7db88a3266c4d9fc2375d"
            ),
            "sizeBytes": 5159,
        },
        "argmax_bool_": {
            "sha256": (
                "fdcb9df68b8cb77f220df8b5663fcf9cc622ff19927f3eaee49ab7d5cd40fe91"
            ),
            "sizeBytes": 5161,
        },
        "argmin_uint8": {
            "sha256": (
                "451cef5cc11bd2572715c1eff945fbab9ac9cca9a36aedeebf7ddf09548c1ebe"
            ),
            "sizeBytes": 4258,
        },
        "argmax_uint8": {
            "sha256": (
                "cbe5c59c3499531f4e4192263c17c083127d74955a85f3677b3c3a724207fece"
            ),
            "sizeBytes": 4254,
        },
        "argmin_uint16": {
            "sha256": (
                "3e36f54049b75512ee2784955cab4a4e94cde38c8dac9268ae882f745444ceef"
            ),
            "sizeBytes": 4299,
        },
        "argmax_uint16": {
            "sha256": (
                "c42352f2b77b323cda5e3b1ef6411c70c40fd97097f6abbacc163b6b7a6a4bae"
            ),
            "sizeBytes": 4291,
        },
        "argmin_uint32": {
            "sha256": (
                "eecaaef0ce843c24cb76a32fdef26fd098a8f16f9b60e1b9fae5a6492c8eb443"
            ),
            "sizeBytes": 4301,
        },
        "argmax_uint32": {
            "sha256": (
                "1ea64f4ebb09c8baf10409a3501f7e949009450117ce723e9891f2357eae906d"
            ),
            "sizeBytes": 4283,
        },
        "argmin_uint64": {
            "sha256": (
                "da2d79272fd1320b1c4fed503fa14848163581958651850ff44fea00bf4851b8"
            ),
            "sizeBytes": 5348,
        },
        "argmax_uint64": {
            "sha256": (
                "84bcf7d986e9b5def16d252c0043557b5455dd9850b2aa58e7e0c6d4e0558cad"
            ),
            "sizeBytes": 5310,
        },
        "argmin_int8": {
            "sha256": (
                "3b6b71214a3573de424637d23de749dd39deb60ae706a2889c427cf6bbb9fc0f"
            ),
            "sizeBytes": 4216,
        },
        "argmax_int8": {
            "sha256": (
                "1063c0ebe784587065e18ad6f025e9a9cd73046e2f5ab29b4526ecd17723a95a"
            ),
            "sizeBytes": 4218,
        },
        "argmin_int16": {
            "sha256": (
                "a2b798e83541462ec119186655081a0ea9cd077fb3c203a2dc881186e30d8660"
            ),
            "sizeBytes": 4257,
        },
        "argmax_int16": {
            "sha256": (
                "ec54bf919f083b5f87a43e75eebffa39076ef86d0e056fd90b01be958b1da26e"
            ),
            "sizeBytes": 4259,
        },
        "argmin_int32": {
            "sha256": (
                "cb9938644ac23dd4d5c23fdd7d389ea0d9aa8cbfb01af83d8f27bae79cfed166"
            ),
            "sizeBytes": 4259,
        },
        "argmax_int32": {
            "sha256": (
                "11e5ab4faeb5714b6f7fb67b90ef3d3bd5f983466449a6107583d3f6653bd5d7"
            ),
            "sizeBytes": 4261,
        },
        "argmin_int64": {
            "sha256": (
                "4a565e19d1b5efaba4af26a3f699964720a36045a9b12bcc1116e718f58714d3"
            ),
            "sizeBytes": 5304,
        },
        "argmax_int64": {
            "sha256": (
                "328ea0d98eb3b17d6726f859b8d45909c3be33cfdaaabbc1f5b1c89fad42a90a"
            ),
            "sizeBytes": 5322,
        },
        "argmin_float16": {
            "sha256": (
                "c7b7e0fb0cf16b820af3a8a55b8bb3592917064486614a35bff30b7bef932210"
            ),
            "sizeBytes": 4114,
        },
        "argmax_float16": {
            "sha256": (
                "56724e2ed446cc62a40dc74746b65d7383ee6f3c5d7703563c4f5974d4d8fdc5"
            ),
            "sizeBytes": 4116,
        },
        "argmin_float32": {
            "sha256": (
                "c91ed3700660707c003c828624ded5d9abfdcc4e7282051aed70959c49c551c6"
            ),
            "sizeBytes": 4134,
        },
        "argmax_float32": {
            "sha256": (
                "7fc5d6a97d8a5aa4bee597c439185ff1d31c86d49395a13f63e1be854fa9c8e2"
            ),
            "sizeBytes": 4136,
        },
        "argmin_bfloat16": {
            "sha256": (
                "42eb1fdd01354adb70cd422e13b7828568f19f86e93c5b90e6da7bff2461badd"
            ),
            "sizeBytes": 4318,
        },
        "argmax_bfloat16": {
            "sha256": (
                "b66521e976f8fbee6abce186c9e53515b21cbbea519f4a44f00a8ebd488933c8"
            ),
            "sizeBytes": 4328,
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
    assert all(
        target["runtimeEntries"] == RUNTIME_ENTRIES
        for target in validation["targets"].values()
    )
    assert all(
        target["compilerValidationRequiredInCi"] is True
        for target in validation["targets"].values()
    )
    runtime = contract["runtimeMatrix"]
    assert runtime["scope"] == "representative-float32-numerical-parity"
    assert runtime["entries"] == RUNTIME_ENTRIES
    assert all(
        target["runtimeRequiredInCi"] is True
        for target in validation["targets"].values()
    )
    assert contract["scope"] == {
        "coveredEntryCount": 24,
        "runtimeCoveredEntryCount": 2,
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
            ROOT / "tests/fixtures/runtime_verification/mlx_arg_reduce_metal.swift",
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
            if entry not in RUNTIME_ENTRIES:
                return
            runner, original = metal_reference
            evidence = []
            for name, request, rows, _storage in _cases():
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
