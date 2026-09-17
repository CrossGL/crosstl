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

ROOT = Path(__file__).resolve().parents[2]
MLX_COMMIT = "d9add9d11f3154111a4c85f267ec2fd307ecd18e"
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
                "9b6eca3aa715896d7cdf4ce3465c96777a5b20d089862ef01baf07f223fc2676"
            ),
            "sizeBytes": 8741,
        },
        "argmax_bool_": {
            "sha256": (
                "a4320903ca7fc56c363946695ac43f2ffb45d0ad85acaee09f0e507639b20b2e"
            ),
            "sizeBytes": 8743,
        },
        "argmin_uint8": {
            "sha256": (
                "42ed837ee36b995b138e18b61251083fbd4401ea777c37b54b3beb33ea77e0de"
            ),
            "sizeBytes": 6690,
        },
        "argmax_uint8": {
            "sha256": (
                "b072c5bef3c99e82f7e8b58183e2d814991d179687cbed711fa1e29ee129cc6a"
            ),
            "sizeBytes": 6686,
        },
        "argmin_uint16": {
            "sha256": (
                "66068689426dffad2564eda40ba11467c269401fc892c2d4743fdcf5efed31b4"
            ),
            "sizeBytes": 7126,
        },
        "argmax_uint16": {
            "sha256": (
                "d41655607b8bc651f4b3a391011cb8f023ca4dd242710a7bcd118685d37e10b9"
            ),
            "sizeBytes": 7118,
        },
        "argmin_uint32": {
            "sha256": (
                "85f52e6cc3f8ae8b3428e7e7460b1a3fdc3a69cf1a55d033984d5d24a94ca6df"
            ),
            "sizeBytes": 6709,
        },
        "argmax_uint32": {
            "sha256": (
                "5245ebe5b423e586e866036ace1d8ddab6951d8e7ebf1f0f0ded470cf586db62"
            ),
            "sizeBytes": 6689,
        },
        "argmin_uint64": {
            "sha256": (
                "a9603d8967aebdb67920f2c25c7cfa1a36b0e663ef0f8db7a3a9b543e6f3897a"
            ),
            "sizeBytes": 9034,
        },
        "argmax_uint64": {
            "sha256": (
                "1d00bf4e89e0a5e3e50708ea92edcd410f45d73e0ea43b9109d6d487b5a001f6"
            ),
            "sizeBytes": 8990,
        },
        "argmin_int8": {
            "sha256": (
                "871d0bd231fd4a844ffb047cbcb649246a5067c122e88b86bc1d21ce3d251012"
            ),
            "sizeBytes": 6978,
        },
        "argmax_int8": {
            "sha256": (
                "a5edb61ee04828f255a755f7d8e8b9fe4a37197f01b375cf6b0db6445519cbd4"
            ),
            "sizeBytes": 6980,
        },
        "argmin_int16": {
            "sha256": (
                "352632c957c26bed94e9adfe1d3c2cea4470852b80d589b6bfce24e2570f6186"
            ),
            "sizeBytes": 7052,
        },
        "argmax_int16": {
            "sha256": (
                "664a0b39119a171fd06988d9d180424c9c311ff1f776da4726b9fa954975b03a"
            ),
            "sizeBytes": 7054,
        },
        "argmin_int32": {
            "sha256": (
                "7afcc44299a8526a95b8a68f075289d0d391e2f3eaa60b604083358b79fb0e07"
            ),
            "sizeBytes": 6972,
        },
        "argmax_int32": {
            "sha256": (
                "249f5e5dca802869f6842bb8b5d9b0a8f814c9628b041e963d472fda9165db6c"
            ),
            "sizeBytes": 6976,
        },
        "argmin_int64": {
            "sha256": (
                "86cb622d330f427f449a8fc4e7c5d9dc232322859aa410160db34d864f918916"
            ),
            "sizeBytes": 8963,
        },
        "argmax_int64": {
            "sha256": (
                "8a575b4cccdf65f9fd52cf068505cae387f1f726edf6a3ac4ac7be9d1b03b6c7"
            ),
            "sizeBytes": 8965,
        },
        "argmin_float16": {
            "sha256": (
                "94219f02373811fcf39fe3062c3e624d332019673b29f2d15b69458fcb59933e"
            ),
            "sizeBytes": 6842,
        },
        "argmax_float16": {
            "sha256": (
                "30891177b7a79922c54e08803cf8af7104de900ed2918c6f10355e1f228da857"
            ),
            "sizeBytes": 6844,
        },
        "argmin_float32": {
            "sha256": (
                "0768cdc9658dd81ab76c02b6d59baf9a7b1103dae09ef56e77e09e181c0ed825"
            ),
            "sizeBytes": 6813,
        },
        "argmax_float32": {
            "sha256": (
                "35be05ff5f8485644cb3cddad86c1e17b19eaa05599611baf95997e9ff646d88"
            ),
            "sizeBytes": 6815,
        },
        "argmin_bfloat16": {
            "sha256": (
                "fc486b3bcbbcaaa07f4b86794b15107bf71c8c82a7bd6bdf26e4139d84b56f9b"
            ),
            "sizeBytes": 8144,
        },
        "argmax_bfloat16": {
            "sha256": (
                "454b48d309341a7e2e6a333cc24f9b2b3d98fe3f6e72276140bad6b4da80da88"
            ),
            "sizeBytes": 8204,
        },
    },
    "opengl": {
        "argmin_bool_": {
            "sha256": (
                "8fcc570c24a8fc3692b2df1324a2897839992e311a909da9c9349fa4f750b756"
            ),
            "sizeBytes": 7843,
        },
        "argmax_bool_": {
            "sha256": (
                "1c782526027fe46a22e4b27c7517748ffd2c8447f86fbb2a87c24db2648b6905"
            ),
            "sizeBytes": 7845,
        },
        "argmin_uint8": {
            "sha256": (
                "3789491c0fe6b0c60e6b16e8ae565d547ab5b416146a9feeb881c1a99bbf4c0a"
            ),
            "sizeBytes": 7634,
        },
        "argmax_uint8": {
            "sha256": (
                "61784ed7517b40bab351bad20407454fc661a51fd81c360f88de0f9d0f7a9874"
            ),
            "sizeBytes": 7630,
        },
        "argmin_uint16": {
            "sha256": (
                "e1f1f2e95903d13c32bf2513d1430f6fce23a618a482352d2bb30da3e838bb5c"
            ),
            "sizeBytes": 7693,
        },
        "argmax_uint16": {
            "sha256": (
                "1d8b2b9a3f579ad9a9acaede3888fae2b29458aacf2d2e0e6c62953f9662ea3e"
            ),
            "sizeBytes": 7685,
        },
        "argmin_uint32": {
            "sha256": (
                "ae710523b1692c8d7be6d48311d07f388e1f8af0989da495440e14024a7ae5f4"
            ),
            "sizeBytes": 7579,
        },
        "argmax_uint32": {
            "sha256": (
                "816dd8c8b95018f364f6ac7b6c9aef86b4459fb4240ca2a78b501739a1bacba0"
            ),
            "sizeBytes": 7559,
        },
        "argmin_uint64": {
            "sha256": (
                "3d437ae9d74ee1c52f605f5b9be0c576b8d2a241656b9e0943dd62e48822bf68"
            ),
            "sizeBytes": 8342,
        },
        "argmax_uint64": {
            "sha256": (
                "052cd1ae53122aa96bd95d94cf589236eaeef8c5afff6abb3e394a1aa0467872"
            ),
            "sizeBytes": 8300,
        },
        "argmin_int8": {
            "sha256": (
                "6f36dfdfbd74cce02181e87d430b3d0ce424b3bd557b59c34eb5e0990092a8ae"
            ),
            "sizeBytes": 7932,
        },
        "argmax_int8": {
            "sha256": (
                "15b8b1864587ca8291e6f0f40b1dc768cbac3ffcbb84b36fb4d7fe05edb4c587"
            ),
            "sizeBytes": 7938,
        },
        "argmin_int16": {
            "sha256": (
                "b1e740224e5104c6bb5f31a52eccc733a1964d62ea44ee9a78750f6a37e21231"
            ),
            "sizeBytes": 7991,
        },
        "argmax_int16": {
            "sha256": (
                "5745a484cd5f934b16fa77001beae46e5b5b6c3ed768444765316643a15127de"
            ),
            "sizeBytes": 7997,
        },
        "argmin_int32": {
            "sha256": (
                "bcd8f70148f40e1025cd8e87d99c8b2d574bc05e4072f1a2f85168aa67db33d1"
            ),
            "sizeBytes": 7905,
        },
        "argmax_int32": {
            "sha256": (
                "57d6879c5247587652ff99732a2b0ac5b848f471cfc0c015ca6806b8a3edaaff"
            ),
            "sizeBytes": 7913,
        },
        "argmin_int64": {
            "sha256": (
                "7ad3a1b4001ceec958a7e572f9f5c54efcfc420b23a955893a71e5c1264c87cb"
            ),
            "sizeBytes": 8287,
        },
        "argmax_int64": {
            "sha256": (
                "449ec891dfd9c355e9806cc42b72bf478f293651e3d8e3fab922c2e55be23a6d"
            ),
            "sizeBytes": 8293,
        },
        "argmin_float16": {
            "sha256": (
                "5591f0d48daf2a1cf546b0f82bf047da9472acdb1e8252530088eba7b760293f"
            ),
            "sizeBytes": 7726,
        },
        "argmax_float16": {
            "sha256": (
                "9aa80e7436b4d4253f72e206a536250ae7b76144e37b7970fac7fe0f0a8a0e1a"
            ),
            "sizeBytes": 7732,
        },
        "argmin_float32": {
            "sha256": (
                "14fcee01e9c7f5b5bb05a32262e6c3932cf80ea8bdd50582f26ae77f8976d320"
            ),
            "sizeBytes": 7750,
        },
        "argmax_float32": {
            "sha256": (
                "4b0595da819da2efaa81672604182e87f870b6fde30cb255bbcf434d2ddc7e21"
            ),
            "sizeBytes": 7756,
        },
        "argmin_bfloat16": {
            "sha256": (
                "77104705bd5981225d23ed627bf62d239fef60a6d74ea8d27684e52043ac1e1e"
            ),
            "sizeBytes": 7974,
        },
        "argmax_bfloat16": {
            "sha256": (
                "ec1977efd5fb41e24de63817977c6f386bd01a6296b288459d8f66775ac0c50f"
            ),
            "sizeBytes": 7980,
        },
    },
    "metal": {
        "argmin_bool_": {
            "sha256": (
                "02ae43a54bd28570f69b3733745c690c02685afdf3591e5f7bfd5acc531cf4af"
            ),
            "sizeBytes": 4923,
        },
        "argmax_bool_": {
            "sha256": (
                "cda6ef6fc99842bca63546bceb052d062e73de01ed93c3183f462bd90dd4e98a"
            ),
            "sizeBytes": 4925,
        },
        "argmin_uint8": {
            "sha256": (
                "c1dcd224bda22c06fe25c90689be79dac5277a87c7f7416530168789ea43e727"
            ),
            "sizeBytes": 4117,
        },
        "argmax_uint8": {
            "sha256": (
                "9542580180bad0a0b201099151a93aa59fe1ef71b7e0b9760f71f122473c37ca"
            ),
            "sizeBytes": 4113,
        },
        "argmin_uint16": {
            "sha256": (
                "e07ecb0f8bcfb12505e946b86a55ec1672ee86f9575d747af1e248e905415b16"
            ),
            "sizeBytes": 4158,
        },
        "argmax_uint16": {
            "sha256": (
                "b5cfbd64a6f95b082d0db194718c563efd9c53a6002321712bf79c17332bfc50"
            ),
            "sizeBytes": 4150,
        },
        "argmin_uint32": {
            "sha256": (
                "f48065265af057f5477bdab5ed6d5232fa9d267b65af8693f5d6c9304053c58e"
            ),
            "sizeBytes": 4160,
        },
        "argmax_uint32": {
            "sha256": (
                "fe8276460c6399e82bf55ae0f44aa8cdfd77073ff47a279373074aec2a334d1a"
            ),
            "sizeBytes": 4142,
        },
        "argmin_uint64": {
            "sha256": (
                "716042ba37fa828f4d5988d0737b3d00156964c38a13dfa969065c7a960b65f4"
            ),
            "sizeBytes": 5112,
        },
        "argmax_uint64": {
            "sha256": (
                "39513c295098950b3ea2c1a7ed975d384dc9e44280af159c361b05b8270bba76"
            ),
            "sizeBytes": 5074,
        },
        "argmin_int8": {
            "sha256": (
                "f7785b93901d5b1a2c6b9370a5c87fded5a1d1d809129bf987e5a002b3036b6c"
            ),
            "sizeBytes": 4075,
        },
        "argmax_int8": {
            "sha256": (
                "e2d4170d03a9c5c1603118c181c97e5de3e39def46d63405dd4d2f471f9c9dad"
            ),
            "sizeBytes": 4077,
        },
        "argmin_int16": {
            "sha256": (
                "98aaad1ec921acd02ba7626f70dcc7ad64e0daad042d6098019ff2ff09fb39d5"
            ),
            "sizeBytes": 4116,
        },
        "argmax_int16": {
            "sha256": (
                "a8b012bdabcdd1ade236f6460a53abf3e1090948e54fc28a972d1e1029286466"
            ),
            "sizeBytes": 4118,
        },
        "argmin_int32": {
            "sha256": (
                "1303af3b44a5a6e2f8a860e7a9840ce2de6ed8ea846c34bac40cb528eb82b582"
            ),
            "sizeBytes": 4118,
        },
        "argmax_int32": {
            "sha256": (
                "9b17151107a518cbf5db58def922848b983da5555c61956c1242f9d21babc1b5"
            ),
            "sizeBytes": 4120,
        },
        "argmin_int64": {
            "sha256": (
                "71d7c03cf2fb66d82fb944e302d49b0542c0cfe033ac32b4522d1f73c8934a15"
            ),
            "sizeBytes": 5068,
        },
        "argmax_int64": {
            "sha256": (
                "f75a4b6ae0fdd783d204638012314e00d85eb03bec441d192fbe340b227babca"
            ),
            "sizeBytes": 5086,
        },
        "argmin_float16": {
            "sha256": (
                "623a6bbb83613892520a68ce7d33b3793f8c34e3121c9a4b16a8ad6417867103"
            ),
            "sizeBytes": 3980,
        },
        "argmax_float16": {
            "sha256": (
                "94c4a3d37c10f093fd1556b8e4c2747d0266cb7aee281455182a6db58fc9c03e"
            ),
            "sizeBytes": 3982,
        },
        "argmin_float32": {
            "sha256": (
                "955ededf34bbf85d26c24b75da7d9a8f2d394eb359af0898a81c379e5ae2205e"
            ),
            "sizeBytes": 4000,
        },
        "argmax_float32": {
            "sha256": (
                "1a9f08f4a607e9f1f45ee95da2d08fd753fac967073f081f6c3bb2d2d3c39d4c"
            ),
            "sizeBytes": 4002,
        },
        "argmin_bfloat16": {
            "sha256": (
                "02197cbd75af21ec86bf9151c7caba697168f0662bea23f3ebe6f219eb8f56e0"
            ),
            "sizeBytes": 4184,
        },
        "argmax_bfloat16": {
            "sha256": (
                "d77f0d264519f6b9868dfe6741013878368044ab3cf10e2a8925730fc2836ae2"
            ),
            "sizeBytes": 4194,
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
        "metalUnitCount": 42,
        "entryCount": 17478,
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
        "completeCorpusEntryCount": 17478,
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


def _metal_library(source, output, root):
    air = output.with_suffix(".air")
    _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
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
    library = _metal_library(root / SOURCE, work / "upstream.metallib", root)
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


def _runtime_executor(target):
    if target == "directx":
        adapter = DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
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
    executor = _runtime_executor(target)
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
