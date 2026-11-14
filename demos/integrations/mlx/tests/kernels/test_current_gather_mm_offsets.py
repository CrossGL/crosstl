"""Translate and execute the pinned MLX ``gather_mm_offsets`` kernel natively."""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/gather_mm_offsets.metal"
SOURCE_SHA256 = "e2b3839ba16c384d65cad76f7426da38f7c11931f5e55bba75c6f70a5f7a98d4"
ENTRY = "gather_mm_offsets"
ENTRIES = (ENTRY,)
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GATHER_MM_OFFSETS"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/gather_mm_offsets.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# MLX dispatches exactly one thread per group; one-thread workgroups reproduce
# that grid without partial-workgroup threads writing past ``offsets``.
WORKGROUP_SIZE = (1, 1, 1)
GUARD = -0x2468ACE
GUARD_COUNT = 16
# Values past ``M`` must never be searched. Zero is below every group id, so a
# read past the bound would move ``lo`` and change the result.
DECOY = 0
DECOY_COUNT = 4


def _dense_indices():
    # Deterministic, sorted expert assignments with empty and repeated groups.
    state = 0x9E3779B9
    values = []
    for _ in range(1000):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        values.append((state >> 16) % 61)
    return sorted(values)


# name -> (indices[:M], group count). Group ids are the dispatched thread ids.
CASES = {
    "empty": ([], 8),
    "single": ([3], 7),
    "grouped": (
        [0, 0, 0, 1, 1, 4, 4, 4, 4, 5, 9, 9, 9, 9, 9, 9, 12, 13, 13]
        + [20, 21, 21, 21, 22, 25, 25, 26, 30, 30, 30, 31, 32, 33, 33, 34]
        + [35, 35],
        40,
    ),
    "dense": (_dense_indices(), 64),
    # Ordered as unsigned 32-bit values. Groups 8-11 visit the sign-bit tail,
    # where a signed comparison would move the search past it.
    "unsigned": (
        [0, 1, 1, 2, 5, 7, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE]
        + [0xFFFFFFFF],
        12,
    ),
    # Outside MLX's sorted-input contract, but deterministic: it checks the
    # exact midpoint and branch direction of the translated search.
    "unsorted": ([9, 2, 7, 7, 0, 15, 3, 3, 11, 1, 6, 14, 4], 16),
}


def _search(indices, group):
    """Mirror the upstream binary search, including its midpoint arithmetic."""
    low, high = 0, len(indices)
    while low < high:
        middle = (low + high) // 2
        if indices[middle] < group:
            low = middle + 1
        else:
            high = middle
    return low


def _case(name):
    indices, groups = CASES[name]
    return {
        "indices": list(indices),
        "storage": list(indices) + [DECOY] * DECOY_COUNT,
        "M": len(indices),
        "groups": groups,
        "expected": [_search(indices, group) for group in range(groups)],
    }


def _config(output, target):
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
workgroup_size = [{", ".join(str(item) for item in WORKGROUP_SIZE)}]
[project.entry_points]
"{SOURCE}" = "{ENTRY}"
"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_current_mlx_gather_mm_offsets_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-gather-mm-offsets-contract"
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
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    validation = contract["validationContract"]
    assert validation["workgroupSize"] == list(WORKGROUP_SIZE)
    assert validation["indexRangeAssertions"] == {target: [] for target in TARGETS}
    assert validation["cases"] == {
        name: {"M": len(indices), "groups": groups}
        for name, (indices, groups) in CASES.items()
    }
    assert validation["outputGuardCount"] == GUARD_COUNT
    assert validation["inputDecoyCount"] == DECOY_COUNT
    assert validation["observedInputCount"] == 2
    assert validation["compilerValidationRequiredInCi"] == list(TARGETS)
    assert validation["runtimeRequiredInCi"] == list(TARGETS)
    assert validation["upstreamMetallibParityTargets"] == ["metal"]
    assert contract["scope"] == {
        "coveredEntryCount": 1,
        "runtimeCoveredEntryCount": 1,
        "caseCount": len(CASES),
        "offsetCountByTarget": {
            target: sum(groups for _, groups in CASES.values()) for target in TARGETS
        },
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_gather_mm_offsets_cases_cover_search_edges():
    for name in CASES:
        case = _case(name)
        expected = case["expected"]
        assert len(expected) == case["groups"]
        assert all(0 <= value <= case["M"] for value in expected)
        if name != "unsorted":
            indices = case["indices"]
            assert indices == sorted(indices)
            # Sorted input reduces the search to the first index >= group.
            assert expected == [
                sum(value < group for value in indices)
                for group in range(case["groups"])
            ]
    assert _case("empty")["expected"] == [0] * 8
    grouped = _case("grouped")
    # Empty groups repeat the next start; groups past the last index end at M.
    assert grouped["expected"][2] == grouped["expected"][3] == 5
    assert grouped["expected"][-1] == grouped["M"]
    unsigned = _case("unsigned")
    assert unsigned["expected"][8:] == [6] * 4
    signed = [
        value - 2**32 if value >= 2**31 else value for value in unsigned["indices"]
    ]
    # A signed comparison would produce different starts for the same groups.
    assert [_search(signed, group) for group in range(12)] != unsigned["expected"]
    unsorted = _case("unsorted")
    assert unsorted["expected"] != [
        sum(value < group for value in unsorted["indices"]) for group in range(16)
    ]


@pytest.mark.parametrize("target", TARGETS)
def test_gather_mm_offsets_config_uses_project_workgroup(tmp_path, target):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert tuple(config.workgroup_size) == WORKGROUP_SIZE
    assert list(config.targets) == [target]
    assert [item.to_json() for item in config.index_range_assertions] == []


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native gather_mm_offsets gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
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
        timeout=30,
    )
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("gather-mm-offsets-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _upstream_metal(work, reference, case, name):
    runner, library = reference
    guarded = [GUARD] * (case["groups"] + GUARD_COUNT)
    payloads = [
        struct.pack(f"<{len(case['storage'])}I", *case["storage"]),
        struct.pack(f"<{len(guarded)}i", *guarded),
        struct.pack("<i", case["M"]),
    ]
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{name}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": [case["groups"], 1, 1],
                "workgroupSize": list(WORKGROUP_SIZE),
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, ENTRY, request, output], work, f"{name}-upstream")
    )
    raw = (output / "buffer-1.bin").read_bytes()
    for index in (0, 2):
        assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]
    return device, list(struct.unpack(f"<{len(raw) // 4}i", raw))


def _dispatch(descriptor, package, target, case):
    data = {
        "indices": {
            "dtype": "uint32",
            "shape": [len(case["storage"])],
            "values": case["storage"],
        },
        "offsets": {
            "dtype": "int32",
            "shape": [case["groups"] + GUARD_COUNT],
            "values": [GUARD] * (case["groups"] + GUARD_COUNT),
        },
        "M": {"dtype": "int32", "shape": [1], "values": [case["M"]]},
    }
    inputs, observed, output_name = {}, [], None
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        # DirectX constant members carry the sanitized entry name as a prefix.
        name = layout.get("memberName", binding["name"]).removeprefix(
            re.sub(r"_+", "_", ENTRY) + "_"
        )
        value = data[name]
        assert layout["elementType"] == value["dtype"], (name, layout)
        inputs[binding["name"]] = value
        if name == "offsets":
            output_name = binding["name"]
        else:
            observed.append(binding["name"])
    assert output_name is not None and len(inputs) == 3 and len(observed) == 2
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_name: dict(data["offsets"])},
        {
            "workgroupCount": [case["groups"], 1, 1],
            "workgroupSize": list(WORKGROUP_SIZE),
        },
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output_name


def test_current_mlx_gather_mm_offsets_native_parity(
    current_mlx, metal_reference, tmp_path
):
    root, target = current_mlx
    record = _contract()["artifacts"][target][ENTRY]
    with tempfile.TemporaryDirectory(
        prefix=".current-gather-mm-offsets-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            config = work / "crosstl.toml"
            config.write_text(_config(f"{work.name}/out", target), encoding="utf-8")
            translation = translate_project(
                load_project_config(root, config),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
            report = translation.to_json()
            translation.write_json(work / "report.json")
            validation = validate_project_report(work / "report.json")
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
                artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
            )
            execution = artifact.get("execution")
            if target == "metal":
                # Metal selects the threadgroup size when the kernel is dispatched.
                assert execution is None
            else:
                assert execution["workgroupSize"] == list(WORKGROUP_SIZE)
                assert execution["sourceEntryPoints"] == [ENTRY]
            generated = root / artifact["path"]
            text = generated.read_text(encoding="utf-8")
            assert "unsupported" not in text
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES:
                    case = _case(name)
                    request, output_name = _dispatch(descriptor, package, target, case)
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    (work / f"{name}-result.json").write_text(
                        json.dumps(
                            {
                                "status": result.status,
                                "outputs": result.outputs,
                                "details": result.details,
                            },
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    assert result.status == "ok"
                    output = result.outputs[output_name]
                    assert output["dtype"] == "int32"
                    expected = case["expected"] + [GUARD] * GUARD_COUNT
                    assert output["values"] == expected
                    observations = result.details["inputObservations"]
                    assert len(observations) == 2
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, values = _upstream_metal(
                            work, metal_reference, case, name
                        )
                        assert values == expected
                        upstream = {"device": device, "matched": True}
                    evidence.append(
                        {
                            "case": name,
                            "M": case["M"],
                            "groups": case["groups"],
                            "expected": case["expected"],
                            "outputGuardCount": GUARD_COUNT,
                            "inputDecoyCount": DECOY_COUNT,
                            "observedInputCount": 2,
                            "upstream": upstream,
                        }
                    )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": ENTRY,
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "indexRangeAssertions": [],
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_gather_mm_offsets_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow,
        "mlx-metal-porting",
        "Prove current MLX new-unit native parity",
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/new-units",
        "--junitxml=mlx-current-results/new-units.xml",
        "demos/integrations/mlx/tests/kernels/test_current_gather_mm_offsets.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_gather_mm_offsets.py",
    )
    upload = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX new-unit native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )
