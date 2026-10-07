"""Execute unchanged pinned gather-front kernels through native packages."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision
from tests.ci_helpers import assert_paths_covered
from tests.runtime_helpers import _prepare_native_package, _validate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_float_storage_encoding import WORDS
from tests.test_translator.test_loop_updates import _execute

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GATHER"
HEADER = "mlx/backend/metal/kernels/indexing/gather_front.h"
GUARD = 0x42F60000


def _source(width):
    entry = f"gather_frontfloat32_int32_int_{width}"
    return (
        entry,
        f"""#include "mlx/backend/metal/kernels/utils.h"
#include "{HEADER}"
template [[host_name("{entry}")]] [[kernel]]
decltype(gather_front<float, int, int, {width}>) gather_front<float, int, int, {width}>;
""",
    )


def _workload(stride, width):
    indices = [-4, -1, 0, 2, 2, 1]
    source = [WORDS[index % len(WORDS)] for index in range(4 * stride)] + [GUARD] * 16
    expected = []
    for index in indices:
        row = index if index >= 0 else index + 4
        for column in range(stride):
            expected.append(source[row * stride + column])
    expected += [GUARD] * 16

    def typed(dtype, values):
        return {
            "dtype": dtype,
            "shape": [len(values)],
            "values": values,
            **({"encoding": FLOAT32_BITS} if dtype == "float32" else {}),
        }

    return {
        "stride": stride,
        "width": width,
        "indices": indices,
        "grid": {
            "workgroupCount": [max(1, (stride + width - 1) // width), len(indices), 1],
            "workgroupSize": [1, 1, 1],
        },
        "inputs": {
            "src": typed("float32", source),
            "indices": typed("int32", indices),
            "out_": typed("float32", [GUARD] * len(expected)),
            "stride": typed("int64", [stride]),
            "size": typed("int32", [4]),
        },
        "outputs": {"out_": typed("float32", expected)},
    }


def _bound_inputs(descriptor, entry, inputs):
    bound, matched = {}, set()
    prefix = entry.rstrip("_") + "_"
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        layout = binding["scalarLayout"]
        member = layout.get("memberName", binding["name"])
        name = member[len(prefix) :] if member.startswith(prefix) else member
        assert name in inputs and name not in matched, (name, binding)
        assert binding["name"] not in bound
        matched.add(name)
        value = inputs[name]
        if value["dtype"] == "bool":
            assert all(type(item) is bool for item in value["values"])
            value = {
                **value,
                "dtype": "uint32",
                "values": [int(item) for item in value["values"]],
            }
        assert value["dtype"] == layout["elementType"], (name, layout)
        bound[binding["name"]] = value
    assert matched == set(inputs)
    return bound


@pytest.mark.parametrize("stride", [0, 1, 3, 9])
@pytest.mark.parametrize("width", [1, 4, 8])
def test_gather_workloads_cover_negative_duplicate_and_tail_indices(stride, width):
    workload = _workload(stride, width)
    source = workload["inputs"]["src"]["values"]
    expected = workload["outputs"]["out_"]["values"]
    assert expected[-16:] == [GUARD] * 16
    assert len(expected) == 6 * stride + 16
    assert expected[:stride] == source[:stride]
    assert expected[stride : 2 * stride] == source[3 * stride : 4 * stride]
    assert expected[3 * stride : 4 * stride] == expected[4 * stride : 5 * stride]
    assert workload["grid"]["workgroupCount"][0] * width >= stride


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("width", (1, 4, 8))
def test_gather_inputs_follow_reflected_constant_names(target, width):
    entry, _ = _source(width)
    inputs = _workload(3, width)["inputs"]
    bindings = []
    for name, value in inputs.items():
        constant = name in {"stride", "size"}
        member = f"{entry}_{name}" if target == "directx" and constant else name
        suffix = "Constants" if target == "directx" else "Args"
        binding_name = (
            f"{entry}_{name}_{suffix}"
            if target != "metal" and constant
            else f"{name}Buffer" if target == "opengl" else name
        )
        bindings.append(
            {
                "name": binding_name,
                "scalarLayout": {"memberName": member, "elementType": value["dtype"]},
            }
        )
    descriptor = {"bindings": bindings}
    bound = _bound_inputs(descriptor, entry, inputs)
    assert list(bound) == [binding["name"] for binding in bindings]
    assert list(bound.values()) == list(inputs.values())
    for missing in ("stride", "size"):
        with pytest.raises(AssertionError):
            _bound_inputs(
                descriptor, entry, {k: v for k, v in inputs.items() if k != missing}
            )
    with pytest.raises(AssertionError):
        _bound_inputs({"bindings": [*bindings, bindings[-1]]}, entry, inputs)


def _verify_source(root):
    require_revision(root)
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
    return hashlib.sha256((root / HEADER).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def gather_source():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native gather gate")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    assert target in {"metal", "opengl", "directx"}
    before = _verify_source(root)
    yield root, target, before
    assert _verify_source(root) == before


def _package(root, target, tmp_path, width):
    entry, source = _source(width)
    (tmp_path / "gather_front.metal").write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix=".gather-proof-", dir=root) as directory:
        work = Path(directory)
        wrapper = work / "gather_front.metal"
        wrapper.write_text(source, encoding="utf-8")
        relative = wrapper.relative_to(root).as_posix()
        report = translate_project(
            ProjectConfig(
                root=root,
                source_roots=(work.name,),
                include_patterns=(relative,),
                include_dirs=(".",),
                targets=(target,),
                output_dir=f"{work.name}/out",
                entry_points={relative: (entry,)},
                workgroup_size=(1, 1, 1),
            ),
            format_output=False,
        )
        report.write_json(tmp_path / "report.json")
        assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
        descriptor, package = _prepare_native_package(report, tmp_path)
    return descriptor, package


@pytest.mark.parametrize("width", [1, 4, 8])
def test_pinned_gather_front_compiles(gather_source, tmp_path, width):
    root, target, _ = gather_source
    descriptor, package = _package(root, target, tmp_path, width)
    artifact = package / descriptor["artifact"]["packagePath"]
    _validate(artifact, tmp_path, target)


@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("stride", [0, 1, 3, 9])
def test_pinned_gather_front_native_parity(gather_source, tmp_path, width, stride):
    root, target, source_hash = gather_source
    entry, source = _source(width)
    workload = _workload(stride, width)
    (tmp_path / "workload.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "header": HEADER,
                "headerSha256": source_hash,
                "entry": entry,
                **workload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    descriptor, package = _package(root, target, tmp_path, width)
    expected = _bound_values(descriptor, workload["outputs"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_inputs(descriptor, entry, workload["inputs"]),
        expected,
        workload["grid"],
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=entry,
        metal_compile_flags=("-I", str(root)),
        validate=_validate,
    )


def test_gather_native_gate_is_required():
    import re

    import yaml

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(workflow, "Validate pinned gather loops")
    for required in (
        "test_current_gather.py",
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT",
        "CROSTL_MLX_CURRENT_TARGET",
        "--timeout-seconds",
        "--junitxml",
        "-n auto",
    ):
        assert required in step
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "demos/integrations/mlx/tests/kernels/test_current_gather.py",
        )
    job = yaml.safe_load(workflow)["jobs"]["portable-host"]
    deadlines = [
        int(value)
        for item in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", item.get("run", ""))
    ]
    assert sum(deadlines) + 1800 < job["timeout-minutes"] * 60 <= 360 * 60
