"""Ordered for-loop updates through project packages and native execution."""

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_verification import RuntimeExecutionState
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_LOOP_UPDATE_RUNTIME"
CASES = {
    "prefix": ("++i, ++position", "result += position;", 4, 6, 4),
    "postfix": ("i++, position++", "result += position;", 4, 6, 4),
    "dependent": ("++i, position += i", "result += position;", 4, 10, 10),
    "reverse-dependent": ("position += i, ++i", "result += position;", 4, 4, 6),
    "compound": ("i += 1, position += 2", "result += position;", 4, 12, 8),
    "continue": (
        "++i, position += i",
        "if (i == 1) { continue; } result += position;",
        4,
        9,
        10,
    ),
    "break": (
        "++i, position += i",
        "if (i == 2) { break; } result += position;",
        4,
        1,
        3,
    ),
    "single": ("++i", "result += i;", 4, 6, 0),
    "empty": ("", "result += position; ++i; position += i;", 4, 10, 10),
    "shadow": (
        "++i, ++position",
        "{ int position = i + 7; result += position; }",
        4,
        34,
        4,
    ),
    "zero-trip": ("++i, ++position", "result += position;", 0, 0, 0),
}


def _source(case):
    update, body, _, _, _ = CASES[case]
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void loop_updates(const device int* limits [[buffer(0)]],
                         device int* results [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {{
    int i = 91;
    int position = 0;
    int result = 0;
    for (int i = 0; i < limits[tid]; {update}) {{
        {body}
    }}
    results[3 * tid + 4] = result;
    results[3 * tid + 5] = position;
    results[3 * tid + 6] = i;
}}
"""


def _values(case):
    _, _, limit, result, position = CASES[case]
    expected = [-123456] * 4 + [result, position, 91] * 3 + [-123456] * 4
    return (
        {
            "limits": {"dtype": "int32", "shape": [3], "values": [limit] * 3},
            "results": {
                "dtype": "int32",
                "shape": [len(expected)],
                "values": [-123456] * len(expected),
            },
        },
        {"results": {"dtype": "int32", "shape": [len(expected)], "values": expected}},
    )


def _request(root, target, case):
    source, descriptor, package = _package(
        root, target, "int", (1, 1, 1), source=_source(case), software_subgroups=False
    )
    inputs, outputs = _values(case)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _bound_values(descriptor, outputs),
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("case", CASES)
def test_ordered_loop_updates_translate_and_compile(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    assert "UnaryOpNode(" not in generated and "AssignmentNode(" not in generated
    assert "is_postfix=" not in generated
    if "," in CASES[case][0]:
        header = next(line for line in generated.splitlines() if "for (" in line)
        assert "," in header.split(";")[2]
    _compile(generated, target, tmp_path)


def _execute(
    request,
    expected,
    work,
    *,
    original_source=None,
    original_entry=None,
    metal_compile_flags=(),
    validate=None,
):
    target = request.artifact["target"]
    executor = _executor(target)
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    records = {}
    try:
        available = executor.is_available(request)
        assert available.available, available
        validation = work / "validation"
        validation.mkdir()
        if validate is None:
            _, module = _compile(request.artifact_path.read_text(), target, validation)
        else:
            module = validate(request.artifact_path, validation, target)
        assert module.is_file() and module.stat().st_size
        native = executor.runtime_adapter.prepare_buffers(state)
        retained = work / native.module_path.name
        retained.write_bytes(native.module_path.read_bytes())
        actual = executor.runtime_adapter.dispatch(state, native)
        records["generated"] = {
            "outputs": actual,
            "details": state.details,
            "request": native.to_json(),
            "adapterSteps": [step.to_json() for step in state.adapter_steps],
            "moduleFile": retained.name,
            "moduleKind": "source" if target == "opengl" else "binary",
            "moduleSha256": hashlib.sha256(retained.read_bytes()).hexdigest(),
            "validationModuleFile": str(module.relative_to(work)),
            "validationModuleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        }
        if target == "metal" and original_source is not None:
            original = work / "original"
            original.mkdir()
            source, library = _compile(
                original_source,
                "metal",
                original,
                metal_compile_flags=("-fno-fast-math", *metal_compile_flags),
            )
            control = replace(
                native,
                artifact_path=source,
                module_path=library,
                entry_point=original_entry,
                loaded_artifact=None,
            )
            outputs = executor.runtime_adapter.runtime.dispatch(None, state, control)
            records["originalMetal"] = {
                "outputs": outputs,
                "compileFlags": ["-fno-fast-math", *metal_compile_flags],
                "moduleFile": str(library.relative_to(work)),
                "sourceFile": str(source.relative_to(work)),
                "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
        (work / "evidence.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "expected": expected,
                    "records": records,
                    "fixture": request.fixture.to_json(),
                    "artifactFile": str(request.artifact_path),
                    "artifactSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        assert actual == expected
        if "originalMetal" in records:
            assert records["originalMetal"]["outputs"] == expected
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


@pytest.mark.parametrize("case", CASES)
def test_ordered_loop_updates_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required loop-update execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="loop_updates",
    )


def test_loop_updates_native_gate_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/mlx-portable-host.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(workflow, "Validate ordered loop updates")
    assert "test_loop_updates.py" in step and f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds" in step and "--junitxml" in step and "-n auto" in step
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert (
            "tests/test_translator/test_loop_updates.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
