"""Returned void expressions retain their calls and argument evaluation."""

import os
import re
import shutil
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_VOID_RETURN_EFFECTS"
CASES = ("entry", "helper", "branch", "argument")


def _source(case):
    extra = ""
    body = "return store(results, tid, 7);"
    if case == "helper":
        extra = """void relay(device uint* results, uint i) {
            return store(results, i, 7);
        }"""
        body = "return relay(results, tid);"
    elif case == "branch":
        body = """if ((tid & 1) != 0) { return store(results, tid, 13); }
        return store(results, tid, 7);"""
    elif case == "argument":
        extra = """void observed(device uint* results, uint i, thread uint& position) {
            results[i + 3] = position;
        }"""
        body = "uint position = tid; return observed(results, position++, position);"
    return f"""#include <metal_stdlib>
using namespace metal;
void store(device uint* results, uint i, uint value) {{ results[i + 3] = value; }}
{extra}
kernel void returned_call(device uint* results [[buffer(0)]],
                          uint tid [[thread_position_in_grid]]) {{ {body} }}
"""


def _request(tmp_path, target, case):
    source, descriptor, package = _package(
        tmp_path,
        target,
        "uint",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
    )
    words = [
        tid + 1 if case == "argument" else (13 if case == "branch" and tid & 1 else 7)
        for tid in range(4)
    ]
    inputs = {
        "results": {"dtype": "uint32", "shape": [10], "values": [0xDEADBEEF] * 10}
    }
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [10],
                "values": [0xDEADBEEF] * 3 + words + [0xDEADBEEF] * 3,
            }
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", ("opengl", "metal", "directx"))
def test_void_return_call_compiles(tmp_path, target, case):
    tool = {
        "opengl": "glslangValidator",
        "metal": "xcrun",
        "directx": "dxc",
    }[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    if target == "opengl":
        entry = generated[generated.rindex("void main()") :]
        callee = {"helper": "relay", "argument": "observed"}.get(case, "store")
        assert re.search(rf"\b{callee}(?:_glsl_\w+)?\(", entry), entry
        assert "return;" in entry
    _, module = _compile(generated, target, tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", CASES)
def test_void_return_call_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required void-return execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="returned_call",
    )


def test_void_return_effects_are_required_on_each_native_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    for name in (
        "Validate general gather and empty arrays",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate indexed OpenGL gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "test_void_return_effects.py" in step
        assert "--timeout-seconds 1200" in step
        assert "if:" not in step and "continue-on-error" not in workflow
