"""Source guards establish representable indices without asserting buffer sizes."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import FunctionCallNode, IdentifierNode
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_GUARDED_RESOURCE_INDICES"
BODIES = {
    "bounded-if": (
        "if (index >= 0 && index < 4) { return cursor.row[index]; } return 7;"
    ),
    "signed-if": "if (index >= 0) { return cursor.row[index]; } return 7;",
    "reversed-if": (
        "if (0 <= index && 4 > index) { return cursor.row[index]; } return 7;"
    ),
    "mixed-if": (
        "if (index >= 0 && uint(index) < 4u) { return cursor.row[index]; } return 7;"
    ),
    "nested-if": (
        "if (index >= 0) { if (index < 4) { return cursor.row[index]; } } return 7;"
    ),
    "else": (
        "if (index < 0 || index >= 4) { return 7; } else { return cursor.row[index]; }"
    ),
    "bounded-loop": (
        "uint value = 0; for (int i = index; i >= 0 && i < 4; --i) { value += cursor.row[i]; } return value;"
    ),
    "signed-loop": (
        "uint value = 0; for (int i = index; i >= 0; --i) { value += cursor.row[i]; } return value;"
    ),
    "while": (
        "uint value = 0; int i = index; while (i >= 0 && i < 4) { value += cursor.row[i]; --i; } return value;"
    ),
    "wide-if": (
        "long i = index; if (i >= 0 && i < 4) { return cursor.row[i]; } return 7;"
    ),
    "wide-loop": (
        "uint value = 0; for (long i = index; i >= 0 && i < 4; --i) { value += cursor.row[i]; } return value;"
    ),
}
REJECTED = {
    "unguarded": "return cursor.row[index];",
    "unsigned-promotion": "if (index >= 0u) { return cursor.row[index]; } return 7;",
    "mutated": (
        "if (index >= 0 && index < 4) { index = -1; return cursor.row[index]; } return 7;"
    ),
    "call-mutation": (
        "if (index >= 0 && index < 4) { change(index); return cursor.row[index]; } return 7;"
    ),
    "condition-mutation": (
        "if (index++ >= 0 && index < 4) { return cursor.row[index]; } return 7;"
    ),
    "wide": "long i = index; if (i >= 0) { return cursor.row[i]; } return 7;",
    "overflow": (
        "if (index >= 0 && index < (2147483647 + 1)) { return cursor.row[long(index) + 1]; } return 7;"
    ),
    "wrapped": "if (uint(index) + 1u < 4u) { return cursor.row[index]; } return 7;",
    "do-while": (
        "uint value = 0; int i = index; do { value += cursor.row[i]; --i; } while (i >= 0 && i < 4); return value;"
    ),
    "loop-mutation": (
        "uint value = 0; for (int i = index; i >= 0 && i < 4; --i) { i = -1; value += cursor.row[i]; } return value;"
    ),
    "after-loop": (
        "int i = index; while (i >= 0 && i < 4) { --i; } return cursor.row[i];"
    ),
    "shadow": (
        "if (index >= 0 && index < 4) { long index = -1; return cursor.row[index]; } return 7;"
    ),
}


def _source(case):
    body = BODIES.get(case, REJECTED.get(case))
    helper = (
        "void change(thread int& value) { value = -1; }"
        if case == "call-mutation"
        else ""
    )
    return f"""#include <metal_stdlib>
using namespace metal;
struct Cursor {{ constant uint* row; }};
{helper}
uint load_value(Cursor cursor, int index) {{ {body} }}
kernel void guarded(constant uint* values [[buffer(0)]],
                    constant int* indices [[buffer(1)]],
                    device uint* results [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {{
    Cursor cursor{{values}};
    results[4u + (tid & 3u)] = load_value(cursor, indices[tid & 3u]);
}}
"""


def _request(root, target, case):
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=_source(case), software_subgroups=False
    )
    guard = 0x5A1B2C3D
    words = [0, 3, 19, 36] if "loop" in case or case == "while" else [7, 3, 11, 17]

    def value(words, dtype="uint32"):
        return {"dtype": dtype, "shape": [len(words)], "values": words}

    inputs = _bound_values(
        descriptor,
        {
            "values": value([3, 5, 11, 17]),
            "indices": value([-1, 0, 2, 3], "int32"),
            "results": value([guard] * 4 + [0] * 4 + [guard] * 4),
        },
    )
    expected = _bound_values(
        descriptor, {"results": value([guard] * 4 + words + [guard] * 4)}
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, expected


@pytest.mark.parametrize("case", BODIES)
@pytest.mark.parametrize("target", ["opengl", "directx", "metal"])
def test_guarded_resource_indices_compile(tmp_path, target, case):
    request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", BODIES)
def test_guarded_resource_indices_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required guarded index execution")
    target = {"linux": "opengl", "darwin": "metal", "win32": "directx"}[sys.platform]
    request, expected = _request(tmp_path, target, case)
    options = {}
    if target == "metal":
        options = {
            "original_source": _source(case),
            "original_entry": "guarded",
            "metal_compile_flags": ("-Wall", "-Wextra", "-Werror"),
        }
    _execute(request, expected, tmp_path, **options)


@pytest.mark.parametrize("case", REJECTED)
def test_guarded_resource_indices_reject_unproven_access(tmp_path, case):
    (tmp_path / "source.metal").write_text(_source(case))
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("source.metal",),
            targets=("opengl",),
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1, report
    assert any(
        d["code"] == "project.translate.opengl-index-type-unsupported"
        for d in report["diagnostics"]
    ), report
    assert not list((tmp_path / "out").rglob("*.glsl"))


@pytest.mark.parametrize(
    "source_type,condition,expected",
    [
        ("int", "index >= 0", (0, 2147483647)),
        ("uint", "index < 4", (0, 3)),
        ("int", "index >= 0u", None),
        ("int", "index >= 0 && index < 4u", (0, 3)),
        ("int", "index >= 0u && index < 4u", None),
        ("int", "index + 1 > 0", None),
        ("uint", "index + 1u > 0u", None),
        ("int64_t", "index >= 0 && index < 4u", (0, 3)),
        ("int", "index < (4294967295u + 1u)", None),
    ],
)
def test_guard_bounds_preserve_source_promotions(source_type, condition, expected):
    assert _observed_intervals(
        f"void check({source_type} index) {{ if ({condition}) {{ observe(index); }} }}"
    ) == [expected]


def _observed_intervals(source, caller_types=None):
    generator = GLSLCodeGen()
    generator.local_variable_source_types = caller_types or {}
    ast = parse("shader Guards { " + source + " }")
    function = generator.collect_functions(ast)[0]
    generator.annotate_glsl_control_flow_intervals(function)
    return [
        getattr(node.arguments[0], "_glsl_control_flow_intervals", {}).get("index")
        for node in generator.walk_ast(function.body)
        if isinstance(node, FunctionCallNode)
        and isinstance(node.function, IdentifierNode)
        and node.function.name == "observe"
    ]


@pytest.mark.parametrize(
    "body",
    [
        "int* alias = &index; if (index >= 0) { *alias = -1; observe(index); }",
        "int& alias = index; if (index >= 0) { alias = -1; observe(index); }",
    ],
)
def test_guard_bounds_do_not_revive_escaped_local_facts(body):
    assert _observed_intervals("void check(int index) { " + body + " }") == [None]


def test_guard_bounds_use_lexical_types_not_caller_types():
    assert _observed_intervals(
        "void check(int index) { if (index >= 0) { observe(index); } }",
        {"index": "uint64_t"},
    ) == [(0, 2147483647)]


def test_guarded_resource_indices_ci_contract():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_guarded_resource_indices.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    assert step.count("python tools/run_bounded_command.py") == 2
    assert step.count("--timeout-seconds 360") == 2
    assert "--junitxml=.mlx-portable-host/resource-values/results.xml" in step
