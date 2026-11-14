"""Helper parameters inherit checked metadata bounds from every call.

A collective loop in a helper, ``for (j = 0; j < limit; j += 32)``, needs a bound
on ``limit`` before the software-subgroup lowerings can prove its terminating
update. When every reachable call passes a bounded entry input or an integer
literal of the parameter's own type, and the helper never rebinds the parameter,
the hull of those bounds holds inside the helper. An unbounded call, a mutated or
shadowed parameter, or a changed integer type leaves the loop unproven.
"""

import os
import sys

import pytest

from crosstl.project import translate_project
from crosstl.translator.ast import (
    BlockNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    ParameterNode,
    PrimitiveType,
    VariableNode,
)
from crosstl.translator.codegen.metadata_call_bounds import (
    call_parameter_bounds,
    inherited_metadata_bindings,
    join_parameter_bounds,
)
from tests.test_translator.test_metadata_loop_bounds import (
    SOURCE,
    TARGETS,
    _loop_config,
    _request,
)
from tests.test_translator.test_project_metadata_entry_scope import _packages

HELPER = """size_t collect_limit(size_t limit) {
    return limit;
}

uint collect(size_t limit, uint lane) {
    uint result = 0;
    BODY
    for (uint j = 0; j < limit; j += 32) {
        result += simd_prefix_inclusive_sum(lane + j);
    }
    return result;
}

"""


def _source(call="collect(axis_size, lane)", body="", prelude=""):
    start = SOURCE.index("    uint result = 0;")
    source = (
        SOURCE[:start]
        + prelude
        + f"    output[group.x * 64 + lane + 4] = {call};\n}}\n"
    )
    return source.replace("kernel void", HELPER.replace("BODY", body) + "kernel void")


# Each collective call is its own statement, as the OpenGL lowering requires.
WIDENED = "    uint widened = collect(64, lane);\n    uint bounded = collect(axis_size, lane);\n"
UNBOUNDED = (
    "    uint other = collect(size_t(lane), lane);\n"
    "    uint bounded = collect(axis_size, lane);\n"
)


def _translate(root, target, source, **kwargs):
    return _packages(_loop_config(root, target, source=source, **kwargs))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "call, prelude",
    [
        ("collect(axis_size, lane)", ""),
        # A literal call widens the bound to cover it.
        ("bounded + widened - widened", WIDENED),
    ],
)
def test_helper_loop_inherits_the_entry_bound(tmp_path, target, call, prelude):
    source = _source(call, prelude=prelude)
    _, package, (descriptor,) = _translate(tmp_path, target, source)
    artifact = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert any("for (" in line and "limit" in line for line in artifact.splitlines())
    for count in (0, 1, 31, 32, 33, 64, 65535):
        request, _ = _request(descriptor, package, count)
        assert not request.execution_plan.diagnostics


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "call, body, prelude, conditions",
    [
        # No call proves a bound.
        ("collect(axis_size, lane)", "", "", ()),
        # A computed argument is not a bounded input.
        ("collect(axis_size + 1, lane)", "", "", None),
        ("collect(collect_limit(axis_size), lane)", "", "", None),
        # One unbounded call leaves the parameter unbounded.
        ("bounded ^ other", "", UNBOUNDED, None),
        # The helper rebinds or shadows its parameter.
        ("collect(axis_size, lane)", "limit += 1;", "", None),
        (
            "collect(axis_size, lane)",
            "{ size_t limit = 7; result += uint(limit); }",
            "",
            None,
        ),
    ],
)
def test_unproven_helper_bounds_stay_rejected(
    tmp_path, target, call, body, prelude, conditions
):
    kwargs = {} if conditions is None else {"conditions": conditions}
    source = _source(call, body, prelude)
    config = _loop_config(tmp_path, target, source=source, **kwargs)
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert any(
        item.get("details", {}).get("executionSpecialization", {}).get("reason")
        in {"potentially-divergent-control-flow", "helper-call-not-uniform"}
        and "top-level" not in item["message"]
        for item in payload["diagnostics"]
    ), payload["diagnostics"]


def _parameter(name, type_name, qualifiers=()):
    return ParameterNode(name, PrimitiveType(type_name), qualifiers=list(qualifiers))


def _literal(value):
    return LiteralNode(value, PrimitiveType("int"))


def test_call_bounds_require_the_parameter_type_and_a_value_parameter():
    bounds = {"n": {"dtype": "int32", "minimum": 0, "maximum": 9}}
    parameters = [
        _parameter("a", "int"),
        _parameter("b", "uint"),
        _parameter("c", "int", ("inout",)),
        _parameter("d", "short"),
        _parameter("e", "float"),
    ]
    arguments = [IdentifierNode("n")] * 3 + [_literal(70000), _literal(1)]
    proven = call_parameter_bounds(parameters, arguments, bounds, lambda t: t.name)
    assert proven == {0: {"dtype": "int32", "minimum": 0, "maximum": 9}}
    literal = call_parameter_bounds(
        [_parameter("d", "short")], [_literal(-3)], {}, lambda t: t.name
    )
    assert literal == {0: {"dtype": "int16", "minimum": -3, "maximum": -3}}


def test_joined_bounds_keep_only_what_every_call_proves():
    first = {
        0: {"dtype": "int32", "minimum": 0, "maximum": 9},
        1: {"dtype": "int32", "minimum": 4, "maximum": 4},
    }
    second = {0: {"dtype": "int32", "minimum": -2, "maximum": 3}}
    assert join_parameter_bounds(None, first) == first
    assert join_parameter_bounds(first, second) == {
        0: {"dtype": "int32", "minimum": -2, "maximum": 9}
    }
    assert join_parameter_bounds(first, {}) == {}


def test_inherited_bounds_skip_rebound_parameters():
    function = FunctionNode(
        "helper",
        PrimitiveType("void"),
        [_parameter("a", "int"), _parameter("b", "int"), _parameter("c", "int")],
        BlockNode([VariableNode("b", PrimitiveType("int"))]),
    )
    record = {"dtype": "int32", "minimum": 0, "maximum": 1}
    inherited = {0: record, 1: record, 2: record}
    assert inherited_metadata_bindings(
        function, inherited, {"c"}, lambda node: node.walk() if node else ()
    ) == {"a": record}


def test_helper_loop_bound_executes_natively(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    _, package, (descriptor,) = _translate(tmp_path, target, _source())
    for count in (0, 33, 95):
        request, outputs = _request(descriptor, package, count)
        work = tmp_path / f"native-{count}"
        work.mkdir()
        _execute(request, outputs, work)


def test_helper_bounds_are_required_on_all_native_platforms():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert 'CROSTL_REQUIRE_METADATA_PRECONDITIONS: "1"' in step
    assert "test_metadata_call_bounds.py" in step
    assert "if:" not in step and "continue-on-error" not in step
