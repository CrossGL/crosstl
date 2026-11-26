"""CrossGL's workgroup collectives become helpers that combine the invocations' values through a shared array.

``crosstl/translator/workgroup_operations.py`` defines the collectives and ``crosstl/translator/semantics/
workgroup_collectives.py`` lowers them: each helper waits at a workgroup barrier, stores the caller's value at its
local invocation index, waits again and combines the stored values in index order.
``tests/conformance/cases/x_workgroup_collectives`` executes the helpers; these tests pin the rewriting, the
workgroup size it combines over, where the helpers go and the refusals.
"""

from __future__ import annotations

import re

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.host_workgroup_size import workgroup_size_dependence
from crosstl.translator.refusals import RegistryRefusal
from crosstl.translator.semantics import workgroup_collectives as collectives
from crosstl.translator.semantics.builtin_function_names import BUILTIN_FUNCTION_NAMES
from crosstl.translator.semantics.hub_types import BOOL, FLOAT32, TypeScope
from crosstl.translator.semantics.workgroup_collectives import (
    PREFIX,
    WorkgroupCollectiveError,
    lower_workgroup_collectives,
    register_native_workgroup_collectives,
)
from crosstl.translator.workgroup_operations import WORKGROUP_COLLECTIVES

LOWERED_TARGETS = (
    "opengl",
    "vulkan",
    "directx",
    "metal",
    "cuda",
    "hip",
    "slang",
    "wgsl",
    "mojo",
    "rust",
)
_EVERY_COLLECTIVE = """
float x = data[i];
int n = ints[i];
uint u = uint(n);
data[i] = workgroupAdd(x) + workgroupMul(x) + workgroupMin(x) + workgroupMax(x)
    + workgroupInclusiveAdd(x) + workgroupInclusiveMul(x) + workgroupInclusiveMin(x) + workgroupInclusiveMax(x)
    + workgroupExclusiveAdd(x) + workgroupExclusiveMul(x) + workgroupExclusiveMin(x) + workgroupExclusiveMax(x)
    + workgroupBroadcast(x, 3u);
ints[i] = workgroupAnd(n) + workgroupOr(n) + workgroupXor(n) + workgroupInclusiveAnd(n) + workgroupInclusiveOr(n)
    + workgroupInclusiveXor(n) + workgroupExclusiveAnd(n) + workgroupExclusiveOr(n) + workgroupExclusiveXor(n)
    + int(workgroupAdd(u)) + (workgroupAll(n > 0) ? 1 : 0) + (workgroupAny(n > 5) ? 2 : 0)
    + (workgroupXor(n > 1) ? 4 : 0);
"""


def _program(
    body,
    size="layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;",
    prelude="",
):
    return f"""
shader Collectives {{
    {prelude}
    compute {{
        {size}
        buffer float* data;
        buffer int* ints;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "collectives.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _lowered(source, target="opengl"):
    program = parse(source)
    count = lower_workgroup_collectives(program, target)
    return program, count


def _stage(program):
    (stage,) = program.stages.values()
    return stage


def _scratch(declarations):
    return {
        v.name: v.var_type.size.value
        for v in declarations
        if getattr(v, "name", "").startswith(PREFIX)
    }


def _helper(functions, operation):
    (helper,) = (f for f in functions if f.name.startswith(f"{PREFIX}{operation}_"))
    return helper


def test_every_collective_is_a_crossgl_operation():
    assert set(WORKGROUP_COLLECTIVES) <= BUILTIN_FUNCTION_NAMES
    assert len(WORKGROUP_COLLECTIVES) == 24


def test_crossgl_output_keeps_the_calls(tmp_path):
    code = _translate(tmp_path, _program(_EVERY_COLLECTIVE), "cgl")

    assert PREFIX not in code
    for operation in WORKGROUP_COLLECTIVES:
        assert re.search(rf"\b{operation}\(", code), operation


@pytest.mark.parametrize("backend", LOWERED_TARGETS)
def test_every_target_calls_the_helpers(tmp_path, backend):
    code = _translate(tmp_path, _program(_EVERY_COLLECTIVE), backend)

    for operation in WORKGROUP_COLLECTIVES:
        assert not re.search(rf"\b{operation}\(", code), operation
    if backend == "vulkan":
        # SPIR-V names no functions; every helper waits at two workgroup barriers.
        assert len(re.findall(r"^\s*OpControlBarrier\b", code, re.M)) >= 2 * 24
        return
    for operation in WORKGROUP_COLLECTIVES:
        assert re.search(rf"\b{PREFIX}{operation}_\w+\(", code), operation
    # One array per stored type: float, int and uint (which also carries the bools).
    arrays = set(re.findall(rf"\b{PREFIX}(float|int|uint)\b", code))
    assert arrays == {"float", "int", "uint"}


def _glsl_helper(code, name):
    """The GLSL text of helper ``name``'s definition."""
    start = re.search(rf"^\w+ {name}\([^;]*\{{", code, re.M).start()
    return code[start : code.index("\n}\n", start)]


def test_helpers_store_wait_and_combine_in_index_order(tmp_path):
    source = _program(
        "data[i] = workgroupAdd(data[i]) + workgroupInclusiveMin(data[i]);"
    )
    program, count = _lowered(source)
    stage = _stage(program)

    assert count == 2
    assert _scratch(stage.local_variables) == {PREFIX + "float": 64}
    (declaration,) = (v for v in stage.local_variables if v.name == PREFIX + "float")
    assert declaration.qualifiers == ["shared"]
    code = _translate(tmp_path, source, "opengl")
    assert re.search(rf"^shared float {PREFIX}float\[64\];", code, re.M)
    for operation, bound in (
        ("workgroupAdd", "lane < 64u"),
        ("workgroupInclusiveMin", "lane <= index"),
    ):
        text = _glsl_helper(code, f"{PREFIX}{operation}_float")
        assert text.count("barrier();") == 2, operation
        store = text.index(f"{PREFIX}float[index] = value;")
        assert text.index("barrier();") < store < text.rindex("barrier();")
        assert text.index("uint index = gl_LocalInvocationIndex;") < store
        assert bound in text, operation


@pytest.mark.parametrize(
    ("operation", "identity"),
    [
        ("workgroupExclusiveAdd", "0"),
        ("workgroupExclusiveMul", "1"),
        ("workgroupExclusiveAnd", "-1"),
        ("workgroupExclusiveMin", "2147483647"),
        ("workgroupExclusiveMax", "-2147483647) - 1"),
    ],
)
def test_exclusive_scans_start_from_the_identity(tmp_path, operation, identity):
    code = _translate(tmp_path, _program(f"ints[i] = {operation}(ints[i]);"), "opengl")

    body = _glsl_helper(code, f"{PREFIX}{operation}_int")
    start = body.split("int result = ", 1)[1].split(";", 1)[0]
    assert start.strip("()") == identity, start
    assert "lane < index" in body


def test_broadcast_reads_what_the_source_invocation_stored(tmp_path):
    code = _translate(
        tmp_path, _program("data[i] = workgroupBroadcast(data[i], 7);"), "opengl"
    )

    body = _glsl_helper(code, f"{PREFIX}workgroupBroadcast_float")
    assert "int source" in body
    assert re.search(r"gl_LocalInvocationIndex == uint\(source\)", body)
    assert f"return {PREFIX}float[0u];" in body


@pytest.mark.parametrize(
    ("size", "prelude", "expected"),
    [
        ("layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;", "", 64),
        ("", "", 1),
        ("layout(local_size_x = N) in;", "const int N = 48;", 48),
    ],
)
def test_the_helpers_combine_over_the_declared_workgroup_size(
    tmp_path, size, prelude, expected
):
    source = _program("data[i] = workgroupMax(data[i]);", size, prelude)
    program, _ = _lowered(source)
    stage = _stage(program)

    assert _scratch(stage.local_variables) == {PREFIX + "float": expected}
    code = _translate(tmp_path, source, "opengl")
    text = _glsl_helper(code, f"{PREFIX}workgroupMax_float")
    # One invocation combines nothing but its own value, and the GLSL generator drops the empty loop.
    assert (f"lane < {expected}u" in text) == (expected > 1)


def test_numthreads_and_compute_function_entries_give_the_size():
    program, _ = _lowered(
        "shader P { compute { buffer float* d; void main() @numthreads(16, 2, 1) { d[0] = workgroupAdd(d[0]); } } }"
    )
    assert _scratch(_stage(program).local_variables) == {PREFIX + "float": 32}

    program, _ = _lowered(
        "shader P { buffer float* d; @compute @workgroup_size(16, 4) fn k() { d[0] = workgroupAdd(d[0]); } }"
    )
    assert _scratch(program.global_variables) == {PREFIX + "float": 64}
    assert any(f.name.startswith(PREFIX) for f in program.functions)


@pytest.mark.parametrize("backend", ["opengl", "metal", "mojo"])
def test_a_shader_scope_caller_takes_the_helpers_of_its_stage(tmp_path, backend):
    source = _program(
        "data[i] = twice(data[i]);",
        prelude="float twice(float v) { return workgroupAdd(v) * 2.0; }",
    )
    program, _ = _lowered(source)
    stage = _stage(program)

    assert _scratch(stage.local_variables) == {PREFIX + "float": 64}
    assert not _scratch(program.global_variables or [])
    assert any(f.name.startswith(PREFIX) for f in stage.local_functions)
    # Mojo has no global variables, so the array must stay in the stage.
    code = _translate(tmp_path, source, backend)
    assert re.search(rf"\b{PREFIX}workgroupAdd_float\(", code)


def test_nested_collectives_take_the_inner_result_type():
    program, count = _lowered(
        _program("ints[i] = workgroupAdd(workgroupExclusiveMax(ints[i]));")
    )

    assert count == 2
    names = sorted(
        f.name for f in _stage(program).local_functions if f.name.startswith(PREFIX)
    )
    assert names == [PREFIX + "workgroupAdd_int", PREFIX + "workgroupExclusiveMax_int"]


def test_type_inference_knows_the_collectives():
    program = parse(
        _program("data[i] = workgroupInclusiveAdd(1.0) + float(workgroupAll(true));")
    )
    calls = [
        node
        for node in program.walk()
        if type(node).__name__ == "FunctionCallNode"
        and getattr(node.function, "name", "").startswith("workgroup")
    ]
    scope = TypeScope(program)

    assert [scope.infer(node) for node in calls] == [FLOAT32, BOOL]


def test_collectives_make_a_kernel_depend_on_its_workgroup_size():
    program = parse(_program("data[i] = workgroupAdd(data[i]);"))

    assert workgroup_size_dependence(program, _stage(program)) == "calls workgroupAdd"


def test_a_function_the_program_defines_under_the_name_is_left_alone(tmp_path):
    source = _program(
        "data[i] = workgroupAdd(data[i]);",
        prelude="float workgroupAdd(float v) { return v + 1.0; }",
    )
    code = _translate(tmp_path, source, "opengl")

    assert PREFIX not in code
    assert re.search(r"\bworkgroupAdd\(", code)


def test_a_native_target_keeps_the_calls(monkeypatch):
    monkeypatch.setattr(collectives, "_NATIVE", dict(collectives._NATIVE))
    register_native_workgroup_collectives("native-test", WORKGROUP_COLLECTIVES)
    program, count = _lowered(
        _program("data[i] = workgroupAdd(data[i]);"), "native-test"
    )

    assert count == 0
    assert not _scratch(_stage(program).local_variables)
    assert any(
        getattr(node.function, "name", None) == "workgroupAdd"
        for node in program.walk()
        if type(node).__name__ == "FunctionCallNode"
    )
    with pytest.raises(RegistryRefusal, match="workgroupBarrier"):
        register_native_workgroup_collectives("native-test", ["workgroupBarrier"])


_REFUSED = {
    "fragment stage": (
        "shader P { struct FO { vec4 c; } fragment { FO main() { FO o; o.c = vec4(workgroupAdd(1.0)); return o; } } }",
        "a fragment stage has no workgroup",
    ),
    "specialization constant size": (
        _program("data[i] = workgroupAdd(data[i]);", "layout(local_size_x_id = 0) in;"),
        "not a specialization constant",
    ),
    "bool vector": (
        _program(
            "bvec2 b = bvec2(true, false); data[i] = workgroupAnd(b).x ? 1.0 : 0.0;"
        ),
        "workgroupAnd: the operand type",
    ),
    "float invocation index": (
        _program("data[i] = workgroupBroadcast(data[i], 1.0);"),
        "the invocation index is not an int or uint",
    ),
    "arity": (
        _program("data[i] = workgroupAdd(data[i], data[i]);"),
        "takes 1 operand, not 2",
    ),
    "bitwise float": (
        _program("data[i] = workgroupXor(data[i]);"),
        "no workgroup helper for a f32x1",
    ),
    "vote of a number": (
        _program("data[i] = workgroupAll(data[i]) ? 1.0 : 0.0;"),
        "no workgroup helper",
    ),
    "reserved name": (
        _program(
            "data[i] = workgroupAdd(data[i]);",
            prelude="float cgl_rule_wgc_x(float v) { return v; }",
        ),
        "reserved for the workgroup collective helpers",
    ),
    "entries of different sizes": (
        "shader P { buffer float* d; float f(float v) { return workgroupAdd(v); }"
        " @compute @workgroup_size(16) fn a() { d[0] = f(d[0]); }"
        " @compute @workgroup_size(32) fn b() { d[1] = f(d[1]); } }",
        "different workgroup sizes (16, 32)",
    ),
}


@pytest.mark.parametrize("case", sorted(_REFUSED))
@pytest.mark.parametrize("backend", ["opengl", "metal"])
def test_refusals(tmp_path, case, backend):
    source, message = _REFUSED[case]

    with pytest.raises(WorkgroupCollectiveError, match=re.escape(message)) as raised:
        _translate(tmp_path, source, backend)
    assert raised.value.code == "project.translate.workgroup-collective-unsupported"


def test_a_host_chosen_workgroup_size_is_refused():
    program = parse(_program("data[i] = workgroupAdd(data[i]);", ""))
    _stage(program).execution_config["host_workgroup_size"] = {
        "entry": "main",
        "reason": "calls workgroupAdd",
    }

    with pytest.raises(WorkgroupCollectiveError, match="the host chooses"):
        lower_workgroup_collectives(program, "opengl")
