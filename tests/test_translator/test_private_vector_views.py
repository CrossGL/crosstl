"""Private scalar pointer views retain the original vector-array storage."""

import os
import pickle
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.ast import AssignmentNode, StructNode, VariableNode
from crosstl.translator.codegen.pointer_reinterpret import PointerReinterpretationError
from crosstl.translator.codegen.private_vector_views import lower_private_vector_views
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_PRIVATE_VECTOR_VIEWS"
CASES = (
    "returned",
    "direct",
    "immediate",
    "alias",
    "offset",
    "capture",
    "loop",
    "readonly",
    "indexed",
    "nested",
)
KINDS = ("float", "uint", "int")
INPUTS = {
    "float": [0.0, 1.5, -2.25, 64.0, 65536.0],
    "uint": [0, 1, 31, 64, 65536],
    "int": [0, 1, -31, 64, -65536],
}


def _source(case, kind, width):
    method = f"""
    thread {kind}* elems() thread {{ return reinterpret_cast<thread {kind}*>(fragments); }}
    const thread {kind}* elems() const thread {{ return reinterpret_cast<const thread {kind}*>(fragments); }}
"""
    initial = ["inputs[tid]", f"{kind}(7)", f"{kind}(99)", f"{kind}(101)"]
    initial += [f"{kind}({200 + i})" for i in range(2 * width - 4)]
    setup = "\n".join(
        f"tile.fragments[{i}] = {kind}{width}({', '.join(initial[i * width:(i + 1) * width])});"
        for i in range(2)
    )
    storage = "Tile tile;"
    declarations = f"thread {kind}* values = tile.elems();"
    helper = ""
    body = f"values[1] += {kind}(3); values[2] = values[0] * {kind}(2);"
    if case == "direct":
        method = ""
        declarations = (
            f"thread {kind}* values = reinterpret_cast<thread {kind}*>(tile.fragments);"
        )
    elif case == "immediate":
        declarations = ""
        body = f"tile.elems()[1] += {kind}(3); tile.elems()[2] = tile.elems()[0] * {kind}(2);"
    elif case == "alias":
        declarations += f"thread {kind}* second = values;"
        body = f"second[1] += {kind}(3); values[2] = second[0] * {kind}(2);"
    elif case == "offset":
        declarations += f"thread {kind}* second = values + 1;"
        body = f"*second += {kind}(3); second[1] = values[0] * {kind}(2);"
    elif case == "capture":
        declarations = f"uint which = tid; thread {kind}* values = tile.elems() + (which & 1u); which ^= 1u;"
    elif case == "loop":
        body = f"for (int i = 0; i < {2 * width}; ++i) {{ values[i] += {kind}(1); }}"
    elif case == "readonly":
        helper = f"{kind} observe(const thread Tile& other) {{ const thread {kind}* observed = other.elems(); return observed[0] + observed[1]; }}"
        body = f"values[1] += {kind}(3); values[2] = observe(tile) * {kind}(2);"
    elif case == "indexed":
        storage = "Tile tiles[2];"
        setup = setup.replace("tile.", "tiles[0].") + setup.replace(
            "tile.", "tiles[1]."
        )
        declarations = f"uint which = tid; thread {kind}* values = reinterpret_cast<thread {kind}*>(tiles[which & 1u].fragments); which ^= 1u;"
    elif case == "nested":
        storage = "Envelope envelope;"
        setup = setup.replace("tile.", "envelope.tile.")
        declarations = declarations.replace("tile.", "envelope.tile.")
    stride = 2 * width + (case == "indexed")
    receiver = (
        "tiles[tid & 1u]"
        if case == "indexed"
        else "envelope.tile" if case == "nested" else "tile"
    )
    output = "\n".join(
        f"results[4 + {stride} * tid + {i}] = {receiver}.fragments[{i // width}][{i % width}];"
        for i in range(2 * width)
    )
    if case == "indexed":
        output += f"results[4 + {stride} * tid + {2 * width}] = tiles[which & 1u].fragments[0][1];"
    envelope = "struct Envelope { Tile tile; };" if case == "nested" else ""
    return f"""#include <metal_stdlib>
using namespace metal;
struct Tile {{ {kind}{width} fragments[2]; {method} }};
{envelope}
{helper}
kernel void views(const device {kind}* inputs [[buffer(0)]],
                  device {kind}* results [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {{
    {storage}
    {setup}
    {declarations}
    {body}
    {output}
}}
"""


def _request(root, target, case, kind, width):
    source, descriptor, package = _package(
        root,
        target,
        kind,
        (1, 1, 1),
        source=_source(case, kind, width),
        software_subgroups=False,
    )
    inputs = INPUTS[kind]
    guard = 123456
    output = [guard] * 4
    for index, value in enumerate(inputs):
        values = [value, 7, 99, 101] + [200 + i for i in range(2 * width - 4)]
        if case == "loop":
            values = [element + 1 for element in values]
        else:
            offset = index & 1 if case == "capture" else 0
            values[offset + 1] += 3
            values[offset + 2] = 2 * values[offset]
            if case == "readonly":
                values[2] += 2 * values[1]
        output.extend(values)
        if case == "indexed":
            output.append(7)
    output += [guard] * 4
    dtype = {"uint": "uint32", "int": "int32", "float": "float32"}[kind]

    def typed(values):
        return {"dtype": dtype, "shape": [len(values)], "values": values}

    expected = _bound_values(descriptor, {"results": typed(output)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "inputs": typed(inputs),
                "results": typed([guard] * 4 + [0] * (len(output) - 8) + [guard] * 4),
            },
        ),
        expected,
        {"workgroupCount": [len(inputs), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", CASES)
def test_private_vector_view_compiles(tmp_path, target, width, kind, case):
    _, request, _ = _request(tmp_path, target, case, kind, width)
    _compile(
        request.artifact_path.read_text(),
        target,
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra", "-fno-fast-math"),
    )


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", CASES)
def test_private_vector_view_executes(tmp_path, width, kind, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required private vector view execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case, kind, width)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="views",
        metal_compile_flags=("-Wall", "-Wextra", "-fno-fast-math"),
    )


def _canonical(body, width=2, element="float", getter=True):
    method = (
        """thread float* elems(inout thread Tile self) {
        return (thread float*)self.fragments;
    }"""
        if getter
        else ""
    )
    ast = parse(f"""shader Views {{
        struct Tile {{ vec{width}[2] fragments; }}
        {method}
        compute {{
            void main() {{
                Tile tile;
                thread {element}* values = (thread {element}*)tile.fragments;
                {body}
            }}
        }}
    }}""")
    assert any(
        isinstance(node, VariableNode) and node.name == "tile" for node in ast.walk()
    )
    return ast


def test_lowering_does_not_mutate_source_ast():
    ast = _canonical("values[1] = values[0] + 1.0;")
    assignment = next(node for node in ast.walk() if isinstance(node, AssignmentNode))
    assignment.annotations = {"source.contract": {"preserve": True}}
    original = pickle.dumps(ast)
    first = lower_private_vector_views(ast, "directx")
    second = lower_private_vector_views(ast, "directx")
    assert pickle.dumps(ast) == original
    assert pickle.dumps(first) == pickle.dumps(second)
    assert not any(function.name == "elems" for function in first.functions)
    lowered_assignment = next(
        node for node in first.walk() if isinstance(node, AssignmentNode)
    )
    assert lowered_assignment.annotations == assignment.annotations
    assert lowered_assignment.target is lowered_assignment.left
    assert lowered_assignment.value is lowered_assignment.right


@pytest.mark.parametrize(
    "body,reason",
    [
        ("values[4] = 1.0;", "index-range-unproven"),
        ("values[-1] = 1.0;", "index-range-unproven"),
        ("uint i; values[i] = 1.0;", "index-range-unproven"),
        ("int i = 0; values[i++] = 1.0;", "index-side-effects"),
        ("thread float* escaped = &values[0];", "view-escape"),
        ("unknown(values);", "view-call-escape"),
        ("values = values + 1;", "view-escape"),
        (
            "const thread float* readonly = values; readonly[0] = 1.0;",
            "write-through-readonly-view",
        ),
        ("{ Tile tile; values[0] = 1.0; }", "backing-shadowed"),
        (
            "for (int i = 0; i < 4; ++i) { i += 10; values[i] = 1.0; }",
            "index-range-unproven",
        ),
        (
            "for (int i = 0; i < 4; ++i) { unknown(i); values[i] = 1.0; }",
            "index-range-unproven",
        ),
        (
            "for (int i = 0; i < 4; ++i) { int i = 7; values[i] = 1.0; }",
            "index-range-unproven",
        ),
    ],
)
def test_private_vector_view_rejects_unproven_access(body, reason):
    with pytest.raises(PointerReinterpretationError) as caught:
        lower_private_vector_views(_canonical(body), "opengl")
    assert caught.value.reason == reason


@pytest.mark.parametrize(
    "width,element,reason",
    [
        (3, "float", "unproven-vector-array-layout"),
        (2, "uint", "incompatible-scalar-layout"),
        (2, "half", "incompatible-scalar-layout"),
    ],
)
def test_private_vector_view_rejects_unproven_layout(width, element, reason):
    with pytest.raises(PointerReinterpretationError) as caught:
        lower_private_vector_views(
            _canonical("", width, element, getter=False), "directx"
        )
    assert caught.value.reason == reason


@pytest.mark.parametrize(
    "body,reason",
    [
        (
            "const uint64_t index = 4294967296; values[index] = 1.0;",
            "index-range-unproven",
        ),
        (
            "uint index; thread float* selected = values + index;",
            "captured-index-unproven",
        ),
        (
            "int index = 0; thread float* selected = values + index++;",
            "offset-side-effects",
        ),
        (
            "const thread float* observed = values; thread float* mutable_view = observed;",
            "readonly-view-conversion",
        ),
        ("values[0] = elems(tile)[0]; unknown(elems(tile));", "view-call-escape"),
    ],
)
def test_private_vector_view_preserves_width_access_and_lifetime(body, reason):
    with pytest.raises(PointerReinterpretationError) as caught:
        lower_private_vector_views(_canonical(body), "directx")
    assert caught.value.reason == reason


@pytest.mark.parametrize(
    "qualifier,member,reason",
    [
        ("const", False, "readonly-view-conversion"),
        ("volatile", False, "backing-storage-unsupported"),
        ("threadgroup", False, "backing-storage-unsupported"),
        ("static", True, "backing-member-unsupported"),
        ("volatile", True, "backing-member-unsupported"),
        ("const", True, "readonly-view-conversion"),
    ],
)
def test_private_vector_view_retains_storage_qualifiers(qualifier, member, reason):
    ast = _canonical("values[0] = 1.0;", getter=False)
    binding = (
        next(node for node in ast.walk() if isinstance(node, StructNode)).members[0]
        if member
        else next(
            node
            for node in ast.walk()
            if isinstance(node, VariableNode) and node.name == "tile"
        )
    )
    binding.qualifiers = [qualifier]
    with pytest.raises(PointerReinterpretationError) as caught:
        lower_private_vector_views(ast, "opengl")
    assert caught.value.reason == reason


def test_private_vector_view_accepts_scalar_value_conversions():
    lowered = lower_private_vector_views(
        _canonical("values[1] = float(values[0]);"), "directx"
    )
    assert lowered is not None


@pytest.mark.parametrize(
    "body,reason",
    [
        ("unknown(((thread float*)tile.fragments)[0]);", "view-call-escape"),
        ("thread float* escaped = &((thread float*)tile.fragments)[0];", "view-escape"),
        ("volatile thread float* qualified = values;", "view-qualifiers-unsupported"),
    ],
)
def test_direct_cast_views_do_not_bypass_escape_checks(body, reason):
    with pytest.raises(PointerReinterpretationError) as caught:
        lower_private_vector_views(_canonical(body), "opengl")
    assert caught.value.reason == reason


def test_private_vector_view_native_gate():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert "test_private_vector_views.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step


def test_native_frontend_ast_is_left_to_the_existing_target_path():
    ast = object()
    assert lower_private_vector_views(ast, "directx") is ast
