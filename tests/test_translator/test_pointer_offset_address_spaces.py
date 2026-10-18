"""Pointer offsets preserve storage provenance and execute the original call."""

import json
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.resource_aggregates import lower_resource_aggregates
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_POINTER_OFFSET_RUNTIME"
CASES = {
    "member": ("", "values + index.offset"),
    "commuted": ("", "index.offset + values"),
    "nested": ("", "(index.offset + values) + 1 - 1"),
    "subtraction": ("", "(values + 4) - (4 - index.offset)"),
    "alias": ("auto alias = values + index.offset;", "alias"),
    "member-pointer": ("View view{values};", "view.data + index.offset"),
    "commuted-member-pointer": ("View view{values};", "index.offset + view.data"),
    "arrow-offset": (
        "thread Index* index_pointer = &index;",
        "values + index_pointer->offset",
    ),
    "conditional-offset": (
        "",
        "values + (tid == 0 ? index.offset : values[tid + 5])",
    ),
    "conditional-pointer": (
        "",
        "(tid == 0 ? values : values + 1) + index.offset - (tid == 0 ? 0 : 1)",
    ),
    "address": ("", "&values[index.offset]"),
    "address-offset": ("", "&values[0] + index.offset"),
    "offset-side-effect": (
        "View view{values}; int offset = index.offset - 1;",
        "(++offset) + view.data",
    ),
}
FOREIGN_DIAGNOSTICS = {
    (
        "arrow-offset",
        "directx",
    ): "project.translate.directx-private-pointer-unsupported",
    ("arrow-offset", "opengl"): "project.translate.opengl-private-pointer-unsupported",
    (
        "conditional-pointer",
        "directx",
    ): "project.translate.directx-resource-pointer-parameter-unsupported",
    (
        "conditional-pointer",
        "opengl",
    ): "project.translate.opengl-storage-pointer-unsupported",
}


def source(case):
    declarations, expression = CASES[case]
    view = (
        "struct View { const constant int* data; };" if "View" in declarations else ""
    )
    effects = " + 100 * (offset - index.offset)" if case == "offset-side-effect" else ""
    return f"""#include <metal_stdlib>
using namespace metal;
struct Index {{ int offset; }};
{view}
int read_value(const constant int* values) {{ return values[0]; }}
kernel void pointer_offset(const constant int* values [[buffer(0)]],
                           device int* results [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {{
    Index index{{int(tid) + 1}};
    {declarations}
    int loaded = read_value({expression});
    results[tid + 1] = loaded + index.offset{effects};
}}
"""


def index_assertions(filename, target):
    return (
        (
            {
                "source": filename,
                "expression": "reference.offset + index",
                "minimum": 0,
                "maximum": 7,
            },
        )
        if target == "opengl"
        else ()
    )


@pytest.mark.parametrize("qualifier", ("buffer", "readonly buffer"))
def test_buffer_helper_parameters_preserve_device_storage(qualifier):
    generated = MetalCodeGen().generate(parse(f"""shader BufferHelper {{
        int read_value({qualifier} int* values) {{ return values[0]; }}
        compute {{
            void main(device int* values @buffer(0)) {{
                values[0] = read_value(values);
            }}
        }}
    }}"""))
    expected = "const device" if "readonly" in qualifier else "device"
    assert f"int read_value({expected} int* values)" in generated
    assert "values[0] = read_value(values);" in generated


def test_generated_resource_helper_indices_match_declared_width():
    ast = lower_resource_aggregates(parse("""shader ResourceIndices {
        struct View { device int* data; };
        compute {
            void main(device int* values @buffer(0)) {
                View view;
                view.data = values;
                view.data[0] = view.data[1];
                device int* alias = &view.data[2];
                int old = atomicAdd(view.data[3], 1);
                buffer_store(view.data, 4, old);
            }
        }
    }"""))
    functions = {
        node.name: node
        for node in ast.walk()
        if isinstance(node, FunctionNode) and node.name.startswith("crosstl_resource_")
    }
    checked = []
    for call in ast.walk():
        if not isinstance(call, FunctionCallNode):
            continue
        name = getattr(call.function, "name", None)
        function = functions.get(name)
        if function is None:
            continue
        for parameter, argument in zip(function.parameters, call.arguments):
            if parameter.name != "index":
                continue
            assert parameter.param_type.name == "int64_t"
            assert isinstance(argument, FunctionCallNode)
            assert argument.function.name == "int64_t"
            checked.append(name)
    assert len(checked) == 5


@pytest.mark.parametrize("space", ("constant", "device", "threadgroup", "thread"))
@pytest.mark.parametrize("commuted", (False, True))
@pytest.mark.parametrize("offset_kind", ("member", "array", "conditional"))
def test_pointer_arithmetic_uses_only_pointer_storage(space, commuted, offset_kind):
    generator = MetalCodeGen()
    generator.local_variable_types.update(
        values="int*", index="Index", offsets="int*", choose="bool"
    )
    generator.struct_member_types["Index"] = {"offset": "int"}
    generator.current_address_space_variables.update(
        values=space, index="thread", offsets="device"
    )
    member = MemberAccessNode(IdentifierNode("index"), "offset")
    loaded = ArrayAccessNode(IdentifierNode("offsets"), LiteralNode("0", "int"))
    offset = {
        "member": member,
        "array": loaded,
        "conditional": TernaryOpNode(IdentifierNode("choose"), member, loaded),
    }[offset_kind]
    pointer = IdentifierNode("values")
    left, right = (offset, pointer) if commuted else (pointer, offset)
    expression = BinaryOpNode(left, "+", right)
    assert generator.expression_result_type(expression) == "int*"
    assert generator.argument_address_space(expression) == space
    assert generator.argument_address_space_conflict(expression) is None
    address = UnaryOpNode("&", ArrayAccessNode(expression, LiteralNode("0", "int")))
    assert generator.argument_address_space(address) == space


@pytest.mark.parametrize("offset_on_left", (False, True))
def test_pointer_arithmetic_retains_actual_pointer_branch_conflicts(offset_on_left):
    generator = MetalCodeGen()
    generator.local_variable_types.update(a="int*", b="int*", offset="int")
    generator.current_address_space_variables.update(a="threadgroup", b="device")
    pointer = TernaryOpNode(
        IdentifierNode("choose"), IdentifierNode("a"), IdentifierNode("b")
    )
    offset = IdentifierNode("offset")
    left, right = (offset, pointer) if offset_on_left else (pointer, offset)
    expression = BinaryOpNode(left, "+", right)
    assert generator.argument_address_space(expression) is None
    assert generator.argument_address_space_conflict(expression) == (
        "threadgroup",
        "device",
        "a",
        "b",
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_project_preserves_pointer_offset_calls(tmp_path, target, scoped, case):
    (tmp_path / "offset.metal").write_text(source(case), encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("offset.metal",),
            targets=(target,),
            entry_points={"offset.metal": ("pointer_offset",)} if scoped else {},
            workgroup_size=(1, 1, 1),
            index_range_assertions=index_assertions("offset.metal", target),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    unsupported = FOREIGN_DIAGNOSTICS.get((case, target))
    if unsupported:
        assert report["summary"]["failedCount"] == 1
        assert [item["code"] for item in report["diagnostics"]] == [unsupported]
        assert not any(item["status"] == "translated" for item in report["artifacts"])
        return
    assert not report["diagnostics"], report["diagnostics"]
    (artifact,) = report["artifacts"]
    assert artifact["status"] == "translated"
    generated = (tmp_path / artifact["path"]).read_text()
    assert "unsupported Metal address-space call" not in generated
    assert generated.count("read_value") >= 2


@pytest.mark.parametrize("return_type", ("void", "int"))
@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("mixed", (False, True))
def test_project_rejects_incompatible_pointer_call_without_artifact(
    tmp_path, return_type, scoped, mixed
):
    result = "" if return_type == "void" else "return values[0];"
    argument = "tid == 0 ? scratch : values" if mixed else "scratch + 1"
    (tmp_path / "offset.cgl").write_text(
        f"""shader InvalidCall {{
        {return_type} read_value(device int* values) {{ {result} }}
        compute {{
            void main(device int* values @buffer(0), uint tid @gl_GlobalInvocationID) {{
                shared int scratch[4];
                read_value({argument});
            }}
        }}
    }}""",
        encoding="utf-8",
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("offset.cgl",),
            targets=("metal",),
            entry_points={"offset.cgl": ("main",)} if scoped else {},
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1, report
    assert all(item["status"] == "failed" for item in report["artifacts"])
    assert not list((tmp_path / "out").rglob("*.metal"))
    (diagnostic,) = report["diagnostics"]
    assert diagnostic["code"] == "project.translate.unsupported-feature"
    assert "requires device" in diagnostic["message"]
    assert ("mixes branches" in diagnostic["message"]) == mixed


def _request(root, target, case):
    original, descriptor, package = _package(
        root,
        target,
        "int",
        (1, 1, 1),
        source=source(case),
        software_subgroups=False,
        index_range_assertions=index_assertions("products.metal", target),
    )
    inputs = {
        "values": {
            "dtype": "int32",
            "shape": [8],
            "values": [-11, 7, 19, 31, 999, 1, 2, 3],
        },
        "results": {"dtype": "int32", "shape": [6], "values": [-12345] * 6},
    }
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "int32",
                "shape": [6],
                "values": [-12345, 8, 21, 34, -12345, -12345],
            },
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, expected


@pytest.mark.parametrize(
    "case,target",
    [
        (case, target)
        for case in CASES
        for target in ("metal", "directx", "opengl")
        if (case, target) not in FOREIGN_DIAGNOSTICS
    ],
)
def test_pointer_offsets_compile(tmp_path, target, case):
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    _, request, _ = _request(tmp_path, target, case)
    _, module = _compile(request.artifact_path.read_text(), target, tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in CASES
        if (
            case,
            {"darwin": "metal", "win32": "directx", "linux": "opengl"}.get(
                sys.platform
            ),
        )
        not in FOREIGN_DIAGNOSTICS
    ],
)
def test_pointer_offsets_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pointer-offset execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="pointer_offset",
    )


def test_pointer_offset_execution_is_required_on_each_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    for name in (
        "Validate general gather and empty arrays",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate indexed OpenGL gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "test_pointer_offset_address_spaces.py" in step
        timeout = 1800 if name == "Validate general gather and empty arrays" else 1200
        assert f"--timeout-seconds {timeout}" in step and "-n auto" in step
        assert "if:" not in step and "continue-on-error" not in workflow


POINTER_VIEW_CASES = ("read", "advance", "forward", "rebase", "shadow")


def _pointer_view_source(case):
    mutation = "rows[0] += 1;" if case != "read" else ""
    forward = (
        """
void forward(thread uint total[1], const device uint* rows[1]) {
    consume(total, rows);
}
"""
        if case == "forward"
        else ""
    )
    callee = "forward" if case == "forward" else "consume"
    rebase = "row += 1;" if case == "rebase" else ""
    shadow = (
        """
    {
        const device uint* row = values + get_offset(metadata + tid + 3, calls);
        consume(&total, &row);
        results[tid * 5 + 4] = total + row[0];
    }
"""
        if case == "shadow"
        else ""
    )
    return f"""#include <metal_stdlib>
using namespace metal;
uint get_offset(const constant uint* metadata, thread uint& calls) {{
    calls += 1;
    return metadata[0] & 3u;
}}
void consume(thread uint total[1], const device uint* rows[1]) {{
    uint first = rows[0][0];
    {mutation}
    total[0] = first + rows[0][0];
}}
{forward}
kernel void pointer_view(const device uint* values [[buffer(0)]],
                         const constant uint* metadata [[buffer(1)]],
                         device uint* results [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {{
    uint calls = 0;
    const device uint* row = values + get_offset(metadata + tid, calls);
    uint total = 0;
    {shadow}
    {rebase}
    {callee}(&total, &row);
    results[tid * 5 + 1] = total;
    results[tid * 5 + 2] = row[0];
    results[tid * 5 + 3] = calls;
}}
"""


@pytest.mark.parametrize("case", POINTER_VIEW_CASES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("scoped", (False, True))
def test_project_preserves_helper_computed_pointer_views(
    tmp_path, case, target, scoped
):
    (tmp_path / "view.metal").write_text(_pointer_view_source(case), encoding="utf-8")
    payload = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("view.metal",),
            targets=(target,),
            entry_points={"view.metal": ("pointer_view",)} if scoped else {},
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert not payload["diagnostics"]


@pytest.mark.parametrize("case", ("read", "advance", "forward"))
def test_helper_computed_pointer_view_rejects_wide_offset(tmp_path, case):
    source = _pointer_view_source(case).replace("uint get_offset", "ulong get_offset")
    source = source.replace("metadata[0] & 3u", "4294967296ul + metadata[0]")
    (tmp_path / "view.metal").write_text(source, encoding="utf-8")
    payload = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("view.metal",),
            targets=("opengl",),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in payload["diagnostics"]
    ), payload
    assert not list((tmp_path / "out").rglob("*.glsl"))


def _pointer_view_request(root, target, case):
    original, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_pointer_view_source(case),
        software_subgroups=False,
    )
    values = [17 + 13 * index for index in range(8)]
    metadata = [0, 1, 3, 2, 0, 1]
    guard = 0xDEADBEEF
    result = [guard] * 47
    expected_words = result.copy()
    for tid in range(3):
        base = metadata[tid] + (case == "rebase")
        advanced = base + (case != "read")
        expected_words[tid * 5 + 1] = values[base] + values[advanced]
        expected_words[tid * 5 + 2] = values[advanced]
        expected_words[tid * 5 + 3] = 2 if case == "shadow" else 1
        if case == "shadow":
            inner = metadata[tid + 3]
            expected_words[tid * 5 + 4] = values[inner] + 2 * values[inner + 1]
    inputs = _bound_values(
        descriptor,
        {
            name: {"dtype": "uint32", "shape": [len(words)], "values": words}
            for name, words in (
                ("values", values),
                ("metadata", metadata),
                ("results", result),
            )
        },
    )
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [len(result)],
                "values": expected_words,
            },
        },
    )
    observed = [name for name in inputs if name not in expected]
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
        observe_inputs=observed,
    )
    assert not request.execution_plan.diagnostics
    return original, request, expected


@pytest.mark.parametrize("case", POINTER_VIEW_CASES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_helper_computed_pointer_views_compile(tmp_path, case, target):
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    _, request, _ = _pointer_view_request(tmp_path, target, case)
    _, module = _compile(request.artifact_path.read_text(), target, tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", POINTER_VIEW_CASES)
def test_helper_computed_pointer_views_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pointer-offset execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _pointer_view_request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="pointer_view",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    for record in evidence["records"].values():
        observations = record["details"]["inputObservations"]
        assert len(observations) == 2
        assert all(item["unchanged"] for item in observations.values())


POINTER_RETURN_BODIES = {
    "final": "rows[0] += 1; return rows[0][0] + rows[1][0];",
    "effect": "return advance(rows, calls);",
    "forward": "rows[0] += 1; return advance(rows, calls);",
    "early": (
        """
        rows[0] += 1;
        if (mode == 0) { return rows[0][0]; }
        rows[1] += 1;
        return rows[1][0];
    """
    ),
    "void": (
        """
        rows[0] += 1;
        if (mode == 0) { return; }
        rows[1] += 1;
        calls += 1;
    """
    ),
    "void-expression": "return advance_void(rows, calls);",
    "switch": (
        """
        rows[0] += 1;
        switch (mode) {
            case 0: return rows[0][0];
            case 1: rows[1] += 1; return rows[1][0];
            default: break;
        }
        rows[0] += 1;
        return rows[0][0] + rows[1][0];
    """
    ),
    "loop": (
        """
        for (uint i = 0; i < 3; ++i) {
            rows[0] += 1;
            if (i == mode) { return rows[0][0]; }
        }
        return 999u;
    """
    ),
    "struct": "return Result{advance(rows, calls), 7u};",
    "shadow": (
        """
        uint crossglReturn = 7;
        rows[0] += 1;
        if (mode == 0) {
            uint crossglReturn = 11;
            return rows[0][0] + crossglReturn;
        }
        return rows[1][0] + crossglReturn;
    """
    ),
}


def _pointer_return_source(case):
    return_type = (
        "void" if case.startswith("void") else "Result" if case == "struct" else "uint"
    )
    invocation = "consume(rows, tid, calls)"
    statement = (
        f"{invocation}; uint total = 101;"
        if return_type == "void"
        else (
            f"Result result = {invocation}; uint total = result.value + result.marker;"
            if return_type == "Result"
            else f"uint total = {invocation};"
        )
    )
    return f"""#include <metal_stdlib>
using namespace metal;
struct Result {{ uint value; uint marker; }};
uint advance(const device uint* rows[2], thread uint& calls) {{
    rows[0] += 1;
    rows[1] += 2;
    calls += 1;
    return rows[0][0] + rows[1][0];
}}
void advance_void(const device uint* rows[2], thread uint& calls) {{
    rows[0] += 1;
    rows[1] += 2;
    calls += 1;
}}
{return_type} consume(const device uint* rows[2], uint mode, thread uint& calls) {{
    {POINTER_RETURN_BODIES[case]}
}}
uint readonly_value(const device uint* rows[2]) {{ return rows[1][0]; }}
kernel void pointer_return(const device uint* values [[buffer(0)]],
                           const constant uint* metadata [[buffer(1)]],
                           device uint* results [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {{
    uint base = metadata[tid] & 3u;
    const device uint* rows[2];
    for (int i = 0; i < 2; ++i) {{
        rows[i] = values + base + i * 2;
    }}
    uint calls = 0;
    {statement}
    results[tid * 6 + 1] = total;
    results[tid * 6 + 2] = rows[0][0];
    results[tid * 6 + 3] = readonly_value(rows);
    results[tid * 6 + 4] = calls;
}}
"""


def _pointer_return_request(root, target, case):
    original, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_pointer_return_source(case),
        software_subgroups=False,
    )
    values = [19, 31, 47, 67, 89, 113, 139, 167, 197, 229]
    metadata = [0, 1, 3]
    result = [0xDEADBEEF] * 50
    expected_words = result.copy()
    for tid, base in enumerate(metadata):
        first, second, calls = base + 1, base + 2, 0
        if case in {"effect", "forward", "struct", "void-expression"}:
            first += case == "forward"
            second += 2
            calls = 1
        elif case in {"early", "void", "switch"}:
            second += tid == 1 or (tid == 2 and case != "switch")
            first += tid == 2 and case == "switch"
            calls = int(tid != 0 and case == "void")
        elif case == "loop":
            first += tid
        total = values[first] + values[second]
        if case == "struct":
            total += 7
        elif case in {"early", "switch"} and tid < (3 if case == "early" else 2):
            total = values[first if tid == 0 else second]
        elif case == "loop":
            total = values[first]
        elif case.startswith("void"):
            total = 101
        elif case == "shadow":
            total = values[first] + 11 if tid == 0 else values[second] + 7
        expected_words[tid * 6 + 1 : tid * 6 + 5] = [
            total,
            values[first],
            values[second],
            calls,
        ]
    inputs = _bound_values(
        descriptor,
        {
            name: {"dtype": "uint32", "shape": [len(words)], "values": words}
            for name, words in (
                ("values", values),
                ("metadata", metadata),
                ("results", result),
            )
        },
    )
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [len(result)],
                "values": expected_words,
            },
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
        observe_inputs=[name for name in inputs if name not in expected],
    )
    assert not request.execution_plan.diagnostics
    return original, request, expected


@pytest.mark.parametrize("case", POINTER_RETURN_BODIES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("scoped", (False, True))
def test_project_preserves_pointer_array_return_writeback(
    tmp_path, case, target, scoped
):
    (tmp_path / "return.metal").write_text(
        _pointer_return_source(case), encoding="utf-8"
    )
    payload = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("return.metal",),
            targets=(target,),
            entry_points={"return.metal": ("pointer_return",)} if scoped else {},
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert not payload["diagnostics"]


@pytest.mark.parametrize("case", POINTER_RETURN_BODIES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_pointer_array_returns_compile(tmp_path, case, target):
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    _, request, _ = _pointer_return_request(tmp_path, target, case)
    _, module = _compile(request.artifact_path.read_text(), target, tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", POINTER_RETURN_BODIES)
def test_pointer_array_returns_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pointer-offset execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _pointer_return_request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="pointer_return",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    for record in evidence["records"].values():
        observations = record["details"]["inputObservations"]
        assert len(observations) == 2
        assert all(item["unchanged"] for item in observations.values())


# Offsetting a pointer keeps its element type: ``(p + n)[i]`` and
# ``*(p + n)`` read the element ``p`` points to whatever the offset's type.
# Exact in binary16 and bfloat16; 0.75 and -1.5 expose integer truncation.
TYPED_OFFSET_VALUES = (9.5, 0.75, -1.5, 2.25, 3.0, -4.5, 5.25, 6.0)
TYPED_OFFSET_READS = (
    ("(f + tid)[1]", 2),
    ("*(f + signed_tid)", 1),
    ("(h + tid)[1]", 2),
    ("(tid + h)[0]", 1),
    ("(b + signed_tid)[1]", 2),
    ("*(b + tid)", 1),
    ("(b + 3)[tid]", 4),
    ("(b - 1)[tid + 2u]", 2),
)
TYPED_OFFSET_THREADS = 3


def _typed_offset_source():
    width = len(TYPED_OFFSET_READS)
    stores = "\n    ".join(
        f"results[{width}u * tid + {slot + 1}u] = static_cast<float>({expression});"
        for slot, (expression, _) in enumerate(TYPED_OFFSET_READS)
    )
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void typed_offsets(const device float* floats [[buffer(0)]],
                          const device half* halves [[buffer(1)]],
                          const device bfloat* brains [[buffer(2)]],
                          device float* results [[buffer(3)]],
                          device bfloat* narrowed [[buffer(4)]],
                          uint tid [[thread_position_in_grid]]) {{
    int signed_tid = int(tid);
    auto f = floats + 1;
    auto h = halves + 1;
    auto b = brains + 1;
    {stores}
    (narrowed + tid)[1] = static_cast<bfloat>((f + tid)[1] * 2.0f);
}}
"""


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _typed_offset_request(root, target):
    from crosstl.project.runtime_value_encoding import FLOAT16_BITS, FLOAT32_BITS
    from tests.test_translator.test_bfloat_buffer_runtime import _storage

    original, descriptor, package = _package(
        root,
        target,
        "float",
        (1, 1, 1),
        source=_typed_offset_source(),
        software_subgroups=False,
    )
    marker = _float_bits(123.0)
    width = len(TYPED_OFFSET_READS)
    results = [marker] * (width * TYPED_OFFSET_THREADS + 2)
    narrowed = [0x3EAA] * (TYPED_OFFSET_THREADS + 2)
    expected_results = list(results)
    expected_narrowed = list(narrowed)
    for tid in range(TYPED_OFFSET_THREADS):
        for slot, (_, index) in enumerate(TYPED_OFFSET_READS):
            value = TYPED_OFFSET_VALUES[tid + index]
            expected_results[width * tid + slot + 1] = _float_bits(value)
        doubled = 2.0 * TYPED_OFFSET_VALUES[tid + 2]
        expected_narrowed[tid + 1] = _float_bits(doubled) >> 16

    def words(values, encoding=FLOAT32_BITS, dtype="float32"):
        values = list(values)
        return {
            "dtype": dtype,
            "encoding": encoding,
            "values": values,
            "shape": [len(values)],
        }

    inputs = {
        "floats": words(_float_bits(value) for value in TYPED_OFFSET_VALUES),
        # OpenGL widens binary16 storage to binary32 words.
        "halves": (
            words(_float_bits(value) for value in TYPED_OFFSET_VALUES)
            if target == "opengl"
            else words(
                (
                    struct.unpack("<H", struct.pack("<e", value))[0]
                    for value in TYPED_OFFSET_VALUES
                ),
                FLOAT16_BITS,
                "float16",
            )
        ),
        "brains": _storage(
            target, [_float_bits(value) >> 16 for value in TYPED_OFFSET_VALUES]
        ),
        "results": words(results),
        "narrowed": _storage(target, narrowed),
    }
    expected = _bound_values(
        descriptor,
        {
            "results": words(expected_results),
            "narrowed": _storage(target, expected_narrowed),
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [TYPED_OFFSET_THREADS, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, expected


def test_directx_offset_pointer_reads_decode_their_element_type(tmp_path):
    _, request, _ = _typed_offset_request(tmp_path, "directx")
    generated = request.artifact_path.read_text(encoding="utf-8")
    body = generated[generated.index("CSMain") :]
    # No float, binary16 or bfloat16 element passes through an integer.
    assert "__crossgl_uint_to_float" not in body
    assert body.count("__crossgl_bfloat16_to_float(") == 4
    assert "__crossgl_bfloat16_from_float(" in body
    assert "__crossgl_bfloat16_to_float(uint(__crossgl_bfloat16_from_float" not in body
    from tests.test_translator.test_half_buffer_runtime import _validate_half

    if shutil.which("dxc"):
        module = _validate_half(request.artifact_path, tmp_path, "directx")
        assert module.is_file() and module.stat().st_size


def test_offset_pointer_reads_keep_their_element_type_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pointer-offset execution")
    from tests.test_translator.test_half_buffer_runtime import _validate_half

    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _typed_offset_request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="typed_offsets",
        validate=_validate_half,
    )
