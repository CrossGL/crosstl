"""Pointer offsets preserve storage provenance and execute the original call."""

import os
import shutil
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
        / ".github/workflows/mlx-gather-roundtrip.yml"
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
