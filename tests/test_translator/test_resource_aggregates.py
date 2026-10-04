"""Resource identities and offsets survive private aggregate operations."""

import os
import pickle
import shutil
import sys
from pathlib import Path

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalSizeofResolutionError
from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import (
    CastNode,
    FunctionCallNode,
    IdentifierNode,
    NamedType,
    PointerReinterpretNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    UnaryOpNode,
    VariableNode,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.resource_aggregates import (
    ResourceAggregateError,
    lower_resource_aggregates,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_member_pointer_provenance import (
    _source as _member_source,
)
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_RESOURCE_AGGREGATES"
CASES = (
    "cursor",
    "copy",
    "nested",
    "array",
    "helper",
    "rebase",
    "constant",
    "reference",
    "returned",
    "shadow",
    "effects",
    "selection",
    "piecewise",
    "template",
    "multidimensional",
    "const-slot",
    "entry-rebase",
    "alias-writeback",
    "typedef",
)


def _source(case):
    declarations = (
        "struct Cursor { const device int* input; device int* output; uint count; };"
    )
    setup = "Cursor cursor{left + 1, first + 2, 4};"
    read, write = "cursor.input[tid]", "cursor.output[tid]"
    if case == "copy":
        setup += "Cursor snapshot = cursor; cursor.input += 2; cursor.output += 1;"
        read = "snapshot.input[tid] + cursor.input[tid]"
    elif case == "nested":
        declarations += "struct Envelope { Cursor cursor; uint tag; };"
        setup = (
            "Envelope envelope{{left + 1, first + 2, 4}, 19}; Envelope copy = envelope;"
        )
        read, write = (
            "copy.cursor.input[tid] + int(copy.tag)",
            "copy.cursor.output[tid]",
        )
    elif case == "array":
        declarations = """struct Cursor {
            array<const device int*, 2> inputs;
            array<device int*, 2> outputs;
        };"""
        setup = "Cursor cursor{{left + 1, right + 2}, {first + 2, second + 3}};"
        read, write = "cursor.inputs[tid & 1][tid]", "cursor.outputs[tid & 1][tid]"
    elif case == "helper":
        declarations += """int load(const device int* values, uint index) { return values[index]; }
        int read_cursor(Cursor cursor, uint index) { return load(&cursor.input[index], 1); }"""
        read = "read_cursor(cursor, tid)"
    elif case == "rebase":
        setup += "cursor.input = right + 4; cursor.input -= 2; cursor.output = second + 5; cursor.output -= 2;"
    elif case == "constant":
        declarations = declarations.replace("const device", "const constant")
        setup = "Cursor cursor{right + 2, first + 2, 4};"
    elif case == "reference":
        declarations += "void advance(thread Cursor& cursor) { cursor.input += 2; cursor.output += 1; }"
        setup += "advance(cursor);"
    elif case == "returned":
        declarations += "Cursor make_cursor(const device int* input, device int* output) { Cursor result{input + 1, output + 2, 4}; return result; }"
        setup = "Cursor cursor = make_cursor(left, first);"
    elif case == "shadow":
        setup = (
            "const device int* left = right + 2; Cursor cursor{left, second + 3, 4};"
        )
    elif case == "effects":
        setup = "uint offset = 1; Cursor cursor{left + offset++, first + offset++, 4};"
    elif case == "selection":
        setup += "if ((tid & 1) != 0) { cursor.input = right + 2; cursor.output = second + 3; }"
    elif case == "piecewise":
        setup = "Cursor cursor; cursor.input = left + 1; cursor.output = first + 2; cursor.count = 4;"
    elif case == "template":
        declarations += "template<typename Pointer> struct Holder { Pointer value; };"
        setup = "Holder<const device int*> holder{left + 1}; Cursor cursor{holder.value, first + 2, 4};"
    elif case == "multidimensional":
        declarations = "struct Cursor { const device int* inputs[2][2]; device int* outputs[2][2]; };"
        setup = """Cursor cursor;
        cursor.inputs[0][0] = left + 1; cursor.inputs[0][1] = right + 2;
        cursor.inputs[1][0] = right + 1; cursor.inputs[1][1] = left + 2;
        cursor.outputs[0][0] = first + 1; cursor.outputs[0][1] = second + 1;
        cursor.outputs[1][0] = second + 2; cursor.outputs[1][1] = first + 2;"""
        read, write = (
            "cursor.inputs[tid / 2][tid & 1][tid]",
            "cursor.outputs[tid / 2][tid & 1][tid]",
        )
    elif case == "const-slot":
        declarations = declarations.replace(
            "device int* output;", "device int* const output;"
        )
    elif case == "entry-rebase":
        setup = "left += 1; first += 2; Cursor cursor{left, first, 4};"
    elif case == "alias-writeback":
        setup = "Cursor cursor{first + 2, first + 2, 4};"
        read = "left[tid + 1]"
    elif case == "typedef":
        declarations = (
            "typedef const device int* Input; typedef Input ReadPointer; "
            + declarations.replace("const device int* input;", "ReadPointer input;")
        )
    body = f"{setup}\n    {write} = {read};"
    if case == "alias-writeback":
        body += "second[tid + 3] = cursor.input[tid] + 5;"
    if case == "effects":
        body += "second[tid + 4] = int(offset);"
    if case == "shadow":
        body = f"if (tid < 4) {{ {body} }}"
    right_space = "constant" if case == "constant" else "device"
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void aggregate_resources(const device int* left [[buffer(0)]],
                                const {right_space} int* right [[buffer(1)]],
                                device int* first [[buffer(2)]],
                                device int* second [[buffer(3)]],
                                uint tid [[thread_position_in_grid]]) {{
    {body}
}}
"""


def _workload(case):
    left = [11, -3, 47, -91, 5, 107, -23, 53, 29, 71]
    right = [-113, 17, 61, 79, -31, 13, -43, 97, 109, -127]
    first, second = [-999] * 12, [-999] * 12
    for tid in range(4):
        value, output, offset = left[tid + 1], first, 2
        if case == "copy":
            value, offset = value + left[tid + 3], 3
        elif case == "nested":
            value += 19
        elif case in {"array", "selection"} and tid & 1:
            value, output, offset = right[tid + 2], second, 3
        elif case == "helper":
            value = left[tid + 2]
        elif case in {"rebase", "shadow"}:
            value, output, offset = right[tid + 2], second, 3
        elif case == "constant":
            value = right[tid + 2]
        elif case == "reference":
            value, offset = left[tid + 3], 3
        elif case == "effects":
            second[tid + 4] = 3
        elif case == "alias-writeback":
            second[tid + 3] = value + 5
        elif case == "multidimensional":
            buffers, targets, offsets = (
                (left, right, right, left),
                (first, second, second, first),
                (1, 2, 1, 2),
            )
            value, output, offset = (
                buffers[tid][tid + offsets[tid]],
                targets[tid],
                (1, 1, 2, 2)[tid],
            )
        output[tid + offset] = value

    def typed(values):
        return {"dtype": "int32", "shape": [len(values)], "values": values}

    return (
        {
            "left": typed(left),
            "right": typed(right),
            "first": typed([-999] * 12),
            "second": typed([-999] * 12),
        },
        {"first": typed(first), "second": typed(second)},
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", ("directx", "metal"))
def test_resource_aggregate_compiles(tmp_path, case, target):
    if not shutil.which("dxc" if target == "directx" else "xcrun"):
        pytest.skip(f"the optional {target} compiler is unavailable")
    source, _descriptor, package = _package(
        tmp_path,
        target,
        "int",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
    )
    (artifact,) = package.rglob("*.hlsl" if target == "directx" else "*.metal")
    validation = tmp_path / "validation"
    validation.mkdir()
    _, module = _compile(artifact.read_text(), target, validation)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", CASES)
def test_resource_aggregate_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required resource-aggregate execution")
    target = {"darwin": "metal", "win32": "directx"}[sys.platform]
    source, descriptor, package = _package(
        tmp_path,
        target,
        "int",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
    )
    inputs, outputs = _workload(case)
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="aggregate_resources",
    )


@pytest.mark.parametrize(
    "body,reason",
    [
        ("cursor.input[tid] = 4;", "write-through-readonly-pointer"),
        ("cursor.input++;", "pointer-unary-operator"),
        ("first[tid] = int(cursor.input - cursor.input);", "pointer-binary-operator"),
        ("cursor.output = cursor.input;", "pointer-contract-mismatch"),
        ("cursor.input = 0;", "pointer-contract-mismatch"),
        ("first[tid] = sizeof(Cursor);", "resource-reference-escape"),
        ("first[tid] = sizeof(cursor);", "resource-reference-escape"),
        ("first[tid] = unknown(cursor);", "resource-reference-escape"),
        ("first[tid] = unknown(cursor.input);", "resource-reference-escape"),
        ("cursor.output[tid] += 4;", "compound-resource-write"),
    ],
)
@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_resource_aggregate_rejects_unproven_operations(tmp_path, body, reason, target):
    source = _source("cursor").replace("cursor.output[tid] = cursor.input[tid];", body)
    path = tmp_path / "unsupported.metal"
    path.write_text(source)
    if "sizeof(" in body:
        with pytest.raises(
            MetalSizeofResolutionError, match="aggregate object layout is not available"
        ):
            translate(str(path), backend=target, format_output=False)
        return
    with pytest.raises(ResourceAggregateError) as error:
        translate(str(path), backend=target, format_output=False)
    assert error.value.reason == reason


def test_resource_aggregate_failure_is_reported_without_an_artifact(tmp_path):
    (tmp_path / "source.metal").write_text(
        _source("cursor").replace(
            "cursor.output[tid] = cursor.input[tid];", "cursor.input[tid] = 4;"
        )
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("source.metal",),
            targets=("directx",),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert report["summary"]["translatedCount"] == 0
    assert [item["code"] for item in report["diagnostics"]] == [
        "project.translate.resource-aggregate-unsupported"
    ]
    assert "write-through-readonly-pointer" in report["diagnostics"][0]["message"]
    assert not list((tmp_path / "out").rglob("*.hlsl"))


def test_resource_lowering_retains_source_ast_and_is_deterministic(tmp_path):
    path = tmp_path / "source.metal"
    path.write_text(_source("array"))
    ast = parse(translate(str(path), backend="cgl", format_output=False))
    original = pickle.dumps(ast)
    lowered = lower_resource_aggregates(ast)
    assert lowered is not ast
    assert pickle.dumps(ast) == original
    generator = HLSLCodeGen()
    first = generator.generate(ast)
    assert generator.generate(ast) == first
    assert pickle.dumps(ast) == original


@pytest.mark.parametrize("stage_local", (False, True))
def test_resource_lowering_removes_only_unreachable_resource_signatures(stage_local):
    ast = parse("""shader ResourceHelpers {
        struct Cursor { device int* data; }
        int read(Cursor cursor, uint index) { return cursor.data[index]; }
        void unused(Cursor cursor, uint index) { cursor.data[index] = 0; }
        int ordinary(int value) { return value + 1; }
        compute { void main(RWStructuredBuffer<int> values @buffer(0),
                            RWStructuredBuffer<int> results @buffer(1)) {
            Cursor cursor = Cursor(values);
            results[0] = read(cursor, 0u);
        } }
    }""")
    stage = next(iter(ast.stages.values()))
    if stage_local:
        stage.local_functions.extend(ast.functions)
        ast.functions = []
    before = pickle.dumps(ast)
    lowered = lower_resource_aggregates(ast)
    names = {f.name for f in lowered.functions}
    names.update(f.name for s in lowered.stages.values() for f in s.local_functions)
    assert "unused" not in names
    assert {"read", "ordinary"} <= names
    assert pickle.dumps(ast) == before


@pytest.mark.parametrize(
    "operation",
    ["local-reference", "reference-return", "address", "cast", "reinterpret"],
)
def test_resource_aggregate_alias_and_layout_escapes_are_diagnostic(operation):
    ast = parse("""shader T {
        struct Cursor { device int* data; }
        Cursor identity(Cursor value) { return value; }
        compute { void main(RWStructuredBuffer<int> output @buffer(0)) {
            Cursor cursor = identity(Cursor(output));
        } }
    }""")
    entry = next(iter(ast.stages.values())).entry_point
    if operation == "local-reference":
        entry.body.statements.append(
            VariableNode(
                "alias", ReferenceType(NamedType("Cursor")), IdentifierNode("cursor")
            )
        )
        reason = "local-reference-alias"
    elif operation == "reference-return":
        ast.functions[0].return_type = ReferenceType(NamedType("Cursor"))
        reason = "reference-return"
    elif operation == "address":
        entry.body.statements.append(UnaryOpNode("&", IdentifierNode("cursor")))
        reason = "aggregate-address-escape"
    else:
        node_type = CastNode if operation == "cast" else PointerReinterpretNode
        entry.body.statements.append(
            node_type(IdentifierNode("cursor"), PointerType(PrimitiveType("int")))
        )
        reason = "resource-reference-cast"
    with pytest.raises(ResourceAggregateError, match=reason):
        lower_resource_aggregates(ast)


def test_resource_aggregate_member_address_executes_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required resource-aggregate execution")
    target = {"darwin": "metal", "win32": "directx"}[sys.platform]
    source, descriptor, package = _package(
        tmp_path,
        target,
        "int",
        (1, 1, 1),
        source=_member_source("constant", "direct"),
        software_subgroups=False,
    )
    values = [-13, 4, 71, -19, 103, 53, 27, -6]
    result = [-123456] * 4
    for tid in range(3):
        result.extend(
            (values[tid + 2] + 11, values[tid + 4] + 11, values[tid + 1] + 101, tid + 3)
        )
    result.extend([-123456] * 4)
    expected = _bound_values(
        descriptor, {"results": {"dtype": "int32", "shape": [20], "values": result}}
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "src": {"dtype": "int32", "shape": [8], "values": values},
                "results": {"dtype": "int32", "shape": [20], "values": [-123456] * 20},
            },
        ),
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="member_pointer",
    )


@pytest.mark.parametrize("member", ["cursor", "Cursor"])
def test_canonical_resource_aggregate_layout_is_not_reinterpreted(member):
    source = f"""shader T {{
        struct Cursor {{ device int* data; }}
        compute {{ void main(RWStructuredBuffer<int> output @buffer(0)) {{
            Cursor cursor = Cursor(output);
            output[0] = sizeof({member});
        }} }}
    }}"""
    with pytest.raises(ResourceAggregateError, match="resource-reference-escape"):
        lower_resource_aggregates(parse(source))


def test_pointer_array_partial_initialization_is_not_a_buffer_reference(tmp_path):
    path = tmp_path / "partial.metal"
    path.write_text(_source("array").replace("{left + 1, right + 2}", "{left + 1}"))
    with pytest.raises(
        ResourceAggregateError, match="pointer-array-initialization-arity"
    ):
        translate(str(path), backend="directx", format_output=False)


def test_aggregate_reference_cannot_change_an_external_buffer_layout():
    source = """shader T {
        struct Cursor { device int* data; }
        compute { void main(StructuredBuffer<Cursor> input @buffer(0)) {} }
    }"""
    with pytest.raises(ResourceAggregateError, match="aggregate-buffer-abi"):
        lower_resource_aggregates(parse(source))


def test_resource_read_proof_retains_effectful_argument_checks(tmp_path):
    path = tmp_path / "source.metal"
    path.write_text(_source("helper"))
    ast = lower_resource_aggregates(
        parse(translate(str(path), backend="cgl", format_output=False))
    )
    generator = HLSLCodeGen()
    generator.current_hlsl_available_functions = {
        function.name: function for function in ast.functions
    }
    reader = generator.current_hlsl_available_functions["load"]
    assert reader.resource_aggregate_nonmutating
    assert not generator.hlsl_expression_has_observable_side_effects(
        FunctionCallNode(IdentifierNode("load"), [])
    )
    assert generator.hlsl_expression_has_observable_side_effects(
        FunctionCallNode(
            IdentifierNode("load"),
            [UnaryOpNode("++", IdentifierNode("index"), is_postfix=True)],
        )
    )
    assert generator.hlsl_expression_has_observable_side_effects(
        FunctionCallNode(IdentifierNode("unresolved"), [])
    )
    assert all(
        not function.resource_aggregate_nonmutating
        for function in ast.functions
        if function.name.startswith("crosstl_resource_store_")
    )


@pytest.mark.parametrize("qualifier", ["static", "volatile", "threadgroup"])
def test_resource_read_proof_excludes_observable_local_storage(qualifier):
    ast = parse("""shader T {
        struct Cursor { device int* data; }
        int advance() { int count = 0; count += 1; return count; }
        compute { void main(RWStructuredBuffer<int> output @buffer(0)) {
            Cursor cursor = Cursor(output); cursor.data[0] = advance();
        } }
    }""")
    ast.functions[0].body.statements[0].qualifiers.append(qualifier)
    lowered = lower_resource_aggregates(ast)
    helper = next(f for f in lowered.functions if f.name == "advance")
    assert not getattr(helper, "resource_aggregate_nonmutating", False)


def test_resource_aggregate_gate_requires_native_execution():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate indexed DirectX gather and resource aggregates"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'CROSTL_REQUIRE_MLX_DIRECTX_GENERAL_GATHER: "1"' in step
    assert (
        "test_general_gather_directx.py" in step
        and "test_resource_aggregates.py" in step
    )
    assert "--timeout-seconds 1200" in step and "--junitxml=" in step
    assert "--basetemp=" in step and "-n auto --dist loadgroup" in step
    assert "if:" not in step and "continue-on-error" not in workflow
    assert "runs-on: windows-2025" in workflow
    assert "Get-FileHash" in workflow and "d3d10warp.dll" in workflow
    directx_job = workflow.split("  directx:", 1)[1]
    assert 'python-version: "3.12"' in directx_job
    assert directx_job.index(
        "Initialize DirectX execution evidence"
    ) < directx_job.index("Install CrossTL and runtime dependencies")
    install = ci_coverage.workflow_step_section(
        workflow, "Install CrossTL and runtime dependencies"
    )
    assert "set -euo pipefail" in install
    assert "tee .mlx-gather-directx/dependencies.log" in install
    for event in ("push", "pull_request"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/**",
        )
