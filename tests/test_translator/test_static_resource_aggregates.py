"""Static metadata is not instance state in resource-bearing helper graphs."""

import os
import pickle
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.codegen.resource_aggregates import (
    ResourceAggregateError,
    lower_resource_aggregates,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_STATIC_RESOURCE_AGGREGATES"
GUARD = 0x5A1B2C3D
INPUTS = [0, 11, 0xFFFFFFFF, 0x80000000, 0x89ABCDEF]
TAG = "struct Tag { static constant constexpr uint bias = 7u; };"
MIXED = "struct Mixed { static constant constexpr uint bias = 7u; uint first; static constant constexpr uint extra = 11u; uint second; };"
CASES = {
    "empty": (TAG, "Tag tag", "tag.bias", "", "{}", 7),
    "named": (TAG, "Tag tag", "tag.bias", "Tag tag{};", "tag", 7),
    "mixed": (
        MIXED,
        "Mixed tag",
        "tag.bias + tag.first + tag.extra + tag.second",
        "",
        "{3u, 5u}",
        26,
    ),
    "nested": (
        TAG + "struct Envelope { Tag tag; uint value; };",
        "Envelope tag",
        "tag.tag.bias + tag.value",
        "",
        "{{}, 3u}",
        10,
    ),
    "array": (
        MIXED,
        "Mixed tag",
        "tag.bias + tag.first + tag.extra + tag.second",
        "Mixed tags[2] = {{3u, 5u}, {9u, 2u}};",
        "tags[tid & 1u]",
        (26, 29),
    ),
    "constructor": (
        MIXED.replace(
            "uint second;", "uint second; Mixed(uint a, uint b): first(a), second(b) {}"
        ),
        "Mixed tag",
        "tag.bias + tag.first + tag.extra + tag.second",
        "Mixed tag(3u, 5u);",
        "tag",
        26,
    ),
    "returned": (
        MIXED + "Mixed make_tag() { return {3u, 5u}; }",
        "Mixed tag",
        "tag.bias + tag.first + tag.extra + tag.second",
        "",
        "make_tag()",
        26,
    ),
    "copy": (
        MIXED,
        "Mixed tag",
        "tag.bias + tag.first + tag.extra + tag.second",
        "Mixed first{3u, 5u}; Mixed tag = first; tag.second += 2u;",
        "tag",
        28,
    ),
    "resource": (TAG, "Tag tag", "tag.bias + c.bias", "", "{}", 20),
    "array-field": (
        "struct Mixed { static constant constexpr uint bias = 7u; uint values[2]; };",
        "Mixed tag",
        "tag.bias + tag.values[0] + tag.values[1]",
        "",
        "{{3u, 5u}}",
        15,
    ),
    "nested-array-field": (
        "struct Mixed { static constant constexpr uint bias = 7u; uint values[2][2]; };",
        "Mixed tag",
        "tag.bias + tag.values[0][0] + tag.values[0][1] + tag.values[1][0] + tag.values[1][1]",
        "",
        "{{{3u, 5u}, {9u, 2u}}}",
        26,
    ),
    "effect": (
        TAG + "Tag make_tag(thread uint& count) { ++count; return {}; }",
        "Tag tag",
        "tag.bias",
        "",
        "{}",
        10,
    ),
    "index-effect": (TAG, "Tag tag", "tag.bias", "", "{}", 10),
    "floating": (
        "struct Tag { static constant constexpr float bias = 7.5f; };",
        "Tag tag",
        "uint(tag.bias * 2.0f)",
        "",
        "{}",
        15,
    ),
    "boolean": (
        "struct Tag { static constant constexpr bool bias = true; };",
        "Tag tag",
        "(tag.bias ? 7u : 19u)",
        "",
        "{}",
        7,
    ),
}


def _source(case):
    declarations, parameter, bias, setup, argument, _ = CASES[case]
    static = "static constant constexpr uint bias = 13u;" if case == "resource" else ""
    evaluation = f"return c.input[index] + {bias};"
    if case == "effect":
        evaluation = "uint count = 0u; uint value = make_tag(count).bias; return c.input[index] + value + 3u * count + tag.bias - 7u;"
    elif case == "index-effect":
        evaluation = "Tag tags[2] = {{}, {}}; uint count = 0u; uint value = tags[count++].bias; return c.input[index] + value + 3u * count + tag.bias - 7u;"
    return f"""#include <metal_stdlib>
using namespace metal;
struct Cursor {{ {static} const device uint* input; }};
{declarations}
uint read_value(Cursor c, uint index, {parameter}) {{ {evaluation} }}
kernel void resources(const device uint* inputs [[buffer(0)]], device uint* results [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{
    Cursor c{{inputs}};
    {setup}
    results[4 + tid] = read_value(c, tid, {argument});
}}
"""


def _request(root, target, case):
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
        index_range_assertions=(
            (
                {
                    "expression": "reference.offset + index",
                    "minimum": 0,
                    "maximum": len(INPUTS) - 1,
                },
            )
            if target == "opengl"
            else ()
        ),
    )
    bias = CASES[case][-1]
    output = (
        [GUARD] * 4
        + [
            (word + (bias[index & 1] if isinstance(bias, tuple) else bias)) & 0xFFFFFFFF
            for index, word in enumerate(INPUTS)
        ]
        + [GUARD] * 4
    )
    inputs = _bound_values(
        descriptor,
        {
            "inputs": {"dtype": "uint32", "shape": [len(INPUTS)], "values": INPUTS},
            "results": {
                "dtype": "uint32",
                "shape": [len(output)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * len(INPUTS) + [GUARD] * 4,
            },
        },
    )
    expected = _bound_values(
        descriptor,
        {"results": {"dtype": "uint32", "shape": [len(output)], "values": output}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [len(INPUTS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", CASES)
def test_static_resource_aggregate_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(
        request.artifact_path.read_text(),
        target,
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra", "-fno-fast-math"),
    )


@pytest.mark.parametrize("case", CASES)
def test_static_resource_aggregate_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required static-member execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="resources",
        metal_compile_flags=("-Wall", "-Wextra", "-fno-fast-math"),
    )


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("initializer", ["{}", "{inputs, inputs}"])
def test_static_metadata_does_not_hide_invalid_instance_arity(
    tmp_path, target, initializer
):
    source = _source("resource").replace("Cursor c{inputs};", f"Cursor c{initializer};")
    (tmp_path / "source.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("source.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.resource-aggregate-unsupported"
        and "aggregate-initialization-arity" in item["message"]
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("representation", ["attribute", "qualifier"])
def test_static_field_classification_preserves_source_and_metadata(representation):
    ast = parse("""shader T {
        struct Tag { static constant uint bias = 7u; uint value; }
        struct Cursor { device uint* data; }
        uint read_tag(Tag tag) { return tag.bias + tag.value; }
        compute { void main(RWStructuredBuffer<uint> output @buffer(0)) {
            Cursor cursor = Cursor(output);
            cursor.data[0] = read_tag({3u});
        } }
    }""")
    tag = next(struct for struct in ast.structs if struct.name == "Tag")
    if representation == "qualifier":
        tag.members[0].qualifiers = ["static", "constant"]
        tag.members[0].attributes = []
    original = pickle.dumps(ast)
    lowered = lower_resource_aggregates(ast)
    assert pickle.dumps(ast) == original
    target = next(struct for struct in lowered.structs if struct.name == "Tag")
    assert pickle.dumps(target.members[0]) == pickle.dumps(tag.members[0])
    factory = next(
        function
        for function in lowered.functions
        if function.name.startswith("crosstl_make_Tag")
    )
    assert len(factory.parameters) == 1


@pytest.mark.parametrize(
    "declaration, reason",
    [
        ("static uint value = 7u;", "static-member-value-unsupported"),
        ("static constant uint value;", "static-member-value-unsupported"),
        (
            "static constant uint value = external_bias;",
            "static-member-value-unsupported",
        ),
        ("static device uint* value;", "static-resource-reference"),
    ],
)
def test_static_state_is_not_replaced_with_default_values(declaration, reason):
    ast = parse(f"""shader T {{
        struct Tag {{ {declaration} }}
        struct Cursor {{ device uint* data; }}
        compute {{ void main(RWStructuredBuffer<uint> output @buffer(0)) {{
            Cursor cursor = Cursor(output);
            Tag tag;
            uint external_bias = 99u;
            cursor.data[0] = tag.value;
        }} }}
    }}""")
    with pytest.raises(ResourceAggregateError) as error:
        lower_resource_aggregates(ast)
    assert error.value.reason == reason


def test_static_resource_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_static_resource_aggregates.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
