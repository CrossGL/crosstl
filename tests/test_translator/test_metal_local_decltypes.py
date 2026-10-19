"""``decltype`` of a function-local variable inside Metal template arguments.

For an unparenthesized id-expression naming a variable, ``decltype`` is the variable's declared type
([dcl.type.decltype]/1.3). Template materialization resolves it after the enclosing template is instantiated, both
in a nested struct template argument (``Limits<typename decltype(tile)::elem_type>``) and in an explicit function
template argument (``tripled<decltype(scale)>(...)``). A parenthesized operand names an lvalue reference and a
pointer local has no spelled class type here, so both stay refused with the typed specialization diagnostic.
"""

import os
import sys

import pytest

from crosstl.backend.Metal.local_decltypes import (
    complete_declared_type,
    substitute_local_decltypes,
)
from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_LOCAL_DECLTYPES"
ENTRY = "lowest_int"
WORKGROUP = (32, 1, 1)
GROUPS = 3
LOWEST = -1000
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [(index * 2654435761 + 7) % 1000003 for index in range(GROUPS * 32)]

STRUCT_ARGUMENT = "Limits<typename decltype(tile)::elem_type>::min"
FUNCTION_ARGUMENT = "tripled<decltype(scale)>(int(index))"


def _source(struct_argument=STRUCT_ARGUMENT, function_argument=FUNCTION_ARGUMENT):
    return f"""#include <metal_stdlib>
using namespace metal;

template <typename T>
struct Limits {{
  static constexpr constant T min = T({LOWEST});
}};

template <typename T, int N>
struct Tile {{
  typedef T elem_type;
  T values[N];
}};

template <typename T>
void fill(thread Tile<T, 2>& tile, T value) {{
  for (int i = 0; i < 2; i++) {{
    tile.values[i] = value + T(i);
  }}
}}

template <typename U>
U tripled(int v) {{
  return U(v) * U(3);
}}

template <typename T>
[[kernel]] void lowest(
    const device T* values [[buffer(0)]],
    device T* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {{
  Tile<T, 2> tile;
  const device T* cursor = values + index;
  fill(tile, *cursor);
  tile.values[1] = {struct_argument};
  T scale = tile.values[0];
  results[index] = tile.values[0] + tile.values[1] + {function_argument};
}}

template [[host_name("{ENTRY}")]] [[kernel]] void lowest<int>(
    const device int*, device int*, uint);
"""


def _expected():
    return [value + LOWEST + 3 * index for index, value in enumerate(VALUES)] + GUARD


def _request(root, target):
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=_source(), software_subgroups=False
    )
    initial = [-0x21524111] * len(VALUES) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "int32", "shape": [len(initial)], "values": initial},
        },
    )
    expected = _expected()
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "int32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


def _translate(root, target, source):
    (root / "decltypes.metal").write_text(source)
    return translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("decltypes.metal",),
            targets=(target,),
            output_dir="out",
            entry_points={"decltypes.metal": (ENTRY,)},
            workgroup_size=WORKGROUP,
        ),
        format_output=False,
    ).to_json()


def test_declared_type_is_spelled_only_for_complete_value_declarations():
    assert complete_declared_type("Tile<float,  2>", "") == "Tile<float, 2>"
    assert complete_declared_type("uint", None) == "uint"
    for type_text, suffix in (
        ("Tile<float, 2>", "[4]"),
        ("const Tile<float, 2>", ""),
        ("thread Tile<float, 2>&", ""),
        ("device float*", ""),
        ("constexpr int", ""),
        ("auto", ""),
        ("threadgroup float", ""),
    ):
        assert complete_declared_type(type_text, suffix) is None, type_text


def test_only_unparenthesized_known_locals_are_substituted():
    types = {"tile": "Tile<float, 2>"}
    assert (
        substitute_local_decltypes("typename decltype( tile )::elem_type", types)
        == "typename Tile<float, 2>::elem_type"
    )
    for argument in (
        "decltype((tile))",
        "decltype(other)",
        "decltype(tile.values)",
        "my_decltype(tile)",
        "ns::decltype(tile)",
    ):
        assert substitute_local_decltypes(argument, types) == argument


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_local_decltype_template_arguments_compile(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "decltype" not in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "struct_argument, function_argument",
    (
        # ``decltype((tile))`` is ``Tile<int, 2>&``, which names no nested member.
        ("Limits<typename decltype((tile))::elem_type>::min", FUNCTION_ARGUMENT),
        # ``cursor`` is a pointer, whose declared type is not a spelled value type here.
        (STRUCT_ARGUMENT, "tripled<decltype(cursor)>(int(index))"),
    ),
)
def test_unspellable_local_decltypes_stay_refused(
    tmp_path, target, struct_argument, function_argument
):
    report = _translate(tmp_path, target, _source(struct_argument, function_argument))
    assert report["summary"]["translatedCount"] == 0
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes == {"project.translate.metal-template-specialization"}, codes


def test_local_decltype_template_arguments_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native local decltypes")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_local_decltypes_are_required_on_all_native_platforms():
    from pathlib import Path

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_local_decltypes.py" in step
    assert "if:" not in step and "continue-on-error" not in step
