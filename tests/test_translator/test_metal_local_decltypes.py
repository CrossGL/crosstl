"""Function-local ``decltype`` and type-trait constants inside Metal template arguments.

For an unparenthesized id-expression naming a variable, ``decltype`` is the variable's declared type
([dcl.type.decltype]/1.3). Template materialization resolves it after the enclosing template is instantiated, both
in a nested struct template argument (``Limits<typename decltype(tile)::elem_type>``), through local aliases
(``using tile_t = decltype(tile);``), as a declared type (``typename decltype(tile)::elem_type step``) and in an
explicit function template argument (``tripled<decltype(scale)>(...)``). A parenthesized operand names an lvalue reference and a
pointer local has no spelled class type here, so both stay refused with the typed specialization diagnostic.

A ``constexpr bool`` local initialized by a type trait over concrete types (``is_same_v<M, bool>``) is a known
constant, so ``conditional_t<is_bool, bool, T>`` selects its branch before the struct it parameterizes materializes,
as MLX's attention kernels select the mask element type.
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


def _source(
    struct_argument=STRUCT_ARGUMENT, function_argument=FUNCTION_ARGUMENT, mask="bool"
):
    return f"""#include <metal_stdlib>
using namespace metal;

template <typename T>
struct Width {{
  static constexpr constant int bytes = sizeof(T);
}};

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

// Metal compilers since 32023.921 qualify decltype of a local with its
// thread address space, which a return type cannot carry; a local can.
template <typename U>
int tripled(int v) {{
  U value = v * 3;
  return value;
}}

template <typename T, typename M>
[[kernel]] void lowest(
    const device T* values [[buffer(0)]],
    device T* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {{
  Tile<T, 2> tile;
  const device T* cursor = values + index;
  fill(tile, *cursor);
  tile.values[1] = {struct_argument};
  // The aliases MLX's attention kernels use to name a tile's element type.
  using tile_t = decltype(tile);
  using elem_t = typename tile_t::elem_type;
  constexpr auto lowest = Limits<elem_t>::min;
  typename decltype(tile)::elem_type step = elem_t(3);
  T scale = tile.values[0];
  constexpr bool is_bool = is_same_v<M, bool>;
  using melem_t = typename metal::conditional_t<is_bool, bool, elem_t>;
  results[index] = tile.values[0] + tile.values[1] + lowest / elem_t(10) +
      step * {function_argument} + Width<melem_t>::bytes;
}}

template [[host_name("{ENTRY}")]] [[kernel]] void lowest<int, {mask}>(
    const device int*, device int*, uint);
"""


# sizeof(bool) and sizeof(int) in the Metal Shading Language specification, table 2.1.
MASK_BYTES = {"bool": 1, "int": 4}


def _expected(mask):
    return [
        value + LOWEST + LOWEST // 10 + 9 * index + MASK_BYTES[mask]
        for index, value in enumerate(VALUES)
    ] + GUARD


def _request(root, target, mask="bool"):
    source, descriptor, package = _package(
        root,
        target,
        "int",
        WORKGROUP,
        source=_source(mask=mask),
        software_subgroups=False,
    )
    initial = [-0x21524111] * len(VALUES) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "int32", "shape": [len(initial)], "values": initial},
        },
    )
    expected = _expected(mask)
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


@pytest.mark.parametrize("mask", sorted(MASK_BYTES))
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_local_decltype_template_arguments_compile(tmp_path, target, mask):
    _, request, _ = _request(tmp_path, target, mask)
    generated = request.artifact_path.read_text()
    assert "decltype" not in generated and "conditional" not in generated
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


@pytest.mark.parametrize("mask", sorted(MASK_BYTES))
def test_local_decltype_template_arguments_execute(tmp_path, mask):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native local decltypes")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target, mask)
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

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_local_decltypes.py" in step
    assert "if:" not in step and "continue-on-error" not in step
