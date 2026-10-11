"""Keep zero-extent standard arrays distinct from C-style array declarations."""

import os
import sys

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.runtime_helpers import _validate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_EMPTY_ARRAYS"
GUARD = 0xDEADBEEF
CASES = (
    ("float", 4, 4),
    ("half", 2, 2),
    ("char", 1, 1),
    ("ushort3", 8, 8),
    ("float3", 16, 16),
    ("packed_float3", 12, 4),
    ("array<float3, 0>", 16, 16),
    ("array<float3, 2>", 32, 16),
    ("Pair", 8, 4),
    ("const device float*", None, 8),
    ("device float*", None, 8),
    ("const thread float*", None, 8),
    ("threadgroup float*", None, 8),
    ("constant float*", None, 8),
)


@pytest.mark.parametrize(
    "element,expected",
    (
        ("const device float*", "const device float*"),
        ("const volatile device float*", "volatile const device float*"),
        ("constant float*", "constant float*"),
        ("const f16", "const half"),
        ("u16", "ushort"),
        ("packed_float3", "packed_float3"),
    ),
)
def test_empty_standard_array_retains_qualified_storage_type(element, expected):
    from crosstl.translator.ast import NamedType, PointerType
    from crosstl.translator.codegen.metal_codegen import MetalCodeGen
    from crosstl.translator.lexer import Lexer
    from crosstl.translator.parser import Parser

    node = Parser(Lexer(f"array<{element}, 0>").tokens).parse_type()
    assert isinstance(node, NamedType) and node.name == "array"
    assert node.generic_args[1].value == 0
    if "*" in element:
        assert isinstance(node.generic_args[0], PointerType)
        assert node.generic_args[0].address_space in {"device", "constant"}
    assert MetalCodeGen().map_type(node) == f"array<{expected}, 0>"


def test_empty_standard_array_does_not_change_c_style_extent():
    from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
    from crosstl.translator.ast import ArrayType
    from crosstl.translator.lexer import Lexer
    from crosstl.translator.parser import Parser

    converter = MetalToCrossGLConverter()
    assert converter.map_type("array<float, 0>") == "array<float, 0>"
    assert converter.map_type("array<float, 1>") == "float[1]"
    assert converter.map_type("float[0]") == "float[0]"
    node = Parser(Lexer("float[0]").tokens).parse_type()
    assert isinstance(node, ArrayType) and node.size.value == 0


@pytest.mark.parametrize("label", ("row_major", "column_major"))
def test_generic_value_labels_are_not_consumed_as_qualifiers(label):
    from crosstl.translator.ast import NamedType
    from crosstl.translator.lexer import Lexer
    from crosstl.translator.parser import Parser

    node = Parser(Lexer(f"Layout<{label}>").tokens).parse_type()
    assert isinstance(node.generic_args[0], NamedType)
    assert node.generic_args[0].name == label


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("extent", ("0", "3 - 3", "(2 * 4) - 8"))
def test_foreign_targets_reject_constant_zero_array_objects(target, extent):
    from crosstl.translator.codegen.array_utils import ZeroExtentArrayUnsupportedError
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
    from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

    generator = HLSLCodeGen() if target == "directx" else GLSLCodeGen()
    with pytest.raises(ZeroExtentArrayUnsupportedError) as error:
        generator.map_type(f"array<float, {extent}>")
    assert error.value.reason == "zero-extent-standard-array"
    assert error.value.missing_capabilities == (f"{target}.zero-extent-standard-array",)


def _source(element, size, form):
    array = f"metal::array<{element}, 0>"
    if form == "expression":
        array = f"array<{element}, 3 - 3>"
    alias = f"using Empty = {array};" if form == "alias" else ""
    if alias:
        array = "Empty"
    size_expression = (
        "sizeof(Payload)" if size is not None else "copied.empty.max_size()"
    )
    return f"""#include <metal_stdlib>
using namespace metal;
struct Pair {{ uint x; uint y; }};
{alias}
struct Payload {{ uint head; {array} empty; uint tail; }};
kernel void empty_arrays(device uint* results [[buffer(0)]],
                         uint tid [[thread_position_in_grid]]) {{
    uint counter = 19 + tid;
    Payload original{{91 + tid, {{}}, counter++}};
    Payload copied = original;
    copied.tail += 7;
    uint offset = 4 + tid * 8;
    results[offset] = original.head;
    results[offset + 1] = original.tail;
    results[offset + 2] = copied.tail;
    results[offset + 3] = counter;
    results[offset + 4] = {size_expression};
    results[offset + 5] = alignof(Payload);
    results[offset + 6] = copied.empty.size();
    results[offset + 7] = copied.empty.empty();
}}
"""


@pytest.mark.parametrize("element,size,alignment", CASES)
@pytest.mark.parametrize("form", ("direct", "alias", "expression"))
def test_zero_extent_standard_arrays_execute_natively(
    tmp_path, element, size, alignment, form
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required empty-array execution")
    assert sys.platform == "darwin", "the round-trip gate requires native Metal"
    source, descriptor, package = _package(
        tmp_path,
        "metal",
        "uint",
        (1, 1, 1),
        source=_source(element, size, form),
        software_subgroups=False,
    )
    aggregate_alignment = max(4, alignment)
    total = 0
    if size is not None:
        tail_end = ((4 + alignment - 1) // alignment) * alignment + size + 4
        total = (
            (tail_end + aggregate_alignment - 1) // aggregate_alignment
        ) * aggregate_alignment
    expected = [GUARD] * 4
    for tid in range(3):
        expected.extend(
            [91 + tid, 19 + tid, 26 + tid, 20 + tid, total, aggregate_alignment, 0, 1]
        )
    expected += [GUARD] * 4

    def words(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    outputs = _bound_values(descriptor, {"results": words(expected)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, {"results": words([GUARD] * len(expected))}),
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="metal",
    )
    assert not request.execution_plan.diagnostics
    assert "array<" in request.artifact_path.read_text()
    assert "[0]" not in request.artifact_path.read_text()
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source,
        original_entry="empty_arrays",
        metal_compile_flags=("-fno-fast-math",),
        validate=_validate,
    )


@pytest.mark.parametrize("target", ("opengl", "directx"))
def test_unsupported_empty_array_targets_report_a_diagnostic(tmp_path, target):
    (tmp_path / "empty.metal").write_text(_source("float", 4, "direct"))
    report = translate_project(
        ProjectConfig(root=tmp_path, targets=(target,), output_dir="out"),
        format_output=False,
    )
    payload = report.to_json()
    assert payload["summary"]["failedCount"] == 1
    diagnostics = [
        item
        for item in payload["diagnostics"]
        if item["code"] == "project.translate.zero-extent-array-unsupported"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0]["severity"] == "error"
    assert diagnostics[0]["missingCapabilities"] == [
        f"{target}.zero-extent-standard-array"
    ]
    assert not list((tmp_path / "out").rglob("*.hlsl"))
    assert not list((tmp_path / "out").rglob("*.glsl"))
