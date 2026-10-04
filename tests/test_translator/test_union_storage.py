"""Overlapping union views preserve bytes through native value operations."""

import os
import sys
from pathlib import Path

import pytest

from crosstl._crosstl import translate
from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.codegen.glsl_union_storage import OpenGLUnionLayoutError
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_UNION_STORAGE"
HEADER = """#include <metal_stdlib>
using namespace metal;
union Bits { uint2 words; uchar4 bytes[2]; };
union Wide { uint4 words; uchar4 bytes[4]; };
union Word { uint word; float value; int signed_word; uchar4 bytes; };
union Flags { uint word; bool4 lanes; };
union FloatPair { float2 values; uint2 words; float scalars[2]; int signed_words[2]; };
struct Holder { Bits values[2]; };
Bits make_bits(uint word) { Bits value; value.words = uint2(word, 0x88776655u); return value; }
Bits copy_bits(Bits value) { value.bytes[1] = uchar4(9u, 10u, 11u, 12u); return value; }
uint next_index(thread uint& count) { uint old = count; count += 1u; return old; }
"""
CASES = {
    "unpack": (
        "Bits value; value.words = uint2(0x04030201u, 0xff807f00u);",
        [
            "value.bytes[0][0]",
            "value.bytes[0][3]",
            "value.bytes[1][2]",
            "value.bytes[1][3]",
        ],
        [1, 4, 128, 255],
    ),
    "pack": (
        "Bits value; value.bytes[0] = uchar4(1u, 2u, 3u, 4u); value.bytes[1] = uchar4(0u, 127u, 128u, 255u);",
        ["value.words.x", "value.words.y"],
        [0x04030201, 0xFF807F00],
    ),
    "wide": (
        "Wide value; value.words = uint4(0u); value.bytes[3] = uchar4(252u, 253u, 254u, 255u);",
        ["value.words.x", "value.words.w", "value.bytes[3].z"],
        [0, 0xFFFEFDFC, 254],
    ),
    "float_bits": (
        "Word value; value.value = -0.0f;",
        ["value.word", "uint(signbit(value.value))"],
        [0x80000000, 1],
    ),
    "signed_bits": (
        "Word value; value.signed_word = -2;",
        ["value.word", "uint(value.signed_word == -2)"],
        [0xFFFFFFFE, 1],
    ),
    "float_array": (
        "FloatPair value; value.values = float2(1.0f, -0.0f); value.scalars[0] = 2.0f;",
        ["value.words.x", "value.words.y", "uint(value.values.x)"],
        [0x40000000, 0x80000000, 2],
    ),
    "signed_array": (
        "FloatPair value; value.words = uint2(0u); value.signed_words[1] = -2;",
        ["value.words.x", "value.words.y", "uint(value.signed_words[1] == -2)"],
        [0, 0xFFFFFFFE, 1],
    ),
    "byte_lane": (
        "Word value; value.word = 0x44332211u; value.bytes.z = 255u;",
        ["value.word", "value.bytes.x", "value.bytes.w"],
        [0x44FF2211, 17, 68],
    ),
    "bool_pack": (
        "Flags value; value.lanes = bool4(true, false, true, true); value.lanes[2] = false;",
        ["value.word", "uint(value.lanes.x)", "uint(value.lanes.z)"],
        [0x01000001, 1, 0],
    ),
    "bool_unpack": (
        "Flags value; value.word = 0x01000100u;",
        [
            "uint(value.lanes.x)",
            "uint(value.lanes.y)",
            "uint(value.lanes.z)",
            "uint(value.lanes.w)",
        ],
        [0, 1, 0, 1],
    ),
    "copy": (
        "Bits first = make_bits(0x04030201u); Bits second = first; first.words.x = 0u;",
        ["second.bytes[0].x", "second.words.y", "first.words.x"],
        [1, 0x88776655, 0],
    ),
    "assignment": (
        "Bits first = make_bits(0x04030201u); Bits second; second = first; first.words.y = 0u;",
        ["second.bytes[1].x", "second.words.y"],
        [85, 0x88776655],
    ),
    "helper_return": (
        "Bits first = make_bits(0x04030201u); Bits second = copy_bits(first);",
        ["second.words.y", "first.words.y", "second.bytes[0].z"],
        [0x0C0B0A09, 0x88776655, 3],
    ),
    "nested_selector": (
        "Holder holder; holder.values[0] = make_bits(0x04030201u); holder.values[1] = make_bits(7u); uint count = 0u; holder.values[next_index(count)].bytes[1] = uchar4(5u, 6u, 7u, 8u);",
        ["holder.values[0].words.y", "holder.values[1].words.y", "count"],
        [0x08070605, 0x88776655, 1],
    ),
    "lane_selector": (
        "Bits value = make_bits(0x04030201u); uint count = 0u; value.bytes[0][next_index(count)] = 255u;",
        ["value.words.x", "value.words.y", "count"],
        [0x040302FF, 0x88776655, 1],
    ),
    "word_compound": (
        "Bits value = make_bits(0x04030201u); value.words.x += 2u; value.words.y++;",
        ["value.bytes[0].x", "value.bytes[1].x"],
        [3, 86],
    ),
    "read_selector": (
        "Holder holder; holder.values[0] = make_bits(0x04030201u); uint count = 0u; uint selected = holder.values[next_index(count)].bytes[1][2];",
        ["selected", "count"],
        [119, 1],
    ),
}


def _case(root, target, case):
    body, expressions, values = CASES[case]
    writes = "\n".join(
        f"results[{index + 1}] = {expr};" for index, expr in enumerate(expressions)
    )
    source = (
        HEADER
        + f"\nkernel void union_values(device uint* results [[buffer(0)]]) {{\n{body}\n{writes}\n}}\n"
    )
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    count = len(values) + 2
    inputs = {"results": {"dtype": "uint32", "shape": [count], "values": [99] * count}}
    outputs = {
        "results": {"dtype": "uint32", "shape": [count], "values": [99, *values, 99]}
    }
    return (
        source,
        _request(descriptor, package, inputs, outputs, 1),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("case", CASES)
def test_union_storage_packages(tmp_path, target, case):
    _, request, _ = _case(tmp_path, target, case)
    if target == "opengl":
        generated = request.artifact_path.read_text()
        assert "uvec2 CrossGLUnionStorage;" in generated
        assert "uvec4 bytes[2];" not in generated


@pytest.mark.parametrize("case", CASES)
def test_union_storage_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native union storage")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="union_values",
    )


@pytest.mark.parametrize(
    ("declaration", "reason"),
    [
        ("union Bad { uint2 words; uint word; };", "member-size-mismatch"),
        (
            "union Bad { uint words[2]; uchar4 bytes[2]; };",
            "aggregate-alignment-unsupported",
        ),
        (
            "union Bad { uint3 words; uchar4 bytes[4]; };",
            "member-storage-shape-mismatch",
        ),
        (
            "struct Pair { uint2 words; }; union Bad { uint2 words; Pair pair; };",
            "member-shape-unsupported",
        ),
    ],
)
def test_union_storage_rejects_unrepresented_layouts(tmp_path, declaration, reason):
    source = tmp_path / "union.metal"
    source.write_text(declaration)
    with pytest.raises(OpenGLUnionLayoutError) as error:
        translate(
            str(source), backend="opengl", source_backend="metal", format_output=False
        )
    assert error.value.reason == reason
    assert error.value.missing_capabilities == ("opengl.union-storage-aliasing",)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("value.value += 1.0f;", "interpreted-assignment-value-unsupported"),
        (
            "results[0] = uint(value.value = 1.0f);",
            "interpreted-assignment-value-unsupported",
        ),
        ("value.value++;", "interpreted-address-or-update-unsupported"),
        (
            "value.bytes.xy = uchar2(1u, 2u);",
            "interpreted-subcomponent-write-unsupported",
        ),
        ("mutate(value.value);", "member-reference-unsupported"),
    ],
)
def test_union_storage_unsupported_mutations_report_diagnostics(tmp_path, body, reason):
    source = tmp_path / "union.metal"
    source.write_text(
        HEADER + "void mutate(thread float& arg) { arg = 2.0f; }\n"
        "kernel void run(device uint* results [[buffer(0)]]) { Word value; value.word = 0u; "
        + body
        + " }"
    )
    with pytest.raises(OpenGLUnionLayoutError) as error:
        translate(
            str(source), backend="opengl", source_backend="metal", format_output=False
        )
    assert error.value.reason == reason
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("union.metal",),
            targets=("opengl",),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert report["diagnostics"][0]["code"] == error.value.project_diagnostic_code
    assert report["diagnostics"][0]["missingCapabilities"] == list(
        error.value.missing_capabilities
    )


def test_union_storage_buffer_contract_is_not_silently_replaced(tmp_path):
    source = tmp_path / "union.metal"
    source.write_text(
        HEADER
        + "kernel void run(device Holder* output [[buffer(0)]]) { output[0].values[0].words = uint2(0u); }"
    )
    with pytest.raises(OpenGLUnionLayoutError, match="buffer reflection"):
        translate(
            str(source), backend="opengl", source_backend="metal", format_output=False
        )


def test_union_storage_overlapping_reference_arguments_fail_closed(tmp_path):
    source = tmp_path / "union.metal"
    source.write_text(
        HEADER
        + "void combine(thread Word& first, thread Word& second) { first.word += 1u; second.word += first.word; }\n"
        "kernel void run(device uint* results [[buffer(0)]]) { Word value; value.word = 3u; combine(value, value); results[0] = value.word; }"
    )
    with pytest.raises(OpenGLUnionLayoutError) as error:
        translate(
            str(source), backend="opengl", source_backend="metal", format_output=False
        )
    assert error.value.reason == "union-reference-unsupported"


@pytest.mark.parametrize("initializer", ["{}", "{uint2(1u, 2u)}"])
def test_union_storage_initializer_uses_one_physical_member(tmp_path, initializer):
    source = tmp_path / "union.metal"
    source.write_text(
        HEADER
        + "kernel void run(device uint* results [[buffer(0)]]) { Bits value = "
        + initializer
        + "; results[0] = value.words.x; }"
    )
    generated = translate(
        str(source), backend="opengl", source_backend="metal", format_output=False
    )
    assert "Bits(" in generated
    assert "uvec4 bytes[2]" not in generated


def test_union_storage_and_opengl_random_are_required_in_ci():
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    assert workflow.count(f'{REQUIRE_ENV}: "1"') == 3
    assert workflow.count("tests/test_translator/test_union_storage.py") == 3
    step = workflow.split("- name: Validate translated OpenGL random kernels\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "--target opengl --output-dir .mlx-gather-opengl/random" in step
    assert "set -euo pipefail" in step
    assert "--timeout-seconds 300" in step
    assert "continue-on-error" not in step
