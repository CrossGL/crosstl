"""Branch-declared counters, uninitialized locals and wrapping counter loops.

``uint start, end; if (mode > 0) { start = s[2 * g]; end = s[2 * g + 1]; }
else { ... }`` leaves two locals uninitialized until each branch writes them,
and ``if (c) { uint k = start + 8; for (; k <= end; k += 8) { ... } } else {
uint k = start + 4; for (...) }`` declares a counter per branch. The DirectX
and OpenGL software-subgroup proofs follow both: an uninitialized local is
uniform only once every path writes it uniformly, and each loop's counter is
the declaration visible where the loop starts. The bound ``end`` has no
range, so the loop need not terminate; a 32-bit unsigned counter wraps alike
in Metal, HLSL and GLSL, so every invocation still runs the same trips and
leaves the loop with the same counter.
"""

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator import parse
from crosstl.translator.codegen.counter_loops import (
    any_start_counter_loops,
    wrapping_counter_loop_uniform,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_metal_builtin_ownership import _compile

ENTRY = "segment_products"
METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;

kernel void segment_products(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device uint* segments [[buffer(2)]],
    constant int& mode [[buffer(3)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  simdgroup_float8x8 A, B, C;
  const device float* in = src + sg * 64;
  device float* out = dst + group.z * 128 + sg * 64;
  simdgroup_load(A, in, 8);
  B = make_filled_simdgroup_matrix<float, 8>(0.5f);
  B.thread_elements()[0] = float(int(lane % 3) - 1);
  C = make_filled_simdgroup_matrix<float, 8>(0.0f);
  uint start, end;
  if (mode > 0) {
    start = segments[2 * group.z];
    end = segments[2 * group.z + 1];
  } else {
    start = segments[group.z];
    end = segments[group.z + 1];
  }
  if (mode > 1) {
    uint k = start + 8;
    for (; k <= end; k += 8) {
      simdgroup_multiply_accumulate(C, A, B, C);
    }
    // The counter stays uniform after the loop.
    if (k > end + 4) {
      simdgroup_multiply_accumulate(C, A, A, C);
    }
  } else {
    uint k = start + 4;
    for (; k <= end; k += 4) {
      simdgroup_multiply_accumulate(C, B, A, C);
    }
  }
  simdgroup_store(C, out, 8);
}
"""
ERRORS = {
    "directx": "project.translate.directx-software-subgroup-invalid",
    "opengl": "project.translate.opengl-software-subgroup-invalid",
}
TARGETS = ("directx", "opengl")


def _translate(root, target, source=METAL_KERNEL):
    root.mkdir(parents=True, exist_ok=True)
    (root / "segments.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("segments.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(32, 2, 1),
            # Nothing writes the segments while the kernel runs.
            immutable_storage=[{"entryPoint": ENTRY, "parameter": "segments"}],
            source_options={
                "metal": {
                    "preserve_resource_origins": True,
                    "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                    "cooperative_matrix_fragment_mapping_provenance": (
                        "unit_lane_coordinates"
                    ),
                    "target_options": {
                        "directx": {
                            "cooperative_matrix_software_lowering": True,
                            "software_subgroup_width": 32,
                            "relative_wave_shuffle_out_of_range": "self",
                        },
                        "opengl": {
                            "cooperative_matrix_software_lowering": True,
                            "software_subgroup_width": 32,
                        },
                    },
                }
            },
        ),
        format_output=False,
    )
    return report.to_json()


def _rewritten(rewrites):
    source = METAL_KERNEL
    for old, new in rewrites:
        assert source.count(old) >= 1, old
        source = source.replace(old, new, 1)
    return source


ACCEPTED = {
    "branch-counters": [],
    # The branches' counters hide an outer, divergent one.
    "shadowed-outer": [("  if (mode > 1) {", "  uint k = lane;\n  if (mode > 1) {")],
    # One declaration before the branches serves both loops.
    "shared-counter": [
        (
            "  if (mode > 1) {\n    uint k = start + 8;",
            "  uint k = start + 8;\n  if (mode > 1) {",
        ),
        ("    uint k = start + 4;\n", ""),
    ],
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("variant", sorted(ACCEPTED))
def test_branch_declared_counters_translate(tmp_path, target, variant):
    report = _translate(tmp_path, target, _rewritten(ACCEPTED[variant]))
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    (artifact,) = report["artifacts"]
    text = (tmp_path / artifact["path"]).read_text()
    assert text.count("for (; ") == 2 and "simdgroup_" not in text
    _compile(text, target, tmp_path)


REJECTED = {
    # The else path never writes ``end``, so it is not uniform at the loops.
    "unwritten-path": [("    end = segments[group.z + 1];\n", "")],
    "divergent-write": [
        ("start = segments[group.z];", "start = segments[group.z] + lane % 2;")
    ],
    "divergent-guard": [("if (mode > 0) {", "if (mode > int(lane % 2)) {")],
    # A signed counter may overflow on an unranged bound, which Metal leaves
    # undefined.
    "signed-counter": [
        ("uint k = start + 8;", "int k = int(start) + 8;"),
        ("k <= end; k += 8", "k <= int(end); k += 8"),
    ],
    "narrow-counter": [
        ("uint k = start + 8;", "ushort k = ushort(start + 8);"),
        ("k <= end; k += 8", "k <= ushort(end); k += 8"),
    ],
    # Without its own declaration, the first loop counts the outer local.
    "outer-divergent-counter": [
        (
            "  if (mode > 1) {\n    uint k = start + 8;",
            "  uint k = start + lane % 2;\n  if (mode > 1) {",
        ),
    ],
    "divergent-after-loop": [
        ("    // The counter stays", "    k += lane % 2;\n    // The counter stays")
    ],
    "written-in-body": [
        (
            "simdgroup_multiply_accumulate(C, A, B, C);\n    }",
            "simdgroup_multiply_accumulate(C, A, B, C);\n      k += lane % 2;\n    }",
        )
    ],
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("variant", sorted(REJECTED))
def test_unproven_branch_counters_are_refused(tmp_path, target, variant):
    report = _translate(tmp_path, target, _rewritten(REJECTED[variant]))
    assert report["summary"]["translatedCount"] == 0, report
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes == {ERRORS[target]}, codes


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (("uint", 0, 16, "<="), True),
        (("uint32_t", 0, 16, "<"), True),
        (("uint", 3, 1, "<"), True),
        (("uint", 0, -1, ">"), True),
        # An always-true test runs forever in every invocation alike.
        (("uint", 0, -1, ">="), True),
        (("uint", 0, (1 << 32) - 1, "<"), True),
        (("uint", 0, 1 << 32, "<"), False),
        (("uint", -1, 1, "<"), False),
        (("uint", 0, 0, "<"), False),
        (("uint", 0, 1, "=="), False),
        (("uint", 0, 1, "!="), False),
        (("uint", 0, True, "<"), False),
        (("uint", 0, 1.0, "<"), False),
        (("int", 0, 1, "<"), False),
        (("int32_t", 0, 16, "<="), False),
        (("ushort", 0, 1, "<"), False),
        (("uint16_t", 0, 1, "<"), False),
        (("min16uint", 0, 1, "<"), False),
        (("ulong", 0, 1, "<"), False),
        (("uint2", 0, 1, "<"), False),
        (("not a type", 0, 1, "<"), False),
        ((None, 0, 1, "<"), False),
    ],
)
def test_wrapping_counter_contract(arguments, expected):
    assert wrapping_counter_loop_uniform(*arguments) is expected


DISCOVERY = """shader Counters {{
    compute {{
        void main(StructuredBuffer<uint> limits @register(t0),
                  RWStructuredBuffer<uint> results @register(u1)) {{
            uint total = 0u;
            {outer}
            if (limits[0] > 0u) {{
                {first}
                for (; k < limits[1]; k++) {{
                    total += k;
                }}
            }} else {{
                {second}
                for (; k < limits[2]; k += 2u) {{
                    total -= k;
                }}
            }}
            results[0] = total;
        }}
    }}
}}"""


def _discovered(outer="", first="uint k = limits[3];", second="uint k = 1u;"):
    ast = parse(DISCOVERY.format(outer=outer, first=first, second=second))
    ((_stage, function, _config),) = GLSLCodeGen().glsl_software_subgroup_stage_entries(
        ast
    )
    walk = HLSLCodeGen().walk_ast
    records = any_start_counter_loops(function, walk, lambda node: set())
    return sorted(
        (name, str(getattr(declaration.initial_value, "value", "")))
        for name, declaration, _loop in records.values()
    )


def test_discovery_pairs_each_loop_with_its_visible_declaration():
    assert len(_discovered()) == 2
    assert {name for name, _ in _discovered()} == {"k"}
    # One outer declaration serves both loops.
    shared = _discovered(outer="uint k = 5u;", first="", second="")
    assert shared == [("k", "5"), ("k", "5")]
    # A branch declaration hides the outer one only in its branch.
    mixed = _discovered(outer="uint k = 5u;", first="", second="uint k = 1u;")
    assert sorted(value for _, value in mixed) == ["1", "5"]


def test_discovery_refuses_uninitialized_redeclared_or_written_counters():
    # Without an initializer the first loop's counter has no start.
    assert len(_discovered(first="uint k;")) == 1
    # Two declarations in one statement list are ill-formed.
    assert _discovered(first="uint k = 1u; uint k = 2u;") == []
    # Any other write disqualifies the name in every scope.
    assert _discovered(outer="uint k = 5u; k = 6u;", first="", second="") == []
    # A loop without any visible declaration counts something else.
    assert len(_discovered(first="")) == 1
