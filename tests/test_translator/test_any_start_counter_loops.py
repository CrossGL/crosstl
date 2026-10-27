"""Initializer-free counter loops entered with a uniform but unknown counter.

``int k = T; if (mode > 0) { for (; k > 0; k--) { ... } } else { ... }``
reaches each loop with whatever value the counter holds. The DirectX and
OpenGL software subgroup proofs accept such a loop when the flow analysis
proves the counter uniform at the loop and the loop terminates for every
value of its type.
"""

import os

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator import parse
from crosstl.translator.codegen.array_utils import _UnsignedLiteralInt
from crosstl.translator.codegen.counter_loops import (
    any_start_counter_loop_terminates,
    any_start_counter_loops,
    counter_update,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1
RANGES = {
    "short": (-(1 << 15), (1 << 15) - 1),
    "ushort": (0, (1 << 16) - 1),
    "int": (INT_MIN, INT_MAX),
    "uint": (0, (1 << 32) - 1),
}
COMPARE = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _terminates_from(type_name, start, stop, step, operator):
    """Run the loop arithmetically: jump to the last admitted counter."""
    low, high = RANGES[type_name]
    test = COMPARE[operator]
    value = start
    if test(value, stop):
        if step > 0:
            limit = stop - (operator == "<")
            trips = (limit - value) // step + 1
        else:
            limit = stop + (operator == ">")
            trips = (value - limit) // -step + 1
        value += trips * step
        if test(value, stop):
            raise AssertionError("trip count is wrong")
    return low <= value <= high


def _searched(type_name, stops, step, operator):
    low, high = RANGES[type_name]
    starts = {low, high, 0}
    for stop in stops:
        for center in (stop, stop - 1, stop + 1):
            starts.update(range(center - 12, center + 13))
    starts = sorted(value for value in starts if low <= value <= high)
    return all(
        _terminates_from(type_name, start, stop, step, operator)
        for stop in stops
        for start in starts
    )


@pytest.mark.parametrize("type_name", sorted(RANGES))
@pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
def test_any_start_termination_matches_every_start(type_name, operator):
    low, high = RANGES[type_name]
    ascending = operator in {"<", "<="}
    accepted = rejected = 0
    for magnitude in (1, 2, 8, 9):
        step = magnitude if ascending else -magnitude
        for first in (low, low + 1, low + 7, 0, 1, 5, high - 8, high - 1, high):
            stops = [first, min(high, first + 2)]
            expected = _searched(type_name, stops, step, operator)
            actual = any_start_counter_loop_terminates(
                type_name, 0, stops, step, operator
            )
            assert actual is expected, (type_name, operator, step, first)
            accepted += expected
            rejected += not expected
    assert accepted > 10 and rejected > 5


@pytest.mark.parametrize(
    "arguments, expected",
    [
        # The descending MLX loop: ``for (; k > 0; k--)`` from any int.
        (("int", 0, [0], -1, ">"), True),
        (("uint", 0, [_UnsignedLiteralInt(0)], -1, ">"), True),
        # ``k >= 0`` on uint is always true and wraps.
        (("uint", 0, [_UnsignedLiteralInt(0)], -1, ">="), False),
        (("int", 0, [INT_MIN], -1, ">="), False),
        (("int", 0, [INT_MIN], -1, ">"), True),
        (("int", 0, [0, INT_MAX], 1, "<"), True),
        (("int", 0, [0, INT_MAX], 1, "<="), False),
        (("int", 0, [0, INT_MAX - 7], 8, "<"), True),
        (("int", 0, [0, INT_MAX - 6], 8, "<"), False),
        # An unknown start could overflow ``k + offset`` on the first test.
        (("int", 8, [0, 64], 8, "<="), False),
        (("int", -1, [0], -1, ">"), False),
        # The direction must match the comparison.
        (("int", 0, [0], 1, ">"), False),
        (("int", 0, [64], -1, "<"), False),
        (("int", 0, [0], 0, ">"), False),
        (("int", 0, [0], -1, "=="), False),
        # A negative signed counter compares as a huge unsigned value.
        (("int", 0, [_UnsignedLiteralInt(0)], -1, ">"), False),
        (("uint", 0, [-1], -1, ">"), False),
        (("int", 0, [], -1, ">"), False),
        (("int", 0, [0.0], -1, ">"), False),
        (("int", True, [0], -1, ">"), False),
        (("long", 0, [0], -1, ">"), False),
        (("int2", 0, [0], -1, ">"), False),
        (("not a type", 0, [0], -1, ">"), False),
    ],
)
def test_any_start_termination_contract(arguments, expected):
    assert any_start_counter_loop_terminates(*arguments) is expected


SHADER = """shader Counter {{
    {helpers}
    compute {{
        void main(StructuredBuffer<int> limits @register(t0),
                  RWStructuredBuffer<int> results @register(u1)) {{
            int total = 0;
            int k = limits[0];
            {before}
            if (limits[1] > 0) {{
                for (; k > 0; k--) {{
                    total += k;
                }}
            }} else {{
                for (; k > 0; k -= 2) {{
                    total -= k;
                }}
            }}
            {after}
            results[0] = total + k;
        }}
    }}
}}"""


def _entry(ast):
    ((_stage, function, _config),) = GLSLCodeGen().glsl_software_subgroup_stage_entries(
        ast
    )
    return function


def _loops(writes=None, *, helpers="", before="", after=""):
    ast = parse(SHADER.format(helpers=helpers, before=before, after=after))
    walk = HLSLCodeGen().walk_ast
    if writes is None:

        def writes(node):
            names = set()
            roots = (
                [node.initial_value]
                if hasattr(node, "initial_value")
                else list(node.arguments or [])
            )
            for root in roots:
                names |= {
                    child.name
                    for child in walk(root)
                    if type(child).__name__ == "IdentifierNode"
                }
            return names

    return any_start_counter_loops(_entry(ast), walk, writes)


def test_discovery_finds_both_branch_loops():
    records = _loops()
    assert sorted(name for name, _declaration, _loop in records.values()) == ["k", "k"]
    assert {loop.init for _name, _declaration, loop in records.values()} == {None}
    assert _loops(after="results[1] = k;") != {}


@pytest.mark.parametrize(
    "fields",
    [
        dict(after="k = 3;"),
        dict(before="k += 1;"),
        dict(after="k--;"),
        dict(before="int& alias = k;"),
        dict(after="int k = 4;"),
        dict(
            helpers="void bump(inout int value) { value += 1; }",
            after="bump(k);",
        ),
    ],
)
def test_discovery_rejects_other_writes_and_aliases(fields):
    assert _loops(**fields) == {}


def test_discovery_skips_loops_inside_other_loops():
    source = SHADER.format(helpers="", before="", after="").replace(
        "if (limits[1] > 0) {", "for (int r = 0; r < 2; r++) if (limits[1] > 0) {", 1
    )
    walk = HLSLCodeGen().walk_ast
    assert (
        any_start_counter_loops(_entry(parse(source)), walk, lambda node: set()) == {}
    )


def test_counter_update_reads_both_directions():
    ast = parse(SHADER.format(helpers="", before="", after=""))
    walk = HLSLCodeGen().walk_ast
    updates = sorted(
        (counter_update(loop.update)[:2])
        for _name, _declaration, loop in any_start_counter_loops(
            _entry(ast), walk, lambda node: set()
        ).values()
    )
    assert updates == [("k", -1), ("k", -1)]


# --- DirectX software subgroups ---------------------------------------------

METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;

kernel void counted_products(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& T [[buffer(2)]],
    constant int& mode [[buffer(3)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 A, B, C;
  const device float* in = src + sg * 64;
  device float* out = dst + sg * 64;
  simdgroup_load(A, in, 8);
  B = make_filled_simdgroup_matrix<float, 8>(0.5f);
  B.thread_elements()[0] = float(int(lane % 3) - 1);
  C = make_filled_simdgroup_matrix<float, 8>(0.0f);
  int k = T;
  if (mode > 0) {
    for (; k > 0; k--) {
      simdgroup_multiply_accumulate(C, A, B, C);
    }
  } else {
    for (; k > 0; k--) {
      simdgroup_multiply_accumulate(C, B, A, C);
    }
  }
  simdgroup_store(C, out, 8);
}
"""
DIRECTX_ERROR = "project.translate.directx-software-subgroup-invalid"
OPENGL_ERROR = "project.translate.opengl-software-subgroup-invalid"


def _translate(root, source=METAL_KERNEL, target="directx"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "counted.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("counted.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(32, 2, 1),
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


def test_directx_accepts_descending_loops_from_a_uniform_counter(tmp_path):
    report = _translate(tmp_path)
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    (artifact,) = report["artifacts"]
    text = (tmp_path / artifact["path"]).read_text()
    assert text.count("for (; ") == 2 and "simdgroup_" not in text
    if os.environ.get("CROSTL_SKIP_NATIVE_VALIDATORS") != "1":
        from tests.test_translator.test_directx_float_atomics import _compile

        work = tmp_path / "validation"
        work.mkdir()
        _compile(text, work, profile="cs_6_2")


REJECTED_REWRITES = [
    [("int k = T;", "int k = T + int(lane % 2);")],
    [("if (mode > 0) {", "if (mode > int(lane % 2)) {")],
    [
        (
            "simdgroup_multiply_accumulate(C, A, B, C);\n    }",
            "simdgroup_multiply_accumulate(C, A, B, C);\n      k -= int(lane % 2);\n    }",
        )
    ],
    [
        (
            "simdgroup_store(C, out, 8);",
            "k = int(lane);\n  simdgroup_store(C, out, 8);",
        )
    ],
    # A signed counter with an unranged bound may overflow, which Metal
    # leaves undefined; 32-bit unsigned counters wrap alike everywhere
    # (test_branch_declared_counters.py).
    [("for (; k > 0; k--)", "for (; k < mode; k += 2)")],
    [
        ("int k = T;", "int k = T;\n  for (int r = 0; r < 2; r++) {"),
        ("  simdgroup_store", "  }\n  simdgroup_store"),
    ],
]
REJECTED_IDS = [
    "divergent-start",
    "divergent-branch",
    "written-in-body",
    "written-elsewhere",
    "signed-unranged-bound",
    "inside-another-loop",
]


def _rewritten(rewrites):
    source = METAL_KERNEL
    for old, new in rewrites:
        assert old in source, old
        source = source.replace(old, new, 1)
    return source


@pytest.mark.parametrize("rewrites", REJECTED_REWRITES, ids=REJECTED_IDS)
def test_directx_rejects_unproven_any_start_loops(tmp_path, rewrites):
    report = _translate(tmp_path, _rewritten(rewrites))
    assert report["summary"]["translatedCount"] == 0, report
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes == {DIRECTX_ERROR}, codes


# --- OpenGL software subgroups ----------------------------------------------


def test_opengl_accepts_descending_loops_from_a_uniform_counter(tmp_path):
    report = _translate(tmp_path, target="opengl")
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    (artifact,) = report["artifacts"]
    text = (tmp_path / artifact["path"]).read_text()
    assert text.count("for (; ") == 2 and "simdgroup_" not in text
    from tests.test_translator.test_metal_builtin_ownership import _compile

    _compile(text, "opengl", tmp_path)


@pytest.mark.parametrize("rewrites", REJECTED_REWRITES, ids=REJECTED_IDS)
def test_opengl_rejects_unproven_any_start_loops(tmp_path, rewrites):
    report = _translate(tmp_path, _rewritten(rewrites), target="opengl")
    assert report["summary"]["translatedCount"] == 0, report
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes == {OPENGL_ERROR}, codes
