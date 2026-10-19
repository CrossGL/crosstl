"""Unsigned wrap is exact modulo the destination width of a stored integer."""

from dataclasses import replace

import pytest

from crosstl.project import translate_project
from crosstl.translator.ast import (
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    TernaryOpNode,
    UnaryOpNode,
)
from crosstl.translator.codegen.congruent_intervals import congruent_ring_interval
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from tests.test_translator.test_project_metadata_entry_scope import _config

UINT32 = (0, (1 << 32) - 1)
INT32 = (-(1 << 31), (1 << 31) - 1)
UINT16 = (0, (1 << 16) - 1)


class _Domain:
    def __init__(self, bounds):
        self.minimum, self.maximum = bounds


def _interval(expression, facts, domains):
    """Mathematical bounds and declared domains for synthetic expressions."""

    def bounds(node):
        if isinstance(node, IdentifierNode):
            return facts.get(node.name)
        if isinstance(node, UnaryOpNode):
            inner = bounds(node.operand)
            return None if inner is None else (-inner[1], -inner[0])
        if isinstance(node, TernaryOpNode):
            left, right = bounds(node.true_expr), bounds(node.false_expr)
            return min(left[0], right[0]), max(left[1], right[1])
        if not isinstance(node, BinaryOpNode):
            return None
        left, right = bounds(node.left), bounds(node.right)
        if left is None or right is None:
            return None
        if node.op == "+":
            return left[0] + right[0], left[1] + right[1]
        if node.op == "-":
            return left[0] - right[1], left[1] - right[0]
        if node.op == "*":
            products = [a * b for a in left for b in right]
            return min(products), max(products)
        if node.op == "/" and right[0] > 0:
            return left[0] // right[1], left[1] // right[0]
        return None

    def domain(node):
        value = domains.get(id(node))
        return None if value is None else _Domain(value)

    def walk(node):
        yield node
        for child in ("left", "right", "operand", "condition", "true_expr"):
            value = getattr(node, child, None)
            if value is not None:
                yield from walk(value)
        if getattr(node, "false_expr", None) is not None:
            yield from walk(node.false_expr)
        for argument in getattr(node, "args", None) or ():
            yield from walk(argument)

    return lambda bits: congruent_ring_interval(
        expression,
        bits,
        bounds=bounds,
        domain=domain,
        operator=lambda node: node.op,
        walk=walk,
    )


def _difference(domain=UINT32):
    left, right = IdentifierNode("count"), IdentifierNode("lane")
    node = BinaryOpNode(left, "-", right)
    return node, {id(node): domain}


def test_unsigned_subtraction_keeps_its_signed_representative():
    node, domains = _difference()
    interval = _interval(node, {"count": (-121, 7), "lane": (0, 124)}, domains)
    assert interval(32) == (-245, 7)
    assert interval(16) == (-245, 7)


@pytest.mark.parametrize(
    "domain, bits",
    [
        (INT32, 32),  # signed overflow is undefined, not modular
        (UINT16, 32),  # a narrower wrap loses the destination's high bits
    ],
)
def test_wrap_must_be_unsigned_and_at_least_destination_width(domain, bits):
    node, domains = _difference(domain)
    facts = {"count": (domain[0], 7), "lane": (0, 124)}
    assert _interval(node, facts, domains)(bits) is None


def test_non_ring_operations_cannot_consume_a_wrapped_value():
    difference, domains = _difference()
    divisor = IdentifierNode("two")
    quotient = BinaryOpNode(difference, "/", divisor)
    domains[id(quotient)] = UINT32
    facts = {"count": (-121, 7), "lane": (0, 124), "two": (2, 2)}
    assert _interval(quotient, facts, domains)(32) is None
    product = BinaryOpNode(difference, "*", divisor)
    domains[id(product)] = UINT32
    assert _interval(product, facts, domains)(32) == (-490, 14)


def test_wrapped_values_inside_calls_are_rejected():
    difference, domains = _difference()
    call = FunctionCallNode("helper", [difference])
    outer = BinaryOpNode(call, "+", IdentifierNode("lane"))
    domains[id(outer)] = UINT32
    facts = {"count": (-121, 7), "lane": (0, 124)}
    assert _interval(outer, facts, domains)(32) is None


def test_select_forwards_wrapped_operands_at_its_common_width():
    difference, domains = _difference()
    select = TernaryOpNode(IdentifierNode("flag"), IdentifierNode("count"), difference)
    domains[id(select)] = UINT32
    facts = {"count": (-121, 7), "lane": (0, 124), "flag": (0, 1)}
    assert _interval(select, facts, domains)(32) == (-245, 7)
    domains[id(select)] = UINT16
    assert _interval(select, facts, domains)(32) is None


def test_negation_and_exact_values_are_unchanged():
    lane = IdentifierNode("lane")
    negated = UnaryOpNode("-", lane)
    domains = {id(negated): UINT32}
    assert _interval(negated, {"lane": (0, 124)}, domains)(32) == (-124, 0)
    assert _interval(lane, {"lane": (0, 124)}, domains)(32) is None


TAIL = """#include <metal_stdlib>
using namespace metal;
template <typename T, typename IdxT = int64_t, int N_READS = 4>
[[kernel]] void tail_sum(const device T* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         const constant size_t& in_size [[buffer(2)]],
                         const constant size_t& row_size [[buffer(3)]],
                         uint3 gid [[threadgroup_position_in_grid]],
                         uint3 lid [[thread_position_in_threadgroup]],
                         uint3 lsize [[threads_per_threadgroup]]) {
  float total = 0.0f;
  IdxT start_idx = gid.y * IdxT(row_size);
  IdxT actual_row =
      (start_idx + row_size <= in_size) ? row_size : in_size - start_idx;
  IdxT blocks = actual_row / (lsize.x * N_READS);
  int extra = actual_row - blocks * (lsize.x * N_READS);
  extra -= lid.x * N_READS;
  start_idx += lid.x * N_READS;
  in += start_idx;
  if (extra >= N_READS) { blocks++; extra = 0; }
  for (IdxT b = 0; b < blocks; b++) {
    for (int i = 0; i < N_READS; i++) { total += float(in[i]); }
    in += lsize.x * N_READS;
  }
  if (extra > 0) {
    for (int i = 0; i < extra; i++) { total += float(in[i]); }
  }
  out[gid.y * lsize.x + lid.x] = total;
}
template [[host_name("tail_sum_f")]] [[kernel]] void tail_sum<float>(
    const device float*, device float*, const constant size_t&,
    const constant size_t&, uint3, uint3, uint3);
"""


def _tail_config(root, source=TAIL, *, count=5000, rows=128):
    row_size = (count + rows - 1) // rows
    config = _config(
        root,
        "opengl",
        source=source,
        entries=("tail_sum_f",),
        conditions=[
            dict(entryPoint="tail_sum_f", parameter=name, minimum=value, maximum=value)
            for name, value in (("in_size", count), ("row_size", row_size))
        ],
    )
    (region,) = plan_dispatch_regions((32, rows, 1), (32, 1, 1))
    metal = dict(config.source_options["metal"])
    metal["target_options"] = {"opengl": {"dispatch_region": region.to_json()}}
    return replace(config, workgroup_size=(32, 1, 1), source_options={"metal": metal})


@pytest.mark.parametrize("count, rows", [(7, 1), (5000, 128)])
def test_unsigned_remainders_bound_checked_tail_reads(tmp_path, count, rows):
    config = _tail_config(tmp_path, count=count, rows=rows)
    report = translate_project(config, format_output=False).to_json()
    assert not config.index_range_assertions
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]


def test_a_divided_wrapped_remainder_still_fails_closed(tmp_path):
    source = TAIL.replace(
        "in_size - start_idx;", "(in_size - start_idx) / 2u;"
    ).replace("IdxT(row_size);", "IdxT(row_size) + 64;")
    report = translate_project(
        _tail_config(tmp_path, source), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )
