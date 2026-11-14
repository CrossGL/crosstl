"""The sweep index for local integral constant bindings answers exactly what the linear scan answers."""

import random

from crosstl.backend.Metal.integral_binding_visibility import (
    IntegralBindingTable,
    indexed_visible_bindings,
)
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    _MetalIntegralConstantBinding,
)

scan = MetalPreprocessor._local_integral_constant_bindings_at


def _binding(declaration, start, end, value=None):
    return _MetalIntegralConstantBinding(
        declaration_position=declaration,
        scope_start=start,
        scope_end=end,
        value=value,
    )


def _plain(table):
    """The same bindings as a plain mapping of lists, which the preprocessor scans linearly."""
    return {name: list(bindings) for name, bindings in table.items()}


def _nested_scopes(rng, start, end, depth):
    scopes = [(start, end)]
    if depth == 0 or end - start < 8:
        return scopes
    cursor = start + 1
    while cursor < end - 4 and rng.random() < 0.8:
        child_start = rng.randrange(cursor, end - 3)
        child_end = rng.randrange(child_start + 2, min(end, child_start + 60) + 1)
        scopes += _nested_scopes(rng, child_start, child_end, depth - 1)
        cursor = child_end + 1
    return scopes


def _random_table(rng, length, overlapping):
    scopes = _nested_scopes(rng, 0, length, 4)
    if overlapping:
        # Scopes that neither nest nor exclude each other, and scopes that open after their declaration.
        for _ in range(6):
            start = rng.randrange(0, length - 2)
            scopes.append((start, rng.randrange(start + 1, length)))
    declarations = []
    for _ in range(rng.randrange(40, 160)):
        start, end = rng.choice(scopes)
        declaration = rng.randrange(start, end)
        if overlapping and rng.random() < 0.1:
            declaration = max(0, start - rng.randrange(1, 6))
        name = rng.choice("abcdefgh")
        declarations.append((declaration, name, start, end, f"{declaration}{name}"))
    declarations.sort(key=lambda item: item[0])
    table = IntegralBindingTable()
    for declaration, name, start, end, value in declarations:
        table.add(name, _binding(declaration, start, end, value))
    return table


def _assert_same(table, positions, names=None):
    reference = _plain(table)
    for position in positions:
        indexed = table.visible_at(position, names)
        assert indexed == scan(reference, position, names=names), position
        assert list(indexed) == list(scan(reference, position, names=names))


def test_sweep_matches_the_scan_for_nested_scopes_in_increasing_order():
    rng = random.Random(7)
    for _ in range(40):
        table = _random_table(rng, 400, overlapping=False)
        _assert_same(table, sorted(rng.randrange(-5, 410) for _ in range(120)))


def test_sweep_matches_the_scan_when_positions_go_back():
    rng = random.Random(11)
    for _ in range(40):
        table = _random_table(rng, 300, overlapping=False)
        _assert_same(table, [rng.randrange(0, 300) for _ in range(80)])


def test_sweep_matches_the_scan_for_overlapping_and_late_opening_scopes():
    rng = random.Random(13)
    for _ in range(60):
        table = _random_table(rng, 200, overlapping=True)
        _assert_same(table, [rng.randrange(0, 200) for _ in range(60)])
        _assert_same(table, sorted(rng.randrange(0, 200) for _ in range(60)))


def test_restricted_names_are_answered_in_sorted_order():
    rng = random.Random(17)
    for _ in range(30):
        table = _random_table(rng, 300, overlapping=False)
        _assert_same(
            table,
            sorted(rng.randrange(0, 300) for _ in range(40)),
            {"h", "c", "a", "z"},
        )


def test_equal_declarations_resolve_to_the_first_binding():
    table = IntegralBindingTable()
    table.add("n", _binding(5, 0, 50, "first"))
    table.add("n", _binding(5, 0, 50, "second"))
    table.add("n", _binding(5, 6, 50, "late"))
    assert table.visible_at(10)["n"].value == "first"
    assert scan(_plain(table), 10)["n"].value == "first"


def test_inner_bindings_shadow_until_their_scope_closes():
    table = IntegralBindingTable()
    table.add("n", _binding(2, 0, 100, "outer"))
    table.add("n", _binding(12, 10, 20, "inner"))
    table.add("m", _binding(15, 10, 20, "inner-only"))
    assert {name: b.value for name, b in table.visible_at(1).items()} == {}
    assert {name: b.value for name, b in table.visible_at(15).items()} == {
        "n": "inner",
        "m": "inner-only",
    }
    assert {name: b.value for name, b in table.visible_at(20).items()} == {"n": "outer"}
    # Going back restarts the sweep; ``m`` is declared after this position.
    assert {name: b.value for name, b in table.visible_at(13).items()} == {"n": "inner"}


def test_bindings_added_between_queries_become_visible():
    table = IntegralBindingTable()
    table.add("a", _binding(0, 0, 100, "a"))
    assert list(table.visible_at(10)) == ["a"]
    table.add("b", _binding(10, 0, 100, "b"))
    table.add("a", _binding(20, 15, 40, "a2"))
    assert {name: b.value for name, b in table.visible_at(10).items()} == {
        "a": "a",
        "b": "b",
    }
    assert {name: b.value for name, b in table.visible_at(30).items()} == {
        "a": "a2",
        "b": "b",
    }


def test_out_of_order_additions_fall_back_to_the_scan():
    table = IntegralBindingTable()
    table.add("a", _binding(10, 0, 100, "late"))
    table.add("a", _binding(5, 0, 100, "early"))
    assert table.visible_at(50) is None
    # The scan stops at the first binding declared after the position, as before.
    assert scan(table, 7) == scan(_plain(table), 7) == {}
    assert scan(table, 50)["a"].value == "late"


def test_direct_edits_fall_back_to_the_scan():
    table = IntegralBindingTable()
    table.add("a", _binding(0, 0, 100, "a"))
    assert table.visible_at(10) is not None
    table["b"] = [_binding(1, 0, 100, "b")]
    assert table.visible_at(10) is None
    assert {name: b.value for name, b in scan(table, 10).items()} == {
        "a": "a",
        "b": "b",
    }


def test_plain_mappings_are_not_indexed():
    assert indexed_visible_bindings({"a": [_binding(0, 0, 9)]}, 3) is None


def test_collected_bindings_are_indexed():
    code = """
kernel void k(device int* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    constexpr int n = 4;
    {
        constexpr int n = 8;
        constexpr int m = n * 2;
        out[tid] = m;
    }
    out[tid] += n;
}
"""
    preprocessor = MetalPreprocessor()
    body = code.index("{")
    end = code.rindex("}") + 1
    bindings = preprocessor._collect_local_integral_constant_bindings(
        code, [(body, end)]
    )
    assert isinstance(bindings, IntegralBindingTable)
    inner = code.index("out[tid] = m")
    outer = code.index("out[tid] += n")
    assert preprocessor._local_integral_constants_at(bindings, inner) == scan_values(
        _plain(bindings), inner
    )
    assert preprocessor._local_integral_constants_at(bindings, inner)["m"] == "16"
    assert preprocessor._local_integral_constants_at(bindings, outer)["n"] == "4"


def scan_values(bindings, position):
    return {
        name: binding.value
        for name, binding in scan(bindings, position).items()
        if binding.value is not None
    }
