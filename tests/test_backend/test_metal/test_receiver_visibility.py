"""The receiver declaration sweep answers exactly what the linear scan over a name's declarations answers."""

import random

from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    _MetalReceiverDeclaration,
)
from crosstl.backend.Metal.receiver_visibility import (
    ReceiverDeclarationSweep,
    visible_receiver_declaration,
)


def _scan(candidates, position):
    """The linear scan the preprocessor used before the sweep."""
    declaration = None
    for candidate in candidates:
        if candidate.declaration_end > position:
            break
        if not (candidate.scope_start <= position < candidate.scope_end):
            continue
        declaration = candidate
    return declaration


def _declaration(position, end, scope_start, scope_end, declared_type="S"):
    return _MetalReceiverDeclaration(
        name="r",
        declaration_position=position,
        declaration_end=end,
        type_position=position,
        scope_start=scope_start,
        scope_end=scope_end,
        declared_type=declared_type,
        raw_type_text=declared_type,
    )


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


def _random_declarations(rng, length, irregular):
    scopes = _nested_scopes(rng, 0, length, 4)
    if irregular:
        # Scopes that neither nest nor exclude each other.
        for _ in range(6):
            start = rng.randrange(0, length - 2)
            scopes.append((start, rng.randrange(start + 1, length)))
    declarations = []
    for _ in range(rng.randrange(20, 120)):
        start, end = rng.choice(scopes)
        position = rng.randrange(start, end)
        if irregular and rng.random() < 0.1:
            # A parameter: declared before the body that is its scope.
            position = max(0, start - rng.randrange(1, 6))
        declarations.append(
            _declaration(position, position + rng.randrange(1, 8), start, end)
        )
    declarations.sort(key=lambda item: item.declaration_end)
    if irregular:
        # Contextual parameters are inserted ahead of the scanned declarations, out of order.
        for _ in range(rng.randrange(0, 3)):
            start, end = rng.choice(scopes)
            declarations.insert(0, _declaration(-1, 0, start, end))
        if rng.random() < 0.3 and len(declarations) > 4:
            index = rng.randrange(1, len(declarations) - 1)
            declarations[index], declarations[index + 1] = (
                declarations[index + 1],
                declarations[index],
            )
    return declarations


def _assert_same(candidates, positions):
    sweep = ReceiverDeclarationSweep(candidates)
    for position in positions:
        assert sweep.visible_at(position) is _scan(candidates, position), position


def test_sweep_matches_the_scan_for_nested_scopes_in_increasing_order():
    rng = random.Random(11)
    for _ in range(60):
        candidates = _random_declarations(rng, 400, irregular=False)
        _assert_same(candidates, sorted(rng.randrange(-5, 410) for _ in range(150)))


def test_sweep_matches_the_scan_for_irregular_scopes_and_lists():
    rng = random.Random(12)
    for _ in range(60):
        candidates = _random_declarations(rng, 400, irregular=True)
        _assert_same(candidates, sorted(rng.randrange(-5, 410) for _ in range(150)))


def test_sweep_restarts_for_earlier_positions():
    rng = random.Random(13)
    for _ in range(40):
        candidates = _random_declarations(rng, 300, irregular=rng.random() < 0.5)
        _assert_same(candidates, [rng.randrange(-5, 310) for _ in range(150)])


def test_a_grown_or_replaced_list_starts_a_new_sweep():
    outer = _declaration(5, 10, 0, 100)
    inner = _declaration(20, 25, 15, 40)
    candidates = [outer, inner]
    sweeps = {}
    assert visible_receiver_declaration(sweeps, candidates, "r", 30) is inner
    later = _declaration(50, 55, 45, 90)
    candidates.append(later)
    assert visible_receiver_declaration(sweeps, candidates, "r", 60) is later
    replacement = [_declaration(1, 2, 0, 100, "T")]
    assert visible_receiver_declaration(sweeps, replacement, "r", 60) is replacement[0]
    assert visible_receiver_declaration(sweeps, candidates, "r", 95) is outer


def test_the_preprocessor_resolves_the_receiver_declared_in_each_instantiation():
    source = """
struct Loader { float value; float get() const { return value; } };
kernel void first(device float* out [[buffer(0)]]) {
    Loader loader{1.0f};
    out[0] = loader.get();
}
kernel void second(device float* out [[buffer(0)]]) {
    { Loader loader{2.0f}; out[1] = loader.get(); }
    Loader loader{3.0f};
    out[2] = loader.get();
}
"""
    preprocessor = MetalPreprocessor()
    declarations = preprocessor._find_receiver_declarations(source).get("loader")
    assert declarations and len(declarations) == 3
    calls = [
        index for index in range(len(source)) if source.startswith("loader.get", index)
    ]
    found = [
        preprocessor._visible_receiver_declaration_at(source, "loader", call)
        for call in calls
    ]
    assert found == [_scan(declarations, call) for call in calls]
    assert [item.declaration_position for item in found] == [
        item.declaration_position for item in declarations
    ]
