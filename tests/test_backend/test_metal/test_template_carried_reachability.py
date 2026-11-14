"""Reachability through template bodies, as the constrained call pass sees it."""

from types import SimpleNamespace

from crosstl.backend.Metal.template_carried_reachability import (
    template_carried_reachable_spans,
)


def _function(code, name):
    start = code.index(f"void {name}(")
    body_start = code.index("{", start)
    end = code.index("}", body_start) + 1
    return SimpleNamespace(
        name=name, span=(start, end), body_span=(body_start + 1, end - 1)
    )


def _template(code, name):
    start = code.index(f"template <typename T> void {name}(")
    body_start = code.index("{", start) + 1
    end = code.index("}", body_start) + 1
    return SimpleNamespace(name=name, span=(start, end), body_start=body_start)


CODE = """
void h() { }
void g() { }
void unused() { }
template <typename T> void inner(T a) { h(); }
template <typename T> void outer(T a) { inner<T>(a); }
template <typename T> void never(T a) { unused(); }
void entry() { outer<float>(1.0f); x.g(); }
"""


def test_functions_reached_through_nested_templates_are_reached():
    functions = [_function(CODE, name) for name in ("h", "g", "unused", "entry")]
    templates = [_template(CODE, name) for name in ("inner", "outer", "never")]
    entry = functions[-1]

    spans = template_carried_reachable_spans(CODE, [entry.span], functions, templates)

    assert spans == [functions[0].span, entry.span]


def test_nothing_is_added_without_template_calls():
    functions = [_function(CODE, name) for name in ("h", "g", "unused", "entry")]

    spans = template_carried_reachable_spans(CODE, [functions[1].span], functions, [])

    assert spans == [functions[1].span]
