"""Struct aliases that select a type with ``conditional_t``.

A materialized template may name a helper struct through
``using L = conditional_t<C, A<...>, B<...>>;`` and then call its methods.
Once ``C`` folds, the alias names one concrete struct and member calls on a
variable of type ``L`` lower to that struct's methods. A condition that does
not fold keeps the alias unresolved, so the call still fails closed.
"""

import re

import pytest

from crosstl.backend.Metal.conditional_types import (
    select_conditional_type,
    split_template_arguments,
)
from crosstl.project import ProjectConfig, translate_project
from tests.test_translator.test_metal_builtin_ownership import _compile


def _fold(values):
    def evaluate(condition):
        condition = condition.strip()
        if condition in values:
            return True, int(values[condition])
        return False, None

    return evaluate


@pytest.mark.parametrize(
    "text,expected",
    [
        ("conditional_t<flag, A<T, 2>, B<T, 2>>", "A<T, 2>"),
        ("metal::conditional_t<other, A, B>", "B"),
        ("typename metal::conditional<flag, A, B>::type", "A"),
        ("conditional_t<flag, conditional_t<other, A, B>, C>", "B"),
        ("conditional_t<other, A, typename conditional<flag, B, C>::type>", "B"),
        ("conditional_t<flag, A<(4 > 2)>, B>", "A<(4 > 2)>"),
    ],
)
def test_select_conditional_type_follows_folded_conditions(text, expected):
    assert select_conditional_type(text, _fold({"flag": 1, "other": 0})) == expected


@pytest.mark.parametrize(
    "text",
    [
        # The condition does not fold.
        "conditional_t<unknown, A, B>",
        "conditional_t<flag, conditional_t<unknown, A, B>, C>",
        # Not a selection, or not a complete one.
        "A<flag, B, C>",
        "metal::conditional<flag, A, B>",
        "conditional_t<flag, A>",
        "conditional_t<flag, A, B>::type",
        "conditional_t<flag, A, B",
    ],
)
def test_select_conditional_type_rejects_unresolved_spellings(text):
    assert select_conditional_type(text, _fold({"flag": 1})) is None


def test_split_template_arguments_keeps_nested_lists_and_operators():
    assert split_template_arguments(
        "a && (N < 4), C<D, E<F>>, (x, y), G<1 >= 0>, 2 < 3"
    ) == ["a && (N < 4)", "C<D, E<F>>", "(x, y)", "G<1 >= 0>", "2 < 3"]
    assert split_template_arguments("A<B, (C>") is None
    assert split_template_arguments("A<B, C") is None


SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T, int N>
struct NarrowLoader {
  T scale;
  T apply(T v) const { return v * scale + T(N); }
};

template <typename T, int N>
struct WideLoader {
  T scale;
  T apply(T v) const { return v * scale - T(N); }
};

template <typename T, int N, bool narrow>
[[kernel]] void run(device T* out [[buffer(0)]],
                    uint i [[thread_position_in_grid]]) {
  using loader_t = SELECT;
  thread loader_t loader{T(2)};
  out[i] = loader.apply(T(i));
}

template [[host_name("run_narrow")]] [[kernel]] void run<float, 3, true>(
    device float*, uint);
template [[host_name("run_wide")]] [[kernel]] void run<float, 3, false>(
    device float*, uint);
"""

SELECTIONS = {
    "conditional_t": (
        "conditional_t<narrow && (N < 4), NarrowLoader<T, N>, WideLoader<T, N>>"
    ),
    "conditional-type": (
        "typename metal::conditional<narrow, NarrowLoader<T, N>, "
        "WideLoader<T, N>>::type"
    ),
    "nested": (
        "typename metal::conditional_t<narrow && N <= 4, NarrowLoader<T, N>, "
        "typename metal::conditional_t<(N > 2), WideLoader<T, N>, "
        "NarrowLoader<T, N>>>"
    ),
}


def _translate(tmp_path, backend, entry, selection):
    source = SOURCE.replace("SELECT", SELECTIONS[selection])
    (tmp_path / "loaders.metal").write_text(source, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        source_roots=(".",),
        include_patterns=("loaders.metal",),
        targets=(backend,),
        output_dir="out",
        entry_points={"loaders.metal": (entry,)},
        workgroup_size_rules={"loaders.metal": (4, 1, 1)},
    )
    return translate_project(config, format_output=False).to_json()


@pytest.mark.parametrize("selection", SELECTIONS)
@pytest.mark.parametrize(
    "entry,chosen,other",
    [
        ("run_narrow", "NarrowLoader", "WideLoader"),
        ("run_wide", "WideLoader", "NarrowLoader"),
    ],
)
@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_conditional_alias_receiver_calls_selected_struct(
    tmp_path, backend, entry, chosen, other, selection
):
    payload = _translate(tmp_path, backend, entry, selection)
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert not payload["diagnostics"]
    (artifact,) = payload["artifacts"]
    generated = (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    calls = re.findall(r"\b(\w+Loader)_float_3_{1,2}apply\(loader\b", generated)
    assert calls == [chosen]
    assert f"{other}_float_3_apply(loader" not in generated
    assert f"{other}_float_3__apply(loader" not in generated
    _compile(generated, backend, tmp_path)


def test_conditional_alias_with_runtime_condition_fails_closed(tmp_path):
    source = SOURCE.replace(
        "SELECT", "conditional_t<narrow, NarrowLoader<T, N>, WideLoader<T, N>>"
    ).replace("using loader_t", "bool runtime = i > 2u;\n  using loader_t")
    source = source.replace("conditional_t<narrow,", "conditional_t<runtime,")
    (tmp_path / "loaders.metal").write_text(source, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        source_roots=(".",),
        include_patterns=("loaders.metal",),
        targets=("opengl",),
        output_dir="out",
        entry_points={"loaders.metal": ("run_wide",)},
        workgroup_size_rules={"loaders.metal": (4, 1, 1)},
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert payload["diagnostics"]
