"""A struct template-id names its matching partial specialization, not the primary template.

``typename tr<device half*>::type`` is the member of ``template <typename T> struct tr<device T*>`` when that
partial specialization matches ([temp.spec.partial.match]), the most specialized match wins
([temp.spec.partial.order]), and a choice that cannot be made where the template-id is spelled is refused
instead of falling back to the primary template.
"""

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalStructAliasResolutionError
from crosstl.backend.Metal.partial_struct_specializations import (
    MATCH,
    NO_MATCH,
    UNDECIDED,
    match_partial_specialization,
    more_specialized,
    select_partial_specialization,
)

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

POINTER_TRAIT = """
template <typename T> struct element { using type = float; };
template <typename T> struct element<device T*> { using type = T; };
template <typename T> using element_t = typename element<T>::type;
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


@pytest.mark.parametrize(
    "pattern, argument, bindings",
    [
        ("device T*", "device half*", {"T": "half"}),
        ("device T*", "const device half*", {"T": "const half"}),
        ("device T*", "device const half *", {"T": "const half"}),
        ("vec<T, 2>", "float2", {"T": "float"}),
        ("vec<T, 2>", "metal::vec<int, 2>", {"T": "int"}),
        ("pair<T, U>", "pair<vec<float, 2>, int>", {"T": "vec<float, 2>", "U": "int"}),
        (
            "integral_constant<T, v>",
            "integral_constant<int, 3u>",
            {"T": "int", "v": "3"},
        ),
    ],
)
def test_patterns_deduce_their_parameters(pattern, argument, bindings):
    outcome = match_partial_specialization([pattern], [argument], list(bindings))

    assert outcome.status == MATCH
    assert dict(outcome.bindings) == bindings


@pytest.mark.parametrize(
    "pattern, argument",
    [
        ("device T*", "thread half*"),
        ("device T*", "half"),
        ("vec<T, 2>", "vec<float, 3>"),
        ("pair<T, T>", "pair<int, float>"),
    ],
)
def test_patterns_that_cannot_deduce_do_not_match(pattern, argument):
    parameters = ["T", "U"] if "U" in pattern else ["T"]

    assert (
        match_partial_specialization([pattern], [argument], parameters).status
        == NO_MATCH
    )


@pytest.mark.parametrize(
    "patterns, arguments, reason",
    [
        (["device T*"], ["P"], "may match"),
        (["vec<T, 2>"], ["vec<float, N>"], "may match"),
        (["T", "8"], ["float", "kRows"], "may match"),
        (
            ["T", "enable_if_t<is_floating_point_v<T>>"],
            ["half", "void"],
            "needs evaluation",
        ),
        (["T", "int"], ["float", "N * 2"], "needs evaluation"),
    ],
)
def test_undecidable_matches_are_reported(patterns, arguments, reason):
    outcome = match_partial_specialization(
        patterns, arguments, ["T"], dependent_names=["P", "N"]
    )

    assert outcome.status == UNDECIDED
    assert reason in outcome.reason


def test_the_most_specialized_match_is_selected():
    status, declaration, bindings, _reason = select_partial_specialization(
        [("any pointer", ["T*"], ["T"]), ("device pointer", ["device T*"], ["T"])],
        ["device half*"],
    )

    assert (status, declaration, bindings) == (MATCH, "device pointer", {"T": "half"})
    assert more_specialized((["device T*"], ["T"]), (["T*"], ["T"]))
    assert not more_specialized((["T*"], ["T"]), (["device T*"], ["T"]))


def test_matches_without_a_most_specialized_one_are_undecided():
    status, declaration, _bindings, reason = select_partial_specialization(
        [("left", ["T", "int"], ["T"]), ("right", ["int", "T"], ["T"])],
        ["int", "int"],
    )

    assert (status, declaration) == (UNDECIDED, None)
    assert "more specialized" in reason


@pytest.mark.parametrize(
    "backend, declaration",
    [
        ("metal", "half x = half(1.25)"),
        ("directx", "float16_t x = float16_t(1.25)"),
        ("cgl", "float16 x = float16(1.25f)"),
    ],
)
def test_alias_through_a_matching_partial_specialization(
    tmp_path, backend, declaration
):
    code = _translate(
        tmp_path,
        POINTER_TRAIT + """
kernel void run(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  element_t<device half*> x = element_t<device half*>(1.25f);
  out[i] = x / element_t<device half*>(3.0f);
}
""",
        backend,
    )

    assert declaration in code
    assert "float x" not in code


def test_primary_template_applies_when_no_partial_specialization_matches(tmp_path):
    code = _translate(
        tmp_path,
        POINTER_TRAIT + """
kernel void run(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  element_t<thread half*> x = element_t<thread half*>(1.25f);
  out[i] = x;
}
""",
        "metal",
    )

    assert "float x = float(1.25)" in code


def test_most_specialized_partial_specialization_is_used(tmp_path):
    code = _translate(
        tmp_path,
        """
template <typename T> struct element { using type = float; };
template <typename T> struct element<T*> { using type = int; };
template <typename T> struct element<device T*> { using type = T; };

kernel void run(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  using U = typename element<device half*>::type;
  using V = typename element<thread half*>::type;
  out[i] = U(V(3));
}
""",
        "metal",
    )

    assert "half(int(3))" in code


def test_vector_shorthand_argument_matches_a_vec_pattern(tmp_path):
    code = _translate(
        tmp_path,
        """
template <typename T> struct scalar_of { using type = int; };
template <typename T> struct scalar_of<vec<T, 2>> { using type = T; };
template <typename T> using scalar_of_t = typename scalar_of<T>::type;

kernel void run(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  scalar_of_t<half2> x = scalar_of_t<half2>(0.5f);
  out[i] = x;
}
""",
        "metal",
    )

    assert "half x = half(0.5)" in code


@pytest.mark.parametrize("backend", ["metal", "cgl"])
def test_dependent_argument_that_may_match_is_refused(tmp_path, backend):
    source = POINTER_TRAIT + """
template <typename P>
void store_one(P dst, float v) {
  using U = element_t<P>;
  dst[0] = static_cast<U>(v);
}

kernel void run(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  store_one(out + i, 1.5f);
}
"""

    with pytest.raises(MetalStructAliasResolutionError) as error:
        _translate(tmp_path, source, backend)

    assert error.value.project_diagnostic_code == (
        "project.translate.metal-struct-alias-unresolved"
    )
    assert "<device T*> may apply" in str(error.value)


def test_specialization_that_needs_evaluation_is_refused(tmp_path):
    source = """
template <typename T, typename = void> struct select { using type = int; };
template <typename T> struct select<T, enable_if_t<is_floating_point_v<T>>> { using type = T; };
template <typename T> using select_t = typename select<T>::type;

kernel void run(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  select_t<half> x = select_t<half>(0.5f);
  out[i] = x;
}
"""

    with pytest.raises(MetalStructAliasResolutionError) as error:
        _translate(tmp_path, source, "metal")

    assert "enable_if_t<is_floating_point_v<T>>> may apply" in str(error.value)
