"""A call whose constrained candidates are all disabled falls back to the Metal library only where lookup finds it.

The importer leaves such a call spelled as written, so ``metal::max`` resolves it once the disabled templates are
removed. C++ lookup stops at the first scope that declares the name and SFINAE removes candidates only afterwards, so
in each ``ILL_FORMED`` source the call has no viable function: the Metal compiler rejects every one of them with "no
matching function for call to 'max'". Binding the library function instead would silently accept a different
program. Every ``WELL_FORMED`` source compiles with the Metal compiler, and its call resolves to ``metal::max``.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.library_fallback_lookup import (
    library_fallback_obstacle,
    library_using_declaration_present,
)
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)

FLOAT_MAX = (
    "template <typename T, metal::enable_if_t<metal::is_floating_point_v<T>, bool> = true>\n"
    "T max(T a, T b) { return a > b ? a : b + T(100); }\n"
)
KERNEL = (
    "kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    "  o[i] = float(h(int(o[i]))) + max(o[i], 1.0f);\n"
    "}\n"
)
NAMESPACED_KERNEL = (
    "kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    "  o[i] = float(ns::h(int(o[i]))) + ns::max(o[i], 1.0f);\n"
    "}\n"
)
HEADER = "#include <metal_stdlib>\n"

ILL_FORMED = {
    # No using-directive: the global template is the only max lookup finds.
    "no_directive": (
        HEADER + FLOAT_MAX + "int h(int a) { return max(a, 4); }\n" + KERNEL
    ),
    # The library names appear in the global namespace, which lookup from ns never reaches.
    "hidden_by_namespace": (
        HEADER
        + "using namespace metal;\nnamespace ns {\n"
        + FLOAT_MAX
        + "int h(int a) { return max(a, 4); }\n}\n"
        + NAMESPACED_KERNEL
    ),
    # A directive inside ns still makes the library names appear in the global namespace.
    "hidden_despite_inner_directive": (
        HEADER
        + "namespace ns {\nusing namespace metal;\n"
        + FLOAT_MAX
        + "int h(int a) { return max(a, 4); }\n}\n"
        + NAMESPACED_KERNEL
    ),
    # Qualified lookup follows the global directive only when the global namespace declares no max.
    "globally_qualified": (
        HEADER
        + "using namespace metal;\n"
        + FLOAT_MAX
        + "int h(int a) { return ::max(a, 4); }\n"
        + KERNEL
    ),
    # Reached only through a template body, as in a call that a template carries.
    "hidden_through_template": (
        HEADER
        + "namespace ns {\n"
        + FLOAT_MAX
        + "int h(int a) { return max(a, 4); }\n"
        + "template <typename T> T foo(T a) { return T(h(int(a))); }\n}\n"
        + "kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
        + "  o[i] = ns::max(o[i], 1.0f) + ns::foo(o[i]);\n}\n"
    ),
}

# Each source compiles with the Metal compiler; the number is how many of its calls the user template serves.
WELL_FORMED = {
    # Lookup from ns reaches the global namespace, where the template and metal::max appear together; the library's
    # non-template overloads win both calls.
    "global_template_called_from_namespace": (
        HEADER
        + "using namespace metal;\n"
        + FLOAT_MAX
        + "namespace ns { int h(int a) { return max(a, 4); } }\n"
        + NAMESPACED_KERNEL.replace("ns::max", "max"),
        0,
    ),
    # The directive inside ns does not apply to the kernel's call, which binds the template.
    "inner_directive_global_template": (
        HEADER
        + FLOAT_MAX
        + "namespace ns { using namespace metal; int h(int a) { return max(a, 4); } }\n"
        + NAMESPACED_KERNEL.replace("ns::max", "max"),
        1,
    ),
    # A using-declaration brings metal::max into ns, beside the template.
    "using_declaration": (
        HEADER
        + "namespace ns {\n"
        + FLOAT_MAX
        + "using metal::max;\nint h(int a) { return max(a, 4); }\n}\n"
        + NAMESPACED_KERNEL,
        0,
    ),
    # `using namespace ns;` at global scope puts ns::max beside metal::max.
    "namespace_directive_at_global_scope": (
        HEADER
        + "using namespace metal;\nnamespace ns {\n"
        + FLOAT_MAX
        + "}\nusing namespace ns;\nint h(int a) { return max(a, 4); }\n"
        + KERNEL,
        0,
    ),
}


def _translate(tmp_path, source, backend="metal"):
    path = tmp_path / "fallback.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("name", sorted(ILL_FORMED))
@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_disabled_candidates_that_hide_the_library_function_are_refused(
    tmp_path, name, backend
):
    with pytest.raises(
        MetalTemplateSpecializationError, match="max\\(a, 4\\)"
    ) as caught:
        _translate(tmp_path, ILL_FORMED[name], backend)

    assert "every constrained candidate is disabled" in str(caught.value)


@pytest.mark.parametrize("name", sorted(WELL_FORMED))
@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_disabled_candidates_fall_back_where_lookup_finds_the_library(
    tmp_path, name, backend
):
    source, template_calls = WELL_FORMED[name]

    code = _translate(tmp_path, source, backend)

    h_body = re.search(r"\bh\s*\(int a\)\s*\{([^}]*)\}", code)
    assert h_body, code
    assert re.search(r"(?<![\w.:])max\(a, 4\)", h_body.group(1))
    assert code.count("100") == template_calls
    assert bool(re.search(r"\bmax_float\(", code)) == bool(template_calls)


def test_globally_qualified_enabled_call_binds_the_template(tmp_path):
    source = ILL_FORMED["globally_qualified"].replace(
        "::max(a, 4)", "int(::max(float(a), 4.0f))"
    )

    code = _translate(tmp_path, source)

    assert re.search(r"\bh\s*\(int a\)\s*\{[^}]*\bmax_float\(", code)


def test_keyword_before_a_global_qualifier_is_not_a_scope():
    preprocessor = MetalPreprocessor()
    calls = preprocessor._find_static_constexpr_calls(
        "int h(int a) { return ::max(a, 4) + ns :: max(a, 5); }", {"max"}
    )

    assert [call.qualified_name for call in calls] == ["::max", "ns::max"]


class _Directive:
    def __init__(self, target, namespace, position=0, end=1000):
        self.target = target
        self.declaration_namespace = namespace
        self.declaration_position = position
        self.scope_start = 0
        self.scope_end = end
        self.globally_qualified = False


def _targets(directive):
    return (directive.target,)


@pytest.mark.parametrize(
    ("reference", "call_namespace", "candidates", "directives", "hidden"),
    [
        (None, "", [""], [_Directive("metal", "")], False),
        (None, "", [""], [], True),
        (None, "ns", ["ns"], [_Directive("metal", "")], True),
        (None, "ns::inner", ["ns"], [_Directive("metal", "")], True),
        (None, "ns", [""], [_Directive("metal", "")], False),
        (None, "", ["ns"], [_Directive("metal", ""), _Directive("ns", "")], False),
        (None, "a", ["a::b"], [_Directive("metal", ""), _Directive("a::b", "a")], True),
        ("::", "", [""], [_Directive("metal", "")], True),
        ("ns", "", ["ns"], [_Directive("metal", "")], True),
        ("metal", "", ["metal"], [], False),
        ("metal", "c10", ["c10::metal"], [], True),
        (None, "metal", ["metal"], [], False),
        (None, "", [], [], False),
    ],
)
def test_library_fallback_obstacle(
    reference, call_namespace, candidates, directives, hidden
):
    obstacle = library_fallback_obstacle(
        name="max",
        reference_namespace=reference,
        call_namespace=call_namespace,
        position=10,
        candidate_namespaces=candidates,
        directives=directives,
        directive_targets=_targets,
    )

    assert (obstacle is not None) is hidden


def test_directive_declared_after_the_call_is_inactive():
    obstacle = library_fallback_obstacle(
        name="max",
        reference_namespace=None,
        call_namespace="",
        position=10,
        candidate_namespaces=[""],
        directives=[_Directive("metal", "", position=20)],
        directive_targets=_targets,
    )

    assert obstacle is not None and "no using-directive" in obstacle


def test_library_using_declaration_present():
    assert library_using_declaration_present(
        "namespace ns { using ::metal :: max; }", "max"
    )
    assert not library_using_declaration_present("using metal::min;", "max")
    assert not library_using_declaration_present("using metal::maximum;", "max")
