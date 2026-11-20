"""Generic-lambda callbacks of a Boolean dispatch helper become their two instantiations.

``dispatch_bool(c, [&](auto k) { ... })`` calls the lambda with ``true_type`` or ``false_type``, so ``k.value`` is a
constant expression in each instantiation of the body. The translator expands such calls before template
materialization; calls it cannot prove equivalent stay unchanged for the callback lowering and its diagnostics.
"""

import os
import re
import sys
from pathlib import Path

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.bool_dispatch_callbacks import expand_bool_dispatch_callbacks
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)
from crosstl.project import build_native_loader_dispatch_request
from tests.ci_helpers import assert_native_step_prerequisites, assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_BOOL_DISPATCH_CALLBACK_RUNTIME"
TARGETS = ("metal", "directx", "opengl")

HELPER = """
template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(integral_constant<bool, true>{});
  } else {
    f(integral_constant<bool, false>{});
  }
}
"""

# Integral constants with the logical operator overloads compile-time tile
# libraries define, so `kA && kB` on two tags is itself a tag.
TAG_LIBRARY = """
namespace tags {
template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;
  constexpr operator value_type() const thread noexcept {
    return value;
  }
};
template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

// Tags count as integral, so the short circuits below take other operands only.
template <class T>
struct is_integral {
  static constexpr constant bool value = metal::is_integral<T>::value;
};
template <class T, T v>
struct is_integral<integral_constant<T, v>> {
  static constexpr constant bool value = metal::is_integral<T>::value;
};
template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <typename T, T tv, typename U, U uv>
constexpr auto operator&&(integral_constant<T, tv>, integral_constant<U, uv>) {
  constexpr auto res = tv && uv;
  using res_t = metal::remove_addrspace_t<decltype(res)>;
  return integral_constant<res_t, res>{};
}
template <typename T, T tv, typename U, U uv>
constexpr auto operator||(integral_constant<T, tv>, integral_constant<U, uv>) {
  constexpr auto res = tv || uv;
  using res_t = metal::remove_addrspace_t<decltype(res)>;
  return integral_constant<res_t, res>{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
constexpr auto operator||(true_type, T) {
  return true_type{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
constexpr auto operator&&(false_type, T) {
  return false_type{};
}

template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}
}  // namespace tags
using namespace tags;
"""

# Metal's own tags with `.value` uses only: they have no conversion to bool.
LIBRARY_TAG_KERNEL = (
    """#include <metal_stdlib>
using namespace metal;
"""
    + HELPER
    + """
template <bool Safe>
float load(device const float* p, uint i, uint n) {
  if constexpr (Safe) {
    return i < n ? p[i] : 0.0f;
  } else {
    return p[i];
  }
}

kernel void copy_tail(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  dispatch_bool(gid + 1 >= n, [&](auto kSafe) {
    float v = load<kSafe.value>(src, gid, n);
    if constexpr (kSafe.value) {
      if (gid < n) dst[gid] = v;
    } else {
      dst[gid] = v;
    }
  });
}
"""
)

KERNEL = (
    """#include <metal_stdlib>
using namespace metal;
"""
    + TAG_LIBRARY
    + """
template <bool Safe>
float load(device const float* p, uint i, uint n) {
  if constexpr (Safe) {
    return i < n ? p[i] : -1.0f;
  } else {
    return p[i];
  }
}

template <bool Scale, bool Shift>
float adjust(float v) {
  if constexpr (Scale) {
    v = v * 2.0f;
  }
  if constexpr (Shift) {
    v = v + 10.0f;
  }
  return v;
}

kernel void dispatch_tail(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          device const uint* sizes [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  const uint n = sizes[0];
  dispatch_bool(gid + 1 >= n, [&](auto kSafe) {
    dispatch_bool((gid & 1u) == 0u, [&](auto kScale) {
      dispatch_bool((gid & 2u) == 0u, [&](auto kShift) {
        float v = adjust<kScale.value, kShift.value>(load<kSafe.value>(src, gid, n));
        if constexpr (kSafe.value && (kScale || kShift)) {
          v = -v;
        }
        if constexpr (kSafe.value) {
          if (gid < n) {
            dst[gid] = v;
          }
        } else {
          dst[gid] = v;
        }
      });
    });
  });
}
"""
)

ELEMENTS = 10
THREADS = 12
GUARD = -12345.0


def _expected():
    result = []
    for gid in range(THREADS):
        if gid >= ELEMENTS:
            result.append(GUARD)
            continue
        safe = gid + 1 >= ELEMENTS
        scale, shift = gid & 1 == 0, gid & 2 == 0
        value = float(gid + 1)
        value = value * 2.0 if scale else value
        value = value + 10.0 if shift else value
        result.append(-value if safe and (scale or shift) else value)
    return result


def _expand(source, **options):
    preprocessor = MetalPreprocessor(**options)
    preprocessor._configure_integral_constant_contracts(source)
    return expand_bool_dispatch_callbacks(preprocessor, source)


def _compact(text):
    return re.sub(r"\s+", " ", text).strip()


def _call(callback, helper=HELPER, condition="c"):
    return (
        helper
        + f"void run(bool c, bool d) {{ dispatch_bool({condition}, {callback}); }}\n"
    )


def test_value_uses_become_constants_in_both_instantiations():
    source = _call("[&](auto k) { use<k.value>(k.value ? 1 : 2); }")
    expanded = _expand(source)
    assert _compact(expanded.replace(HELPER, "")) == _compact(
        "void run(bool c, bool d) { { if ((c)) { use<true>(true ? 1 : 2); } "
        "else { use<false>(false ? 1 : 2); } } }"
    )


def test_value_uses_combine_with_logical_operators():
    source = _call(
        "[&](auto k) { dispatch_bool(d, [&](auto j) {"
        " if constexpr (k.value && j.value) { use(); } else if (!k.value || d) { other(); } }); }"
    )
    expanded = _compact(_expand(source).replace(HELPER, ""))
    assert "auto" not in expanded
    assert "if constexpr (false && true)" in expanded and "if (!true || d)" in expanded


def test_nested_calls_expand_inside_out_and_inner_parameters_shadow():
    source = _call(
        "[&](auto k) { dispatch_bool(d, [&](auto k) { use<k.value>(); }); use<!k.value>(); }"
    )
    expanded = _compact(_expand(source).replace(HELPER, ""))
    assert "dispatch_bool" not in expanded and "auto k" not in expanded
    # The inner parameter shadows the outer one; the outer value reaches only
    # the outer use.
    assert expanded.count("use<true>()") == 2 and expanded.count("use<false>()") == 2
    assert expanded.count("use<!true>()") == 1 and expanded.count("use<!false>()") == 1
    assert expanded.count("if ((c))") == 1 and expanded.count("if ((d))") == 2


def test_bare_tags_in_conditions_use_their_value_under_logical_overloads():
    source = (
        TAG_LIBRARY + "void run(bool c, bool d) { dispatch_bool(c, [&](auto a) {"
        " dispatch_bool(d, [&](auto b) { if constexpr (a && b) { x(); }"
        " if (!(a || b)) { y(); } }); }); }\n"
    )
    expanded = _compact(_expand(source).replace(TAG_LIBRARY, ""))
    assert "auto a" not in expanded and "auto b" not in expanded
    for a in ("true", "false"):
        for b in ("true", "false"):
            assert f"if constexpr ({a} && {b})" in expanded
            assert f"if (!({a} || {b}))" in expanded


@pytest.mark.parametrize(
    "callback",
    [
        # By-value captures copy the enclosing variables.
        "[=](auto k) { use<k.value>(); }",
        # `return` would leave the enclosing function instead of the lambda.
        "[&](auto k) { if (k.value) { return; } use(); }",
        # The tag's type matters to a call or an initializer.
        "[&](auto k) { use(k); }",
        "[&](auto k) { bool b = k; use(b); }",
        "[&](auto k) { use(k + 1); }",
        # The address of the static member is not a constant value.
        "[&](auto k) { use(&k.value); }",
        "[&](auto k) { use(d && &k.value); }",
        "[&](auto k) { use(sizeof(k.value)); }",
        # A nested lambda redeclares the parameter.
        "[&](auto k) { auto g = [](auto k) { return k.value; }; use(g(1)); }",
        # Not one deduced parameter.
        "[&]() { use(); }",
        "[&](bool k) { use(k); }",
    ],
)
def test_callbacks_that_are_not_provably_equivalent_stay_unchanged(callback):
    source = _call(callback)
    assert _expand(source) == source


@pytest.mark.parametrize(
    "helper",
    [
        # One branch only.
        "template <typename F> void dispatch_bool(bool v, F f) { if (v) { f(true_type{}); } }",
        # Swapped tags.
        "template <typename F> void dispatch_bool(bool v, F f) {"
        " if (v) { f(false_type{}); } else { f(true_type{}); } }",
        # Not a Boolean condition.
        "template <typename F> void dispatch_bool(int v, F f) {"
        " if (v) { f(true_type{}); } else { f(false_type{}); } }",
        # Overloaded helper.
        HELPER + "template <typename F> void dispatch_bool(int v, F f) { f(v); }",
        HELPER + "void dispatch_bool(bool v, int f) {}",
    ],
)
def test_unverified_helpers_keep_their_calls(helper):
    source = _call("[&](auto k) { use<k.value>(); }", helper=helper)
    assert _expand(source) == source


@pytest.mark.parametrize(
    "library",
    [
        # A logical overload that does not compute the logical value.
        TAG_LIBRARY.replace(
            "constexpr auto res = tv && uv;", "constexpr auto res = !(tv && uv);"
        ),
        # A tag alias that is not the standard one.
        TAG_LIBRARY.replace(
            "using true_type = bool_constant<true>;",
            "using true_type = bool_constant<false>;",
        ),
        # An integral constant whose `value` is not its argument.
        TAG_LIBRARY.replace(
            "static constexpr constant T value = v;",
            "static constexpr constant T value = !v;",
        ),
    ],
)
def test_nonstandard_tag_libraries_keep_their_calls(library):
    assert library != TAG_LIBRARY
    source = (
        library
        + "void run(bool c) { dispatch_bool(c, [&](auto k) { if constexpr (k && k) { use(); } }); }\n"
    )
    assert _expand(source) == source


def test_standard_tag_library_expands():
    source = (
        TAG_LIBRARY
        + "void run(bool c) { dispatch_bool(c, [&](auto k) { if constexpr (k && k) { use(); } }); }\n"
    )
    expanded = _expand(source)
    assert "if constexpr (true && true)" in expanded
    assert "if constexpr (false && false)" in expanded


def test_short_circuit_overloads_require_effect_free_right_operands():
    source = (
        TAG_LIBRARY
        + "void run(bool c) { dispatch_bool(c, [&](auto k) { if (k && next()) { use(); } }); }\n"
    )
    assert _expand(source) == source
    effect_free = source.replace("next()", "ready")
    assert "if (true && ready)" in _expand(effect_free)


def test_expansion_is_bounded_by_the_specialization_limit():
    source = _call(
        "[&](auto a) { dispatch_bool(d, [&](auto b) { use<a.value, b.value>(); }); }"
    )
    # Two inner bodies in each of the two outer copies, plus the outer copies.
    assert _expand(source, max_template_specializations=6) != source
    with pytest.raises(MetalTemplateSpecializationError, match="Boolean dispatch"):
        _expand(source, max_template_specializations=5)


def test_bare_library_tags_stay_unchanged():
    # metal::integral_constant has no conversion to its value, so only `.value`
    # names the constant.
    source = _call("[&](auto k) { if constexpr (k && k) { use(); } }")
    assert _expand(source) == source


@pytest.mark.parametrize(
    "source", [LIBRARY_TAG_KERNEL, KERNEL], ids=["library", "source"]
)
@pytest.mark.parametrize("target", TARGETS)
def test_file_translation_instantiates_both_callbacks(tmp_path, target, source):
    path = tmp_path / "dispatch.metal"
    path.write_text(source, encoding="utf-8")
    generated = translate(str(path), backend=target)
    assert "dispatch_bool" not in generated and "kSafe" not in generated
    if target != "metal":
        _compile(generated, target, tmp_path)


def _request(root, target):
    source, descriptor, package = _package(
        root, target, "float", (1, 1, 1), source=KERNEL, software_subgroups=False
    )
    inputs = {
        "src": {
            "dtype": "float32",
            "shape": [ELEMENTS],
            "values": [float(i + 1) for i in range(ELEMENTS)],
        },
        "dst": {"dtype": "float32", "shape": [THREADS], "values": [GUARD] * THREADS},
        "sizes": {"dtype": "uint32", "shape": [1], "values": [ELEMENTS]},
    }
    outputs = {"dst": {"dtype": "float32", "shape": [THREADS], "values": _expected()}}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _bound_values(descriptor, outputs),
        {"workgroupCount": [THREADS, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, _bound_values(descriptor, outputs)


def test_expected_values_cover_every_instantiation():
    expected = _expected()
    assert expected[10:] == [GUARD, GUARD]
    # Every (kScale, kShift) pair occurs with and without kSafe.
    assert (
        expected[0] == 12.0
        and expected[1] == 12.0
        and expected[2] == 6.0
        and expected[3] == 4.0
    )
    assert expected[9] == -20.0


@pytest.mark.parametrize("target", TARGETS)
def test_project_translation_compiles_on_every_target(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "dispatch_bool" not in generated and "kSafe" not in generated
    _compile(generated, target, tmp_path)


def test_dispatch_matches_the_source_kernel_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required Boolean dispatch execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="dispatch_tail",
    )


def test_native_dispatch_gate_is_required():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate Metal argument inference"
    )
    assert "test_metal_bool_dispatch_callbacks.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert_native_step_prerequisites(step, job="portable-host")
    assert "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_metal_bool_dispatch_callbacks.py",
        )
