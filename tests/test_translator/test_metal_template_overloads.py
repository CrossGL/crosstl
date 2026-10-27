"""Explicit instantiations of overloaded templates and qualified member types."""

import textwrap

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.template_overloads import (
    declared_type_names,
    select_instantiated_overload,
)
from crosstl.project import translate_project
from tests.test_translator.test_project_translation import (
    assert_compute_glsl_validates_if_available,
    assert_directx_compute_validates_if_available,
    assert_metal_validates_if_available,
)

TARGETS = ("directx", "opengl", "metal")

OVERLOADED_KERNELS = """
#include <metal_stdlib>
using namespace metal;

struct _NoMask {};
typedef struct _NoMask nomask_t;

template <typename T, typename mask_t, int N>
[[kernel]] void scale(
    device T* out [[buffer(0)]],
    const device mask_t* mask [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  constexpr bool has_mask = !metal::is_same_v<mask_t, nomask_t>;
  T value = T(N);
  if (has_mask) {
    value = value * T(2);
  }
  out[gid] = value;
}

template <typename T, int N, bool has_mask = false>
[[kernel]] void scale(
    device T* out [[buffer(0)]],
    const device bool* mask [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  out[gid] = T(N + (has_mask ? 100 : 200));
}

template [[host_name("scale_float_nomask")]] [[kernel]] void
scale<float, nomask_t, 3>(
    device float* out [[buffer(0)]],
    const device nomask_t* mask [[buffer(1)]],
    uint gid [[thread_position_in_grid]]);
"""

QUALIFIED_RECEIVER_KERNEL = """
#include <metal_stdlib>
using namespace metal;

template <typename T, int N>
struct Loader {
  const device T* src;
  int stride;
  Loader(const device T* src_, int stride_) : src(src_), stride(stride_) {}
  T load_safe(int index) const {
    return index < N ? src[index * stride] : T(0);
  }
};

template <typename T, int N>
struct Kernel {
  using loader_t = Loader<T, N>;
  static constant constexpr const int size = N;
};

template <typename T, int N>
[[kernel]] void copy_tile(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  using kernel_t = Kernel<T, N>;
  thread {declared_type} loader(src, 2);
  dst[gid] = loader.load_safe(int(gid) % kernel_t::size);
}

template [[host_name("copy_tile_float_4")]] [[kernel]] void copy_tile<float, 4>(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]);
"""

UNMATERIALIZED_KERNEL_WITH_DISPATCH_BOOL = """
#include <metal_stdlib>
using namespace metal;

template <typename F>
void dispatch_bool(bool flag, F callback) {
  if (flag) {
    callback(true_type{});
  } else {
    callback(false_type{});
  }
}

template <typename T, int N>
[[kernel]] void fill(
    device T* out [[buffer(0)]],
    uint gid [[thread_position_in_grid]]) {
  out[gid] = T(N);
}

template [[host_name("fill_float")]] [[kernel]] void fill<float>(
    device float* out [[buffer(0)]],
    uint gid [[thread_position_in_grid]]);
"""


def _translate(tmp_path, source):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.metal").write_text(textwrap.dedent(source).strip() + "\n")
    payload = translate_project(
        repo,
        targets=list(TARGETS),
        output_dir="out",
        format_output=False,
    ).to_json()
    outputs = {
        artifact["target"]: (repo / artifact["path"]).read_text(encoding="utf-8")
        for artifact in payload["artifacts"]
        if (repo / artifact["path"]).exists()
    }
    return payload, outputs


def _assert_validates(outputs, tmp_path):
    assert_directx_compute_validates_if_available(outputs["directx"], tmp_path)
    assert_compute_glsl_validates_if_available(outputs["opengl"], tmp_path)
    assert_metal_validates_if_available(outputs["metal"], tmp_path)


def test_explicit_instantiation_selects_overload_by_template_parameter_kinds(
    tmp_path,
):
    payload, outputs = _translate(tmp_path, OVERLOADED_KERNELS)

    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == len(TARGETS)
    for target, output in outputs.items():
        assert "nomask_t +" not in output, target
        assert "100" not in output and "200" not in output, target
        assert "float(3)" in output, target
    _assert_validates(outputs, tmp_path)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["float", "nomask_t", "3"], 0),
        (["float", "3"], 1),
        (["float", "3", "true"], 1),
        (["float", "nomask_t", "3", "4"], None),
        (["4", "nomask_t", "3"], None),
    ],
)
def test_overload_selection_uses_arity_defaults_and_certain_kinds(arguments, expected):
    source = textwrap.dedent(OVERLOADED_KERNELS)
    preprocessor = MetalPreprocessor()
    templates = [
        template
        for template in preprocessor._find_template_functions(source)
        if template.name == "scale"
    ]
    assert len(templates) == 2

    selected = select_instantiated_overload(
        preprocessor,
        declared_type_names(source),
        templates,
        arguments,
    )

    assert selected is (None if expected is None else templates[expected])


def test_overload_selection_keeps_single_candidate_and_last_ambiguous_overload():
    source = textwrap.dedent(OVERLOADED_KERNELS)
    preprocessor = MetalPreprocessor()
    templates = [
        template
        for template in preprocessor._find_template_functions(source)
        if template.name == "scale"
    ]

    assert (
        select_instantiated_overload(preprocessor, frozenset(), templates[:1], ["4"])
        is templates[0]
    )
    # `unknown_t` has no visible declaration, so both overloads still fit and
    # the last declaration is kept, as the single-name lookup did.
    assert (
        select_instantiated_overload(
            preprocessor, frozenset(), templates, ["float", "unknown_t", "3"]
        )
        is templates[1]
    )
    assert (
        select_instantiated_overload(
            preprocessor, frozenset(), templates, ["float", "unknown_n"]
        )
        is templates[1]
    )


def test_declared_type_names_reads_structs_aliases_and_typedefs():
    names = declared_type_names(
        "struct A {}; typedef struct A a_t; using b_t = float; enum class C { x };"
    )

    assert {"A", "a_t", "b_t", "C"} <= names


@pytest.mark.parametrize(
    "declared_type",
    [
        "typename kernel_t::loader_t",
        "kernel_t::loader_t",
        "typename Kernel<T, N>::loader_t",
    ],
)
def test_receiver_declared_through_qualified_member_type_alias(tmp_path, declared_type):
    payload, outputs = _translate(
        tmp_path,
        QUALIFIED_RECEIVER_KERNEL.replace("{declared_type}", declared_type),
    )

    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == len(TARGETS)
    for target, output in outputs.items():
        assert "loader_t" not in output, target
        assert "load_safe" in output, target
    _assert_validates(outputs, tmp_path)


def test_dispatch_bool_helper_without_concrete_entry_root_fails_closed(tmp_path):
    payload, outputs = _translate(tmp_path, UNMATERIALIZED_KERNEL_WITH_DISPATCH_BOOL)

    assert outputs == {}
    assert payload["summary"]["translatedCount"] == 0
    codes = {diagnostic["code"] for diagnostic in payload["diagnostics"]}
    assert codes == {"project.translate.template-materialization-unsupported"}
    for diagnostic in payload["diagnostics"]:
        assert "fill missing N" in diagnostic["message"]
