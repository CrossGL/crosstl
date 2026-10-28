"""Private pointers to member lvalues lower to ``inout`` member arguments.

A Metal helper may take ``thread uint*`` and its caller may pass ``&value.bits``,
the address of one member of a private struct or union value. DirectX and
OpenGL have no private pointers: the scalar view becomes an ``inout`` argument
of the member lvalue. Before this, both targets refused such calls with
``missing-view-backing`` because only whole locals and arrays could back a
private pointer view. The compare-and-swap loop below is the shape that packs
sub-word atomics into 32-bit words.

A union member is the canonical storage word only when it is a 32-bit unsigned
member; any other member is a reinterpretation and must fail closed. Two scalar
views of one object cannot be copied in and out independently, so such calls
fail closed on both targets.
"""

import re

import pytest

from crosstl._crosstl import translate
from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.codegen.directx_codegen import (
    DirectXPrivatePointerParameterError,
)
from crosstl.translator.codegen.GLSL_codegen import OpenGLPrivatePointerParameterError
from crosstl.translator.codegen.glsl_union_storage import OpenGLUnionLayoutError
from tests.test_translator.test_metal_builtin_ownership import _compile

PRELUDE = """#include <metal_stdlib>
using namespace metal;

union packed_word { uchar val[4]; uint bits; };
union float_word { uint bits; float value; };
struct word_box { uint bits; uint other; };
struct nested_box { word_box inner; };

static inline bool cas_word(device atomic<uint>* object, thread uint* expected, uint desired) {
  return atomic_compare_exchange_weak_explicit(
      object, expected, desired, memory_order_relaxed, memory_order_relaxed);
}

static inline void pair(thread uint* first, thread uint* second) {
  first[0] += 1u;
  second[0] += first[0];
}

static inline void scale(thread float* value) { value[0] *= 2.0f; }
"""

CAS_KERNEL = """
kernel void member_views(
    device atomic<uint>* bytes [[buffer(0)]],
    device atomic<uint>* words [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  packed_word expected;
  expected.bits = atomic_load_explicit(&bytes[gid / 4u], memory_order_relaxed);
  packed_word desired = expected;
  desired.val[gid % 4u] = uchar(desired.val[gid % 4u] + 1u);
  while (!cas_word(&bytes[gid / 4u], &(expected.bits), desired.bits)) {
    desired = expected;
    desired.val[gid % 4u] = uchar(desired.val[gid % 4u] + 1u);
  }
  nested_box current;
  current.inner.bits = atomic_load_explicit(&words[gid / 4u], memory_order_relaxed);
  while (!cas_word(&words[gid / 4u], &current.inner.bits, current.inner.bits + gid + 1u)) {
  }
}
"""

EXPECTED_ARGUMENTS = (
    "expected.CrossGLUnionStorage, desired.CrossGLUnionStorage",
    "current.inner.bits, ((current.inner.bits + gid) + 1u)",
)


def _translate(tmp_path, body, backend):
    source = tmp_path / "member_views.metal"
    source.write_text(PRELUDE + body, encoding="utf-8")
    return translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_member_address_views_lower_to_inout_member_arguments(tmp_path, target):
    generated = _translate(tmp_path, CAS_KERNEL, target)
    compact = " ".join(generated.split())
    for arguments in EXPECTED_ARGUMENTS:
        assert arguments in compact
    # The view is passed in place; no member copy or address remains.
    assert "&(expected" not in generated and "&current" not in generated
    _compile(generated, target, tmp_path)


def test_member_address_views_keep_metal_round_trip(tmp_path):
    generated = _translate(tmp_path, CAS_KERNEL, "metal")
    assert "&(expected.bits)" in generated or "&expected.bits" in generated
    _compile(generated, "metal", tmp_path)


@pytest.mark.parametrize(
    ("target", "error_type"),
    (
        ("directx", DirectXPrivatePointerParameterError),
        ("opengl", OpenGLPrivatePointerParameterError),
    ),
)
@pytest.mark.parametrize(
    "body",
    (
        "uint x = 3u; pair(&x, &x); results[0] = x;",
        "word_box b; b.bits = 3u; b.other = 1u; pair(&b.bits, &b.other); results[0] = b.bits;",
        "word_box b; b.bits = 3u; pair(&b.bits, &b.bits); results[0] = b.bits;",
    ),
)
def test_aliased_scalar_views_fail_closed(tmp_path, target, error_type, body):
    kernel = f"kernel void run(device uint* results [[buffer(0)]]) {{ {body} }}\n"
    with pytest.raises(error_type) as error:
        _translate(tmp_path, kernel, target)
    assert error.value.reason == "aliased-view-backing"


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_interpreted_union_member_address_fails_closed(tmp_path, target):
    kernel = (
        "kernel void run(device uint* results [[buffer(0)]]) "
        "{ float_word w; w.bits = 0x3f800000u; scale(&w.value); results[0] = w.bits; }\n"
    )
    source = tmp_path / "member_views.metal"
    source.write_text(PRELUDE + kernel, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        translate(
            str(source), backend=target, source_backend="metal", format_output=False
        )
    if target == "opengl":
        assert isinstance(error.value, OpenGLUnionLayoutError)
    assert error.value.reason == "interpreted-member-address-unsupported"
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("member_views.metal",),
            targets=(target,),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert report["diagnostics"][0]["code"] == error.value.project_diagnostic_code


# A constrained compare-and-swap overload whose ``&(expected.bits)`` argument is
# typed only after member lowering. Its late specialization still spells
# ``device wrapped_atomic<uchar>*`` and must be bound to the concrete struct.
LATE_HELPER_SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T>
constexpr constant bool is_native = is_same_v<T, int> || is_same_v<T, uint> || is_same_v<T, float>;

template <typename T, typename = void>
struct wrapped_atomic {
  atomic<uint> val;
};

template <typename T>
struct wrapped_atomic<T, enable_if_t<is_native<T>>> {
  atomic<T> val;
};

template <typename T, enable_if_t<!is_native<T>, bool> = true>
METAL_FUNC bool cas_word(device wrapped_atomic<T>* object, thread uint* expected, uint val, uint offset) {
  return atomic_compare_exchange_weak_explicit(
      &(object[offset].val), expected, val, memory_order_relaxed, memory_order_relaxed);
}

template <typename T, enable_if_t<is_native<T>, bool> = true>
METAL_FUNC bool cas_word(device wrapped_atomic<T>* object, thread T* expected, T val, uint offset) {
  return atomic_compare_exchange_weak_explicit(
      &(object[offset].val), expected, val, memory_order_relaxed, memory_order_relaxed);
}

template <typename T>
union packed_lanes {
  T val[4];
  uint bits;
};

template <typename T>
struct Add {
  static bool condition(T a, T b) { return a != T(0) || b == b; }
  T operator()(T a, T b) thread { return a + b; }
};

template <typename T, typename Op>
struct update_helper {
  uint operator()(packed_lanes<T> init, T update, uint elem) thread {
    Op op;
    init.val[elem] = op(update, init.val[elem]);
    return init.bits;
  }
};

template <typename T, typename Op>
METAL_FUNC void update_and_store(device wrapped_atomic<T>* object, T update, uint offset) {
  uint pack_offset = offset / 4u;
  uint elem_offset = offset % 4u;
  update_helper<T, Op> helper;
  packed_lanes<T> expected;
  expected.bits = atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);
  while (Op::condition(update, expected.val[elem_offset]) &&
         !cas_word(object, &(expected.bits), helper(expected, update, elem_offset), pack_offset)) {
  }
}

template <typename T, enable_if_t<!is_native<T>, bool> = true>
METAL_FUNC void fetch_add(device wrapped_atomic<T>* object, T val, uint offset) {
  update_and_store<T, Add<T>>(object, val, offset);
}

template <typename T, enable_if_t<is_native<T>, bool> = true>
METAL_FUNC void fetch_add(device wrapped_atomic<T>* object, T val, uint offset) {
  atomic_fetch_add_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename U>
struct Summer {
  template <typename T>
  void atomic_update(device wrapped_atomic<T>* out, T val, uint offset) thread {
    fetch_add(out, val, offset);
  }
};

[[kernel]] void bump_bytes(
    const device uchar* updates [[buffer(0)]],
    device wrapped_atomic<uchar>* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  Summer<uchar> op;
  op.atomic_update(out, updates[gid], gid);
}
"""


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_late_constrained_helper_binds_concrete_structs(tmp_path, target):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "late.metal").write_text(LATE_HELPER_SOURCE, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            source_roots=("src",),
            include_patterns=("src/late.metal",),
            targets=(target,),
            output_dir="out",
            entry_points={"src/late.metal": ("bump_bytes",)},
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 0, report["diagnostics"][:1]
    suffix = {"directx": ".hlsl", "opengl": ".glsl"}[target]
    (artifact,) = (
        path for path in (tmp_path / "out").rglob(f"*{suffix}") if path.is_file()
    )
    generated = artifact.read_text()
    assert "wrapped_atomic<" not in generated
    # GLSL reserves double underscores, so the lowered functor name differs.
    assert re.search(r"\bAdd_uchar_+operator_call\(op,", generated)
    assert "expected.CrossGLUnionStorage" in generated
    _compile(generated, target, tmp_path)
