"""Metal round trips restore literal ``operator bool()`` members.

The Metal frontend moves a struct's ``operator bool()`` into a free helper.
Metal still type-checks branches a folded template constant makes dead, as
MLX's masked GEMV's ``if (has_output_mask) { if (!mask_out) ... }`` and
``bool(mat_mask[i])`` over its ``_NoMask`` type, so the regenerated struct
needs the members again.
"""

import pytest

from crosstl.project import ProjectConfig, translate_project
from tests.test_translator.test_metal_builtin_ownership import _compile

NO_MASK = """#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

struct _NoMask {
  char x;

  constexpr METAL_FUNC operator bool() thread {
    return true;
  }
  constexpr METAL_FUNC operator bool() const threadgroup {
    return true;
  }
  constexpr METAL_FUNC operator bool() const device {
    return true;
  }
  constexpr METAL_FUNC operator bool() const constant {
    return true;
  }
};

typedef struct _NoMask nomask_t;

struct Never {
  char x;

  constexpr METAL_FUNC operator bool() const {
    return false;
  }
};

template <typename T, typename mask_t>
struct MaskedCopy {
  MLX_MTL_CONST bool has_mask = !metal::is_same_v<mask_t, nomask_t>;

  static METAL_FUNC void run(const device mask_t* mask, device T* out, uint i) {
    if (has_mask) {
      auto m = mask[i];
      if (!m) {
        out[i] = T(0);
        return;
      }
    }
    if (!has_mask || bool(mask[i])) {
      out[i] = T(1);
    }
  }
};

template <typename T, typename mask_t>
[[kernel]] void masked_copy(
    const device mask_t* mask [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  MaskedCopy<T, mask_t>::run(mask, out, i);
}

template <typename mask_t>
[[kernel]] void never_copy(
    const device mask_t* mask [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (metal::is_same_v<mask_t, bool>) {
    out[i] = 1.0f;
  } else {
    const thread mask_t local = mask[i];
    out[i] = bool(local) ? 2.0f : 3.0f;
  }
}

template [[host_name("masked_copy_nomask")]] [[kernel]] void
masked_copy<float, nomask_t>(
    const device nomask_t* mask [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]);

template [[host_name("masked_copy_bool")]] [[kernel]] void
masked_copy<float, bool>(
    const device bool* mask [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]);

template [[host_name("never_copy_never")]] [[kernel]] void never_copy<Never>(
    const device Never* mask [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]);
"""

NO_MASK_MEMBERS = (
    "    constexpr operator bool() thread { return true; }\n"
    "    constexpr operator bool() const threadgroup { return true; }\n"
    "    constexpr operator bool() const device { return true; }\n"
    "    constexpr operator bool() const constant { return true; }\n"
    "};\n"
)


def _round_trip(tmp_path, entry):
    (tmp_path / "masks.metal").write_text(NO_MASK, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=("masks.metal",),
        targets=("metal",),
        entry_points={"masks.metal": (entry,)},
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert payload["diagnostics"] == []
    (artifact,) = payload["artifacts"]
    return (tmp_path / artifact["path"]).read_text(encoding="utf-8")


def _struct(generated, name):
    start = generated.index(f"struct {name} {{\n")
    return generated[start : generated.index("};\n", start) + 3]


def test_per_receiver_literal_conversions_are_restored(tmp_path):
    generated = _round_trip(tmp_path, "masked_copy_nomask")
    # The folded constant leaves both conversions in dead code.
    assert "if (false) {" in generated
    assert "if (!m) {" in generated
    assert "bool(mask[i])" in generated
    assert _struct(generated, "_NoMask") == (
        "struct _NoMask {\n    char x;\n" + NO_MASK_MEMBERS
    )
    _compile(generated, "metal", tmp_path)


def test_lone_literal_conversion_becomes_a_const_member(tmp_path):
    generated = _round_trip(tmp_path, "never_copy_never")
    # ``local`` is const, so a mutable member could not be called on it.
    assert "bool(local)" in generated
    assert _struct(generated, "Never") == (
        "struct Never {\n"
        "    char x;\n"
        "    constexpr operator bool() const thread { return false; }\n"
        "};\n"
    )
    _compile(generated, "metal", tmp_path)


def test_entries_without_the_struct_are_unchanged(tmp_path):
    generated = _round_trip(tmp_path, "masked_copy_bool")
    assert "operator bool" not in generated
    assert "_NoMask" not in generated
    _compile(generated, "metal", tmp_path)


@pytest.mark.parametrize(
    "body",
    ("return x != 0;", "return bool(x);", "char y = x; return true;"),
)
def test_only_literal_bodies_are_restored(tmp_path, body):
    source = NO_MASK.replace(
        "constexpr METAL_FUNC operator bool() const {\n    return false;\n  }",
        f"METAL_FUNC operator bool() const {{\n    {body}\n  }}",
    )
    assert source != NO_MASK
    (tmp_path / "masks.metal").write_text(source, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=("masks.metal",),
        targets=("metal",),
        entry_points={"masks.metal": ("never_copy_never",)},
    )
    payload = translate_project(config, format_output=False).to_json()
    (artifact,) = payload["artifacts"]
    generated = (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    assert "operator bool" not in _struct(generated, "Never")
