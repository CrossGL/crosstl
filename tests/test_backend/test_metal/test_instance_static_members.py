"""Static data members read through an object expression.

``t.value`` for a ``static constexpr`` member names the member's constant. The
importer used to emit it as a field read, and the lowered struct has no such
field (an empty one is padded with ``int _crossgl_empty``), so HLSL read a
field that does not exist.
"""

import re

import pytest

import crosstl

PLAIN_OBJECTS = """
#include <metal_stdlib>
using namespace metal;

struct Tag {
  static constexpr constant bool value = true;
  static constexpr constant int count = 3;
  float payload;
};
struct Holder {
  Tag inner;
  Tag many[2];
};

float scaled(thread const Tag& t) {
  return t.value ? float(t.count) * t.payload : 0.0f;
}

kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  Holder h;
  h.inner.payload = 2.0f;
  h.many[1].payload = 4.0f;
  int n = h.inner.count + h.many[1].count;
  out[tid] = scaled(h.inner) + float(n) + (h.many[1].value ? 1.0f : 0.0f);
}
"""


def _translate(tmp_path, code, backend, name="static_members.metal"):
    source = tmp_path / name
    source.write_text(code)
    return crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal", "cgl"])
def test_static_members_read_through_plain_objects_fold(tmp_path, backend):
    output = _translate(tmp_path, PLAIN_OBJECTS, backend)

    assert not re.search(r"\.(?:value|count)\b", output)
    assert re.search(r"return \(?true \? \(?float\(3\) \* t\.payload", output)
    assert "int n = 3 + 3;" in output or "int n = (3 + 3);" in output
    assert "t.payload" in output


def test_static_member_read_through_pointer_folds(tmp_path):
    code = PLAIN_OBJECTS.replace(
        "int n = h.inner.count + h.many[1].count;",
        "thread Tag* p = &h.many[1];\n  int n = h.inner.count + p->count;",
    )

    output = _translate(tmp_path, code, "metal")

    assert "int n = 3 + 3;" in output
    assert "->count" not in output and ").count" not in output


def test_empty_tag_static_value_through_parameter(tmp_path):
    code = """
    #include <metal_stdlib>
    using namespace metal;
    struct tag_true { static constexpr constant bool value = true; };
    float pick(const device float* src, tag_true t) {
      return t.value ? src[1] : src[0];
    }
    kernel void k(const device float* src [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
      out[tid] = pick(src, tag_true{});
    }
    """

    output = _translate(tmp_path, code, "directx")

    # The HLSL struct is `struct tag_true { int _crossgl_empty; };`.
    assert "t.value" not in output
    assert re.search(r"return \(true \? src\[", output)


def test_non_static_member_reads_are_unchanged(tmp_path):
    code = PLAIN_OBJECTS.replace(
        "static constexpr constant int count = 3;", "int count;"
    ).replace(
        "h.inner.payload = 2.0f;", "h.inner.payload = 2.0f;\n  h.inner.count = 5;"
    )

    output = _translate(tmp_path, code, "directx")

    assert "h.inner.count" in output
    assert "t.count" in output
