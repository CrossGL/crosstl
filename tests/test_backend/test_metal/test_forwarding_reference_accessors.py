"""Reference accessors that forward to another accessor of the same struct.

``thread T& at_t(i, j) { return at(j, i); }`` names the storage ``at(j, i)``
names. It used to be refused because its body is a call rather than a member
access; it is now inlined through the accessor it forwards to.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError

FORWARDING = """
#include <metal_stdlib>
using namespace metal;

struct Tile {
  float2 frags[4];
  thread float2& at(const short i, const short j) thread { return frags[i * 2 + j]; }
  const thread float2& at(const short i, const short j) const thread {
    return frags[i * 2 + j];
  }
  thread float2& at_t(const short i, const short j) thread { return at(j, i); }
  const thread float2& at_t(const short i, const short j) const thread {
    return at(j, i);
  }
  thread float2& at_tt(const short i, const short j) thread { return this->at_t(j, i); }
  float2 sum() const thread { return at_t(0, 1) + at(1, 1); }
  void bump() thread { at_t(1, 0) *= 2.0f; }
};

kernel void k(device float2* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  Tile t;
  for (short i = 0; i < 4; i++) {
    t.frags[i] = float2(float(i), float(tid));
  }
  t.at_tt(0, 1) += float2(5.0f);
  t.at_t(1, 1) -= float2(1.0f);
  t.bump();
  out[tid] = t.sum();
}
"""


def _compact(text):
    return re.sub(r"[\s()]", "", text)


def test_forwarding_accessors_inline_the_storage_they_forward():
    output = _compact(MetalPreprocessor().preprocess(FORWARDING))
    kernel = output.split("kernelvoidk", 1)[1]

    # at_tt(0, 1) -> at_t(1, 0) -> at(0, 1); at_t(1, 1) -> at(1, 1).
    assert "t.frags[0*2+1]+=float25.0f;" in kernel
    assert "t.frags[1*2+1]-=float21.0f;" in kernel
    # Inside members: at_t(1, 0) -> at(0, 1) and at_t(0, 1) -> at(1, 0).
    assert "self.frags[0*2+1]*=2.0f;" in output
    assert "returnself.frags[1*2+0]+self.frags[1*2+1];" in output
    assert not re.search(r"Tile__at\w*\(", kernel)


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_forwarding_accessors_translate(tmp_path, backend):
    source = tmp_path / "forwarding.metal"
    source.write_text(FORWARDING)

    output = _compact(
        crosstl.translate(
            str(source), backend=backend, source_backend="metal", format_output=False
        )
    )

    assert "t.frags[0*2+1]+=" in output
    assert "t.frags[1*2+1]-=" in output
    assert "self.frags[0*2+1]*=2.0" in output
    assert "at_t" not in output and "at_tt" not in output


@pytest.mark.parametrize(
    "members, call",
    [
        # A cycle never reaches storage.
        (
            "thread float2& a(short i) thread { return b(i); }\n"
            "thread float2& b(short i) thread { return a(i); }",
            "t.a(0) = float2(1.0f);",
        ),
        # The forwarded argument has a side effect.
        (
            "thread float2& at(short i) thread { return v[i]; }\n"
            "thread float2& next(short i) thread { return at(i++); }",
            "t.next(0) = float2(1.0f);",
        ),
        # The callee does not name one storage element.
        (
            "thread float2& pick(short i) thread { return i == 0 ? v[0] : v[1]; }\n"
            "thread float2& fwd(short i) thread { return pick(i); }",
            "t.fwd(0) = float2(1.0f);",
        ),
        # The callee is not a member of this struct.
        (
            "thread float2& fwd(short i) thread { return elsewhere(v, i); }",
            "t.fwd(0) = float2(1.0f);",
        ),
    ],
)
def test_unprovable_forwarding_still_fails_closed(members, call):
    code = f"""
    thread float2& elsewhere(thread float2* v, short i) {{ return v[i]; }}
    struct T {{
      float2 v[2];
      {members}
    }};
    kernel void k(device float2* out [[buffer(0)]]) {{
      T t;
      {call}
      out[0] = t.v[0];
    }}
    """

    with pytest.raises(MetalStructMethodError) as caught:
        MetalPreprocessor().preprocess(code)

    assert caught.value.reason == "reference-return-identity-unsupported"
