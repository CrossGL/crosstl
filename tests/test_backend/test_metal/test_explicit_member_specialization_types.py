"""Explicit member specializations match members spelled with the owner's type aliases.

A class template may declare a member with its own alias (``using pair_T = vec<T, 2>;``) or with an alias of another
template's member (``using scalar_T = typename Traits<T>::scalar_T;``), while an explicit specialization of that member
for one owner spells the concrete type (``float2``). C++ matches the two by type ([temp.expl.spec]). The preprocessor
compared the spellings, so the specialization was silently ignored and the primary member body was emitted for the
specialized owner, computing something else.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)

TRAITS = """
template <typename T>
struct Storage {
  using scalar_T = T;
};

template <>
struct Storage<float2> {
  using scalar_T = float;
};

template <typename In, typename Out>
struct Traits {
  using scalar_T = typename Storage<In>::scalar_T;
};
"""

OWNER = """
template <typename In, typename Out>
struct Writer {
  using scalar_T = typename Traits<In, Out>::scalar_T;
  using pair_T = vec<scalar_T, 2>;
  device pair_T* out;
  Writer(device pair_T* out_) thread : out(out_) {}
  void put(uint i, const device pair_T* w) const thread {
    out[i] = w[i];
  }
  pair_T scaled(pair_T value) const thread {
    return value;
  }
};
"""


def _source(specialization):
    return f"""
#include <metal_stdlib>
using namespace metal;
{TRAITS}
{OWNER}
{specialization}

[[kernel]] void run(
    device float2* out [[buffer(0)]],
    const device float2* w [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {{
  Writer<float, float2> writer(out);
  writer.put(i, w);
  out[i + 1] = writer.scaled(w[i]);
}}
"""


SPECIALIZED = _source("""
template <>
void Writer<float, float2>::put(uint i, const device float2* w) const thread {
  out[i] = w[i] * 2.0f;
}

template <>
metal::vec<float, 2> Writer<float, float2>::scaled(vec<float, 2> value) const thread {
  return value * 3.0f;
}
""")


def _member_body(code, member):
    """The body of the free function the owner's ``member`` lowers to."""
    match = re.search(
        rf"static inline [^\n]*\bWriter_float_float2__{member}\([^)]*\) \{{(?P<body>[^}}]*)\}}",
        code,
    )
    assert match, code
    return match.group("body")


def test_specialized_members_replace_alias_spelled_primary_members():
    code = MetalPreprocessor().preprocess(SPECIALIZED)
    assert "w[i] * 2.0f" in _member_body(code, "put")
    assert "value * 3.0f" in _member_body(code, "scaled")


@pytest.mark.parametrize("target", ["metal", "directx"])
def test_specialized_member_bodies_reach_the_targets(tmp_path, target):
    source = tmp_path / "writer.metal"
    source.write_text(SPECIALIZED, encoding="utf-8")
    code = crosstl.translate(str(source), backend=target)
    assert "2.0" in code and "3.0" in code
    if target == "metal" and shutil.which("xcrun"):
        artifact = tmp_path / "out.metal"
        artifact.write_text(code, encoding="utf-8")
        completed = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(artifact),
                "-o",
                str(tmp_path / "out.air"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0 and "unable to find utility" in completed.stderr:
            pytest.skip("the Metal toolchain is not installed")
        assert completed.returncode == 0, completed.stderr[-2000:]


def test_a_specialization_matching_no_overload_is_refused():
    source = _source("""
template <>
void Writer<float, float2>::put(uint i, const device float4* w) const thread {
  out[i] = w[i].xy;
}
""")
    with pytest.raises(MetalTemplateSpecializationError, match="matches no member"):
        MetalPreprocessor().preprocess(source)


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("vec<float,2>", "float2"),
        ("const device metal::vec<half,4>*", "const device half4*"),
        ("::metal::vec<uint,3>", "uint3"),
        ("vec<bfloat,2>", "bfloat2"),
        ("other::vec<float,2>", "other::vec<float,2>"),
        ("vec<float,5>", "vec<float,5>"),
        ("vec<T,2>", "vec<T,2>"),
    ],
)
def test_concrete_vector_spellings_compare_by_type(spelling, expected):
    assert MetalPreprocessor()._canonical_metal_vector_spelling(spelling) == expected
