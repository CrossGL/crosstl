"""Receivers declared through a struct's member type alias of a Metal library template.

``typedef typename Frag_t::mat_type mat_type; mat_type m; m.thread_elements()[i] = ...`` declares ``m`` with the
type ``Frag::mat_type`` names: when ``Frag`` declares ``typedef metal::simdgroup_matrix<T, 8, 8> mat_type;`` the
receiver is a SIMD-group matrix, and its member calls are library operations rather than struct methods. A member
alias that names no library template, or no member at all, still fails closed.
"""

import re
import shutil
import struct
import subprocess
from types import SimpleNamespace

import pytest

import crosstl
from crosstl.backend.Metal.preprocessing.member_typedef_receivers import (
    member_typedef_target,
    names_library_template,
)
from crosstl.backend.Metal.preprocessing.support import (
    METAL_BUILTIN_MEMBER_RECEIVER_TEMPLATE_RE,
    MetalStructMethodError,
)

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

TILE_MATRIX = PREFIX + """
template <typename T, int R, int C>
struct Frag {};

template <typename T>
struct Frag<T, 8, 8> {
  static constant constexpr const int kRows = 8;
  static constant constexpr const int kCols = 8;
  typedef metal::vec<T, 2> frag_type;
  typedef metal::simdgroup_matrix<T, kRows, kCols> mat_type;
};

template <typename T, int R, int C, typename Frag_t = Frag<T, 8, 8>>
struct Tile {
  typedef typename Frag_t::mat_type mat_type;
  typedef typename Frag_t::frag_type frag_type;
  frag_type val_frags[R * C];

  METAL_FUNC mat_type mat_at(const short i, const short j) thread {
    mat_type val_mat;
    for (short ii = 0; ii < 2; ++ii) {
      val_mat.thread_elements()[ii] = val_frags[i * C + j][ii];
    }
    return val_mat;
  }
};

kernel void k(device float* out [[buffer(0)]],
              uint lane [[thread_index_in_simdgroup]]) {
  Tile<float, 1, 1> tile;
  tile.val_frags[0] = float2(float(lane), 2.0f * float(lane) + 1.0f);
  simdgroup_float8x8 m = tile.mat_at(0, 0);
  simdgroup_store(m, out, 8);
}
"""

# The member alias names another alias of its own struct, which names the library template.
CHAINED_ALIAS = PREFIX + """
struct Frag {
  typedef simdgroup_matrix<float, 8, 8> base_type;
  typedef base_type mat_type;
};

struct Tile {
  float2 values;
  METAL_FUNC Frag::mat_type load() thread {
    Frag::mat_type m;
    m.thread_elements()[0] = values.x;
    m.thread_elements()[1] = values.y;
    return m;
  }
};

kernel void k(device float* out [[buffer(0)]],
              uint lane [[thread_index_in_simdgroup]]) {
  Tile tile;
  tile.values = float2(3.0f * float(lane), float(lane) - 7.0f);
  simdgroup_float8x8 m = tile.load();
  simdgroup_store(m, out, 8);
}
"""

# ``Frag::value_type`` names ``float4``, not a library object or a struct, so ``v.norm()`` is not a library call.
NON_LIBRARY_ALIAS = PREFIX + """
struct Frag {
  typedef float4 value_type;
};

struct Tile {
  float4 values;
  METAL_FUNC float total() thread {
    Frag::value_type v = values;
    return v.norm();
  }
};

kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  Tile tile;
  tile.values = float4(float(i));
  out[i] = tile.total();
}
"""

THREADS = 32
WORDS = 64


def _translate(tmp_path, source, backend="metal", name="unit.metal"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _xcrun_metal(tmp_path, source, name):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / f"{name}.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "--sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / f"{name}.air")],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    return result


def _run(source, entry):
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        source, metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    size = 4 * WORDS
    output = device.newBufferWithLength_options_(size, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(output, 0, 0)
    # One SIMD-group: every lane holds two elements of the 8x8 matrix.
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(THREADS, 1, 1), metal.MTLSizeMake(THREADS, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack(f"<{WORDS}f", output.contents().as_buffer(size))


def _entry(metal):
    found = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", metal)
    assert found, metal
    return found.group(1)


@pytest.mark.parametrize(
    "source", [TILE_MATRIX, CHAINED_ALIAS], ids=["dependent-typedef", "chained-alias"]
)
def test_member_alias_receivers_compute_the_source_results_on_metal(tmp_path, source):
    metal = _translate(tmp_path, source)
    assert ".thread_elements()" in metal, metal
    compiled = _xcrun_metal(tmp_path, metal, "translated")
    assert compiled.returncode == 0, compiled.stderr + metal
    expected = _run(source, "k")
    assert len(set(expected)) > 2, expected
    assert _run(metal, _entry(metal)) == expected


@pytest.mark.parametrize(
    "source", [TILE_MATRIX, CHAINED_ALIAS], ids=["dependent-typedef", "chained-alias"]
)
def test_member_alias_receivers_import_to_crossgl(tmp_path, source):
    crossgl = _translate(tmp_path, source, backend="cgl")
    assert "thread_elements" in crossgl or "CooperativeMatrix" in crossgl, crossgl


def test_member_alias_of_a_plain_type_still_fails_closed(tmp_path):
    with pytest.raises(MetalStructMethodError, match=r"'v\.norm'"):
        _translate(tmp_path, NON_LIBRARY_ALIAS)


def _struct(name, aliases, qualified_name=""):
    return SimpleNamespace(
        name=name, type_aliases=aliases, qualified_name=qualified_name or name
    )


def test_member_typedef_target_follows_struct_aliases():
    structs = {
        "Frag": _struct(
            "Frag",
            {"base_t": "metal::simdgroup_matrix<float, 8, 8>", "mat_t": "base_t"},
        ),
        "Tile": _struct("Tile", {"mat_t": "typename Frag::mat_t"}),
        "Inner": _struct("Inner", {"tex_t": "texture2d<float>"}, "ns::Inner"),
        "Loop": _struct("Loop", {"a": "Loop::b", "b": "Loop::a"}),
    }
    assert member_typedef_target("Tile::mat_t", structs) == (
        "metal::simdgroup_matrix<float, 8, 8>"
    )
    assert member_typedef_target("ns :: Inner :: tex_t", structs) == (
        "texture2d<float>"
    )
    assert member_typedef_target("::ns::Inner::tex_t", structs) == "texture2d<float>"
    assert member_typedef_target("other::Inner::tex_t", structs) is None
    assert member_typedef_target("Frag::missing", structs) is None
    assert member_typedef_target("Loop::a", structs) is None
    assert member_typedef_target("mat_t", structs) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("metal::simdgroup_matrix<float, 8, 8>", True),
        ("simdgroup_matrix<half,8,8>", True),
        ("texture2d<float, access::read>", True),
        ("metal::vec<float, 2>", False),
        ("simdgroup_matrix<float, 8, 8>::value_type", False),
        ("float4", False),
    ],
)
def test_names_library_template_requires_one_whole_template_id(text, expected):
    assert (
        names_library_template(text, METAL_BUILTIN_MEMBER_RECEIVER_TEMPLATE_RE)
        is expected
    )
