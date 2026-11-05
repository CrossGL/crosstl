"""DirectX output moves binary16 workgroup and private array values as raw bits.

The D3D11.3 functional specification lets every operation except a raw move
change a NaN's bit pattern (section 3.1.3.2), so typed ``float16_t`` loads,
stores and selections may quiet a signaling NaN that a kernel only copies.
"""

import re
import shutil
import subprocess

import pytest

from crosstl import translate
from crosstl.translator.codegen import directx_codegen
from tests.test_translator.test_codegen.test_directx_codegen import (
    assert_directx_native_16_bit_compute_validates_if_available,
)

TILE_COPY = """#include <metal_stdlib>
using namespace metal;

kernel void half_tile_copy(const device half* values [[buffer(0)]],
                           device half* results [[buffer(1)]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint group [[threadgroup_position_in_grid]]) {
  threadgroup half tile[32];
  tile[lid] = values[group * 32u + lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  half staged[2];
  staged[lid & 1u] = tile[31u - lid];
  staged[(lid & 1u) ^ 1u] = lid < 32u ? tile[lid] : half(NAN);
  results[group * 32u + 31u - lid] = staged[lid & 1u];
}
"""

SORT_NETWORK = """#include <metal_stdlib>
using namespace metal;

template <typename T>
METAL_FUNC void thread_swap(thread T& a, thread T& b) {
  T w = a;
  a = b;
  b = w;
}

template <typename T, short N>
struct ThreadSort {
  static METAL_FUNC void sort(thread T (&vals)[N]) {
    for (short i = 0; i < N; ++i) {
      for (short j = i & 1; j < N - 1; j += 2) {
        if (vals[j + 1] < vals[j]) {
          thread_swap(vals[j + 1], vals[j]);
        }
      }
    }
  }
};

METAL_FUNC void merge_step(const threadgroup half* As, const threadgroup half* Bs,
                           short A_sz, short B_sz, thread half (&vals)[4]) {
  short a_idx = 0;
  short b_idx = 0;
  for (int i = 0; i < 4; ++i) {
    auto a = (a_idx < A_sz) ? As[a_idx] : half(NAN);
    auto b = (b_idx < B_sz) ? Bs[b_idx] : half(NAN);
    bool pred = (b_idx < B_sz) && (a_idx >= A_sz || b < a);
    vals[i] = pred ? b : a;
    b_idx += short(pred);
    a_idx += short(!pred);
  }
}

kernel void sort_tile(const device half* inp [[buffer(0)]],
                      device half* out [[buffer(1)]],
                      uint lid [[thread_index_in_threadgroup]]) {
  threadgroup half tgp_vals[128];
  for (uint i = lid; i < 128u; i += 32u) {
    tgp_vals[i] = inp[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  thread half thread_vals[4];
  for (int i = 0; i < 4; ++i) {
    thread_vals[i] = tgp_vals[lid * 4u + uint(i)];
  }
  ThreadSort<half, 4>::sort(thread_vals);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; ++i) {
    tgp_vals[lid * 4u + uint(i)] = thread_vals[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup half* As = tgp_vals + (lid & ~1u) * 4u;
  merge_step(As, As + 4, 4, 4, thread_vals);
  for (int i = 0; i < 4; ++i) {
    out[lid * 4u + uint(i)] = thread_vals[i];
  }
}
"""

# Typed binary16 memory traffic, phi nodes and selections in a DXIL listing.
TYPED_HALF_TRANSPORT = re.compile(
    r"\b(?:load|store) half\b|\bphi half\b|\bselect i1 [^,]*, half\b"
    r"|x half\]|\bhalf addrspace\(3\)\*|\balloca half\b"
)
ARRAY_DECLARATION = re.compile(r"\b(uint16_t|float16_t)[234]? (\w+)\[")


def _kernel(helpers, body):
    return f"""#include <metal_stdlib>
using namespace metal;
{helpers}
kernel void k(const device half* values [[buffer(0)]], device half* results [[buffer(1)]],
              uint lid [[thread_index_in_threadgroup]]) {{
  threadgroup half tile[32];
  tile[lid] = values[lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  half staged[2];
  staged[lid & 1u] = tile[31u - lid];
  {body}
  results[lid] = staged[lid & 1u];
}}
"""


def _hlsl(root, source):
    path = root / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend="directx",
        format_output=False,
        source_options={"threadgroup_size": 32},
    )


def _arrays(generated):
    """Map each declared binary16 array to the element type it is declared with."""
    found = {}
    for line in generated.splitlines():
        if "(" in line.split("[")[0]:
            continue
        match = ARRAY_DECLARATION.search(line)
        if match:
            found[match.group(2)] = match.group(1)
    return found


def _optimized_listing(generated, root):
    """Return DXC's optimized DXIL listing, or ``None`` without DXC."""
    dxc = shutil.which("dxc")
    if dxc is None:
        return None
    shader = root / "storage.hlsl"
    listing = root / "storage.ll"
    shader.write_text(generated, encoding="utf-8")
    result = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_2",
            "-HV",
            "2021",
            "-enable-16bit-types",
            "-O3",
            "-E",
            "CSMain",
            str(shader),
            "-Fc",
            str(listing),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return listing.read_text(encoding="utf-8")


def _assert_bit_moves(generated, root):
    assert_directx_native_16_bit_compute_validates_if_available(generated, root)
    listing = _optimized_listing(generated, root)
    if listing is not None:
        assert TYPED_HALF_TRANSPORT.findall(listing) == []


def test_workgroup_and_private_copies_move_bits(tmp_path):
    generated = _hlsl(tmp_path, TILE_COPY)
    assert _arrays(generated) == {
        "half_tile_copy_tile": "uint16_t",
        "staged": "uint16_t",
    }
    assert "half_tile_copy_tile[(31u - lid)];" in generated
    assert ": asuint16(__crossgl_round_half1(" in generated
    _assert_bit_moves(generated, tmp_path)


def test_sort_network_moves_bits_through_pointers_swaps_and_selections(tmp_path):
    generated = _hlsl(tmp_path, SORT_NETWORK)
    arrays = _arrays(generated)
    assert arrays["sort_tile_tgp_vals"] == "uint16_t"
    assert arrays["thread_vals"] == "uint16_t"
    assert "inout uint16_t vals[4]" in generated
    assert "float16_t vals[" not in generated
    _assert_bit_moves(generated, tmp_path)


LOWERED = {
    "element-copy": ("", "staged[0] = staged[1];"),
    "loop": ("", "for (int i = 0; i < 2; ++i) { staged[i] = tile[i + int(lid)]; }"),
    "nested-selection": (
        "",
        "staged[0] = lid < 3u ? staged[1] : (lid < 5u ? tile[0] : half(NAN));",
    ),
    "comparison": ("", "if (staged[0] == tile[1]) { staged[1] = tile[2]; }"),
    "array-copy": ("", "half other[2]; other = staged; results[lid + 32u] = other[0];"),
    "pointer-walk": (
        "",
        "const threadgroup half* p = tile; p += 3; staged[0] = *p; staged[1] = p[1];",
    ),
    "pointer-parameter": (
        "half first(const threadgroup half* p) { return p[0]; }",
        "staged[0] = first(tile + 4);",
    ),
    "reference-parameter": (
        "void put(thread half& x, half v) { x = v; }",
        "put(staged[0], tile[3]);",
    ),
    "vector": (
        "",
        "threadgroup half4 quad[8]; quad[lid & 7u] = half4(staged[0]);"
        " threadgroup_barrier(mem_flags::mem_threadgroup);"
        " staged[1] = quad[(lid + 1u) & 7u].y;",
    ),
}


@pytest.mark.parametrize("case", LOWERED)
def test_copies_through_supported_uses_move_bits(tmp_path, case):
    generated = _hlsl(tmp_path, _kernel(*LOWERED[case]))
    assert set(_arrays(generated).values()) == {"uint16_t"}
    assert "float16_t" not in "".join(
        line for line in generated.splitlines() if "staged[2]" in line
    )
    _assert_bit_moves(generated, tmp_path)


def test_reference_arguments_copy_back_as_bits(tmp_path):
    generated = _hlsl(tmp_path, _kernel(*LOWERED["reference-parameter"]))
    assert "float16_t __crossgl_half_bits1 = asfloat16(staged[0]);" in generated
    assert "put(__crossgl_half_bits1, asfloat16(k_tile[3]));" in generated
    assert "staged[0] = asuint16(__crossgl_half_bits1);" in generated


def test_compound_updates_encode_the_typed_result(tmp_path):
    generated = _hlsl(tmp_path, _kernel("", "staged[0] += tile[2];"))
    assert _arrays(generated) == {"k_tile": "uint16_t", "staged": "uint16_t"}
    assert (
        "staged[0] = asuint16(float16_t((asfloat16(staged[0]) + asfloat16(k_tile[2]))));"
        in generated
    )
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)


def test_array_parameters_take_the_bits_representation(tmp_path):
    generated = _hlsl(
        tmp_path,
        _kernel(
            "half sum2(thread half (&v)[2]) { return v[0] + v[1]; }",
            "staged[0] = sum2(staged);",
        ),
    )
    assert "float16_t sum2(inout uint16_t v[2]) {" in generated
    assert "return (asfloat16(v[0]) + asfloat16(v[1]));" in generated
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)


KEPT = {
    "increment": ("", "++staged[0];", {"k_tile": "uint16_t", "staged": "float16_t"}),
    "out-builtin": (
        "",
        "half whole; staged[0] = modf(staged[1], whole);",
        {"k_tile": "uint16_t", "staged": "float16_t"},
    ),
    "out-builtin-element": (
        "",
        "half part = modf(staged[1], staged[0]);",
        {"k_tile": "uint16_t", "staged": "float16_t"},
    ),
    "pointer-increment": (
        "",
        "threadgroup half* p = tile + 4; ++p[0];",
        {"k_tile": "float16_t", "staged": "uint16_t"},
    ),
}


@pytest.mark.parametrize("case", KEPT)
def test_uses_without_a_bit_rewrite_keep_their_whole_group_typed(tmp_path, case):
    helpers, body, expected = KEPT[case]
    generated = _hlsl(tmp_path, _kernel(helpers, body))
    assert _arrays(generated) == expected
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)


def test_typed_groups_are_emitted_as_without_the_lowering(tmp_path, monkeypatch):
    source = TILE_COPY.replace(
        "results[group * 32u + 31u - lid] = staged[lid & 1u];",
        "++staged[0]; ++tile[lid];\n  results[group * 32u + 31u - lid] = staged[lid & 1u];",
    )
    lowered = _hlsl(tmp_path, source)
    monkeypatch.setattr(
        directx_codegen, "lower_half_private_storage", lambda ast, map_type: ast
    )
    assert lowered == _hlsl(tmp_path, source)


@pytest.mark.parametrize("scalar", ["float", "ushort", "short", "bfloat"])
def test_other_element_types_are_unchanged(tmp_path, scalar, monkeypatch):
    source = TILE_COPY.replace("half(NAN)", f"{scalar}(0)").replace("half", scalar)
    lowered = _hlsl(tmp_path, source)
    monkeypatch.setattr(
        directx_codegen, "lower_half_private_storage", lambda ast, map_type: ast
    )
    assert lowered == _hlsl(tmp_path, source)
