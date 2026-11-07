"""Deduced calls that select explicit function template specializations.

A call reaches ``template <> float twice<float>(float)`` whenever the primary's template arguments, explicit or
deduced from the call's arguments, are exactly ``float`` ([temp.expl.spec]/1, [temp.deduct.call]). The importer
keeps the primary generic for the CrossGL and Metal targets, so a deduced call has to name the specialization
itself; otherwise the generators instantiate the primary's body.
"""

import crosstl

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

SOURCE = """
template <typename T> T twice(T x) { return x * 2; }
template <> float twice<float>(float x) { return x + x + 1.0f; }
template <typename U, typename T> U cast_to(T v) { return U(v); }
template <> bool cast_to<bool, float>(float v) { return (as_type<uint>(v) & 0x7FFFFFFF) != 0; }
template <typename T> void put(device T* p, uint i, T v) { p[i] = v; }
template <> void put<float>(device float* p, uint i, float v) { p[i] = v * 2.0f; }

kernel void k(device float* f [[buffer(0)]], device int* n [[buffer(1)]],
              device bool* b [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  f[i] = twice(f[i]);
  n[i] = twice(n[i]);
  f[i + 1] = twice<float>(f[i]);
  half h = 1.0h;
  f[i + 2] = float(twice(h));
  b[i] = cast_to<bool>(f[i]);
  n[i + 1] = cast_to<int>(f[i]);
  put(f, i, 3.0f);
  put(n, i, 3);
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + SOURCE)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def test_deduced_calls_name_the_matching_specialization(tmp_path):
    cgl = _translate(tmp_path, "cgl")
    assert "buffer_store(f, i, twice_u3cfloat_u3e(buffer_load(f, i)));" in cgl
    assert "buffer_store(f, i + 1, twice_u3cfloat_u3e(buffer_load(f, i)));" in cgl
    assert "buffer_store(b, i, cast_to_u3cbool_u2cfloat_u3e(buffer_load(f, i)));" in cgl
    assert "put_u3cfloat_u3e(f, i, 3.0f);" in cgl


def test_calls_without_a_matching_specialization_keep_the_primary(tmp_path):
    cgl = _translate(tmp_path, "cgl")
    assert "buffer_store(n, i, twice(buffer_load(n, i)));" in cgl
    # half deduces T = half, which has no specialization; float's does not apply.
    assert "float(twice(h))" in cgl
    assert "cast_to<int>(buffer_load(f, i))" in cgl
    assert "put(n, i, 3);" in cgl


def test_metal_round_trip_runs_the_specialized_bodies(tmp_path):
    metal = _translate(tmp_path, "metal")
    assert "f[i] = twice_u3cfloat_u3e(f[i]);" in metal
    assert "b[i] = cast_to_u3cbool_u2cfloat_u3e(f[i]);" in metal
    assert "return x + x + 1.0;" in metal
