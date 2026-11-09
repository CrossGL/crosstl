"""Imported Metal explicit specializations carry ``@explicit_specialization``.

The importer keeps ``template <typename T> T twice(T)`` generic for the CrossGL and Metal targets and emits
``template <> float twice<float>(float)`` as a concrete function. The annotation lets generators call that function
for every ``twice<float>`` instantiation, including ones a generic caller such as ``apply<T>`` only reaches once it
is instantiated.
"""

import crosstl

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

SOURCE = """
template <typename T> T twice(T x) { return x * 2; }
template <> float twice<float>(float x) { return x + x + 1.0f; }
template <typename U, typename T> U cast_to(T v) { return U(v); }
template <> bool cast_to<bool, float>(float v) { return (as_type<uint>(v) & 0x7FFFFFFF) != 0; }
template <typename T> T apply(T x) { return twice(x); }
template <typename T> bool truthy(T x) { return cast_to<bool>(x); }

kernel void k(device float* f [[buffer(0)]], device bool* b [[buffer(1)]],
              device int* n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  f[i] = apply(f[i]);
  b[i] = truthy(f[i]);
  n[i] = apply(n[i]);
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + SOURCE)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def test_specializations_are_annotated(tmp_path):
    cgl = _translate(tmp_path, "cgl")
    assert (
        "float twice_u3cfloat_u3e(float x) @explicit_specialization(twice, float)"
        in cgl
    )
    assert "@explicit_specialization(cast_to, bool, float)" in cgl


def test_generic_callers_reach_the_specializations(tmp_path):
    metal = _translate(tmp_path, "metal")
    assert "return twice_u3cfloat_u3e(x);" in metal
    assert "return cast_to_u3cbool_u2cfloat_u3e(x);" in metal
    # int has no specialization, so apply<int> keeps the primary's body.
    assert "return twice_int(x);" in metal
    assert "twice_float" not in metal
