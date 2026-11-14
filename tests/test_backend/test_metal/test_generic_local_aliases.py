"""Body-local aliases of dependent types inside generic Metal functions.

``using U = typename elem<device T*>::type;`` inside ``template <typename T>`` names ``T`` once the member alias
is resolved through the matching partial specialization.  Every later use, as a declaration type or as a
constructor, must name that generic type.  An alias that CrossGL cannot spell must be refused where it is used,
instead of reaching the output as an undeclared name.
"""

import pytest

import crosstl
from crosstl.backend.Metal.generic_local_aliases import (
    MetalLocalTypeAliasResolutionError,
)

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

ELEMENT_TRAIT = """
template <typename T> struct elem { using type = float; };
template <typename T> struct elem<device T*> { using type = T; };
"""

STORE = ELEMENT_TRAIT + """
template <typename T>
void store(device T* dst, float v) {
  using U = typename elem<device T*>::type;
  U a = U(v);
  dst[0] = a + U(1.0f);
}

kernel void k(device half* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  store(out, float(id));
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _body_lines(code, marker):
    return [line.strip() for line in code.splitlines() if marker in line]


def test_partial_specialization_member_alias_names_the_type_parameter(tmp_path):
    code = _translate(tmp_path, STORE, "cgl")

    assert "U(" not in code
    assert "U a" not in code
    assert _body_lines(code, " a = ") == ["T a = T(v);"]
    assert _body_lines(code, "dst[0]") == ["dst[0] = a + T(1.0f);"]


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_specialized_targets_use_the_partial_specialization_element(tmp_path, backend):
    code = _translate(tmp_path, STORE, backend)

    assert "U(" not in code
    assert "U a" not in code
    if backend == "metal":
        assert "half a = half(v);" in code


def test_alias_to_user_struct_template_names_the_generic_struct(tmp_path):
    source = """
template <typename T> struct wrap { T value; };
template <typename T> struct tr { using type = wrap<T>; };

template <typename T>
T direct(T v) {
  using U = wrap<T>;
  U w;
  w.value = v;
  return w.value;
}

template <typename T>
T member(T v) {
  using U = typename tr<T>::type;
  U w;
  w.value = v;
  return w.value;
}

kernel void k(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  out[id] = direct(float(id)) + member(float(id));
}
"""
    code = _translate(tmp_path, source, "cgl")

    assert "U w" not in code
    assert _body_lines(code, " w;") == ["wrap<T> w;", "wrap<T> w;"]

    metal = _translate(tmp_path, source, "metal")
    assert "U w" not in metal
    assert _body_lines(metal, " w;") == ["wrap_float w;", "wrap_float w;"]


def test_alias_to_dependent_vector_constructs_the_generic_vector(tmp_path):
    source = """
template <typename T> struct vt { using type = vec<T, 4>; };

template <typename T>
T widen(T v) {
  using V = typename vt<T>::type;
  V w = V(v);
  return w.x + w.w;
}

kernel void k(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  out[id] = widen(float(id));
}
"""
    code = _translate(tmp_path, source, "cgl")

    assert "V(" not in code
    assert _body_lines(code, " w = ") == ["vec<T, 4> w = vec<T, 4>(v);"]


@pytest.mark.parametrize(
    "alias, use",
    [
        ("typename T::value_type", "U x = U(0);"),
        ("typename T::value_type", "U x;"),
        ("typename ptr<T>::type", "U p;"),
    ],
)
def test_used_alias_without_a_crossgl_spelling_is_refused(tmp_path, alias, use):
    source = f"""
template <typename T> struct ptr {{ using type = device T*; }};

template <typename T>
T pick(T v) {{
  using U = {alias};
  {use}
  return v;
}}

kernel void k(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {{
  out[id] = pick(float(id));
}}
"""
    with pytest.raises(MetalLocalTypeAliasResolutionError) as caught:
        _translate(tmp_path, source, "cgl")

    assert caught.value.alias_name == "U"
    assert "T" in caught.value.alias_type


def test_alias_resolved_by_an_explicit_specialization_is_not_refused(tmp_path):
    source = """
struct holder { using value_type = float; };

template <typename T>
float pick(float v) {
  using U = typename T::value_type;
  U x = U(1);
  return v + x;
}

kernel void k(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  out[id] = pick<holder>(float(id));
}
"""
    code = _translate(tmp_path, source, "cgl")

    assert "U x" not in code
    assert "float x = float(1);" in code


def test_unused_unresolvable_alias_is_not_refused(tmp_path):
    source = """
template <typename T>
T keep(T v) {
  using U = typename T::value_type;
  return v;
}

kernel void k(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  out[id] = keep(float(id));
}
"""
    code = _translate(tmp_path, source, "cgl")

    assert "value_type" not in code


def test_unresolved_alias_does_not_leak_into_another_function(tmp_path):
    source = """
struct U { float value; };

template <typename T>
T keep(T v) {
  using U = typename T::value_type;
  return v;
}

float fetch_value(float v) {
  U u;
  u.value = v;
  return u.value;
}

kernel void k(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  out[id] = keep(float(id)) + fetch_value(float(id));
}
"""
    code = _translate(tmp_path, source, "cgl")

    assert "U u;" in code
