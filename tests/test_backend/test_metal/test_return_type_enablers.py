"""Function templates constrained through an ``enable_if_t`` return type.

``enable_if_t<B, T>`` names ``T`` only when ``B`` holds, so a template whose return type is an enabler leaves the
overload set for arguments that make ``B`` false (C++14 [temp.deduct]/8). Overloads selected this way, an explicit
specialization declared in a namespace, and the enabler a materialized instantiation keeps in concrete code are
covered here.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
    MetalTypeTraitError,
)
from crosstl.backend.Metal.return_type_enablers import fold_concrete_enablers

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    namespace util {
    template <typename T>
    ::metal::enable_if_t<::metal::is_floating_point_v<T>, T> scale(T v) {
      return v * T(0.5);
    }
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T>, T> scale(T v) {
      return v / T(2);
    }
    template <typename T>
    metal::enable_if_t<metal::is_floating_point_v<T>, T> twice(T v) {
      return v + v;
    }
    template <>
    inline half twice(half v) {
      return v * 3.0h;
    }
    } // namespace util
    kernel void k(device float* f [[buffer(0)]],
                  device int* i [[buffer(1)]],
                  device half* h [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
      float s = util::scale(f[id]);
      f[id] = util::twice(s);
      i[id] = util::scale(i[id]);
      h[id] = util::twice(h[id]);
    }
    """)


def test_return_type_enablers_select_overloads_and_name_their_types():
    output = MetalPreprocessor().preprocess(KERNEL)
    assert "float s = ::util::scale_float(f[id]);" in output
    assert "i[id] = ::util::scale_int(i[id]);" in output
    assert "inline float scale_float(float v) {\n  return v * float(0.5);" in output
    assert "inline int scale_int(int v) {\n  return v / int(2);" in output
    assert "enable_if" not in output


def test_namespace_scope_explicit_specialization_supplies_the_body():
    output = MetalPreprocessor().preprocess(KERNEL)
    # The specialization is no competing overload; it is the body of twice<half>.
    assert "inline half twice_half(half v) {\n  return v * 3.0h;" in output
    assert "inline float twice_float(float v) {\n  return v + v;" in output


def test_return_type_enabler_with_two_enabled_overloads_is_refused():
    ambiguous = KERNEL.replace(
        "metal::enable_if_t<metal::is_integral_v<T>, T> scale",
        "metal::enable_if_t<!metal::is_integral_v<T>, T> scale",
    )
    with pytest.raises(
        MetalTemplateSpecializationError,
        match="multiple constrained overloads are enabled",
    ):
        MetalPreprocessor().preprocess(ambiguous)


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("::metal::enable_if_t<true, float> f(float v)", "float f(float v)"),
        ("metal::enable_if_t<(8 < 16) && !false, int2> f()", "int2 f()"),
        ("typename metal::enable_if<true, half>::type f()", "half f()"),
        ("enable_if_t<true> f()", "void f()"),
    ],
)
def test_concrete_enablers_fold_to_the_type_they_name(spelling, expected):
    assert fold_concrete_enablers(MetalPreprocessor(), spelling + " {}") == (
        expected + " {}"
    )


def test_enablers_inside_templates_are_left_to_instantiation():
    code = "template <typename T> metal::enable_if_t<sizeof(T) == 4, T> f(T v) { return v; }"
    assert fold_concrete_enablers(MetalPreprocessor(), code) == code


@pytest.mark.parametrize(
    "spelling, reason",
    [
        ("metal::enable_if_t<false, float> f()", "its condition is false"),
        ("metal::enable_if_t<N == 2, float> f()", "does not fold to a constant"),
    ],
)
def test_concrete_enablers_that_name_no_type_are_refused(spelling, reason):
    with pytest.raises(MetalTypeTraitError, match=reason) as excinfo:
        fold_concrete_enablers(MetalPreprocessor(), spelling + " {}")
    assert excinfo.value.trait == "enable_if"
    assert excinfo.value.source_location["line"] == 1


def test_a_source_defined_enable_if_t_keeps_its_own_meaning():
    code = (
        "template <bool B, typename T> struct my_enable_if {};\n"
        "template <bool B, typename T = void> using enable_if_t = typename my_enable_if<B, T>::type;\n"
        "enable_if_t<true, float> f() {}"
    )
    assert fold_concrete_enablers(MetalPreprocessor(), code) == code


_VALIDATORS = {
    "opengl": (
        "enablers.comp",
        [["glslangValidator", "-S", "comp", "enablers.comp"]],
    ),
    "vulkan": (
        "enablers.spvasm",
        [
            ["spirv-as", "--target-env", "vulkan1.1", "enablers.spvasm", "-o", "e.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "e.spv"],
        ],
    ),
    "directx": (
        "enablers.hlsl",
        [["dxc", "-T", "cs_6_2", "-enable-16bit-types", "-E", "k", "enablers.hlsl"]],
    ),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_return_type_enablers_reach_every_target(tmp_path, backend):
    source = tmp_path / "enablers.metal"
    source.write_text(KERNEL, encoding="utf-8")
    code = crosstl.translate(str(source), backend=backend)
    file_name, commands = _VALIDATORS[backend]
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
