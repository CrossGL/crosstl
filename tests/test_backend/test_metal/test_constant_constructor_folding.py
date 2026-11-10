"""Constructor calls in the initializers of ``static constexpr`` members.

The member is constant-initialized, so the importer folds a constructor whose body is empty into the CrossGL
aggregate of its member values instead of calling the runtime constructor factory. Every use of the member reads
the same folded value, including uses that the importer inlines.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

COMPLEX = """
template <typename T>
struct complex_t {
  T real;
  T imag;
  T bias = T(0.5f);

  constexpr complex_t(T real, T imag) thread : real(real), imag(imag) {};

  template <typename U, typename = typename enable_if<is_convertible_v<U, T>>::type>
  constexpr complex_t(U x) thread : real(static_cast<T>(x) * T(2)), imag(static_cast<T>(7)) {}
};
"""

STATIC_MEMBERS = COMPLEX + """
template <typename U>
struct Sum {
  static constexpr constant U init = U(3);
  static constexpr constant U pair = U(1.0f, 2.0f);
};

kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  complex_t<float> value = Sum<complex_t<float>>::init;
  complex_t<float> pair = Sum<complex_t<float>>::pair;
  out[i] = value.real * 100.0f + value.imag * 10.0f + value.bias + pair.imag * 1000.0f
      + float(i) * 10000.0f;
}
"""

# 6 * 100 + 7 * 10 + 0.5 + 2 * 1000 for thread 0, plus 10000 per thread.
EXPECTED = tuple(2670.5 + 10000.0 * index for index in range(4))


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _static_initializer(crossgl, member):
    match = re.search(
        rf"static constant complex_t_float {member} = (?P<value>[^;]*);", crossgl
    )
    assert match, crossgl
    return match.group("value")


def _run(tool, *args):
    executable = shutil.which(tool)
    if executable is None:
        pytest.skip(f"{tool} is not installed")
    result = subprocess.run([executable, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _run_metal_kernel(source, count):
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    options = metal.MTLCompileOptions.new()
    options.setMathMode_(metal.MTLMathModeSafe)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_("k"), None
    )
    assert pipeline is not None, error
    buffer = device.newBufferWithLength_options_(4 * count, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(count, 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack(f"{count}f", buffer.contents().as_buffer(4 * count))


def test_static_member_constructor_calls_fold_to_member_values(tmp_path):
    crossgl = _translate(tmp_path, STATIC_MEMBERS)
    init = _static_initializer(crossgl, "init")
    pair = _static_initializer(crossgl, "pair")
    assert init.startswith("complex_t_float(") and "crosstl_ctor" not in init
    assert pair.startswith("complex_t_float(") and "crosstl_ctor" not in pair
    # The converting constructor scales its argument, sets imag to 7 and leaves
    # bias to its default member initializer.
    assert "(3)" in init and "* float(2)" in init and "7" in init
    assert init.endswith("float(0.5f))")
    assert re.search(r"complex_t_float\(float\(1\.0f\), float\(2\.0f\), ", pair)


def test_inlined_static_member_uses_the_constructor_value(tmp_path):
    # The use site inlines the member's value. Before the fix it repeated the
    # unconverted spelling `complex_t_float(3)`, which every generator reads as
    # field-wise aggregate initialization (real = 3, imag = 0).
    crossgl = _translate(tmp_path, STATIC_MEMBERS)
    init = _static_initializer(crossgl, "init")
    assert f"complex_t_float value = ({init});" in crossgl
    assert "complex_t_float(3)" not in crossgl


def test_static_member_initializers_compile_and_read_back_on_metal(tmp_path):
    metal = _translate(tmp_path, STATIC_MEMBERS, "metal")
    assert "crosstl_ctor_complex_t_float_2" not in metal
    artifact = tmp_path / "roundtrip.metal"
    artifact.write_text(metal, encoding="utf-8")
    _run(
        "xcrun",
        "--sdk",
        "macosx",
        "metal",
        "-Werror",
        "-fno-fast-math",
        "-c",
        str(artifact),
        "-o",
        str(tmp_path / "roundtrip.air"),
    )
    assert _run_metal_kernel(metal, len(EXPECTED)) == EXPECTED


def test_static_member_initializers_compile_for_directx(tmp_path):
    hlsl = tmp_path / "kernel.hlsl"
    hlsl.write_text(_translate(tmp_path, STATIC_MEMBERS, "directx"), encoding="utf-8")
    _run(
        "dxc",
        "-T",
        "cs_6_0",
        "-E",
        "CSMain",
        str(hlsl),
        "-Fo",
        str(tmp_path / "k.dxil"),
    )


def test_static_member_initializers_compile_for_opengl(tmp_path):
    glsl = tmp_path / "kernel.comp"
    glsl.write_text(_translate(tmp_path, STATIC_MEMBERS, "opengl"), encoding="utf-8")
    _run("glslangValidator", "-S", "comp", str(glsl))


@pytest.mark.parametrize(
    "constructor",
    [
        # Statements in the body run after the mem-initializers.
        "constexpr Cell(float v) thread : a(v), b(v) { b = b + 1.0f; }",
        # `b(a)` reads the member `a`, which only exists inside the factory.
        "constexpr Cell(float v) thread : a(v), b(a) {}",
    ],
)
def test_constructors_outside_the_fold_keep_the_factory_call(tmp_path, constructor):
    source = f"""
struct Cell {{
  float a;
  float b;
  {constructor}
}};
struct Holder {{
  static constexpr constant Cell init = Cell(2.0f);
}};
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
  Cell c = Holder::init;
  out[i] = c.a + c.b;
}}
"""
    crossgl = _translate(tmp_path, source)
    match = re.search(r"static constant Cell init = (?P<value>[^;]*);", crossgl)
    assert match, crossgl
    assert match.group("value") == "crosstl_ctor_Cell_1(2.0f)"
    assert "Cell c = (crosstl_ctor_Cell_1(2.0f));" in crossgl


CONVERTED_PARAMETER = """
struct Cell {
  float a;
  float b;
  constexpr Cell(int v) thread : a(v), b(v * 2) {}
};
struct Holder {
  static constexpr constant Cell init = Cell(2.75f);
};
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  Cell c = Holder::init;
  out[i] = c.a * 10.0f + c.b + float(i) * 100.0f;
}
"""


def test_folded_arguments_convert_to_the_parameter_type(tmp_path):
    # The int parameter holds 2, not 2.75, so a = 2 and b = 4.
    crossgl = _translate(tmp_path, CONVERTED_PARAMETER)
    match = re.search(r"static constant Cell init = (?P<value>[^;]*);", crossgl)
    assert match, crossgl
    assert match.group("value") == "Cell(float(int(2.75f)), float(int(2.75f) * 2))"
    metal = _translate(tmp_path, CONVERTED_PARAMETER, "metal")
    assert _run_metal_kernel(metal, 4) == tuple(24.0 + 100.0 * i for i in range(4))
