"""Compile-time global elision leaves functions alone.

The scan for ``constexpr`` globals of stateless struct types also matched function definitions such as
``constexpr small one_small() { ... }``, a template helper materialized for a struct that a using-declaration brings
in from a namespace. It then looked the return type up in the global namespace, found no struct there and refused
the source as an "ambiguous type". PyTorch's ``Eye.metal`` hit this for ``float8_e4m3fn``. Each case fails on that
code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

SOURCE = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    namespace lib {
    struct small {
      uchar bits;
      constexpr small(float value) : bits(uchar(value)) {}
      operator float() const { return float(bits); }
    };
    } // namespace lib
    using lib::small;

    template <typename T>
    constexpr T one() {
      return static_cast<T>(1);
    }

    template <typename T>
    kernel void fill(device T* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
      out[i] = one<T>();
    }

    template [[host_name("fill_small")]] kernel void fill<small>(device small*, uint);
    template [[host_name("fill_float")]] kernel void fill<float>(device float*, uint);
""")


def test_constexpr_functions_are_not_compile_time_globals():
    output = MetalPreprocessor().preprocess(SOURCE)

    assert "constexpr small one_small()" in output
    assert "one_small()" in output.split("fill_small", 1)[1]


def test_kernels_returning_namespaced_structs_translate(tmp_path):
    path = tmp_path / "fill.metal"
    path.write_text(SOURCE)

    output = crosstl.translate(str(path), backend="metal", format_output=False)

    assert "kernel void fill_small(" in output and "kernel void fill_float(" in output
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(output)
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(artifact), "-o", "/dev/null"],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr
