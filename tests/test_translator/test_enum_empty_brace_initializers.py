"""An empty brace initializer of an enumeration is the zero of its underlying type on every target (#3752).

C++ value-initializes ``Mode m = {};`` to zero, and every generator writes an enumeration-typed variable as its
underlying integer type. HLSL output copied ``int m = {};`` (DXC rejects it), WGSL output wrote ``array()`` (naga
rejects it) and GLSL output refused it as an aggregate initializer.
"""

import textwrap

import pytest

import crosstl

SHADER = textwrap.dedent("""
    shader P {
        enum Mode { First, Second }
        compute {
            buffer float* outp;
            void main() {
                Mode m = {};
                outp[0] = float(int(m));
            }
        }
    }
""")

KERNEL = textwrap.dedent("""
    enum Mode { A, B };
    extern "C" __global__ void first(float* out) {
        Mode m = {};
        out[0] = (float)m;
    }
""")

EXPECTED = {
    "directx": "int m = 0;",
    "opengl": "int m = 0;",
    "metal": "int m = 0;",
    "wgsl": "var m: i32 = 0;",
}


def translate(tmp_path, source, backend, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_empty_brace_enum_initializer_is_zero(tmp_path, backend):
    output = translate(tmp_path, SHADER, backend, ".cgl")
    assert EXPECTED[backend] in output
    assert "{}" not in output
    assert "array()" not in output


@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_empty_brace_enum_initializer_from_cuda_is_zero(tmp_path, backend):
    output = translate(tmp_path, KERNEL, backend, ".cu")
    assert EXPECTED[backend] in output
    assert "{}" not in output
    assert "array()" not in output


def test_empty_brace_enum_initializer_translates_to_spirv(tmp_path):
    translate(tmp_path, SHADER, "spirv", ".cgl")
