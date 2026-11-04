"""An empty brace initializer of a plain enumeration is zero on the targets that write it as ``int`` (#3752).

C++ value-initializes ``Mode m = {};`` to zero, and the generators write a variable of a plain enumeration as ``int``
(``i32`` in WGSL). HLSL output copied ``int m = {};`` (DXC rejects it), WGSL output wrote ``array()`` (naga rejects
it) and GLSL output refused it as an aggregate initializer. CUDA output declares ``Mode m`` with the enumeration type,
for which ``{}`` is already zero, and an enumeration whose variants carry data keeps its braces.
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


@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_empty_brace_initializer_of_an_enum_with_an_underlying_type_is_int_zero(
    tmp_path, backend
):
    # The generators write a ``Mode`` variable as ``int`` whatever type the enumeration names; glslang rejects
    # ``int m = 0u;``.
    source = SHADER.replace("enum Mode {", "enum Mode : uint {")
    output = translate(tmp_path, source, backend, ".cgl")
    assert EXPECTED[backend] in output


@pytest.mark.parametrize(
    "suffix, source", [(".cgl", SHADER), (".cu", KERNEL)], ids=["crossgl", "cuda"]
)
def test_cuda_output_keeps_the_empty_brace_initializer_of_an_enum(
    tmp_path, suffix, source
):
    # CUDA output declares ``Mode m`` with the enumeration type: ``{}`` is zero there, and ``0`` does not convert.
    output = translate(tmp_path, source, "cuda", suffix)
    assert "Mode m = {};" in output


PAYLOAD_ENUM = textwrap.dedent("""
    shader P {
        enum Mode { First(float), Second }
        compute {
            buffer float* outp;
            void main() {
                Mode m = {};
                outp[0] = 1.0;
            }
        }
    }
""")


@pytest.mark.parametrize("backend", ["metal", "hip", "slang"])
def test_empty_brace_initializer_of_an_enum_with_payloads_keeps_its_braces(
    tmp_path, backend
):
    # Such an enumeration is a structure on these targets, which value-initialize it from ``{}``.
    output = translate(tmp_path, PAYLOAD_ENUM, backend, ".cgl")
    assert "Mode m = {};" in output
    assert "Mode m = 0;" not in output
