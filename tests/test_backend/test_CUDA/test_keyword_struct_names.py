"""Tests that CUDA structs named after CrossGL keywords translate correctly.

The CUDA importer renames a struct whose name would be rejected as an
identifier by the CrossGL parser. Each renamed declaration and every use
of the struct as a type must share the same renamed name.

``kernel`` IS in CUDA's narrow ``CROSSGL_RESERVED_IDENTIFIERS`` set, so it
is renamed everywhere (not just struct names), and a CUDA local named
``kernel`` must also be renamed. ``half``, ``fn``, and ``mesh`` are NOT in
the narrow set but ARE CrossGL keywords, so they must be renamed only when
used as struct names (via the ``_CROSSGL_ALL_KEYWORDS`` + ``cuda_record_names``
branch).
"""

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

# Keywords that only appear as struct names (not in CROSSGL_RESERVED_IDENTIFIERS).
STRUCT_ONLY_KEYWORDS = ("half", "fn", "mesh")

# Keywords in CROSSGL_RESERVED_IDENTIFIERS that must be renamed everywhere.
RESERVED_KEYWORDS = ("let", "var", "kernel")

ALL_STRUCT_KEYWORDS = (*STRUCT_ONLY_KEYWORDS, *RESERVED_KEYWORDS)

STRUCT_SOURCE = """\
struct {name} {{
    float a;
}};

__device__ float get({name} value) {{
    return value.a;
}}

__global__ void probe(float* outp) {{
    {name} t;
    t.a = 2.0f;
    outp[threadIdx.x] = get(t);
}}
"""


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "probe.cu"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("name", ALL_STRUCT_KEYWORDS)
def test_cuda_keyword_struct_renamed_in_declaration(tmp_path, name):
    """The struct declaration uses the renamed name."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name))

    assert f"struct {name}_ {{" in code
    assert f"struct {name} {{" not in code


@pytest.mark.parametrize("name", ALL_STRUCT_KEYWORDS)
def test_cuda_keyword_struct_renamed_in_parameter(tmp_path, name):
    """The parameter type uses the same renamed name as the declaration."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name))

    assert f"{name}_ value" in code


@pytest.mark.parametrize("name", ALL_STRUCT_KEYWORDS)
def test_cuda_keyword_struct_renamed_in_local_variable(tmp_path, name):
    """The local variable type uses the same renamed name as the declaration."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name))

    # The local variable ``t`` is declared with the renamed type
    assert f"var t: {name}_;" in code


@pytest.mark.parametrize("name", ALL_STRUCT_KEYWORDS)
def test_cuda_keyword_struct_crossgl_parses(tmp_path, name):
    """The translated CrossGL output must be syntactically valid."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name))
    Parser(Lexer(code).tokens).parse()


@pytest.mark.parametrize("name", ALL_STRUCT_KEYWORDS)
def test_cuda_keyword_struct_translates_to_opengl(tmp_path, name):
    """The renamed struct must survive translation to OpenGL GLSL."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name), backend="opengl")

    assert f"{name}_" in code


def test_cuda_kernel_local_renamed(tmp_path):
    """A local variable named ``kernel`` must still be renamed."""
    source = """\
__device__ float use_kernel() {
    int kernel = 42;
    return (float)kernel;
}

__global__ void probe(float* outp) {
    outp[threadIdx.x] = use_kernel();
}
"""
    code = _translate(tmp_path, source)
    # ``kernel`` as a local must be renamed (it is in CROSSGL_RESERVED_IDENTIFIERS)
    assert "var kernel_:" in code


def test_cuda_float_type_not_renamed(tmp_path):
    """The ``float`` type spelling must not be renamed to ``float_``."""
    source = """\
__global__ void probe(float* outp) {
    float x = 1.0f;
    outp[threadIdx.x] = x;
}
"""
    code = _translate(tmp_path, source)
    assert "float_" not in code
