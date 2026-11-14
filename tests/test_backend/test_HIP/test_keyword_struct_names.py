"""Tests that HIP structs named after CrossGL keywords translate correctly.

The HIP importer renames a struct whose name would be rejected as an
identifier by the CrossGL parser. Each renamed declaration and every use
of the struct as a type must share the same renamed name.

``kernel`` is intentionally **not** in HIP's narrow
``CROSSGL_RESERVED_IDENTIFIERS`` set: HIP uses ``kernel`` as the name of
kernel-launch handles, so renaming every identifier called ``kernel`` would
break launch-related code. Only struct/type uses are handled via the
``_CROSSGL_ALL_KEYWORDS`` + ``record_definitions`` branch. This module
verifies that a function named ``kernel`` keeps its name.
"""

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

# ``kernel`` is NOT in HIP's CROSSGL_RESERVED_IDENTIFIERS, so it is excluded
# from the struct-rename tests (struct kernel is renamed via _CROSSGL_ALL_KEYWORDS).
# ``let``, ``var``, ``match`` are in the narrow set.
STRUCT_ONLY_KEYWORDS = ("half", "fn", "mesh", "kernel")
RESERVED_KEYWORDS = ("let", "var", "match")
ALL_STRUCT_KEYWORDS = (*STRUCT_ONLY_KEYWORDS, *RESERVED_KEYWORDS)
KEYWORDS = ALL_STRUCT_KEYWORDS

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

FUNCTION_SOURCE = """\
__device__ float {name}(float x) {{
    return x * 2.0f;
}}

__global__ void probe(float* outp) {{
    outp[threadIdx.x] = {name}(1.0f);
}}
"""


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "probe.hip"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("name", KEYWORDS)
def test_hip_keyword_struct_renamed_in_declaration(tmp_path, name):
    """The struct declaration uses the renamed name."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name))

    assert f"struct {name}_ {{" in code
    assert f"struct {name} {{" not in code


@pytest.mark.parametrize("name", KEYWORDS)
def test_hip_keyword_struct_renamed_in_parameter(tmp_path, name):
    """The parameter type uses the same renamed name as the declaration."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name))

    assert f"{name}_ value" in code


@pytest.mark.parametrize("name", KEYWORDS)
def test_hip_keyword_struct_renamed_in_local_variable(tmp_path, name):
    """The local variable type uses the same renamed name as the declaration."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name))

    # The local variable ``t`` is declared with the renamed type
    assert f"var t: {name}_;" in code


@pytest.mark.parametrize("name", KEYWORDS)
def test_hip_keyword_struct_crossgl_parses(tmp_path, name):
    """The translated CrossGL output must be syntactically valid."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name))
    Parser(Lexer(code).tokens).parse()


@pytest.mark.parametrize("name", KEYWORDS)
def test_hip_keyword_struct_translates_to_opengl(tmp_path, name):
    """The renamed struct must survive translation to OpenGL GLSL."""
    code = _translate(tmp_path, STRUCT_SOURCE.format(name=name), backend="opengl")

    assert f"{name}_" in code


def test_hip_function_named_kernel_keeps_name(tmp_path):
    """A *function* (or kernel) named ``kernel`` must NOT be renamed.

    ``kernel`` was removed from HIP's ``CROSSGL_RESERVED_IDENTIFIERS`` because
    including it renamed every HIP identifier called ``kernel``, including
    ``__global__ void kernel(...)`` launch entry points. This test verifies
    that the fix holds: ``__global__ void kernel(...)`` must remain as
    ``kernel`` in the CrossGL output.
    """
    source = """\
__global__ void kernel(float* outp) {
    outp[threadIdx.x] = 1.0f;
}
"""
    code = _translate(tmp_path, source)
    # The function name must appear unchanged (not renamed to kernel_)
    assert "fn kernel(" in code
    assert "fn kernel_(" not in code
