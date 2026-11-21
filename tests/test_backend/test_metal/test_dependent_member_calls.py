"""Qualified calls whose qualifier depends on a template parameter of a generic function.

``F::add(d, a)`` and ``Tile<T>::Frag_type::add(d, a)`` in a function template call a member of a type known only once
the template is instantiated ([temp.dep.type], [temp.res]/6). Such a statement is an expression, not a declaration:
it has no declarator ([dcl.decl]). CrossGL generic functions have no dependent member call, so the Metal importer
refuses one instead of dropping the qualifier, which would bind the call to an unrelated unqualified ``add``.
Instantiating the function first gives the qualifier a concrete owner.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.MetalAst import FunctionCallNode
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.static_member_calls import MetalDependentMemberCallRefusal

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

PLUS = """
struct Plus {
  static METAL_FUNC void add(thread float& d, thread float& a) { d += a; }
};

METAL_FUNC void add(thread float& d, thread float& a) { d -= a; }
"""

DEDUCED = PLUS + """
template <class F>
METAL_FUNC void apply(F f, thread float& d, thread float& a) {
  F::add(d, a);
}

kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  float a = float(tid);
  float b = 2.0f;
  Plus p;
  apply(p, a, b);
  out[tid] = a;
}
"""

EXPLICIT = DEDUCED.replace(
    "Plus p;\n  apply(p, a, b);", "Plus p;\n  apply<Plus>(p, a, b);"
)

NESTED_ALIAS = """
template <typename T>
struct Frag {
  static METAL_FUNC void add(thread T& d, thread T& a) { d += a; }
};

template <typename T, class F = Frag<T>>
struct Tile {
  using Frag_type = F;
  T v[2];
};

template <typename T>
METAL_FUNC void tile_add(thread Tile<T>& D, thread Tile<T>& A) {
  Tile<T>::Frag_type::add(D.v[0], A.v[0]);
}

kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  Tile<float> P, Q;
  P.v[0] = float(tid);
  Q.v[0] = 2.0f;
  tile_add(P, Q);
  out[tid] = P.v[0];
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PRELUDE + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _function(source, name):
    tokens = MetalLexer(PRELUDE + source).tokenize()
    program = MetalParser(tokens).parse()
    return next(function for function in program.functions if function.name == name)


def _statements(function):
    body = function.body
    return body.statements if hasattr(body, "statements") else body


def test_qualified_template_id_call_statement_is_an_expression():
    (statement,) = _statements(_function(NESTED_ALIAS, "tile_add"))
    assert isinstance(statement, FunctionCallNode)
    assert statement.name == "Tile<T>::Frag_type::add"


@pytest.mark.parametrize(
    "source, call", [(DEDUCED, "F::add"), (NESTED_ALIAS, "Tile<T>::Frag_type::add")]
)
@pytest.mark.parametrize("backend", ["metal", "cgl"])
def test_dependent_member_call_in_generic_function_is_refused(
    tmp_path, source, call, backend
):
    with pytest.raises(MetalDependentMemberCallRefusal, match=re.escape(f"'{call}'")):
        _translate(tmp_path, source, backend)


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_instantiated_targets_call_the_member(tmp_path, backend):
    # Template-hostile targets instantiate `apply` first, so `F::add` names `Plus::add`.
    code = _translate(tmp_path, DEDUCED, backend)
    assert "Plus_add(d, a);" in code.replace("Plus__add", "Plus_add"), code
    assert "{ add(d, a); }" not in code


def test_explicit_template_arguments_call_the_member(tmp_path):
    metal = _translate(tmp_path, EXPLICIT, "metal")
    assert "Plus__add(d, a);" in metal, metal
    assert " add(d, a);" not in metal


UNCALLED = PLUS + """
template <class F>
METAL_FUNC void apply(F f, thread float& d, thread float& a) {
  F::add(d, a);
}

kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  float a = float(tid);
  float b = 2.0f;
  Plus::add(a, b);
  out[tid] = a;
}
"""


@pytest.mark.parametrize("backend", ["metal", "cgl", "opengl"])
def test_an_uninstantiated_template_with_a_dependent_call_is_left_out(
    tmp_path, backend
):
    # Nothing instantiates `apply`, so it contributes no code ([temp.inst]/4) and is not refused.
    code = _translate(tmp_path, UNCALLED, backend)
    assert "apply" not in code
    assert re.search(r"Plus_+add\(", code), code


def test_a_named_template_with_a_dependent_call_is_still_refused(tmp_path):
    # A mention keeps the template: here a call that Metal output keeps generic.
    source = UNCALLED.replace("Plus::add(a, b);", "Plus p;\n  apply(p, a, b);")
    with pytest.raises(MetalDependentMemberCallRefusal):
        _translate(tmp_path, source, "cgl")
