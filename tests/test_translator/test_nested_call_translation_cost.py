"""Translation work stays polynomial in the nesting depth of calls.

``asint(asuint(x) + 0u)`` nested ``d`` levels deep cost a constant factor more work per level in three shared
places, so a few levels of the bit casts that wrapped integer arithmetic polyfills write took seconds:

* the HLSL generator's two type queries re-asked the type of the same operand in several branches (memoized for
  the outermost query by :mod:`crosstl.translator.codegen.type_query_memo`);
* the reference-alias lowering copied ``FunctionCallNode.arguments`` and its alias ``args`` into two lists, so every
  later walk that follows each distinct container visited the arguments twice;
* the initializer-scope renamer followed both ``arguments`` and ``args`` of the parsed call.
"""

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.codegen import directx_codegen
from crosstl.translator.codegen.reference_aliases import lower_reference_aliases
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import declaration_scopes


def _program(depth):
    expression = "x"
    for level in range(depth):
        expression = f"asint(asuint({expression}) + {level}u)"
    return (
        "shader S { compute { layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;"
        " buffer int* data; void main() { int x = data[0]; data[1] = %s; } } }"
        % expression
    )


def _parse(code):
    return Parser(Lexer(code).tokens).parse()


def _walk(node):
    yield node
    for child in node.child_nodes():
        yield from _walk(child)


def test_reference_alias_lowering_keeps_call_aliases_shared():
    lowered = lower_reference_aliases(_parse(_program(3)))
    calls = [n for n in _walk(lowered) if isinstance(n, A.FunctionCallNode)]
    assert len(calls) == 6
    for call in calls:
        assert call.args is call.arguments
        assert call.name is call.function


def test_initializer_scope_walk_follows_each_child_once():
    call = _parse(_program(1)).stages
    call = next(
        n
        for n in _walk(next(iter(call.values())).entry_point)
        if isinstance(n, A.FunctionCallNode)
    )
    children = list(declaration_scopes._children(call))
    assert sum(1 for c in children if c is call.arguments) == 1
    assert len({id(c) for c in children}) == len(children)


def test_initializer_hazard_check_visits_each_call_once(monkeypatch):
    # The check that lets the pass skip the renamer followed both ``arguments`` and ``args`` of each call, so it
    # visited the innermost call 2**depth times.
    entry = next(iter(_parse(_program(6)).stages.values())).entry_point
    calls = [n for n in _walk(entry) if isinstance(n, A.FunctionCallNode)]
    assert len(calls) == 12
    visits = {}
    children = declaration_scopes._children

    def counting(node):
        visits[id(node)] = visits.get(id(node), 0) + 1
        return children(node)

    monkeypatch.setattr(declaration_scopes, "_children", counting)
    assert not declaration_scopes._has_hazard([entry.body.statements])
    assert [visits.get(id(call)) for call in calls] == [1] * len(calls)


def _bitcast_queries(monkeypatch, tmp_path, depth):
    count = [0]
    original = directx_codegen.HLSLCodeGen.hlsl_bitcast_result_type

    def counting(self, *args, **kwargs):
        count[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        directx_codegen.HLSLCodeGen, "hlsl_bitcast_result_type", counting
    )
    path = tmp_path / f"nested{depth}.cgl"
    path.write_text(_program(depth))
    code = crosstl.translate(str(path), backend="directx", format_output=False)
    assert "data[1]" in code
    return count[0]


def test_hlsl_type_queries_grow_polynomially_with_call_depth(monkeypatch, tmp_path):
    shallow = _bitcast_queries(monkeypatch, tmp_path, 3)
    deep = _bitcast_queries(monkeypatch, tmp_path, 9)
    # Without the memo every level quadruples the queries (4**6 = 4096 times from depth 3 to 9); with it the
    # count grows with the square of the depth (9 times).
    assert deep <= 16 * shallow, (shallow, deep)


@pytest.mark.parametrize("backend", ["directx", "vulkan"])
def test_deeply_nested_bit_casts_translate(tmp_path, backend):
    path = tmp_path / "nested.cgl"
    path.write_text(_program(12))
    code = crosstl.translate(str(path), backend=backend, format_output=False)
    assert code
