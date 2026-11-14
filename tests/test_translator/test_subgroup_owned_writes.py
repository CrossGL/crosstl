"""Uniformity proofs follow writes to values a function owns.

Helper summaries follow loops and in-place writes to by-value parameters and
initialized locals. Entry flows follow member writes of followed records, and
guards fold when a compile-time constant decides them: the DirectX and OpenGL
proofs then follow only the branch every invocation takes.
"""

import pytest

from crosstl.translator import parse
from tests.test_translator.test_directx_software_reductions import _source
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_subgroup_uniform_returns import (  # noqa: F401
    RETURN_ERRORS,
    _body,
    _record_source,
    _return_codegen,
    return_target,
)

CURSOR = "struct Cursor { uint start; uint offset; };"

LOOP_HELPERS = [
    # A local accumulated over a loop the argument bounds.
    "uint bound(uint value) { uint total = 0u; for (uint i = 0u; i < value; ++i) { total += i; } return min(total, 3u); }",
    # A by-value parameter divided in place, as an index decomposition.
    "uint bound(uint value) { uint loc = 0u; for (int i = 2; i >= 0 && value > 0u; --i) { loc += value % 3u; value /= 3u; } return min(loc, 3u); }",
    # A while loop over a copy of the argument.
    "uint bound(uint value) { uint rest = value; uint digits = 0u; while (rest > 0u) { rest = rest / 2u; digits += 1u; } return min(digits, 3u); }",
    # A by-value record moved in place and returned.
    CURSOR
    + " Cursor shift(Cursor cursor, uint step) { cursor.offset += step; return cursor; }"
    + " Cursor start(uint value) { Cursor cursor; cursor.start = value; cursor.offset = 0u; return cursor; }"
    + " uint bound(uint value) { Cursor cursor = shift(start(value), 1u); return min(cursor.offset + cursor.start, 3u); }",
]


@pytest.mark.parametrize("helper", LOOP_HELPERS)
@pytest.mark.parametrize("expression", ["bound(group.x)", "bound(2u)"])
def test_helpers_writing_owned_values_return_uniform_bounds(
    tmp_path, helper, expression, return_target
):
    source = parse(_source("uint", _body(expression), helper))
    generated = _return_codegen(return_target).generate_stage(source, "compute")
    assert "WaveActiveSum" not in generated
    _compile(generated, return_target, tmp_path)


@pytest.mark.parametrize("helper", LOOP_HELPERS)
def test_helpers_writing_owned_values_keep_divergent_arguments(helper, return_target):
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_source("uint", _body("bound(invocation)"), helper)), "compute"
        )


@pytest.mark.parametrize(
    "helper",
    [
        # The trip count depends on the divergent limit even though the
        # returned value is assigned from the uniform one.
        "uint bound(uint value, uint limit) { uint total = 0u; for (uint i = 0u; i < limit; ++i) { total = value; } return total; }",
        "uint bound(uint value, uint limit) { while (limit > 0u) { value += 1u; limit -= 1u; } return value; }",
        "uint bound(uint value, uint limit) { if (limit > 3u) { value += 1u; } return value; }",
        # The loop writes the uniform value from the divergent one.
        "uint bound(uint value, uint limit) { for (uint i = 0u; i < 2u; ++i) { value += limit; } return value; }",
    ],
)
def test_loop_and_branch_writes_carry_every_dependency(helper, return_target):
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_source("uint", _body("bound(group.x, invocation)"), helper)),
            "compute",
        )


@pytest.mark.parametrize(
    "helper",
    [
        # Element writes, unknown calls and reads before initialization fail
        # closed even when every argument is uniform.
        "uint bound(uint value) { uint copies[2]; for (uint i = 0u; i < 2u; ++i) { copies[i] = value; } return copies[1]; }",
        "uint bound(uint value) { uint total = 0u; for (uint i = 0u; i < value; ++i) { total += unknown(i); } return total; }",
        CURSOR
        + " uint bound(uint value) { Cursor cursor; for (uint i = 0u; i < value; ++i) { cursor.offset += 1u; } return cursor.offset; }",
        "uint bound(uint value) { uint total = 0u; for (uint i = 0u; i < value; ++i) { total += i; } }",
    ],
)
def test_unowned_or_unknown_loop_writes_remain_unproven(helper, return_target):
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_source("uint", _body("bound(group.x)"), helper)), "compute"
        )


def test_rewritten_members_of_initialized_records_compile(tmp_path, return_target):
    initializer = "Metadata result; result.count = count; result.offset = 0u; result.offset = 1u; return result;"
    generated = _return_codegen(return_target).generate_stage(
        parse(_record_source(initializer)), "compute"
    )
    _compile(generated, return_target, tmp_path)


def _cursor_source(writes):
    helpers = (
        CURSOR
        + " Cursor start(uint value) { Cursor cursor; cursor.start = value; cursor.offset = 0u; return cursor; }"
    )
    return _source(
        "uint",
        "Cursor cursor = start(group.x);"
        + writes
        + _body("cursor.start + cursor.offset"),
        helpers,
    )


@pytest.mark.parametrize(
    "writes",
    [
        "cursor.offset += group.y;",
        "if (group.y > 1u) { cursor.offset += 2u; } else { cursor.start -= 1u; }",
        "cursor.offset = group.z; cursor.offset *= 2u;",
    ],
)
def test_directx_entry_records_moved_by_uniform_member_writes_stay_uniform(
    tmp_path, writes
):
    generated = _return_codegen("directx").generate_stage(
        parse(_cursor_source(writes)), "compute"
    )
    assert "WaveActiveSum" not in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    "writes",
    [
        "cursor.offset += invocation;",
        "if (invocation > 3u) { cursor.offset += 1u; }",
        "cursor.offset = 0u; cursor.start = invocation;",
    ],
)
def test_entry_records_moved_by_divergent_member_writes_are_refused(
    writes, return_target
):
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_cursor_source(writes)), "compute"
        )


def _guard_source(declaration, guard):
    body = f"""{declaration}
        if ({guard}) {{
            return;
        }}
        outputWords[index] = WaveActiveSum(invocation);"""
    return _source("uint", body)


@pytest.mark.parametrize(
    "declaration, guard",
    [
        ("const bool masked = false;", "masked && inputWords[invocation] > 0u"),
        ("const bool masked = !true;", "masked && inputWords[invocation] > 0u"),
        ("const bool enabled = true;", "!enabled && invocation > 2u"),
        ("const bool open = true;", "!(open || inputWords[invocation] > 0u)"),
        (
            "const bool masked = false; const bool both = masked || false;",
            "both && invocation > 2u",
        ),
    ],
)
def test_guards_decided_by_constants_fold(tmp_path, declaration, guard, return_target):
    generated = _return_codegen(return_target).generate_stage(
        parse(_guard_source(declaration, guard)), "compute"
    )
    assert "WaveActiveSum" not in generated
    _compile(generated, return_target, tmp_path)


@pytest.mark.parametrize(
    "declaration, guard",
    [
        ("const bool masked = true;", "masked && inputWords[invocation] > 0u"),
        ("const bool open = false;", "!(open || inputWords[invocation] > 0u)"),
        ("const bool enabled = false;", "!enabled && invocation > 2u"),
        # A written local is not a constant.
        ("bool masked = false; masked = true;", "masked && invocation > 2u"),
    ],
)
def test_guards_left_to_runtime_values_are_refused(declaration, guard, return_target):
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_guard_source(declaration, guard)), "compute"
        )


def test_directx_omits_branches_constants_rule_out():
    body = """const bool masked = false;
        if (masked) { outputWords[index] = 12345u; }
        if (!masked || inputWords[index] > 0u) { outputWords[index] = 23456u; }
        outputWords[index] = WaveActiveSum(invocation);"""
    generated = _return_codegen("directx").generate_stage(
        parse(_source("uint", body)), "compute"
    )
    assert "12345" not in generated
    assert "23456" in generated
    # The constant decides the second guard, so its runtime side is gone.
    assert "inputWords[" not in generated


def test_constant_branches_follow_lexical_scope(return_target):
    body = """{ const bool masked = false; }
        if (masked) { outputWords[index] = 12345u; }
        if (!masked) { outputWords[index] = 23456u; }
        outputWords[index] = WaveActiveSum(invocation);"""
    generated = _return_codegen(return_target).generate_stage(
        parse(_source("uint", body, "const bool masked = true;")), "compute"
    )
    # The block's constant ends with the block; the shader constant decides.
    assert "12345" in generated
    assert "23456" not in generated


def test_directx_constant_guards_keep_their_condition_and_drop_dead_branches(
    tmp_path,
):
    body = """const bool masked = false;
        if (!masked) { outputWords[index] = 23456u; } else { outputWords[index] = 12345u; }
        if (masked) { outputWords[index] = 34567u; } else { outputWords[index] = 45678u; }
        outputWords[index] = WaveActiveSum(invocation);"""
    generated = _return_codegen("directx").generate_stage(
        parse(_source("uint", body)), "compute"
    )
    # Every operand is constant, so the condition is still emitted as written.
    assert "if (!masked)" in generated
    assert "if (masked)" in generated
    assert "23456" in generated and "45678" in generated
    assert "12345" not in generated and "34567" not in generated
    _compile(generated, "directx", tmp_path)


def test_opengl_omits_branches_constants_rule_out(tmp_path):
    body = """const bool masked = false;
        if (masked) { outputWords[index] = 12345u; }
        if (!masked || inputWords[index] > 0u) { outputWords[index] = 23456u; }
        outputWords[index] = WaveActiveSum(invocation);"""
    generated = _return_codegen("opengl").generate_stage(
        parse(_source("uint", body)), "compute"
    )
    entry = generated.split("void main")[1]
    assert "12345" not in entry
    assert "23456" in entry
    # The constant decides the second guard, so its runtime side is gone.
    assert "inputWords[" not in entry
    _compile(generated, "opengl", tmp_path)


DEAD_BRANCH_HELPER = "uint total(uint value) { return WaveActiveSum(value); }"
DEAD_BRANCHES = [
    # A divergent return before collective work.
    """if (masked) {
            if (invocation > 3u) { return; }
            outputWords[index] = WaveActiveSum(invocation);
        }""",
    # A collective helper called under a divergent guard.
    """if (masked) {
            if (invocation > 3u) { outputWords[index] = total(invocation); }
        }""",
    # The same, in the else branch of a guard the constant takes.
    """if (!masked) {
            outputWords[index] = 1u;
        } else if (invocation > 3u) {
            outputWords[index] = total(invocation);
        }""",
]


@pytest.mark.parametrize("branch", DEAD_BRANCHES[:2])
def test_branches_constants_rule_out_need_no_uniform_collectives(
    tmp_path, branch, return_target
):
    body = f"""const bool masked = false;
        {branch}
        outputWords[index] = total(invocation);"""
    generated = _return_codegen(return_target).generate_stage(
        parse(_source("uint", body, DEAD_BRANCH_HELPER)), "compute"
    )
    _compile(generated, return_target, tmp_path)


@pytest.mark.parametrize("branch", DEAD_BRANCHES)
@pytest.mark.parametrize(
    "declaration",
    [
        "bool masked = invocation > 100u;",
        # A written local is not a constant.
        "bool masked = false; masked = invocation > 100u;",
    ],
)
def test_branches_runtime_values_select_keep_their_collectives_checked(
    branch, declaration, return_target
):
    body = f"""{declaration}
        {branch}
        outputWords[index] = total(invocation);"""
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_source("uint", body, DEAD_BRANCH_HELPER)), "compute"
        )
